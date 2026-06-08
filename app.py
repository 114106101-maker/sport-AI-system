import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from PIL import Image, ImageDraw, ImageFont
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==================================================
# 0. 模型下載與初始化
# ==================================================
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with urllib.request.urlopen(MODEL_URL) as response, open(MODEL_PATH, 'wb') as out_file:
            out_file.write(response.read())

# ==================================================
# 1. 幾何運算邏輯 (保留原程式)
# ==================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_torso_angle(shoulder, hip):
    torso_x, torso_y = shoulder[0] - hip[0], shoulder[1] - hip[1]
    cosine_angle = -torso_y / (np.sqrt(torso_x**2 + torso_y**2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==================================================
# 2. 繪製邏輯 (適配 Streamlit/OpenCV)
# ==================================================
def draw_skeleton(img, landmarks, error_knee, error_back):
    h, w, _ = img.shape
    points = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in [11,12,23,24,25,26,27,28]}
    
    connections = [
        (11, 12, False, False), (23, 24, False, False),
        (11, 23, True, False), (12, 24, True, False),
        (23, 25, False, True), (25, 27, False, True),
        (24, 26, False, True), (26, 28, False, True),
    ]
    for p1, p2, is_back, is_leg in connections:
        if p1 in points and p2 in points:
            color = (0, 0, 255) if (is_back and error_back) or (is_leg and error_knee) else (235, 206, 135)
            cv2.line(img, points[p1], points[p2], color, 3, cv2.LINE_AA)

def draw_ui_overlay(img, counter, stage, side, knee_angle, torso_angle, error_knee, error_back, avg_visibility):
    # 由於 Streamlit 部署環境可能沒有 Windows 字體，這裡改用 OpenCV 預設字體
    # 若要中文，需上傳 .ttf 檔至 GitHub 並使用 PIL 讀取
    h, w, _ = img.shape
    color = (0, 0, 255) if (error_knee or error_back) else (0, 255, 0) if knee_angle < 140 else (255, 120, 0)
    cv2.rectangle(img, (0, 0), (w, h), color, 8)
    
    # 簡易資訊面板
    cv2.rectangle(img, (10, 10), (300, 200), (20, 20, 20), -1)
    cv2.putText(img, f"Count: {counter}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Stage: {stage}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Knee: {int(knee_angle)} deg", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, f"Torso: {int(torso_angle)} deg", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    if error_knee: cv2.putText(img, "KNEE OVER TOES!", (w//2-100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    if error_back: cv2.putText(img, "BACK NOT STRAIGHT!", (w//2-100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
    return img

# ==================================================
# 3. WebRTC 實時處理類別
# ==================================================
class SquatAnalyzer:
    def __init__(self):
        download_model()
        self.base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        self.options = vision.PoseLandmarkerOptions(
            base_options=self.base_options, running_mode=vision.RunningMode.IMAGE)
        self.detector = vision.PoseLandmarker.create_from_options(self.options)
        
        self.counter = 0
        self.stage = None
        self.knee_hist = []
        self.torso_hist = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        # MediaPipe 處理
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        error_knee, error_back = False, False
        side, avg_vis = 'unknown', 0.0
        knee_angle, torso_angle = 180.0, 0.0

        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            # 側面識別
            l_vis = sum([landmarks[i].visibility for i in [11, 23, 25, 27]])
            r_vis = sum([landmarks[i].visibility for i in [12, 24, 26, 28]])
            
            if l_vis > r_vis:
                side, indices = 'left', [11, 23, 25, 27]
                avg_vis = l_vis / 4
            else:
                side, indices = 'right', [12, 24, 26, 28]
                avg_vis = r_vis / 4

            if avg_vis >= 0.5:
                sh = [landmarks[indices[0]].x, landmarks[indices[0]].y]
                hp = [landmarks[indices[1]].x, landmarks[indices[1]].y]
                kn = [landmarks[indices[2]].x, landmarks[indices[2]].y]
                ak = [landmarks[indices[3]].x, landmarks[indices[3]].y]

                cur_knee = calculate_angle(hp, kn, ak)
                cur_torso = calculate_torso_angle(sh, hp)
                
                self.knee_hist.append(cur_knee)
                self.torso_hist.append(cur_torso)
                if len(self.knee_hist) > 3:
                    self.knee_hist.pop(0); self.torso_hist.pop(0)
                
                knee_angle = np.mean(self.knee_hist)
                torso_angle = np.mean(self.torso_hist)

                # 計數
                if knee_angle > 150:
                    if self.stage == 'Down': self.counter += 1
                    self.stage = 'Up'
                elif knee_angle < 100:
                    self.stage = 'Down'

                # 糾錯
                if knee_angle < 140:
                    if (side == 'left' and kn[0] < ak[0] - 0.01) or (side == 'right' and kn[0] > ak[0] + 0.01):
                        error_knee = True
                    if torso_angle > 35:
                        error_back = True

        draw_skeleton(img, results.pose_landmarks[0] if results.pose_landmarks else None, error_knee, error_back)
        img = draw_ui_overlay(img, self.counter, self.stage, side, knee_angle, torso_angle, error_knee, error_back, avg_vis)
        
        return img

# ==================================================
# 4. Streamlit UI
# ==================================================
st.set_page_config(page_title="AI Squat Coach", layout="wide")
st.title("🏋️ AI 深蹲姿勢糾錯系統")
st.markdown("請將鏡頭對準身體側面，開始進行深蹲訓練。")

# 初始化分析器
analyzer = SquatAnalyzer()

webrtc_streamer(
    key="squat-analysis",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_transformer=analyzer.transform,
)

st.info("注意：本系統在網頁端僅提供視覺反饋，語音功能已移除以適配雲端環境。")