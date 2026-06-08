import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
import urllib.request
from PIL import Image
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==================================================
# 0. 設定與模型下載
# ==================================================
MODEL_PATH = "pose_landmarker_lite.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("正在下載 AI 模型，請稍候...")
        with urllib.request.urlopen(MODEL_URL) as response, open(MODEL_PATH, 'wb') as out_file:
            out_file.write(response.read())

# ==================================================
# 1. 幾何運算工具
# ==================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def calculate_torso_angle(shoulder, hip):
    torso_x, torso_y = shoulder[0] - hip[0], shoulder[1] - hip[1]
    # 計算相對於垂直線 [0, -1] 的夾角
    cosine_angle = -torso_y / (np.sqrt(torso_x**2 + torso_y**2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# ==================================================
# 2. 視覺化繪製
# ==================================================
def draw_ai_overlay(img, counter, stage, side, knee_angle, torso_angle, error_knee, error_back, avg_vis):
    h, w, _ = img.shape
    
    # 1. 邊框顏色反饋
    color = (0, 0, 255) if (error_knee or error_back) else (0, 255, 0) if knee_angle < 140 else (255, 120, 0)
    cv2.rectangle(img, (0, 0), (w, h), color, 10)
    
    # 2. 資訊面板 (深色半透明背景)
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 220), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    # 3. 寫入文字 (使用 OpenCV 預設字體以確保兼容性)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "AI SQUAT COACH", (20, 40), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Count: {counter}", (20, 80), font, 0.7, (255, 255, 100), 2)
    cv2.putText(img, f"Stage: {stage}", (20, 110), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Knee Angle: {int(knee_angle)}deg", (20, 140), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Torso Angle: {int(torso_angle)}deg", (20, 170), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Side: {side}", (20, 200), font, 0.6, (200, 200, 200), 1)

    # 4. 錯誤警告大字
    if error_knee:
        cv2.putText(img, "WARNING: KNEE OVER TOES!", (w//2-200, 60), font, 0.8, (0, 0, 255), 3)
    if error_back:
        cv2.putText(img, "WARNING: BACK NOT STRAIGHT!", (w//2-200, 100), font, 0.8, (0, 0, 255), 3)
        
    return img

def draw_skeleton(img, landmarks, error_knee, error_back):
    if landmarks is None: return
    h, w, _ = img.shape
    pts = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in [11,12,23,24,25,26,27,28]}
    
    # 連線定義: (p1, p2, is_back, is_leg)
    conn = [(11,12,0,0), (23,24,0,0), (11,23,1,0), (12,24,1,0), 
            (23,25,0,1), (25,27,0,1), (24,26,0,1), (26,28,0,1)]
            
    for p1, p2, is_back, is_leg in conn:
        if p1 in pts and p2 in pts:
            color = (0, 0, 255) if (is_back and error_back) or (is_leg and error_knee) else (235, 206, 135)
            cv2.line(img, pts[p1], pts[p2], color, 3, cv2.LINE_AA)

# ==================================================
# 3. 核心分析類別 (WebRTC Transformer)
# ==================================================
class SquatAnalyzer:
    def __init__(self):
        download_model()
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options, running_mode=vision.RunningMode.IMAGE)
        self.detector = vision.PoseLandmarker.create_from_options(options)
        
        # 狀態變數
        self.counter = 0
        self.stage = "Standing"
        self.knee_hist = []
        self.torso_hist = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # MediaPipe 預測
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)

        error_knee, error_back = False, False
        side, avg_vis = 'unknown', 0.0
        knee_angle, torso_angle = 180.0, 0.0

        if results.pose_landmarks:
            lms = results.pose_landmarks[0]
            # 判斷側面
            l_vis = sum([lms[i].visibility for i in [11, 23, 25, 27]])
            r_vis = sum([lms[i].visibility for i in [12, 24, 26, 28]])
            
            if l_vis > r_vis:
                side, idxs = 'Left', [11, 23, 25, 27]
                avg_vis = l_vis / 4
            else:
                side, idxs = 'Right', [12, 24, 26, 28]
                avg_vis = r_vis / 4

            if avg_vis >= 0.5:
                sh = [lms[idxs[0]].x, lms[idxs[0]].y]
                hp = [lms[idxs[1]].x, lms[idxs[1]].y]
                kn = [lms[idxs[2]].x, lms[idxs[2]].y]
                ak = [lms[idxs[3]].x, lms[idxs[3]].y]

                # 角度計算與平滑
                cur_knee = calculate_angle(hp, kn, ak)
                cur_torso = calculate_torso_angle(sh, hp)
                self.knee_hist.append(cur_knee)
                self.torso_hist.append(cur_torso)
                if len(self.knee_hist) > 3:
                    self.knee_hist.pop(0); self.torso_hist.pop(0)
                
                knee_angle = np.mean(self.knee_hist)
                torso_angle = np.mean(self.torso_hist)

                # 計數狀態機
                if knee_angle > 150:
                    if self.stage == "Down": self.counter += 1
                    self.stage = "Up"
                elif knee_angle < 100:
                    self.stage = "Down"

                # 糾錯邏輯
                if knee_angle < 140:
                    if (side == 'Left' and kn[0] < ak[0] - 0.01) or (side == 'Right' and kn[0] > ak[0] + 0.01):
                        error_knee = True
                    if torso_angle > 35:
                        error_back = True

        # 繪製骨架與 UI
        draw_skeleton(img, results.pose_landmarks[0] if results.pose_landmarks else None, error_knee, error_back)
        img = draw_ai_overlay(img, self.counter, self.stage, side, knee_angle, torso_angle, error_knee, error_back, avg_vis)
        
        return img

# ==================================================
# 4. Streamlit 界面佈局
# ==================================================
st.set_page_config(page_title="AI Squat Coach", layout="wide")
st.title("🏋️ AI 深蹲姿勢糾錯系統")
st.markdown("""
### 使用指南：
1. 點擊 **Start** 開啟鏡頭。
2. 請將手機/電腦放在**身體側面** (左側或右側均可)。
3. 保持背部挺直，下蹲時注意膝蓋不要過度超過腳尖。
""")

# 初始化分析類別
analyzer = SquatAnalyzer()

# 啟動 WebRTC 串流
webrtc_streamer(
    key="squat-coach",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_transformer=analyzer.transform,
)

st.warning("⚠️ 注意：本網頁版僅提供視覺反饋，語音功能因雲端權限限制已移除。")
