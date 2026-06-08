import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import os
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ==================================================
# 0. 初始化 MediaPipe 經典版
# ==================================================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ==================================================
# 1. 幾何運算工具 (保持不變)
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
# 2. 視覺化繪製
# ==================================================
def draw_ai_overlay(img, counter, stage, side, knee_angle, torso_angle, error_knee, error_back):
    h, w, _ = img.shape
    color = (0, 0, 255) if (error_knee or error_back) else (0, 255, 0) if knee_angle < 140 else (255, 120, 0)
    cv2.rectangle(img, (0, 0), (w, h), color, 10)
    
    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 220), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "AI SQUAT COACH (Classic)", (20, 40), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Count: {counter}", (20, 80), font, 0.7, (255, 255, 100), 2)
    cv2.putText(img, f"Stage: {stage}", (20, 110), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Knee: {int(knee_angle)}deg", (20, 140), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Torso: {int(torso_angle)}deg", (20, 170), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Side: {side}", (20, 200), font, 0.6, (200, 200, 200), 1)

    if error_knee: cv2.putText(img, "WARNING: KNEE OVER TOES!", (w//2-200, 60), font, 0.8, (0, 0, 255), 3)
    if error_back: cv2.putText(img, "WARNING: BACK NOT STRAIGHT!", (w//2-200, 100), font, 0.8, (0, 0, 255), 3)
    return img

# ==================================================
# 3. 核心分析類別 (切換至 mp.solutions.pose)
# ==================================================
class SquatAnalyzer:
    def __init__(self):
        # 這裡不再使用 PoseLandmarker，改用經典 Pose
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.counter = 0
        self.stage = "Standing"
        self.knee_hist = []
        self.torso_hist = []

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 使用經典 API 進行偵測
        results = self.pose.process(rgb_frame)

        error_knee, error_back = False, False
        side, avg_vis = 'unknown', 0.0
        knee_angle, torso_angle = 180.0, 0.0

        if results.pose_landmarks:
            lms = results.pose_landmarks.landmark
            
            # 判斷側面 (透過座標可見度/位置簡易判斷)
            # 經典版 API 沒有直接的 visibility 屬性，我們用座標權重
            l_vis = sum([1 for i in [11, 23, 25, 27] if lms[i].visibility > 0.5])
            r_vis = sum([1 for i in [12, 24, 28, 26] if lms[i].visibility > 0.5])
            
            if l_vis >= r_vis:
                side, idxs = 'Left', [11, 23, 25, 27]
            else:
                side, idxs = 'Right', [12, 24, 26, 28]

            # 提取關鍵點座標
            sh = [lms[idxs[0]].x, lms[idxs[0]].y]
            hp = [lms[idxs[1]].x, lms[idxs[1].y]]
            kn = [lms[idxs[2]].x, lms[idxs[2]].y]
            ak = [lms[idxs[3]].x, lms[idxs[3]].y]

            cur_knee = calculate_angle(hp, kn, ak)
            cur_torso = calculate_torso_angle(sh, hp)
            
            self.knee_hist.append(cur_knee)
            self.torso_hist.append(cur_torso)
            if len(self.knee_hist) > 3:
                self.knee_hist.pop(0); self.torso_hist.pop(0)
            
            knee_angle = np.mean(self.knee_hist)
            torso_angle = np.mean(self.torso_hist)

            if knee_angle > 150:
                if self.stage == "Down": self.counter += 1
                self.stage = "Up"
            elif knee_angle < 100:
                self.stage = "Down"

            if knee_angle < 140:
                if (side == 'Left' and kn[0] < ak[0] - 0.01) or (side == 'Right' and kn[0] > ak[0] + 0.01):
                    error_knee = True
                if torso_//L's torso_angle > 35:
                    error_back = True

            # 繪製經典骨架
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        img = draw_ai_overlay(img, self.counter, self.stage, side, knee_angle, torso_angle, error_knee, error_back)
        return img

# ==================================================
# 4. Streamlit UI (保持不變)
# ==================================================
st.set_page_config(page_title="AI Squat Coach", layout="wide")
st.title("🏋️ AI 深蹲姿勢糾錯系統 (穩定版)")
st.markdown("### 已切換至兼容模式，現在應可正常運行。")

analyzer = SquatAnalyzer()

webrtc_streamer(
    key="squat-classic",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    video_transformer=analyzer.transform,
)
