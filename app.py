import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import RunningMode
import numpy as np
import urllib.request
import os
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av

# =================================================================
# 0. Download MediaPipe pose landmarker model (Tasks API)
# =================================================================
MODEL_PATH = "pose_landmarker_full.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

@st.cache_resource(show_spinner="Downloading pose model...")
def download_model():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH

# MediaPipe Tasks landmark indices (same as classic API)
# 11=L_SHOULDER, 12=R_SHOULDER, 23=L_HIP, 24=R_HIP
# 25=L_KNEE,     26=R_KNEE,     27=L_ANKLE, 28=R_ANKLE
_L_SIDE = [11, 23, 25, 27]
_R_SIDE = [12, 24, 26, 28]

# =================================================================
# 1. Geometry helpers
# =================================================================
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def calculate_torso_angle(shoulder, hip):
    dx, dy = shoulder[0] - hip[0], shoulder[1] - hip[1]
    cosine = -dy / (np.sqrt(dx**2 + dy**2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

# =================================================================
# 2. HUD overlay
# =================================================================
def draw_ai_overlay(img, counter, stage, side, knee_angle, torso_angle,
                    error_knee, error_back):
    h, w = img.shape[:2]
    color = (0, 0, 255) if (error_knee or error_back) else (
            (0, 255, 0) if knee_angle < 140 else (255, 120, 0))
    cv2.rectangle(img, (0, 0), (w, h), color, 10)

    overlay = img.copy()
    cv2.rectangle(overlay, (10, 10), (320, 220), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, "AI SQUAT COACH",        (20,  40), font, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"Count : {counter}",    (20,  80), font, 0.7, (255, 255, 100), 2)
    cv2.putText(img, f"Stage : {stage}",      (20, 110), font, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Knee  : {int(knee_angle)} deg", (20, 140), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Torso : {int(torso_angle)} deg",(20, 170), font, 0.6, (255, 255, 255), 1)
    cv2.putText(img, f"Side  : {side}",       (20, 200), font, 0.6, (200, 200, 200), 1)

    if error_knee:
        cv2.putText(img, "WARNING: KNEE OVER TOES!",
                    (w//2 - 200, 60), font, 0.8, (0, 0, 255), 3)
    if error_back:
        cv2.putText(img, "WARNING: BACK NOT STRAIGHT!",
                    (w//2 - 200, 100), font, 0.8, (0, 0, 255), 3)
    return img

# =================================================================
# 3. Draw skeleton manually from Tasks landmarks
# =================================================================
_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),   # shoulders / arms
    (11,23),(12,24),(23,24),                   # torso
    (23,25),(25,27),(24,26),(26,28),           # legs
    (27,29),(27,31),(28,30),(28,32),           # feet
]

def draw_landmarks_tasks(img, landmarks):
    h, w = img.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in _CONNECTIONS:
        if a < len(pts) and b < len(pts):
            cv2.line(img, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(img, pt, 4, (255, 255, 0), -1)

# =================================================================
# 4. Core analyser — uses Tasks LIVE_STREAM mode
# =================================================================
class SquatAnalyzer:
    def __init__(self, model_path: str):
        self.counter   = 0
        self.stage     = "Standing"
        self.knee_hist : list[float] = []
        self.torso_hist: list[float] = []
        self._last_result = None
        self._ts = 0

        # Build landmarker in LIVE_STREAM mode so it's non-blocking
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.LIVE_STREAM,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._on_result,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    # Callback stores the latest result
    def _on_result(self, result, output_image, timestamp_ms):
        self._last_result = result

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Feed frame to landmarker (async; result arrives via callback)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._ts += 33                          # ~30 fps synthetic timestamp
        self.landmarker.detect_async(mp_img, self._ts)

        result = self._last_result
        error_knee, error_back = False, False
        side = "unknown"
        knee_angle, torso_angle = 180.0, 0.0

        if result and result.pose_landmarks:
            lms = result.pose_landmarks[0]      # first (only) pose

            # Choose side with better visibility
            l_vis = sum(lms[i].visibility for i in _L_SIDE)
            r_vis = sum(lms[i].visibility for i in _R_SIDE)
            idxs  = _L_SIDE if l_vis >= r_vis else _R_SIDE
            side  = "Left"  if l_vis >= r_vis else "Right"

            sh = [lms[idxs[0]].x, lms[idxs[0]].y]
            hp = [lms[idxs[1]].x, lms[idxs[1]].y]
            kn = [lms[idxs[2]].x, lms[idxs[2]].y]
            ak = [lms[idxs[3]].x, lms[idxs[3]].y]

            cur_knee  = calculate_angle(hp, kn, ak)
            cur_torso = calculate_torso_angle(sh, hp)

            self.knee_hist.append(cur_knee)
            self.torso_hist.append(cur_torso)
            if len(self.knee_hist) > 3:
                self.knee_hist.pop(0)
                self.torso_hist.pop(0)

            knee_angle  = float(np.mean(self.knee_hist))
            torso_angle = float(np.mean(self.torso_hist))

            # State machine
            if knee_angle > 150:
                if self.stage == "Down":
                    self.counter += 1
                self.stage = "Up"
            elif knee_angle < 100:
                self.stage = "Down"

            # Error detection (only when in squat)
            if knee_angle < 140:
                if (side == "Left"  and kn[0] < ak[0] - 0.01) or \
                   (side == "Right" and kn[0] > ak[0] + 0.01):
                    error_knee = True
                if torso_angle > 35:
                    error_back = True

            draw_landmarks_tasks(img, lms)

        img = draw_ai_overlay(img, self.counter, self.stage, side,
                              knee_angle, torso_angle, error_knee, error_back)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# =================================================================
# 5. Streamlit UI
# =================================================================
st.set_page_config(page_title="AI Squat Coach", layout="wide")
st.title("🏋️ AI 深蹲姿勢糾錯系統")
st.markdown(
    "使用 **MediaPipe Tasks API**（最新穩定版）。"
    "請將鏡頭對準側面，讓全身入鏡。"
)

model_path = download_model()
analyzer   = SquatAnalyzer(model_path)

webrtc_streamer(
    key="squat-tasks-v1",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    video_frame_callback=analyzer.transform,   # new kwarg (replaces video_transformer)
    media_stream_constraints={"video": True, "audio": False},
)
