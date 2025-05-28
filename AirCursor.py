import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller, Button
from screeninfo import get_monitors
from filterpy.kalman import KalmanFilter
import time
import os
import sys

# === Load calibration data ===
try:
    calibration_data = np.load("calibration.npy", allow_pickle=True).item()
    top_left = np.array(calibration_data["top_left"], dtype=np.float32)
    top_right = np.array(calibration_data["top_right"], dtype=np.float32)
    bottom_left = np.array(calibration_data["bottom_left"], dtype=np.float32)
    bottom_right = np.array(calibration_data["bottom_right"], dtype=np.float32)
    threshold_slope = calibration_data["threshold_slope"]
    threshold_intercept = calibration_data["threshold_intercept"]
except FileNotFoundError:
    print("Calibration data not found. Please run calibrate_corners_and_pinch.py first.")
    exit()

# === Initialize mouse and screen size ===
mouse = Controller()
screen = get_monitors()[0]
screen_width, screen_height = screen.width, screen.height

# === MediaPipe Hands ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8, model_complexity=0)

# === Webcam setup ===
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Frame resize for processing ===
frame_width, frame_height = 320, 240

# === Perspective Transform ===
src_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
dst_points = np.array([[0, 0], [screen_width, 0], [screen_width, screen_height], [0, screen_height]], dtype=np.float32)
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# === Kalman Filter for smoothing ===
kf = KalmanFilter(dim_x=4, dim_z=2)
kf.x = np.zeros(4)
kf.P *= 1000.
kf.F = np.array([[1, 0, 1, 0],
                 [0, 1, 0, 1],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]])
kf.H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]])
kf.R = np.eye(2) * 5
kf.Q = np.eye(4) * 0.01

# === Pinch detection state ===
pinch_start_time = None
debounce_delay = 0.1
is_pinching = False

# Clear the console initially
os.system("cls" if os.name == "nt" else "clear")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    small_frame = cv2.resize(frame, (frame_width, frame_height))
    small_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    results = hands.process(small_rgb)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        # Thumb and Index
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        thumb_x, thumb_y = int(thumb_tip.x * frame_width), int(thumb_tip.y * frame_height)
        index_x, index_y = int(index_tip.x * frame_width), int(index_tip.y * frame_height)

        # Hand size for dynamic threshold
        wrist = hand_landmarks.landmark[0]
        middle_mcp = hand_landmarks.landmark[9]
        hand_size = np.linalg.norm(
            np.array([wrist.x - middle_mcp.x, wrist.y - middle_mcp.y])
        ) * frame_width

        # Midpoint
        midpoint_x = (thumb_x + index_x) / 2
        midpoint_y = (thumb_y + index_y) / 2

        # Apply perspective transform
        point = np.array([[[midpoint_x, midpoint_y]]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point, perspective_matrix)
        screen_x, screen_y = transformed_point[0][0]

        # Clamp
        screen_x = np.clip(screen_x, 0, screen_width)
        screen_y = np.clip(screen_y, 0, screen_height)

        # Kalman smoothing
        kf.predict()
        kf.update(np.array([screen_x, screen_y]))
        smooth_x, smooth_y = kf.x[0], kf.x[1]

        # Move mouse
        mouse.position = (smooth_x, smooth_y)

        # === Pinch Detection ===
        distance = np.hypot(index_x - thumb_x, index_y - thumb_y)
        pinch_threshold = threshold_slope * hand_size + threshold_intercept
        current_time = time.time()

        if distance < pinch_threshold:
            if pinch_start_time is None:
                pinch_start_time = current_time
            elif current_time - pinch_start_time > debounce_delay and not is_pinching:
                mouse.press(Button.left)
                is_pinching = True
        else:
            if is_pinching:
                mouse.release(Button.left)
                is_pinching = False
            pinch_start_time = None

        # Debug text in console
        sys.stdout.write(f"\rDistance: {distance:.2f} | Threshold: {pinch_threshold:.2f}   ")
        sys.stdout.flush()

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
