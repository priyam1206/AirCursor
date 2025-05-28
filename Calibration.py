import cv2
import numpy as np
import time
import mediapipe as mp

# === Webcam setup ===
cap = cv2.VideoCapture(0)
frame_width, frame_height = 320, 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# === Corner calibration ===
points = []
print("\n🖱️ Click 4 corners of your desired control area in this order:")
print("   1. Top-Left\n   2. Top-Right\n   3. Bottom-Left\n   4. Bottom-Right")

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))
        print(f"Point {len(points)}: {x}, {y}")

cv2.namedWindow("Calibrate")
cv2.setMouseCallback("Calibrate", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    resized = cv2.resize(frame, (frame_width, frame_height))

    for p in points:
        cv2.circle(resized, p, 5, (0, 255, 0), -1)

    cv2.imshow("Calibrate", resized)
    if len(points) == 4:
        break
    if cv2.waitKey(1) & 0xFF == 27:
        break

# === Pinch threshold calibration ===
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
min_data_points = 10  # Minimum valid detections per gesture
distances = ["close (about 30 cm)", "medium (about 60 cm)", "far (about 90 cm)"]
calibration_data_points = []

for dist in distances:
    # Open hand calibration
    print(f"\n✋ Hold your hand open (no pinch) at {dist} for 4 seconds...")
    open_distances = []
    open_hand_sizes = []
    start_time = time.time()
    while time.time() - start_time < 4:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        resized = cv2.resize(frame, (frame_width, frame_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            t = hand.landmark[4]  # Thumb tip
            i = hand.landmark[8]  # Index finger tip
            w = hand.landmark[0]  # Wrist
            m = hand.landmark[9]  # Middle finger MCP
            d = np.hypot((t.x - i.x) * frame_width, (t.y - i.y) * frame_height)
            hand_size = np.hypot((w.x - m.x) * frame_width, (w.y - m.y) * frame_height)
            open_distances.append(d)
            open_hand_sizes.append(hand_size)

            # Visual feedback
            cv2.circle(resized, (int(t.x * frame_width), int(t.y * frame_height)), 5, (0, 255, 0), -1)
            cv2.circle(resized, (int(i.x * frame_width), int(i.y * frame_height)), 5, (0, 255, 0), -1)
            cv2.putText(resized, f"Dist: {d:.1f}px | Size: {hand_size:.1f}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Calibrate", resized)
        cv2.waitKey(1)

    # Pinch calibration
    print(f"👉 Now pinch at {dist} for 4 seconds...")
    pinch_distances = []
    pinch_hand_sizes = []
    start_time = time.time()
    while time.time() - start_time < 4:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)
        resized = cv2.resize(frame, (frame_width, frame_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            t = hand.landmark[4]  # Thumb tip
            i = hand.landmark[8]  # Index finger tip
            w = hand.landmark[0]  # Wrist
            m = hand.landmark[9]  # Middle finger MCP
            d = np.hypot((t.x - i.x) * frame_width, (t.y - i.y) * frame_height)
            hand_size = np.hypot((w.x - m.x) * frame_width, (w.y - m.y) * frame_height)
            pinch_distances.append(d)
            pinch_hand_sizes.append(hand_size)

            # Visual feedback
            cv2.circle(resized, (int(t.x * frame_width), int(t.y * frame_height)), 5, (0, 255, 0), -1)
            cv2.circle(resized, (int(i.x * frame_width), int(i.y * frame_height)), 5, (0, 255, 0), -1)
            cv2.putText(resized, f"Dist: {d:.1f}px | Size: {hand_size:.1f}px", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Calibrate", resized)
        cv2.waitKey(1)

    # Validate and store data
    if len(open_distances) < min_data_points or len(pinch_distances) < min_data_points:
        print(f"Warning: Insufficient data at {dist} (Open: {len(open_distances)}, Pinch: {len(pinch_distances)})")
        continue
    avg_open = np.mean(open_distances)
    avg_pinch = np.mean(pinch_distances)
    avg_hand_size = (np.mean(open_hand_sizes) + np.mean(pinch_hand_sizes)) / 2
    if avg_open <= avg_pinch:
        print(f"Warning: Invalid data at {dist} (Open: {avg_open:.2f} <= Pinch: {avg_pinch:.2f})")
        continue
    # Use a threshold biased toward pinch
    threshold = avg_pinch + 0.3 * (avg_open - avg_pinch)
    calibration_data_points.append((avg_hand_size, threshold))

# === Compute threshold slope and intercept ===
if len(calibration_data_points) < 2:
    print("Error: Insufficient valid data points for linear fit. Using defaults.")
    threshold_slope = 0.5
    threshold_intercept = 20
else:
    hand_sizes, thresholds = zip(*calibration_data_points)
    coeffs = np.polyfit(hand_sizes, thresholds, 1)
    threshold_slope, threshold_intercept = coeffs[0], coeffs[1]
    print(f"Saved Calibration with slope: {threshold_slope:.2f}, intercept: {threshold_intercept:.2f}")

# === Save data ===
calibration_data = {
    "top_left": points[0],
    "top_right": points[1],
    "bottom_left": points[2],
    "bottom_right": points[3],
    "threshold_slope": threshold_slope,
    "threshold_intercept": threshold_intercept
}
np.save("calibration.npy", calibration_data)

cap.release()
cv2.destroyAllWindows()
