# AirCursor
# AirCursor

AirCursor is a Python-based hands-free mouse control system that uses your webcam and simple hand gestures. It allows you to move the mouse pointer and perform clicks by pinching your thumb and index finger, making use of computer vision and dynamic calibration for accurate and smooth control.

---

## Features

- **Hands-free mouse movement**: Control your cursor with your hand in the air.
- **Gesture-based clicking**: Pinch your thumb and index finger to click, release to stop.
- **Custom calibration**: Define your control area and pinch sensitivity for your unique setup.
- **Smooth pointer movement**: Uses a Kalman filter to reduce jitter.
- **Dynamic pinch detection**: Adapts to your hand size and distance from the camera.

---

## Requirements

- Python 3.7+
- Webcam
- Python packages:
  - opencv-python
  - mediapipe
  - numpy
  - pynput
  - screeninfo
  - filterpy

Install dependencies with:
```bash
pip install opencv-python mediapipe numpy pynput screeninfo filterpy
```

---

## Usage

### 1. Calibrate Your Setup

Run the calibration script to define your control area and pinch gesture threshold:

```bash
python Calibration.py
```

**Calibration steps:**
- Click the four corners of your desired control area on the webcam feed (Top-Left, Top-Right, Bottom-Left, Bottom-Right).
- For each distance (close, medium, far), hold your hand open and then pinch as prompted to calibrate the pinch detection.
- Calibration data is saved to `calibration.npy`.

### 2. Start AirCursor

After calibration, run the main AirCursor script:

```bash
python AirCursor.py
```

**How to use:**
- Move your hand within the calibrated area to control the mouse pointer.
- Pinch your thumb and index finger together to click and hold; release to stop clicking.

---

## File Overview

| File            | Description                                   |
|-----------------|-----------------------------------------------|
| Calibration.py  | Calibration tool for area and pinch threshold |
| AirCursor.py    | Main script for mouse control                 |
| calibration.npy | Calibration data (auto-generated)             |

---

## Troubleshooting

- **No calibration data**: If you see "Calibration data not found. Please run calibrate_corners_and_pinch.py first.", run `Calibration.py` first.
- **Webcam issues**: Ensure your webcam is connected and accessible.
- **Dependency errors**: Make sure all required Python packages are installed.

---

## Credits

- Built with [MediaPipe](https://mediapipe.dev/) for hand tracking.
- Uses [OpenCV](https://opencv.org/) for image processing.
- Mouse control via [pynput](https://pynput.readthedocs.io/).
- Kalman filter from [filterpy](https://filterpy.readthedocs.io/).

---

## License

MIT License (add your license details here if desired)

---

**Enjoy hands-free mouse control with AirCursor!**

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/51063976/6912ae90-c96a-496f-ad55-ccdc05646a21/Calibration.py
[2] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/51063976/89592a81-72ad-4ce4-919e-49717017f1eb/AirCursor.py

---
Answer from Perplexity: pplx.ai/share
