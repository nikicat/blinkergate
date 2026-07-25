#!/usr/bin/env python
"""Download the face models (~3.9 MB total).

The refined 478-point MediaPipe variant is required: its iris landmarks locate
each eye reliably on infrared, where eyelid geometry is not trustworthy. YuNet is
a second, independent face detector - see bgate/detect.py for why two.
"""

import os
import urllib.request

MODELS = {
    "face_landmarker.task":
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task",
    # Second face detector. Measured on an IR laptop camera it found a face in
    # 210/210 frames against MediaPipe's 209, at half the latency.
    "face_detection_yunet.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
}

for dest, url in MODELS.items():
    if os.path.exists(dest):
        print(f"{dest} already present ({os.path.getsize(dest) / 1e6:.1f} MB)")
        continue
    print(f"downloading {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  wrote {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")
