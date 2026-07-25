#!/usr/bin/env python
"""Download the MediaPipe face landmarker bundle (~3.6 MB).

The refined 478-point variant is required: the iris landmarks it adds are what
locate each eye reliably on infrared, where eyelid geometry is not trustworthy.
"""

import os
import sys
import urllib.request

URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
       "face_landmarker/float16/1/face_landmarker.task")
DEST = "face_landmarker.task"

if os.path.exists(DEST):
    print(f"{DEST} already present ({os.path.getsize(DEST) / 1e6:.1f} MB)")
    sys.exit(0)

print(f"downloading {URL}")
urllib.request.urlretrieve(URL, DEST)
print(f"wrote {DEST} ({os.path.getsize(DEST) / 1e6:.1f} MB)")
