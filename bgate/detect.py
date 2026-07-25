"""Locating the eyes, from more than one detector at once.

All the classifier needs is where each eye is and how wide it is. Two detectors
can supply that, and they fail in different places:

* **MediaPipe** (BlazeFace short-range + the 478-point mesh) is what the training
  crops were cut with. It is trained on frontal RGB selfies, so it struggles on a
  tilted head - and worse, on a head that is tilted *with one eye shut*, which is
  exactly the gesture we are trying to detect.
* **YuNet** (OpenCV, 228 KB) returns a face box plus five keypoints, two of them
  eye centres. Measured on this camera it detected 200/200 frames against
  MediaPipe's 199/200, at half the latency.

Running both is more useful than choosing. The pair agree closely enough to be
interchangeable, so whenever both succeed we measure the difference and use it to
correct YuNet's estimate - then when MediaPipe drops out mid-wink, YuNet can take
over with crops the model still recognises.

That calibration is learned online rather than hardcoded, because it depends on
the camera and on where you sit.
"""

from collections import deque

import cv2
import numpy as np

from . import ui

# MediaPipe eye rings, keyed by the side names the dataset uses. The names are
# historical and misleading: "img_left" is the SUBJECT'S left eye, which on an
# unmirrored feed appears on the RIGHT of the picture. The mapping from name to
# ring has never changed, so existing labels stay valid.
RINGS = {"img_left": ui.RING_L, "img_right": ui.RING_R}

# YuNet keypoint order: [right eye, left eye, nose, right mouth, left mouth],
# "right" meaning the subject's right. Measured against the mesh: kp0 lines up
# with RING_R, kp1 with RING_L.
YUNET_KP = {"img_left": 1, "img_right": 0}

# Eye width as a fraction of interocular distance, measured at 0.375 +/- 0.019
# over 400 eyes. Only a starting value: it is refined online.
WIDTH_RATIO = 0.3752

CALIB_WINDOW = 120        # frames of agreement kept for the running correction
CALIB_MIN = 20            # before that, fall back to the measured defaults


class Eye:
    """Where one eye is, in pixels. All a crop needs."""

    __slots__ = ("cx", "cy", "width")

    def __init__(self, cx, cy, width):
        self.cx, self.cy, self.width = float(cx), float(cy), float(width)

    def as_tuple(self):
        return self.cx, self.cy, self.width

    def __repr__(self):
        return f"Eye({self.cx:.1f}, {self.cy:.1f}, w={self.width:.1f})"


class Detection:
    """One detector's answer for one frame."""

    def __init__(self, source, eyes, landmarks=None, box=None):
        self.source = source
        self.eyes = eyes                # {side: Eye}
        self.landmarks = landmarks      # mesh landmarks, when the source has them
        self.box = box

    def __bool__(self):
        return bool(self.eyes)


class MediaPipeLocator:
    """The mesh. Also the only source of eyelid contours, which the UI draws."""

    name = "mediapipe"

    def __init__(self, landmarker=None):
        if landmarker is None:
            from .model import make_landmarker
            landmarker = make_landmarker()
        self.landmarker = landmarker
        import mediapipe as mp
        self._mp = mp

    def locate(self, gray):
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB,
                               data=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
        res = self.landmarker.detect(image)
        if not res.face_landmarks:
            return None
        lm = res.face_landmarks[0]
        h, w = gray.shape[:2]
        eyes = {}
        for side, ring in RINGS.items():
            pts = np.array([[lm[i].x * w, lm[i].y * h] for i in ring])
            eyes[side] = Eye(pts[:, 0].mean(), pts[:, 1].mean(),
                             pts[:, 0].max() - pts[:, 0].min())
        return Detection(self.name, eyes, landmarks=lm)


class YuNetLocator:
    """OpenCV's YuNet. Face box plus five keypoints, two of them eye centres."""

    name = "yunet"

    def __init__(self, model_path, size, score_threshold=0.6):
        self.model_path = model_path
        self._size = None
        self.detector = cv2.FaceDetectorYN_create(model_path, "", size,
                                                  score_threshold, 0.3, 5000)
        self._set_size(size)

    def _set_size(self, size):
        if size != self._size:
            self.detector.setInputSize(size)
            self._size = size

    def locate(self, gray):
        h, w = gray.shape[:2]
        self._set_size((w, h))
        _rc, faces = self.detector.detect(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
        if faces is None or not len(faces):
            return None
        fa = max(faces, key=lambda f: f[14])       # highest score
        kp = [np.array([fa[4 + 2 * i], fa[5 + 2 * i]]) for i in range(5)]
        interocular = float(np.linalg.norm(kp[0] - kp[1]))
        width = interocular * WIDTH_RATIO
        eyes = {side: Eye(kp[i][0], kp[i][1], width) for side, i in YUNET_KP.items()}
        box = (float(fa[0]), float(fa[1]), float(fa[2]), float(fa[3]))
        return Detection(self.name, eyes, box=box)


class DualLocator:
    """Runs both detectors and reconciles them.

    ``primary`` decides whose geometry is used when both succeed; the other still
    runs, both so its disagreement is visible and so the correction stays fresh.
    When the primary fails, the fallback is used with that correction applied.
    """

    def __init__(self, mediapipe, yunet, primary="mediapipe"):
        self.locators = {mediapipe.name: mediapipe, "yunet": yunet} if yunet else \
                        {mediapipe.name: mediapipe}
        self.mediapipe, self.yunet = mediapipe, yunet
        self.primary = primary if primary in self.locators else mediapipe.name
        # Per-side history of (dx, dy, width_ratio) between the two detectors.
        self._agree = {side: deque(maxlen=CALIB_WINDOW) for side in RINGS}
        self.last = {}

    def toggle_primary(self):
        names = list(self.locators)
        self.primary = names[(names.index(self.primary) + 1) % len(names)]
        return self.primary

    def _observe(self, mp_det, yn_det):
        for side in RINGS:
            a, b = mp_det.eyes.get(side), yn_det.eyes.get(side)
            if a and b and b.width > 1e-6:
                self._agree[side].append((a.cx - b.cx, a.cy - b.cy, a.width / b.width))

    def correction(self, side):
        """Learned (dx, dy, width_scale) taking YuNet's estimate onto MediaPipe's.
        Returns None until enough agreement has been seen."""
        hist = self._agree[side]
        if len(hist) < CALIB_MIN:
            return None
        arr = np.array(hist)
        return tuple(np.median(arr, axis=0))

    def _corrected(self, det):
        """YuNet geometry mapped onto the crops the model was trained with."""
        eyes = {}
        for side, eye in det.eyes.items():
            fix = self.correction(side)
            if fix is None:
                eyes[side] = eye
            else:
                dx, dy, scale = fix
                eyes[side] = Eye(eye.cx + dx, eye.cy + dy, eye.width * scale)
        return Detection(det.source, eyes, landmarks=det.landmarks, box=det.box)

    def locate(self, gray):
        """Returns (chosen, per-detector results). ``chosen`` may be None."""
        results = {}
        for name, loc in self.locators.items():
            try:
                results[name] = loc.locate(gray)
            except Exception:
                results[name] = None
        self.last = results

        mp_det, yn_det = results.get("mediapipe"), results.get("yunet")
        if mp_det and yn_det:
            self._observe(mp_det, yn_det)

        chosen = results.get(self.primary)
        if chosen is None:
            # Primary lost the face: fall back to whoever still has it.
            for name, det in results.items():
                if det is not None:
                    chosen = self._corrected(det) if name == "yunet" else det
                    break
        elif self.primary == "yunet":
            chosen = self._corrected(chosen)
        return chosen, results

    def agreement(self):
        """Median centre disagreement per side, in pixels - for display."""
        out = {}
        for side, hist in self._agree.items():
            if len(hist) >= CALIB_MIN:
                arr = np.array(hist)
                out[side] = float(np.median(np.hypot(arr[:, 0], arr[:, 1])))
        return out
