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

import math
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
        self.roll = roll_of(eyes) or 0.0

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

    # 0.6 rejected faces that are plainly there; 0.3 finds them with no false
    # positive on any of 39 faceless inputs.
    DEFAULT_SCORE = 0.3

    def __init__(self, model_path, size, score_threshold=DEFAULT_SCORE):
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


class _Landmark:
    """A landmark mapped back into the original frame's coordinates.

    MediaPipe's own objects are read-only, so rotating the image means rebuilding
    them - cheap for 478 points, and it keeps everything downstream (crops, iris
    circles, the eyelid contours the UI draws) working on unrotated coordinates
    exactly as before.
    """

    __slots__ = ("x", "y", "z")

    def __init__(self, x, y, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)


def roll_of(eyes):
    """In-plane head roll in degrees, from the line between the eye centres.
    0 is level; positive means the subject's left eye sits lower in the image."""
    a, b = eyes.get("img_right"), eyes.get("img_left")
    if a is None or b is None:
        return None
    return math.degrees(math.atan2(b.cy - a.cy, b.cx - a.cx))


class RollTracker:
    """Wraps a locator and cancels head roll before asking it to detect.

    Face detectors are trained on upright faces. Measured on 43 frames where
    detection was struggling, rotating to upright took MediaPipe from 67% to 93%.

    The angle is *tracked*, not swept: once the head is found at some roll, the
    next frame almost certainly has a similar one, so the usual cost is a single
    detection. The sweep only runs after the tracked angle and level have both
    failed, which is when a frame is being lost anyway.
    """

    SWEEP = (-20.0, 20.0, -35.0, 35.0)
    MAX_ANGLE = 50.0
    # Below this the correction is not worth a warpAffine; measured roll on a
    # roughly level head wanders by a degree or two frame to frame.
    DEADZONE = 3.0

    def __init__(self, locator, sweep=SWEEP):
        self.locator = locator
        self.sweep = sweep
        self.angle = 0.0          # last roll that worked
        self.last_used = 0.0      # rotation applied on the last successful frame
        self.name = getattr(locator, "name", "locator")

    def _candidates(self):
        """Level FIRST, then the tracked angle, then the sweep - each only once.

        Trying the tracked angle first is self-reinforcing and drifts: rotating by
        a remembered angle makes the mesh fit the rotated image, which then
        measures roughly the angle that was applied, which is stored as the new
        truth. Measured over 180 frames it invented rolls up to 35 degrees off,
        each frame handing its error to the next.

        An unrotated fit cannot be wrong that way, so it is always tried first and
        rotation is left to rescue the frames where it genuinely fails.
        """
        tracked = 0.0 if abs(self.angle) < self.DEADZONE else self.angle
        out, seen = [], set()
        for angle in (0.0, tracked, *self.sweep):
            key = round(angle, 1)
            if key not in seen and abs(angle) <= self.MAX_ANGLE:
                seen.add(key)
                out.append(angle)
        return out

    @staticmethod
    def _rotate(gray, angle):
        h, w = gray.shape[:2]
        m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE), m

    @staticmethod
    def _unrotate(det, m, shape):
        """Map a detection made on the rotated image back to the original."""
        h, w = shape[:2]
        inv = cv2.invertAffineTransform(m)

        def back(px, py):
            return (inv[0, 0] * px + inv[0, 1] * py + inv[0, 2],
                    inv[1, 0] * px + inv[1, 1] * py + inv[1, 2])

        eyes = {}
        for side, eye in det.eyes.items():
            x, y = back(eye.cx, eye.cy)
            eyes[side] = Eye(x, y, eye.width)      # rotation does not change scale

        landmarks = det.landmarks
        if landmarks is not None:
            pts = np.array([[l.x * w, l.y * h, 1.0] for l in landmarks])
            mapped = pts @ inv.T
            landmarks = [_Landmark(px / w, py / h, l.z)
                         for (px, py), l in zip(mapped, landmarks)]
        return Detection(det.source, eyes, landmarks=landmarks, box=det.box)

    def locate(self, gray):
        for angle in self._candidates():
            if angle == 0.0:
                det = self.locator.locate(gray)
                if det:
                    self.last_used = 0.0
                    measured = roll_of(det.eyes)
                    if measured is not None:
                        self.angle = measured
                    return det
                continue

            rotated, m = self._rotate(gray, angle)
            det = self.locator.locate(rotated)
            if det:
                det = self._unrotate(det, m, gray.shape)
                self.last_used = angle
                measured = roll_of(det.eyes)
                # Track what the face actually is, not the angle that happened to
                # work, so the next frame starts from the truth.
                if measured is not None:
                    self.angle = measured
                return det
        return None


class BoxedMeshLocator:
    """Find the face with one detector, then run the mesh inside that box.

    The mesh is the only thing that gives eyelid contours and iris landmarks, so
    the crops it produces are the ones the model was trained on. Using a second
    detector merely to say *where* to look keeps that intact - unlike taking the
    second detector's own eye keypoints, whose crops agreed with the mesh on only
    80% of verdicts.

    It also makes the mesh's job easier: a face that is small and low-contrast in
    the full frame becomes large and centred once its box is cropped and upscaled.
    Measured on 88 frames where detection was struggling, the mesh succeeded
    inside every single box the proposer found.
    """

    name = "boxed-mesh"
    MARGIN = 0.45       # of the box size, added on every side
    TARGET = 256        # upscale the crop to about this, if it is smaller

    def __init__(self, proposer, mesh):
        self.proposer = proposer
        self.mesh = mesh

    def locate(self, gray):
        proposal = self.proposer.locate(gray)
        if not proposal or not proposal.box:
            return None
        h, w = gray.shape[:2]
        x, y, bw, bh = proposal.box
        cx, cy = x + bw / 2, y + bh / 2
        half = max(bw, bh) * (0.5 + self.MARGIN)
        x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
        x1, y1 = int(min(w, cx + half)), int(min(h, cy + half))
        if x1 <= x0 + 4 or y1 <= y0 + 4:
            return None

        roi = gray[y0:y1, x0:x1]
        scale = self.TARGET / max(roi.shape[:2])
        if scale > 1:
            roi = cv2.resize(roi, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
        else:
            scale = 1.0

        det = self.mesh.locate(roi)
        if not det:
            return None

        # Back to full-frame normalised coordinates.
        rh, rw = roi.shape[:2]
        landmarks = [_Landmark((x0 + l.x * rw / scale) / w,
                               (y0 + l.y * rh / scale) / h, l.z)
                     for l in det.landmarks]
        eyes = {}
        for side, ring in RINGS.items():
            pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in ring])
            eyes[side] = Eye(pts[:, 0].mean(), pts[:, 1].mean(),
                             pts[:, 0].max() - pts[:, 0].min())
        return Detection(self.name, eyes, landmarks=landmarks, box=proposal.box)


class Cascade:
    """Try locators in order and take the first that finds a face.

    They fail in different places: roll-compensated mesh handles a tilted head
    but not a dark low-contrast one; YuNet handles the dark frame but not the
    tilt. Measured on 88 hard frames - 72% and 44% alone, 77% together.
    """

    def __init__(self, locators):
        self.locators = list(locators)
        self.used = None        # which one answered on the last frame

    def locate(self, gray):
        for loc in self.locators:
            try:
                det = loc.locate(gray)
            except Exception:
                det = None
            if det:
                self.used = getattr(loc, "name", type(loc).__name__)
                return det
        self.used = None
        return None

    @property
    def angle(self):
        """Roll tracked by whichever member tracks it, for display."""
        for loc in self.locators:
            if isinstance(loc, RollTracker):
                return loc.angle
        return 0.0
