"""IR camera capture.

Windows Hello style cameras strobe their infrared emitter, so alternate frames
arrive unlit and useless. This drops them, which is why the effective frame rate
is half the delivered one. On the reference hardware the sensor delivers 15fps
(not the 30 it advertises), leaving ~7.5 usable frames per second - that ceiling
drives everything downstream, so gestures have to be held rather than flicked.
"""

import time

import cv2

DEFAULT_DEVICE = "/dev/video2"


class IRCamera:
    """Yields only emitter-lit grayscale frames."""

    def __init__(self, device=DEFAULT_DEVICE, calibrate_seconds=1.0):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {device}")
        # Without this the driver hands back a queue several frames deep, which
        # shows up as input lag rather than as dropped frames.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lit_cutoff = self._calibrate(calibrate_seconds)

    def _read_gray(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _calibrate(self, seconds):
        """Learn the split between lit and unlit frames from the feed itself."""
        levels = []
        t0 = time.time()
        while time.time() - t0 < seconds:
            g = self._read_gray()
            if g is not None:
                levels.append(float(g.mean()))
        if not levels:
            raise RuntimeError("camera opened but delivered no frames")
        return (max(levels) + min(levels)) / 2

    def read(self):
        """One lit frame, or None if this frame was unlit or the read failed."""
        g = self._read_gray()
        if g is None or g.mean() < self.lit_cutoff:
            return None
        return g

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()
