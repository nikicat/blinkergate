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
    """Yields only emitter-lit grayscale frames.

    With ``subtract=True`` the unlit frames stop being waste: each lit frame has
    the most recent dark one subtracted from it, which cancels whatever the scene
    is lit by that is *not* the emitter. That is what the strobe is for. It costs
    nothing extra - the dark frames were being read and discarded anyway.
    """

    # A dark frame older than this is not a usable reference for the scene.
    DARK_MAX_AGE = 0.25

    def __init__(self, device=DEFAULT_DEVICE, calibrate_seconds=1.0, subtract=False):
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {device}")
        # Without this the driver hands back a queue several frames deep, which
        # shows up as input lag rather than as dropped frames.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.subtract = subtract
        self.dark = None            # most recent unlit frame
        self._dark_t = 0.0
        self.lit = None             # most recent lit frame, before subtraction
        # read() returns None for an unlit frame and for a failed read alike, so
        # only this class can tell them apart. Counted here because a camera that
        # has stopped delivering and a camera that is merely strobing look the
        # same from outside, and the difference is worth being able to see.
        self.lit_count = 0
        self.unlit_count = 0
        self.fail_count = 0
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
        """One lit frame, or None if this frame was unlit or the read failed.

        Unlit frames are never returned, but with subtraction on they are kept as
        the ambient reference for the next lit one.
        """
        g = self._read_gray()
        if g is None:
            self.fail_count += 1
            return None
        if g.mean() < self.lit_cutoff:
            self.unlit_count += 1
            self.dark = g
            self._dark_t = time.time()
            return None
        self.lit_count += 1
        self.lit = g
        if not self.subtract:
            return g
        if self.dark is None or time.time() - self._dark_t > self.DARK_MAX_AGE:
            return g        # no fresh ambient reference; raw beats nothing
        # Saturating subtract: ambient brighter than emitter+ambient clamps to 0
        # rather than wrapping around into spurious highlights.
        return cv2.subtract(g, self.dark)

    def read_pair(self):
        """One lit frame and its ambient-subtracted twin, or None.

        Both halves of the comparison from the SAME exposure - running the two
        modes in separate sessions compares different pose and lighting as well
        as different preprocessing, which is what made the question hard to
        settle. Requires ``subtract=False``, or the raw half is already
        subtracted and there is nothing to compare.
        """
        g = self.read()
        if g is None:
            return None
        if self.dark is None or time.time() - self._dark_t > self.DARK_MAX_AGE:
            return g, g         # no fresh reference: identical halves, honest
        return g, cv2.subtract(g, self.dark)

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.release()
