#!/usr/bin/env python
"""Does subtracting the unlit strobe frame recover contrast in your lighting?

The emitter strobes, so every other frame shows the scene lit by ambient alone.
Subtracting it should cancel whatever is not the emitter - which is exactly the
problem when ambient infrared floods the sensor and the retroreflection stops
standing out from the skin around it.

Whether that helps depends entirely on how much ambient there is, so this
measures both on the same frames rather than assuming. Run it wherever detection
is struggling.

  uv run ambient_test.py --seconds 15
"""

import argparse
import time

from bgate import qtfix  # noqa: F401  - before cv2

import cv2
import numpy as np

from bgate.camera import IRCamera, DEFAULT_DEVICE
from bgate.detect import MediaPipeLocator, RollTracker
from bgate.model import make_landmarker


def glint_contrast(gray, eye, k=0.7):
    """How far the brightest point near the eye rises above its surroundings.
    This is the signal the whole detector rests on."""
    r = max(int(eye.width * k), 5)
    x0, y0 = int(max(0, eye.cx - r)), int(max(0, eye.cy - r))
    x1, y1 = int(min(gray.shape[1], eye.cx + r)), int(min(gray.shape[0], eye.cy + r))
    roi = gray[y0:y1, x0:x1]
    if roi.size < 25:
        return None
    blur = cv2.GaussianBlur(roi, (0, 0), 1.0)
    return float(blur.max()) - float(np.median(blur))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args()

    # Subtraction off: we want the raw lit frame and the dark one separately, and
    # will do the arithmetic here so both are measured on identical pixels.
    cam = IRCamera(args.device, subtract=False)
    locator = RollTracker(MediaPipeLocator(make_landmarker()))
    print(f"strobe cutoff {cam.lit_cutoff:.1f}\nhold still and look at the camera "
          f"for {args.seconds:.0f}s ...\n")

    raw_c, sub_c, raw_stats, sub_stats = [], [], [], []
    found_raw = found_sub = frames = 0
    t0 = time.time()
    while time.time() - t0 < args.seconds:
        lit = cam.read()
        if lit is None or cam.dark is None:
            continue
        frames += 1
        sub = cv2.subtract(lit, cam.dark)
        raw_stats.append((lit.mean(), lit.std()))
        sub_stats.append((sub.mean(), sub.std()))
        for tag, img, store in (("raw", lit, raw_c), ("sub", sub, sub_c)):
            det = locator.locate(img)
            if not det:
                continue
            if tag == "raw":
                found_raw += 1
            else:
                found_sub += 1
            for eye in det.eyes.values():
                c = glint_contrast(img, eye)
                if c is not None:
                    store.append(c)
        print(f"  {frames} frames", end="\r", flush=True)
    cam.release()

    if not frames:
        raise SystemExit("no paired lit/dark frames - is this the IR camera?")
    r, s = np.array(raw_stats), np.array(sub_stats)
    print(f"\n{frames} lit frames, each paired with the unlit frame beside it\n")
    print(f"  {'':<22}{'raw':>10}{'ambient-subtracted':>22}")
    print(f"  {'frame brightness':<22}{r[:,0].mean():>10.1f}{s[:,0].mean():>22.1f}")
    print(f"  {'frame contrast (sd)':<22}{r[:,1].mean():>10.1f}{s[:,1].mean():>22.1f}")
    print(f"  {'face found':<22}{found_raw:>10}{found_sub:>22}")
    if raw_c and sub_c:
        rc, sc = np.median(raw_c), np.median(sub_c)
        print(f"  {'glint contrast':<22}{rc:>10.1f}{sc:>22.1f}")
        print(f"\n  glint contrast {'improved' if sc > rc else 'got worse'} "
              f"by {abs(sc - rc) / max(rc, 1e-6):.0%}")
        print("  (training frames sat around 40; below ~20 the eye stops"
              " standing out from the skin)")
    else:
        print("  no eye located in one of the two - nothing to compare")


if __name__ == "__main__":
    main()
