#!/usr/bin/env python
"""Run every model generation on the same live frames, side by side.

Switching generations one at a time cannot tell you much: your position and the
lighting change between trials, and that is exactly the variable in question.
This scores them all on identical pixels.

Use it to answer "is it the blur?" - gen-0001 is sharp, later ones are blurred,
and if they disagree on the same frame the difference is the model, not the
conditions.

  uv run compare_gens.py                 20s, both eyes
  uv run compare_gens.py --seconds 40

Open and close your eyes a few times while it runs. It prints P(open) per
generation and, at the end, how often each called a closed eye closed.
"""

import argparse
import time

from bgate import qtfix  # noqa: F401  - before cv2

import cv2
import numpy as np

from bgate import data
from bgate.camera import IRCamera, DEFAULT_DEVICE
from bgate.model import make_landmarker
from bgate.registry import Registry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--models", default="models")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--save", metavar="DIR",
                    help="also save the frames, so disagreements can be replayed")
    args = ap.parse_args()

    reg = Registry(args.models)
    gens = reg.ids()
    if not gens:
        raise SystemExit("no generations - train one first")
    models = {g: reg.load(g) for g in gens}
    print("generations:")
    for g, m in models.items():
        print(f"  {g}  blur={m.blur:<4} held-out={m.accuracy:.2%}"
              f"{'   <- active' if g == reg.active() else ''}")

    landmarker = make_landmarker()
    cam = IRCamera(args.device)
    import mediapipe as mp
    if args.save:
        data.ensure(args.save)

    print(f"\nopen and close your eyes for {args.seconds:.0f}s ...\n")
    header = "  " + "".join(f"{g:>22}" for g in gens)
    print(header + "\n  " + "".join(f"{'IMG-L   IMG-R':>22}" for _ in gens))

    log = {g: [] for g in gens}
    t0 = time.time()
    n = 0
    while time.time() - t0 < args.seconds:
        gray = cam.read()
        if gray is None:
            continue
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)))
        if not res.face_landmarks:
            print("  no face", end="\r")
            continue
        lm = res.face_landmarks[0]
        line = "  "
        for g in gens:
            probs, _crops, _boxes = models[g].eyes(gray, lm)
            log[g].append(probs)
            line += f"{probs[0]:>10.2f}{probs[1]:>12.2f}"
        print(line, end="\r", flush=True)
        if args.save:
            cv2.imwrite(f"{args.save}/frames/cmp_{int(t0 * 1000)}_{n:04d}.png", gray)
        n += 1

    cam.release()
    print("\n\nsummary over", n, "frames")
    print(f"  {'generation':<12}{'blur':>6}{'frames called shut':>22}{'min P(open)':>14}")
    for g in gens:
        a = np.array(log[g], dtype=float)
        shut = float((a < 0.5).mean())
        print(f"  {g:<12}{models[g].blur:>6}{shut:>21.1%}{np.nanmin(a):>14.3f}")
    print("\nIf you closed your eyes for roughly half the run, a working generation")
    print("should report close to that. One reporting ~0% never saw them shut.")


if __name__ == "__main__":
    main()
