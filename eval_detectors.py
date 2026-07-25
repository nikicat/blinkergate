#!/usr/bin/env python
"""Score face detectors on the frames where detection actually failed.

Ordinary captures are useless for this: the app only records them when a face was
found, so the dataset is a survey of the cases that already work. Press `h` in the
live view on frames it misses, then run this.

Ground truth is that a face is present in every frame here - you asserted that by
pressing the key - so the only number that matters is what fraction each variant
finds.

  uv run eval_detectors.py
  uv run eval_detectors.py --montage misses.png     # look at what still fails
"""

import argparse
import os

import cv2
import numpy as np

from bgate.data import Dataset, frames_dir
from bgate.detect import MediaPipeLocator, YuNetLocator

YUNET = "face_detection_yunet.onnx"


def roll_variants(gray, angles):
    """The same frame rotated. Face detectors are trained on upright faces, so
    cancelling head roll is the cheapest thing that can recover a miss."""
    h, w = gray.shape[:2]
    centre = (w / 2, h / 2)
    for angle in angles:
        if angle == 0:
            yield angle, gray
        else:
            m = cv2.getRotationMatrix2D(centre, angle, 1.0)
            yield angle, cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--montage", metavar="PNG", help="write the remaining misses")
    ap.add_argument("--angles", default="0,-20,20,-35,35",
                    help="roll angles to try, degrees")
    args = ap.parse_args()

    ds = Dataset(args.dataset)
    cases = ds.detector_cases()
    if not cases:
        raise SystemExit(
            "no detector cases yet - press h in the live view on frames it misses")
    angles = [float(a) for a in args.angles.split(",")]

    mp_loc = MediaPipeLocator()
    yn_loc = YuNetLocator(YUNET, (640, 360)) if os.path.exists(YUNET) else None
    if yn_loc is None:
        print(f"{YUNET} missing - run: uv run fetch_model.py\n")

    variants = {"mediapipe": [], "mediapipe+roll": []}
    if yn_loc:
        variants |= {"yunet": [], "yunet+roll": []}
    misses = []

    for case in cases:
        gray = cv2.imread(os.path.join(frames_dir(args.dataset), case["frame"]),
                          cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        found = {}
        for name, loc in (("mediapipe", mp_loc), ("yunet", yn_loc)):
            if loc is None:
                continue
            found[name] = bool(loc.locate(gray))
            # Roll: accept if ANY rotation finds it. A real implementation would
            # track the last known roll rather than sweep, but this measures
            # whether roll is the thing standing in the way.
            hit = found[name]
            if not hit:
                for _angle, rot in roll_variants(gray, angles[1:]):
                    if loc.locate(rot):
                        hit = True
                        break
            found[name + "+roll"] = hit
        for k, v in found.items():
            variants[k].append(v)
        if not any(found.values()):
            misses.append((case["frame"], gray))

    n = len(cases)
    print(f"{n} frames you marked as 'face present, detection struggled'\n")
    print(f"  {'variant':<18}{'found':>8}{'rate':>9}")
    for name, hits in variants.items():
        if hits:
            print(f"  {name:<18}{sum(hits):>8}{sum(hits) / len(hits):>9.0%}")
    both = [any(v[i] for v in variants.values()) for i in range(n)]
    print(f"  {'any of them':<18}{sum(both):>8}{sum(both) / max(n, 1):>9.0%}")
    print(f"\n  none of them found a face in {len(misses)} frames")

    if args.montage:
        if not misses:
            print(f"  nothing to write to {args.montage} - every frame was found")
            return
        Z = 2
        tiles = [cv2.resize(g, (g.shape[1] // Z, g.shape[0] // Z)) for _, g in misses[:12]]
        # Pad the last row rather than dropping it: with fewer than a full row of
        # misses the obvious loop writes nothing at all, silently.
        cols = min(4, len(tiles))
        blank = np.zeros_like(tiles[0])
        while len(tiles) % cols:
            tiles.append(blank)
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(args.montage, np.vstack(rows))
        print(f"  wrote {args.montage} ({len(misses)} frames)")


if __name__ == "__main__":
    main()
