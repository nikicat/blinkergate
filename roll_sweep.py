#!/usr/bin/env python
"""What head roll costs, measured against exactly known ground truth.

Rotating a labelled frame by a known angle gives the true eye position for free:
it is the same affine that made the image. So this can ask the question a
detection rate cannot - the face is found in every frame here, the question is
whether the eye is found *within* it - and it can ask it without labelling
anything by hand.

Two numbers come out. Localisation error, in source pixels, which matters because
the classifier is a template matcher with a cliff at 4px. And the verdicts, scored
against the labels with whole capture bursts held out, which is the one that
decides anything.

  uv run roll_sweep.py                  # localisation error per variant
  uv run roll_sweep.py --score          # and what it costs in accuracy (slow)

The sweep is synthetic, which is the point: a real tilted head brings its own
change of expression, lighting and gaze along with the tilt, and then no error can
be attributed to the roll. It is also the limit - rotating a frame cannot
reproduce the perspective change of a genuinely tilted head, so treat these as the
cost of roll alone.

The variants here wrap the bare mesh, not `detect.default_locator`. Deliberately:
the cascade's other stages exist to rescue frames that would otherwise be lost,
and letting them in would mix a detection rate back into a measurement whose whole
point is that a detection rate cannot see this.
"""

import argparse
import math
import os

import cv2
import numpy as np

from bgate import data
from bgate.data import Dataset, crop_at, frames_dir
from bgate.detect import LevelRefit, MediaPipeLocator, rotate_frame
from bgate.model import features, make_landmarker

ANGLES = (-35, -20, 0, 20, 35)

# (label, locator, levelled width). The two width conventions come off the same
# detection, so a locator runs once per frame however many variants read it.
VARIANTS = (
    ("as found", "mesh", False),        # what ran before any of this
    ("as found/level", "mesh", True),
    ("refit", "refit", False),
    ("refit/level", "refit", True),
)


def sweep(frames, ds_dir, locators, variants, angles, progress=True):
    """Per (variant, angle): localisation errors and the crops that resulted."""
    err = {(v[0], a): [] for v in variants for a in angles}
    crops = {(v[0], a): [] for v in variants for a in angles}
    keys = {(v[0], a): [] for v in variants for a in angles}
    n_base = 0

    for i, fn in enumerate(frames):
        gray = cv2.imread(os.path.join(frames_dir(ds_dir), fn), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        # Truth is the upright fit of the plain mesh: the same detector, the same
        # frame, nothing rotated. Every error below is what the rotation added.
        base = locators["mesh"].locate(gray)
        if not base:
            continue
        n_base += 1
        base_eyes, _roll = data.eye_geometry(base.landmarks, *gray.shape[1::-1])

        for a in angles:
            if a:
                rotated, m = rotate_frame(gray, a)
            else:
                rotated, m = gray, np.array([[1.0, 0, 0], [0, 1.0, 0]])
            found = {name: loc.locate(rotated) for name, loc in locators.items()}
            for vname, lname, level_width in variants:
                det = found[lname]
                if not det:
                    continue
                eyes, roll = data.eye_geometry(det.landmarks, *rotated.shape[1::-1],
                                               level_width=level_width)
                th = math.radians(roll)
                cos, sin = math.cos(th), math.sin(th)
                for side, (cx, cy, width) in eyes.items():
                    bx, by, bw = base_eyes[side]
                    truth = m @ np.array([bx, by, 1.0])
                    e = np.array([cx, cy]) - truth
                    err[(vname, a)].append((
                        e[0] * cos + e[1] * sin,        # along the eye line
                        -e[0] * sin + e[1] * cos,       # +down the face
                        width / max(bw, 1e-6),
                        roll + a,                       # residual tilt
                    ))
                    crop, _quad = crop_at(rotated, cx, cy, width, roll=roll)
                    if crop is not None:
                        crops[(vname, a)].append(crop)
                        keys[(vname, a)].append((fn, side))
        if progress and i % 25 == 0:
            print(f"  {i}/{len(frames)}", flush=True)
    return n_base, err, crops, keys


def report_error(labels_, angles, err):
    print(f"\n  {'variant':<16}{'roll':>6}{'n':>6}{'along':>8}{'perp':>8}"
          f"{'|err|':>8}{'p90':>8}{'width':>8}{'residual':>10}")
    for vname in labels_:
        for a in angles:
            v = np.array(err[(vname, a)], dtype=float)
            if not len(v):
                print(f"  {vname:<16}{a:>6}{0:>6}")
                continue
            mag = np.hypot(v[:, 0], v[:, 1])
            print(f"  {vname:<16}{a:>6}{len(v):>6}{v[:,0].mean():>8.2f}"
                  f"{v[:,1].mean():>8.2f}{mag.mean():>8.2f}"
                  f"{np.percentile(mag, 90):>8.2f}{v[:,2].mean():>8.3f}"
                  f"{v[:,3].mean():>10.1f}")
        print()
    print("  along/perp: mean signed error in the levelled frame, source px."
          "  perp is +down the face.")
    print("  width: measured eye width as a fraction of the true one"
          " - the crop's scale.")
    print("  residual: degrees of tilt the crop is left holding, after"
          " levelling by the measured roll.")


def report_accuracy(labels_, angles, crops, keys, labels):
    """Verdicts per variant per angle, whole capture bursts held out.

    Each variant is fitted on its own upright crops, because a model fitted on one
    crop convention says nothing about another - which is the same reason blur and
    normalisation travel inside a generation.
    """
    from sklearn.model_selection import GroupKFold

    from bgate.model import _make_clf

    def usable(vname, a):
        """Positions with a trainable label. Both eyes of a frame get cropped,
        but one of them may be skipped, and a skip has no true class to score."""
        return [i for i, k in enumerate(keys[(vname, a)]) if k in labels]

    print(f"\n  {'variant':<16}" + "".join(f"{a:>8}" for a in angles))
    for vname in labels_:
        base = usable(vname, 0)
        if not base:
            continue
        base_keys = [keys[(vname, 0)][i] for i in base]
        X = features([crops[(vname, 0)][i] for i in base])
        y = np.array([labels[k] == data.CLOSED for k in base_keys], dtype=int)
        groups = np.array([data.burst_of(f) for f, _s in base_keys])
        if len(set(y)) < 2 or len(set(groups)) < 2:
            continue

        hits = {a: [0, 0] for a in angles}
        for train, test in GroupKFold(n_splits=min(5, len(set(groups)))).split(
                X, y, groups):
            clf = _make_clf()
            clf.fit(X[train], y[train])
            held = set(groups[test])
            for a in angles:
                sel = [i for i in usable(vname, a)
                       if data.burst_of(keys[(vname, a)][i][0]) in held]
                if not sel:
                    continue
                Xa = features([crops[(vname, a)][i] for i in sel])
                ya = np.array([labels[keys[(vname, a)][i]] == data.CLOSED
                               for i in sel], dtype=int)
                hits[a][0] += int((clf.predict(Xa) == ya).sum())
                hits[a][1] += len(sel)
        line = f"  {vname:<16}"
        for a in angles:
            hit, tot = hits[a]
            line += f"{hit / tot:>8.1%}" if tot else f"{'-':>8}"
        print(line)
    print("\n  bursts held out; each variant fitted on its own upright crops.")


def main():
    ap = argparse.ArgumentParser(
        description="Measure what head roll costs, against known ground truth.")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--angles", default=",".join(str(a) for a in ANGLES))
    ap.add_argument("--frames", type=int, default=0,
                    help="sample this many frames (0 = all)")
    ap.add_argument("--score", action="store_true",
                    help="also score verdicts - slower, and the number that counts")
    args = ap.parse_args()

    angles = [int(a) for a in args.angles.split(",")]
    if 0 not in angles:
        raise SystemExit("0 must be among the angles - it is the baseline")

    ds = Dataset(args.dataset)
    labels = {(r["frame"], r["side"]): r["label"] for r in ds.rows
              if r["label"] in data.TRAINABLE}
    frames = sorted({f for f, _s in labels})
    if not frames:
        raise SystemExit("no labelled frames - capture some first")
    if args.frames:
        frames = frames[::max(1, len(frames) // args.frames)][:args.frames]

    mesh = MediaPipeLocator(make_landmarker())
    locators = {"mesh": mesh, "refit": LevelRefit(mesh)}
    names = [v[0] for v in VARIANTS]

    print(f"{len(frames)} labelled frames, {len(angles)} angles, "
          f"{len(VARIANTS)} variants")
    n_base, err, crops, keys = sweep(frames, args.dataset, locators, VARIANTS,
                                     angles)
    print(f"\n{n_base} frames had a baseline fit to compare against")

    refit = locators["refit"]
    print(f"the refit ran on {refit.refits} fits and was rejected on "
          f"{refit.rejected} as having moved too far")

    report_error(names, angles, err)
    if args.score:
        report_accuracy(names, angles, crops, keys, labels)
    else:
        print("\n  --score adds what this costs in verdicts, which is the number"
              " that decides.")


if __name__ == "__main__":
    main()
