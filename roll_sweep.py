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


def _rows(crops, keys, labels, idx):
    """Sample dicts in the shape the fitters want.

    Every row gets the same burst on purpose: the outer fold has already held
    bursts out, so an inner cross-validation would be five extra fits for a
    number nobody reads.
    """
    return [{"crop": crops[i],
             "label": labels[keys[i]],
             "burst": "one", "frame": "one"} for i in idx]


def _score_job(kind, train_crops, train_keys, train_idx, per_angle, labels,
               angles):
    """One (variant, kind, fold): fit on upright crops, score at every angle."""
    from bgate.registry import fit_model

    fitted, _acc = fit_model(kind, _rows(train_crops, train_keys, labels,
                                         train_idx), cv=False)
    hits = {}
    for a in angles:
        crops_a, keys_a, idx_a = per_angle[a]
        if not idx_a:
            hits[a] = (0, 0)
            continue
        p = fitted.p_closed([crops_a[i] for i in idx_a])
        truth = np.array([labels[keys_a[i]] == data.CLOSED for i in idx_a])
        hits[a] = (int(((p > 0.5) == truth).sum()), len(idx_a))
    return hits


def report_accuracy(labels_, angles, crops, keys, labels, kinds, n_jobs=-1):
    """Verdicts per variant per angle, whole capture bursts held out.

    Each variant is fitted on its own upright crops, because a model fitted on one
    crop convention says nothing about another - which is the same reason blur and
    normalisation travel inside a generation. Every (variant, kind, fold) is
    independent, so they all run at once.
    """
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold

    def usable(vname, a):
        """Positions with a trainable label. Both eyes of a frame get cropped,
        but one of them may be skipped, and a skip has no true class to score."""
        return [i for i, k in enumerate(keys[(vname, a)]) if k in labels]

    jobs, owner = [], []
    for vname in labels_:
        base = usable(vname, 0)
        if not base:
            continue
        base_keys = [keys[(vname, 0)][i] for i in base]
        y = np.array([labels[k] == data.CLOSED for k in base_keys], dtype=int)
        groups = np.array([data.burst_of(f) for f, _s in base_keys])
        if len(set(y)) < 2 or len(set(groups)) < 2:
            continue
        splits = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(
            base, y, groups))
        for kind in kinds:
            for train, test in splits:
                held = set(groups[test])
                per_angle = {}
                for a in angles:
                    sel = [i for i in usable(vname, a)
                           if data.burst_of(keys[(vname, a)][i][0]) in held]
                    per_angle[a] = (crops[(vname, a)], keys[(vname, a)], sel)
                jobs.append(delayed(_score_job)(
                    kind, crops[(vname, 0)], keys[(vname, 0)],
                    [base[i] for i in train], per_angle, labels, angles))
                owner.append((vname, kind))

    results = Parallel(n_jobs=n_jobs)(jobs)
    totals = {}
    for key, hits in zip(owner, results):
        acc = totals.setdefault(key, {a: [0, 0] for a in angles})
        for a in angles:
            acc[a][0] += hits[a][0]
            acc[a][1] += hits[a][1]

    print(f"\n  {'variant':<16}{'model':<9}" + "".join(f"{a:>8}" for a in angles))
    for vname in labels_:
        for kind in kinds:
            if (vname, kind) not in totals:
                continue
            line = f"  {vname:<16}{kind:<9}"
            for a in angles:
                hit, tot = totals[(vname, kind)][a]
                line += f"{hit / tot:>8.1%}" if tot else f"{'-':>8}"
            print(line)
    print("\n  bursts held out; each row fitted on its own upright crops.")


CACHE = ".rollsweep.npz"


def cached_sweep(frames, ds_dir, locators, variants, angles, refresh=False):
    """``sweep`` with its result kept on disk.

    The detection pass is minutes of MediaPipe - every frame at every angle
    through every locator - and it does not change between runs unless the crop
    convention or the frame set does, both of which are in the key.
    """
    path = os.path.join(ds_dir, CACHE)
    key = (f"{data.CROP_CONVENTION}|{len(frames)}|{frames[0]}|{frames[-1]}|"
           f"{','.join(str(a) for a in angles)}|"
           f"{','.join(v[0] for v in variants)}")
    if not refresh and os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            if str(z["key"]) == key:
                err, crops, keys = {}, {}, {}
                for vname, _l, _w in variants:
                    for a in angles:
                        tag = f"{vname}|{a}"
                        err[(vname, a)] = z[f"err|{tag}"]
                        crops[(vname, a)] = list(z[f"crops|{tag}"])
                        keys[(vname, a)] = [tuple(r) for r in z[f"keys|{tag}"]]
                print(f"reusing the cached detection pass ({path})")
                return int(z["n_base"]), err, crops, keys
        except Exception:
            pass

    n_base, err, crops, keys = sweep(frames, ds_dir, locators, variants, angles)
    out = {"key": key, "n_base": n_base}
    for vname, _l, _w in variants:
        for a in angles:
            tag = f"{vname}|{a}"
            out[f"err|{tag}"] = np.array(err[(vname, a)], dtype=np.float32)
            out[f"crops|{tag}"] = (np.stack(crops[(vname, a)])
                                   if crops[(vname, a)]
                                   else np.zeros((0, data.IN_H, data.IN_W), np.uint8))
            out[f"keys|{tag}"] = np.array(keys[(vname, a)] or [["", ""]], dtype="<U64")
    np.savez(path, **out)
    return n_base, err, crops, keys


def main():
    ap = argparse.ArgumentParser(
        description="Measure what head roll costs, against known ground truth.")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--angles", default=",".join(str(a) for a in ANGLES))
    ap.add_argument("--frames", type=int, default=0,
                    help="sample this many frames (0 = all)")
    ap.add_argument("--score", action="store_true",
                    help="also score verdicts - slower, and the number that counts")
    ap.add_argument("--kinds", default="pixel,pooled",
                    help="model kinds to score")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--refresh", action="store_true",
                    help="ignore the cached detection pass")
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
    n_base, err, crops, keys = cached_sweep(frames, args.dataset, locators,
                                            VARIANTS, angles, args.refresh)
    print(f"\n{n_base} frames had a baseline fit to compare against")
    if locators["refit"].refits:
        refit = locators["refit"]
        print(f"the refit ran on {refit.refits} fits and was rejected on "
              f"{refit.rejected} as having moved too far")

    report_error(names, angles, err)
    if args.score:
        report_accuracy(names, angles, crops, keys, labels,
                        [k for k in args.kinds.split(",") if k], args.jobs)
    else:
        print("\n  --score adds what this costs in verdicts, which is the number"
              " that decides.")


if __name__ == "__main__":
    main()
