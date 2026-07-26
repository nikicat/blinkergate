#!/usr/bin/env python
"""Does pooling beat the pixel template, and is it actually shift-invariant?

Two questions, and the second is the one that matters. A model can score well on
crops cut exactly where it was trained and still be a template matcher; the test
is what happens when the crop moves. So every variant is fitted on unshifted
crops and then scored at deliberate offsets, with whole capture bursts held out.

  uv run pooled_eval.py
  uv run pooled_eval.py --shifts 0,2,4,6,8 --direction vertical
  uv run pooled_eval.py --ablate           # which channels earn their place

Crops are cut once per (shift, direction) and cached under dataset/.shifts.npz,
because the expensive part is decoding 600 PNGs and it does not change between
runs. Folds are independent, so the whole (variant x fold) grid runs in parallel.
"""

import argparse
import os

import cv2
import numpy as np

from bgate import data, model, pooled
from bgate.data import Dataset, crop_at, frames_dir

CACHE = ".shifts.npz"


def _cut_all(ds, samples, shifts, direction, seed_base=0):
    """``{shift: [crop or None]}``, every frame decoded exactly once.

    Re-cut from the source rather than translating the finished 40x24: shifting
    the small crop would drag in border padding instead of the pixels genuinely
    next to the eye, and teach the comparison about the padding.
    """
    # One direction per SAMPLE, reused at every magnitude, so the shift levels
    # are nested: 4px is the 3px crop pushed further along the same line. Drawing
    # a fresh direction per level instead makes each column an independent
    # sample, and the curve then wanders non-monotonically - measured, the pixel
    # model read 54.4% at 3px and 65.0% at 4px, which is noise, not tolerance.
    rng = np.random.default_rng(seed_base)
    if direction == "random":
        th = rng.uniform(0, 2 * np.pi, len(samples))
        unit = np.stack([np.cos(th), np.sin(th)], axis=1)
    elif direction == "horizontal":
        unit = np.stack([rng.choice([-1.0, 1.0], len(samples)),
                         np.zeros(len(samples))], axis=1)
    else:
        unit = np.stack([np.zeros(len(samples)),
                         rng.choice([-1.0, 1.0], len(samples))], axis=1)

    by_frame = {}
    for i, s in enumerate(samples):
        by_frame.setdefault(s["frame"], []).append(i)

    out = {s: [None] * len(samples) for s in shifts}
    for frame, group in by_frame.items():
        gray = cv2.imread(os.path.join(frames_dir(ds.dir), frame),
                          cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        for i in group:
            cx, cy, width, roll = (float(v) for v in samples[i]["geom"])
            for shift in shifts:
                out[shift][i], _quad = crop_at(
                    gray, cx + shift * unit[i, 0], cy + shift * unit[i, 1],
                    width, roll=roll)
    return out


def shifted_crops(ds, samples, shifts, direction, refresh=False):
    """Cached ``_cut_all``. The key covers everything that changes the pixels,
    so a stale cache is a mismatch rather than a silent wrong answer."""
    path = os.path.join(ds.dir, CACHE)
    key = (f"{data.CROP_CONVENTION}|{direction}|{len(samples)}|"
           f"{','.join(str(s) for s in shifts)}|"
           f"{samples[0]['id']}|{samples[-1]['id']}")
    if not refresh and os.path.exists(path):
        try:
            z = np.load(path, allow_pickle=False)
            if str(z["key"]) == key:
                blank = z["blank"]
                return {s: [None if blank[j, i] else z["crops"][j, i]
                            for i in range(len(samples))]
                        for j, s in enumerate(shifts)}
        except Exception:
            pass
    print("  cutting crops (cached for next time) ...", flush=True)
    out = _cut_all(ds, samples, shifts, direction)
    stack = np.zeros((len(shifts), len(samples), data.IN_H, data.IN_W), np.uint8)
    blank = np.zeros((len(shifts), len(samples)), bool)
    for j, s in enumerate(shifts):
        for i, c in enumerate(out[s]):
            if c is None:
                blank[j, i] = True
            else:
                stack[j, i] = c
    np.savez(path, key=key, crops=stack, blank=blank)
    return out


# -- the variants ---------------------------------------------------------
# Module level and parameterised by a plain string, so a fold can be shipped to
# a worker process. A closure could not be.

def _fit_variant(kind, bank, crops, y, epochs):
    rows = [{"crop": c, "label": data.CLOSED if v else data.OPEN,
             "burst": "b", "frame": "b"} for c, v in zip(crops, y)]
    if kind == "pixel":
        coef, b, _a = model.fit(rows)
        return model.EyeModel(coef, b)
    kern, coef, b, _a = pooled.fit(rows, bank=bank, epochs=epochs,
                                   learn_kernels=(kind == "learned"), cv=False)
    return pooled.PooledModel(kern, coef, b)


def _fold_job(kind, bank, train_crops, ytr, test_by_shift, yte, shifts, epochs):
    """One variant, one fold: fit unshifted, score at every shift."""
    fitted = _fit_variant(kind, bank, train_crops, ytr, epochs)
    hits = {}
    for s in shifts:
        crops = test_by_shift[s]
        keep = [j for j, c in enumerate(crops) if c is not None]
        if not keep:
            hits[s] = (0, 0)
            continue
        p = fitted.p_closed([crops[j] for j in keep])
        hits[s] = (int(((p > 0.5) == (yte[keep] > 0.5)).sum()), len(keep))
    return hits


def run(variants, crops_by_shift, y, groups, shifts, epochs, n_jobs):
    """Every (variant, fold) pair in parallel; they share nothing."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold

    base = crops_by_shift[0]
    splits = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(
        base, y, groups))
    jobs, owner = [], []
    for name, kind, bank in variants:
        for tr, te in splits:
            jobs.append(delayed(_fold_job)(
                kind, bank, [base[i] for i in tr], y[tr],
                {s: [crops_by_shift[s][i] for i in te] for s in shifts},
                y[te], shifts, epochs))
            owner.append(name)
    results = Parallel(n_jobs=n_jobs, verbose=0)(jobs)

    totals = {name: {s: [0, 0] for s in shifts} for name, _k, _b in variants}
    for name, hits in zip(owner, results):
        for s in shifts:
            totals[name][s][0] += hits[s][0]
            totals[name][s][1] += hits[s][1]
    return totals


def table(variants, totals, shifts):
    print(f"\n  {'model':<24}" + "".join(f"{s:>7}px" for s in shifts))
    for name, _k, _b in variants:
        line = f"  {name:<24}"
        for s in shifts:
            hit, tot = totals[name][s]
            line += f"{hit / tot:>9.1%}" if tot else f"{'-':>9}"
        print(line)


def main():
    ap = argparse.ArgumentParser(
        description="Pooled filter responses against the pixel template.")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--shifts", default="0,2,3,4,6,8")
    ap.add_argument("--direction", default="random",
                    choices=["random", "horizontal", "vertical"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--refresh", action="store_true", help="ignore the crop cache")
    ap.add_argument("--ablate", action="store_true",
                    help="blob channel against ridge channel")
    ap.add_argument("--shuffle", action="store_true",
                    help="negative control: permute the labels. Anything above "
                         "the majority class is a leak, not a model.")
    ap.add_argument("--kernels", metavar="PNG", default="",
                    help="write the learned kernels beside their starting shapes")
    args = ap.parse_args()

    shifts = [int(s) for s in args.shifts.split(",")]
    if 0 not in shifts:
        raise SystemExit("0 must be among the shifts - it is the baseline")

    ds = Dataset(args.dataset)
    samples = [s for s in ds.trainable() if s.get("geom") is not None]
    y = np.array([1.0 if s["label"] == data.CLOSED else 0.0 for s in samples])
    groups = np.array([s["burst"] for s in samples])
    print(f"{len(samples)} samples in {len(set(groups))} bursts, "
          f"{int((y == 0).sum())} open / {int(y.sum())} closed, "
          f"shift direction: {args.direction}")

    crops_by_shift = shifted_crops(ds, samples, shifts, args.direction,
                                   args.refresh)

    if args.shuffle:
        # Held-out accuracy of 99.5% on a dataset where the pixel model reads
        # 96.3% is worth disbelieving until this says otherwise. With the labels
        # permuted there is nothing left to learn, so anything above the majority
        # class means the pipeline is leaking rather than the model working.
        y = np.random.default_rng(0).permutation(y)
        major = max(y.mean(), 1 - y.mean())
        print(f"  LABELS PERMUTED - the majority class is {major:.1%}; "
              f"a working pipeline scores about that and no more")

    K = pooled.KERNEL
    if args.ablate:
        variants = [
            ("blobs only (2)", "learned",
             np.stack([pooled.log_kernel(K, 1.5), pooled.log_kernel(K, 2.4)])),
            ("ridges only (3)", "learned",
             np.stack([pooled.ridge_kernel(K, 1.4, 3.2, a)
                       for a in (0.0, 22.0, -22.0)])),
            ("both (5)", "learned", pooled.default_bank(K)),
        ]
    else:
        variants = [
            ("pixel template", "pixel", None),
            ("pooled, fixed kernels", "fixed", pooled.default_bank(K)),
            ("pooled, learned kernels", "learned", pooled.default_bank(K)),
        ]

    totals = run(variants, crops_by_shift, y, groups, shifts, args.epochs,
                 args.jobs)
    table(variants, totals, shifts)
    print("\n  fitted on unshifted crops, whole capture bursts held out;"
          f"\n  shift is in source px, {args.direction} direction.")

    if args.kernels:
        rows = [{"crop": s["crop"], "label": s["label"], "burst": s["burst"],
                 "frame": s["frame"]} for s in samples]
        kern, coef, _b, acc = pooled.fit(rows, epochs=args.epochs,
                                         n_jobs=args.jobs)
        print(f"\n  final fit on everything: held-out {acc:.1%}, "
              f"{kern.size + coef.size + 1} weights "
              f"(pixel model: {data.IN_H * data.IN_W + 1})")
        start = pooled.default_bank(kern.shape[-1])
        names = ["blob small", "blob large", "ridge 0", "ridge +22", "ridge -22"]
        print(f"\n  {'kernel':<12}{'moved':>8}{'|w| head':>10}")
        for i, nm in enumerate(names[:len(kern)]):
            print(f"  {nm:<12}{float(np.linalg.norm(kern[i] - start[i])):>8.3f}"
                  f"{float(abs(coef[i])):>10.3f}")
        print("  moved: distance from the textbook kernel it started at"
              " (both unit-norm, so 2.0 is opposite)")
        tiles = []
        for row in (start, kern):
            strip = []
            for k in row:
                v = k / (np.abs(k).max() + 1e-9)
                img = np.zeros((*k.shape, 3), np.uint8)
                img[..., 2] = np.clip(v, 0, 1) * 255
                img[..., 0] = np.clip(-v, 0, 1) * 255
                strip.append(cv2.copyMakeBorder(
                    cv2.resize(img, (88, 88), interpolation=cv2.INTER_NEAREST),
                    4, 4, 4, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
            tiles.append(np.hstack(strip))
        cv2.imwrite(args.kernels, np.vstack(tiles))
        print(f"\n  wrote {args.kernels} - top row as initialised, "
              f"bottom row after fitting (red +, blue -)")


if __name__ == "__main__":
    main()
