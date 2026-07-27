"""Export a model generation for the Rust daemon, and the conformance data that
proves the port agrees with this code.

Two outputs, deliberately from one script:

* ``model.json`` - weights **and** the preprocessing they were fitted with (crop
  geometry, the crop convention, normalisation, detection confidence). Every one
  of those is a convention that would otherwise silently invalidate the weights:
  a model fitted one way and applied another loses accuracy without crashing,
  which is the failure mode this file exists to prevent.
* ``golden.json`` - fixed inputs and the crops and probabilities *this* code
  produces for them, asserted by ``rust/gate/tests/golden.rs``.

Everything in the golden file is synthetic, including the weights: they come from
a seeded generator, not from a fit. That is on purpose twice over. Nothing derived
from your face is committed, and the test then asserts what it is actually for -
that the port's *arithmetic* (crop sampling, standardisation, pooling, the
logistic head) matches, which a fitted model would not test any harder than a
random one. Whether the real model still reads your eyes correctly is a live
question answered by ``bgatectl``, not by a golden file.

The detector half needs no golden data: both implementations call the same
``libmediapipe.so``, and landmarks were measured to agree within 1.14px, iris
centres within 0.62px.

    uv run export_model.py                      # active generation -> model.json
    uv run export_model.py --golden-only        # refresh the golden data alone
"""

import argparse
import json
import os

import cv2
import numpy as np

from bgate import data, model, pooled, registry, ui
from bgate.pooled import PooledModel

FORMAT = "blinkergate-model/1"
GOLDEN_FORMAT = "blinkergate-golden/1"
GOLDEN_PATH = "rust/gate/tests/golden.json"

# The synthetic frame, as an integer formula rather than committed pixels: both
# sides compute it, so there is nothing to keep in sync but the expression.
GOLDEN_W, GOLDEN_H = 96, 72
FORMULA = "((x*7 + y*13) % 251) ^ ((x*y) % 97)"


def rings():
    """bgate's eye rings, checked against MediaPipe's own connection constants.

    ``ui.RING_L`` is 16 integers with no derivation written next to them, and a
    transcription error in one of them moves the crop by a fraction of an eye -
    the kind of defect that has no failure mode and grows smoothly. MediaPipe
    ships the rings as *edges* rather than as ordered loops, so this asserts both
    halves of the claim: the same vertices, and our order is a closed walk of
    those edges.

    The constants live only in MediaPipe's Python wrapper - the C API the Rust
    side calls carries no landmark indices - so the numbers have to be copied.
    Copied and then checked is the difference between a transcription and a magic
    number, and the check travels to the port through golden.json.
    """
    from mediapipe.tasks.python.vision.face_landmarker import FaceLandmarksConnections as C

    out = {}
    for side, ring, conn in (("left", ui.RING_L, C.FACE_LANDMARKS_LEFT_EYE),
                             ("right", ui.RING_R, C.FACE_LANDMARKS_RIGHT_EYE)):
        edges = {frozenset((e.start, e.end)) for e in conn}
        verts = {v for edge in edges for v in edge}
        assert verts == set(ring), (
            f"{side} eye ring disagrees with MediaPipe: "
            f"ours {sorted(set(ring))}, theirs {sorted(verts)}")
        assert all(frozenset((ring[i], ring[(i + 1) % len(ring)])) in edges
                   for i in range(len(ring))), (
            f"{side} eye ring is not a closed walk of MediaPipe's edges")
        out[side] = list(ring)
    return out


def golden_frame(w=GOLDEN_W, h=GOLDEN_H):
    x = np.arange(w)[None, :]
    y = np.arange(h)[:, None]
    return (((x * 7 + y * 13) % 251) ^ ((x * y) % 97)).astype(np.uint8)


def shaped_crop(bright, w=data.IN_W, h=data.IN_H):
    """A crop with the structure of a real one - smooth gradients, a saturated
    disc, a dark bar - rather than formula noise, because that is a different
    pixel distribution for the pooling maximum to land in."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 60.0)
    img += 25 * np.sin(xx / 6.0)
    img[np.abs(yy - 5) < 1.5] -= 35                   # lash line
    if bright:
        img[(xx - w / 2) ** 2 + (yy - h / 2) ** 2 < 16] += 180   # saturates
    return np.clip(img, 0, 255).astype(np.uint8)


def warp_crop(frame, cx, cy, width, roll):
    """The crop, forced through ``cv2.warpAffine``.

    ``data.crop_at`` branches at *exactly* ``roll == 0`` onto an integer box plus
    an ``INTER_AREA`` resize, which is a different resampler. Nothing in the live
    pipeline reaches that branch - roll comes from ``atan2`` of two eye centres
    and is never exactly zero - and the Rust port implements the warp alone, so
    the warp is what the golden data records. The divergence itself is pinned by
    ``crop_box`` below, so this note cannot go stale unattended.
    """
    half_w = max(width * data.PAD_W / 2, 8)
    half_h = half_w * (data.IN_H / data.IN_W) * data.PAD_H
    m = data.level_affine(cx, cy, half_w, half_h, roll, data.IN_W, data.IN_H)
    return cv2.warpAffine(frame, m, (data.IN_W, data.IN_H), flags=cv2.INTER_AREA,
                          borderMode=cv2.BORDER_REPLICATE)


def as_json(obj, gen, meta):
    """The exported form of a pooled model."""
    return {
        "format": FORMAT,
        "kind": obj.kind,
        "generation": gen,
        # NaN is not JSON, and an unfitted model legitimately has no accuracy.
        "accuracy": None if np.isnan(obj.accuracy) else float(obj.accuracy),
        "crop": {
            "in_w": data.IN_W, "in_h": data.IN_H,
            "pad_w": data.PAD_W, "pad_h": data.PAD_H,
            "level_width": bool(obj.level_width),
            "convention": data.CROP_CONVENTION,
        },
        "detector": {"confidence": model.DETECTION_CONFIDENCE},
        "normalise": obj.normalise,
        "kernel_size": int(obj.size),
        "use_position": bool(obj.use_position),
        "kernels": [k.reshape(-1).tolist() for k in obj.kernels],
        "coef": obj.coef.tolist(),
        "intercept": float(obj.intercept),
        "trained_at": meta.get("created_at"),
        "n_train": meta.get("n_train"),
    }


def export(gen, root="models"):
    reg = registry.Registry(root)
    obj = registry.load_model(reg.path(gen, "model.npz"))
    if not isinstance(obj, PooledModel):
        raise SystemExit(
            f"{gen} is a {obj.kind} model. The daemon implements the pooled one "
            "only: the pixel template reads 44.2% at a 4px crop shift against "
            "98.3% pooled, and the pupil is genuinely up to 9.9px off centre.")
    return as_json(obj, gen, reg.meta(gen) or {})


def synthetic_model(seed=7):
    """A pooled model with made-up weights, holding the invariants a fitted one
    holds: kernels zero-mean and unit-norm (so a flat region scores exactly 0 and
    no kernel can inflate its own feature)."""
    rng = np.random.default_rng(seed)
    kernels = np.stack([pooled._prep(rng.standard_normal((pooled.KERNEL,) * 2))
                        for _ in range(5)])
    coef = rng.standard_normal(3 * len(kernels)).astype(np.float32) * 0.5
    return PooledModel(kernels, coef, 0.3, accuracy=float("nan"))


def golden():
    obj = synthetic_model()
    frame = golden_frame()
    cases = []
    # Rolls either side of level, and one at exactly zero - which takes a
    # different path in cv2 (an integer box plus resize rather than a warp) and is
    # therefore the case most likely to disagree with the port.
    for cx, cy, width, roll in [
        (48.0, 36.0, 18.0, 0.0),
        (47.3, 35.6, 18.4, -12.5),
        (52.1, 30.2, 16.0, 7.25),
        (30.5, 44.9, 21.0, 33.0),
        (8.0, 6.0, 19.0, -22.0),        # against the border, so replicate matters
        (90.0, 68.0, 12.0, 5.0),        # and the opposite border
    ]:
        box, _quad = data.crop_at(frame, cx, cy, width, roll=roll)
        assert box is not None, f"degenerate golden case at {cx},{cy}"
        crop = warp_crop(frame, cx, cy, width, roll)
        if roll:
            # Same code, so it had better be the same pixels: this is what makes
            # warp_crop a restatement of crop_at rather than a second convention.
            assert np.array_equal(crop, box), f"warp_crop disagrees at roll {roll}"
        case = {"cx": cx, "cy": cy, "width": width, "roll": roll,
                "crop": crop.reshape(-1).tolist(),
                "p_open": float(obj.predict([crop])[0])}
        if not roll:
            case["crop_box"] = box.reshape(-1).tolist()
        cases.append(case)
    shaped = [{"note": f"disc {'present' if b else 'absent'}",
               "crop": shaped_crop(b).reshape(-1).tolist(),
               "p_open": float(obj.predict([shaped_crop(b)])[0])}
              for b in (True, False)]
    return {
        "format": GOLDEN_FORMAT,
        "frame": {"w": GOLDEN_W, "h": GOLDEN_H, "formula": FORMULA},
        "rings": rings(),
        "model": as_json(obj, "synthetic", {}),
        "cases": cases,
        "shaped": shaped,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gen", help="generation id (default: the active one)")
    ap.add_argument("--root", default="models")
    ap.add_argument("--out", default="model.json")
    ap.add_argument("--golden", action="store_true", help=f"also write {GOLDEN_PATH}")
    ap.add_argument("--golden-only", action="store_true",
                    help="write the golden data and nothing else")
    args = ap.parse_args()

    if not args.golden_only:
        gen = args.gen or registry.Registry(args.root).active()
        if not gen:
            raise SystemExit("no active generation - train one first")
        exported = export(gen, args.root)
        with open(args.out, "w") as fh:
            json.dump(exported, fh, indent=1, allow_nan=False)
            fh.write("\n")
        print(f"{args.out}: {gen}, {exported['kind']}, "
              f"held-out {exported['accuracy']:.4f}")

    if args.golden or args.golden_only:
        os.makedirs(os.path.dirname(GOLDEN_PATH), exist_ok=True)
        with open(GOLDEN_PATH, "w") as fh:
            json.dump(golden(), fh, allow_nan=False)
            fh.write("\n")
        print(f"{GOLDEN_PATH}: written")


if __name__ == "__main__":
    main()
