"""Model generations, kept forever.

Training never overwrites: it appends a new generation. Each one holds its
weights, a snapshot of every label at the moment it was trained, and enough
metadata to tell whether it is still reproducible.

That last part matters. Crops are *derived* from frames, landmarks and crop
geometry, so a snapshot only means what it meant if those are unchanged. The
crop parameters, library versions and a hash of the exact training crops all go
into ``meta.json``, which turns a silent divergence into a detectable one.

    models/
      gen-0001/  model.npz  dataset.csv  meta.json
      gen-0002/  ...
      active                -> which generation the live view uses
"""

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone

import numpy as np

from . import data

ACTIVE = "active"
GEN_RE = re.compile(r"^gen-(\d{4,})$")


def _versions():
    out = {}
    try:
        import mediapipe
        out["mediapipe"] = mediapipe.__version__
    except Exception:
        pass
    try:
        import sklearn
        out["scikit-learn"] = sklearn.__version__
    except Exception:
        pass
    from . import __version__
    out["blinkergate"] = __version__
    return out


def crops_hash(samples):
    """Content hash of the exact crops a fit consumed, in a stable order."""
    h = hashlib.sha256()
    for s in sorted(samples, key=lambda s: s["id"]):
        h.update(s["id"].encode())
        h.update(s["label"].encode())
        h.update(np.ascontiguousarray(s["crop"]).tobytes())
    return h.hexdigest()


class Registry:
    def __init__(self, root="models"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    # -- generations -----------------------------------------------------
    def ids(self):
        return sorted(d for d in os.listdir(self.root) if GEN_RE.match(d))

    def path(self, gen, *parts):
        return os.path.join(self.root, gen, *parts)

    def next_id(self):
        existing = [int(GEN_RE.match(g).group(1)) for g in self.ids()]
        return f"gen-{max(existing, default=0) + 1:04d}"

    def meta(self, gen):
        try:
            with open(self.path(gen, "meta.json")) as fh:
                m = json.load(fh)
        except Exception:
            return None
        m["id"] = gen
        m["active"] = gen == self.active()
        return m

    def list(self):
        return [m for m in (self.meta(g) for g in self.ids()) if m]

    def create(self, coef, intercept, accuracy, samples, all_rows, blur=None,
               normalise=None):
        """Write a new generation. ``samples`` are the ones that trained it;
        ``all_rows`` is the whole label file, skips included, so the snapshot
        records what was excluded as well as what was used."""
        gen = self.next_id()
        os.makedirs(self.path(gen), exist_ok=True)

        from .model import BLUR_SIGMA, NORMALISE
        blur = BLUR_SIGMA if blur is None else blur
        normalise = NORMALISE if normalise is None else normalise
        np.savez(self.path(gen, "model.npz"), coef=coef, intercept=intercept,
                 in_h=data.IN_H, in_w=data.IN_W, blur=blur, normalise=normalise,
                 pad_w=data.PAD_W, pad_h=data.PAD_H, cv_accuracy=accuracy)

        with open(self.path(gen, "dataset.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=data.FIELDS)
            w.writeheader()
            w.writerows(data.snapshot_rows(all_rows))

        counts = {label: sum(1 for r in all_rows if r["label"] == label)
                  for label in data.LABELS}
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "accuracy": None if accuracy != accuracy else round(float(accuracy), 4),
            "n_train": len(samples),
            "n_bursts": len({s["burst"] for s in samples}),
            "counts": counts,
            "crop": {"pad_w": data.PAD_W, "pad_h": data.PAD_H,
                     "in_h": data.IN_H, "in_w": data.IN_W, "blur": blur,
                     "normalise": normalise},
            "crops_sha256": crops_hash(samples),
            "versions": _versions(),
        }
        with open(self.path(gen, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        return gen

    def snapshot(self, gen):
        """The label assignment a generation was trained on."""
        try:
            with open(self.path(gen, "dataset.csv")) as fh:
                return list(csv.DictReader(fh))
        except Exception:
            return []

    def unseen_by(self, gen, samples):
        """Samples labelled since ``gen`` was trained - genuinely held out for
        it, and so a fair way to compare an old generation against a new one on
        today's labels rather than on its own."""
        snap = {data.sample_id(r["frame"], r["side"]): r["label"]
                for r in self.snapshot(gen)}
        return [s for s in samples
                if snap.get(s["id"]) != s["label"] or s["id"] not in snap]

    # -- the active generation -------------------------------------------
    def active(self):
        try:
            with open(os.path.join(self.root, ACTIVE)) as fh:
                gen = fh.read().strip()
        except OSError:
            return None
        return gen if gen in self.ids() else None

    def set_active(self, gen):
        if gen not in self.ids():
            raise ValueError(f"unknown generation {gen!r}")
        with open(os.path.join(self.root, ACTIVE), "w") as fh:
            fh.write(gen + "\n")

    def dir_mtime(self):
        """Changes when a generation is added or removed. Lets a second process
        notice new generations without listing the directory every frame."""
        try:
            return os.path.getmtime(self.root)
        except OSError:
            return 0.0

    def active_mtime(self):
        """For cheap change detection: the live view polls this instead of
        needing any channel back from the web UI."""
        try:
            return os.path.getmtime(os.path.join(self.root, ACTIVE))
        except OSError:
            return 0.0

    def load(self, gen):
        from .model import EyeModel
        return EyeModel.load(self.path(gen, "model.npz"))

    def load_active(self):
        gen = self.active()
        return (gen, self.load(gen)) if gen else (None, None)
