"""The dataset: captured frames and one label per eye.

A *sample* is a single eye from a single frame, identified by ``frame|side``.
Sides are sides of the picture, not of your face: the feed is not mirrored, so
your left eye appears on the image's right.

Labels are ``open``, ``closed`` or ``skip``. Only open and closed train a model;
``skip`` marks a sample as junk without removing anything. **Captures are never
deleted** - relabelling a skipped sample brings it straight back, which is why
this layer needs no undo.

The dataset is mutable and has no history of its own. History lives on the model
side: see :mod:`bgate.registry`, where every generation keeps a snapshot of the
labels it was trained on.
"""

import csv
import os

import cv2
import numpy as np

from . import ui

# Crop geometry the model is trained and evaluated with. Recorded into every
# model generation, because changing it silently changes what a snapshot means.
PAD_W, PAD_H = 1.6, 1.0
IN_H, IN_W = 24, 40

# (side key, landmark ring, display tag)
SIDES = (
    ("img_left", ui.RING_L, "IMG-L"),
    ("img_right", ui.RING_R, "IMG-R"),
)
OPEN, CLOSED, SKIP = "open", "closed", "skip"
LABELS = (OPEN, CLOSED, SKIP)
TRAINABLE = (OPEN, CLOSED)

SAMPLES_CSV = "samples.csv"
LEGACY_CSV = "labels.csv"          # one row per frame, img_left / img_right columns
CACHE = ".crops.npz"
FIELDS = ["frame", "side", "label"]


def sample_id(frame, side):
    return f"{frame}|{side}"


def burst_of(frame):
    """The capture burst a frame belongs to. Frames from one key press share a
    stamp, which is what lets validation hold out whole bursts."""
    return frame.split("_")[0]


def frames_dir(ds_dir):
    return os.path.join(ds_dir, "frames")


def ensure(ds_dir):
    os.makedirs(frames_dir(ds_dir), exist_ok=True)


def make_crop(gray, lm, ring, pad_w=PAD_W, pad_h=PAD_H):
    """Eye crop, plus the box it came from. Landmarks locate the eye reliably on
    infrared even where they track eyelid *shape* poorly, so position is all this
    asks of them."""
    h, w = gray.shape[:2]
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in ring])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    half_w = max((pts[:, 0].max() - pts[:, 0].min()) * pad_w / 2, 8)
    half_h = half_w * (IN_H / IN_W) * pad_h
    x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    x1, y1 = min(w, int(cx + half_w)), min(h, int(cy + half_h))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None, (x0, y0, x1, y1)
    crop = cv2.resize(gray[y0:y1, x0:x1], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    return crop, (x0, y0, x1, y1)


def _migrate_legacy(ds_dir):
    """Convert the old one-row-per-frame file to one row per eye.

    The legacy file is left in place untouched; captures are never destroyed and
    that goes for the record of them too.
    """
    legacy = os.path.join(ds_dir, LEGACY_CSV)
    if not os.path.exists(legacy):
        return []
    rows = []
    with open(legacy) as fh:
        for r in csv.DictReader(fh):
            for side, _ring, _tag in SIDES:
                label = r.get(side, "")
                if label in LABELS:
                    rows.append({"frame": r["file"], "side": side, "label": label})
    return rows


class Dataset:
    """In-memory dataset. Edits mutate memory and rewrite the CSV; nothing here
    re-reads from disk, because doing that per edit is what made the old UI
    unusable."""

    def __init__(self, ds_dir):
        self.dir = ds_dir
        ensure(ds_dir)
        self.rows = []          # [{frame, side, label}]
        self.crops = {}         # id -> uint8 crop
        self.load()

    # -- persistence -----------------------------------------------------
    @property
    def path(self):
        return os.path.join(self.dir, SAMPLES_CSV)

    def load(self):
        if os.path.exists(self.path):
            with open(self.path) as fh:
                self.rows = [r for r in csv.DictReader(fh) if r.get("label") in LABELS]
        else:
            self.rows = _migrate_legacy(self.dir)
            if self.rows:
                self.save()
        self._load_crops()

    def save(self):
        with open(self.path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)

    # -- crops -----------------------------------------------------------
    def _cache_path(self):
        return os.path.join(self.dir, CACHE)

    def _load_crops(self, landmarker=None, progress=None):
        cached = {}
        if os.path.exists(self._cache_path()):
            try:
                z = np.load(self._cache_path(), allow_pickle=False)
                # Read each array ONCE: NpzFile lookups are lazy, so indexing
                # inside a comprehension decompresses the archive per sample.
                names, crops = z["names"], z["crops"]
                cached = {str(n): crops[i] for i, n in enumerate(names)}
            except Exception:
                cached = {}     # a corrupt cache is never worth failing over

        wanted = {sample_id(r["frame"], r["side"]) for r in self.rows}
        missing_frames = sorted({r["frame"] for r in self.rows
                                 if sample_id(r["frame"], r["side"]) not in cached})
        if missing_frames:
            import mediapipe as mp
            from .model import make_landmarker
            landmarker = landmarker or make_landmarker()
            for i, frame in enumerate(missing_frames):
                g = cv2.imread(os.path.join(frames_dir(self.dir), frame),
                               cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                res = landmarker.detect(mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)))
                if not res.face_landmarks:
                    continue
                lm = res.face_landmarks[0]
                for side, ring, _tag in SIDES:
                    crop, _box = make_crop(g, lm, ring)
                    if crop is not None:
                        cached[sample_id(frame, side)] = crop
                if progress and i % 10 == 0:
                    progress(i, len(missing_frames))
            self._save_crops(cached)

        self.crops = {k: v for k, v in cached.items() if k in wanted}

    def _save_crops(self, cache):
        if not cache:
            return
        names = sorted(cache)
        np.savez_compressed(self._cache_path(), names=np.array(names),
                            crops=np.stack([cache[n] for n in names]))

    # -- queries ---------------------------------------------------------
    def samples(self):
        """Every labelled eye that has a usable crop."""
        out = []
        for r in self.rows:
            sid = sample_id(r["frame"], r["side"])
            if sid in self.crops:
                out.append(dict(id=sid, frame=r["frame"], side=r["side"],
                                label=r["label"], burst=burst_of(r["frame"]),
                                crop=self.crops[sid]))
        return out

    def trainable(self):
        return [s for s in self.samples() if s["label"] in TRAINABLE]

    def counts(self):
        c = {label: 0 for label in LABELS}
        for r in self.rows:
            if r["label"] in c:
                c[r["label"]] += 1
        return c

    # -- mutation --------------------------------------------------------
    def set_labels(self, ids, label):
        """Relabel samples. Nothing is deleted, so this is always reversible by
        relabelling again - which is why there is no undo endpoint."""
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        wanted = set(ids)
        n = 0
        for r in self.rows:
            if sample_id(r["frame"], r["side"]) in wanted:
                r["label"] = label
                n += 1
        if n:
            self.save()
        return n

    def add_capture(self, gray, label_left, label_right, stamp, index):
        """Save one captured frame and its two labels."""
        ensure(self.dir)
        frame = f"{stamp}_{index:03d}.png"
        cv2.imwrite(os.path.join(frames_dir(self.dir), frame), gray)
        for side, label in (("img_left", label_left), ("img_right", label_right)):
            self.rows.append({"frame": frame, "side": side, "label": label})
        self.save()
        return frame

    def refresh_crops(self, landmarker=None):
        """Pick up crops for frames added since load."""
        self._load_crops(landmarker=landmarker)


def snapshot_rows(rows):
    """A copy of the label assignment, for attaching to a model generation."""
    return [dict(frame=r["frame"], side=r["side"], label=r["label"]) for r in rows]
