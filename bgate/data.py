"""The dataset: captured frames and one label per eye.

A *sample* is a single eye from a single frame, identified by ``frame|side``.

Sides are **yours**, not the picture's. ``img_left`` is your left eye, which on an
unmirrored feed appears on the RIGHT of the image (measured: mean x 347 of 640,
against 307 for ``img_right``). That follows MediaPipe, whose "left eye" ring is
the subject's left. The key names are historical and stay as they are, because
they are what every existing label is keyed by.

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

# (side key, landmark ring, display tag). Tags are subject-relative because that
# is what the rings actually are; the keys are historical.
SIDES = (
    ("img_left", ui.RING_L, "YOUR LEFT"),
    ("img_right", ui.RING_R, "YOUR RIGHT"),
)
OPEN, CLOSED, SKIP = "open", "closed", "skip"
LABELS = (OPEN, CLOSED, SKIP)
TRAINABLE = (OPEN, CLOSED)

SAMPLES_CSV = "samples.csv"
LEGACY_CSV = "labels.csv"          # one row per frame, img_left / img_right columns
CACHE = ".crops.npz"
FIELDS = ["frame", "side", "label"]

# Jitter used to make the classifier tolerant of crop misalignment. Measured
# without it, a 4px shift flipped 14% of verdicts: a linear model on raw pixels
# is a template matcher, and every source of geometry noise - head pose, a
# different detector, distance - lands on that cliff.
JITTER_PX = 3.0        # translation, in source pixels
JITTER_SCALE = 0.08    # fractional eye-width


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


def crop_at(gray, cx, cy, eye_width, pad_w=PAD_W, pad_h=PAD_H):
    """The crop, given only where the eye is and how wide it is.

    This is the single definition of the geometry: any detector that can supply a
    centre and a width produces crops the model recognises, which is what lets a
    second detector stand in for the one the model was trained with.
    """
    h, w = gray.shape[:2]
    half_w = max(eye_width * pad_w / 2, 8)
    half_h = half_w * (IN_H / IN_W) * pad_h
    x0, y0 = max(0, int(cx - half_w)), max(0, int(cy - half_h))
    x1, y1 = min(w, int(cx + half_w)), min(h, int(cy + half_h))
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None, (x0, y0, x1, y1)
    crop = cv2.resize(gray[y0:y1, x0:x1], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    return crop, (x0, y0, x1, y1)


def make_crop(gray, lm, ring, pad_w=PAD_W, pad_h=PAD_H):
    """Eye crop from mesh landmarks. Landmarks locate the eye reliably on infrared
    even where they track eyelid *shape* poorly, so position is all this asks."""
    h, w = gray.shape[:2]
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in ring])
    return crop_at(gray, pts[:, 0].mean(), pts[:, 1].mean(),
                   pts[:, 0].max() - pts[:, 0].min(), pad_w, pad_h)


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
    """In-memory dataset. Edits mutate memory and rewrite the CSV, because
    reloading everything per edit is what made the first UI unusable.

    Two processes share it - the camera app captures, the web UI labels - so each
    checks ``samples.csv``'s mtime before serving or training and reloads only
    when the other has actually written. A reload is ~2ms (0.7ms to parse the CSV,
    1ms to read the crop cache), so the check is cheap enough to do per request.
    """

    def __init__(self, ds_dir):
        self.dir = ds_dir
        ensure(ds_dir)
        self.rows = []          # [{frame, side, label}]
        self.crops = {}         # id -> uint8 crop
        self.geom = {}          # id -> (cx, cy, eye_width) in the source frame
        self._mtime = 0.0       # mtime of samples.csv as of the last load
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
        self._mtime = self._disk_mtime()
        self._load_crops()

    def _disk_mtime(self):
        try:
            return os.path.getmtime(self.path)
        except OSError:
            return 0.0

    def reload_if_changed(self):
        """Pick up what the other process wrote. Returns True if anything did."""
        if self._disk_mtime() <= self._mtime:
            return False
        self.load()
        return True

    def save(self):
        with open(self.path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(self.rows)
        # Our own write must not look like somebody else's.
        self._mtime = self._disk_mtime()

    # -- crops -----------------------------------------------------------
    def _cache_path(self):
        return os.path.join(self.dir, CACHE)

    def _load_crops(self, landmarker=None, progress=None):
        cached, geom = {}, {}
        if os.path.exists(self._cache_path()):
            try:
                z = np.load(self._cache_path(), allow_pickle=False)
                # Read each array ONCE: NpzFile lookups are lazy, so indexing
                # inside a comprehension decompresses the archive per sample.
                names, crops = z["names"], z["crops"]
                # Older caches predate stored geometry; without it we cannot
                # re-crop for jitter, so rebuild rather than silently skip it.
                geoms = z["geom"]
                cached = {str(n): crops[i] for i, n in enumerate(names)}
                geom = {str(n): geoms[i] for i, n in enumerate(names)}
            except Exception:
                cached, geom = {}, {}   # corrupt or old cache: rebuild

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
                gh, gw = g.shape[:2]
                for side, ring, _tag in SIDES:
                    pts = np.array([[lm[i].x * gw, lm[i].y * gh] for i in ring])
                    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                    width = pts[:, 0].max() - pts[:, 0].min()
                    crop, _box = crop_at(g, cx, cy, width)
                    if crop is not None:
                        sid = sample_id(frame, side)
                        cached[sid] = crop
                        geom[sid] = np.array([cx, cy, width], dtype=np.float32)
                if progress and i % 10 == 0:
                    progress(i, len(missing_frames))
            self._save_crops(cached, geom)

        self.crops = {k: v for k, v in cached.items() if k in wanted}
        self.geom = {k: v for k, v in geom.items() if k in wanted}

    def _save_crops(self, cache, geom):
        if not cache:
            return
        names = sorted(n for n in cache if n in geom)
        # Uncompressed on purpose: 1ms against 50ms for 0.28MB more on a dataset
        # of ~700 crops. That is the difference between flushing after every
        # capture and having to defer it.
        np.savez(self._cache_path(), names=np.array(names),
                 crops=np.stack([cache[n] for n in names]),
                 geom=np.stack([geom[n] for n in names]))

    # -- queries ---------------------------------------------------------
    def samples(self):
        """Every labelled eye that has a usable crop."""
        out = []
        for r in self.rows:
            sid = sample_id(r["frame"], r["side"])
            if sid in self.crops:
                out.append(dict(id=sid, frame=r["frame"], side=r["side"],
                                label=r["label"], burst=burst_of(r["frame"]),
                                crop=self.crops[sid], geom=self.geom.get(sid)))
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

    def add_capture(self, gray, label_left, label_right, stamp, index=0, lm=None):
        """Save one captured frame and its two labels.

        ``lm`` are the landmarks already computed for this frame by the caller.
        Passing them lets the crop be derived here instead of re-detecting the
        face immediately afterwards, which is what makes a capture feel instant.
        Both the CSV and the crop cache are written straight away, so the web UI
        sees the sample on its next request.
        """
        ensure(self.dir)
        frame = f"{stamp}_{index:03d}.png"
        cv2.imwrite(os.path.join(frames_dir(self.dir), frame), gray)
        for side, label in (("img_left", label_left), ("img_right", label_right)):
            self.rows.append({"frame": frame, "side": side, "label": label})
        self.save()

        if lm is not None:
            h, w = gray.shape[:2]
            for side, ring, _tag in SIDES:
                pts = np.array([[lm[i].x * w, lm[i].y * h] for i in ring])
                cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                width = pts[:, 0].max() - pts[:, 0].min()
                crop, _box = crop_at(gray, cx, cy, width)
                if crop is not None:
                    sid = sample_id(frame, side)
                    self.crops[sid] = crop
                    self.geom[sid] = np.array([cx, cy, width], dtype=np.float32)
            self.flush()
        return frame

    def flush(self):
        """Write the crop cache. Cheap enough to call after every capture."""
        self._save_crops(self.crops, self.geom)

    def refresh_crops(self, landmarker=None):
        """Pick up crops for frames added since load."""
        self._load_crops(landmarker=landmarker)

    def jittered(self, samples, copies, seed=0):
        """Extra training rows, re-cropped from the source frames at perturbed
        geometry.

        Re-cropping rather than warping the finished 40x24: a translation applied
        to the small crop would drag in border padding instead of the pixels that
        are genuinely next to the eye, teaching the model about the padding.

        Frames are read once each and shared across every sample and copy taken
        from them.
        """
        if copies <= 0:
            return []
        rng = np.random.default_rng(seed)
        by_frame = {}
        for s in samples:
            if s.get("geom") is not None:
                by_frame.setdefault(s["frame"], []).append(s)

        out = []
        for frame, group in by_frame.items():
            gray = cv2.imread(os.path.join(frames_dir(self.dir), frame),
                              cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            for s in group:
                cx, cy, width = (float(v) for v in s["geom"])
                for _ in range(copies):
                    dx, dy = rng.uniform(-JITTER_PX, JITTER_PX, 2)
                    scale = 1.0 + rng.uniform(-JITTER_SCALE, JITTER_SCALE)
                    crop, _box = crop_at(gray, cx + dx, cy + dy, width * scale)
                    if crop is not None:
                        out.append(dict(id=s["id"], frame=frame, side=s["side"],
                                        label=s["label"], burst=s["burst"],
                                        crop=crop, geom=s["geom"]))
        return out


def snapshot_rows(rows):
    """A copy of the label assignment, for attaching to a model generation."""
    return [dict(frame=r["frame"], side=r["side"], label=r["label"]) for r in rows]
