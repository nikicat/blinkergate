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
import math
import os
import time

import cv2
import numpy as np

from . import ui

# Crop geometry the model is trained and evaluated with. Recorded into every
# model generation, because changing it silently changes what a snapshot means.
PAD_W, PAD_H = 1.6, 1.0
IN_H, IN_W = 24, 40

# How a crop is derived from landmarks, as a single string. Stamped into the crop
# cache and into every model generation: crops built one way are not crops built
# the other way, and the difference is invisible in a 40x24 thumbnail.
CROP_CONVENTION = "roll-levelled/level-width"

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
# Frames captured because a detector FAILED on them. They carry no eye labels -
# there are no eye positions to crop - so they live in their own manifest and are
# never part of classifier training. Ground truth is "a face is present", which
# you assert by pressing the key.
DETECTOR_CSV = "detector.csv"
DETECTOR_FIELDS = ["frame", "mediapipe", "note"]
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


def crop_at(gray, cx, cy, eye_width, pad_w=PAD_W, pad_h=PAD_H, roll=0.0):
    """The crop, given only where the eye is, how wide it is, and its tilt.

    This is the single definition of the geometry: any detector that can supply a
    centre and a width produces crops the model recognises, which is what lets a
    second detector stand in for the one the model was trained with.

    ``roll`` levels the eye. Locating a tilted eye correctly is not enough - an
    axis-aligned rectangle around it still contains a *rotated* eye, and every
    training crop holds a near-upright one. Passing the head roll cuts the crop in
    the levelled frame instead, so tilt stops being a variable the classifier has
    to have seen. roll=0 reproduces the plain rectangle exactly.

    Returns the crop and the four corners of the region it actually sampled, in
    source pixels. Corners rather than a rectangle because when the crop is
    levelled the sampled region is *tilted*, and a live view drawing an
    axis-aligned box around a rotated crop cannot show you a misplaced one.
    """
    h, w = gray.shape[:2]
    half_w = max(eye_width * pad_w / 2, 8)
    half_h = half_w * (IN_H / IN_W) * pad_h

    if roll:
        # One warp straight to the output size: rotate about the eye, scale each
        # axis so the crop maps to IN_W x IN_H, then put the eye at the centre of
        # the output. Scaling per axis rather than uniformly is what makes this
        # agree with the branch below for any pad_h, not only pad_h=1.
        th = math.radians(roll)
        cos, sin = math.cos(th), math.sin(th)
        sx, sy = IN_W / (2 * half_w), IN_H / (2 * half_h)
        m = np.array([[sx * cos, sx * sin, 0.0],
                      [-sy * sin, sy * cos, 0.0]])
        m[:, 2] = (IN_W / 2, IN_H / 2) - m[:, :2] @ (cx, cy)
        crop = cv2.warpAffine(gray, m, (IN_W, IN_H), flags=cv2.INTER_AREA,
                              borderMode=cv2.BORDER_REPLICATE)
        return crop, crop_quad(cx, cy, half_w, half_h, roll)

    box = (max(0, int(cx - half_w)), max(0, int(cy - half_h)),
           min(w, int(cx + half_w)), min(h, int(cy + half_h)))
    x0, y0, x1, y1 = box
    quad = crop_quad(cx, cy, half_w, half_h, 0.0)
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return None, quad
    crop = cv2.resize(gray[y0:y1, x0:x1], (IN_W, IN_H), interpolation=cv2.INTER_AREA)
    return crop, quad


def crop_quad(cx, cy, half_w, half_h, roll):
    """Corners of the sampled region in source pixels, in draw order."""
    th = math.radians(roll)
    cos, sin = math.cos(th), math.sin(th)
    corners = np.array([[-half_w, -half_h], [half_w, -half_h],
                        [half_w, half_h], [-half_w, half_h]])
    # Inverse of the levelling rotation: these are levelled-frame offsets being
    # put back into the source frame.
    back = np.stack([corners[:, 0] * cos - corners[:, 1] * sin,
                     corners[:, 0] * sin + corners[:, 1] * cos], axis=1)
    return back + (cx, cy)


def eye_centre(lm, ring, w, h, roll=0.0):
    """Centre and width of one eye in pixels, given the roll the crop will be
    levelled by.

    The width is measured along that levelled axis. The obvious alternative - the
    width of an axis-aligned box around the ring, which is what this did - shrinks
    as the head tilts, because a rotated ellipse projects narrower onto x.
    Measured over a synthetic roll sweep (`roll_sweep.py`) it read 0.81x the true
    width at 35 degrees while the levelled measure stayed within 3%, and since the
    width sets the crop's scale, that silently zoomed every crop on a tilted head
    by about 1.2x against everything the model was fitted on.

    ``roll=0`` reproduces the axis-aligned measure exactly, which is what
    generations fitted before this change need.
    """
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in ring])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    if roll:
        th = math.radians(roll)
        proj = (pts[:, 0] - cx) * math.cos(th) + (pts[:, 1] - cy) * math.sin(th)
    else:
        proj = pts[:, 0] - cx
    return cx, cy, float(proj.max() - proj.min())


def roll_from(lm, w, h):
    """Head roll in degrees, from the line joining the two eye centres."""
    lx, ly, _ = eye_centre(lm, SIDES[0][1], w, h)
    rx, ry, _ = eye_centre(lm, SIDES[1][1], w, h)
    return math.degrees(math.atan2(ly - ry, lx - rx))


def eye_geometry(lm, w, h, level_width=True):
    """``({side: (cx, cy, width)}, roll)`` - the whole crop geometry for a face.

    One function so the mesh locator, the capture path and the crop cache cannot
    drift apart on how a width is measured; they did, and a width is a scale.
    """
    roll = roll_from(lm, w, h)
    return ({side: eye_centre(lm, ring, w, h, roll if level_width else 0.0)
             for side, ring, _tag in SIDES}, roll)


def make_crop(gray, lm, ring, pad_w=PAD_W, pad_h=PAD_H, roll=None,
              level_width=True):
    """Eye crop from mesh landmarks. Landmarks locate the eye reliably on infrared
    even where they track eyelid *shape* poorly, so position is all this asks.

    Roll is taken from the landmarks unless given, so the crop is levelled by
    default: a correctly located but tilted eye still breaks the classifier
    (measured 56.9% at 30 degrees against 98.3% levelled).
    """
    h, w = gray.shape[:2]
    if roll is None:
        roll = roll_from(lm, w, h)
    cx, cy, width = eye_centre(lm, ring, w, h, roll if level_width else 0.0)
    return crop_at(gray, cx, cy, width, pad_w, pad_h, roll)


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
                if geoms.shape[1] < 4:
                    raise ValueError("cache predates roll-levelled crops")
                # A crop cut under an older convention is the same shape and the
                # same 40x24 of plausible-looking eye, so nothing downstream can
                # notice. Rebuild on any mismatch.
                if str(z["convention"]) != CROP_CONVENTION:
                    raise ValueError("cache predates the current crop convention")
                cached = {str(n): crops[i] for i, n in enumerate(names)}
                geom = {str(n): geoms[i] for i, n in enumerate(names)}
            except Exception:
                cached, geom = {}, {}   # corrupt or old cache: rebuild

        wanted = {sample_id(r["frame"], r["side"]) for r in self.rows}
        missing_frames = sorted({r["frame"] for r in self.rows
                                 if sample_id(r["frame"], r["side"]) not in cached})
        if missing_frames:
            # The same locator the live view runs, not a bare mesh call. A
            # training crop cut one way and an inference crop cut another is a
            # template matcher being shown a template it was never fitted on, and
            # nothing about a 40x24 thumbnail makes that visible. A frame this
            # cannot locate loses its labels from training entirely.
            from .detect import default_locator
            locator = default_locator(landmarker)
            for i, frame in enumerate(missing_frames):
                g = cv2.imread(os.path.join(frames_dir(self.dir), frame),
                               cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                det = locator.locate(g)
                if not det:
                    continue
                lm = det.landmarks
                gh, gw = g.shape[:2]
                eyes, roll = eye_geometry(lm, gw, gh)
                for side, (cx, cy, width) in eyes.items():
                    crop, _quad = crop_at(g, cx, cy, width, roll=roll)
                    if crop is not None:
                        sid = sample_id(frame, side)
                        cached[sid] = crop
                        geom[sid] = np.array([cx, cy, width, roll], dtype=np.float32)
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
                 geom=np.stack([geom[n] for n in names]),
                 convention=CROP_CONVENTION)

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
            eyes, roll = eye_geometry(lm, w, h)
            for side, (cx, cy, width) in eyes.items():
                crop, _quad = crop_at(gray, cx, cy, width, roll=roll)
                if crop is not None:
                    sid = sample_id(frame, side)
                    self.crops[sid] = crop
                    self.geom[sid] = np.array([cx, cy, width, roll], dtype=np.float32)
            self.flush()
        return frame

    def flush(self):
        """Write the crop cache. Cheap enough to call after every capture."""
        self._save_crops(self.crops, self.geom)

    # -- detector test cases ---------------------------------------------
    def detector_path(self):
        return os.path.join(self.dir, DETECTOR_CSV)

    def add_detector_case(self, gray, detected, note=""):
        """Save a frame that a detector should have found a face in.

        Deliberately separate from the labelled samples: this is a benchmark for
        *locating* the eyes, and a frame is only interesting here when locating
        failed - which is exactly when there is nothing to label.
        """
        ensure(self.dir)
        # A millisecond stamp alone collides when two land in the same one, which
        # silently overwrites the earlier frame. Suffix until the name is free.
        base = int(time.time() * 1000)
        n = 0
        while True:
            frame = f"hard_{base}_{n:02d}.png"
            if not os.path.exists(os.path.join(frames_dir(self.dir), frame)):
                break
            n += 1
        cv2.imwrite(os.path.join(frames_dir(self.dir), frame), gray)
        new = not os.path.exists(self.detector_path())
        with open(self.detector_path(), "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=DETECTOR_FIELDS)
            if new:
                w.writeheader()
            w.writerow({"frame": frame, "mediapipe": int(bool(detected)), "note": note})
        return frame

    def detector_cases(self):
        try:
            with open(self.detector_path()) as fh:
                return list(csv.DictReader(fh))
        except OSError:
            return []

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
                cx, cy, width, roll = (float(v) for v in s["geom"])
                for _ in range(copies):
                    dx, dy = rng.uniform(-JITTER_PX, JITTER_PX, 2)
                    scale = 1.0 + rng.uniform(-JITTER_SCALE, JITTER_SCALE)
                    crop, _quad = crop_at(gray, cx + dx, cy + dy, width * scale,
                                          roll=roll)
                    if crop is not None:
                        out.append(dict(id=s["id"], frame=frame, side=s["side"],
                                        label=s["label"], burst=s["burst"],
                                        crop=crop, geom=s["geom"]))
        return out


def snapshot_rows(rows):
    """A copy of the label assignment, for attaching to a model generation."""
    return [dict(frame=r["frame"], side=r["side"], label=r["label"]) for r in rows]
