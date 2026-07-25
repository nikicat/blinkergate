"""Eye open/closed classifier, fitted on the camera it will run on.

Off-the-shelf IR eye-state models reach roughly chance-to-63% on this hardware:
they are trained on well-exposed close-up infrared eyes, and a laptop Hello
camera gives you an 18-pixel-wide, low-contrast, often oblique eye instead. They
fail *confidently*, returning 1.000 with no uncertainty, which reads as a broken
detector rather than an unsure one.

A linear model fitted on a couple of hundred crops from the actual sensor reaches
~98% on held-out poses. This is enrolment, not tuning - there is no threshold you
adjust by eye - and it is the same shape as Howdy already enrolling your face.
"""

import os

import cv2
import numpy as np

from . import data, ui

DEFAULT_PATH = "eyemodel.npz"

# Gaussian blur applied to a crop before it becomes a feature vector.
#
# A linear model on raw pixels is a template matcher, and templates are brittle:
# measured on held-out bursts, a 4px crop shift cost 13 points of accuracy. Every
# source of geometry noise - head pose, distance, a different detector - lands
# there. Blurring makes each feature cover more area, so a shift moves the content
# a smaller fraction of a feature.
#
# Measured across shifts of 0-6px, sigma 1.2-2.0 is a plateau; 1.6 sits in the
# middle and costs nothing at zero shift:
#     sigma      0px     2px     3px     4px     6px
#     none     98.8%   97.3%   94.5%   85.4%   78.3%
#     1.6      98.8%   98.0%   97.2%   92.1%   81.0%
#
# Jitter augmentation was tried first and made things WORSE (81.3% at 4px with
# 4 copies): averaging a template over shifted copies blurs it without teaching
# invariance, which needs pooling. See Dataset.jittered.
BLUR_SIGMA = 1.6

# Per-crop contrast normalisation.
#
# Without it the features are absolute pixel values, so the decision boundary is
# calibrated to the illumination of whichever session was captured. Ambient IR
# changes - time of day, distance to the emitter, the angle your face presents -
# and the model silently stops working. Measured on held-out bursts with the
# frame scaled to simulate that:
#
#     features        x0.6    x0.75   as-is    x1.3   gamma1.4
#     raw            57.2%    61.4%   98.8%   88.3%     66.3%
#     standardise    98.8%    98.8%   98.8%   98.8%     98.8%
#
# Subtracting the mean and dividing by the standard deviation removes gain and
# offset exactly, which is what a lighting change mostly is. CLAHE and DoG also
# help but less completely (76.9% and 96.6% worst-case).
#
# Like the blur, this is carried in the model: features normalised one way are
# meaningless to a model fitted the other way.
NORMALISE = "standardise"


# The detector inside the landmarker bundle, not the mesh, is what rejects a
# tilted or low-contrast face. At the default 0.5 it found 49% of the frames
# captured because detection was struggling; at 0.2 that is 74%, and 100% once
# roll compensation is applied. It costs nothing measurable: accuracy on ordinary
# frames went 99.2% -> 99.6%, and across 39 faceless inputs (sensor noise, flat
# fields, real frames with the face blanked out) there were zero false positives
# at any threshold down to 0.1.
DETECTION_CONFIDENCE = 0.2


def make_landmarker(model_path=None, confidence=DETECTION_CONFIDENCE):
    """MediaPipe face landmarker in IMAGE mode.

    IMAGE, not VIDEO: VIDEO seeds each frame from the previous one, which damps
    frame-to-frame iris motion about tenfold. That smoothing is exactly the lag
    that leaves the eye box trailing behind a head turn.
    """
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    path = model_path or ui.FACE_MODEL
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run: python fetch_model.py")
    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=path),
            running_mode=vision.RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=confidence))


def features(crops, blur=BLUR_SIGMA, normalise=NORMALISE):
    out = []
    for crop in crops:
        x = crop.astype(np.float32)
        if blur:
            x = cv2.GaussianBlur(x, (0, 0), blur)
        if normalise == "standardise":
            x = (x - x.mean()) / (x.std() + 1e-6)
        else:
            x = x / 255.0
        out.append(x.reshape(-1))
    return np.stack(out)


def _make_clf():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, C=0.05))


def _flatten(clf):
    """Fold the scaler into the linear layer: inference becomes one dot product."""
    scaler = clf.named_steps["standardscaler"]
    lr = clf.named_steps["logisticregression"]
    coef = (lr.coef_[0] / scaler.scale_).astype(np.float32)
    intercept = float(lr.intercept_[0] -
                      np.dot(lr.coef_[0], scaler.mean_ / scaler.scale_))
    return coef, intercept


def fit(samples, augment=None, blur=BLUR_SIGMA, normalise=NORMALISE):
    """Fit on labelled samples. Returns (coef, intercept, held_out_accuracy).

    Validation groups by capture burst: frames from one key press are near
    duplicates, so splitting them across train and test would score memorisation
    rather than generalisation.

    ``augment`` is an optional callable taking a list of samples and returning
    extra ones. It is applied **to training folds only** - never to the fold being
    scored - so the reported accuracy stays comparable with generations trained
    without it, and cannot be inflated by testing on near-copies of training rows.
    Measured, jitter augmentation hurt; the hook is kept because the measurement
    is worth being able to repeat, not because it is recommended.

    ``blur`` and ``normalise`` must be carried into the model: features built one
    way are meaningless to a model fitted the other way.
    """
    from sklearn.model_selection import GroupKFold

    y = np.array([1 if s["label"] == data.CLOSED else 0 for s in samples])
    groups = np.array([s.get("burst") or data.burst_of(s["frame"]) for s in samples])
    if len(set(y)) < 2:
        raise ValueError("need both open and closed samples")
    X = features([s["crop"] for s in samples], blur, normalise)

    def matrix(rows):
        return features([r["crop"] for r in rows], blur, normalise), \
               np.array([1 if r["label"] == data.CLOSED else 0 for r in rows])

    accuracy = float("nan")
    n_groups = len(set(groups))
    if n_groups >= 2:
        scores = []
        for train_idx, test_idx in GroupKFold(
                n_splits=min(5, n_groups)).split(X, y, groups):
            rows = [samples[i] for i in train_idx]
            extra = augment(rows) if augment else []
            Xtr, ytr = matrix(rows + extra) if extra else (X[train_idx], y[train_idx])
            clf = _make_clf()
            clf.fit(Xtr, ytr)
            scores.append(clf.score(X[test_idx], y[test_idx]))
        accuracy = float(np.mean(scores))

    extra = augment(samples) if augment else []
    Xall, yall = matrix(samples + extra) if extra else (X, y)
    clf = _make_clf()
    clf.fit(Xall, yall)
    coef, intercept = _flatten(clf)
    return coef, intercept, accuracy


class EyeModel:
    """Loaded classifier. ``predict`` returns P(open) per crop."""

    def __init__(self, coef, intercept, accuracy=float("nan"), blur=BLUR_SIGMA,
                 normalise=NORMALISE, level_width=True):
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = float(intercept)
        self.accuracy = float(accuracy)
        self.blur = float(blur)
        self.normalise = normalise
        self.level_width = bool(level_width)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        d = np.load(path)
        # Generations predating the blur were fitted on sharp features; defaulting
        # to 0.0 keeps them meaning what they meant when they were trained.
        blur = float(d["blur"]) if "blur" in d.files else 0.0
        # Generations predating normalisation were fitted on absolute pixel
        # values; "none" keeps them meaning what they meant when trained.
        norm = str(d["normalise"]) if "normalise" in d.files else "none"
        # Generations predating the levelled width were fitted on crops whose
        # scale came from an axis-aligned box, so they must keep being fed those.
        level_width = bool(d["level_width"]) if "level_width" in d.files else False
        return cls(d["coef"], d["intercept"],
                   d.get("cv_accuracy", float("nan")), blur, norm, level_width)

    @classmethod
    def exists(cls, path=DEFAULT_PATH):
        return os.path.exists(path)

    def save(self, path=DEFAULT_PATH):
        np.savez(path, coef=self.coef, intercept=self.intercept,
                 in_h=data.IN_H, in_w=data.IN_W, blur=self.blur,
                 normalise=self.normalise, pad_w=data.PAD_W, pad_h=data.PAD_H,
                 level_width=self.level_width, cv_accuracy=self.accuracy)

    def p_closed(self, crops):
        crops = [c for c in crops if c is not None]
        if not crops:
            return np.array([])
        return 1.0 / (1.0 + np.exp(
            -(features(crops, self.blur, self.normalise) @ self.coef
              + self.intercept)))

    def predict(self, crops):
        """P(open) per crop, preserving None entries."""
        out, usable = [], [c for c in crops if c is not None]
        probs = iter(1.0 - self.p_closed(usable)) if usable else iter(())
        for c in crops:
            out.append(float(next(probs)) if c is not None else None)
        return out

    def eyes(self, gray, lm, roll=None):
        """P(open) and the sampled quad for both eyes, from mesh landmarks.

        The crop is levelled using the head roll, because locating a tilted eye
        is not the same as being able to classify it.
        """
        h, w = gray.shape[:2]
        if roll is None:
            roll = data.roll_from(lm, w, h)
        crops, quads = [], []
        for _side, ring, _tag in data.SIDES:
            crop, quad = data.make_crop(gray, lm, ring, roll=roll,
                                        level_width=self.level_width)
            crops.append(crop)
            quads.append(quad)
        return self.predict(crops), crops, quads

    def eyes_from(self, gray, detection):
        """P(open) and the sampled quad per side, from any detector's eye
        positions.

        Keyed by side rather than positional, so a detector that finds only one
        eye still contributes the one it found.
        """
        sides, crops, quads = [], [], []
        for side, _ring, _tag in data.SIDES:
            eye = detection.eyes.get(side)
            if eye is None:
                continue
            crop, quad = data.crop_at(gray, eye.cx, eye.cy, eye.width,
                                      roll=getattr(detection, "roll", 0.0) or 0.0)
            sides.append(side)
            crops.append(crop)
            quads.append(quad)
        probs = self.predict(crops)
        return dict(zip(sides, probs)), dict(zip(sides, crops)), dict(zip(sides, quads))
