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


def make_landmarker(model_path=None):
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
            running_mode=vision.RunningMode.IMAGE, num_faces=1))


def features(crops):
    return np.stack([c.reshape(-1).astype(np.float32) / 255.0 for c in crops])


def fit(samples):
    """Fit on labelled samples. Returns (coef, intercept, held_out_accuracy).

    Validation groups by capture burst: frames from one key press are near
    duplicates, so splitting them across train and test would score memorisation
    rather than generalisation.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = features([s["crop"] for s in samples])
    y = np.array([1 if s["label"] == data.CLOSED else 0 for s in samples])
    groups = np.array([s.get("burst") or data.burst_of(s["frame"]) for s in samples])
    if len(set(y)) < 2:
        raise ValueError("need both open and closed samples")

    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.05))
    accuracy = float("nan")
    n_groups = len(set(groups))
    if n_groups >= 2:
        accuracy = float(cross_val_score(
            clf, X, y, groups=groups, cv=GroupKFold(n_splits=min(5, n_groups))).mean())

    clf.fit(X, y)
    scaler = clf.named_steps["standardscaler"]
    lr = clf.named_steps["logisticregression"]
    # Fold the scaler into the linear layer so inference is a single dot product.
    coef = (lr.coef_[0] / scaler.scale_).astype(np.float32)
    intercept = float(lr.intercept_[0] - np.dot(lr.coef_[0], scaler.mean_ / scaler.scale_))
    return coef, intercept, accuracy


class EyeModel:
    """Loaded classifier. ``predict`` returns P(open) per crop."""

    def __init__(self, coef, intercept, accuracy=float("nan")):
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = float(intercept)
        self.accuracy = float(accuracy)

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        d = np.load(path)
        return cls(d["coef"], d["intercept"], d.get("cv_accuracy", float("nan")))

    @classmethod
    def exists(cls, path=DEFAULT_PATH):
        return os.path.exists(path)

    def save(self, path=DEFAULT_PATH):
        np.savez(path, coef=self.coef, intercept=self.intercept,
                 in_h=data.IN_H, in_w=data.IN_W,
                 pad_w=data.PAD_W, pad_h=data.PAD_H, cv_accuracy=self.accuracy)

    def p_closed(self, crops):
        crops = [c for c in crops if c is not None]
        if not crops:
            return np.array([])
        return 1.0 / (1.0 + np.exp(-(features(crops) @ self.coef + self.intercept)))

    def predict(self, crops):
        """P(open) per crop, preserving None entries."""
        out, usable = [], [c for c in crops if c is not None]
        probs = iter(1.0 - self.p_closed(usable)) if usable else iter(())
        for c in crops:
            out.append(float(next(probs)) if c is not None else None)
        return out

    def eyes(self, gray, lm):
        """P(open) and crop box for both eyes of one frame."""
        crops, boxes = [], []
        for _side, ring, _tag in data.SIDES:
            crop, box = data.make_crop(gray, lm, ring)
            crops.append(crop)
            boxes.append(box)
        return self.predict(crops), crops, boxes
