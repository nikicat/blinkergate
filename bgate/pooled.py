"""Eye state from pooled filter responses, rather than from fixed pixels.

The pixel model is a template matcher: pixel (12, 7) always multiplies weight
(12, 7). Move the pupil and it multiplies a different weight, which is why a 4px
crop shift flips 14% of its verdicts. Measured on this dataset the bright pupil
sits a median 1.8px from the crop centre and up to 9.9px away, and re-centring the
crop does not fix it - the offset is scatter (gaze), not bias.

This model convolves instead. Each kernel slides over every position, producing a
response map, and the feature is the **maximum of that map**. Move the pupil and
the peak moves with it; the maximum does not change. That reduce step is the
pooling, and it is the only reason the result is shift-invariant.

Two things are learned jointly: the kernels themselves and the logistic layer on
top. Max-pooling makes the backward pass trivial - only the winning position
contributed, so the gradient is a single patch per kernel per sample - which is
why this needs numpy and not a framework.

Size, for scale: four 11x11 kernels plus the head is ~490 weights, against 960 for
the pixel model it replaces. Fewer parameters *and* the invariance. The usual
"a convnet needs more data" caution is about stacked layers, not about this.
"""

import numpy as np

from . import data, model

# Kernels are square and odd. 11 on a 40x24 crop leaves a 30x14 response map,
# which is the pooling window: the pupil was inside the crop on 636 of 636
# confident detections, so pooling over the whole map loses nothing.
KERNEL = 11

# Per-crop standardisation, as the pixel model uses, for the same reason: raw
# features pin the decision boundary to one session's illumination.
NORMALISE = "standardise"


def _prep(k):
    """Zero-mean and unit-norm.

    Zero-mean so a flat region scores exactly 0 - otherwise the pooling maximum
    will happily lock onto the brightest patch of cheek. Unit-norm so the kernel
    cannot grow to inflate its own feature, which would leave the logistic layer
    fitting a scale it does not control.
    """
    k = np.asarray(k, dtype=np.float32)
    k = k - k.mean()
    n = float(np.linalg.norm(k))
    return k / n if n > 1e-9 else k


def log_kernel(size=KERNEL, sigma=1.8):
    """Bright disc with a darker ring - a Laplacian of Gaussian, negated.

    The raw Laplacian dips negative at the centre of a bright blob, so a bright
    pupil would score negative; negating makes "bright disc, dark ring" the thing
    that scores high. sigma is the blob radius over sqrt(2), in crop pixels.
    """
    ax = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    r2 = xx ** 2 + yy ** 2
    lap = (r2 - 2 * sigma ** 2) / sigma ** 4 * np.exp(-r2 / (2 * sigma ** 2))
    return _prep(-lap)


def ridge_kernel(size=KERNEL, thickness=1.4, length=3.2, angle=0.0):
    """Dark elongated bar - negative along the line, positive above and below.

    Not negated, unlike the blob: the second derivative across a Gaussian is
    already negative at the centre, so a *dark* line scores high. Anisotropic on
    purpose, so it answers to a lash line and ignores a round blob. An arc is
    locally a tilted line, which is what the angle is for.
    """
    ax = np.arange(size) - (size - 1) / 2
    xx, yy = np.meshgrid(ax, ax)
    th = np.radians(angle)
    along = xx * np.cos(th) + yy * np.sin(th)
    across = -xx * np.sin(th) + yy * np.cos(th)
    profile = ((across ** 2 - thickness ** 2) / thickness ** 4
               * np.exp(-across ** 2 / (2 * thickness ** 2)))
    return _prep(profile * np.exp(-along ** 2 / (2 * length ** 2)))


def default_bank(size=KERNEL):
    """Starting kernels: the two patterns, at the scales this camera gives.

    Training starts here rather than from noise, so the fit begins at the
    hand-designed baseline and can only improve on it - and so a learned kernel
    stays interpretable, because you can render it and see whether it is still a
    disc and a bar or has wandered into something else.
    """
    return np.stack([
        log_kernel(size, 1.5),                      # pupil, near
        log_kernel(size, 2.4),                      # pupil, close up
        ridge_kernel(size, 1.4, 3.2, 0.0),          # lid, level
        ridge_kernel(size, 1.4, 3.2, 22.0),         # lid, arced one way
        ridge_kernel(size, 1.4, 3.2, -22.0),        # and the other
    ])


def patches(crops, size=KERNEL, normalise=NORMALISE):
    """``(N, positions, size*size)`` sliding windows over standardised crops."""
    x = np.asarray(crops, dtype=np.float32)
    if x.ndim == 2:
        x = x[None]
    if normalise == "standardise":
        m = x.mean(axis=(1, 2), keepdims=True)
        s = x.std(axis=(1, 2), keepdims=True) + 1e-6
        x = (x - m) / s
    else:
        x = x / 255.0
    win = np.lib.stride_tricks.sliding_window_view(x, (size, size), axis=(1, 2))
    n, ph, pw = win.shape[0], win.shape[1], win.shape[2]
    return np.ascontiguousarray(win.reshape(n, ph * pw, size * size)), ph, pw


def pooled_features(pat, ph, pw, kernels, use_position=True):
    """``(features, peak responses, argmax indices)``.

    Per kernel: the maximum response anywhere, and optionally where it was. The
    position is *not* differentiable - it is an index - so it is passed through as
    a constant. It earns its place by letting the head learn a soft prior:
    a blob anywhere is fine, a blob jammed against the top edge is an eyebrow.
    """
    flat = kernels.reshape(len(kernels), -1)
    resp = pat @ flat.T                                  # (N, positions, K)
    idx = resp.argmax(axis=1)                            # (N, K)
    peak = np.take_along_axis(resp, idx[:, None, :], axis=1)[:, 0, :]
    feats = [peak]
    if use_position:
        feats.append((idx // pw) / max(ph - 1, 1) * 2.0 - 1.0)
        feats.append((idx % pw) / max(pw - 1, 1) * 2.0 - 1.0)
    return np.concatenate(feats, axis=1), peak, idx


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


class PooledModel(model.CropSource):
    """Learned kernels, max-pooled, with a logistic head. P(open) per crop.

    Interchangeable with :class:`bgate.model.EyeModel` everywhere - same
    ``predict``, same ``eyes``, same crop geometry - so the live view and the
    registry do not need to know which kind they are holding. ``blur`` is None
    because a pooled model has none; that is what the live view checks.
    """

    kind = "pooled"

    def __init__(self, kernels, coef, intercept, accuracy=float("nan"),
                 use_position=True, normalise=NORMALISE, level_width=True):
        self.kernels = np.asarray(kernels, dtype=np.float32)
        self.coef = np.asarray(coef, dtype=np.float32)
        self.intercept = float(intercept)
        self.accuracy = float(accuracy)
        self.use_position = bool(use_position)
        self.normalise = normalise
        self.level_width = bool(level_width)

    @property
    def size(self):
        return self.kernels.shape[-1]

    def features(self, crops):
        pat, ph, pw = patches(crops, self.size, self.normalise)
        feats, _peak, _idx = pooled_features(pat, ph, pw, self.kernels,
                                             self.use_position)
        return feats

    def p_closed(self, crops):
        crops = [c for c in crops if c is not None]
        if not crops:
            return np.array([])
        return _sigmoid(self.features(crops) @ self.coef + self.intercept)

    def predict(self, crops):
        """P(open) per crop, preserving None entries."""
        out, usable = [], [c for c in crops if c is not None]
        probs = iter(1.0 - self.p_closed(usable)) if usable else iter(())
        for c in crops:
            out.append(float(next(probs)) if c is not None else None)
        return out

    def peaks(self, crop):
        """Per kernel: (response, row, col) of the winning position, in crop
        pixels. For drawing what the model actually keyed on."""
        pat, ph, pw = patches([crop], self.size, self.normalise)
        _f, peak, idx = pooled_features(pat, ph, pw, self.kernels,
                                        self.use_position)
        half = self.size // 2
        return [(float(peak[0, k]), int(idx[0, k] // pw) + half,
                 int(idx[0, k] % pw) + half) for k in range(len(self.kernels))]

    def describe(self):
        """One line for the live view header."""
        k = len(self.kernels)
        return (f"pooled  {k}x{self.size}x{self.size} kernels"
                f"{' + position' if self.use_position else ''}"
                f"  norm {self.normalise}")

    def save(self, path):
        np.savez(path, kernels=self.kernels, coef=self.coef,
                 intercept=self.intercept, cv_accuracy=self.accuracy,
                 use_position=self.use_position, normalise=self.normalise,
                 level_width=self.level_width, in_h=data.IN_H, in_w=data.IN_W,
                 pad_w=data.PAD_W, pad_h=data.PAD_H, kind=self.kind)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        return cls(d["kernels"], d["coef"], d["intercept"],
                   d.get("cv_accuracy", float("nan")),
                   bool(d["use_position"]), str(d["normalise"]),
                   bool(d["level_width"]) if "level_width" in d.files else True)


def _train_once(X_crops, y, size, bank, use_position, epochs, batch, lr, l2,
                learn_kernels, seed):
    """Adam on the kernels and the head together. Returns (kernels, coef, b)."""
    rng = np.random.default_rng(seed)
    pat, ph, pw = patches(X_crops, size)
    n = len(y)
    kernels = bank.copy()
    k = len(kernels)
    n_feat = k * (3 if use_position else 1)
    coef = np.zeros(n_feat, dtype=np.float32)
    intercept = 0.0

    params = [kernels.reshape(k, -1), coef]
    moments = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    mb, vb, step = 0.0, 0.0, 0

    for _epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            sel = order[start:start + batch]
            pb, yb = pat[sel], y[sel]
            flat = params[0]
            resp = pb @ flat.T
            idx = resp.argmax(axis=1)
            peak = np.take_along_axis(resp, idx[:, None, :], axis=1)[:, 0, :]
            feats = [peak]
            if use_position:
                feats.append((idx // pw) / max(ph - 1, 1) * 2.0 - 1.0)
                feats.append((idx % pw) / max(pw - 1, 1) * 2.0 - 1.0)
            F = np.concatenate(feats, axis=1)

            p = _sigmoid(F @ params[1] + intercept)
            g_logit = (p - yb) / len(sel)                       # (b,)

            g_coef = F.T @ g_logit + l2 * params[1]
            g_b = float(g_logit.sum())
            # Only the winning position contributed to a max, so the gradient
            # into a kernel is exactly the patch that won, per sample.
            g_peak = g_logit[:, None] * params[1][None, :k]     # (b, k)
            win = pb[np.arange(len(sel))[:, None], idx]         # (b, k, size^2)
            g_flat = np.einsum("bk,bkd->kd", g_peak, win)

            step += 1
            for i, g in enumerate((g_flat, g_coef)):
                if i == 0 and not learn_kernels:
                    continue
                m, v = moments[i]
                m *= 0.9
                m += 0.1 * g
                v *= 0.999
                v += 0.001 * g * g
                mh = m / (1 - 0.9 ** step)
                vh = v / (1 - 0.999 ** step)
                params[i] -= lr * mh / (np.sqrt(vh) + 1e-8)
            mb = 0.9 * mb + 0.1 * g_b
            vb = 0.999 * vb + 0.001 * g_b * g_b
            intercept -= lr * (mb / (1 - 0.9 ** step)) / \
                (np.sqrt(vb / (1 - 0.999 ** step)) + 1e-8)

            if learn_kernels:
                # Re-project every step: the zero-mean and unit-norm properties
                # are what make a flat region score 0 and stop a kernel from
                # inflating its own feature, and gradient steps do not preserve
                # either one.
                flat = params[0]
                flat -= flat.mean(axis=1, keepdims=True)
                norms = np.linalg.norm(flat, axis=1, keepdims=True)
                np.divide(flat, np.maximum(norms, 1e-9), out=flat)

    return params[0].reshape(bank.shape), params[1], intercept


def fit(samples, size=KERNEL, bank=None, use_position=True, epochs=40,
        batch=64, lr=0.02, l2=1e-3, learn_kernels=True, seed=0, cv=True,
        n_jobs=-1):
    """Fit on labelled samples. Returns (kernels, coef, intercept, accuracy).

    Validation groups by capture burst, exactly as the pixel model does: frames
    from one key press are near duplicates, and splitting them across train and
    test scores memorisation.

    ``cv=False`` skips that and returns nan for the accuracy. Use it whenever the
    caller is doing its own held-out scoring, which is most of them - the cross
    validation is five extra trainings, so leaving it on inside an outer fold
    loop costs 6x for a number that gets thrown away.

    ``learn_kernels=False`` freezes the bank and fits only the head, which is the
    comparison that says whether learning the kernels bought anything over the
    textbook ones. Keep that honest - a fitted kernel that cannot beat a
    Laplacian of Gaussian has not earned its place.
    """
    bank = default_bank(size) if bank is None else np.asarray(bank, np.float32)
    crops = [s["crop"] for s in samples]
    y = np.array([1.0 if s["label"] == data.CLOSED else 0.0 for s in samples],
                 dtype=np.float32)
    groups = np.array([s.get("burst") or data.burst_of(s["frame"])
                       for s in samples])
    if len(set(y)) < 2:
        raise ValueError("need both open and closed samples")

    accuracy = float("nan")
    if cv and len(set(groups)) >= 2:
        accuracy = cross_val_accuracy(
            crops, y, groups, size, bank, use_position, epochs, batch, lr, l2,
            learn_kernels, seed, n_jobs)

    kern, coef, b = _train_once(crops, y, size, bank, use_position, epochs,
                                batch, lr, l2, learn_kernels, seed)
    return kern, coef, b, accuracy


def _fold(crops, y, tr, te, size, bank, use_position, epochs, batch, lr, l2,
          learn_kernels, seed):
    kern, coef, b = _train_once([crops[i] for i in tr], y[tr], size, bank,
                                use_position, epochs, batch, lr, l2,
                                learn_kernels, seed)
    m = PooledModel(kern, coef, b, use_position=use_position)
    pc = m.p_closed([crops[i] for i in te])
    return float(((pc > 0.5) == (y[te] > 0.5)).mean())


def cross_val_accuracy(crops, y, groups, size=KERNEL, bank=None,
                       use_position=True, epochs=40, batch=64, lr=0.02,
                       l2=1e-3, learn_kernels=True, seed=0, n_jobs=-1):
    """Held-out accuracy over capture bursts. Folds are independent, so they run
    in parallel - a fit is a few seconds and there are five of them."""
    from joblib import Parallel, delayed
    from sklearn.model_selection import GroupKFold

    bank = default_bank(size) if bank is None else np.asarray(bank, np.float32)
    splits = list(GroupKFold(n_splits=min(5, len(set(groups)))).split(
        crops, y, groups))
    scores = Parallel(n_jobs=n_jobs)(
        delayed(_fold)(crops, y, tr, te, size, bank, use_position, epochs,
                       batch, lr, l2, learn_kernels, seed)
        for tr, te in splits)
    return float(np.mean(scores))
