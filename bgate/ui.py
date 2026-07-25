"""Drawing helpers.

Everything is authored in "design pixels" and multiplied by :data:`S` at draw
time, so the interface stays sharp on a high-DPI screen instead of being drawn
small and upscaled. The one thing deliberately *not* smoothed is the IR frame
itself, which is enlarged nearest-neighbour: it should look pixelated, because
that is honestly how little data the decision rests on.
"""

import math

import cv2
import numpy as np

S = 2.0                                   # UI scale, set from --scale
FACE_MODEL = "face_landmarker.task"       # MediaPipe bundle, see fetch_model.py

# MediaPipe FaceMesh contours. RING_L is the SUBJECT'S LEFT eye, which on an
# unmirrored feed appears on the RIGHT of the image. Verified against the pixels,
# not assumed: RING_L centres at x=347 of 640, RING_R at x=307.
RING_L = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RING_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# Iris landmarks exist only in the refined 478-point mesh: centre + 4 boundary.
IRIS_L = [473, 474, 475, 476, 477]
IRIS_R = [468, 469, 470, 471, 472]

INSET_W, INSET_H = 300, 170


def P(x, y):
    return (int(x * S), int(y * S))


def txt(img, s, org, size, color, thick=1.0):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size * S, color,
                max(1, int(round(thick * S * 0.6))), cv2.LINE_AA)


def enlarge(gray):
    """IR frame at UI scale, nearest-neighbour so the real pixel grid stays visible."""
    return cv2.resize(gray, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)


def iris_circle(shape, lm, iris):
    """Centre and radius of the iris in pixels of the given image shape."""
    h, w = shape[:2]
    p = np.array([[lm[i].x * w, lm[i].y * h] for i in iris])
    c = p[0]
    return float(c[0]), float(c[1]), max(float(np.linalg.norm(p[1:] - c, axis=1).mean()), 1.5)


def face_quad(lm, w, h, roll=0.0, pad=0.06):
    """Corners of a box around the face, tilted with the head.

    Measured in the levelled frame and rotated back, so the box follows your roll
    instead of growing as you tilt - which also makes the tracked roll visible.
    """
    pts = np.array([[l.x * w, l.y * h] for l in lm])
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    theta = math.radians(-roll)
    cos, sin = math.cos(theta), math.sin(theta)
    rel = pts - (cx, cy)
    level = np.stack([rel[:, 0] * cos - rel[:, 1] * sin,
                      rel[:, 0] * sin + rel[:, 1] * cos], axis=1)
    x0, y0 = level.min(0)
    x1, y1 = level.max(0)
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    corners = np.array([[x0 - mx, y0 - my], [x1 + mx, y0 - my],
                        [x1 + mx, y1 + my], [x0 - mx, y1 + my]])
    back = np.stack([corners[:, 0] * cos + corners[:, 1] * sin,
                     -corners[:, 0] * sin + corners[:, 1] * cos], axis=1)
    return (back + (cx, cy)).astype(np.int32)


def draw_face_frame(img, lm, roll=0.0, colour=(0, 200, 255), label=None):
    """The face frame, drawn on an already-enlarged image."""
    h, w = img.shape[:2]
    quad = face_quad(lm, w, h, roll)
    cv2.polylines(img, [quad], True, colour, max(1, int(round(S * 0.6))), cv2.LINE_AA)
    # Corner ticks, so the frame reads as a frame rather than as a shape in the
    # scene - the IR image is busy and a plain rectangle disappears into it.
    for i, corner in enumerate(quad):
        nxt = quad[(i + 1) % 4]
        vec = (nxt - corner).astype(np.float32)
        vec /= max(np.linalg.norm(vec), 1e-6)
        tip = (corner + vec * (np.linalg.norm(nxt - corner) * 0.22)).astype(np.int32)
        cv2.line(img, tuple(corner), tuple(tip), colour,
                 max(2, int(round(S))), cv2.LINE_AA)
    if label:
        txt(img, label, (int(quad[:, 0].min()), int(quad[:, 1].min()) - int(6 * S)),
            0.42, colour, 1.2)
    return quad


def eye_inset(gray, lm, ring, iris, title, p_open, shut, extra=None):
    """One eye, magnified, with its state. Reports the true source width so it is
    always clear how many real pixels the decision came from."""
    out_w, out_h = int(INSET_W * S), int(INSET_H * S)
    h, w = gray.shape[:2]
    pts = np.array([[lm[i].x * w, lm[i].y * h] for i in ring])
    cx, cy, _rad = iris_circle(gray.shape, lm, iris)

    half_w = max((pts[:, 0].max() - pts[:, 0].min()) * 0.9, 10)
    half_h = half_w * out_h / out_w
    x0, x1 = int(max(0, cx - half_w)), int(min(w, cx + half_w))
    y0, y1 = int(max(0, cy - half_h)), int(min(h, cy + half_h))
    if x1 <= x0 + 1 or y1 <= y0 + 1:
        return np.zeros((out_h, out_w, 3), np.uint8)

    src_w = x1 - x0
    crop = cv2.cvtColor(gray[y0:y1, x0:x1], cv2.COLOR_GRAY2BGR)
    sx, sy = out_w / crop.shape[1], out_h / crop.shape[0]
    crop = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    color = (0, 0, 255) if shut else (0, 255, 0)
    lid = np.array([[(p[0] - x0) * sx, (p[1] - y0) * sy] for p in pts], np.int32)
    cv2.polylines(crop, [lid], True, color, max(1, int(round(S * 0.7))), cv2.LINE_AA)

    txt(crop, f"{title}  {'SHUT' if shut else 'OPEN'}", P(6, 22), 0.6, color, 1.6)
    txt(crop, f"P(open) {p_open:.3f}", P(6, 46), 0.5, color, 1.2)
    if extra:
        txt(crop, extra, P(6, 68), 0.42, (200, 200, 120), 1.2)
    txt(crop, f"{src_w}px wide  ->  {out_w // max(src_w, 1)}x",
        P(6, INSET_H - 10), 0.42, (185, 185, 185))
    return crop


def inset_row(left_in, right_in, width, lines):
    """Both eye insets side by side with a text column to their right."""
    band = np.full((int(INSET_H * S), width, 3), 18, np.uint8)
    iw = int(INSET_W * S)
    x1, x2 = int(20 * S), int(40 * S) + iw
    band[:, x1:x1 + iw] = left_in
    band[:, x2:x2 + iw] = right_in
    tx = int(60 * S) + 2 * iw
    for i, (line, size, color) in enumerate(lines):
        txt(band, line, (tx, int((34 + i * 32) * S)), size, color, 1.2)
    return band
