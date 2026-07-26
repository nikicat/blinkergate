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


def text_width(s, size, thick=1.0):
    """Rendered width in pixels, for fitting a label into a known space rather
    than guessing and having it clipped."""
    (w, _h), _base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, size * S,
                                     max(1, int(round(thick * S * 0.6))))
    return w


def enlarge(gray):
    """IR frame at UI scale, nearest-neighbour so the real pixel grid stays visible."""
    return cv2.resize(gray, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)


def iris_circle(shape, lm, iris):
    """Centre and radius of the iris in pixels of the given image shape.

    Nothing in the live view uses this - the preview is framed by the crop's own
    affine now. It stays because it is the mesh's *independent* estimate of where
    the pupil is, and comparing it against a blob found in the pixels is what
    separates a mislocated crop from an off-centre gaze (see FINDINGS, "Where the
    pupil actually is"). Deleting it would remove one side of that measurement.
    """
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


# How much of the eye's surroundings the preview shows, in eye widths. Wider than
# the crop's own PAD_W so the crop outline has margin around it and you can see
# what was left out.
INSET_PAD = 2.6


def eye_inset(gray, m, lm, ring, quad, title, p_open, shut, extra=None):
    """One eye, magnified and **levelled the way the classifier sees it**.

    ``m`` is the affine into this window, from :func:`bgate.data.level_affine`, so
    the preview shares its definition of upright with the training crop. ``quad``
    is the region actually sampled, in source pixels; drawn here it shows exactly
    which pixels the verdict came from and how much context was discarded.

    This used to be an axis-aligned cut centred on the iris, which disagreed with
    the crop on rotation, centre and scale at once - a preview that could not
    display the misframing it exists to reveal.
    """
    out_w, out_h = int(INSET_W * S), int(INSET_H * S)
    h, w = gray.shape[:2]
    view = cv2.warpAffine(gray, m, (out_w, out_h), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_REPLICATE)
    view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    def into_view(pts):
        return (np.asarray(pts, np.float32) @ m[:, :2].T + m[:, 2]).astype(np.int32)

    color = (0, 0, 255) if shut else (0, 255, 0)
    lid = into_view([[lm[i].x * w, lm[i].y * h] for i in ring])
    cv2.polylines(view, [lid], True, color, max(1, int(round(S * 0.7))), cv2.LINE_AA)
    # The crop, in the preview's own coordinates. Axis-aligned here by
    # construction, because both were levelled by the same angle - if it ever
    # looks tilted, the two have come apart.
    cv2.polylines(view, [into_view(quad)], True, (0, 200, 255),
                  max(1, int(round(S * 0.6))), cv2.LINE_AA)

    src_w = int(round(float(np.linalg.norm(np.asarray(quad[1]) - np.asarray(quad[0])))))
    txt(view, f"{title}  {'SHUT' if shut else 'OPEN'}", P(6, 22), 0.6, color, 1.6)
    txt(view, f"P(open) {p_open:.3f}", P(6, 46), 0.5, color, 1.2)
    if extra:
        txt(view, extra, P(6, 68), 0.42, (200, 200, 120), 1.2)
    txt(view, f"crop {src_w}px wide  ->  {out_w // max(src_w, 1)}x",
        P(6, INSET_H - 10), 0.42, (185, 185, 185))
    return view


# Timeline geometry, in design pixels.
LANE_H = 54                 # one eye's lane
LANE_H_COMPACT = 36         # four lanes (two modes compared) in the same strip
TIMELINE_HEAD = 17          # the row of second markers above them
TIMELINE_GUTTER = 84        # left margin, for the lane labels
# A dropout longer than this is drawn as a hole rather than painted across. At
# ~7.5 usable frames a second the normal gap is 0.13s, so this is three missed
# frames - long enough to be a real loss, short enough not to smear one.
TIMELINE_GAP = 0.4

OPEN_BG = (28, 74, 36)      # BGR, muted so the trace reads on top of it
SHUT_BG = (38, 30, 98)
NODATA_BG = (34, 34, 38)
TRACE = (232, 232, 232)

# Frame-time lane. Milliseconds rather than frames per second: this hardware is
# described in ms throughout - 66 for the raw camera, ~133 once unlit strobe
# frames are dropped, ~150 for a natural blink - and a dropped frame reads as a
# clear spike to 266ms where on an fps axis it collapses towards zero.
FRAME_LANE_H = 46
FRAME_MS_MAX = 300.0        # top of the scale
FRAME_REFS = (66, 133)      # measured: camera interval, and after the strobe drop
FRAME_BAR = (150, 116, 60)
FRAME_OVER = (60, 150, 220)     # off the top of the scale, so it cannot be read
FRAME_BG = (30, 30, 34)
CAPTURE_TICK = (215, 215, 225)   # one per frame the camera delivered


def _frame_lane(panel, top, bot, x0, x1, xof, pts, captures=None):
    """Frame interval as bars, one per gap between consecutive samples.

    A bar spans exactly the interval it measures, so it lines up with whatever
    the eye lanes were doing while the camera took that long. Nothing extra is
    stored for this: the intervals are the differences between the timestamps the
    lanes are already plotted against, which is also why it cannot drift out of
    sync with them.
    """
    span = bot - top - 2
    panel[top:bot, x0:x1] = FRAME_BG

    for a, b in zip(pts, pts[1:]):
        ms = (b[0] - a[0]) * 1000.0
        xa = max(xof(a[0]), x0)
        xb = min(max(xof(b[0]), xof(a[0]) + 1), x1)
        if xb <= xa:
            continue
        h = max(int(min(ms / FRAME_MS_MAX, 1.0) * span), 1)
        panel[bot - h - 1:bot - 1, xa:xb] = (
            FRAME_OVER if ms > FRAME_MS_MAX else FRAME_BAR)

    # A tick per captured frame, along the top of the lane. A tick with no bar
    # starting under it is a frame the camera delivered and the pipeline dropped -
    # which is the one thing a frame rate on its own can never tell you.
    if captures:
        for t in captures:
            x = xof(t)
            if x0 <= x < x1:
                panel[top:top + max(2, int(3 * S)), x:x + 1] = CAPTURE_TICK

    # After the bars, not before: at the normal 133ms the bars are taller than the
    # 66ms line and were hiding it, which left the lane with one reference instead
    # of two.
    for ref in FRAME_REFS:
        y = bot - int(min(ref / FRAME_MS_MAX, 1.0) * span) - 1
        cv2.line(panel, (x0, y), (x1, y), (210, 210, 216), 1)
        label = f"{ref}ms"
        txt(panel, label, (x1 - text_width(label, 0.26) - int(4 * S),
                           y - int(3 * S)), 0.26, (215, 215, 220))


def time_x(t, now, window, x0, plot_w):
    """The timeline's time -> x pixel mapping.

    A module function rather than a closure so the click handler can use the
    SAME rounding: selecting by inverting the maths in time space picked the
    wrong frame on boundary pixels, where int(round()) had moved an edge.
    """
    frac = 1.0 - (now - t) / window
    return int(round(x0 + min(max(frac, 0.0), 1.0) * plot_w))


def axis_now(pts, fallback=0.0):
    """The right edge of the time axis: one median interval past the newest
    sample.

    With the edge AT the newest stamp, the current frame's span was empty and it
    drew as a sliver. Held for a median frame it gets the width every other
    sample got when it was newest. Capped by TIMELINE_GAP so a stall does not
    stretch it, and shared with the click handler, which must map pixels with
    the same axis the bars were drawn on.
    """
    if not pts:
        return fallback
    gaps = sorted(b[0] - a[0] for a, b in zip(pts, list(pts)[1:]))
    hold = min(gaps[len(gaps) // 2], TIMELINE_GAP) if gaps else TIMELINE_GAP
    return pts[-1][0] + hold


def timeline(width, history, now, thresh, window, lanes, frame_time=True,
             captures=None, cursor=None, lane_height=None):
    """Time-synced lanes of P(open), most recent at the right edge.

    ``history`` is an iterable of ``(t, p0, p1)`` with ``None`` for a frame where
    there was nothing to classify. ``lanes`` is ``[(position, tag), ...]`` in the
    order to draw them, where position indexes the probabilities in an entry.

    The position and the tag travel together on purpose. The lanes are laid out to
    match the picture, so the *display* order is the reverse of the dataset's, and
    passing only the labels reversed would have drawn the subject's left eye under
    the words "YOUR RIGHT". Every left/right bug in this project has come from a
    name and an index being reordered separately.

    Plotted against **time, not sample index**. The sensor delivers ~7.5 usable
    frames a second and drops them unevenly, so equal-width samples would stretch
    and squash the exact durations this exists to show: a natural blink is ~150 ms,
    about one frame. Against an index axis a one-frame blink and a four-frame hold
    look the same width.

    A gap in detection is drawn as a gap. A frame with no face is not an open eye,
    and painting it green would invent the one reading that matters most.

    ``frame_time`` adds a third lane of the interval between frames, sharing the
    same x mapping so a stall lines up with whatever the eyes were doing during it.

    ``cursor`` is the stamp of the frame selected while paused, highlighted as a
    bar over that frame's whole span.
    """
    lane_h = int((lane_height or LANE_H) * S)
    head_h = int(TIMELINE_HEAD * S)
    frame_h = int(FRAME_LANE_H * S) if frame_time else 0
    panel = np.full((head_h + len(lanes) * lane_h + frame_h, width, 3), 18,
                    np.uint8)
    x0, x1 = int(TIMELINE_GUTTER * S), width - int(10 * S)
    plot_w = max(x1 - x0, 1)
    pts = list(history)
    now = max(now, axis_now(pts, now))

    def xof(t):
        return time_x(t, now, window, x0, plot_w)

    # How long each sample stands for: until the next one, or until now for the
    # newest, and never across a dropout. Computed once and shared, because the
    # background and the trace have to agree about it - two loops deriving the
    # same intervals separately is how they come apart.
    spans = []
    for j, entry in enumerate(pts):
        t = entry[0]
        t_end = min(pts[j + 1][0] if j + 1 < len(pts) else now, t + TIMELINE_GAP)
        spans.append((t, t_end))

    for i, (pos, tag) in enumerate(lanes):
        top = head_h + i * lane_h
        bot = top + lane_h - int(3 * S)
        span = bot - top - 2
        panel[top:bot, x0:x1] = NODATA_BG
        thick = max(1, int(round(S * 0.7)))

        def bounds(t, t_end):
            return (max(xof(t), x0), min(max(xof(t_end), xof(t) + 1), x1))

        # Piecewise-constant background: each sample holds for its span. That is
        # what a discrete sample means.
        for (t, t_end), entry in zip(spans, pts):
            p = entry[1 + pos]
            xa, xb = bounds(t, t_end)
            if xb > xa:
                panel[top:bot, xa:xb] = (
                    NODATA_BG if p is None else
                    SHUT_BG if p < thresh else OPEN_BG)

        cv2.line(panel, (x0, bot - int(thresh * span) - 1),
                 (x1, bot - int(thresh * span) - 1), (110, 110, 120), 1)

        # A staircase, not a polyline: the trace holds flat across a sample's span
        # and steps at the boundary, which is the same claim the background colour
        # makes. Sloping between samples asserts a ramp through time nobody
        # measured, and it disagreed with the block of colour underneath it.
        prev_y = prev_end = None
        for (t, t_end), entry in zip(spans, pts):
            p = entry[1 + pos]
            if p is None:
                prev_y, prev_end = None, t_end
                continue
            y = bot - int(p * span) - 1
            xa, xb = bounds(t, t_end)
            # No riser across a hole: the previous span was cut short by a dropout,
            # so there is nothing to step from.
            if prev_y is not None and prev_end is not None \
                    and t <= prev_end + 1e-9 and prev_y != y:
                cv2.line(panel, (xa, prev_y), (xa, y), TRACE, thick, cv2.LINE_AA)
            if xb > xa:
                cv2.line(panel, (xa, y), (xb, y), TRACE, thick, cv2.LINE_AA)
            prev_y, prev_end = y, t_end

        latest = next((e[1 + pos] for e in reversed(pts)
                       if e[1 + pos] is not None),
                      None)
        shut = latest is not None and latest < thresh
        colour = (0, 0, 255) if shut else (0, 255, 0) if latest is not None \
            else (130, 130, 130)
        txt(panel, tag, (int(6 * S), top + int(20 * S)), 0.36, colour, 1.2)
        txt(panel, "-" if latest is None else f"{latest:.2f}",
            (int(6 * S), top + int(40 * S)), 0.38, colour, 1.2)

    if frame_time:
        top = head_h + len(lanes) * lane_h
        _frame_lane(panel, top, top + frame_h - int(3 * S), x0, x1, xof,
                    pts, captures)
        gaps = sorted((b[0] - a[0]) * 1000.0 for a, b in zip(pts, pts[1:]))
        # The median, not the newest: one interval is noise, and the median is
        # what the frame budget actually is.
        mid = gaps[len(gaps) // 2] if gaps else None
        txt(panel, "FRAME TIME", (int(6 * S), top + int(17 * S)), 0.34,
            (150, 150, 160), 1.2)
        txt(panel, "-" if mid is None else f"{mid:.0f}ms {1000 / mid:.1f}/s",
            (int(6 * S), top + int(36 * S)), 0.34,
            (150, 150, 160) if mid else (110, 110, 118), 1.2)

    # Second markers last, so they cross every lane and the three read as one
    # time axis rather than three graphs that happen to be stacked.
    for s in range(int(window), -1, -1):
        x = xof(now - s)
        cv2.line(panel, (x, head_h), (x, panel.shape[0]), (86, 86, 94), 1)
        if s % 2 == 0:
            label = f"-{s}s" if s else "now"
            # Clamped, not offset blindly: the "now" tick sits at the right edge
            # and its label ran off it.
            lx = min(x + int(2 * S), width - text_width(label, 0.32) - int(3 * S))
            txt(panel, label, (lx, int(12 * S)), 0.32, (120, 120, 128))

    if cursor is not None and pts:
        # The span runs to the NEXT sample, deliberately ignoring TIMELINE_GAP:
        # this is a selection highlight, not a data claim, and only bars that
        # tile the axis keep the promise that the click lands inside the bar it
        # selected. Tinted rather than painted, so the trace stays readable
        # under it.
        nxt = next((e[0] for e in pts if e[0] > cursor), now)
        xa = xof(cursor)
        xb = max(min(xof(nxt), x1), xa + 1)     # at least a pixel, regardless
        tint = np.full((panel.shape[0], xb - xa, 3), (0, 220, 255), np.uint8)
        panel[:, xa:xb] = cv2.addWeighted(panel[:, xa:xb], 0.65, tint, 0.35, 0)
    return panel


def inset_strip(width, groups):
    """One pair of eye insets under each camera pane above it.

    ``groups`` is a pair of insets per pane, in the order they should read
    left to right - which is the REVERSE of the dataset's, because your left
    eye appears on the image's right. Centred under its own pane so the eyes
    sit beneath the face they were cut from.
    """
    band = np.full((int(INSET_H * S), width, 3), 18, np.uint8)
    iw, gap = int(INSET_W * S), int(8 * S)
    span = width // max(len(groups), 1)
    for g, pair in enumerate(groups):
        block = len(pair) * iw + (len(pair) - 1) * gap
        x = g * span + max((span - block) // 2, 0)
        for inset in pair:
            if x + iw <= width:
                band[:, x:x + iw] = inset
            x += iw + gap
    return band


def text_pane(width, lines):
    """The status lines, full width, in their own pane.

    A pane rather than a column beside the insets: comparing two modes puts
    four insets across the frame and leaves no width to sit next to.
    Row height follows text size, so the big state line gets its room and the
    small legends do not waste any.
    """
    rows = [int((size * 30 + 16) * S) for _line, size, _colour in lines]
    pane = np.full((sum(rows) + int(10 * S), width, 3), 18, np.uint8)
    y = 0
    for (line, size, colour), h in zip(lines, rows):
        y += h
        txt(pane, line, (int(20 * S), y), size, colour, 1.2)
    return pane


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
