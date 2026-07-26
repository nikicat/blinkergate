#!/usr/bin/env python
"""blinkergate live view - watch detection and capture labelled samples.

Labelling and model history live in the web UI (``uv run serve.py``). This window
does the two things that need the camera: showing detection in real time, and
recording labelled bursts.

The two share state through ``models/active``. Switch generation in the browser
and this picks it up within a second, without a restart.

  o  both open          c  both closed
  1  YOUR LEFT shut     2  YOUR RIGHT shut       (one frame per press)
  h  save this frame as a detector test case (press when it says NO FACE)
  t  train a new PIXEL generation from the current dataset
  p  train a new POOLED generation (learned kernels, shift-invariant)
  g  cycle generation: follow-the-web-UI, then each one in turn
  b  override the blur (diagnostic, pixel models only - see below)
  q  quit               + -  resize              f  fullscreen

Two kinds of model, and `g` is how you compare them. The pixel one reads 960 fixed
pixels, so the pupil has to be where it was during training; the pooled one
convolves learned kernels and takes the maximum, so it does not care where the
pupil is. Measured on deliberate crop shifts, 96.3% -> 44.2% at 4px for the pixel
model against 99.5% -> 98.3% for the pooled one. The header names which kind is
loaded.

Blur is a property of a pixel model, not a switch: one fitted on blurred features
is meaningless applied to sharp ones. So the honest blur A/B is `g`, cycling
between generations that were *trained* differently - the header shows each one's
blur. `b` forces a different blur onto the current model and is labelled MISMATCH
when it disagrees with what that model was trained with; it is there to show what
the mismatch looks like, not to be run in. A pooled model has no blur.

Sides are YOURS. The feed is not mirrored, so your left eye appears on the RIGHT
of the picture - the insets and the timeline lanes are laid out to match the
picture, so the one on the right of the strip is the one on the right of the frame.

The timeline at the bottom is the last 10 seconds of P(open) per eye, plotted
against *time* rather than against frame number. That matters here: the sensor
gives ~7.5 usable frames a second and drops them unevenly, so on an index axis a
one-frame blink and a four-frame hold are the same width. A frame where nothing
could be classified is drawn as a hole, because a lost face is not an open eye.

A third lane shows the interval between verdicts in milliseconds, with the measured
66ms and 133ms marked, plus a tick per frame the camera delivered. A tick with no
bar under it is a frame the pipeline dropped - which is the difference between "the
detector lost the face" and "no frame arrived to look at".

Capture runs in its own thread, feeding a short queue that discards its oldest
frame when full. Serially, a pass over 133ms missed the next lit frame outright and
the next usable one came 266ms later, so the cost of being slightly slow was
quantised; the queue absorbs that and the verdict rate becomes the processing time
instead. `--workers` adds processing threads (they overlap MediaPipe's inference
but complete out of order) and `--queue` sets the buffer depth, where 1 is
always-freshest and never-buffered.
"""

import argparse
import sys
import time
from collections import deque

from bgate import qtfix  # noqa: F401  - must be imported before cv2

import cv2
import numpy as np

from bgate import data, ui
from bgate.camera import IRCamera, DEFAULT_DEVICE
from bgate.data import Dataset
from bgate.detect import default_locator
from bgate.pipeline import Pipeline
from bgate.registry import Registry, train_and_register

WIN = "blinkergate"
POLL_EVERY = 1.0              # seconds between checks for a generation switch
HISTORY_SECONDS = 10.0        # how far back the timeline reaches
# Design pixels. Tall enough for the two frames, the inset band and the
# timeline and its frame-time lane stacked - they pushed the content past 600.
WIN_W, WIN_H = 1280, 740
CAPTURES = {
    ord("o"): ("both OPEN", data.OPEN, data.OPEN),
    ord("c"): ("both CLOSED", data.CLOSED, data.CLOSED),
    ord("1"): ("YOUR LEFT shut", data.CLOSED, data.OPEN),
    ord("2"): ("YOUR RIGHT shut", data.OPEN, data.CLOSED),
}


class Verdict:
    """One processed frame, carried whole from a worker to the display.

    The frame travels with its result: rendering needs the pixels, and a capture
    keypress should record the frame you were looking at when you pressed - not
    whatever the camera has produced since.
    """

    __slots__ = ("stamp", "gray", "det", "probs", "quads", "locator", "angle")

    def __init__(self, stamp, gray, det, probs, quads, locator, angle):
        self.stamp, self.gray, self.det = stamp, gray, det
        self.probs, self.quads = probs, quads
        self.locator, self.angle = locator, angle


class App:
    def __init__(self, args):
        self.args = args
        self.dataset = Dataset(args.dataset)
        self.registry = Registry(args.models)
        self.model = None
        self.model_gen = None
        self.trained_blur = None
        self.active_seen = -1.0
        self.gens = []
        self.gens_seen = -1.0
        self.last_poll = 0.0
        self.status = ""
        self.captured = 0
        self.hard = 0
        self.roll = 0.0
        self.locator_used = None
        self.pinned = False          # generation chosen here, not in the web UI
        self.blur_override = None    # None = whatever the model was trained with
        # (t, P(open) left, P(open) right), one entry per frame, None where there
        # was nothing to classify. Timestamped rather than merely ordered: the
        # timeline plots against time, and monotonic() because this is an elapsed
        # interval and must not move if the wall clock does.
        self.history = deque()
        self.gens = self.registry.ids()
        self.load_model()

    def make_processor(self):
        """A ``process(stamp, gray)`` for one worker thread.

        Called inside the worker, so the locator it builds belongs to that thread
        alone. ``self.model`` is re-read per frame rather than captured, because
        `g` and `t` swap it from the main thread - and a reference read is the one
        thing that is safe to do across threads without ceremony.
        """
        locator = default_locator()

        def process(stamp, gray):
            det = locator.locate(gray)
            probs = quads = None
            model = self.model
            if det and model:
                probs, _crops, quads = model.eyes(gray, det.landmarks,
                                                  roll=det.roll)
            return Verdict(stamp, gray, det, probs, quads, locator.used,
                           getattr(locator, "angle", 0.0))
        return process

    def absorb(self, v):
        """Take a worker's verdict into the display state.

        Stamped with the time the frame was CAPTURED, not the time we finished
        thinking about it. The timeline is about when the eye was in that state;
        dating it by completion would smear every reading by the processing time
        and make a slow pass look like a slow blink.
        """
        self.history.append((v.stamp, *(v.probs or (None, None))))
        cutoff = v.stamp - HISTORY_SECONDS
        while self.history and self.history[0][0] < cutoff:
            self.history.popleft()
        if v.det:
            self.roll = v.angle
            self.locator_used = v.locator

    def load_model(self, gen=None):
        if gen is None:
            gen, model = self.registry.load_active()
            self.pinned = False
        else:
            model = self.registry.load(gen)
            self.pinned = True
        self.model, self.model_gen = model, gen
        self.active_seen = self.registry.active_mtime()
        self.apply_blur()
        self.status = (f"{gen} ({model.accuracy:.1%} held-out)" if model
                       else "no model - capture samples, then press t")

    def apply_blur(self):
        """Push the override, if any, onto the loaded model.

        A pooled model has no blur - it convolves learned kernels instead - so
        there is nothing to override and nothing to restore.
        """
        if self.model is None or self.model.blur is None:
            return
        if self.trained_blur is None:
            self.trained_blur = self.model.blur
        self.model.blur = (self.trained_blur if self.blur_override is None
                           else self.blur_override)

    def cycle_generation(self):
        """Step through the generations, and back to following the web UI.

        ``None`` is a real option in the cycle, not an absence of one: pinning
        needs to be reversible or the live view silently stops tracking whatever
        you activate in the browser.
        """
        self.gens = self.registry.ids()
        if not self.gens:
            return
        options = [None] + self.gens          # None = follow models/active
        current = self.model_gen if self.pinned else None
        i = options.index(current) if current in options else 0
        chosen = options[(i + 1) % len(options)]
        self.trained_blur = None
        self.load_model(chosen)
        if chosen is None:
            print(f"  following the web UI -> {self.model_gen}")
        else:
            print(f"  pinned {chosen}  {self.model.describe()}"
                  f"  ({self.model.accuracy:.1%})")

    def cycle_blur(self):
        """Force a blur onto the current model. Diagnostic only."""
        if self.model is None or self.model.blur is None:
            print("  this model has no blur to override")
            return
        options = [None, 0.0, 0.8, 1.6, 2.4]
        i = options.index(self.blur_override) if self.blur_override in options else 0
        self.blur_override = options[(i + 1) % len(options)]
        self.apply_blur()
        print(f"  blur override -> {self.blur_override}"
              f"  (model trained with {self.trained_blur})")

    def poll(self):
        """Follow what the web UI does, without needing a channel to it.

        Two things can change under us: which generation is active (it writes
        ``models/active``) and which generations exist (training adds a directory).
        Both are noticed by mtime, so this costs a stat every second.
        """
        now = time.time()
        if now - self.last_poll < POLL_EVERY:
            return
        self.last_poll = now

        mtime = self.registry.dir_mtime()
        if mtime != self.gens_seen:
            self.gens_seen = mtime
            gens = self.registry.ids()
            if gens != self.gens:
                new = [g for g in gens if g not in self.gens]
                self.gens = gens
                if new:
                    print(f"  new generation{'s' if len(new) > 1 else ''}: "
                          f"{', '.join(new)}")

        if self.pinned:
            return          # chosen here; the web UI does not override it
        if self.registry.active_mtime() != self.active_seen:
            self.trained_blur = None
            self.load_model()
            print(f"  switched to {self.model_gen}")

    def capture(self, gray, det, name, label_left, label_right):
        """One frame, now. Holding a pose for a burst produced ~15 near-identical
        frames worth about one frame of information (measured: training on 1 per
        press scored 88.1% against 86.5% for all 15), so a press takes exactly
        the frame you pressed on."""
        if not det:
            self.status = f"{name}: no face - nothing captured"
            return
        stamp = int(time.time() * 1000)
        self.dataset.add_capture(gray, label_left, label_right, stamp,
                                 lm=det.landmarks)
        self.captured += 1
        counts = self.dataset.counts()
        self.status = (f"captured {name}  ({self.captured} this session, "
                       f"{counts[data.OPEN]}/{counts[data.CLOSED]} open/closed)")
        print(f"  {self.status}")

    def capture_hard(self, gray, det):
        """Record a frame the detector should have handled.

        Captures whether or not a face was found - the failures are the whole
        point, and the ordinary capture path refuses exactly those.
        """
        frame = self.dataset.add_detector_case(gray, bool(det))
        self.hard += 1
        found = "found" if det else "NO FACE"
        self.status = f"detector case {self.hard} saved ({found}): {frame}"
        print(f"  {self.status}")

    def train(self, kind="pixel"):
        # Shared with the web UI, so the two cannot drift on what trains a model.
        try:
            gen, accuracy, n = train_and_register(self.dataset, self.registry, kind)
        except Exception as exc:
            self.status = f"training failed: {exc}"
            print(f"  {self.status}")
            return
        self.trained_blur = None
        self.blur_override = None
        self.load_model()
        self.status = f"{gen} ({kind}): {n} samples, held-out {accuracy:.1%}"
        print(f"  {self.status}")

    def blur_mismatch(self):
        """Is the loaded model being fed features it was not fitted on?

        Only a pixel model can be: a pooled one has no blur, so a leftover
        override from the generation before it is not a mismatch. Reporting one
        anyway painted the header red on a model that was working perfectly.
        """
        return (self.model is not None and self.model.blur is not None
                and self.blur_override is not None
                and self.blur_override != self.trained_blur)

    def model_line(self):
        if self.model is None:
            return self.status
        text = (f"{self.model_gen}  {self.model.describe()}"
                f"  ({self.model.accuracy:.1%} held-out)")
        text += "  [pinned]" if self.pinned else "  [web UI]"
        if len(self.gens) > 1:
            text += f"  ({len(self.gens)} gens)"
        if self.blur_mismatch():
            text += f"  MISMATCH: trained with {self.trained_blur}"
        return text

    def model_colour(self):
        # A mismatched blur is not a valid reading, so say so in red.
        return (0, 0, 255) if self.blur_mismatch() else (0, 200, 255)

    def render(self, v, pipe=None):
        """Draw one verdict. Pure display - the probabilities were computed in a
        worker and are not recomputed here, so a slow render costs smoothness and
        nothing else."""
        gray, det = v.gray, v.det
        raw = cv2.cvtColor(ui.enlarge(gray), cv2.COLOR_GRAY2BGR)
        annot = raw.copy()
        state = "no face"
        blank = np.zeros((int(ui.INSET_H * ui.S), int(ui.INSET_W * ui.S), 3), np.uint8)
        insets = [blank, blank]

        if det and det.landmarks is not None:
            ui.draw_face_frame(annot, det.landmarks, det.roll,
                               label=f"{v.locator or ''} {det.roll:+.0f}deg".strip())

        if det and v.probs is not None:
            lm = det.landmarks
            probs, quads = v.probs, v.quads
            shut = [p is not None and p < self.args.thresh for p in probs]
            # The quad, not a rectangle around it: the crop is cut in the levelled
            # frame, so an axis-aligned box would draw something the classifier
            # never sees and hide exactly the misplacement worth noticing.
            for quad, is_shut in zip(quads, shut):
                cv2.polylines(annot, [(quad * ui.S).astype(np.int32)], True,
                              (0, 0, 255) if is_shut else (0, 255, 0),
                              max(1, int(ui.S * 0.6)), cv2.LINE_AA)
            # Framed by the same affine the crop was cut with, so the preview is
            # levelled exactly as the classifier's input is - just larger, and
            # with the sampled region outlined inside it.
            h, w = gray.shape[:2]
            out_w = int(ui.INSET_W * ui.S)
            out_h = int(ui.INSET_H * ui.S)
            insets = []
            for (side, ring, tag), p, sh, quad in zip(data.SIDES, probs, shut,
                                                      quads):
                cx, cy, width = data.eye_centre(
                    lm, ring, w, h, det.roll if self.model.level_width else 0.0)
                half_w = max(width * ui.INSET_PAD / 2, 12)
                m = data.level_affine(cx, cy, half_w, half_w * out_h / out_w,
                                      det.roll, out_w, out_h)
                insets.append(ui.eye_inset(gray, m, lm, ring, quad, tag,
                                           p if p is not None else 1.0, sh))
            state = ("BOTH SHUT" if all(shut) else
                     "YOUR LEFT shut" if shut[0] else
                     "YOUR RIGHT shut" if shut[1] else "open")
        elif not self.model:
            state = "no model - capture samples and press t"

        ui.txt(annot, "DETECTION", ui.P(8, 22), 0.5, (0, 255, 0), 1.2)
        top = np.hstack([raw, annot])

        counts = self.dataset.counts()
        # Laid out to match the picture: your left eye is on the image's right.
        band = ui.inset_row(insets[1], insets[0], top.shape[1], [
            (state, 0.85,
             (0, 165, 255) if state not in ("open", "no face") else (150, 150, 150)),
            # This frame's roll, not the tracked one: it is the angle the crops
            # were actually levelled by, and showing the tracker's instead put two
            # different numbers on screen under the same name.
            (f"dataset: {counts[data.OPEN]} open / {counts[data.CLOSED]} closed / "
             f"{counts[data.SKIP]} skip    roll "
             f"{det.roll if det else self.roll:+.0f}deg"
             + (f"  via {v.locator}" if v.locator else ""),
             0.45, (200, 200, 120)),
            (self.model_line(), 0.42, self.model_colour()),
            # Pipeline health. A drop count is the whole reason for the threads:
            # a frame we chose not to process and a frame that never arrived look
            # identical in a frame rate, and only one of them is our fault.
            (pipe.summary() if pipe else "", 0.38, (140, 170, 190)),
            # Built from CAPTURES rather than written out, so the legend cannot
            # come to disagree with what the keys actually do.
            ("capture   " + "   ".join(f"{chr(k)} {name}"
                                       for k, (name, _l, _r) in CAPTURES.items()),
             0.38, (180, 180, 140)),
            ("h hard case   t train pixel   p train pooled   g cycle generation"
             "   b blur   q quit", 0.38, (150, 150, 150)),
        ])
        # Reversed to match the insets above, which match the picture. The index
        # into the history travels with its label so the two cannot be reordered
        # apart - drawing the subject's left eye under "YOUR RIGHT" is the
        # standing hazard in this codebase.
        lanes = [(i, tag) for i, (_side, _ring, tag) in enumerate(data.SIDES)][::-1]
        # "now" is the newest CAPTURE, not the wall clock: the right edge should be
        # the last thing the camera saw, so a stalled camera freezes the plot
        # instead of scrolling an ever-widening blank off to the right.
        strip = ui.timeline(top.shape[1], self.history,
                            self.history[-1][0] if self.history else v.stamp,
                            self.args.thresh, HISTORY_SECONDS, lanes,
                            captures=pipe.capture.stamps if pipe else None)
        return np.vstack([top, band, strip])


def main():
    ap = argparse.ArgumentParser(
        description="blinkergate live view: detection and labelled capture.")
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--models", default="models", help="model generation directory")
    ap.add_argument("--scale", type=float, default=2.0, help="UI scale")
    ap.add_argument("--thresh", type=float, default=0.5,
                    help="P(open) below this counts as shut")
    ap.add_argument("--subtract", action="store_true",
                    help="subtract the unlit strobe frame to cancel ambient IR")
    ap.add_argument("--workers", type=int, default=1,
                    help="processing threads. More than one overlaps MediaPipe's "
                         "inference but completes out of order")
    ap.add_argument("--queue", type=int, default=2,
                    help="frames buffered between capture and processing (1 = "
                         "always the freshest, never one in hand)")
    args = ap.parse_args()
    ui.S = args.scale

    app = App(args)
    try:
        cam = IRCamera(args.device, subtract=args.subtract)
    except RuntimeError as exc:
        sys.exit(str(exc))

    counts = app.dataset.counts()
    print(f"strobe cutoff {cam.lit_cutoff:.1f}   dataset {sum(counts.values())} samples"
          + ("   ambient subtraction ON" if args.subtract else ""))
    print(f"{app.status}\nlabel and manage generations at:  uv run serve.py\n")

    # Capture runs in its own thread so waiting for the camera and processing a
    # frame stop being the same wall-clock time. Serial, a pass over 133ms missed
    # the next lit frame entirely and the following usable one was 266ms later;
    # the queue absorbs that, so the verdict interval is the processing time
    # instead of the next multiple of 133ms.
    pipe = Pipeline(cam, app.make_processor, workers=args.workers,
                    frame_depth=args.queue).start()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, int(WIN_W * ui.S), int(WIN_H * ui.S))
    try:
        loop(app, pipe)
    except KeyboardInterrupt:
        # A stack trace for a deliberate Ctrl-C is noise. The newline is so the
        # shell prompt does not land on top of the "^C".
        print("\ninterrupted")
    finally:
        # Threads first, then the device: releasing the camera under a capture loop
        # still reading from it is how you get a segfault on the way out. In a
        # finally so that Ctrl-C, a closed window and a crash all clean up alike.
        pipe.stop()
        for name, err in pipe.errors():
            print(f"  {name} thread ended with: {err!r}")
        cam.release()
        cv2.destroyAllWindows()


def window_gone():
    """Has the user closed the window?

    cv2.imshow *recreates* a window that has been closed, so without this the
    window reappeared and the app ran on forever. Some builds raise instead of
    reporting a missing window, which counts the same.
    """
    try:
        return cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def loop(app, pipe):
    fullscreen = False
    latest = None
    # Only a window that has *been* visible can have been closed. Backends differ
    # on what they report between namedWindow and the first imshow, and reading
    # that as a close would have quit before drawing anything.
    was_visible = False

    while True:
        item = pipe.results.get(0.05)
        if item is not None:
            latest = item[1]
            app.absorb(latest)
            app.poll()
            cv2.imshow(WIN, app.render(latest, pipe))
        elif latest is None:
            # Nothing has been processed yet. Say why rather than showing a blank
            # window: a pipeline that died looks exactly like one still starting.
            for name, err in pipe.errors():
                sys.exit(f"{name} thread died: {err!r}")

        key = cv2.waitKey(1) & 0xFF
        if window_gone():
            if was_visible:
                return
        else:
            was_visible = True

        if key == ord("q"):
            return
        elif key == ord("t"):
            app.status = "training ..."
            app.train("pixel")
        elif key == ord("p"):
            app.status = "training pooled ..."
            app.train("pooled")
        elif key == ord("g"):
            app.cycle_generation()
        elif key == ord("b"):
            app.cycle_blur()
        elif latest is None:
            pass            # the keys below all act on a frame; there is none yet
        elif key == ord("h"):
            app.capture_hard(latest.gray, latest.det)
        elif key in CAPTURES:
            app.capture(latest.gray, latest.det, *CAPTURES[key])
        elif key in (ord("+"), ord("=")):
            ui.S = min(ui.S + 0.25, 4.0)
            cv2.resizeWindow(WIN, int(WIN_W * ui.S), int(WIN_H * ui.S))
        elif key == ord("-"):
            ui.S = max(ui.S - 0.25, 0.75)
            cv2.resizeWindow(WIN, int(WIN_W * ui.S), int(WIN_H * ui.S))
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else
                                  cv2.WINDOW_NORMAL)


if __name__ == "__main__":
    main()
