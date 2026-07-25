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
  t  train a new generation from the current dataset
  g  cycle generation: follow-the-web-UI, then each one in turn
  b  override the blur (diagnostic - see below)
  q  quit               + -  resize              f  fullscreen

Blur is a property of a model, not a switch: one fitted on blurred features is
meaningless applied to sharp ones. So the honest blur A/B is `g`, cycling between
generations that were *trained* differently - the header shows each one's blur.
`b` forces a different blur onto the current model and is labelled MISMATCH when
it disagrees with what that model was trained with; it is there to show what the
mismatch looks like, not to be run in.

Sides are YOURS. The feed is not mirrored, so your left eye appears on the RIGHT
of the picture - the insets are laid out to match the picture, so the one on the
right of the strip is the one on the right of the frame.
"""

import argparse
import sys
import time

from bgate import qtfix  # noqa: F401  - must be imported before cv2

import cv2
import numpy as np

from bgate import data, ui
from bgate.camera import IRCamera, DEFAULT_DEVICE
from bgate.data import Dataset
from bgate.detect import MediaPipeLocator, RollTracker
from bgate.model import fit, make_landmarker
from bgate.registry import Registry

WIN = "blinkergate"
POLL_EVERY = 1.0              # seconds between checks for a generation switch
CAPTURES = {
    ord("o"): ("both OPEN", data.OPEN, data.OPEN),
    ord("c"): ("both CLOSED", data.CLOSED, data.CLOSED),
    ord("1"): ("YOUR LEFT shut", data.CLOSED, data.OPEN),
    ord("2"): ("YOUR RIGHT shut", data.OPEN, data.CLOSED),
}


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
        self.pinned = False          # generation chosen here, not in the web UI
        self.blur_override = None    # None = whatever the model was trained with
        self.gens = self.registry.ids()
        self.load_model()

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
        """Push the override, if any, onto the loaded model."""
        if self.model is None:
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
            print(f"  pinned {chosen}  blur={self.model.blur}"
                  f"  ({self.model.accuracy:.1%})")

    def cycle_blur(self):
        """Force a blur onto the current model. Diagnostic only."""
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

    def train(self):
        # The web UI may have relabelled since we loaded; train on what is there.
        if self.dataset.reload_if_changed():
            print("  picked up label changes from the web UI")
        self.dataset.refresh_crops()
        samples = self.dataset.trainable()
        counts = self.dataset.counts()
        if min(counts[data.OPEN], counts[data.CLOSED]) < 10:
            self.status = f"need 10+ of each: {counts[data.OPEN]}/{counts[data.CLOSED]}"
            return
        try:
            coef, intercept, accuracy = fit(samples)
        except Exception as exc:
            self.status = f"training failed: {exc}"
            return
        gen = self.registry.create(coef, intercept, accuracy, samples,
                                   self.dataset.rows)
        self.registry.set_active(gen)
        self.trained_blur = None
        self.blur_override = None
        self.load_model()
        self.status = f"{gen}: {len(samples)} samples, held-out {accuracy:.1%}"
        print(f"  {self.status}")

    def model_line(self):
        if self.model is None:
            return self.status
        mismatch = (self.blur_override is not None
                    and self.blur_override != self.trained_blur)
        text = (f"{self.model_gen}  blur {self.model.blur}"
                f"  norm {self.model.normalise}"
                f"  ({self.model.accuracy:.1%} held-out)")
        text += "  [pinned]" if self.pinned else "  [following web UI]"
        if len(self.gens) > 1:
            text += f"  ({len(self.gens)} generations)"
        if mismatch:
            text += f"  MISMATCH: trained with {self.trained_blur}"
        return text

    def model_colour(self):
        if (self.blur_override is not None
                and self.blur_override != self.trained_blur):
            return (0, 0, 255)      # a mismatched blur is not a valid reading
        return (0, 200, 255)

    def render(self, gray, det):
        raw = cv2.cvtColor(ui.enlarge(gray), cv2.COLOR_GRAY2BGR)
        annot = raw.copy()
        state = "no face"
        blank = np.zeros((int(ui.INSET_H * ui.S), int(ui.INSET_W * ui.S), 3), np.uint8)
        insets = [blank, blank]

        if det and self.model:
            lm = det.landmarks
            probs, _crops, boxes = self.model.eyes(gray, lm, roll=det.roll)
            shut = [p is not None and p < self.args.thresh for p in probs]
            for box, is_shut in zip(boxes, shut):
                x0, y0, x1, y1 = [int(v * ui.S) for v in box]
                cv2.rectangle(annot, (x0, y0), (x1, y1),
                              (0, 0, 255) if is_shut else (0, 255, 0),
                              max(1, int(ui.S * 0.6)))
            insets = [
                ui.eye_inset(gray, lm, ring,
                             ui.IRIS_L if side == "img_left" else ui.IRIS_R,
                             tag, p if p is not None else 1.0, sh)
                for (side, ring, tag), p, sh in zip(data.SIDES, probs, shut)
            ]
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
            (f"dataset: {counts[data.OPEN]} open / {counts[data.CLOSED]} closed / "
             f"{counts[data.SKIP]} skip    roll {self.roll:+.0f}deg",
             0.45, (200, 200, 120)),
            (self.model_line(), 0.42, self.model_colour()),
            ("o/c/1/2 capture   h hard case   t train   g gen   q quit",
             0.38, (150, 150, 150)),
        ])
        return np.vstack([top, band])


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
    args = ap.parse_args()
    ui.S = args.scale

    try:
        # Roll tracking: face detectors want upright faces, and cancelling the
        # tracked head roll took detection from 67% to 93% on the frames that
        # were failing. The angle is tracked, so it is usually one detect.
        locator = RollTracker(MediaPipeLocator(make_landmarker()))
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    app = App(args)
    try:
        cam = IRCamera(args.device, subtract=args.subtract)
    except RuntimeError as exc:
        sys.exit(str(exc))

    counts = app.dataset.counts()
    print(f"strobe cutoff {cam.lit_cutoff:.1f}   dataset {sum(counts.values())} samples"
          + ("   ambient subtraction ON" if args.subtract else ""))
    print(f"{app.status}\nlabel and manage generations at:  uv run serve.py\n")

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, int(1280 * ui.S), int(600 * ui.S))
    fullscreen = False

    while True:
        gray = cam.read()
        if gray is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        app.poll()
        det = locator.locate(gray)
        app.roll = locator.angle if det else app.roll

        cv2.imshow(WIN, app.render(gray, det))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            app.status = "training ..."
            app.train()
        elif key == ord("g"):
            app.cycle_generation()
        elif key == ord("b"):
            app.cycle_blur()
        elif key == ord("h"):
            app.capture_hard(gray, det)
        elif key in CAPTURES:
            app.capture(gray, det, *CAPTURES[key])
        elif key in (ord("+"), ord("=")):
            ui.S = min(ui.S + 0.25, 4.0)
            cv2.resizeWindow(WIN, int(1280 * ui.S), int(600 * ui.S))
        elif key == ord("-"):
            ui.S = max(ui.S - 0.25, 0.75)
            cv2.resizeWindow(WIN, int(1280 * ui.S), int(600 * ui.S))
        elif key == ord("f"):
            fullscreen = not fullscreen
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else
                                  cv2.WINDOW_NORMAL)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
