#!/usr/bin/env python
"""blinkergate live view - watch detection and capture labelled samples.

Labelling and model history live in the web UI (``uv run serve.py``). This window
does the two things that need the camera: showing detection in real time, and
recording labelled bursts.

The two share state through ``models/active``. Switch generation in the browser
and this picks it up within a second, without a restart.

  o  both open          c  both closed
  1  IMG-LEFT shut      2  IMG-RIGHT shut        (2s burst per press)
  t  train a new generation from the current dataset
  q  quit               + -  resize              f  fullscreen

Sides are sides of the IMAGE. The feed is not mirrored, so your left eye appears
on the image's right.
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
from bgate.model import fit, make_landmarker
from bgate.registry import Registry

WIN = "blinkergate"
HOLD = 2.0                    # seconds of frames recorded per capture press
POLL_EVERY = 1.0              # seconds between checks for a generation switch
CAPTURES = {
    ord("o"): ("both OPEN", data.OPEN, data.OPEN),
    ord("c"): ("both CLOSED", data.CLOSED, data.CLOSED),
    ord("1"): ("IMG-LEFT shut", data.CLOSED, data.OPEN),
    ord("2"): ("IMG-RIGHT shut", data.OPEN, data.CLOSED),
}


class App:
    def __init__(self, args):
        self.args = args
        self.dataset = Dataset(args.dataset)
        self.registry = Registry(args.models)
        self.model = None
        self.model_gen = None
        self.active_seen = -1.0
        self.last_poll = 0.0
        self.capture = None
        self.status = ""
        self.load_model()

    def load_model(self):
        gen, model = self.registry.load_active()
        self.model, self.model_gen = model, gen
        self.active_seen = self.registry.active_mtime()
        self.status = (f"{gen} active ({model.accuracy:.1%} held-out)" if model
                       else "no model - capture samples, then press t")

    def poll_active(self):
        """The web UI switches generations by writing models/active; noticing it
        by mtime avoids needing any channel between the two processes."""
        now = time.time()
        if now - self.last_poll < POLL_EVERY:
            return
        self.last_poll = now
        if self.registry.active_mtime() != self.active_seen:
            self.load_model()
            print(f"  switched to {self.model_gen}")

    def train(self):
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
        self.load_model()
        self.status = f"{gen}: {len(samples)} samples, held-out {accuracy:.1%}"
        print(f"  {self.status}")

    def render(self, gray, res):
        raw = cv2.cvtColor(ui.enlarge(gray), cv2.COLOR_GRAY2BGR)
        annot = raw.copy()
        state = "no face"
        blank = np.zeros((int(ui.INSET_H * ui.S), int(ui.INSET_W * ui.S), 3), np.uint8)
        insets = [blank, blank]

        if res.face_landmarks and self.model:
            lm = res.face_landmarks[0]
            probs, _crops, boxes = self.model.eyes(gray, lm)
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
                     "IMG-LEFT shut" if shut[0] else
                     "IMG-RIGHT shut" if shut[1] else "open")
        elif not self.model:
            state = "no model - capture samples and press t"

        ui.txt(annot, "DETECTION", ui.P(8, 22), 0.5, (0, 255, 0), 1.2)
        top = np.hstack([raw, annot])

        banner = state
        if self.capture:
            banner = (f"RECORDING {self.capture['name']}  "
                      f"{HOLD - (time.time() - self.capture['t0']):.1f}s")
        counts = self.dataset.counts()
        band = ui.inset_row(insets[0], insets[1], top.shape[1], [
            (banner, 0.85,
             (0, 220, 255) if self.capture else
             (0, 165, 255) if state not in ("open", "no face") else (150, 150, 150)),
            (f"dataset: {counts[data.OPEN]} open / {counts[data.CLOSED]} closed / "
             f"{counts[data.SKIP]} skip", 0.45, (200, 200, 120)),
            (self.status, 0.42, (0, 200, 255)),
            ("o open   c closed   1 img-left   2 img-right   t train   q quit",
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
    args = ap.parse_args()
    ui.S = args.scale

    try:
        landmarker = make_landmarker()
    except FileNotFoundError as exc:
        sys.exit(str(exc))

    app = App(args)
    try:
        cam = IRCamera(args.device)
    except RuntimeError as exc:
        sys.exit(str(exc))

    counts = app.dataset.counts()
    print(f"strobe cutoff {cam.lit_cutoff:.1f}   dataset {sum(counts.values())} samples")
    print(f"{app.status}\nlabel and manage generations at:  uv run serve.py\n")

    import mediapipe as mp
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, int(1280 * ui.S), int(600 * ui.S))
    fullscreen = False

    while True:
        gray = cam.read()
        if gray is None:
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        app.poll_active()
        res = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)))

        if app.capture:
            since = time.time() - app.capture["t0"]
            if since < HOLD and res.face_landmarks:
                app.dataset.add_capture(gray, app.capture["l"], app.capture["r"],
                                        int(app.capture["t0"] * 1000),
                                        app.capture["n"])
                app.capture["n"] += 1
            elif since >= HOLD:
                app.status = f"captured {app.capture['n']} as {app.capture['name']}"
                print(f"  {app.status}")
                app.capture = None
                app.dataset.refresh_crops(landmarker)

        cv2.imshow(WIN, app.render(gray, res))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("t"):
            app.status = "training ..."
            app.train()
        elif key in CAPTURES and not app.capture:
            name, left, right = CAPTURES[key]
            app.capture = {"name": name, "l": left, "r": right,
                           "t0": time.time(), "n": 0}
            print(f"  recording {name} for {HOLD:.0f}s ...")
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
