"""Local web UI for labelling the dataset and managing model generations.

Markup only: the live camera view stays in ``eyetrainer.py``. The two surfaces
share state through the filesystem - this writes ``models/active``, and the live
view polls its mtime and hot-reloads.

    uv run serve.py            then open http://127.0.0.1:8000
"""

import base64
import os

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import data
from .data import Dataset
from .model import fit
from .registry import Registry

WEB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web")


class LabelRequest(BaseModel):
    ids: list[str]
    label: str


class State:
    """Dataset and registry held in memory. Labelling mutates this; nothing in
    the request path re-reads the dataset from disk."""

    def __init__(self, ds_dir="dataset", models_dir="models"):
        self.dataset = Dataset(ds_dir)
        self.registry = Registry(models_dir)
        self.model = None
        self.model_gen = None
        self.reload_model()

    def reload_model(self):
        gen, model = self.registry.load_active()
        self.model_gen, self.model = gen, model

    def predictions(self, samples):
        if not self.model or not samples:
            return {}
        p = self.model.p_closed([s["crop"] for s in samples])
        return {s["id"]: float(v) for s, v in zip(samples, p)}


def thumb_b64(crop):
    """Crops go over the wire at their native 40x24 and are scaled by CSS, which
    is set to `image-rendering: pixelated`. Enlarging server-side would multiply
    the payload for pixels the browser can generate itself.

    Inline rather than one endpoint per crop: a few hundred samples then cost one
    request instead of hundreds."""
    ok, buf = cv2.imencode(".png", crop)
    if not ok:
        return ""
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def create_app(ds_dir="dataset", models_dir="models"):
    state = State(ds_dir, models_dir)
    app = FastAPI(title="blinkergate")

    @app.get("/api/samples")
    def samples():
        rows = state.dataset.samples()
        preds = state.predictions(rows)
        return {
            "active_generation": state.model_gen,
            "counts": state.dataset.counts(),
            "samples": [
                {
                    "id": s["id"],
                    "frame": s["frame"],
                    "side": s["side"],
                    "label": s["label"],
                    "burst": s["burst"],
                    "p_closed": preds.get(s["id"]),
                    "thumb": thumb_b64(s["crop"]),
                }
                for s in rows
            ],
        }

    @app.post("/api/label")
    def label(req: LabelRequest):
        if req.label not in data.LABELS:
            raise HTTPException(400, f"label must be one of {data.LABELS}")
        n = state.dataset.set_labels(req.ids, req.label)
        return {"updated": n, "counts": state.dataset.counts()}

    @app.get("/api/models")
    def models():
        # Only open/closed can be scored - a skipped sample has no true class,
        # and including them would report noise as accuracy.
        current = state.dataset.trainable()
        out = []
        for meta in state.registry.list():
            # Scored on samples labelled since this generation was trained, so
            # generations stay comparable on today's labels rather than each on
            # its own snapshot.
            unseen = state.registry.unseen_by(meta["id"], current)
            meta["n_unseen"] = len(unseen)
            meta["unseen_accuracy"] = None
            if unseen:
                try:
                    model = state.registry.load(meta["id"])
                    p = model.p_closed([s["crop"] for s in unseen])
                    truth = np.array([s["label"] == data.CLOSED for s in unseen])
                    meta["unseen_accuracy"] = round(float(((p > 0.5) == truth).mean()), 4)
                except Exception:
                    pass
            out.append(meta)
        return {"generations": out, "active": state.registry.active()}

    @app.post("/api/models")
    def train():
        samples = state.dataset.trainable()
        counts = state.dataset.counts()
        if min(counts[data.OPEN], counts[data.CLOSED]) < 10:
            raise HTTPException(400, f"need at least 10 of each label: {counts}")
        coef, intercept, accuracy = fit(samples)
        gen = state.registry.create(coef, intercept, accuracy, samples,
                                    state.dataset.rows)
        state.registry.set_active(gen)
        state.reload_model()
        return {"generation": gen, "accuracy": None if accuracy != accuracy else accuracy,
                "n_train": len(samples)}

    @app.post("/api/models/{gen}/activate")
    def activate(gen: str):
        try:
            state.registry.set_active(gen)
        except ValueError as exc:
            raise HTTPException(404, str(exc))
        state.reload_model()
        return {"active": gen}

    @app.get("/")
    def index():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app
