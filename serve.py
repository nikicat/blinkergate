#!/usr/bin/env python
"""Start the labelling web UI.

  uv run serve.py            then open http://127.0.0.1:8000

Markup only - run eyetrainer.py alongside it for the live camera view. They
share state through models/active, which the live view polls.
"""

import argparse

import uvicorn

from bgate.server import create_app

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="dataset")
ap.add_argument("--models", default="models")
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8000)
args = ap.parse_args()

print(f"http://{args.host}:{args.port}")
uvicorn.run(create_app(args.dataset, args.models), host=args.host, port=args.port)
