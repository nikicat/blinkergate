"""Build a directory of PGM frames that spells out a gesture.

The end-to-end test runs in a virtual machine, which has no ``/dev/video*``, and a
gesture is the one thing a synthetic frame cannot contain. So the frames come from
the labelled dataset: real frames of a real face, each with a known per-eye label,
put in an order that spells the pattern out.

**This is a plumbing test, not an accuracy measurement.** The frames are ones the
active generation was fitted on, so the classifier has seen them - see
``docs/PITFALLS.md``, "Testing on the training set". What it proves is that the
camera source, the mesh, the crop, the classifier, the matcher and the socket agree
end to end, which is exactly what nothing else checks. It says nothing about whether
the gesture works on frames from tomorrow.

    uv run record_replay.py --pattern l,r --out /tmp/blinkergate-replay

A recording of a gesture actually performed is strictly better and can replace this
directory without any other change: any directory of same-sized PGMs will do.
"""

import argparse
import os
import shutil

import cv2

from bgate import data

# Seconds between frames, written into the recording so the daemon paces it the way it
# was built rather than at some other camera's rate. Measured on the reference camera:
# 15 fps delivered with the emitter strobing, so 133ms between *lit* frames. A recording
# made on other hardware needs its own measurement here - the number decides whether six
# shut frames are a held closure or a blink.
INTERVAL = 0.133

# How many frames each state is held for. A wink scores the moment the eye shuts, so
# one frame would do; two survives a dropped frame, and at 133ms a frame the whole
# sequence still lands well inside the 6s window.
HOLD = 2


def by_state(ds):
    """Frames grouped by what both eyes are doing, as the matcher would read them."""
    labels = {}
    for row in ds.rows:
        if row["label"] in data.TRAINABLE:
            labels.setdefault(row["frame"], {})[row["side"]] = row["label"]
    states = {"o": [], "l": [], "r": [], "b": []}
    for frame, sides in sorted(labels.items()):
        left, right = sides.get("img_left"), sides.get("img_right")
        if left is None or right is None:
            continue        # one eye labelled is not a state
        shut_l, shut_r = left == data.CLOSED, right == data.CLOSED
        state = "b" if shut_l and shut_r else "l" if shut_l else "r" if shut_r else "o"
        states[state].append(frame)
    return states


def sequence(pattern):
    """The states to replay, with eyes open either side of every event.

    Open frames between events are not decoration: the matcher requires every event
    to start and end at open eyes, which is what stops a slide from one wink to the
    other counting as two.
    """
    out = ["o"]
    for symbol in pattern:
        if symbol in ("l", "r"):
            out += [symbol, "o"]
        elif symbol == "short":
            out += ["b", "o"]
        elif symbol == "long":
            # Long enough to pass long_min at 133ms a frame, with margin.
            out += ["b"] * 6 + ["o"]
        else:
            raise SystemExit(f"unknown symbol {symbol!r}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pattern", default="l,r", help="comma separated, l/r/long/short")
    ap.add_argument("--out", default="/tmp/blinkergate-replay")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--hold", type=int, default=HOLD)
    ap.add_argument("--interval", type=float, default=INTERVAL,
                    help="seconds between frames, as the daemon should replay them")
    args = ap.parse_args()

    ds = data.Dataset(args.dataset)
    states = by_state(ds)
    print("labelled frames per state:",
          {k: len(v) for k, v in states.items()})

    wanted = sequence(args.pattern.split(","))
    missing = [s for s in set(wanted) if not states[s]]
    if missing:
        raise SystemExit(
            f"the dataset has no frames for {missing} - capture some with "
            "eyetrainer.py before building a replay for this pattern")

    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out)

    # Different frames for repeats of the same state where possible: reusing one
    # frame would make a held closure look like a frozen camera, which is a thing
    # the pipeline may one day want to reject.
    picked, used = [], {s: 0 for s in states}
    for state in wanted:
        for _ in range(args.hold):
            pool = states[state]
            picked.append((state, pool[used[state] % len(pool)]))
            used[state] += 1

    for i, (state, frame) in enumerate(picked):
        gray = cv2.imread(os.path.join(data.frames_dir(args.dataset), frame),
                          cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise SystemExit(f"{frame}: unreadable")
        # PGM, because it is the simplest real image format and the daemon reads it
        # without a decoder.
        cv2.imwrite(os.path.join(args.out, f"{i:04d}_{state}.pgm"), gray)

    # The manifest travels with the frames: a recording that does not say how fast it
    # was made cannot be replayed honestly, and the daemon refuses one that is missing.
    spelled = "".join(s for s, _ in picked)
    with open(os.path.join(args.out, "replay.toml"), "w") as fh:
        fh.write(f"""# Written by record_replay.py - do not pace these frames any other way.
interval = {args.interval}

# Provenance, for a human. Nothing reads these.
spells = "{args.pattern}"
states = "{spelled}"
frames = {len(picked)}
source = "labelled dataset frames, composed - not a performed gesture"
""")

    print(f"{args.out}: {len(picked)} frames, {spelled}")
    print(f"pattern {args.pattern} at {args.interval * 1000:.0f}ms a frame is "
          f"{len(picked) * args.interval:.1f}s of replay")


if __name__ == "__main__":
    main()
