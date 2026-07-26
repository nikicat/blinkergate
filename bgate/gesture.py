"""Deliberate eye gestures, matched in the live verdict stream.

The point of the project: a sequence like l,r,r,l (winks) or
long,long,short (both-eye closures) as a consent signal. This module only
decides whether the stream contains the pattern; what to do about a match
is the caller's.

Four symbols. ``l`` and ``r`` are winks - YOUR left, YOUR right - scored the
moment the eye shuts. ``long`` and ``short`` are both-eye closures, scored
the moment the eyes REopen, because duration is what tells them apart:
``long_min`` seconds and over is long. The two kinds compose in one pattern.

Strictness is the point - a consent signal must be hard to produce by
accident:

- every event starts and ends at open eyes. A wink is open -> exactly one
  shut; a closure is open -> both shut -> open. Leaving a closure one eye
  at a time (or sliding l->r without opening) is not an event, it aborts
  the attempt.
- an event the pattern did not expect resets it, so for a winks-only
  pattern a natural blink - a short both-eye closure - still resets, as it
  always did. A pattern that USES ``short`` accepts that a natural blink is
  indistinguishable from it; the surrounding sequence is the protection.
- the whole sequence must land within ``window`` seconds of its first event.
- a frame where nothing was classified is neutral: it neither completes nor
  splits a run of frames. The window still bounds a long loss of the face.

Holding a wink is one wink, however many frames it spans - a run of the same
symbol advances the pattern once, so l,r,r,l needs the eyes OPEN between the
two rights.

Needs verdicts in capture order: more than one pipeline worker completes out
of order and would shuffle the events (see pipeline.py, which leaves one
worker as the default for exactly this).

Sides are YOURS and index 0 is your left, as in data.SIDES - the standing
left/right hazard of this codebase.
"""

SYMBOLS = ("l", "r", "long", "short")


class GestureMatcher:
    def __init__(self, pattern, thresh=0.5, window=6.0, long_min=0.5):
        if not pattern or any(s not in SYMBOLS for s in pattern):
            raise ValueError(f"pattern must be {'/'.join(SYMBOLS)} symbols, "
                             f"got {pattern!r}")
        self.pattern = list(pattern)
        self.thresh = thresh          # P(open) below this counts as shut
        self.window = window          # seconds from first event to full match
        self.long_min = long_min      # a closure this long or over is 'long'
        self.sym = None               # the current run's symbol
        self.shut_at = None           # when the current both-eye closure began
        self.progress = 0             # events matched so far
        self.started = 0.0            # stamp of the attempt's first event
        self.matched_at = None        # stamp of the last full match

    def feed(self, stamp, probs):
        """One verdict. True exactly when this frame completes the pattern."""
        sym = self._classify(probs)
        if sym is None or sym == self.sym:
            return False              # lost face, or a run continuing
        prev, self.sym = self.sym, sym
        if self.progress and stamp - self.started > self.window:
            self.progress = 0

        if sym == "o":
            if prev != "b" or self.shut_at is None:
                return False          # opening after a wink: already scored
            # A closure is classified by duration at the moment it opens.
            event = ("long" if stamp - self.shut_at >= self.long_min
                     else "short")
            self.shut_at = None
        elif prev != "o":
            # l->r, l->b, b->l ...: the eyes never fully opened between
            # states. Not a clean event - strict means it aborts the attempt.
            self.progress = 0
            self.shut_at = None
            return False
        elif sym == "b":
            self.shut_at = stamp      # closure begins; classified when it ends
            return False
        else:
            event = sym               # a wink scores on entry: no duration

        if event == self.pattern[self.progress]:
            if self.progress == 0:
                self.started = stamp
            self.progress += 1
        else:
            # ponytail: a wrong event restarts with itself rather than a KMP
            # failure table - patterns are a few symbols, and only one with an
            # overlapping prefix (l,l,r) would ever lose a step to this.
            self.progress = 1 if event == self.pattern[0] else 0
            self.started = stamp
        if self.progress == len(self.pattern):
            self.progress = 0
            self.matched_at = stamp
            return True
        return False

    def _classify(self, probs):
        """'l'/'r' = that eye shut alone, 'b' both, 'o' open, None = no read."""
        if not probs:
            return None
        left, right = probs
        if left is None or right is None:
            return None
        shut_l, shut_r = left < self.thresh, right < self.thresh
        return ("b" if shut_l and shut_r else "l" if shut_l else
                "r" if shut_r else "o")


def _run(pattern, symbols, dt=0.133, window=6.0):
    """Indices at which feeding the symbol stream completes the pattern."""
    P = {"l": (0.1, 0.9), "r": (0.9, 0.1), "b": (0.1, 0.1),
         "o": (0.9, 0.9), "?": None}
    m = GestureMatcher(list(pattern), window=window)
    return [i for i, s in enumerate(symbols.split()) if m.feed(i * dt, P[s])]


if __name__ == "__main__":
    assert _run("lrrl", "o l o r o r o l") == [7]
    assert _run("lrrl", "o l l l o r o r o l") == [9]      # a held wink is one
    assert _run("lrrl", "o l o r o b o r o l") == []       # a blink resets
    assert _run("lrrl", "b l o l o r o r o l") == [9]      # b->l is not a wink
    assert _run("lrrl", "o l ? l o r o r o l") == [9]      # lost face: neutral
    assert _run("lr", "o r o l o r") == [5]                # wrong wink restarts
    assert _run("lr", "o l o r", dt=4.0) == []             # over the window
    assert _run("lrrl", "o l o r o r o l o l o r o r o l") == [7, 15]

    # Both-eye closures: 4 shut frames = 0.532s >= long_min, 1 = 0.133s short.
    LLS = ["long", "long", "short"]
    assert _run(LLS, "o b b b b o b b b b o b o") == [12]
    assert _run(LLS, "o b b b b o b o b b b b o") == []    # short in the middle
    assert _run(["short"], "o b l o b o") == [5]           # b->l aborts, clean
    assert _run(["l", "short"], "o l o b o") == [4]        # kinds compose
    assert _run(["short", "l"], "o b o l") == [3]          # blink then wink
    try:
        GestureMatcher(["l", "x"])
        raise SystemExit("accepted a bad symbol")
    except ValueError:
        pass
    print("gesture.py self-check ok")
