"""Deliberate wink sequences, matched in the live verdict stream.

The point of the project: a sequence like l,r,r,l - YOUR left, right, right,
left, one wink each - as a consent signal. This module only decides whether
the stream contains the pattern; what to do about a match is the caller's.

Strictness is the point - a consent signal must be hard to produce by
accident:

- a wink counts only from open eyes: open -> exactly one eye shut. Coming
  from both-shut (opening the other eye) is not a wink.
- both eyes shut resets the attempt. A natural blink is both-eyed, and a
  pattern of single-eye winks contains none, so a 'b' is either an accident
  or not a gesture.
- the whole sequence must land within ``window`` seconds of its first wink.
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


class GestureMatcher:
    def __init__(self, pattern, thresh=0.5, window=6.0):
        if not pattern or any(s not in ("l", "r") for s in pattern):
            raise ValueError(f"pattern must be l/r symbols, got {pattern!r}")
        self.pattern = list(pattern)
        self.thresh = thresh          # P(open) below this counts as shut
        self.window = window          # seconds from first wink to full match
        self.sym = None               # the current run's symbol
        self.progress = 0             # winks matched so far
        self.started = 0.0            # stamp of the attempt's first wink
        self.matched_at = None        # stamp of the last full match

    def feed(self, stamp, probs):
        """One verdict. True exactly when this frame completes the pattern."""
        sym = self._classify(probs)
        if sym is None or sym == self.sym:
            return False              # lost face, or a run continuing
        prev, self.sym = self.sym, sym
        if self.progress and stamp - self.started > self.window:
            self.progress = 0
        if sym == "b":
            self.progress = 0
            return False
        if sym == "o" or prev != "o":
            return False
        if sym == self.pattern[self.progress]:
            if self.progress == 0:
                self.started = stamp
            self.progress += 1
        else:
            # ponytail: a wrong wink restarts with itself rather than a KMP
            # failure table - patterns are a few symbols, and only one with an
            # overlapping prefix (l,l,r) would ever lose a step to this.
            self.progress = 1 if sym == self.pattern[0] else 0
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
    print("gesture.py self-check ok")
