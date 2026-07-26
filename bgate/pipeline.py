"""Concurrent capture and processing, with every discard counted.

Serial capture-then-process pays for its own latency twice. Lit frames arrive
every ~133ms; if one pass takes longer than that, the next lit frame goes by
unread, the strobe hands us an unlit frame to discard, and the following usable
one is 266ms after the last. **The cost is quantised** - a millisecond over budget
costs 133ms - which is why the measured frame time sat at 260ms rather than
anywhere in between.

Splitting the two removes the quantisation. The capture thread drains the camera
at its own cadence and never waits for us. A worker takes a queued frame and
processes back to back, so the interval between verdicts is the processing time
itself rather than the next multiple of 133ms. Measured processing is ~50-90ms
typical and ~210ms worst case, so this is the difference between 3.8 verdicts a
second and 7.5.

Nothing here makes processing faster. It makes falling behind cost what it
actually costs, and it makes the frames we drop *countable* rather than leaving
them to show up as a frame rate nobody can account for.
"""

import threading
import time
from collections import deque


class DropQueue:
    """Bounded queue that discards the OLDEST item to make room.

    On a live feed a stale frame is worthless, so when the consumer falls behind
    the thing to throw away is the thing nobody wants any more.

    Every drop is counted and timestamped. A queue that silently discards is
    indistinguishable from a camera that never delivered, and that ambiguity is
    precisely what made the frame rate unexplainable before.

    ``maxsize=1`` is latest-only: the worker always gets the freshest frame at the
    cost of never having one buffered. Larger absorbs a slow pass without dropping,
    at the cost of handing the worker something slightly stale.
    """

    def __init__(self, name, maxsize=2):
        self.name = name
        self.maxsize = max(1, int(maxsize))
        self._items = deque()
        self._ready = threading.Condition()
        self.put_count = 0
        self.drop_count = 0
        self.get_count = 0
        self.dropped_stamps = deque(maxlen=512)

    def put(self, stamp, payload):
        """Enqueue, dropping from the head if full. True if anything was dropped."""
        with self._ready:
            dropped = False
            while len(self._items) >= self.maxsize:
                old_stamp, _old = self._items.popleft()
                self.drop_count += 1
                self.dropped_stamps.append(old_stamp)
                dropped = True
            self._items.append((stamp, payload))
            self.put_count += 1
            self._ready.notify()
            return dropped

    def get(self, timeout=0.2):
        """``(stamp, payload)``, or None if nothing arrived within ``timeout``."""
        with self._ready:
            if not self._items:
                self._ready.wait(timeout)
            if not self._items:
                return None
            self.get_count += 1
            return self._items.popleft()

    @property
    def depth(self):
        return len(self._items)

    def stats(self):
        return dict(name=self.name, depth=self.depth, maxsize=self.maxsize,
                    put=self.put_count, got=self.get_count,
                    dropped=self.drop_count)


class _Loop(threading.Thread):
    """A daemon thread that records why it stopped.

    A worker that dies on an exception and says nothing looks exactly like a
    camera delivering nothing, which is the confusion this module exists to end -
    so the error is kept and the owner can surface it.
    """

    def __init__(self, name):
        super().__init__(name=name, daemon=True)
        self._stop = threading.Event()
        self.error = None

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self.work()
        except BaseException as exc:            # noqa: BLE001 - reported, not swallowed
            self.error = exc

    def work(self):
        raise NotImplementedError


class CaptureLoop(_Loop):
    """Drains the camera and enqueues lit frames, stamped when they arrived.

    Runs at the sensor's own cadence - the unlit half of the strobe is discarded
    here, cheaply, and never reaches a worker.
    """

    def __init__(self, camera, out, name="capture"):
        super().__init__(name)
        self.camera = camera
        self.out = out
        self.paused = False                 # set from the UI thread; a plain bool
        self.stamps = deque(maxlen=512)     # every lit frame we captured

    def work(self):
        while not self._stop.is_set():
            gray = self.camera.read()
            if gray is None:                # unlit, or a failed read; the camera
                continue                    # counts those two separately
            if self.paused:
                # Still reading: the drain keeps the driver's buffer fresh so
                # resume shows now, not a frame from when the pause began. But
                # nothing is enqueued, so the workers sit idle instead of
                # spending inference on frames nobody will look at.
                continue
            stamp = time.monotonic()
            self.stamps.append(stamp)
            self.out.put(stamp, gray)


class WorkerLoop(_Loop):
    """Pulls a frame, processes it, enqueues the result.

    ``make_process`` is called **inside this thread** and must return a
    ``process(stamp, gray)`` callable. Each worker therefore builds its own
    detector: a MediaPipe landmarker is not documented as thread-safe, and sharing
    one across threads is the kind of thing that works until it does not.
    """

    def __init__(self, source, sink, make_process, name="worker"):
        super().__init__(name)
        self.source = source
        self.sink = sink
        self.make_process = make_process
        self.busy_ms = deque(maxlen=256)
        self.done = 0

    def work(self):
        process = self.make_process()
        while not self._stop.is_set():
            item = self.source.get(0.2)
            if item is None:
                continue
            stamp, gray = item
            t0 = time.perf_counter()
            result = process(stamp, gray)
            self.busy_ms.append((time.perf_counter() - t0) * 1000.0)
            self.done += 1
            if result is not None:
                self.sink.put(stamp, result)


class Pipeline:
    """Camera -> frames queue -> worker(s) -> results queue.

    The GUI stays on the main thread, which is where OpenCV wants it, and which is
    also why rendering is not a stage here: it consumes ``results`` at whatever
    rate it can and its own slowness costs only display smoothness.

    More than one worker will overlap MediaPipe's C++ inference, which releases the
    GIL - but results then complete out of order. The timeline is plotted against
    capture time so it tolerates that; a gesture decision would not. Left at one
    worker by default for that reason.
    """

    def __init__(self, camera, make_process, workers=1, frame_depth=2,
                 result_depth=2):
        self.frames = DropQueue("frames", frame_depth)
        self.results = DropQueue("results", result_depth)
        self.capture = CaptureLoop(camera, self.frames)
        self.workers = [WorkerLoop(self.frames, self.results, make_process,
                                   name=f"worker{i}")
                        for i in range(max(1, int(workers)))]

    @property
    def threads(self):
        return [self.capture, *self.workers]

    def start(self):
        for t in self.threads:
            t.start()
        return self

    def stop(self, timeout=1.0):
        for t in self.threads:
            t.stop()
        for t in self.threads:
            t.join(timeout)

    def errors(self):
        """Whatever killed a thread, so a dead pipeline cannot look like a quiet
        one."""
        return [(t.name, t.error) for t in self.threads if t.error is not None]

    def dead(self):
        return [t.name for t in self.threads if not t.is_alive()]

    def rate(self, stamps, window=4.0):
        """Events per second over the last ``window`` seconds of ``stamps``."""
        if len(stamps) < 2:
            return 0.0
        now = stamps[-1]
        recent = [s for s in stamps if s >= now - window]
        if len(recent) < 2:
            return 0.0
        return (len(recent) - 1) / max(recent[-1] - recent[0], 1e-6)

    def summary(self):
        """One line of pipeline health, for the live view."""
        cam = getattr(self.capture, "camera", None)
        busy = [ms for w in self.workers for ms in w.busy_ms]
        busy_med = sorted(busy)[len(busy) // 2] if busy else 0.0
        parts = [f"capture {self.rate(self.capture.stamps):.1f}/s"]
        if cam is not None:
            parts.append(f"lit {cam.lit_count} unlit {cam.unlit_count}"
                         + (f" failed {cam.fail_count}" if cam.fail_count else ""))
        parts.append(f"work {busy_med:.0f}ms")
        parts.append(f"drop {self.frames.drop_count}f/{self.results.drop_count}r")
        parts.append(f"q {self.frames.depth}/{self.frames.maxsize}")
        if len(self.workers) > 1:
            parts.append(f"{len(self.workers)} workers")
        return "   ".join(parts)
