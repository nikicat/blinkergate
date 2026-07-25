# Pitfalls

Every entry here is a mistake actually made while building this, caught later by a
measurement or by looking at an image. They are recorded because each one produced
a *confident, plausible, wrong* conclusion — several of which survived for hours
and got built upon.

The common thread: **the measurement was easier to get than the right measurement,
and it agreed with what was expected.**

---

## Measuring on data selected by the thing you are measuring

The worst one, made twice.

**YuNet "detects better than MediaPipe" — 210/210 against 209/210.** Those frames
came from the labelled dataset. Captures are only written *when a face is found*,
so the dataset is by construction a survey of frames where detection already
worked. On frames where it was failing, YuNet scored 44% against MediaPipe's 67%.

The second time, the same trap with the fix applied: after roll compensation, the
hard-frame set was still the one collected against the *pre-roll* detector, so it
over-represented tilted heads — exactly YuNet's weakness. Two batches, opposite
conclusions, neither wrong about its own data.

> Before comparing detectors, ask what selected the frames. If the answer involves
> one of the detectors, the comparison is circular.

This also means **a capture pipeline that only records successes cannot measure
its own failures.** `h` in the live view exists solely to record frames regardless
of detection, because the ordinary path refuses exactly the interesting ones.

## Averaging over a window that contains both states

"Blink detection doesn't separate" — because a 2-second phase where the subject
repeatedly closes and opens their eyes contains mostly *open* frames, and the
phase mean washes the closures out. Per-phase means said 0.239 for both-eye
closure against 0.181 for a left wink, i.e. nonsense. Per-*event* analysis on the
same data showed clean separation.

Made this one, caught it, and then reintroduced it an hour later when scoring the
learned classifier by phase mean.

> Aggregate over the event, not the window that contains it.

## Testing on the training set

Crop-shift sensitivity measured "jitter augmentation helps, 86% → 90% at 4 px".
It was scored on the same samples the model trained on. With whole capture bursts
held out, jitter was **worse** — monotonically, more copies worse than fewer.

The first result was not noise; it was memorisation being rewarded.

> A number computed on training data is not a weak version of the real number. It
> can have the opposite sign.

## Not looking at the images

Several rounds of increasingly elaborate metrics were computed on frames nobody
had looked at. The first contact sheet answered two open questions immediately:
the bright pupil is unmistakable, and the "closed eyes" phases contain plenty of
open eyes because the subject was blinking repeatedly.

Later, a crop montage looked like the eye box was badly misplaced. The measurement
said the box was centred to within 3.7 px — the same as training. The crops merely
*looked* wrong because contrast had collapsed. Eyes and metrics each caught what
the other missed.

> Look at the data early. Then keep measuring, because looking is also wrong
> sometimes.

## Building a new mechanism instead of checking the existing one's configuration

Roll compensation and a second-detector fallback were built, measured and
committed to lift hard-frame detection from 49% to 77%. Then a sweep of nine
detectors found that **BlazeFace at confidence 0.2 — the detector already inside
the bundle — found 100% of the remaining misses on raw pixels.**
`min_face_detection_confidence` had been at its 0.5 default the whole time.
Setting it to 0.2 took the cascade to 100% with no measured false positives.

The two mechanisms still earn their place (74% → 100%), but a one-line default was
doing more damage than either was repairing.

> Exhaust the parameters of what you have before adding something new. Print the
> defaults you never chose.

## An optimisation that corrupts the signal it optimises

`RollTracker` tried its remembered angle first and returned on the first success —
so it rotated by a remembered angle, the mesh fitted the rotated image, that fit
measured roughly the angle just applied, and it was stored as the new truth for
the next frame. It invented rotations up to 35° off on an upright face.

The metric used to validate roll compensation was *detection rate*, which could
not see this: the tracker was finding faces perfectly well, just at the wrong
angle. Only a human watching the live view noticed.

> When caching or tracking state feeds back into the input, ask what stops an
> error compounding. And check whether your success metric can even observe the
> failure you would care about.

## Measuring a fix with the input it was supposed to earn

Roll levelling was validated by rotating crops whose centre and width were already
known, and scored 99.6% at 30° of tilt against 61.9% axis-aligned. Both numbers
are correct, and levelling is a real fix.

But nothing in the live pipeline hands you a centre and a width. They come from a
mesh fit on the same tilted image — degraded by exactly the tilt being corrected.
Re-measured end to end, with the geometry re-derived the way production has to
derive it, 35° of roll delivered **69.7%**, not 99.6%. Half the failure was in the
stage that was never in the experiment.

The synthetic tilt sweep was the right instrument. It was pointed at the second
half of the pipeline and the number was read as though it covered both.

> When a fix is measured with one stage's input supplied by hand, the number is
> that stage's ceiling. Ask where that input comes from in the real thing.

## A number that only ever moves smoothly, so nothing ever looks broken

The eye width fed to the crop was the width of an axis-aligned box around the
eyelid ring. On a tilted head a rotated ellipse projects narrower onto x, so the
width read 0.81× true at 35° and every crop was zoomed by about 1.2×.

It survived because it has no failure mode. It does not throw, it does not return
None, it does not spike; it degrades smoothly with an angle nobody was plotting it
against, and the result is a 40×24 eye that looks exactly like a 40×24 eye. The
three-copies-of-the-same-expression problem made it worse — the mesh locator, the
capture path and the crop cache each computed the width inline, so there was no
single place where it was written down and could be questioned. It is now one
function, and the convention it implements is stamped into the crop cache and into
every model generation.

> A quantity with no failure mode needs a plot against the variable it depends on,
> because it will never announce itself. And a definition that is inline in three
> places is not a definition.

## A cache rebuild that quietly shrank the dataset

Crops are written at capture time by the full locator cascade, and rebuilt on a
cache miss by — until now — a bare mesh call. So the two disagreed twice over: on
the crop convention, and on which frames get a crop at all. A frame that only the
roll rescue or the YuNet box could locate simply had no crop after a rebuild, and
a sample with no crop is skipped by `Dataset.samples`, so its labels left the
training set without anything being logged.

It surfaced only because a convention change forced a rebuild and the next
generation trained on **951 samples across 186 bursts instead of 983 and 200**.
Nothing errored. A training set that is 3% smaller does not look like a bug, it
looks like a training set.

Both callers now build the stack through one function.

> A cache that can be rebuilt by a different code path than the one that filled it
> has two definitions of its contents. Ask what happens to the entries the rebuild
> cannot produce — silently dropping them is the default, and it is the worst one.

## A drawing that cannot express the failure it is there to show

The live view drew an axis-aligned rectangle around each eye. The crop it stood
for is cut in the *levelled* frame, so on a tilted head the rectangle was not the
region the classifier saw — it could not be, having no way to be tilted. The
person watching the live view for a misplaced crop was watching a shape incapable
of showing one. It now draws the four corners actually sampled.

The same watching is what found this whole class of bug in the first place, which
is the argument for keeping the drawing honest rather than tidy.

> A diagnostic drawing has to live in the same coordinate system as the thing it
> reports on. Otherwise it is decoration that looks like evidence.

## Silent-zero and empty-collection bugs in diagnostics

Two separate cases where a *diagnostic* lied:

- `(p or 1) < 0.5` — a probability of exactly `0.0` is falsy, so the most
  confident detections were rewritten to 1.0 and counted as open. Reported 0%
  detection for a working classifier. Use `is not None`.
- A montage batched tiles into rows of four with `range(0, len(tiles) - 3, 4)`,
  which is empty below four items — so it wrote nothing and said nothing. The tool
  whose job was showing failures failed silently when there were few.

> A diagnostic that can be wrong is worse than no diagnostic, because it is
> trusted. Make them loud when they have nothing to say.

## Lazy loaders inside comprehensions

`{name: z["crops"][i] for i, name in enumerate(z["names"])}` on an `NpzFile`.
Lookups are lazy, so this decompressed the entire archive **once per sample** —
2355 ms to load 418 crops, on the interactive path after every label edit. Hoisting
the two arrays: 370 ms. Later, `savez` instead of `savez_compressed` for the write
path: 1.0 ms instead of 49.6 ms, for 0.28 MB more.

> Anything indexed inside a comprehension may not be the cheap array access it
> looks like.

## Two processes, one file, no coordination

The web UI held the dataset in memory (deliberately — reloading per edit was the
2.4 s bug above) and never re-read it, so captures made in the camera app were
invisible until restart. The mirror image also existed: labels changed in the
browser were ignored by the camera app's next training run.

Fixed by stamping the file's mtime on load and reloading only when the *other*
process has written. A no-op check costs 0.002 ms.

> An in-memory cache is a decision to go stale. Write down what invalidates it at
> the moment you add it.

## Naming that contradicts the data

`img_left` / `img_right` were documented as picture-sides. Measured, `img_left`
sits at mean x=347 of 640 — the *right* half. They are subject-relative,
following MediaPipe, and every stored label is keyed by them.

The fix was to correct the display and the comments, **not** the keys: renaming
them would have silently inverted the entire dataset.

> When a name and the data disagree, find out which one the stored records depend
> on before changing either.

---

## Habits that paid off

- **Re-measure after every change, on held-out data.** Most reversals in this
  document came from doing this, not from insight.
- **Prefer the experiment that can embarrass you.** Detection *rate* could not see
  the roll bug; a synthetic tilt sweep could.
- **Keep negative results in the code**, off by default, with the numbers in the
  docstring (`Dataset.jittered`). Otherwise the next person re-derives them.
- **Record the preprocessing inside the model.** `blur`, `normalise`,
  `roll_levelled` travel with the weights, so old generations keep meaning what
  they meant. Every one of those was a convention change that would otherwise have
  silently invalidated earlier models.
- **Never delete captures.** `skip` excludes a sample from training and is
  reversible; deletion is not. It also means the dataset needs no undo.
