# Findings

Everything here was measured on a Dell `Integrated_Webcam_HD` infrared sensor
(`/dev/video2`, 640×360 `GREY`) with one subject. Numbers are from this hardware;
the *shapes* of the results are likely general, the constants are not.

Read [PITFALLS.md](PITFALLS.md) alongside this. Several conclusions below replaced
earlier ones that were wrong, and the ways they were wrong repeat.

---

## The camera

| | |
|---|---|
| advertised | 30 fps |
| **actually delivered** | **15.1 fps**, uniform 66 ms intervals |
| after discarding unlit strobe frames | **~7.5 fps** |
| eye width at normal seating distance | 16–19 px |
| V4L2 controls exposed | `region_of_interest` only — no exposure, gain or rate |

`grab()` alone (no decode, no conversion) also costs 66 ms, so the 15 fps is the
camera, not our processing. There is no knob to change it.

**The emitter strobes.** Alternate frames are unlit — around 5 brightness against
40–95 for lit ones. They are discarded, which is where half the frame rate goes.
Their intended purpose is ambient subtraction (see below).

**A natural blink is ~150 ms**, so it occupies roughly *one* usable frame. Any
gesture built on this hardware must be **held**, not flicked. Related trap: a
temporal median over 3 frames silently erases anything shorter than ~0.27 s,
which is an excellent way to convince yourself a working detector never fires.

**Capture latency**: the V4L2 buffer defaults to about 4 frames deep — ~264 ms of
lag. `CAP_PROP_BUFFERSIZE = 1` brings it to one frame, 66 ms.

### Falling behind costs 133 ms, not a millisecond

The "~7.5 usable fps" above is a **camera** measurement — `grab()` alone, no
processing. The end-to-end rate was never measured until the live view grew a
frame-time lane, and it showed 260 ms intervals, not 133.

The reason is that the cost is **quantised**. Lit frames arrive every 133 ms; a
pass that runs 1 ms over misses one entirely, the strobe then hands over an unlit
frame to discard, and the next usable frame is 266 ms after the last. There is no
value between.

Per-frame cost, measured on real frames at UI scale 2:

| stage | median |
|---|---|
| mesh, one pass | 31.8 ms |
| `LevelRefit(mesh)` — two passes | 64.3 ms |
| `default_locator` (what runs) | 61.2 ms |
| `model.eyes` (classify) | 0.9 ms |
| render | 22.8 ms |

| `default_locator` on | median | p90 | max |
|---|---|---|---|
| dataset frames (easy) | 49.5 ms | 69.4 ms | 110.1 ms |
| **frames where detection struggles** | 68.4 ms | **125.7 ms** | **211.6 ms** |

Adding render, the easy case sits at ~73 ms and the hard case at ~91 ms median but
149 ms at p90 — straddling the line, which is what makes the observed rate flicker
between 133 and 266.

**`LevelRefit` fires on 54% of frames**, because its deadzone is 3° and the median
|head roll| here is 3.2° — the threshold sits at the middle of the distribution. It
was chosen from a noise argument ("measured roll wanders a degree or two") without
looking at how much roll a head actually has.

| deadzone | fires on | mean detection cost |
|---|---|---|
| 3° (as shipped) | 54% | 49 ms |
| 8° | 15% | 37 ms |
| 20° | 5% | 33 ms |

From the roll table, the refit is worth 7–9 points at ±20° and **0.3 points at 0°**,
so most of those 54% are paying 33 ms for nothing. Raising the deadzone is
unmeasured but cheap; `roll_sweep.py --angles=-15,-10,-5,0,5,10,15` would place it.

### Concurrency removes the quantisation

`bgate/pipeline.py` runs capture in its own thread, feeding a bounded queue that
drops its oldest frame when full. Waiting for the camera and processing a frame stop
being the same wall-clock time, so throughput becomes **1/T** instead of
1/(⌈T/133⌉·133). Measured against a synthetic strobing camera:

| processing time | serial (quantised) | concurrent |
|---|---|---|
| 60 ms | 7.5/s | 7.7/s |
| 180 ms | 3.8/s | **5.7/s** |
| 233 ms | 3.8/s | **4.2/s** |

The gain is largest just above a multiple of 133 ms, and it cannot exceed the
capture rate — you can only process frames that exist. **It does not make anything
faster.** What it changes is that falling behind now costs what it costs, and every
dropped frame is counted rather than showing up as an unexplainable frame rate.

---

## MediaPipe

`face_landmarker.task` is a bundle of three models:

- `face_detector.tflite` (224 KB) — BlazeFace short-range, finds the face box
- `face_landmarks_detector.tflite` (2.5 MB) — the **refined 478-point** mesh
- `face_blendshapes.tflite` (933 KB)

The dependency is strictly sequential: **no box → no mesh → no eye position → no
prediction.** The classifier never gets a chance to be wrong, because it never
runs. That makes detection failures look like classifier failures.

### Settings that matter

**`min_face_detection_confidence` — default 0.5 is far too strict here.** At 0.5
it found 49% of the frames captured because detection was struggling; at 0.2,
74%, and 100% once roll compensation is applied. Ordinary-frame accuracy went
*up* (99.2% → 99.6%), and 39 faceless inputs (sensor noise, flat fields, real
frames with the face blanked out) produced **zero** false positives at any
threshold down to 0.1. This single default was doing more damage than two
elaborate mechanisms built to work around it.

**`RunningMode.VIDEO` lags; use `IMAGE`.** VIDEO seeds each frame from the
previous one and damps frame-to-frame iris motion about tenfold (0.11 px vs
1.05 px). That smoothing is exactly what leaves the eye box trailing behind a
head turn. IMAGE mode costs ~1.4 ms more and has no lag.

### Landmark facts

**`RING_L` is the SUBJECT'S left eye, which appears on the RIGHT of an unmirrored
picture.** Measured: `RING_L` centres at x=347 of 640, `RING_R` at x=307. The
dataset's `img_left` / `img_right` keys follow MediaPipe's convention, so they are
subject-relative despite the names. Do not "fix" the key names — every stored
label is keyed by them.

**Eyelid landmarks do not track closure on infrared.** The mesh regresses a
canonical topology; when local evidence is weak it relaxes toward the mean face,
which has open eyes. Observed open-eye EAR of 0.576 against 0.18–0.39 on a clean
RGB reference — geometrically implausible. EAR is also not scale-stable here: the
same reference face measured 0.177 at full resolution and 0.369 downscaled.

**Position is reliable even where deformation is not.** That asymmetry is what
the whole design rests on: use the mesh to *locate* the eye, and decide open vs
closed from the pixels. Reliable on an upright head — see [Finding a face and
locating the eyes](#finding-a-face-and-locating-the-eyes-inside-it-are-different-problems)
for what roll does to it.

**The eyelid ring's mean is a fair centre for the eye.** It sits 0.6 px *above*
the iris centre over 395 labelled eyes, 0.034 of an eye width and the same for
open and closed, so the crop is not systematically framed low. Worth having
measured, because it is a constant offset and the synthetic roll sweep is blind to
constants by construction.

### Where the pupil actually is

Measured independently of the mesh, by finding the bright disc directly with a
scale-normalised Laplacian of Gaussian, over 636 confident detections. Offsets are
in the levelled frame, so they are the shift the crop actually suffers.

| | mean | sd | p50 | p90 | p99 | max |
|---|---|---|---|---|---|---|
| dx | −0.39 | 1.93 | −0.17 | 1.50 | 4.50 | 6.00 |
| dy | −1.38 | 0.89 | −1.33 | −0.50 | 0.94 | 3.00 |
| \|d\| | 2.20 | 1.33 | 1.82 | 3.68 | 6.70 | **9.94** |

**Re-centring the crop does not fix it.** There is a real vertical bias — the
pupil sits 1.4 px above the crop centre — but removing it moves the median from
1.82 px to 1.17 px and the fraction past the 4 px cliff from 8.8% to 8.6%. The
dominant term is horizontal scatter of sd 1.9 px, and that is **gaze**: where the
subject is looking inside their own eye socket. No detector can remove it, because
it is not a detection error. That is the case for shift invariance in the
classifier rather than more work on the geometry.

**It is the same for both eyes** (img_left dx −0.64 ± 2.02, img_right −0.13 ±
1.81), so there is no per-eye correction to make.

**The crop is already wide enough.** 0 of 636 pupils fell outside the current
`PAD_W = 1.6` crop, and the worst offset in the dataset is 9.94 px. Pooling has
the margin it needs without widening anything.

Two controls, because a blob detector always returns a maximum: on closed eyes the
response is 0.68 median against 1.90 for open (76% of open above the pooled
median, 4% of closed), and the blob agrees with the mesh's own iris landmark to a
median of 1.8 px with 86% inside 4 px. Where the two *disagree* the crop is
mislocated; where they agree and the pupil is still off-centre, that is gaze.

**13% of open eyes have no confident pupil at all** — low contrast, nothing for a
blob detector to find. A model keyed on the pupil alone would abstain on one open
eye in eight, which is why the ridge channel is not optional.

**Blendshapes carry a left/right bias on grayscale IR** — `eyeBlinkLeft` reads
about 2× `eyeBlinkRight` at rest (0.132 vs 0.063) and during *symmetric* both-eye
closure (0.598 vs 0.359). Fatal for any differential L−R decision.

---

## Classifiers

### Off-the-shelf IR eye-state models do not transfer

[OCEC](https://github.com/PINTO0309/OCEC), trained on ~245k eye crops, F1 0.9933
on its own data:

| | balanced accuracy | closed eyes caught |
|---|---|---|
| all six OCEC size variants | 50–63% | 0–68% |
| + CLAHE / equalise / contrast-stretch | 50% (chance) | 0% |
| + crop padding swept 1.0→3.0 | no improvement | — |
| **logistic regression fitted on this camera** | **~98%** | **~98%** |

They are trained on well-exposed close-up infrared eyes; a laptop Hello camera
gives an 18-pixel-wide, low-contrast, often oblique one. They fail **confidently**
— returning 1.000 with no uncertainty — which reads as a broken detector rather
than an unsure one. Contrast normalisation made them *worse*, pushing the input
further from what they knew.

A plain logistic regression on raw pixels, fitted on ~300 crops from the sensor it
runs on, reaches 97.7% with whole capture bursts held out. 5 KB, 0.04 ms for both
eyes. **Enrolment beats transfer here.**

### The model is a template matcher, and that has consequences

A linear model on raw pixels *is* a template. It is brittle to anything that
moves or rescales the content:

| crop shift | verdict flips | accuracy |
|---|---|---|
| 1 px | 0.7% | 99.3% |
| 3 px | 3.3% | 96.6% |
| **4 px** | **13.9%** | **86.1%** |
| 6 px | 20.3% | 79.6% |

Graceful to ~3 px, then a cliff. Every source of geometry noise — head pose,
distance, a different detector — lands there.

> **These numbers do not state a shift direction, and the direction matters more
> than anything else in the table.** Re-measured, the same model reads 75.4% at
> 4 px shifted horizontally and **44.2%** shifted in a random direction. The crop
> is 29 source px wide but only 17 tall, so a vertical shift is proportionally
> 1.7× the insult. Assume the tables in this section are the horizontal, gentler
> case. `pooled_eval.py --direction` measures either.

**Three fixes, all measured, all carried inside the model** (a model fitted one
way is meaningless applied another, so `blur`, `normalise` and `roll_levelled`
are recorded per generation and applied from the model at inference):

**1. Gaussian blur σ=1.6 before featurisation.** Each feature covers more area, so
a shift moves content a smaller fraction of a feature. σ 1.2–2.0 is a plateau.

| σ | 0 px | 2 px | 3 px | 4 px | 6 px |
|---|---|---|---|---|---|
| none | 98.8% | 97.3% | 94.5% | 85.4% | 78.3% |
| 1.6 | 98.8% | 98.0% | 97.2% | **92.1%** | 81.0% |

**2. Per-crop standardisation `(x − mean) / std`.** Raw features are absolute pixel
values, so the decision boundary is pinned to the illumination of whichever
session was captured. Change the light and it silently stops working.

| features | ×0.6 | ×0.75 | as-is | ×1.3 | γ1.4 |
|---|---|---|---|---|---|
| raw | 57.2% | 61.4% | 98.8% | 88.3% | 66.3% |
| **standardise** | **98.8%** | **98.8%** | **98.8%** | **98.8%** | **98.8%** |

Flat, because gain and offset are what a lighting change mostly *is*. CLAHE and
DoG also help but less completely (76.9% and 96.6% worst case).

**3. Level the crop by head roll.** Locating a tilted eye is not enough — an
axis-aligned rectangle around it still contains a *rotated* eye, while every
training crop holds a near-upright one.

| head tilt | axis-aligned | levelled |
|---|---|---|
| 0° | 98.7% | 99.6% |
| 15° | 89.5% | 99.2% |
| **30°** | **61.9%** | **99.6%** |
| −30° | 87.9% | 98.7% |

Those figures are with the eye's centre and width *given*. The pipeline has to
re-derive them from the tilted image, and most of the remaining cost turned out
to live there — see [Finding a face is not locating the
eyes](#finding-a-face-and-locating-the-eyes-inside-it-are-different-problems).
99.6% is the ceiling levelling can reach, not the number it delivered.

### Jitter augmentation does not work here

Tried before the blur, and it made things **worse** — monotonically, with more
copies worse than fewer (81.3% at 4 px with ×4, against 85.4% with none).
Averaging a template over shifted copies blurs it without teaching invariance;
invariance needs pooling, which a linear model does not have. Applying the blur
deliberately, at training *and* inference, is the version that works.

The hook is kept in `fit(augment=...)`, off, so the measurement can be repeated —
not because it is recommended.

**That reasoning stops applying the moment there is pooling.** "Invariance needs
pooling, which a linear model does not have" is a statement about the *pixel*
model. Do not carry the anti-augmentation conclusion over to the pooled one
without re-measuring it.

---

## Pooling, instead of a template

A linear model can encode "bright **here**". It cannot encode "bright **anywhere
near here**" — that needs a maximum over positions, and a sum is not a maximum.
Every geometry fix above is a prosthetic for that missing operation.

`bgate/pooled.py` supplies it: convolve a kernel over every position, take the
maximum of the response map, feed those maxima to the same logistic layer. Move
the pupil and the peak moves with it; the maximum does not change.

Two channels, which are the two things visible in an IR eye crop — a bright disc
with a darker ring (Laplacian of Gaussian) and a dark elongated bar (an
anisotropic ridge filter, at three orientations because an arc is locally a
tilted line). Both start at their textbook shapes and are then fitted.

Scored on deliberate crop shifts, random direction, whole bursts held out:

| model | 0 px | 2 px | 3 px | 4 px | 6 px | 8 px |
|---|---|---|---|---|---|---|
| pixel template | 96.3% | 79.3% | 56.3% | 44.2% | 46.3% | 54.0% |
| pooled, fixed kernels | 99.4% | 98.3% | 98.1% | 95.6% | 81.4% | 68.5% |
| **pooled, learned kernels** | **99.5%** | **99.4%** | **99.1%** | **98.3%** | **94.4%** | **82.4%** |

And on real head roll, where the geometry is re-derived from a rotated frame
rather than perturbed by a known amount (`roll_sweep.py --score`):

| geometry | model | −35° | −20° | 0° | +20° | +35° |
|---|---|---|---|---|---|---|
| as found | pixel | 62.4% | 83.3% | 95.1% | 81.6% | 74.2% |
| as found | pooled | 88.6% | 97.1% | 99.7% | 94.8% | 93.2% |
| refit/level | pixel | 86.1% | 94.0% | 96.0% | 90.9% | 88.6% |
| refit/level | **pooled** | **92.1%** | **98.6%** | **99.8%** | **97.9%** | **93.8%** |

**Learning the kernels earns its place**, which was the open question: +2.7 points
at 4 px, +13 at 6 px, +14 at 8 px over the textbook bank. Max-pooling makes the
backward pass trivial — only the winning position contributed — so this is numpy
and an Adam loop, not a framework.

It is also *smaller*: 621 weights against 961. The "a convnet needs more data"
caution is about stacked layers; one learned filter bank with global max-pooling
has fewer parameters than the model it replaces.

**Pooling substitutes for most of the geometry work.** The pooled model on raw
geometry — no levelling refit, no levelled width — beats the pixel model *with*
the full geometry stack at every roll angle. Both attack the same failure, so
credit for it cannot be claimed twice. The geometry fixes still stack (88.6% →
92.1% at −35°) and they still fix *detection*, which pooling cannot touch: a face
that is not found has no crop to pool over.

### Two guards that are not optional

**Kernels are held zero-mean and unit-norm**, re-projected every step. Zero-mean
so a flat region scores exactly 0 — otherwise the pooling maximum locks onto the
brightest patch of cheek. Unit-norm so a kernel cannot inflate its own feature,
leaving the logistic layer fitting a scale it does not control.

**A negative control, because 99.5% deserves disbelief.** With the labels
permuted there is nothing to learn, so anything above the majority class means the
harness leaks. Measured: pixel 59.9%, pooled-fixed 57.0%, pooled-learned 62.9%,
against a 63.7% majority. `pooled_eval.py --shuffle`.

### The ablation is not clean, and says so

| bank | 0 px | 4 px | 6 px | 8 px |
|---|---|---|---|---|
| blobs only (2 kernels) | 99.4% | 96.9% | 86.8% | 74.1% |
| ridges only (3 kernels) | 99.8% | 98.2% | 87.8% | 74.0% |
| both (5) | 99.5% | 98.3% | **94.4%** | **82.4%** |

Ridges edging ahead of blobs is **not** a finding — 3 kernels against 2, not
parameter-matched. What is solid: both channels together beat either alone by 7–8
points in the tail.

The learned kernels muddy it further. The blobs stayed blobs (0.70 and 0.83 of
unit norm from where they started) but the ridges moved 1.31–1.52 and two of the
three visibly stopped being bars. Given spare capacity the fit spends it on blob
detectors.

---

## Detectors

**Neither MediaPipe nor YuNet is better. They fail in different places.**

| | hard batch A (tilted) | hard batch B (dark) |
|---|---|---|
| mediapipe | 67% | 64% |
| mediapipe + roll | 93% | 64% |
| yunet | 44% | **100%** |

Batch A was collected against the pre-roll mesh, so it is full of tilted heads —
which is precisely YuNet's weakness. Any comparison is only as good as the frames
it is run on.

**Use a second detector as a region proposer, not as a source of eye positions.**
YuNet's own eye keypoints produce crops that agree with the mesh on only ~80% of
verdicts, i.e. a second crop convention. Running the mesh *inside* YuNet's box
keeps one convention and makes the mesh's job easier (a small face becomes large
and centred). **The mesh succeeded inside every box YuNet proposed — 39/39.**

**Roll compensation: rotate to level, detect, map back.** Took detection from 67%
to 93% on frames that were failing. But see PITFALLS — tracking the angle to make
it cheap introduced a feedback loop that invented rotations up to 35° off. Always
try level *first*.

**Do not assume a failure needs a bigger model.** Haar cascades — 2001, no neural
network — found 100% of the frames that the whole modern cascade missed. So did
BlazeFace at confidence 0.2, which is the *same detector already inside the
bundle*. The fix was a threshold, not a model.

### Finding a face and locating the eyes inside it are different problems

Detection rate reads 100% on a tilted head. The eye positions *inside* that
detection are still wrong, and the classifier is a template matcher with a cliff
at 4 px — so it does not need the face to be missed in order to fail. This is the
failure a detection rate cannot see, and it is the second time that sentence has
had to be written here.

Measured by `roll_sweep.py`: rotate a labelled frame by a known angle, which makes
the true eye position exactly known — it is the same affine that made the image —
then score the verdicts against the labels with whole capture bursts held out.

| roll | as found | + refit | + levelled width |
|---|---|---|---|
| −35° | 60.9% | 76.7% | **84.4%** |
| −20° | 80.4% | 93.1% | **94.6%** |
| 0° | 94.2% | 95.4% | **95.4%** |
| +20° | 78.2% | 88.5% | **89.4%** |
| +35° | 69.7% | 77.7% | **86.7%** |

477 frames, 954 labelled eyes per angle. The upright column is not the 97.7%
headline: this re-detects every frame from scratch rather than reusing the crop
cut when it was captured, which is the honest thing to compare against and about
a point and a half more pessimistic.

Two independent defects, neither of them a detection failure:

**The mesh under-measures roll on a head it can still find.** At 35° of real roll
the crop was left holding 6° of tilt and the eye centre was out by 4.5 px mean,
12.2 px at the 90th percentile — well past where 14% of verdicts flip. Re-fitting
**once** on the levelled image takes that to 3.6 px / 9.9 px. The error is also
strongly asymmetric between roll directions (−35° is worse than +35°: 6.4 px and
13.6° of residual tilt, against 4.5 px and 6.2°), which one subject cannot
distinguish from a property of the mesh.

Once, and anchored on the unrotated fit. The angle is always taken from the fit
made on the frame as it arrived, so it cannot have been suggested by a rotation we
applied — that is the entire safety argument. Written the obvious other way, with
a second pass adding its newly measured roll to the angle already applied, it
double counts: a detection mapped back into the original frame reports the
*absolute* roll, not the residual. That version invented 48.5° on a face rolled
33.7°, and mean eye error at 35° went to 9.2 px against 3.9 px for not refitting
at all — the `RollTracker` feedback loop from PITFALLS, one class down and in a
different costume.

The refit is not a rescue. On the worst frames at −35° the tail barely moves
(p90 17.9 px → 16.9 px): a refit anchored on a bad first fit stays bad, and 82 of
1744 refits were rejected outright for moving an eye more than its own width.

**An eye width measured as the width of an axis-aligned box shrinks as the head
tilts.** A rotated ellipse projects narrower onto x. Measured, 0.81× the true
width at 35°, and since that width sets the crop's scale, every crop on a tilted
head was zoomed by about 1.2× against everything the model was fitted on.
Projecting the eyelid ring onto the *levelled* axis instead holds within 3% across
the whole sweep, and costs nothing: the roll it needs is already being computed in
order to level the crop.

This one is worth its own note because it is the kind of defect that hides
forever. It is deterministic, it is invisible in a 40×24 thumbnail, and it grows
smoothly with tilt rather than failing anywhere.

---

## Ambient infrared is the outer limit

Under ambient flooding, frame brightness roughly doubled (93.6 vs 47.4) and
contrast fell to **0.26×** (14.2 vs 54.7). Glint contrast at the eye went from ~40
to 16–25. Per-crop standardisation rescales that back to unit variance but cannot
create signal that is not there.

This matches [irlume](https://github.com/archledger/irlume)'s independently
measured ceiling: reliable below ambient IR ~120, **0% genuine acceptance above
~170**.

Ambient subtraction (`IRCamera(subtract=True)`) is implemented and **unmeasured**.
A quick check moved brightness 94.0 → 30.8 but contrast 15.3 → 11.9, i.e.
slightly down: removing a constant pedestal buys nothing that standardisation was
not already doing, and differencing two noisy frames costs SNR. Where it should
help is *non-uniform* ambient — shadows, window reflections. `ambient_test.py`
answers this on real frames; run it before trusting either way.

---

## Project context

- **[Howdy](https://github.com/boltgolt/howdy)** — 7.7k stars, but **zero commits
  in 12 months** and the last tagged release was v2.6.1 in 2020. Its rubberstamp
  system (`compare.py` calls it after a successful match; a stamp returning False
  exits 15 → `PAM_AUTH_ERR`) is the natural extension point, and is upstream-
  described as not finished.
- **[irlume](https://github.com/archledger/irlume)** — Rust, ships a nod-to-approve
  consent gesture for polkit plus opt-in blink liveness. Its published PAD results
  independently corroborate the blink findings here: prompted deliberate blinks
  rejected on UX grounds, passive EAR measured 0/12 in darkness and 11/11
  non-response in daylight, fixed only by a bespoke temporal-shape detector.
- **The argument for any of this**: a face match proves *presence*, not *consent*.
  Any process can fire a `sudo` while you happen to be sitting in front of the
  laptop. A deliberate gesture closes that gap — which is why the gesture only has
  to be deliberate, not secret.

## Still unmeasured

**The false-positive rate during ordinary blinking.** This decides whether any of
it can gate privilege escalation, and posed captures cannot produce it — it needs
a passive recording while the camera is ignored. Every accuracy number in this
document is conditioned on a deliberate pose.
