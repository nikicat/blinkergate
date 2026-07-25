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
