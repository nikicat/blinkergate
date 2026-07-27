//! Finding a face, and locating the eyes inside it.
//!
//! Those are different jobs, and the difference is the most expensive lesson in
//! `docs/PITFALLS.md`: the mesh will happily fit a tilted head - detection rate stays
//! 100% - and fit it badly, and every error lands on a classifier with a cliff at
//! 4px. A detection *rate* cannot see that. So this module has two rescues, aimed at
//! the two different failures:
//!
//! - the rotation rescue is allowed to fail. It exists so a frame is not lost at all:
//!   rotating to level took detection from 67% to 93% on frames where the mesh was
//!   failing. (`RollTracker` on the Python side.)
//! - the levelled refit is only allowed to improve. It exists because a frame that
//!   was *not* lost can still be located badly, which cost 24 points of accuracy on a
//!   rolled head. (`LevelRefit` there.)
//!
//! YuNet, the second detector the Python side runs, is not ported: it needs an ONNX
//! runtime, and what it rescues is dark low-contrast frames rather than the tilted
//! ones this gate sees. A frame nothing can locate falls through to the password.

use std::num::NonZeroU32;

use mediapipe::{Confidence, FaceLandmarker, Image, ModelSource};

use crate::crop::{Geometry, Landmarks, Located, Pt, Roll, roll_of};
use crate::error::{Error, Result};
use crate::frame::Frame;
use crate::model::Probability;

/// The mesh, wrapped so the rest of this crate never sees MediaPipe's types.
pub struct Mesh {
    landmarker: FaceLandmarker,
    /// What it was built with, so a request whose model asks for a different
    /// threshold rebuilds rather than being quietly answered at the old one.
    confidence: Probability,
}

impl Mesh {
    /// Build a landmarker in IMAGE mode.
    ///
    /// IMAGE, not VIDEO: VIDEO seeds each frame from the previous one, which damps
    /// frame-to-frame iris motion about tenfold (0.11px against 1.05px). That
    /// smoothing is exactly the lag that leaves the eye box trailing behind a head
    /// turn, and it costs 1.4ms to not have it.
    pub fn open(bundle: &std::path::Path, confidence: Probability) -> Result<Mesh> {
        let mp = |e: mediapipe::Error| Error::Mesh { why: e.to_string() };
        let landmarker = FaceLandmarker::builder(ModelSource::path(bundle))
            .num_faces(NonZeroU32::new(1).expect("1 is not zero"))
            .min_face_detection_confidence(Confidence::new(confidence.get()).map_err(mp)?)
            .build()
            .map_err(mp)?;
        Ok(Mesh { landmarker, confidence })
    }

    pub fn confidence(&self) -> Probability {
        self.confidence
    }

    /// The landmarks for the one face in this frame, if there is one.
    ///
    /// `None` covers both "no face" and "a mesh this build cannot read" - see
    /// [`Landmarks::new`], which refuses a landmarker returning too few points rather
    /// than indexing past the end of one.
    pub fn locate(&mut self, frame: &Frame) -> Result<Option<Landmarks>> {
        let size = frame.size();
        // MediaPipe takes 8-bit RGB; a grayscale frame becomes three equal channels,
        // which is what `cv2.cvtColor(gray, COLOR_GRAY2RGB)` does on the Python side.
        let mut rgb = Vec::with_capacity(size.area() * 3);
        for p in frame.px() {
            rgb.extend_from_slice(&[p.raw(); 3]);
        }
        let mp = |e: mediapipe::Error| Error::Mesh { why: e.to_string() };
        let size = mediapipe::Size { width: size.width as u32, height: size.height as u32 };
        let image = Image::from_rgb(size, &rgb).map_err(mp)?;
        let result = self.landmarker.detect(&image).map_err(mp)?;
        let Some(face) = result.landmarks.first() else { return Ok(None) };
        let points = face.iter().map(|l| Pt { x: l.point.x(), y: l.point.y() }).collect();
        Ok(Landmarks::new(points))
    }
}

/// The locator stack: the mesh, rescued twice.
pub struct Locator {
    mesh: Mesh,
    geometry: Geometry,
    /// The last roll a face was actually found at, which is the first angle the
    /// rotation rescue tries once level has failed.
    tracked: Roll,
    /// Counted because a rescue that never fires and a rescue that fires on every
    /// frame are both worth knowing about, and neither shows up in a verdict.
    pub rescues: u32,
    pub refits: u32,
    pub refits_rejected: u32,
}

/// Below this the correction is not worth a warp and a second mesh pass: measured
/// roll on a level head wanders a degree or two frame to frame.
///
/// It is also the number `FINDINGS.md` calls out as chosen badly - the median |roll|
/// on the reference camera is 3.2 degrees, so a 3 degree deadzone fires on 54% of
/// frames for 0.3 points at 0 degrees. Raising it to 8 was measured to cost nothing
/// and save 12ms a frame, but not measured *end to end*, so it stays where the
/// accuracy numbers were taken.
const DEADZONE: Roll = Roll::degrees(3.0);

/// Angles to try when the mesh finds nothing at all, after level and after the angle
/// the last face was found at.
const SWEEP: [Roll; 4] =
    [Roll::degrees(-20.0), Roll::degrees(20.0), Roll::degrees(-35.0), Roll::degrees(35.0)];

/// A refit that moves an eye further than this, as a fraction of its own width, has
/// not refined the same face - keep the fit that was not built on a guess.
const MAX_MOVE: f32 = 1.0;

impl Locator {
    pub fn new(mesh: Mesh, geometry: Geometry) -> Locator {
        Locator { mesh, geometry, tracked: Roll::LEVEL, rescues: 0, refits: 0, refits_rejected: 0 }
    }

    /// Where the eyes are in this frame, or `None` if no face was found.
    pub fn locate(&mut self, frame: &Frame) -> Result<Option<Located>> {
        let Some(found) = self.find(frame)? else { return Ok(None) };
        Ok(Some(self.refit(frame, found)?))
    }

    /// Find the face, rotating the frame only if the frame as it arrived fails.
    ///
    /// Level *first*, always. Trying the tracked angle first is self-reinforcing and
    /// drifts: rotating by a remembered angle makes the mesh fit the rotated image,
    /// which then measures roughly the angle that was applied, which is stored as the
    /// new truth. Measured over 180 frames that invented rolls up to 35 degrees off,
    /// each frame handing its error to the next. An unrotated fit cannot be wrong that
    /// way, so rotation is left to rescue the frames that genuinely fail.
    fn find(&mut self, frame: &Frame) -> Result<Option<Located>> {
        if let Some(lm) = self.mesh.locate(frame)? {
            let at = self.geometry.locate(&lm, frame.size());
            self.tracked = at.roll;
            return Ok(Some(at));
        }
        // A tracked angle inside the deadzone is the level attempt that just failed.
        let tracked = (self.tracked.abs() >= DEADZONE).then_some(self.tracked);
        for angle in tracked.into_iter().chain(SWEEP) {
            let rotated = frame.rotated(angle);
            let Some(lm) = self.mesh.locate(&rotated)? else { continue };
            // Map the eyes back into the frame as it arrived: the crop is always cut
            // from the original, so a resampling artefact cannot reach the classifier.
            let at = unrotate(self.geometry.locate(&lm, rotated.size()), angle, frame);
            self.rescues += 1;
            // Track what the face actually is, not the angle that happened to work.
            self.tracked = at.roll;
            return Ok(Some(at));
        }
        Ok(None)
    }

    /// Re-fit once on the levelled image, and keep the result only if it moved the
    /// eyes plausibly.
    ///
    /// Once, and anchored on the unrotated fit: the angle always comes from the fit
    /// made on the frame as it arrived, so it cannot have been suggested by a rotation
    /// we applied. That is the entire safety argument. Written the obvious other way -
    /// a second pass adding its newly measured roll to the angle already applied - it
    /// double counts, because a detection mapped back into the original frame reports
    /// the *absolute* roll and not the residual. That version invented 48.5 degrees on
    /// a face rolled 33.7.
    fn refit(&mut self, frame: &Frame, found: Located) -> Result<Located> {
        if found.roll.abs() < DEADZONE {
            return Ok(found);
        }
        let rotated = frame.rotated(found.roll);
        let Some(lm) = self.mesh.locate(&rotated)? else { return Ok(found) };
        let refit = unrotate(self.geometry.locate(&lm, rotated.size()), found.roll, frame);
        if moved_too_far(&found, &refit) {
            self.refits_rejected += 1;
            return Ok(found);
        }
        self.refits += 1;
        Ok(refit)
    }

    /// Hand the mesh back, so the daemon can keep it warm for the next
    /// authentication while this locator and its counters are discarded.
    pub fn into_mesh(self) -> Mesh {
        self.mesh
    }
}

/// Map eyes located on a rotated frame back into the frame as it arrived.
///
/// Rotation does not change scale, so the widths carry over untouched; only the
/// centres move, and the roll is re-derived from them rather than reused - the two
/// centres are what the angle means.
fn unrotate(at: Located, by: Roll, frame: &Frame) -> Located {
    // The very map `Frame::rotated` sampled through: it already answers "where in the
    // original did this pixel come from", so there is nothing to invert here.
    let back = crate::frame::Sampling::rotation(frame.size(), by);
    let eyes = at
        .eyes
        .map(|_side, eye| crate::crop::Eye { centre: back.at_point(eye.centre), width: eye.width });
    Located { roll: roll_of(&eyes), eyes }
}

fn moved_too_far(found: &Located, refit: &Located) -> bool {
    found.eyes.iter().any(|(side, before)| {
        let after = refit.eyes.get(side);
        let moved = (after.centre.x - before.centre.x).hypot(after.centre.y - before.centre.y);
        moved > MAX_MOVE * before.width.max(1.0)
    })
}
