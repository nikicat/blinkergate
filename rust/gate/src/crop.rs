//! Where an eye is, and the crop the classifier reads.
//!
//! A port of `bgate/data.py`'s geometry, which is the single definition of that
//! on the Python side for a reason: the mesh locator, the capture path and the
//! crop cache each computed the eye width inline once, they drifted, and a width
//! is a scale - every crop on a tilted head was silently zoomed 1.2x against
//! everything the model was fitted on. Nothing about a 40x24 eye makes that
//! visible.
//!
//! So the same rules apply here as there:
//!
//! - the crop is cut in the **levelled** frame, because locating a tilted eye is
//!   not the same as being able to classify it (98.7% -> 61.9% axis-aligned at
//!   30 degrees of roll, 99.6% levelled);
//! - the eye width is measured along that **levelled axis**, because a rotated
//!   ellipse projects narrower onto x (0.81x true width at 35 degrees);
//! - which convention was used is stamped into every model generation, and
//!   [`Geometry::check`] refuses a model that was fitted under a different one.

use serde::Deserialize;

use crate::frame::{Frame, Grey, Px, Raster, Sampling, Size};
use crate::model::ModelError;
use crate::side::PerEye;

/// An index into one mesh fit's landmarks.
///
/// Its own type because a landmark id, a position within a ring, a pixel offset and
/// a crop coordinate are all bare integers, and only the first may be used to read a
/// point out of a [`Landmarks`]. `u16` because the refined mesh has 478 points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(transparent)]
pub struct Landmark(u16);

impl Landmark {
    pub const fn new(index: u16) -> Landmark {
        Landmark(index)
    }

    const fn get(self) -> usize {
        self.0 as usize
    }
}

/// A ring written as plain integers, so the list below stays readable - it is checked
/// digit by digit against MediaPipe's own constants.
const fn ring(raw: [u16; 16]) -> [Landmark; 16] {
    let mut out = [Landmark(0); 16];
    let mut i = 0;
    while i < raw.len() {
        out[i] = Landmark(raw[i]);
        i += 1;
    }
    out
}

/// MediaPipe eyelid rings, copied from `FaceLandmarksConnections` in MediaPipe's
/// Python wrapper by way of `bgate/ui.py`.
///
/// Copied because they cannot be imported: MediaPipe ships the rings only in its
/// Python and JS wrappers, and the C API this daemon calls carries no landmark
/// indices at all. So `export_model.py` checks them against those constants - same
/// vertices, and this order is a closed walk of MediaPipe's edges - and writes them
/// into `golden.json`, where `rings_are_mediapipes_own` holds this copy to them.
///
/// `RING_L` is the SUBJECT'S left eye, which on an unmirrored feed appears on the
/// RIGHT of the picture (measured: mean x 347 of 640, against 307 for the right
/// eye). Sides are the subject's throughout this codebase; the dataset's `img_left`
/// / `img_right` keys follow the same convention despite reading like picture
/// sides.
pub const RING_L: [Landmark; 16] =
    ring([362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]);
pub const RING_R: [Landmark; 16] =
    ring([33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]);

/// The highest landmark the rings above reach. The refined mesh has 478 points;
/// anything shorter is a different model, and these indices would then be reading
/// whatever happens to sit there.
pub const RING_MAX: Landmark = Landmark::new(466);

/// How a crop is derived from landmarks. Any change to the geometry above must
/// change this string, so that a model fitted under the old rules is rejected
/// rather than silently fed crops it has never seen.
pub const CONVENTION: &str = "roll-levelled/level-width";

/// In-plane head roll, in **degrees**. Positive means the subject's left eye sits
/// lower in the image.
///
/// A type because the unit is not obvious and the two halves of this codebase
/// disagree about it: every measurement in `FINDINGS.md` is in degrees, every
/// trigonometric call needs radians, and a `f32` that has been through
/// `to_radians()` once too often is a crop rotated by 0.06 of a degree that looks
/// exactly like a crop rotated correctly.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Roll(f32);

impl Roll {
    /// A level head. Not reachable from the mesh, which measures roll with
    /// `atan2` and never returns exactly zero - see the note in `Geometry`.
    pub const LEVEL: Roll = Roll(0.0);

    pub const fn degrees(deg: f32) -> Roll {
        Roll(deg)
    }

    /// Magnitude, for the deadzone comparisons that do not care which way a head is
    /// tilted.
    pub fn abs(self) -> Roll {
        Roll(self.0.abs())
    }

    /// The rotation this roll stands for.
    pub fn turn(self) -> Turn {
        let (sin, cos) = self.0.to_radians().sin_cos();
        Turn { sin, cos }
    }
}

/// The sine and cosine of a rotation, named.
///
/// `f32::sin_cos` returns them positionally, and every rotation in this crate is
/// written out by hand - so swapping the two is a rotation by its complement,
/// which is a plausible-looking crop of the wrong part of the face.
#[derive(Debug, Clone, Copy)]
pub struct Turn {
    pub sin: f32,
    pub cos: f32,
}

/// A landmark in normalized coordinates, as MediaPipe reports them.
#[derive(Debug, Clone, Copy)]
pub struct Pt {
    pub x: f32,
    pub y: f32,
}

impl Pt {
    /// The only route into pixel space, and it takes the image the normalization
    /// is relative to - so a landmark cannot be multiplied by the wrong frame.
    #[inline]
    pub fn to_pixels(self, size: Size) -> Px {
        let (fw, fh) = size.as_f32();
        Px { x: self.x * fw, y: self.y * fh }
    }
}

/// One mesh fit's landmarks, checked once so the ring indices cannot miss.
///
/// The check is the whole reason this type exists: `RING_L` reads point 466, so a
/// result from some other landmarker - or a truncated one - would either panic in
/// a root daemon or silently crop from the wrong place. Finiteness is checked here
/// too, at the one boundary where a NaN could enter, rather than being carried
/// into a probability that compares false against every threshold and looks like
/// an eye that is merely open.
#[derive(Debug, Clone)]
pub struct Landmarks(Vec<Pt>);

impl Landmarks {
    pub fn new(points: Vec<Pt>) -> Option<Landmarks> {
        let ok = points.len() > RING_MAX.get()
            && points.iter().all(|p| p.x.is_finite() && p.y.is_finite());
        ok.then_some(Landmarks(points))
    }

    /// One landmark. In range for any [`Landmark`] up to [`RING_MAX`], which is what
    /// `new` checked - hence not public, since nothing outside this crate builds
    /// indices.
    #[inline]
    pub(crate) fn at(&self, i: Landmark) -> Pt {
        self.0[i.get()]
    }
}

/// Crop geometry, carried in the model file rather than hardcoded here: crops cut
/// one way are not crops cut the other way.
#[derive(Debug, Clone, Deserialize)]
pub struct Geometry {
    /// Crop width the model was fitted at, in crop pixels. 40 for the reference
    /// camera, against an eye that is 16-19 source pixels wide there - so a crop is
    /// an upscale, and the classifier sees interpolation rather than sensor detail.
    pub in_w: usize,
    /// Crop height, 24. Wider than tall because an eye is, and because the aspect is
    /// what makes a *vertical* crop shift 1.7x the insult of a horizontal one: the
    /// same pixel error covers more of the eye.
    pub in_h: usize,
    /// How much wider than the eye the crop reaches, as a multiple of eye width.
    /// 1.6 here, and measured to be enough: 0 of 636 confident pupils fell outside
    /// it, so pooling has the margin it needs without widening anything.
    pub pad_w: f32,
    /// The same for height, as a multiple of what the crop's aspect already implies.
    /// 1.0 - the aspect alone frames the eye, and this exists so that a generation
    /// fitted on taller crops keeps meaning what it meant.
    pub pad_h: f32,
    /// Whether the eye width was measured along the levelled axis rather than as the
    /// width of an axis-aligned box. False only for generations that predate the
    /// fix, which [`Geometry::check`] refuses: an axis-aligned width reads 0.81x true
    /// at 35 degrees of roll, so every crop on a tilted head was zoomed by 1.2x
    /// against everything the model was fitted on.
    pub level_width: bool,
    /// Deliberately an opaque string and not an enum: the Python side mints these,
    /// so the set is open, and the only thing this build can say about one is
    /// whether it is [`CONVENTION`]. An enum would be a closed guess at an open set,
    /// and would report a future convention as "unknown variant" during parsing -
    /// before there is a model to name it in.
    pub convention: String,
}

impl Geometry {
    pub fn check(&self) -> Result<(), ModelError> {
        if self.convention != CONVENTION {
            return Err(ModelError::Convention {
                found: self.convention.clone(),
                expected: CONVENTION,
            });
        }
        if self.in_w == 0 || self.in_h == 0 || self.in_w * self.in_h > 1 << 16 {
            return Err(ModelError::Shape(format!(
                "implausible crop size {}x{}",
                self.in_w, self.in_h
            )));
        }
        if !self.level_width {
            // Pre-levelled-width generations exist and are still loadable in
            // Python for comparison; refusing them here keeps one convention in
            // the daemon rather than two code paths that look alike.
            return Err(ModelError::LevelWidth);
        }
        Ok(())
    }

    /// Centre and width of one eye, in source pixels, measured along the axis the
    /// crop will be levelled to.
    /// `ring` must hold indices within [`RING_MAX`], which is exactly what
    /// [`Landmarks`] guarantees is present - hence not public.
    pub(crate) fn eye(&self, lm: &Landmarks, ring: &[Landmark], size: Size, roll: Roll) -> Eye {
        let pts: Vec<Px> = ring.iter().map(|&i| lm.at(i).to_pixels(size)).collect();
        let centre = centroid(&pts);
        let t = if self.level_width { roll.turn() } else { Roll::LEVEL.turn() };
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for p in &pts {
            let proj = (p.x - centre.x) * t.cos + (p.y - centre.y) * t.sin;
            lo = lo.min(proj);
            hi = hi.max(proj);
        }
        Eye { centre, width: hi - lo }
    }

    /// The crop, given only where the eye is, how wide it is, and its tilt.
    ///
    /// The one definition of the geometry, so any detector that can supply a centre
    /// and a width produces crops the model recognises.
    pub fn crop_at(&self, frame: &Frame, eye: &Eye, roll: Roll) -> Crop {
        let sampling = self.sampling(eye, roll);
        Crop::from_positions(self.size(), |at| Grey::round(frame.sample(sampling.at_point(at))))
    }

    /// Which part of the frame this crop's pixels come from.
    fn sampling(&self, eye: &Eye, roll: Roll) -> Sampling {
        let size = self.size();
        let (ow, oh) = size.as_f32();
        // Half the region, in source pixels: as wide as the eye plus padding, and as
        // tall as the crop's own aspect requires so that nothing is stretched.
        let half_w = (eye.width * self.pad_w / 2.0).max(MIN_HALF_W);
        let half_h = half_w * (oh / ow) * self.pad_h;
        // Inverse of data.level_affine, written as the two source-space vectors that
        // one step right and one step down in the crop correspond to. Same
        // arithmetic as a 2x3 matrix and it says what it does: the crop walks the
        // region along the region's own, levelled axes. Scaling per axis rather than
        // uniformly is what makes this agree with an axis-aligned box for any pad_h,
        // not only pad_h = 1.
        let t = roll.turn();
        let (sx, sy) = (2.0 * half_w / ow, 2.0 * half_h / oh);
        Sampling {
            origin: eye.centre,
            mid: size.centre(),
            step_x: Px { x: t.cos * sx, y: t.sin * sx },
            step_y: Px { x: -t.sin * sy, y: t.cos * sy },
        }
    }

    /// The size a crop comes out at, as the model file recorded it.
    pub fn size(&self) -> Size {
        Size { width: self.in_w, height: self.in_h }
    }

    /// Where both eyes are in one frame, and the roll they were measured along.
    pub fn locate(&self, lm: &Landmarks, size: Size) -> Located {
        let roll = roll_from(self, lm, size);
        Located { eyes: PerEye::from_fn(|side| self.eye(lm, side.ring(), size, roll)), roll }
    }

    /// Both crops, levelled by the roll the same fit measured.
    ///
    /// Takes a [`Located`] rather than landmarks so the roll cannot be some other
    /// roll than the one the widths were measured along - that pairing is the
    /// convention, and mixing it is invisible in a 40x24 eye.
    pub fn cut(&self, frame: &Frame, at: &Located) -> PerEye<Crop> {
        at.eyes.map(|_side, eye| self.crop_at(frame, &eye, at.roll))
    }
}

/// A crop is never cut narrower than this, in source pixels either side of the eye.
///
/// The width comes from a mesh fit, and a fit that has gone wrong reports a tiny
/// eye rather than an error - the floor keeps that from becoming a 3-pixel crop
/// upscaled into something the classifier will confidently misread.
const MIN_HALF_W: f32 = 8.0;

/// The tag for pixels cut for the classifier. See [`Crop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cut;

/// One eye, cut to the size the model was fitted at.
///
/// The same [`Raster`] as a [`Frame`] and a different type on purpose: the
/// classifier derives its pooling window from the raster it is handed, so a whole
/// frame reaching it would read as a very large eye rather than as an error.
///
/// That a crop matches the geometry the *model* was fitted with is a separate
/// claim, checked once at load time from the stamp in the model file.
pub type Crop = Raster<Cut>;

/// Both eyes as located in one frame, with the roll that goes with them.
///
/// One type rather than a pair returned side by side: the roll is not a property
/// of either eye - it comes from the line between them - and it is the axis their
/// widths were measured along, so a crop cut with a different roll is cut under a
/// different convention.
#[derive(Debug, Clone, Copy)]
pub struct Located {
    pub eyes: PerEye<Eye>,
    pub roll: Roll,
}

/// Where one eye is, in source pixels. All a crop needs.
#[derive(Debug, Clone, Copy)]
pub struct Eye {
    pub centre: Px,
    /// Along the levelled axis, not the image's x - see the module header.
    pub width: f32,
}

/// The mean of some pixel positions. The one place a count becomes a float.
fn centroid(pts: &[Px]) -> Px {
    let n = pts.len() as f32;
    Px { x: pts.iter().map(|p| p.x).sum::<f32>() / n, y: pts.iter().map(|p| p.y).sum::<f32>() / n }
}

/// Head roll in degrees, from the line joining the two eye centres. Positive
/// means the subject's left eye sits lower in the image.
pub fn roll_from(geom: &Geometry, lm: &Landmarks, size: Size) -> Roll {
    // Level, because the width this discards is the only thing roll would change,
    // and the centres are what the angle is measured between.
    roll_of(&PerEye::from_fn(|side| geom.eye(lm, side.ring(), size, Roll::LEVEL)))
}

/// Roll of a pair of eyes already located, for the detector stack's own use.
pub fn roll_of(eyes: &PerEye<Eye>) -> Roll {
    let (l, r) = (eyes.left.centre, eyes.right.centre);
    Roll::degrees((l.y - r.y).atan2(l.x - r.x).to_degrees())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pts(n: usize) -> Vec<Pt> {
        (0..n).map(|i| Pt { x: i as f32 / n as f32, y: 0.5 }).collect()
    }

    #[test]
    fn landmarks_refuse_a_mesh_the_rings_would_overrun() {
        // The refined mesh is 478 points and RING_L reads 466. A landmarker
        // returning the 468-point model is fine; the 6-point face detector is not,
        // and indexing it would panic inside a root daemon.
        assert!(Landmarks::new(pts(478)).is_some());
        // A mesh has to *contain* the highest index a ring reads, so its length is
        // one more than that index.
        assert!(Landmarks::new(pts(RING_MAX.get() + 1)).is_some());
        assert!(Landmarks::new(pts(RING_MAX.get())).is_none());
        assert!(Landmarks::new(pts(6)).is_none());
        assert!(Landmarks::new(vec![]).is_none());
    }

    #[test]
    fn landmarks_refuse_a_non_finite_point() {
        // A NaN here would reach the sampler, then the probability, and then
        // compare false against the shut threshold - an eye that reads open
        // forever rather than an error.
        let mut p = pts(478);
        p[10].y = f32::NAN;
        assert!(Landmarks::new(p).is_none());
        let mut p = pts(478);
        p[0].x = f32::INFINITY;
        assert!(Landmarks::new(p).is_none());
    }
}
