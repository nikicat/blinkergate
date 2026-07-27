//! Pixels: a size, a buffer that matches it, and the two ways this code resamples
//! one.
//!
//! Everything holding pixels here is a [`Raster`] - the camera's frame and the
//! classifier's crop alike. They are the same representation and deliberately *not*
//! the same type: [`crate::Model::p_open`] takes its pooling window from the raster
//! it is given, so a 640x360 frame reaching it would pool over the whole image and
//! return a plausible probability rather than an error. Same shape, different space,
//! so the tag parameter keeps them apart.
//!
//! The resamplers are ports of the OpenCV calls the Python pipeline makes, and they
//! have to agree with them closely: the classifier is fitted on crops cut by
//! `cv2.warpAffine`, and a crop cut a different way is a template shown to a model
//! that was never fitted on it. `warpAffine` with `INTER_AREA` silently falls back
//! to `INTER_LINEAR` inside OpenCV, so bilinear is what the training crops actually
//! are, not an approximation of them. `tests/golden.rs` holds that to numbers
//! produced by the Python side.

use std::marker::PhantomData;

use serde::Deserialize;

use crate::crop::Roll;

/// One 8-bit grayscale sample.
///
/// Its own type because a `u8` in this crate could otherwise be anything - a byte off
/// a v4l buffer, one channel of an RGB triple, a mode bit. Every pixel here is
/// single-channel infrared, and the two conversions that matter belong to the type
/// rather than being spelled out at each use: [`Grey::value`] for the arithmetic, and
/// [`Grey::round`] for coming back from it the way OpenCV does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(transparent)]
pub struct Grey(u8);

impl Grey {
    pub const fn new(sample: u8) -> Grey {
        Grey(sample)
    }

    /// The byte, for an API that speaks bytes - MediaPipe's RGB input, a file.
    pub const fn raw(self) -> u8 {
        self.0
    }

    /// As a float, which is what every resampling and every feature is computed in.
    #[inline]
    pub fn value(self) -> f32 {
        f32::from(self.0)
    }

    /// From a float, as OpenCV's `saturate_cast<uchar>`: round half away from zero,
    /// then clamp. The rounding is load-bearing - the crops the model was fitted on
    /// were rounded this way.
    #[inline]
    pub fn round(v: f32) -> Grey {
        Grey(v.round().clamp(0.0, 255.0) as u8)
    }
}

/// An image size in pixels.
///
/// Named fields and no positional constructor, because two `usize`s in a row are
/// exactly the swap this type exists to prevent - write `Size { width, height }`.
/// It is also the one place the integer-to-float conversion the geometry needs
/// happens, instead of at every use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Size {
    pub width: usize,
    pub height: usize,
}

impl Size {
    /// The size as the floats every coordinate below is measured in.
    #[inline]
    pub fn as_f32(self) -> (f32, f32) {
        (self.width as f32, self.height as f32)
    }

    #[inline]
    pub fn area(self) -> usize {
        self.width * self.height
    }

    /// The middle of the image, in pixels.
    #[inline]
    pub fn centre(self) -> Px {
        let (fw, fh) = self.as_f32();
        Px { x: fw / 2.0, y: fh / 2.0 }
    }

    /// Index of the rightmost column, one less than the width - the off-by-one that
    /// clamping against a size rather than against a coordinate makes.
    #[inline]
    pub fn last_column(self) -> usize {
        self.width.saturating_sub(1)
    }

    /// Index of the bottom row.
    #[inline]
    pub fn last_row(self) -> usize {
        self.height.saturating_sub(1)
    }

    /// Every pixel position, row-major.
    ///
    /// The one place a pixel *index* becomes a pixel *coordinate*, which is why the
    /// resamplers below take positions rather than indices: the conversion is the
    /// operation, and doing it once here means neither of them repeats it.
    #[inline]
    pub fn positions(self) -> impl Iterator<Item = Px> {
        (0..self.height)
            .flat_map(move |y| (0..self.width).map(move |x| Px { x: x as f32, y: y as f32 }))
    }
}

/// A point in source pixels.
///
/// Distinct from [`crate::crop::Pt`], which is normalized to the image: the only
/// route from one to the other is `Pt::to_pixels`, which forces you to name the
/// image the normalization is relative to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Px {
    pub x: f32,
    pub y: f32,
}

/// The mean brightness of a frame, on the same 0..=255 scale as a sample but not a
/// sample itself.
///
/// Its own type because the only thing it is ever compared against is another one of
/// these - the strobe's lit/dark cutoff - and on the reference camera those two numbers
/// are 5 for an unlit frame against 40-95 for a lit one. A bare `f32` there sits next
/// to a probability, a roll and a pixel coordinate.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Brightness(f32);

impl Brightness {
    /// A brightness written down rather than measured: a test, or a cutoff that is
    /// configured instead of learned.
    pub const fn level(mean: f32) -> Brightness {
        Brightness(mean)
    }

    /// Halfway between two, which is how the strobe's cutoff is learned from the feed.
    pub fn midpoint(a: Brightness, b: Brightness) -> Brightness {
        Brightness((a.0 + b.0) / 2.0)
    }

    /// How far apart two are - which is what says whether a feed holds two populations
    /// of frames or only one.
    pub fn spread(a: Brightness, b: Brightness) -> Spread {
        Spread((a.0 - b.0).abs())
    }
}

/// The difference between two brightnesses, in grey levels.
///
/// Not a [`Brightness`]: ten levels of separation between two frames is not a frame
/// whose mean is ten, and the two are never usefully compared - the same reason a
/// distance is not a position. `FINDINGS.md` reserves "contrast" for a different
/// measure of the same pixels, so this is a spread.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Spread(f32);

impl Spread {
    pub const fn levels(grey: f32) -> Spread {
        Spread(grey)
    }
}

/// 8-bit grayscale pixels, row-major, exactly `size.area()` of them.
///
/// The tag `K` says which space they are in - see [`Frame`] and [`crate::Crop`].
/// Fields are private and every constructor either checks the length or produces
/// it, because [`Frame::sample`] indexes on the size and a short buffer would be an
/// out-of-bounds panic inside a root daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Raster<K> {
    size: Size,
    px: Vec<Grey>,
    tag: PhantomData<K>,
}

/// The tag for pixels as the camera delivered them. See [`Frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Captured;

/// A whole captured image, and the only thing crops are cut from.
pub type Frame = Raster<Captured>;

impl<K> Raster<K> {
    /// `None` unless `px` is exactly `size.area()` samples - for pixels that arrive
    /// from somewhere else, like a golden file.
    pub fn new(size: Size, px: Vec<Grey>) -> Option<Raster<K>> {
        let ok = px.len() == size.area() && size.width > 0 && size.height > 0;
        ok.then_some(Raster { size, px, tag: PhantomData })
    }

    /// The same, from raw bytes - which is the one place this crate asserts that a
    /// buffer of bytes is single-channel grayscale. A v4l buffer and a PGM raster both
    /// arrive that way.
    pub fn from_bytes(size: Size, bytes: &[u8]) -> Option<Raster<K>> {
        Raster::new(size, bytes.iter().copied().map(Grey::new).collect())
    }

    /// A raster produced one pixel at a time, so its length cannot be wrong.
    ///
    /// The closure is handed each pixel's position rather than its indices, because
    /// every producer here is a resampling that immediately wants coordinates.
    pub fn from_positions(size: Size, mut f: impl FnMut(Px) -> Grey) -> Raster<K> {
        Raster { size, px: size.positions().map(&mut f).collect(), tag: PhantomData }
    }

    pub fn size(&self) -> Size {
        self.size
    }

    pub fn px(&self) -> &[Grey] {
        &self.px
    }

    /// The mean sample, which is what tells a lit frame from an unlit one.
    pub fn brightness(&self) -> Brightness {
        let total: f32 = self.px.iter().map(|p| p.value()).sum();
        Brightness(total / self.px.len() as f32)
    }
}

impl Frame {
    /// Bilinear sample at a fractional pixel position, replicating the border.
    ///
    /// OpenCV clamps each of the four taps into the image; this clamps the sample
    /// point instead, which is the same function. Outside the image both give the
    /// edge pixel, inside neither clamps at all, and the axes factorise so a point
    /// outside in only one of them behaves the same way. Doing it once means the two
    /// indices below are in range because the clamp put them there, rather than
    /// because four more comparisons did. The golden cases deliberately crop across
    /// both borders, so a mistake here fails `crops_match_opencv` rather than
    /// lurking at the edges.
    #[inline]
    pub fn sample(&self, at: Px) -> f32 {
        let x = Tap::at(at.x, self.size.last_column());
        let y = Tap::at(at.y, self.size.last_row());
        // Mix along x inside each of the two rows, then mix the two rows along y.
        let row = |row: usize| {
            let start = row * self.size.width;
            mix(self.grey(start + x.lo), self.grey(start + x.hi), x.frac)
        };
        mix(row(y.lo), row(y.hi), y.frac)
    }

    #[inline]
    fn grey(&self, i: usize) -> f32 {
        self.px[i].value()
    }

    /// The frame rotated about its centre, as `cv2.warpAffine` with
    /// `getRotationMatrix2D` produces it.
    ///
    /// Only used to *find* a face - rotating to level takes detection from 67% to
    /// 93% on frames where the mesh was failing. The crop is never cut from this; it
    /// is cut from the original frame once the eyes are mapped back, so a resampling
    /// artefact cannot reach the classifier.
    pub fn rotated(&self, by: Roll) -> Frame {
        let map = Sampling::rotation(self.size, by);
        Frame::from_positions(self.size, |at| Grey::round(self.sample(map.at_point(at))))
    }
}

/// Where an output raster's pixels come from in a source frame.
///
/// `origin` is the source position the output's *centre* maps to; `step_x` and
/// `step_y` are what one step right and one step down in the output correspond to
/// there. Two basis vectors rather than a 2x3 matrix: the same arithmetic, and it
/// says what it does - the output walks a region along that region's own axes.
///
/// Both resamplings in this crate are one of these, which is also the only place an
/// output pixel index becomes a coordinate.
pub(crate) struct Sampling {
    pub(crate) origin: Px,
    /// The output's centre, in output pixels.
    pub(crate) mid: Px,
    pub(crate) step_x: Px,
    pub(crate) step_y: Px,
}

impl Sampling {
    /// Rotation about the centre, as `cv2.getRotationMatrix2D` inverted.
    pub(crate) fn rotation(size: Size, by: Roll) -> Sampling {
        let t = by.turn();
        let mid = size.centre();
        Sampling {
            origin: mid,
            mid,
            step_x: Px { x: t.cos, y: t.sin },
            step_y: Px { x: -t.sin, y: t.cos },
        }
    }

    /// The source position an output position samples.
    ///
    /// Fractional, which is also how a detection made on a rotated frame is brought
    /// back to the frame as it arrived: the rotation's own mapping already answers
    /// "where in the original did this come from", so nothing has to be inverted a
    /// second time.
    #[inline]
    pub(crate) fn at_point(&self, at: Px) -> Px {
        let dx = at.x - self.mid.x;
        let dy = at.y - self.mid.y;
        Px {
            x: self.origin.x + dx * self.step_x.x + dy * self.step_y.x,
            y: self.origin.y + dx * self.step_x.y + dy * self.step_y.y,
        }
    }
}

/// One axis of a bilinear sample: the two pixel indices to mix along that axis, and
/// how far between them the sample point fell.
#[derive(Debug, Clone, Copy)]
struct Tap {
    lo: usize,
    hi: usize,
    frac: f32,
}

impl Tap {
    /// Where `coord` lands on an axis whose last pixel is `last`.
    ///
    /// The clamp is the border replication, and doing it here - once per axis, on
    /// the coordinate rather than on each of the four taps - is why `lo` and `hi`
    /// are known to be in range.
    #[inline]
    fn at(coord: f32, last: usize) -> Tap {
        let c = coord.clamp(0.0, last as f32);
        let lo = c.floor();
        Tap { lo: lo as usize, hi: (lo as usize + 1).min(last), frac: c - lo }
    }
}

/// `frac` of the way from `a` to `b`.
///
/// Written as two weights summing to one, which is the form OpenCV's fixed-point
/// interpolation weights come from - `a + (b - a) * frac` is the same function and
/// rounds differently in the last place, which is a grey level of disagreement with
/// the crops the model was fitted on.
#[inline]
fn mix(a: f32, b: f32, frac: f32) -> f32 {
    a * (1.0 - frac) + b * frac
}
