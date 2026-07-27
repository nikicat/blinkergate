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

use crate::crop::Roll;

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

/// 8-bit grayscale pixels, row-major, exactly `size.area()` of them.
///
/// The tag `K` says which space they are in - see [`Frame`] and [`crate::Crop`].
/// Fields are private and every constructor either checks the length or produces
/// it, because [`Frame::sample`] indexes on the size and a short buffer would be an
/// out-of-bounds panic inside a root daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Raster<K> {
    size: Size,
    px: Vec<u8>,
    tag: PhantomData<K>,
}

/// The tag for pixels as the camera delivered them. See [`Frame`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Captured;

/// A whole captured image, and the only thing crops are cut from.
pub type Frame = Raster<Captured>;

impl<K> Raster<K> {
    /// `None` unless `px` is exactly `size.area()` bytes - for pixels that arrive
    /// from somewhere else, like a camera buffer or a golden file.
    pub fn new(size: Size, px: Vec<u8>) -> Option<Raster<K>> {
        let ok = px.len() == size.area() && size.width > 0 && size.height > 0;
        ok.then_some(Raster { size, px, tag: PhantomData })
    }

    /// A raster produced pixel by pixel, so its length cannot be wrong.
    pub fn from_fn(size: Size, mut f: impl FnMut(usize, usize) -> u8) -> Raster<K> {
        let mut px = Vec::with_capacity(size.area());
        for y in 0..size.height {
            for x in 0..size.width {
                px.push(f(x, y));
            }
        }
        Raster { size, px, tag: PhantomData }
    }

    pub fn size(&self) -> Size {
        self.size
    }

    pub fn px(&self) -> &[u8] {
        &self.px
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
        f32::from(self.px[i])
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
        Frame::from_fn(self.size, |x, y| round_u8(self.sample(map.at(x, y))))
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

    /// The source position that output pixel `(x, y)` samples.
    #[inline]
    pub(crate) fn at(&self, x: usize, y: usize) -> Px {
        let dx = x as f32 - self.mid.x;
        let dy = y as f32 - self.mid.y;
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

/// OpenCV's `saturate_cast<uchar>`: round half away from zero, then clamp.
#[inline]
pub fn round_u8(v: f32) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}
