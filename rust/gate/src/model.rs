//! The classifier: learned kernels, max-pooled, with a logistic head.
//!
//! A port of `bgate/pooled.py` inference. The pixel model is deliberately not
//! ported - it is a template matcher whose accuracy falls to 44.2% at a 4px crop
//! shift, against 98.3% for this one, and the pupil genuinely sits up to 9.9px off
//! centre because gaze moves it. A `kind = "pixel"` export loads and then fails
//! with that measurement, rather than half-working.
//!
//! Weights arrive as JSON exported by `export_model.py`, carrying the
//! preprocessing (crop geometry, normalisation, detection confidence) alongside
//! them. That pairing is not decoration: a model fitted one way is meaningless
//! applied another, and the failure is silent - it costs accuracy without
//! crashing. [`Model::load`] checks every one of them.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Deserializer, de};

use crate::crop::{Crop, Geometry};
use crate::error::{Error, Result};
use crate::frame::Size;

/// Bumped when the file's *meaning* changes, not when a field is added.
pub const FORMAT: &str = "blinkergate-model/1";

/// Why a model file cannot be used. Separate from [`Error`] so the reason
/// survives being wrapped with the path it came from, and so an enrolment tool can
/// tell "re-train this" from "this file is damaged".
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("unknown format {found:?}, expected {FORMAT:?}")]
    Format { found: String },

    #[error(
        "{found} is a template matcher: 44.2% at a 4px crop shift against 98.3% pooled, \
         and the pupil sits up to 9.9px off centre. Re-train pooled."
    )]
    Kind { found: Kind },

    #[error(
        "{found} features are absolute pixel values, so the boundary is pinned to one \
         session's illumination (98.8% -> 57.2% at 0.6x). Re-train standardised."
    )]
    Normalise { found: Normalise },

    #[error("fitted under crop convention {found:?}, this build cuts {expected:?}")]
    Convention { found: String, expected: &'static str },

    #[error("predates the levelled eye width, so every crop on a tilted head is zoomed 1.2x")]
    LevelWidth,

    /// Weights that do not line up with each other or with the crop.
    #[error("{0}")]
    Shape(String),
}

/// Which classifier the weights belong to.
///
/// `Pixel` is a variant so that loading one produces the measurement above rather
/// than "unknown variant"; anything not in this list is a corrupt file and serde
/// says so, naming the field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Pooled,
    Pixel,
}

/// What was done to a crop's pixels before they became features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Normalise {
    Standardise,
    None,
}

impl fmt::Display for Kind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Kind::Pooled => "pooled",
            Kind::Pixel => "pixel",
        })
    }
}

impl fmt::Display for Normalise {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Normalise::Standardise => "standardise",
            Normalise::None => "raw",
        })
    }
}

/// A probability, checked once where it is written down.
///
/// The classifier's output and every threshold it meets are the same kind of
/// number, and the failure of a bad one is not an exception: `thresh = 5.0` means
/// "always shut", `confidence = 50` means "never find a face". Both are configs
/// that work, quietly, wrongly.
///
/// Constructed unchecked only inside this module, where a sigmoid is the source.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Probability(f32);

impl Probability {
    pub fn new(v: f32) -> Option<Probability> {
        // NaN fails `contains`, which is the point: it would compare false
        // against every threshold and read as an eye that is merely open.
        (0.0..=1.0).contains(&v).then_some(Probability(v))
    }

    pub fn get(self) -> f32 {
        self.0
    }
}

impl fmt::Display for Probability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Three places by default - enough to read a verdict off a log line - but a
        // caller asking for `{:.6}` gets six, which a hardcoded precision silently
        // ignored while looking like it had worked.
        write!(f, "{:.*}", f.precision().unwrap_or(3), self.0)
    }
}

impl<'de> Deserialize<'de> for Probability {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let v = f32::deserialize(d)?;
        Probability::new(v).ok_or_else(|| de::Error::custom(format!("{v} is not a probability")))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Detector {
    /// `min_face_detection_confidence`. MediaPipe's default of 0.5 is far too strict
    /// on the reference camera - it found 49% of the frames that were captured
    /// *because* detection was struggling, against 74% at 0.2, with zero false
    /// positives across 39 faceless inputs (sensor noise, flat fields, real frames
    /// with the face blanked out).
    pub confidence: Probability,
}

/// Unknown fields are ignored here, unlike in [`crate::Config`]: this file is
/// generated by `export_model.py` and a future exporter may add to it, whereas a
/// config is typed by a human and a silently ignored key is a setting that did
/// not take.
#[derive(Debug, Clone, Deserialize)]
pub struct Model {
    /// What this file's fields *mean*, as [`FORMAT`]. Not the model version: a new
    /// generation of weights is an everyday event and does not touch this, while a
    /// change to what `coef` is indexed by must.
    pub format: String,
    /// Which classifier the weights belong to. Only [`Kind::Pooled`] runs here.
    pub kind: Kind,
    /// Which generation this came from, so a verdict can be traced back to a fit.
    pub generation: String,
    /// Held-out accuracy of the fit, or `None` for a model that was not scored.
    ///
    /// Scored with whole capture bursts withheld: frames from one key press are near
    /// duplicates, and splitting them across train and test measures memorisation.
    /// Carried for the record only - nothing here reads it to make a decision.
    pub accuracy: Option<f64>,
    /// How the crops this was fitted on were cut. Not a preference: crops cut another
    /// way are a template shown to a model that never saw it, and the mismatch costs
    /// accuracy without failing.
    pub crop: Geometry,
    /// How hard to look for a face - also part of the preprocessing, because the
    /// threshold decides *which* frames ever reach the classifier.
    pub detector: Detector,
    /// What was done to a crop's pixels before they became features. Only
    /// [`Normalise::Standardise`] runs here.
    pub normalise: Normalise,
    /// Side of each square kernel, in crop pixels. 11 on the reference camera, which
    /// leaves a 30x14 window of positions to pool over on a 40x24 crop.
    pub kernel_size: usize,
    /// Whether the head reads *where* each peak was as well as how strong it was.
    /// Three features per kernel instead of one, which is what [`Model::check`]
    /// verifies `coef` against.
    pub use_position: bool,
    /// `K` kernels, each `kernel_size * kernel_size`, row-major.
    ///
    /// Zero-mean and unit-norm as fitted, and not re-normalised here: zero-mean is
    /// why a flat region scores exactly 0 instead of the pooling maximum locking onto
    /// the brightest patch of cheek, and unit-norm is why a kernel cannot inflate its
    /// own feature past a scale the head does not control. A file that broke either
    /// would still run - it would just quietly mean something else.
    pub kernels: Vec<Vec<f32>>,
    /// The logistic head, one weight per feature: every kernel's peak response, then
    /// every peak's row, then every peak's column. That order is the Python feature
    /// vector's, and reading it in another order is a model that works and is wrong.
    pub coef: Vec<f32>,
    /// The head's bias. Positive means "closed" is the default reading, since the
    /// head predicts P(closed).
    pub intercept: f32,
}

impl Model {
    /// The enrolled model for one user, or `None` if that user has not enrolled.
    ///
    /// Root-owned, checked here rather than assumed of enrolment: these weights
    /// *are* the decision boundary, so a user who can rewrite them can pick one
    /// that says every frame is a wink.
    pub fn load(path: impl AsRef<Path>) -> Result<Option<Model>> {
        let path = path.as_ref();
        let Some(text) = crate::trust::read_root_owned(path)? else { return Ok(None) };
        let model: Model = serde_json::from_str(&text)
            .map_err(|source| Error::Json { path: path.into(), source })?;
        model.check().map_err(|source| Error::Model { path: path.into(), source })?;
        Ok(Some(model))
    }

    /// Everything that has to be true before these weights mean anything.
    ///
    /// Public because enrolment should refuse a bad model *before* installing it
    /// root-owned, not at the next authentication. The order is deliberate: a pixel
    /// model has the wrong coefficient count too, and "re-train pooled" is the
    /// useful half of that.
    pub fn check(&self) -> std::result::Result<(), ModelError> {
        self.check_supported()?;
        self.crop.check()?;
        self.check_shapes()?;
        self.check_finite()
    }

    /// Is this a model this build knows how to run at all?
    ///
    /// The pixel template and unstandardised features are both *loadable* and both
    /// wrong here, and each carries the measurement that says so - see
    /// [`ModelError::Kind`] and [`ModelError::Normalise`].
    fn check_supported(&self) -> std::result::Result<(), ModelError> {
        if self.format != FORMAT {
            return Err(ModelError::Format { found: self.format.clone() });
        }
        if self.kind != Kind::Pooled {
            return Err(ModelError::Kind { found: self.kind });
        }
        if self.normalise != Normalise::Standardise {
            return Err(ModelError::Normalise { found: self.normalise });
        }
        Ok(())
    }

    /// Do the weights line up with each other and with the crop they read?
    fn check_shapes(&self) -> std::result::Result<(), ModelError> {
        let shape = |why: String| Err(ModelError::Shape(why));
        // Zero would pass the per-kernel length check below - nought weights each -
        // and then underflow the pooling window.
        if self.kernel_size == 0 {
            return shape("kernel_size is 0".into());
        }
        let k = self.kernels.len();
        let d = self.kernel_size * self.kernel_size;
        if k == 0 || self.kernels.iter().any(|w| w.len() != d) {
            return shape(format!("{k} kernels do not all have {d} weights"));
        }
        if self.kernel_size > self.crop.in_w || self.kernel_size > self.crop.in_h {
            return shape(format!(
                "{}x{} kernel does not fit a {}x{} crop",
                self.kernel_size, self.kernel_size, self.crop.in_w, self.crop.in_h
            ));
        }
        // Three features per kernel with positions on: the peak, and where it was.
        let want = k * if self.use_position { 3 } else { 1 };
        if self.coef.len() != want {
            return shape(format!("{} coefficients for {want} features", self.coef.len()));
        }
        Ok(())
    }

    /// Are the values usable numbers?
    ///
    /// JSON has no NaN, but it has `1e400`. A non-finite weight makes every
    /// probability NaN, and NaN compares false against the shut threshold - so both
    /// eyes would read open forever and the gesture would simply never complete,
    /// which looks like a user who cannot wink rather than like a broken file.
    fn check_finite(&self) -> std::result::Result<(), ModelError> {
        let weights = self.kernels.iter().flatten().chain(&self.coef);
        if !weights.chain([&self.intercept]).all(|v| v.is_finite()) {
            return Err(ModelError::Shape("weights contain a non-finite value".into()));
        }
        Ok(())
    }

    /// P(open) for one crop, or `None` if the crop is smaller than the kernels.
    ///
    /// `None` rather than a number, and never a silent zero: it means "not
    /// classified", which the gesture matcher already treats as a neutral frame, and
    /// it is the convention `pooled.predict` uses for a crop that could not be cut.
    /// A crop from the geometry this model carries always fits - [`Model::check`]
    /// guarantees it - so the case exists to be honest at the boundary rather than
    /// because it is expected.
    pub fn p_open(&self, crop: &Crop) -> Option<Probability> {
        let window = self.window(crop.size())?;
        let pixels = standardise(crop);
        let peaks = self.pool(&pixels, &window);
        Some(logistic(self.score(&peaks, &window)))
    }

    /// The positions a kernel can sit at on this crop.
    ///
    /// Derived from the crop in hand rather than from the geometry the model was
    /// fitted with, so a size mismatch cannot silently index into the wrong rows.
    fn window(&self, crop: Size) -> Option<Window> {
        Some(Window {
            rows: crop.height.checked_sub(self.kernel_size - 1)?,
            cols: crop.width.checked_sub(self.kernel_size - 1)?,
            stride: crop.width,
            kernel: self.kernel_size,
        })
    }

    /// Each kernel's strongest response anywhere in the window.
    ///
    /// The maximum *is* the pooling: move the pupil and the peak moves with it,
    /// while the maximum does not change. That reduce step is the only reason this
    /// model is shift-invariant, and it is what the pixel template lacked - 44.2% at
    /// a 4px crop shift against 98.3% here.
    fn pool(&self, pixels: &[f32], window: &Window) -> Vec<Peak> {
        self.kernels.iter().map(|kernel| window.best(pixels, kernel)).collect()
    }

    /// The logistic head's input: peak responses, then where each peak sat.
    ///
    /// The position is not differentiable - it is an index - so the fit sees it as a
    /// constant. It earns its place by letting the head learn a soft prior: a blob
    /// anywhere is fine, a blob jammed against the top edge is an eyebrow.
    ///
    /// Summed in the order the Python side lays its feature vector out (all peaks,
    /// then all rows, then all columns) because f32 addition is not associative and
    /// the conformance test compares to six decimal places.
    fn score(&self, peaks: &[Peak], window: &Window) -> f32 {
        let k = peaks.len();
        let mut z = self.intercept;
        for (peak, coef) in peaks.iter().zip(&self.coef) {
            z += peak.response * coef;
        }
        if self.use_position {
            for (i, peak) in peaks.iter().enumerate() {
                let place = window.place(peak);
                z += place.row * self.coef[k + i] + place.col * self.coef[2 * k + i];
            }
        }
        z
    }
}

/// Where a peak sat, scaled to -1..=1 on each axis - the form the head was fitted
/// against, and the reason a position near an edge is a usable signal.
#[derive(Debug, Clone, Copy)]
struct Place {
    row: f32,
    col: f32,
}

/// One kernel's pooled response: how strongly it fired, and where.
#[derive(Debug, Clone, Copy)]
struct Peak {
    response: f32,
    row: usize,
    col: usize,
}

/// The positions a kernel can occupy on one crop, and how to walk them.
struct Window {
    rows: usize,
    cols: usize,
    /// Row length of the crop the window sits on, not of the window.
    stride: usize,
    kernel: usize,
}

impl Window {
    /// The kernel's response with its top-left corner at one position.
    #[inline]
    fn response(&self, pixels: &[f32], kernel: &[f32], row: usize, col: usize) -> f32 {
        let mut acc = 0.0;
        for ky in 0..self.kernel {
            let from = (row + ky) * self.stride + col;
            let weights = ky * self.kernel;
            for kx in 0..self.kernel {
                acc += pixels[from + kx] * kernel[weights + kx];
            }
        }
        acc
    }

    /// The strongest response anywhere, and where it was.
    fn best(&self, pixels: &[f32], kernel: &[f32]) -> Peak {
        let mut best = Peak { response: f32::NEG_INFINITY, row: 0, col: 0 };
        for row in 0..self.rows {
            for col in 0..self.cols {
                let response = self.response(pixels, kernel, row, col);
                // Strictly greater, and rows before columns: numpy's argmax keeps
                // the FIRST maximum, and the position is a feature.
                if response > best.response {
                    best = Peak { response, row, col };
                }
            }
        }
        best
    }

    fn place(&self, peak: &Peak) -> Place {
        let span = |i: usize, n: usize| i as f32 / (n - 1).max(1) as f32 * 2.0 - 1.0;
        Place { row: span(peak.row, self.rows), col: span(peak.col, self.cols) }
    }
}

/// A crop's pixels, standardised.
///
/// `(x - mean) / std` per crop, for the reason it exists on the Python side: raw
/// features are absolute pixel values, so the decision boundary would be pinned to
/// the illumination of whichever session was captured. Measured, that is the
/// difference between 98.8% and 57.2% under a 0.6x lighting change - and gain and
/// offset are most of what a lighting change is.
fn standardise(crop: &Crop) -> Vec<f32> {
    let px = crop.px();
    let n = px.len() as f32;
    let mean = px.iter().map(|&p| f32::from(p)).sum::<f32>() / n;
    let var = px.iter().map(|&p| (f32::from(p) - mean).powi(2)).sum::<f32>() / n;
    let scale = 1.0 / (var.sqrt() + 1e-6);
    px.iter().map(|&p| (f32::from(p) - mean) * scale).collect()
}

/// The head predicts P(closed); the pipeline speaks P(open) throughout.
///
/// Unchecked because a sigmoid of a finite `z` is in (0, 1) by construction, and
/// [`Model::check`] refused non-finite weights.
fn logistic(z: f32) -> Probability {
    Probability(1.0 - 1.0 / (1.0 + (-z.clamp(-60.0, 60.0)).exp()))
}
