//! blinkergate: an eye gesture as the consent step for privilege escalation.
//!
//! A face match proves you are present, not that you consented - any process can
//! fire a `run0` while you happen to be sitting in front of the laptop. A
//! deliberate gesture closes that gap, which is why it only has to be deliberate,
//! not secret.
//!
//! This crate is the part that watches: crop geometry, the classifier and the
//! gesture matcher, all ported from the Python in `bgate/` which is where they
//! were measured. `pam_blinkergate.so` deliberately links none of it - see
//! `../pam`.
//!
//! Read `docs/FINDINGS.md` and `docs/PITFALLS.md` before changing anything here.
//! Most of the comments in these modules are load-bearing: they record what was
//! measured, and which plausible alternative was already tried and was wrong.
//!
//! **Every number in those comments comes from the reference camera** - a Dell
//! `Integrated_Webcam_HD` infrared sensor, 640x360 `GREY`, 15 fps delivered with
//! alternate frames unlit, one subject. The shapes of the results are likely
//! general; the constants are not, and a different sensor needs its own enrolment
//! and its own measurements. Where a comment below says a figure was measured, it
//! means measured there.

pub mod camera;
pub mod config;
pub mod crop;
pub mod daemon;
pub mod detect;
pub mod error;
pub mod frame;
pub mod gesture;
pub mod model;
pub mod side;
pub mod trust;
pub mod watch;

pub use camera::{Frames, Read};
pub use config::Config;
pub use crop::{Crop, Cut, Eye, Geometry, Landmark, Landmarks, Located, Pt, Roll, Turn};
pub use daemon::Daemon;
pub use detect::{Locator, Mesh};
pub use error::{Error, Result};
pub use frame::{Brightness, Captured, Frame, Grey, Px, Raster, Size, Spread};
pub use gesture::{Matcher, Progress, Sym, Verdict};
pub use model::{Kind, Model, ModelError, Normalise, Probability};
pub use side::{PerEye, Side};
pub use trust::Mode;
pub use watch::{Outcome, Tally, watch};
