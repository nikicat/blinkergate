//! Conformance against the Python implementation.
//!
//! `golden.json` is written by `uv run export_model.py --golden`: fixed synthetic
//! inputs, the crops `cv2.warpAffine` cuts from them, and the probabilities
//! `bgate.pooled` reads off those crops. Everything in it - including the model
//! weights - is synthetic, so this asserts the port's arithmetic rather than any
//! particular fit, and nothing derived from a face is committed.
//!
//! Two levels of claim, because they can fail independently:
//!
//! 1. **the crop pixels**, which prove the sampler. Not bit-exact by design:
//!    OpenCV's `remap` interpolates in fixed point (5-bit coordinate fractions)
//!    where this uses f32, so a tap weight can differ in the last place and a
//!    pixel by one grey level. The bound is asserted, not assumed.
//! 2. **the probability**, which proves standardisation, pooling and the head.
//!
//! Every crop in the file is cut by `cv2.warpAffine`, including the one at
//! `roll = 0` - `data.crop_at` branches at exactly zero onto an integer box plus
//! an `INTER_AREA` resize, which is a *different resampler*, and the live
//! pipeline never reaches it (roll comes from `atan2` and is never exactly zero).
//! That branch is pinned by [`python_box_path_is_a_different_resampler`] so the
//! claim cannot quietly stop being true.

use blinkergate::crop::{Landmark, RING_L, RING_MAX, RING_R};
use blinkergate::{Crop, Eye, Frame, Kind, Model, ModelError, Normalise, Px, Roll, Size};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    format: String,
    frame: FrameSpec,
    model: Model,
    cases: Vec<Case>,
    shaped: Vec<Shaped>,
    rings: Rings,
}

#[derive(Deserialize)]
struct Rings {
    left: Vec<Landmark>,
    right: Vec<Landmark>,
}

#[derive(Deserialize)]
struct FrameSpec {
    w: usize,
    h: usize,
    formula: String,
}

#[derive(Deserialize)]
struct Case {
    cx: f32,
    cy: f32,
    width: f32,
    roll: f32,
    crop: Vec<u8>,
    p_open: f32,
    /// Present only where `data.crop_at` would take its integer-box path.
    #[serde(default)]
    crop_box: Option<Vec<u8>>,
}

#[derive(Deserialize)]
struct Shaped {
    note: String,
    crop: Vec<u8>,
    p_open: f32,
}

fn load() -> Golden {
    let text = include_str!("golden.json");
    let golden: Golden = serde_json::from_str(text).expect("golden.json parses");
    assert_eq!(golden.format, "blinkergate-golden/1");
    // The frame is a formula on both sides; if the expression ever changes on one
    // of them, every crop below would still agree with itself.
    assert_eq!(golden.frame.formula, "((x*7 + y*13) % 251) ^ ((x*y) % 97)");
    golden
}

/// The same integer expression the exporter uses. Kept as a literal formula on
/// both sides so the committed file stays small and there is nothing to drift but
/// this line - which `golden.frame.formula` then pins.
fn synth_frame(size: Size) -> Frame {
    Frame::from_fn(size, |x, y| u8::try_from(((x * 7 + y * 13) % 251) ^ ((x * y) % 97)).unwrap())
}

fn diff(got: &[u8], want: &[u8]) -> (u32, f64) {
    assert_eq!(got.len(), want.len());
    let mut max = 0u32;
    let mut sum = 0u64;
    for (a, b) in got.iter().zip(want) {
        let d = u32::from(a.abs_diff(*b));
        max = max.max(d);
        sum += u64::from(d);
    }
    (max, sum as f64 / got.len() as f64)
}

#[test]
fn rings_are_mediapipes_own() {
    // 32 integers with no derivation next to them, and one wrong digit moves a crop
    // by a fraction of an eye - which grows smoothly and never announces itself.
    // The exporter checks them against MediaPipe's connection constants (the C API
    // carries no landmark indices, so they cannot be imported) and writes them here,
    // so the port is pinned to the same source rather than to my typing.
    let g = load();
    assert_eq!(RING_L.as_slice(), g.rings.left, "left eye ring");
    assert_eq!(RING_R.as_slice(), g.rings.right, "right eye ring");
    assert_eq!(RING_MAX, *g.rings.left.iter().chain(&g.rings.right).max().unwrap());
}

#[test]
fn crops_match_opencv() {
    let g = load();
    let frame = synth_frame(Size { width: g.frame.w, height: g.frame.h });
    for case in &g.cases {
        let eye = Eye { centre: Px { x: case.cx, y: case.cy }, width: case.width };
        let got = g.model.crop.crop_at(&frame, &eye, Roll::degrees(case.roll));
        let (max, mean) = diff(got.px(), &case.crop);
        println!(
            "roll {:>6.2}  centre ({:.1}, {:.1})  max |d| {max}  mean |d| {mean:.4}",
            case.roll, case.cx, case.cy
        );
        assert!(max <= 1, "crop differs by {max} grey levels, not fixed-point noise");
        assert!(mean <= 0.01, "crop differs by {mean:.4} on average");
    }
}

#[test]
fn python_box_path_is_a_different_resampler() {
    // Not a bug and not our path - but if this ever starts agreeing, the Python
    // branch has changed and the note in this file's header is stale. Asserting
    // the divergence is what stops that note rotting unnoticed.
    let g = load();
    let cases: Vec<&Case> = g.cases.iter().filter(|c| c.crop_box.is_some()).collect();
    assert!(!cases.is_empty(), "no case exercises data.crop_at's roll==0 branch");
    for case in cases {
        let (max, mean) = diff(case.crop_box.as_ref().unwrap(), &case.crop);
        println!("box vs warp at roll 0: max |d| {max}  mean |d| {mean:.4}");
        assert!(max > 10, "the two Python paths now agree to {max} grey levels");
    }
}

#[test]
fn probabilities_match_python() {
    let g = load();
    let frame = synth_frame(Size { width: g.frame.w, height: g.frame.h });
    let mut worst = 0.0f32;
    for case in &g.cases {
        // From OUR crop, not the golden pixels: a conformance test that fed the
        // stored crop in would pass with a broken sampler.
        let eye = Eye { centre: Px { x: case.cx, y: case.cy }, width: case.width };
        let crop = g.model.crop.crop_at(&frame, &eye, Roll::degrees(case.roll));
        let got = g.model.p_open(&crop).expect("a crop this model cut fits its own kernels");
        let d = (got.get() - case.p_open).abs();
        println!("roll {:>6.2}  python {:.6}  rust {:.6}  d {d:.6}", case.roll, case.p_open, got);
        worst = worst.max(d);
    }
    for s in &g.shaped {
        // Pixels straight from the file here: these cases exist to exercise the
        // classifier on a different pixel distribution (smooth, saturated), and
        // there is no crop geometry involved.
        let crop = Crop::new(g.model.crop.size(), s.crop.clone())
            .expect("golden crops are the model's own size");
        let got = g.model.p_open(&crop).expect("40x24 fits an 11x11 kernel");
        let d = (got.get() - s.p_open).abs();
        println!("{:<14} python {:.6}  rust {:.6}  d {d:.6}", s.note, s.p_open, got);
        worst = worst.max(d);
    }
    // The crops that reach here differ by at most one grey level, and the pooled
    // model is shift-invariant, so the probability should be tight. A gap of more
    // than this is a port bug, not resampling noise.
    assert!(worst < 1e-4, "worst probability gap {worst:.6}");
}

#[test]
fn model_rejects_a_mismatched_convention() {
    // The silent-failure guard from PITFALLS: a crop cut under an older
    // convention is the same 40x24 of plausible-looking eye, so nothing
    // downstream can notice. Only the stamp can.
    let mut g = load();
    g.model.check().expect("the golden model is usable as it stands");
    g.model.crop.convention = "roll-levelled".into();
    assert!(matches!(g.model.check(), Err(ModelError::Convention { .. })));
}

#[test]
fn model_rejects_weights_it_cannot_run() {
    // Each of these produced a wrong answer rather than an error at some point
    // while this was being written, which is the argument for checking them at
    // load rather than trusting the exporter.
    let mut g = load();
    g.model.kind = Kind::Pixel;
    assert!(matches!(g.model.check(), Err(ModelError::Kind { .. })));

    let mut g = load();
    g.model.normalise = Normalise::None;
    assert!(matches!(g.model.check(), Err(ModelError::Normalise { .. })));

    // 0 would pass "every kernel has size*size weights" and then underflow the
    // pooling window.
    let mut g = load();
    g.model.kernel_size = 0;
    g.model.kernels = vec![vec![]];
    assert!(matches!(g.model.check(), Err(ModelError::Shape(_))));

    // JSON has no NaN, but 1e400 parses as infinity - and a non-finite weight
    // makes every probability NaN, which compares false against the shut
    // threshold and reads as an eye that is merely open.
    let mut g = load();
    g.model.coef[0] = f32::INFINITY;
    assert!(matches!(g.model.check(), Err(ModelError::Shape(_))));

    let mut g = load();
    g.model.kernels[2].push(0.0);
    assert!(matches!(g.model.check(), Err(ModelError::Shape(_))));
}
