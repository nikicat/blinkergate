//! The pipeline end to end, on real frames, without a socket or a camera.
//!
//! Ignored by default because it needs three things this repository deliberately does
//! not carry - your model, your face and MediaPipe's 3.6MB bundle:
//!
//! ```sh
//! uv run export_model.py                                   # model.json
//! uv run record_replay.py --pattern l,r --out /tmp/replay   # frames
//! cd rust && BGATE_REPLAY=/tmp/replay cargo test --test replay -- --ignored --nocapture
//! ```
//!
//! What it proves is that the camera source, the mesh, the crop geometry, the
//! classifier and the matcher agree end to end: a directory of frames whose labels
//! spell `l, r` is read as the gesture `l, r`. It is **not** an accuracy measurement -
//! the frames are ones the model was fitted on, which `docs/PITFALLS.md` has a whole
//! entry about. Accuracy lives in the Python evaluation harnesses; this is plumbing.

use std::path::{Path, PathBuf};
use std::time::Duration;

use blinkergate::camera::Replay;
use blinkergate::{Locator, Matcher, Mesh, Model, Outcome, Sym, watch};

/// Where the uncommitted pieces live, relative to this crate rather than to whatever
/// directory the test was started from - cargo runs a test with the *package* as its
/// working directory, which is two levels below the repository.
fn repo(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(name)
}

fn model() -> Model {
    let path = repo("model.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e} - run `uv run export_model.py`", path.display()));
    Model::parse(&text, &path).expect("the exported model is one this build can run")
}

#[test]
#[ignore = "needs model.json, a replay directory in BGATE_REPLAY, and face_landmarker.task"]
fn a_replayed_gesture_is_matched() {
    let dir = std::env::var("BGATE_REPLAY").expect("set BGATE_REPLAY to a replay directory");
    let model = model();
    let bundle = repo("face_landmarker.task");
    let mesh = Mesh::open(&bundle, model.detector.confidence).expect("mediapipe loads");
    let mut locator = Locator::new(mesh, model.crop.clone());

    // The pattern the replay was built to spell, and the thresholds the daemon would
    // use by default.
    let mut matcher = Matcher::new(
        vec![Sym::L, Sym::R],
        blinkergate::Probability::new(0.5).unwrap(),
        Duration::from_secs(6),
        Duration::from_millis(500),
    )
    .unwrap();

    let mut frames = Replay::open(Path::new(&dir)).expect("the replay opens");
    let (outcome, tally) =
        watch(&mut frames, &model, &mut matcher, Duration::from_secs(15), |frame| {
            locator.locate(frame)
        })
        .expect("the pipeline runs");

    println!(
        "{outcome:?}  {tally}  rescues {} refits {}/{}",
        locator.rescues,
        locator.refits,
        locator.refits + locator.refits_rejected,
    );
    // Every frame in a replay is lit by construction, and every one of them is a real
    // face - a faceless frame here is the locator failing on a frame the Python side
    // located well enough to label.
    assert_eq!(tally.faceless, 0, "the locator lost a face the dataset had labelled");
    assert_eq!(outcome, Outcome::Matched, "the frames spell l,r but were not read as it");
}

/// The same frames, against a pattern they do not spell.
///
/// The half of the claim that the matching test cannot make: a gate that answers
/// `Matched` to everything would pass the test above.
#[test]
#[ignore = "needs the same local assets"]
fn a_pattern_the_frames_do_not_spell_is_not_matched() {
    let dir = std::env::var("BGATE_REPLAY").expect("set BGATE_REPLAY to a replay directory");
    let model = model();
    let mesh =
        Mesh::open(&repo("face_landmarker.task"), model.detector.confidence).expect("mediapipe");
    let mut locator = Locator::new(mesh, model.crop.clone());
    let mut matcher = Matcher::new(
        vec![Sym::R, Sym::R, Sym::L],
        blinkergate::Probability::new(0.5).unwrap(),
        Duration::from_secs(6),
        Duration::from_millis(500),
    )
    .unwrap();
    let mut frames = Replay::open(Path::new(&dir)).expect("the replay opens");
    let (outcome, _) =
        watch(&mut frames, &model, &mut matcher, Duration::from_secs(15), |f| locator.locate(f))
            .expect("the pipeline runs");
    // Ended, not TimedOut: the replay runs out first, which is itself the honest answer
    // for a recording that does not contain the gesture.
    assert_ne!(outcome, Outcome::Matched, "a gesture the frames do not contain was accepted");
}
