//! Watching for one gesture.
//!
//! The daemon decides on **frames it captured itself**, and this is the loop that
//! captures them. It never sees the socket, and nothing a client sends reaches it
//! beyond the challenge to demand: a client that could assert what its eyes did would
//! hand the decision to anyone who can reach the daemon.
//!
//! What it costs, per lit frame, on the reference camera: ~61ms to locate (the mesh
//! runs twice on a rolled head), ~1ms to classify both eyes. Lit frames arrive every
//! 133ms, so there is room - but the margin is thin enough that p90 on hard frames
//! (126ms) straddles the line, and falling behind costs a whole frame rather than the
//! overshoot: the strobe hands over an unlit frame next, so the following usable one
//! is 266ms after the last.

use std::time::{Duration, Instant};

use crate::camera::{Frames, Read};
use crate::error::Result;
use crate::gesture::{Matcher, Verdict};
use crate::model::Model;
use crate::side::PerEye;

/// How one attempt at the gesture ended.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The challenge was performed. This is the only value that authenticates.
    Matched,
    /// Watched for the whole timeout and it was not.
    TimedOut,
    /// The frames stopped arriving - the device failed, or a replay ran out. Told
    /// apart from a timeout because it is a fault rather than a decision, and a fault
    /// belongs in the log.
    Ended,
}

/// What one attempt did, for the log line that follows it.
///
/// Counted rather than summarised: a session that timed out having classified two
/// frames is a camera problem, and one that timed out having classified a hundred is a
/// gesture problem. A single verdict cannot tell those apart.
///
/// Three counters and not four. Every lit frame is either located or it is not, so a
/// separate `lit` field would be a fourth number that has to agree with two others -
/// and the daemon prints all of them, where a disagreement would look like a fact.
#[derive(Debug, Clone, Copy, Default)]
pub struct Tally {
    /// Frames the strobe delivered dark, and this discarded.
    pub dark: u32,
    /// Lit frames with no face found, by anything in the locator stack.
    pub faceless: u32,
    /// Lit frames that produced a pair of probabilities.
    pub classified: u32,
}

impl Tally {
    /// Frames the strobe delivered lit, which is every frame that was looked at.
    ///
    /// A method and not a field, which is the whole point of there being three counters
    /// here rather than four: a number derived where it is read cannot disagree with the
    /// two it is derived from.
    pub fn lit(&self) -> u32 {
        self.faceless + self.classified
    }
}

impl std::fmt::Display for Tally {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "lit {} dark {} faceless {} classified {}",
            self.lit(),
            self.dark,
            self.faceless,
            self.classified
        )
    }
}

/// Watch until the pattern is matched or the deadline passes.
///
/// `locate` is the locator stack, taken as a closure so that this loop does not have
/// to own MediaPipe - the daemon keeps that warm across authentications while the
/// camera opens and closes inside each one.
pub fn watch(
    frames: &mut dyn Frames,
    model: &Model,
    matcher: &mut Matcher,
    timeout: Duration,
    mut locate: impl FnMut(&crate::frame::Frame) -> Result<Option<crate::crop::Located>>,
) -> Result<(Outcome, Tally)> {
    let deadline = Instant::now() + timeout;
    let mut tally = Tally::default();
    while Instant::now() < deadline {
        let frame = match frames.read() {
            Read::Lit(frame) => frame,
            Read::Dark => {
                tally.dark += 1;
                continue;
            }
            Read::Ended => return Ok((Outcome::Ended, tally)),
        };

        // A frame with no face is not a frame with open eyes: it feeds `None`, which
        // the matcher treats as neutral. Reporting it as "open" would break a run of
        // shut frames and quietly make a held wink unmatchable.
        let probs = match locate(&frame)? {
            None => {
                tally.faceless += 1;
                PerEye { left: None, right: None }
            }
            Some(at) => {
                tally.classified += 1;
                model.crop.cut(&frame, &at).map(|_side, crop| model.p_open(&crop))
            }
        };
        // The stamp is when the frame was *processed*, which is within a frame
        // interval of when it was captured - the queue is one deep, by configuration.
        if matcher.feed(Instant::now(), probs) == Verdict::Matched {
            return Ok((Outcome::Matched, tally));
        }
    }
    Ok((Outcome::TimedOut, tally))
}
