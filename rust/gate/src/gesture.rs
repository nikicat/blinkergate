//! Deliberate eye gestures, matched in the live verdict stream.
//!
//! A port of `bgate/gesture.py`, including its self-check cases - see
//! `mod tests` at the bottom, which is the same table.
//!
//! Four symbols. `l` and `r` are winks - YOUR left, YOUR right - scored the moment
//! the eye shuts. `Long` and `Short` are both-eye closures, scored the moment the
//! eyes REopen, because duration is what tells them apart. The two kinds compose
//! in one pattern.
//!
//! Strictness is the point - a consent signal must be hard to produce by
//! accident:
//!
//! - every event starts and ends at open eyes. Leaving a closure one eye at a
//!   time (or sliding l->r without opening) is not an event, it aborts the
//!   attempt.
//! - an event the pattern did not expect resets it, so for a winks-only pattern a
//!   natural blink still resets.
//! - the whole sequence must land within `window` seconds of its first event.
//! - a frame where nothing was classified is neutral: it neither completes nor
//!   splits a run of frames.
//!
//! Needs verdicts in capture order. The daemon is single-threaded per
//! authentication, which is what supplies that.

use std::fmt;
use std::time::{Duration, Instant};

use serde::Deserialize;

use crate::model::Probability;
use crate::side::PerEye;

/// A pattern symbol, and therefore an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sym {
    L,
    R,
    Long,
    Short,
}

impl Sym {
    pub fn parse(s: &str) -> Option<Sym> {
        match s {
            "l" => Some(Sym::L),
            "r" => Some(Sym::R),
            "long" => Some(Sym::Long),
            "short" => Some(Sym::Short),
            _ => None,
        }
    }

    /// What the dialog says to do. The user reads this, so it is words rather
    /// than the pattern's own letters.
    pub fn instruction(self) -> &'static str {
        match self {
            Sym::L => "wink LEFT",
            Sym::R => "wink RIGHT",
            Sym::Long => "close BOTH and hold",
            Sym::Short => "blink BOTH",
        }
    }
}

impl fmt::Display for Sym {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Sym::L => "l",
            Sym::R => "r",
            Sym::Long => "long",
            Sym::Short => "short",
        })
    }
}

/// What one frame did to the attempt in flight.
///
/// Not a `bool`: this is the value a privilege escalation turns on, and at the call
/// site `if m.feed(stamp, probs)` gives a reader no way to tell "the gesture was
/// performed" from "still waiting" - while a stray `!` or an inverted convention
/// would grant on every frame. `#[must_use]` for the same reason: a dropped match is
/// a gesture the user performed and the daemon ignored.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The pattern completed on this frame, and the attempt has been reset.
    Matched,
    /// Nothing conclusive - keep feeding frames until the caller's timeout.
    Waiting,
}

/// What a change of eye state means for the pattern being matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Transition {
    /// A completed event, ready to advance the pattern.
    Event(Sym),
    /// Not an event, and not a failure either: a closure beginning, or eyes opening
    /// after a wink that was already scored.
    Nothing,
    /// The eyes moved between states without opening in between. Strictness means
    /// this abandons the attempt rather than being ignored.
    Abort,
}

/// What one frame's pair of probabilities says the eyes are doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    L,
    R,
    Both,
    Open,
}

/// How far into the pattern the attempt in flight has got.
///
/// The step count and the start time are one type because neither means anything
/// without the other: a count with no start cannot be aged out of the window, and
/// a start with no count is not an attempt. As two fields they can disagree, and
/// the disagreement is silent - the start is only ever read when the count is
/// non-zero, so a stale one sits there being wrong until some later change reads
/// it.
///
/// Two invariants, held by every method below rather than by its callers:
///
/// - `done < pattern.len()`, so [`Progress::expected`] cannot be out of bounds -
///   completing the pattern resets rather than sitting at the end;
/// - `started` is `Some` exactly when `done > 0`.
#[derive(Debug, Clone)]
pub struct Progress {
    /// The sequence being demanded, in order. Never empty - [`Progress::new`]
    /// refuses that, because a gate that asks for nothing is not a gate.
    pattern: Vec<Sym>,
    /// How many of those symbols have been performed, cleanly and in order. Always
    /// less than `pattern.len()`: reaching the end is a match, and a match resets.
    done: usize,
    /// When the attempt's *first* event landed, which is what the window is measured
    /// from - not the most recent event, so a slow sequence cannot be stretched
    /// indefinitely by keeping it going. `Some` exactly when `done > 0`.
    started: Option<Instant>,
}

impl Progress {
    /// `None` for an empty pattern: a gate that asks for nothing is not a gate.
    pub fn new(pattern: Vec<Sym>) -> Option<Progress> {
        (!pattern.is_empty()).then_some(Progress { pattern, done: 0, started: None })
    }

    /// The symbol being waited for.
    pub fn expected(&self) -> Sym {
        self.pattern[self.done] // in range by the first invariant
    }

    pub fn done(&self) -> usize {
        self.done
    }

    /// When the attempt's first event landed, if there is an attempt.
    pub fn started(&self) -> Option<Instant> {
        self.started
    }

    pub fn is_idle(&self) -> bool {
        self.done == 0
    }

    /// Abandon the attempt.
    fn reset(&mut self) {
        self.done = 0;
        self.started = None;
    }

    /// Abandon an attempt whose window has passed.
    ///
    /// Returns nothing: whether it fired is observable as [`Progress::is_idle`], and
    /// a bool here would only be a second way to ask.
    fn expire(&mut self, now: Instant, window: Duration) {
        // saturating_duration_since, not `-`: subtracting Instants panics if they
        // arrive out of order, and a daemon that dies on a clock oddity is worse
        // than one that treats it as no elapsed time.
        if let Some(t) = self.started
            && now.saturating_duration_since(t) > window
        {
            self.reset();
        }
    }

    /// Record one event. [`Verdict::Matched`] completes the pattern, which also
    /// leaves this idle, so the next attempt starts from nothing.
    fn advance(&mut self, stamp: Instant, event: Sym) -> Verdict {
        if event == self.expected() {
            if self.is_idle() {
                self.started = Some(stamp);
            }
            self.done += 1;
        } else {
            // ponytail: a wrong event restarts with itself rather than a KMP
            // failure table - patterns are a few symbols, and only one with an
            // overlapping prefix (l,l,r) would ever lose a step to this.
            self.done = usize::from(event == self.pattern[0]);
            self.started = (!self.is_idle()).then_some(stamp);
        }
        if self.done == self.pattern.len() {
            self.reset();
            return Verdict::Matched;
        }
        Verdict::Waiting
    }
}

pub struct Matcher {
    /// The attempt in flight. Holds the pattern too, since one is meaningless
    /// without the other.
    progress: Progress,
    /// P(open) below which an eye counts as shut.
    thresh: Probability,
    /// How long the whole sequence may take, from its first event.
    window: Duration,
    /// A both-eye closure this long or over is `long` rather than `short`.
    long_min: Duration,
    /// What the eyes were doing on the last frame that could be classified. `None`
    /// before the first one - and a frame where nothing could be classified leaves
    /// this alone, which is what makes a lost face neutral rather than a break in a
    /// run of the same symbol.
    state: Option<State>,
    /// When the current both-eye closure began, so that its duration is known when
    /// the eyes reopen - which is the only thing separating `long` from `short`.
    /// `None` whenever the eyes are not currently shut.
    shut_at: Option<Instant>,
}

impl Matcher {
    /// `thresh` is the P(open) below which an eye counts as shut; `window` bounds
    /// the whole sequence from its first event; a closure of `long_min` seconds or
    /// more is `Long`.
    pub fn new(
        pattern: Vec<Sym>,
        thresh: Probability,
        window: Duration,
        long_min: Duration,
    ) -> Result<Self, String> {
        if pattern.is_empty() {
            return Err("pattern is empty".into());
        }
        Ok(Matcher {
            progress: Progress::new(pattern).ok_or("pattern is empty")?,
            thresh,
            window,
            long_min,
            state: None,
            shut_at: None,
        })
    }

    /// What this matcher is waiting for, and how much of it is done.
    pub fn progress(&self) -> &Progress {
        &self.progress
    }

    /// One frame's pair of probabilities, in capture order. `None` for an eye that
    /// could not be classified.
    pub fn feed(&mut self, stamp: Instant, probs: PerEye<Option<Probability>>) -> Verdict {
        let Some(state) = self.classify(probs) else {
            return Verdict::Waiting; // lost the face: neither completes nor splits
        };
        if Some(state) == self.state {
            return Verdict::Waiting; // a run of one symbol advances the pattern once
        }
        let prev = self.state.replace(state);
        self.progress.expire(stamp, self.window);
        match self.transition(prev, state, stamp) {
            Transition::Event(event) => self.progress.advance(stamp, event),
            Transition::Abort => {
                self.abandon();
                Verdict::Waiting
            }
            Transition::Nothing => Verdict::Waiting,
        }
    }

    /// What moving from `prev` to `state` means for the pattern.
    ///
    /// The order of these arms is the strictness: every event has to start and end at
    /// open eyes, so anything reached from a non-open state at all is an abort, and
    /// only the arms above that guard are events. Sliding l->r without opening in
    /// between, or leaving a closure one eye at a time, lands on the abort.
    fn transition(&mut self, prev: Option<State>, state: State, stamp: Instant) -> Transition {
        match state {
            State::Open => {
                if prev != Some(State::Both) {
                    return Transition::Nothing; // opening after a wink: already scored
                }
                // A closure is classified by duration at the moment it opens, which
                // is the only thing that separates a long one from a blink.
                let Some(shut_at) = self.shut_at.take() else {
                    return Transition::Nothing;
                };
                Transition::Event(if stamp.saturating_duration_since(shut_at) >= self.long_min {
                    Sym::Long
                } else {
                    Sym::Short
                })
            }
            _ if prev != Some(State::Open) => Transition::Abort,
            // A closure scores when it ends; a wink scores on entry, having no
            // duration to wait for.
            State::Both => {
                self.shut_at = Some(stamp);
                Transition::Nothing
            }
            State::L => Transition::Event(Sym::L),
            State::R => Transition::Event(Sym::R),
        }
    }

    /// Give up on the attempt in flight, including any closure it was timing.
    fn abandon(&mut self) {
        self.progress.reset();
        self.shut_at = None;
    }

    fn classify(&self, probs: PerEye<Option<Probability>>) -> Option<State> {
        let shut_l = probs.left? < self.thresh;
        let shut_r = probs.right? < self.thresh;
        Some(match (shut_l, shut_r) {
            (true, true) => State::Both,
            (true, false) => State::L,
            (false, true) => State::R,
            (false, false) => State::Open,
        })
    }
}

/// A challenge drawn from `alphabet`, one symbol per random byte.
///
/// `None` if the alphabet is empty or there is not a byte for every symbol -
/// silently shortening the challenge is the one failure here that would weaken the
/// gate rather than break it.
///
/// The modulo bias is deliberate and harmless: with a 2-symbol alphabet there is
/// none, and with 3 the worst case over 256 bytes is 86 draws against 85. This is a
/// liveness challenge shown to the user, not a key - what it has to be is
/// unpredictable to a recording made yesterday.
pub fn draw(alphabet: &[Sym], length: usize, randomness: &[u8]) -> Option<Vec<Sym>> {
    if alphabet.is_empty() || randomness.len() < length {
        return None;
    }
    Some(randomness[..length].iter().map(|b| alphabet[usize::from(*b) % alphabet.len()]).collect())
}

/// The pattern spelled out for a human: "wink LEFT, wink RIGHT, ...".
pub fn describe(pattern: &[Sym]) -> String {
    pattern.iter().map(|s| s.instruction()).collect::<Vec<_>>().join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Indices at which feeding the symbol stream completes the pattern - the
    /// same harness as `gesture.py`'s `_run`, so the cases below can be compared
    /// line by line with the Python ones.
    fn run(pattern: &str, symbols: &str, dt: f64, window: f64) -> Vec<usize> {
        let pat = pattern.split(',').map(|s| Sym::parse(s).unwrap()).collect();
        let window = Duration::from_secs_f64(window);
        let half = Probability::new(0.5).unwrap();
        let mut m = Matcher::new(pat, half, window, Duration::from_millis(500)).unwrap();
        // Stamps are computed rather than sampled, so the cases stay exact - a
        // frame interval of 0.133s is the reference camera's, not this machine's.
        let base = Instant::now();
        symbols
            .split(' ')
            .enumerate()
            .filter(|(i, s)| {
                // `Probability::new` returns the Option these arrays are made of,
                // so a shut eye and an unclassifiable one are written the same way.
                let (shut, open) = (Probability::new(0.1), Probability::new(0.9));
                let probs = match *s {
                    "l" => PerEye { left: shut, right: open },
                    "r" => PerEye { left: open, right: shut },
                    "b" => PerEye { left: shut, right: shut },
                    "o" => PerEye { left: open, right: open },
                    "?" => PerEye { left: None, right: None },
                    other => panic!("unknown symbol {other}"),
                };
                m.feed(base + Duration::from_secs_f64(*i as f64 * dt), probs) == Verdict::Matched
            })
            .map(|(i, _)| i)
            .collect()
    }

    fn r(pattern: &str, symbols: &str) -> Vec<usize> {
        run(pattern, symbols, 0.133, 6.0)
    }

    #[test]
    fn winks() {
        assert_eq!(r("l,r,r,l", "o l o r o r o l"), [7]);
        assert_eq!(r("l,r,r,l", "o l l l o r o r o l"), [9]); // a held wink is one
        assert!(r("l,r,r,l", "o l o r o b o r o l").is_empty()); // a blink resets
        assert_eq!(r("l,r,r,l", "b l o l o r o r o l"), [9]); // b->l is not a wink
        assert_eq!(r("l,r,r,l", "o l ? l o r o r o l"), [9]); // lost face: neutral
        assert_eq!(r("l,r", "o r o l o r"), [5]); // wrong wink restarts
        assert!(run("l,r", "o l o r", 4.0, 6.0).is_empty()); // over the window
        assert_eq!(r("l,r,r,l", "o l o r o r o l o l o r o r o l"), [7, 15]);
    }

    #[test]
    fn closures() {
        // 4 shut frames = 0.532s >= long_min, 1 = 0.133s short.
        assert_eq!(r("long,long,short", "o b b b b o b b b b o b o"), [12]);
        assert!(r("long,long,short", "o b b b b o b o b b b b o").is_empty());
        assert_eq!(r("short", "o b l o b o"), [5]); // b->l aborts, clean
        assert_eq!(r("l,short", "o l o b o"), [4]); // kinds compose
        assert_eq!(r("short,l", "o b o l"), [3]); // blink then wink
    }

    #[test]
    fn draws_a_challenge_or_nothing() {
        let alphabet = [Sym::L, Sym::R, Sym::Long];
        assert_eq!(draw(&alphabet, 3, &[0, 1, 2]), Some(vec![Sym::L, Sym::R, Sym::Long]));
        // The same bytes always draw the same challenge, which is what makes a
        // daemon's decision reproducible from its log.
        assert_eq!(draw(&alphabet, 2, &[7, 8, 9]), draw(&alphabet, 2, &[7, 8, 9]));
        // Too few bytes, or nothing to draw from: no challenge rather than a short
        // one.
        assert!(draw(&alphabet, 4, &[0, 1, 2]).is_none());
        assert!(draw(&[], 1, &[0]).is_none());
        assert_eq!(draw(&alphabet, 0, &[]), Some(vec![]));
    }

    #[test]
    fn progress_holds_its_invariants() {
        let t = Instant::now();
        let mut p = Progress::new(vec![Sym::L, Sym::R]).unwrap();
        assert!(p.is_idle() && p.started().is_none() && p.expected() == Sym::L);

        assert_eq!(p.advance(t, Sym::L), Verdict::Waiting);
        assert_eq!((p.done(), p.expected()), (1, Sym::R));
        assert_eq!(p.started(), Some(t));

        // A wrong event that is not the first symbol either clears the start with
        // the count...
        assert_eq!(p.advance(t, Sym::Long), Verdict::Waiting);
        assert!(p.is_idle() && p.started().is_none());

        // ...or restarts the attempt as its own first step.
        let later = t + Duration::from_secs(1);
        assert_eq!(p.advance(later, Sym::L), Verdict::Waiting);
        assert_eq!(p.started(), Some(later));

        // Completing resets, so the next attempt cannot inherit a start time - and
        // `expected` is never the index past the end.
        assert_eq!(p.advance(later, Sym::R), Verdict::Matched);
        assert!(p.is_idle() && p.started().is_none() && p.expected() == Sym::L);

        // An idle attempt has nothing to age out...
        p.expire(later + Duration::from_secs(99), Duration::from_secs(6));
        assert!(p.is_idle());
        // ...and a live one past its window is abandoned.
        assert_eq!(p.advance(later, Sym::L), Verdict::Waiting);
        p.expire(later + Duration::from_secs(7), Duration::from_secs(6));
        assert!(p.is_idle());

        assert!(Progress::new(vec![]).is_none());
    }

    #[test]
    fn rejects_bad_pattern() {
        assert!(Sym::parse("x").is_none());
        let half = Probability::new(0.5).unwrap();
        assert!(
            Matcher::new(vec![], half, Duration::from_secs(6), Duration::from_millis(500)).is_err()
        );
        assert!(Probability::new(1.5).is_none());
        assert!(Probability::new(f32::NAN).is_none());
    }
}
