//! Which eye, and one value per eye.
//!
//! Sides are **yours**, not the picture's. Your left eye appears on the RIGHT of
//! an unmirrored frame - measured, the left eye's ring centres at x=347 of 640
//! against 307 for the right. The dataset's `img_left` / `img_right` keys follow
//! the same convention despite reading like picture sides, and every stored label
//! is keyed by them, so they cannot be renamed.
//!
//! That is the standing hazard of this codebase, and it is why nothing here is a
//! `[T; 2]`: a pair indexed by number is a swap waiting to happen, and the swap is
//! invisible - two eyes classified the wrong way round still produce a plausible
//! stream of verdicts, just of the other eye.

use crate::crop::{Landmark, RING_L, RING_R};

/// One of the subject's eyes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Left,
    Right,
}

impl Side {
    /// Both eyes, in the order `data.SIDES` uses: yours-left first.
    pub const BOTH: [Side; 2] = [Side::Left, Side::Right];

    /// The key the dataset and the model files use. Historical and misleading, and
    /// load-bearing: renaming it would silently invert every stored label.
    pub const fn key(self) -> &'static str {
        match self {
            Side::Left => "img_left",
            Side::Right => "img_right",
        }
    }

    /// What to call it to a human, who has two of their own.
    pub const fn label(self) -> &'static str {
        match self {
            Side::Left => "YOUR LEFT",
            Side::Right => "YOUR RIGHT",
        }
    }

    /// The MediaPipe eyelid ring for this eye.
    pub const fn ring(self) -> &'static [Landmark; 16] {
        match self {
            Side::Left => &RING_L,
            Side::Right => &RING_R,
        }
    }
}

/// One value per eye, reached by [`Side`] rather than by index.
///
/// Named fields and no positional constructor, for the same reason
/// [`crate::Size`] has none: `new(a, b)` is the swap this type exists to prevent.
/// Build one with a struct literal or [`PerEye::from_fn`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PerEye<T> {
    pub left: T,
    pub right: T,
}

impl<T> PerEye<T> {
    /// One value per eye, from a closure that is told which eye it is computing.
    pub fn from_fn(mut f: impl FnMut(Side) -> T) -> PerEye<T> {
        PerEye { left: f(Side::Left), right: f(Side::Right) }
    }

    pub fn get(&self, side: Side) -> &T {
        match side {
            Side::Left => &self.left,
            Side::Right => &self.right,
        }
    }

    /// Both values, each with the side it belongs to.
    pub fn iter(&self) -> impl Iterator<Item = (Side, &T)> {
        Side::BOTH.into_iter().map(move |s| (s, self.get(s)))
    }

    pub fn map<U>(self, mut f: impl FnMut(Side, T) -> U) -> PerEye<U> {
        PerEye { left: f(Side::Left, self.left), right: f(Side::Right, self.right) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sides_stay_with_their_values() {
        // The whole point: nothing below can be written positionally, so a mapped
        // pair cannot come back the other way round.
        let p = PerEye::from_fn(|s| s.key());
        assert_eq!(p.left, "img_left");
        assert_eq!(*p.get(Side::Right), "img_right");
        let labelled = p.map(|s, key| format!("{}={key}", s.label()));
        assert_eq!(labelled.left, "YOUR LEFT=img_left");
        assert_eq!(labelled.iter().map(|(s, v)| format!("{s:?}{v}")).collect::<Vec<_>>().len(), 2);
        // The rings are the measured mapping: left eye, right of the picture.
        assert_eq!(Side::Left.ring()[0], Landmark::new(362));
        assert_eq!(Side::Right.ring()[0], Landmark::new(33));
    }
}
