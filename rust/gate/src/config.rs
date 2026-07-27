//! Configuration, root-owned, with a per-user layer.
//!
//! Everything the Python CLI carried on the command line is a key here, because
//! the daemon has no command line worth speaking of - it is started by systemd.
//!
//! **Both files must be root-owned**, and [`Config::load`] refuses them
//! otherwise. Under `auth sufficient` the gesture *replaces* the password, so a
//! user-writable config is a path to root: set the pattern to a single natural
//! blink, wait, and any process that can reach the polkit stack is authenticated
//! by an accident of physiology.

use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Deserializer, de};

use blinkergate_proto::Uid;

use crate::error::{Error, Result};
use crate::gesture::{self, Sym};
use crate::model::Probability;
use crate::trust;

// Functions rather than consts: `Path::new` is not a const fn on stable yet
// (rust#143874), and a `&str` that everything immediately wraps in `Path::new` is
// a path with the type filed off.
/// System-wide defaults, read first.
pub fn system_config() -> &'static Path {
    Path::new("/etc/blinkergate/config.toml")
}

/// The longest random challenge worth asking for: at ~7.5 usable frames a second
/// each symbol needs the eyes open either side of it, so more than this does not
/// fit the window a person will tolerate.
pub const MAX_LENGTH: usize = 8;

/// Longer than this is not a wait, it is a camera left running.
const MAX_TIMEOUT: Duration = Duration::from_secs(300);

/// Root-owned per-user state: `<uid>/model.json` and `<uid>/config.toml`.
pub fn state_dir() -> &'static Path {
    Path::new("/var/lib/blinkergate")
}

/// The daemon's whole configuration, as the two files supply it.
///
/// Every field has a working default, so an absent file is not an error and the
/// per-user layer only has to name what it changes.
///
/// **Every duration here is written in seconds** - `timeout = 15`, `long_min = 0.25`.
/// That is what the `secs` attribute on those fields is for: `Duration`'s own serde
/// form is `{ secs, nanos }`, which nobody is typing into a config file.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Config {
    /// The **infrared** camera, not the colour one. `v4l2-ctl --list-devices`
    /// lists them; the IR sensor is usually the one advertising `GREY`.
    pub device: PathBuf,
    /// Subtract the unlit strobe frame from the lit one. Implemented and
    /// **unmeasured** on the reference camera: a quick check moved brightness 94 -> 31
    /// but contrast 15.3 -> 11.9, i.e. slightly down. Where it should help is
    /// non-uniform ambient - shadows, a window reflection.
    pub subtract: bool,
    /// How long to watch before giving up and letting the password prompt have it.
    #[serde(deserialize_with = "secs")]
    pub timeout: Duration,
    /// `[gesture]`: what to demand, and what counts as having performed it.
    pub gesture: GestureConfig,
    /// `[model]`: where enrolment put the weights and the landmarker bundle.
    pub model: ModelConfig,
    /// `[detector]`: overrides for what the model file already records about
    /// finding a face. Normally empty.
    pub detector: DetectorConfig,
    /// `[limit]`: what happens after repeated failures, so a photograph cannot be
    /// retried all night.
    pub limit: LimitConfig,
    /// `[replay]`: read frames from disk instead of the camera. Absent normally.
    pub replay: ReplayConfig,
}

/// The `[replay]` section: frames from a directory instead of a camera.
///
/// For the end-to-end test. A virtual machine has no `/dev/video*`, and a gesture is
/// the one thing a synthetic frame cannot contain - so record a real one on the host
/// and replay it wherever the PAM stack is being exercised.
///
/// Its own section rather than a magic `device` value, because a device path and a
/// directory of recordings are different things and a daemon that silently accepted
/// either would be one typo away from authenticating against yesterday's frames.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ReplayConfig {
    /// When set, the camera is not opened at all.
    ///
    /// Deliberately *not* checked for ownership, unlike the model and the config: the
    /// frames are recorded by hand on the host and replayed in the e2e guest, so they
    /// are the user's own by construction and [`read_root_owned`] would refuse the one
    /// case this exists for. What keeps that from being a way in is the config itself,
    /// which only root can write - a real gate never has this set, and a recording is
    /// evidence about the plumbing rather than about a face.
    ///
    /// [`read_root_owned`]: crate::trust::read_root_owned
    ///
    /// The rate is *not* configured here - it is in the recording's own `replay.toml`,
    /// because replaying frames faster than they were recorded changes a held closure
    /// into a blink.
    pub dir: Option<PathBuf>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            device: "/dev/video2".into(),
            subtract: false,
            timeout: Duration::from_secs(15),
            gesture: GestureConfig::default(),
            model: ModelConfig::default(),
            detector: DetectorConfig::default(),
            limit: LimitConfig::default(),
            replay: ReplayConfig::default(),
        }
    }
}

/// The `[gesture]` section: the challenge, and the thresholds that decide whether
/// it happened.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct GestureConfig {
    /// A list of symbols, or the word `random`.
    ///
    /// Randomised is recommended: because the dialog *tells the user what to
    /// perform*, the daemon can pick the sequence per request, and a pre-recorded
    /// video of a fixed gesture then fails. That is cheap liveness a static
    /// pattern cannot provide.
    pub pattern: Pattern,
    /// Symbols drawn from, when `pattern = "random"`. Winks only by default:
    /// `short` is indistinguishable from a natural blink, so a challenge
    /// containing it is easier to satisfy by accident.
    pub alphabet: Vec<Sym>,
    /// How many symbols a random challenge asks for. Every symbol costs a wink
    /// and about a second of the window, so this is not free to raise.
    pub length: NonZeroUsize,
    /// How long the whole sequence may take, measured from its first event. A
    /// frame where nothing could be classified does not reset the attempt, so
    /// this is also what bounds a long loss of the face.
    #[serde(deserialize_with = "secs")]
    pub window: Duration,
    /// A both-eye closure this long or over is `long` rather than `short`.
    /// Closures are classified when the eyes REopen, because duration is the only
    /// thing that separates the two.
    #[serde(deserialize_with = "secs")]
    pub long_min: Duration,
    /// P(open) below this counts as an eye being shut. Not a tuning knob in the
    /// usual sense - the classifier is enrolled, so 0.5 is the fitted boundary;
    /// raising it makes the gate quicker to believe an eye is closed.
    pub thresh: Probability,
}

impl Default for GestureConfig {
    fn default() -> Self {
        GestureConfig {
            pattern: Pattern::Random,
            alphabet: vec![Sym::L, Sym::R],
            length: NonZeroUsize::new(3).expect("3 is not zero"),
            window: Duration::from_secs(6),
            long_min: Duration::from_millis(500),
            thresh: Probability::new(0.5).expect("0.5 is a probability"),
        }
    }
}

impl GestureConfig {
    /// Symbols, the zero length and the probability range are serde's, so what is
    /// left are the two nonsense values that are still well-typed.
    fn check(&self) -> Result<()> {
        match &self.pattern {
            Pattern::Fixed(syms) if syms.is_empty() => {
                Err(Error::config("gesture.pattern", "is empty"))
            }
            Pattern::Fixed(_) => Ok(()),
            Pattern::Random if self.alphabet.is_empty() => {
                Err(Error::config("gesture.alphabet", "is empty"))
            }
            Pattern::Random if self.length.get() > MAX_LENGTH => Err(Error::config(
                "gesture.length",
                format!("{} symbols is more than a person will hold still for", self.length),
            )),
            Pattern::Random => Ok(()),
        }
    }
}

/// What to ask for: a list of symbols, or the word `random`.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// This sequence, every time. Simple to remember, and replayable by anyone
    /// holding a recording of you performing it.
    Fixed(Vec<Sym>),
    /// A fresh sequence per request, drawn from `alphabet` and shown in the
    /// dialog.
    Random,
}

// Hand-written rather than `#[serde(untagged)]`, which reports every mistake as
// "data did not match any variant of untagged enum Pattern" - true, and no help
// at all to someone editing a config file. This way a bad symbol comes back as
// serde's own "unknown variant `blink`, expected one of `l`, `r`, `long`,
// `short`", and a bad keyword says what the alternative is.
impl<'de> Deserialize<'de> for Pattern {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;

        impl<'de> de::Visitor<'de> for V {
            type Value = Pattern;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(r#"the word "random", or a list like ["l", "r", "long", "short"]"#)
            }

            fn visit_str<E: de::Error>(self, s: &str) -> std::result::Result<Pattern, E> {
                match s {
                    "random" => Ok(Pattern::Random),
                    other => Err(E::custom(format!(
                        r#"{other:?} is not "random"; a fixed pattern is a list like ["l", "r"]"#
                    ))),
                }
            }

            fn visit_seq<A: de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Pattern, A::Error> {
                let mut out = Vec::new();
                while let Some(sym) = seq.next_element::<Sym>()? {
                    out.push(sym);
                }
                Ok(Pattern::Fixed(out))
            }
        }

        d.deserialize_any(V)
    }
}

/// The `[model]` section: where enrolment left things.
///
/// Both paths are read by a root daemon on behalf of an authentication, so both
/// must be root-owned - enrolment installs them that way, and a user-writable
/// model is a user-chosen decision boundary.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ModelConfig {
    /// Root of the per-user weights: `<dir>/<uid>/model.json`, one enrolment per
    /// uid, because a model fitted on your eyes says nothing about anyone else's.
    pub dir: PathBuf,
    /// The MediaPipe landmarker bundle. Not under `/home`: the PAM stack runs
    /// with `ProtectHome=yes`, and while the daemon itself is not sandboxed,
    /// keeping every asset out of `/home` also survives a late-mounted home.
    pub face_model: PathBuf,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            dir: state_dir().to_path_buf(),
            face_model: "/usr/lib/blinkergate/face_landmarker.task".into(),
        }
    }
}

/// The `[detector]` section: how hard to look for a face.
///
/// Deliberately thin. Detection settings belong *in the model file*, because they
/// are part of the preprocessing the weights were fitted with; this exists for
/// re-measuring, not for tuning.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DetectorConfig {
    /// Overrides the confidence the model file carries. Normally absent: the
    /// model records the preprocessing it was fitted with, and this exists for
    /// the case where a camera has moved and you are re-measuring.
    pub confidence: Option<Probability>,
}

/// The `[limit]` section: how much retrying the gesture path allows.
///
/// Being refused here costs the user a password prompt and nothing else, which is
/// why it can afford to be strict.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LimitConfig {
    /// Consecutive failures for one user before the gesture path is refused, so a
    /// photograph or a screen cannot simply be presented until it works.
    pub failures: u32,
    /// How long that refusal lasts, counted from the last failure. During it every
    /// request answers immediately and the camera is never opened.
    #[serde(deserialize_with = "secs")]
    pub cooldown: Duration,
}

impl Default for LimitConfig {
    fn default() -> Self {
        LimitConfig { failures: 3, cooldown: Duration::from_secs(60) }
    }
}

/// A duration written as seconds, which is how a human writes one.
///
/// `pub(crate)` because a replay's manifest declares its frame interval the same way.
///
/// `Duration`'s own serde form is `{ secs, nanos }`; nobody is typing that into a
/// config file. Hand-written for the same reason [`Pattern`]'s is: an integer stays
/// an integer (`Duration::from_secs`, exact), a float is checked by
/// `try_from_secs_f64`, and each way of being wrong says so in its own words.
pub(crate) fn secs<'de, D: Deserializer<'de>>(d: D) -> std::result::Result<Duration, D::Error> {
    struct V;

    impl de::Visitor<'_> for V {
        type Value = Duration;

        fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("a number of seconds")
        }

        fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Duration, E> {
            Ok(Duration::from_secs(v))
        }

        fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Duration, E> {
            u64::try_from(v)
                .map(Duration::from_secs)
                .map_err(|_| E::custom(format!("{v} seconds is negative")))
        }

        fn visit_f64<E: de::Error>(self, v: f64) -> std::result::Result<Duration, E> {
            Duration::try_from_secs_f64(v).map_err(E::custom)
        }
    }

    d.deserialize_any(V)
}

impl Config {
    /// System defaults, then the per-user file layered on top of them.
    ///
    /// Missing files are not an error - the built-in defaults are a working
    /// configuration. An unreadable or user-writable one is.
    pub fn load(uid: Uid) -> Result<Config> {
        let user = Self::user_config(uid);
        let mut merged = toml::Table::new();
        for path in [system_config(), user.as_path()] {
            if let Some(layer) = read_layer(path)? {
                merge(&mut merged, layer);
            }
        }
        // Cannot fail once both layers have type-checked, and is where a merge bug
        // would surface if one ever did.
        let cfg: Config = toml::Table::try_into(merged)
            .map_err(|source| Error::Toml { path: user.into_boxed_path(), source })?;
        cfg.check()?;
        Ok(cfg)
    }

    /// Where this user's own overrides live, if any.
    pub fn user_config(uid: Uid) -> PathBuf {
        state_dir().join(uid.to_string()).join("config.toml")
    }

    /// Where this user's enrolled weights live.
    pub fn model_path(&self, uid: Uid) -> PathBuf {
        self.model.dir.join(uid.to_string()).join("model.json")
    }

    /// The values that are well-typed and still not workable.
    fn check(&self) -> Result<()> {
        // Negative and NaN cannot get here - `secs` rejected them - so what is left
        // is zero, which never watches, and absurdly long, which holds the camera.
        if self.timeout.is_zero() || self.timeout > MAX_TIMEOUT {
            return Err(Error::config(
                "timeout",
                format!("{:?} is not a workable wait", self.timeout),
            ));
        }
        self.gesture.check()
    }

    /// The sequence to demand of this request.
    ///
    /// Randomised patterns are drawn per request, so the same challenge is not
    /// replayable; `randomness` supplies the bytes (see `crate::rand`).
    pub fn challenge(&self, randomness: &[u8]) -> Result<Vec<Sym>> {
        let g = &self.gesture;
        match &g.pattern {
            Pattern::Fixed(syms) => Ok(syms.clone()),
            Pattern::Random => {
                gesture::draw(&g.alphabet, g.length.get(), randomness).ok_or_else(|| {
                    Error::config(
                        "gesture.length",
                        format!(
                            "{} random bytes for a {}-symbol challenge",
                            randomness.len(),
                            g.length
                        ),
                    )
                })
            }
        }
    }
}

/// One configuration layer: root-owned, parsed, and type-checked on its own.
///
/// Checked before merging so that an unknown key or a string where a number belongs
/// names *the file that holds it*. Every field has a default, so a layer that sets
/// one key deserialises fine on its own.
fn read_layer(path: &Path) -> Result<Option<toml::Table>> {
    let Some(text) = trust::read_root_owned(path)? else { return Ok(None) };
    let toml = |source| Error::Toml { path: path.into(), source };
    let layer: toml::Table = text.parse().map_err(toml)?;
    let _: Config = toml::Table::try_into(layer.clone()).map_err(toml)?;
    Ok(Some(layer))
}

/// Table-wise overlay, so a per-user file can set one key without restating the
/// section it lives in.
fn merge(base: &mut toml::Table, over: toml::Table) {
    for (k, v) in over {
        match (base.get_mut(&k), v) {
            (Some(toml::Value::Table(b)), toml::Value::Table(o)) => merge(b, o),
            (_, v) => {
                base.insert(k, v);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        Config::default().check().unwrap();
    }

    #[test]
    fn user_layer_overrides_one_key() {
        let mut base: toml::Table =
            "device = \"/dev/video2\"\n[gesture]\nwindow = 6.0\nthresh = 0.5\n".parse().unwrap();
        merge(&mut base, "[gesture]\nthresh = 0.4\n".parse().unwrap());
        let cfg: Config = toml::Table::try_into(base).unwrap();
        assert_eq!(cfg.gesture.thresh.get(), 0.4);
        assert_eq!(cfg.gesture.window, Duration::from_secs(6)); // the sibling survived
        assert_eq!(cfg.device, Path::new("/dev/video2"));
    }

    #[test]
    fn seconds_parse_as_written_and_nonsense_does_not() {
        let cfg: Config = toml::from_str("timeout = 8\n[gesture]\nlong_min = 0.25\n").unwrap();
        assert_eq!(cfg.timeout, Duration::from_secs(8)); // an integer, as typed
        assert_eq!(cfg.gesture.long_min, Duration::from_millis(250));
        assert!(toml::from_str::<Config>("timeout = -1.0").is_err());
        assert!(toml::from_str::<Config>("timeout = \"soon\"").is_err());
        let zero: Config = toml::from_str("timeout = 0.0").unwrap();
        assert!(zero.check().is_err());
    }

    #[test]
    fn random_challenge_is_drawn_from_the_alphabet() {
        let cfg = Config::default();
        let syms = cfg.challenge(&[0, 1, 2, 3, 4]).unwrap();
        assert_eq!(syms.len(), cfg.gesture.length.get());
        assert!(syms.iter().all(|s| matches!(s, Sym::L | Sym::R)));
        // Different bytes, different challenge - the point of randomising.
        assert_ne!(syms, cfg.challenge(&[1, 0, 1, 0, 1]).unwrap());
        // Too few bytes would silently shorten the challenge, the one failure
        // here that weakens the gate rather than breaking it.
        assert!(cfg.challenge(&[0, 1]).is_err());
    }

    #[test]
    fn fixed_pattern_parses_and_bad_symbols_do_not() {
        let cfg: Config = toml::from_str("[gesture]\npattern = [\"l\", \"long\"]\n").unwrap();
        cfg.check().unwrap();
        assert_eq!(cfg.challenge(&[]).unwrap(), vec![Sym::L, Sym::Long]);
        // Symbols and the "random" keyword are types now, so these fail in serde
        // naming gesture.pattern rather than in a check further down.
        assert!(toml::from_str::<Config>("[gesture]\npattern = [\"blink\"]\n").is_err());
        assert!(toml::from_str::<Config>("[gesture]\npattern = \"fixed\"\n").is_err());
        assert!(toml::from_str::<Config>("[gesture]\nlength = 0\n").is_err());
        assert!(toml::from_str::<Config>("[gesture]\nthresh = 1.5\n").is_err());
        assert!(toml::from_str::<Config>("[detector]\nconfidence = -0.1\n").is_err());
        let empty: Config = toml::from_str("[gesture]\npattern = []\n").unwrap();
        assert!(empty.check().is_err());
    }
}
