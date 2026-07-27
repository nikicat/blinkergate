//! What can go wrong reading the two root-owned files this daemon trusts.
//!
//! One enum, because every caller does the same thing with the result: log it and
//! let the password prompt have the authentication. The variants are still worth
//! distinguishing - [`Error::Model`] carries a [`ModelError`] that says whether
//! you need to re-enrol or the file is corrupt, and an enrolment tool wants to
//! tell those apart.

use std::path::Path;

use blinkergate_proto::Uid;

use crate::model::ModelError;
use crate::trust::Mode;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{}: {source}", path.display())]
    Io {
        path: Box<Path>,
        #[source]
        source: std::io::Error,
    },

    #[error("{}: {source}", path.display())]
    Json {
        path: Box<Path>,
        #[source]
        source: serde_json::Error,
    },

    #[error("{}: {source}", path.display())]
    Toml {
        path: Box<Path>,
        #[source]
        source: toml::de::Error,
    },

    #[error("{}: {source}", path.display())]
    Model {
        path: Box<Path>,
        #[source]
        source: ModelError,
    },

    /// Not root-owned. Under `auth sufficient` the gesture replaces the password,
    /// so a file the user can rewrite is a path to root - set the pattern to a
    /// single natural blink and wait.
    #[error("{} is owned by uid {uid}, not root", path.display())]
    NotRootOwned { path: Box<Path>, uid: Uid },

    #[error("{} is writable by non-root (mode {mode})", path.display())]
    Writable { path: Box<Path>, mode: Mode },

    /// A key whose value is syntactically fine and operationally nonsense.
    #[error("{key}: {why}")]
    Config { key: &'static str, why: String },
}

impl Error {
    pub(crate) fn config(key: &'static str, why: impl Into<String>) -> Error {
        Error::Config { key, why: why.into() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_the_file_and_the_reason() {
        // These strings are what a user sees when authentication quietly falls
        // back to a password prompt, so they are worth pinning. `{path}` also
        // relies on thiserror rendering a `PathBuf` through `.display()`, which
        // std itself will not do.
        let e = Error::NotRootOwned {
            path: Path::new("/etc/blinkergate/config.toml").into(),
            uid: Uid::new(1000),
        };
        assert_eq!(e.to_string(), "/etc/blinkergate/config.toml is owned by uid 1000, not root");

        let e = Error::Model {
            path: Path::new("/var/lib/blinkergate/1000/model.json").into(),
            source: ModelError::LevelWidth,
        };
        assert!(e.to_string().starts_with("/var/lib/blinkergate/1000/model.json: predates"), "{e}");

        let e = Error::config("gesture.length", "9 symbols is more than a 8-symbol window");
        assert_eq!(e.to_string(), "gesture.length: 9 symbols is more than a 8-symbol window");
    }
}
