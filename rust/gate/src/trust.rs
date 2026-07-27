//! Files a root daemon is willing to believe.
//!
//! Under `auth sufficient` the gesture *replaces* the password, so every asset the
//! gate reads is a path to root if the user can rewrite it: a config asking for one
//! natural blink, or a model whose decision boundary the user chose. Both go
//! through [`read_root_owned`], which is the one place that rule is written down.

use std::fmt;
use std::path::Path;

use blinkergate_proto::Uid;

use crate::error::{Error, Result};

/// A file mode, as `stat(2)` reports it.
///
/// Its own type because the interesting question is never the number - it is
/// "could anyone but root have written this" - and because a mode printed in
/// decimal (`420`) is unreadable where `0644` is not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mode(u32);

impl Mode {
    pub const fn new(bits: u32) -> Mode {
        Mode(bits)
    }

    /// Writable by the group or by everyone. The ownership check covers the owner.
    pub const fn writable_by_others(self) -> bool {
        self.0 & 0o022 != 0
    }
}

impl fmt::Display for Mode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04o}", self.0 & 0o7777)
    }
}

/// The contents of a file that only root can have written, or `None` if it is not
/// there at all.
///
/// `symlink_metadata`, not `metadata`: a symlink reports its own ownership, which
/// is the user's, so this refuses one instead of following it. A user who can write
/// the containing directory cannot create a root-owned file - but they can create a
/// link to one, and have its contents believed.
pub fn read_root_owned(path: &Path) -> Result<Option<String>> {
    use std::os::unix::fs::MetadataExt;

    let io = |source| Error::Io { path: path.into(), source };
    let meta = match std::fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(io(e)),
    };
    let owner = Uid::new(meta.uid());
    if owner != Uid::ROOT {
        return Err(Error::NotRootOwned { path: path.into(), uid: owner });
    }
    let mode = Mode::new(meta.mode());
    if mode.writable_by_others() {
        return Err(Error::Writable { path: path.into(), mode });
    }
    Ok(Some(std::fs::read_to_string(path).map_err(io)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modes_read_as_octal_and_answer_the_only_question() {
        assert_eq!(Mode::new(0o644).to_string(), "0644");
        assert_eq!(Mode::new(0o100644).to_string(), "0644"); // the file-type bits
        assert!(!Mode::new(0o644).writable_by_others());
        assert!(!Mode::new(0o600).writable_by_others());
        assert!(Mode::new(0o664).writable_by_others()); // group
        assert!(Mode::new(0o646).writable_by_others()); // world
    }

    #[test]
    fn an_absent_file_is_not_an_error_but_our_own_is_not_trusted() {
        assert!(
            read_root_owned(Path::new("/nonexistent/blinkergate/config.toml")).unwrap().is_none()
        );
        // Written by this test, so owned by whoever runs it - and refused for that
        // reason, unless the suite is running as root.
        let mine = std::env::temp_dir().join("blinkergate-trust-test.toml");
        std::fs::write(&mine, "device = \"/dev/video2\"\n").unwrap();
        let got = read_root_owned(&mine);
        let _ = std::fs::remove_file(&mine);
        match got {
            Err(Error::NotRootOwned { .. }) => {}
            Ok(_) if Uid::new(unsafe_geteuid()) == Uid::ROOT => {}
            other => panic!("expected a refusal, got {other:?}"),
        }
    }

    /// Only the test needs this, and only to explain itself when run as root.
    fn unsafe_geteuid() -> u32 {
        std::fs::metadata("/proc/self")
            .map(|m| std::os::unix::fs::MetadataExt::uid(&m))
            .unwrap_or(u32::MAX)
    }
}
