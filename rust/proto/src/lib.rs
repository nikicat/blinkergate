//! What the PAM module and the daemon say to each other.
//!
//! Line-oriented ASCII, one request and then a stream of replies, because the
//! module has to display a prompt *before* the verdict exists:
//!
//! ```text
//! ->  AUTH <uid> <nonce>
//! <-  PROMPT <text>            (once, immediately)
//! <-  OK <nonce> | FAIL | BUSY | ERR <text>
//! ```
//!
//! Text rather than a serialisation format, and one dependency in total - see
//! this crate's Cargo.toml for why that matters. That excludes `thiserror`: the
//! only failures here are "this line is not a request" ([`Option`]) and "the
//! CSPRNG refused" ([`getrandom::Error`]), both already typed.
//!
//! The nonce is echoed in `OK` and nowhere else. It is the whole defence against
//! one gesture satisfying a request other than the one it was performed for: two
//! authentications can genuinely overlap (polkit-agent-helper@.service is a
//! template unit), and an approval that is not bound to a request is a confused
//! deputy waiting to happen.

use std::fmt;
use std::path::Path;

/// Where the daemon listens. Under `/run` because the PAM stack runs with
/// `PrivateTmp=yes`, which rules out `/tmp`, and `ProtectHome=yes`, which rules
/// out anything under a user's home.
///
/// A function because `Path::new` is not a const fn on stable yet (rust#143874).
pub fn socket() -> &'static Path {
    Path::new("/run/blinkergate/sock")
}

/// Bytes a nonce carries, hex-encoded on the wire.
pub const NONCE_BYTES: usize = 16;

/// Nothing on this wire is long. A prompt is capped by what polkit will carry
/// (512 bytes through the PAM conversation), and every other message is a word
/// and a number - so a peer sending more than this is not a peer we want to keep
/// reading from.
pub const MAX_LINE: usize = 1024;

/// A user id.
///
/// Its own type because this protocol also carries a pid (in `SO_PEERCRED`, on
/// the daemon's side) and both are bare `u32`s upstream. The one that decides
/// *whose enrolled model is consulted* should not be interchangeable with
/// anything.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Uid(u32);

impl Uid {
    pub const ROOT: Uid = Uid(0);

    pub const fn new(uid: u32) -> Uid {
        Uid(uid)
    }
}

impl fmt::Display for Uid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

/// A single-use value binding one gesture to one authentication.
///
/// Lowercase hex, exactly [`NONCE_BYTES`] of it, enforced on the way in. That is
/// not tidiness: the daemon echoes this string back into a newline-delimited
/// reply stream, so a nonce that could contain a space or a newline could forge a
/// reply. Having no other constructor is what makes that impossible rather than
/// merely checked in the right place.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Nonce(String);

impl Nonce {
    /// A fresh nonce from the kernel's CSPRNG.
    pub fn new() -> Result<Nonce, getrandom::Error> {
        Ok(Nonce(hex(&random_bytes(NONCE_BYTES)?)))
    }

    /// Parse one off the wire, or `None` if it is not exactly what we emit.
    pub fn parse(s: &str) -> Option<Nonce> {
        let ok = s.len() == NONCE_BYTES * 2
            && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        ok.then(|| Nonce(s.to_string()))
    }
}

impl fmt::Display for Nonce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// A request, as parsed by the daemon.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// The user being authenticated, resolved by the module via `pam_get_user` -
    /// not the uid of whoever opened the socket.
    pub uid: Uid,
    pub nonce: Nonce,
}

impl Request {
    pub fn line(&self) -> String {
        format!("AUTH {} {}\n", self.uid, self.nonce)
    }

    pub fn parse(line: &str) -> Option<Self> {
        let mut it = line.trim().split(' ');
        if it.next()? != "AUTH" {
            return None;
        }
        let uid = Uid::new(it.next()?.parse().ok()?);
        let nonce = Nonce::parse(it.next()?)?;
        if it.next().is_some() {
            return None;
        }
        Some(Request { uid, nonce })
    }
}

/// A reply, as parsed by the PAM module.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reply {
    /// Show this to the user. The daemon sends it before watching, because the
    /// challenge may be randomised per request - the dialog is what tells you
    /// which gesture to perform.
    Prompt(String),
    /// Matched. Only trusted when the nonce is the one we sent.
    Ok(Nonce),
    /// Watched and did not match, or timed out.
    Fail,
    /// Another authentication holds the camera. Never queued: a queued request
    /// would display "blink now" while the daemon is not watching for it.
    Busy,
    Err(String),
}

impl Reply {
    pub fn line(&self) -> String {
        match self {
            // A prompt with an embedded newline would be two replies.
            Reply::Prompt(t) => format!("PROMPT {}\n", one_line(t)),
            Reply::Ok(nonce) => format!("OK {nonce}\n"),
            Reply::Fail => "FAIL\n".to_string(),
            Reply::Busy => "BUSY\n".to_string(),
            Reply::Err(t) => format!("ERR {}\n", one_line(t)),
        }
    }

    pub fn parse(line: &str) -> Option<Self> {
        let line = line.trim_end_matches(['\r', '\n']);
        let (head, rest) = match line.split_once(' ') {
            Some((h, r)) => (h, r),
            None => (line, ""),
        };
        match head {
            "PROMPT" => Some(Reply::Prompt(rest.to_string())),
            "OK" => Some(Reply::Ok(Nonce::parse(rest)?)),
            "FAIL" => Some(Reply::Fail),
            "BUSY" => Some(Reply::Busy),
            "ERR" => Some(Reply::Err(rest.to_string())),
            _ => None,
        }
    }
}

fn one_line(t: &str) -> String {
    t.chars().map(|c| if c == '\n' || c == '\r' { ' ' } else { c }).collect()
}

/// Bytes from the kernel's CSPRNG, via `getrandom(2)`.
///
/// The syscall rather than reading `/dev/urandom`: no file descriptor to open
/// inside a sandboxed PAM stack, no partial read to handle, and nothing that a
/// `DevicePolicy=strict` private `/dev` could take away.
pub fn random_bytes(n: usize) -> Result<Vec<u8>, getrandom::Error> {
    let mut buf = vec![0u8; n];
    getrandom::fill(&mut buf)?;
    Ok(buf)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let req = Request { uid: Uid::new(1000), nonce: Nonce::new().unwrap() };
        assert_eq!(Request::parse(&req.line()), Some(req.clone()));
        for r in [
            Reply::Prompt("blink: left, right".into()),
            Reply::Ok(req.nonce.clone()),
            Reply::Fail,
            Reply::Busy,
            Reply::Err("no model enrolled".into()),
        ] {
            assert_eq!(Reply::parse(&r.line()), Some(r));
        }
    }

    #[test]
    fn nonces_are_hex_and_only_hex() {
        assert!(Nonce::parse(&"a".repeat(NONCE_BYTES * 2)).is_some());
        assert!(Nonce::parse(&"A".repeat(NONCE_BYTES * 2)).is_none()); // we emit lower
        assert!(Nonce::parse(&"g".repeat(NONCE_BYTES * 2)).is_none());
        assert!(Nonce::parse("abcd").is_none()); // short
        assert!(Nonce::parse(&format!("{} ", "a".repeat(NONCE_BYTES * 2 - 1))).is_none());
        assert_ne!(Nonce::new().unwrap(), Nonce::new().unwrap());
    }

    #[test]
    fn rejects_junk() {
        assert!(Request::parse("AUTH 1000 nothex...").is_none());
        assert!(Request::parse("AUTH 1000").is_none());
        assert!(Request::parse("AUTH root aabb").is_none());
        assert!(Request::parse("HELLO 1000 00").is_none());
        assert!(Request::parse(&format!("AUTH 1000 {} extra", "a".repeat(32))).is_none());
        // An OK whose nonce is not one we could have issued is not an OK at all.
        assert!(Reply::parse("OK yes").is_none());
        // A prompt cannot forge a second reply, whatever the daemon puts in it.
        assert_eq!(Reply::Prompt("two\nlines".into()).line(), "PROMPT two lines\n");
    }
}
