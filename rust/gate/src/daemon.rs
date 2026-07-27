//! The daemon: one socket, one authentication at a time.
//!
//! ## Exactly one
//!
//! Two authentications genuinely overlap - `polkit-agent-helper@.service` is a
//! template unit, so each one gets its own PAM stack, and two `run0` calls or a GUI
//! prompt alongside a terminal one will collide.
//!
//! **The dangerous handling is to multiplex**: one camera session with two waiters,
//! where a match wakes both. That is the confused deputy in its purest form - you
//! blink to approve `run0 ls` and simultaneously approve an attacker's `run0 rm -rf
//! /`. One gesture must satisfy exactly one request, so the daemon holds a single
//! in-flight slot: a second request is answered [`Reply::Busy`] at once and its caller
//! falls through to the password.
//!
//! Not queued. A queued request would display "blink now" while the daemon is not
//! watching for it, and a dialog that lies about what it is waiting for is worse than
//! a password prompt.
//!
//! Denial of the gesture path - an attacker deliberately holding the slot - is
//! accepted: it degrades to typing a password, which is exactly today's behaviour.
//!
//! ## What stops a user asking directly
//!
//! The socket is root-owned and 0600 inside a 0700 directory, so every request
//! necessarily comes from a real PAM stack, which means a visible dialog. A user-level
//! attacker cannot ask the daemon; they must go through polkit and put a prompt on
//! your screen. [`Daemon::serve`] verifies the peer with `SO_PEERCRED` anyway, since
//! the cost is one syscall and the alternative is trusting a mode bit.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use blinkergate_proto::{self as proto, MAX_LINE, Reply, Request, Uid};

use crate::camera::{Camera, Replay};
use crate::config::Config;
use crate::detect::{Locator, Mesh};
use crate::error::{Error, Result};
use crate::gesture::{self, Matcher};
use crate::model::Model;
use crate::watch::{Outcome, watch};

/// Everything one authentication needs, and the lock that makes it one at a time.
///
/// The mesh lives here because it is expensive to build and cheap to keep: the model
/// and the MediaPipe graph stay warm between requests, and only the camera opens and
/// closes inside each one. So there is no always-on infrared camera, and no
/// hundred-millisecond model load in the middle of a login either.
pub struct Session {
    mesh: Option<Mesh>,
    /// Where each user stands with the rate limit. Absent means a clean slate.
    failures: HashMap<Uid, Failures>,
}

/// One user's run of consecutive failures.
///
/// The count and the stamp travel together because neither decides anything alone: the
/// count is what starts a cooldown and the stamp is what ends it, so a count without a
/// stamp would refuse forever.
#[derive(Debug, Clone, Copy)]
struct Failures {
    count: u32,
    /// When the most recent one was, which is what the cooldown is measured from - so a
    /// user who keeps trying keeps pushing it out.
    last: Instant,
}

/// The daemon's state, shared by every connection thread.
pub struct Daemon {
    config: Config,
    /// The single in-flight slot. Held from accept to reply, bounded by the timeout,
    /// with the camera opened and closed inside it.
    slot: Mutex<Session>,
}

impl Daemon {
    pub fn new(config: Config) -> Daemon {
        Daemon { config, slot: Mutex::new(Session { mesh: None, failures: HashMap::new() }) }
    }

    /// Bind the socket, root-only, and answer until killed.
    ///
    /// Also accepts a socket systemd already opened - `LISTEN_FDS` - which is how the
    /// unit avoids a window at boot where the path does not exist yet.
    pub fn serve(self, socket: &Path) -> Result<()> {
        let listener = bind(socket)?;
        // Announced here, after the bind, and not by the caller: a line claiming to be
        // listening while the socket is still a plan is a diagnostic that lies, and it
        // would print immediately before the error explaining that it could not.
        match listener.local_addr().ok().and_then(|a| a.as_pathname().map(Path::to_path_buf)) {
            Some(path) => log(&format!("listening on {}", path.display())),
            // An inherited socket has no pathname if systemd passed an abstract one.
            None => log("listening on the socket systemd passed"),
        }
        let daemon = Arc::new(self);
        notify_ready();
        for stream in listener.incoming() {
            let stream = match stream {
                Ok(s) => s,
                // One refused connection is not a reason to stop answering.
                Err(e) => {
                    log(&format!("accept failed: {e}"));
                    continue;
                }
            };
            let daemon = Arc::clone(&daemon);
            // A thread per connection, so that a second request can be told it is busy
            // *while* the first is watching. Serving them one at a time would leave the
            // second waiting in the backlog and then prompt for a gesture nobody is
            // expecting any more.
            std::thread::spawn(move || {
                if let Err(e) = daemon.answer(stream) {
                    log(&format!("{e}"));
                }
            });
        }
        Ok(())
    }

    /// One connection: read the request, reply, close.
    fn answer(&self, stream: UnixStream) -> Result<()> {
        // The mode bits should already make this impossible; a check that costs one
        // syscall is worth more than the assumption. `std`'s own `peer_cred` is still
        // unstable (rust#42839), so this is rustix's wrapper around the same
        // `SO_PEERCRED` - a getsockopt whose failure mode, written by hand, is reading
        // something else as a uid inside a root daemon.
        let peer = rustix::net::sockopt::socket_peercred(&stream)
            .map_err(|e| Error::Io { path: Path::new("<socket>").into(), source: e.into() })?;
        if Uid::new(peer.uid.as_raw()) != Uid::ROOT {
            return say(&stream, &Reply::Err("only root may ask".into()));
        }
        let mut reader = BufReader::new(
            stream
                .try_clone()
                .map_err(|source| Error::Io { path: Path::new("<socket>").into(), source })?,
        );
        let mut line = String::new();
        // Bounded: every message here is a word and a number, so a peer sending more
        // is not a peer worth reading further.
        reader
            .by_ref()
            .take(MAX_LINE as u64)
            .read_line(&mut line)
            .map_err(|source| Error::Io { path: Path::new("<socket>").into(), source })?;
        let Some(request) = Request::parse(&line) else {
            return say(&stream, &Reply::Err("not a request".into()));
        };

        let Ok(mut session) = self.slot.try_lock() else {
            // Busy, immediately. The caller falls through to the password rather than
            // waiting for a gesture the daemon is not watching for.
            return say(&stream, &Reply::Busy);
        };
        self.authenticate(&mut session, &stream, &request)
    }

    /// The slot is held for all of this.
    fn authenticate(
        &self,
        session: &mut Session,
        stream: &UnixStream,
        request: &Request,
    ) -> Result<()> {
        let uid = request.uid;
        if let Some(left) = self.cooling_off(session, uid) {
            // Answered without opening the camera, so a photograph cannot be presented
            // until it works.
            log(&format!("uid {uid}: refused, {left:?} of cooldown left"));
            return say(stream, &Reply::Err("too many failures".into()));
        }

        // Per-user configuration and per-user weights, both root-owned. Read per
        // request rather than cached, so re-enrolment takes effect without a restart -
        // and so a model that has become user-writable is caught now, not at load.
        let config = Config::load(uid)?;
        let path = config.model_path(uid);
        let Some(model) = Model::load(&path)? else {
            log(&format!("uid {uid}: no model at {}", path.display()));
            return say(stream, &Reply::Err("not enrolled".into()));
        };

        let challenge = config.challenge(&proto::random_bytes(config.gesture.length.get())?)?;
        let prompt = format!(
            "blinkergate: {} - within {}s",
            gesture::describe(&challenge),
            config.timeout.as_secs()
        );
        say(stream, &Reply::Prompt(prompt))?;

        let mut matcher = Matcher::new(
            challenge.clone(),
            config.gesture.thresh,
            config.gesture.window,
            config.gesture.long_min,
        )
        .map_err(|why| Error::config("gesture.pattern", why))?;

        let mesh = session.mesh(&config, &model)?;
        let mut locator = Locator::new(mesh, model.crop.clone());
        let (outcome, tally) = self.look(&config, &model, &mut matcher, &mut locator)?;

        let pattern: Vec<String> = challenge.iter().map(|s| s.to_string()).collect();
        // Log before the mesh goes back, since the counters live on the locator.
        log(&format!(
            "uid {uid}: {outcome:?} [{}] {tally} rescues {} refits {}/{} model {}",
            pattern.join(","),
            locator.rescues,
            locator.refits,
            locator.refits + locator.refits_rejected,
            model.generation,
        ));
        // Warm for the next authentication; the counters and this locator are not.
        session.mesh = Some(locator.into_mesh());

        match outcome {
            Outcome::Matched => {
                session.failures.remove(&uid);
                // The nonce, echoed: this approval belongs to the request that asked
                // for it and to no other.
                say(stream, &Reply::Ok(request.nonce.clone()))
            }
            Outcome::TimedOut => {
                session.fail(uid);
                say(stream, &Reply::Fail)
            }
            // A fault, not a decision. Still fails to the password.
            Outcome::Ended => say(stream, &Reply::Err("camera stopped".into())),
        }
    }

    /// Open the frames, watch, and close them again.
    fn look(
        &self,
        config: &Config,
        model: &Model,
        matcher: &mut Matcher,
        locator: &mut Locator,
    ) -> Result<(Outcome, crate::watch::Tally)> {
        let timeout = config.timeout;
        if let Some(dir) = &config.replay.dir {
            let mut frames = Replay::open(dir)?;
            return watch(&mut frames, model, matcher, timeout, |f| locator.locate(f));
        }
        let camera = Camera::open(&config.device)?;
        let mut frames = camera.stream()?;
        watch(&mut frames, model, matcher, timeout, |f| locator.locate(f))
    }

    /// How long this user still has to wait, if they are in cooldown.
    fn cooling_off(&self, session: &Session, uid: Uid) -> Option<Duration> {
        let failures = *session.failures.get(&uid)?;
        if failures.count < self.config.limit.failures {
            return None;
        }
        // `checked_sub` returning None is the cooldown having passed, which is also how
        // it ends: nothing sweeps the map, the arithmetic just stops refusing.
        self.config.limit.cooldown.checked_sub(failures.last.elapsed())
    }
}

impl Session {
    /// The warm mesh, rebuilt only if this model wants a different threshold.
    fn mesh(&mut self, config: &Config, model: &Model) -> Result<Mesh> {
        // The model records the confidence it was fitted with; the config may override
        // it while a camera is being re-measured.
        let wanted = config.detector.confidence.unwrap_or(model.detector.confidence);
        match self.mesh.take() {
            Some(mesh) if mesh.confidence() == wanted => Ok(mesh),
            _ => Mesh::open(&config.model.face_model, wanted),
        }
    }

    fn fail(&mut self, uid: Uid) {
        let now = Instant::now();
        let failures = self.failures.entry(uid).or_insert(Failures { count: 0, last: now });
        failures.count += 1;
        failures.last = now;
    }
}

/// Bind the socket root-only, or adopt the one systemd passed.
fn bind(path: &Path) -> Result<UnixListener> {
    if let Some(listener) = inherited()? {
        return Ok(listener);
    }
    let io = |source| Error::Io { path: path.into(), source };
    // A stale socket from a killed daemon would otherwise make bind fail forever.
    if path.exists() {
        std::fs::remove_file(path).map_err(io)?;
    }
    let listener = UnixListener::bind(path).map_err(io)?;
    // 0600 as well as the 0700 directory systemd makes: two ways for this to be
    // wrong, and the directory is the one an operator might change.
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)).map_err(io)?;
    Ok(listener)
}

/// The socket systemd opened for us, if it did.
///
/// `LISTEN_FDS=1` means fd 3 is already a listening socket - so the path exists from
/// the moment the unit is enabled, and a `run0` during boot finds either a working
/// daemon or nothing, never a half-created socket.
fn inherited() -> Result<Option<UnixListener>> {
    use std::os::fd::FromRawFd;

    let Ok(count) = std::env::var("LISTEN_FDS") else { return Ok(None) };
    if count.trim() != "1" {
        return Ok(None);
    }
    // Only if this process is the one systemd told, not a child that inherited the
    // variable.
    if std::env::var("LISTEN_PID").ok().as_deref() != Some(&std::process::id().to_string()) {
        return Ok(None);
    }
    // SAFETY: systemd's protocol says fd 3 is a listening socket it opened for this
    // process, and nothing else in this daemon has taken it.
    Ok(Some(unsafe { UnixListener::from_raw_fd(3) }))
}

/// Tell systemd we are ready, if it asked to be told.
///
/// `Type=notify` so that `systemctl start` blocks until the socket answers, which is
/// what makes the verification steps ("stop the daemon, then...") mean anything.
fn notify_ready() {
    let Ok(socket) = std::env::var("NOTIFY_SOCKET") else { return };
    // An abstract socket, which systemd spells with a leading @.
    let path = socket.replacen('@', "\0", 1);
    if let Ok(sock) = std::os::unix::net::UnixDatagram::unbound() {
        let _ = sock.send_to(b"READY=1", path);
    }
}

fn say(stream: &UnixStream, reply: &Reply) -> Result<()> {
    let mut stream = stream;
    stream
        .write_all(reply.line().as_bytes())
        .map_err(|source| Error::Io { path: Path::new("<socket>").into(), source })
}

/// One line to stderr, which systemd files under the unit.
///
/// Never stdout: for the PAM module stdout *is* the wire protocol to
/// polkit-agent-helper-1, and while this daemon is not the module, sharing the habit
/// means a line moved from one to the other cannot break an authentication.
fn log(message: &str) {
    eprintln!("blinkergated: {message}");
}
