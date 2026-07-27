//! Talk to the daemon without PAM in the way.
//!
//! This is verification step 2 of the plan, and the tool to reach for whenever an
//! authentication fails and it is not yet clear whether the fault is in the camera, the
//! model, the daemon or the PAM stack. Each subcommand answers one of those:
//!
//! ```text
//! bgatectl auth [user]     one request against the running daemon, printing the verdict
//! bgatectl check <model>   would the daemon accept this model file?
//! bgatectl race [user]     two requests at once - exactly one must be told BUSY
//! ```
//!
//! `auth` and `race` need root, because the socket does.
//!
//! Exit codes, since these are meant to be scripted:
//!
//! | code | meaning |
//! |---|---|
//! | 0 | confirmed - the gesture matched, the model loads, the slot rule held |
//! | 1 | denied - it ran, and the answer was no |
//! | 2 | could not ask - no daemon, no permission, an unreadable file, bad usage |
//!
//! Denied and failed are separate because a gesture that did not match is a *verdict*,
//! and a daemon that is not running is not.

use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;

use blinkergate::Model;
use blinkergate_proto::{Nonce, Reply, Request, Uid, socket};

/// What a subcommand concluded.
///
/// Not a bare integer: while this was being written `auth` returned `true` for a match,
/// which became exit 1, so `bgatectl auth && echo ok` printed nothing exactly when the
/// gesture had worked. An enum cannot be inverted by an `as u8`.
enum Exit {
    /// The thing asked for happened.
    Confirmed,
    /// It ran, and the answer was no.
    Denied,
    /// Could not ask at all.
    Failed(String),
}

impl Exit {
    fn code(self) -> std::process::ExitCode {
        match self {
            Exit::Confirmed => std::process::ExitCode::SUCCESS,
            Exit::Denied => std::process::ExitCode::from(1),
            Exit::Failed(why) => {
                eprintln!("bgatectl: {why}");
                std::process::ExitCode::from(2)
            }
        }
    }
}

/// `?` on a `Result<_, String>` inside a subcommand becomes [`Exit::Failed`].
impl From<String> for Exit {
    fn from(why: String) -> Exit {
        Exit::Failed(why)
    }
}

fn main() -> std::process::ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = args.get(1).map(String::as_str);
    match args.first().map(String::as_str) {
        Some("auth") => for_user(arg, auth),
        Some("race") => for_user(arg, race),
        Some("check") => check(arg),
        _ => Exit::Failed("usage: bgatectl auth|race [user] | bgatectl check <model.json>".into()),
    }
    .code()
}

/// Resolve the user once, here, and hand the subcommands a [`Uid`].
///
/// So that nothing downstream holds a name it might resolve again - `race` sends two
/// requests that have to differ only in when they arrive, and two lookups of one name
/// are not the same thing as one lookup.
fn for_user(arg: Option<&str>, run: impl FnOnce(Uid) -> Exit) -> Exit {
    match resolve(arg) {
        Ok(uid) => run(uid),
        Err(why) => Exit::Failed(why),
    }
}

/// One authentication, printing every reply as it arrives.
///
/// Prints the prompt separately from the verdict because that is the order the PAM
/// module sees them in: the instruction has to reach the dialog before the daemon knows
/// anything, and a daemon that only spoke at the end could not be used at all.
fn auth(uid: Uid) -> Exit {
    match watch_one(uid) {
        Ok(exit) => exit,
        Err(why) => Exit::Failed(why),
    }
}

/// The conversation itself. Split from [`auth`] so that every step can use `?`: the
/// difference between a verdict and a broken connection is the whole point of [`Exit`],
/// and it survives here as `Ok(Exit::Denied)` against `Err`.
fn watch_one(uid: Uid) -> Result<Exit, String> {
    let nonce = Nonce::new().map_err(|e| e.to_string())?;
    let request = Request { uid, nonce: nonce.clone() };

    let mut stream = UnixStream::connect(socket()).map_err(|e| {
        format!("{}: {e} (is blinkergated running, and are you root?)", socket().display())
    })?;
    stream.write_all(request.line().as_bytes()).map_err(|e| e.to_string())?;

    let started = std::time::Instant::now();
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    while reader.read_line(&mut line).map_err(|e| e.to_string())? > 0 {
        let at = started.elapsed();
        match Reply::parse(&line) {
            Some(Reply::Prompt(text)) => println!("[{at:>7.2?}] prompt: {text}"),
            Some(Reply::Ok(echoed)) => {
                println!("[{at:>7.2?}] MATCHED");
                // The check the PAM module makes, made here too: an approval that is
                // not bound to this request is not an approval of it.
                if echoed != nonce {
                    return Err("the daemon echoed a different nonce".into());
                }
                return Ok(Exit::Confirmed);
            }
            Some(Reply::Fail) => {
                println!("[{at:>7.2?}] no match");
                return Ok(Exit::Denied);
            }
            // Busy and an error are the daemon declining to decide, not deciding no.
            Some(Reply::Busy) => {
                println!("[{at:>7.2?}] busy - another authentication holds the camera");
                return Ok(Exit::Failed("the daemon was busy".into()));
            }
            Some(Reply::Err(why)) => {
                return Ok(Exit::Failed(format!("the daemon refused: {why}")));
            }
            None => return Err(format!("unparseable reply {line:?}")),
        }
        line.clear();
    }
    Err("the daemon closed without answering".into())
}

/// Two requests at once. Exactly one must be refused, and refused *before* any frame
/// is captured - which is the property that stops one gesture satisfying two requests.
fn race(uid: Uid) -> Exit {
    match race_two(uid) {
        Ok(exit) => exit,
        Err(why) => Exit::Failed(why),
    }
}

/// The two requests. Split from [`race`] for the same reason, and it is where the
/// timing lives: the second is delayed rather than simultaneous, because "at the same
/// time" would sometimes mean "before the first has taken the slot", which proves
/// nothing.
fn race_two(uid: Uid) -> Result<Exit, String> {
    let second = std::thread::spawn(move || {
        // Long enough that the first request is certainly inside the slot, short enough
        // to be well within its timeout.
        std::thread::sleep(std::time::Duration::from_millis(300));
        probe(uid)
    });
    let first = probe(uid);
    let second = second.join().map_err(|_| "the second request panicked".to_string())?;
    let (first, second) = (first?, second?);
    println!("first: {first:?}\nsecond: {second:?}");
    match (&first, &second) {
        // The one that arrived second must be the one turned away, and it must be
        // turned away as Busy rather than by timing out behind the first.
        (_, Reply::Busy) => {
            println!("exactly one request got the camera");
            Ok(Exit::Confirmed)
        }
        _ => {
            println!("BOTH requests were served - the single-slot rule is broken");
            Ok(Exit::Denied)
        }
    }
}

/// Send a request and return the first reply, ignoring the prompt.
fn probe(uid: Uid) -> Result<Reply, String> {
    let request = Request { uid, nonce: Nonce::new().map_err(|e| e.to_string())? };
    let mut stream =
        UnixStream::connect(socket()).map_err(|e| format!("{}: {e}", socket().display()))?;
    stream.write_all(request.line().as_bytes()).map_err(|e| e.to_string())?;
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    while reader.read_line(&mut line).map_err(|e| e.to_string())? > 0 {
        match Reply::parse(&line) {
            // Skip the prompt: this is about which request got the slot.
            Some(Reply::Prompt(_)) => line.clear(),
            Some(reply) => return Ok(reply),
            None => return Err(format!("unparseable reply {line:?}")),
        }
    }
    Err("the daemon closed without answering".into())
}

/// Would the daemon accept this model file? Answered without the daemon, so enrolment
/// can ask before installing one.
fn check(path: Option<&str>) -> Exit {
    match inspect(path) {
        Ok(exit) => exit,
        Err(why) => Exit::Failed(why),
    }
}

/// Reading and reporting on the file. Split from [`check`] so that "cannot read it" is
/// an `Err` while "read it, and this build will not run it" is `Ok(Exit::Denied)` -
/// which is the answer enrolment needs, and the opposite of a missing file.
fn inspect(path: Option<&str>) -> Result<Exit, String> {
    let path = path.ok_or("which model file?")?;
    // Read it directly rather than through the daemon's trust rule: the point of this
    // subcommand is to judge a model before it is installed, when it is still owned by
    // whoever exported it. Ownership is reported below instead of refused.
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    match Model::parse(&text, std::path::Path::new(path)) {
        Ok(model) => {
            println!(
                "{path}: {} from {}, {}x{} crops, {} kernels of {}x{}, detector {}",
                model.kind,
                model.generation,
                model.crop.in_w,
                model.crop.in_h,
                model.kernels.len(),
                model.kernel_size,
                model.kernel_size,
                model.detector.confidence,
            );
            match model.accuracy {
                Some(a) => println!("  held-out accuracy {a:.4}"),
                None => println!("  no held-out score recorded"),
            }
            // Not an error here, but it would be at authentication time: these weights
            // are the decision boundary, so the daemon reads them only from a file the
            // user cannot rewrite.
            match std::fs::metadata(path).map(|m| std::os::unix::fs::MetadataExt::uid(&m)) {
                Ok(0) => println!("  root-owned, so the daemon will read it"),
                Ok(uid) => println!("  NOT root-owned (uid {uid}) - enrol it before use"),
                Err(e) => println!("  cannot stat: {e}"),
            }
            Ok(Exit::Confirmed)
        }
        // A model this build cannot run is a verdict about the file, not a failure to
        // look at it.
        Err(e) => {
            println!("{e}");
            Ok(Exit::Denied)
        }
    }
}

/// A user name, a uid, or the caller's own.
///
/// Names go through `getpwnam` in the PAM module; here the common case is "me", and
/// spelling a name is a convenience for testing another account.
fn resolve(user: Option<&str>) -> Result<Uid, String> {
    let Some(user) = user else {
        return Ok(Uid::new(rustix::process::getuid().as_raw()));
    };
    if let Ok(uid) = user.parse::<u32>() {
        return Ok(Uid::new(uid));
    }
    // No getpwnam without libc; /etc/passwd is what it reads anyway, and this is a
    // testing tool rather than the authentication path.
    let passwd = std::fs::read_to_string("/etc/passwd").map_err(|e| e.to_string())?;
    for line in passwd.lines() {
        let mut fields = line.split(':');
        if fields.next() == Some(user) {
            let uid = fields.nth(1).and_then(|f| f.parse().ok());
            return uid.map(Uid::new).ok_or_else(|| format!("{user}: unreadable passwd entry"));
        }
    }
    Err(format!("{user}: no such user"))
}
