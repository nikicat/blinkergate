//! The daemon. Started by systemd, spoken to by `pam_blinkergate.so`.
//!
//! Deliberately without a command line: everything is in
//! `/etc/blinkergate/config.toml`, root-owned, because a flag is a thing an operator
//! can pass and a config file is a thing that can be checked for ownership.
//!
//! Runs as root, and needs to: it opens the infrared camera, reads per-user models out
//! of `/var/lib/blinkergate`, and binds a socket only root may connect to.

use blinkergate::{Config, Daemon};
use blinkergate_proto::{Uid, socket};

fn main() -> std::process::ExitCode {
    // The daemon's own config layer is root's, since there is no user yet - each
    // request loads the layer belonging to the user being authenticated.
    let config = match Config::load(Uid::ROOT) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("blinkergated: {e}");
            // Refusing to start beats starting with a configuration that is not what
            // the file says: every authentication would then be decided by defaults
            // nobody chose.
            return std::process::ExitCode::FAILURE;
        }
    };
    // The configuration, which is true as soon as it is read. Whether anything is
    // listening is the daemon's own claim to make, once it is.
    eprintln!(
        "blinkergated: {}, {:?} timeout, {} challenge",
        match &config.replay.dir {
            Some(dir) => format!("REPLAYING {}", dir.display()),
            None => format!("camera {}", config.device.display()),
        },
        config.timeout,
        match &config.gesture.pattern {
            blinkergate::config::Pattern::Random => {
                format!("{}-symbol random", config.gesture.length)
            }
            blinkergate::config::Pattern::Fixed(syms) => {
                syms.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",")
            }
        },
    );
    if let Err(e) = Daemon::new(config).serve(socket()) {
        eprintln!("blinkergated: {e}");
        return std::process::ExitCode::FAILURE;
    }
    std::process::ExitCode::SUCCESS
}
