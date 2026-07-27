# blinkergate sudo-gate: eye gesture for run0 / polkit

## Where this stands

Phase 1 is two thirds built.

**Keep this section and the checklist below current as part of the work**, not
afterwards - the decisions recorded here are the ones most likely to be
re-litigated by someone who cannot see why they were taken.

| commit | what landed |
|---|---|
| *Port the crop, the classifier and the gesture matcher to Rust* | crop geometry, pooled classifier, gesture matcher, wire protocol, conformance against Python |
| *Watch for a gesture* | camera, MediaPipe mesh + both rescues, watch loop, daemon (socket, single slot, nonce, rate limit), `bgatectl`, replay source |

**Verified with numbers, not assumed:**

- crops agree with `cv2.warpAffine` to at most **1 grey level on 1 pixel in 960**;
  probabilities with `bgate.pooled` to **3e-6** (`cargo test --test golden`).
- the gesture matcher passes the same 13 cases as `gesture.py`'s self-check.
- the eye rings are checked against MediaPipe's own connection constants - same
  vertices, and our order a closed walk of their edges - and pinned in the golden
  file.
- **end to end on real pixels**: ten dataset frames whose labels spell
  `o o l l o o r r o o` are read as the gesture `l,r`, matching on the seventh,
  no face lost, `LevelRefit` firing on 5 of 7 (which is the 54%-of-frames
  behaviour FINDINGS predicts for a 3 degree deadzone against a 3.2 degree
  median). The negative control passes: the same frames against `r,r,l` are not
  matched.

**Not yet done, in dependency order** - see "What is left" at the bottom for the
detail: the PAM module itself, the VM test harness, enrolment, packaging, and
verification steps 3 to 8. Nothing has met the real camera yet; everything so far
went through the replay source.

## Context

blinkergate can already classify per-eye open/closed on this camera and match a
deliberate gesture (`bgate/gesture.py`). The point of the project is to use that
as the confirmation for privilege escalation. This plan turns it into a working
gate for `run0`, which authorizes through **polkit**, which authenticates through
**PAM** (`/etc/pam.d/polkit-1`) — so the hook is a PAM module, and the same
module covers `sudo`, `su`, and GUI escalation by being listed in their stacks
too.

Two decisions taken:

- **The gesture replaces the password** (`auth sufficient`, exactly like
  `pam_fprintd.so` which is already in this stack). This drives every security
  requirement below: the daemon runs as root and the model and config must be
  root-owned, because anything the user can write becomes a path to root.
- **The dialog is GNOME's own.** Prompt text travels through the PAM
  conversation and gnome-shell's polkit dialog renders it. Modal and native for
  free, no UI code, no fight over the session's registered agent. The live
  timeline is deferred to phase 2.

The false-positive rate during ordinary blinking — the open question
`docs/FINDINGS.md` names as deciding whether any of this can gate privilege
escalation — has already been measured and is acceptable, so it is not a
blocker on this work.

### Verified on this machine (not assumed)

- `/usr/lib/polkit-1/polkit-agent-helper-1` forwards `PAM_TEXT_INFO`,
  `PAM_ERROR_MSG`, `PAM_PROMPT_ECHO_OFF/ON` — this is what makes the GNOME
  dialog render our text.
- `/etc/pam.d/polkit-1` already runs `auth sufficient pam_fprintd.so`: the exact
  shape we are copying, working today.
- `pam_fprintd.so` is 26 KB and `fprintd` is a root D-Bus system service. Thin
  module, fat daemon — the architecture below.
- systemd 261 (`run0` present), polkit 127, gnome-shell 50.3, Wayland session.
- `mediapipe` crate 0.1.0 works: same frame, same `libmediapipe.so`, landmarks
  match Python within **1.14 px** worst case and iris centres within **0.62 px**
  (tolerance is 4 px — `FINDINGS` measures 96.3% → 44.2% at a 4 px crop shift).
- `pamtester` is installed, so the module can be tested without touching the
  real polkit stack.

### The polkit 127 sandbox — and why this design is immune to it

`polkit-agent-helper@.service` sandboxes the whole PAM stack:
`PrivateDevices=yes`, `DevicePolicy=strict` (only `/dev/null`), `ProtectHome=yes`,
`ProtectSystem=strict`, `PrivateTmp=yes`, `PrivateNetwork=yes`,
`NoNewPrivileges=yes`, `SystemCallFilter=@system-service`.

**This is what broke Howdy on polkit 127** ([howdy#1077](https://github.com/boltgolt/howdy/issues/1077)) — Howdy opens the camera *inside* the PAM module, and there are no `/dev/video*` nodes in that private `/dev`. A `PrivateDevices=no` drop-in is the usual workaround, and `howdy-git` has one active on this machine right now.

Putting the camera in a **separate daemon** avoids this entirely: the module only
needs a Unix socket, and the sandbox explicitly permits
`RestrictAddressFamilies=AF_UNIX`. So blinkergate ships **no drop-in**, does not
depend on `howdy-git` staying installed, and survives a polkit upgrade that
re-tightens the sandbox. It is the main reason to prefer this architecture over
Howdy's, beyond not loading MediaPipe into a setuid helper.

Three constraints that follow, all of which the design already satisfies:

- **`ProtectHome=yes`** — the PAM stack cannot see `/home`. Nothing may be loaded
  from the checkout under `/home`; assets live in `/var/lib/blinkergate` and
  `/usr/lib/blinkergate`. (The daemon is a normal service and is not sandboxed,
  but keeping assets out of `/home` also avoids a late-mounted home at boot.)
- **`PrivateTmp=yes`** — rules out `/tmp` for IPC. The socket goes under `/run`.
- **`ProtectSystem=strict`** makes the hierarchy read-only. `connect()` does not
  write to the filesystem so this should be fine, but it is load-bearing enough
  to prove in verification step 3 rather than assume.

### Why not a custom polkit agent, definitively

Only **uid 0** may call `AuthenticationAgentResponse2`, so the gesture decision
has to live in PAM no matter who draws the UI — a custom agent could not
authenticate on its own. polkit also authenticates only `unix-user` identities,
its `authfw` is a build-time meson option rather than a runtime plugin, and the
conversation carries just four message styles with text capped at 512 bytes
(no `PAM_BINARY_PROMPT`). Text really is the entire vocabulary, which is why the
timeline cannot go in the dialog.

## Phase 1 — daemon + PAM module

`rust/` subtree, a Cargo workspace. **Three crates, not four** - `core`'s only
consumer was the daemon, so they are one:

| crate | output | contents | state |
|---|---|---|---|
| `proto` | lib | `Uid`, `Nonce`, `Request`, `Reply`, socket path. One dependency (`getrandom`) and a high bar for a second | done |
| `gate` | `blinkergated`, `bgatectl` | config, trust, crop geometry, classifier, gesture matcher, camera, mesh, watch loop, daemon | done |
| `pam` | `pam_blinkergate.so` | cdylib, ~200 lines, no camera and no ML | **not started** |

**`pam_blinkergate.so`** — `pam_sm_authenticate` resolves the target user via
`pam_get_user`, connects to the daemon socket, sends `{uid, nonce}`, pushes the
instruction text through the PAM conversation as `PAM_TEXT_INFO`, waits for the
verdict, returns `PAM_SUCCESS` on a match and **`PAM_IGNORE` on anything else**
so the password path always survives cleanly. Also needs `pam_sm_setcred` as a
no-op returning `PAM_SUCCESS`.

Two rules that are easy to violate and expensive to debug:

- **Never write to stdout.** In the helper, stdout *is* the wire protocol to the
  agent; a stray line produces `Unknown line '…' from helper` and aborts the
  authentication. Log to syslog or stderr only, and pull in no library that
  prints a banner. (Being a thin module with no OpenCV is most of this
  protection already.)
- **`catch_unwind` at every entry point.** A panic unwinding across the FFI
  boundary is UB, and this is loaded into setuid `polkit-agent-helper-1`.

**`blinkergated`** — systemd service, `Type=notify`, running as root. Listens on
`/run/blinkergate/sock`. On a request it opens the camera, runs the loop until
match or timeout, replies, and **closes the camera**. The model and MediaPipe
graph stay warm between requests; the camera does not, so there is no
always-open IR camera. The daemon decides only on **frames it captured itself** —
it must never grant on a client's assertion, or any process able to reach the
socket inherits the decision. Port of `bgate/gesture.py` for matching, and of
`data.eye_centre` / `level_affine` / `crop_at` plus `model.eyes` standardisation
for the crop — the classifier itself is a 5 KB logistic regression, a matmul in
`ndarray` with no ML framework needed.

**Model handoff.** Add a Python export step producing a versioned file the
daemon reads, carrying weights *and* preprocessing parameters together (blur,
normalise, `level_width`, `WIDTH_RATIO`, detection confidence) — `model.py:58`
insists preprocessing is carried in the model, and a silent mismatch costs
accuracy with no crash. Pair it with a **golden-file conformance test**: fixed
frames plus the probabilities Python produces, asserted against the Rust
implementation. The detector half needs no such test, since both call the same
`libmediapipe.so`.

**Enrollment** is a privileged step, not a file copy the user can do:
`blinkergate-enroll --user nb --from models/gen-0032`, run under `run0`, which
installs into root-owned `/var/lib/blinkergate/<uid>/`.

### Concurrency: exactly one authentication at a time

Two requests can genuinely overlap — `polkit-agent-helper@.service` is a
template, so each authentication gets its own PAM stack, and two `run0`
invocations or a GUI prompt alongside a terminal one will collide.

**The dangerous handling is to multiplex**: one camera session with two waiters,
where a match wakes both. That is the confused deputy in its purest form — you
blink to approve `run0 ls` and simultaneously approve an attacker's
`run0 rm -rf /`. One gesture must satisfy exactly one request.

So the daemon holds a **single in-flight slot**. A second request is answered
immediately with `Busy`, the module returns `PAM_IGNORE`, and that caller falls
through to the password. No queueing: a queued request would display "blink
now" while the daemon is not watching for it, and a dialog that lies about what
it is waiting for is worse than a password prompt. The slot is held from accept
to reply and bounded by the timeout, with the camera opened and closed inside it.

Two properties worth stating, because they carry most of the safety here:

- **There is no silent request path.** The socket is root-only, so every request
  necessarily comes from a real PAM stack, which means a visible dialog. A
  user-level attacker cannot ask the daemon directly; they must go through
  polkit and put a dialog on your screen.
- **The nonce stops an approval being reused** for a request other than the one
  it was issued for.

But note what the dialog does *not* give you — see below.

Denial of the gesture path — an attacker deliberately keeping the slot busy — is
accepted: it degrades to typing a password, which is exactly today's behaviour.

### What the user actually sees, and the limit of consent (verified)

`run0` triggers `org.freedesktop.systemd1.manage-units`, whose message is fixed:

> Authentication is required to manage system services or other units.

**The command is never shown.** `run0 ls` and `run0 rm -rf /` render an identical
dialog: there is no per-invocation message, `systemd-run` passes no
`polkit.message` override, and polkit does not hand the action id to PAM either,
so our instruction line cannot name it. A custom agent would not fix this — it
can only display what polkit passes.

Worse for consent, that action's `allow_active` default is **`auth_admin_keep`**:
one success is cached for a few minutes and later `run0` calls do not prompt at
all. So by default the gesture approves *a window of admin activity*, not a
command.

Two consequences to accept deliberately rather than discover later:

- **What the gesture actually proves** is that a live human was present and
  consented to an administrative action at that moment. That is a real gain over
  a password an attacker can capture or a ticket they can ride — but it is not
  per-command consent, and the plan should not pretend otherwise.
- **Recommended hardening**: ship a rules file in `/etc/polkit-1/rules.d/` that
  overrides this action to plain `auth_admin` for the gated user, removing the
  cache so every escalation needs a fresh gesture. That is the single change
  that most tightens the binding, and it is config rather than code. Make it
  optional in the package — it costs a gesture per command.

### Configuration

`/etc/blinkergate/config.toml` for defaults, `/var/lib/blinkergate/<uid>/config.toml`
for per-user overrides — **both root-owned**, since under `auth sufficient` a
user-writable pattern is a path to root (set it to a single natural blink and
wait). Everything currently hardcoded or on the Python CLI becomes a key:

```toml
device = "/dev/video2"
subtract = false
timeout = 15.0            # seconds to wait for the gesture

[gesture]
pattern = ["l", "r", "r", "l"]   # or randomized, see below
window = 6.0                     # GestureMatcher default, not currently on CLI
long_min = 0.5                   # ditto
thresh = 0.5

[model]
dir = "/var/lib/blinkergate"     # per-uid subdirectories

[detector]
confidence = 0.2                 # model.py DETECTION_CONFIDENCE
```

### Security requirements (all forced by "replaces the password")

- **Socket** `/run/blinkergate/sock` root-owned, mode 0600. Only root — i.e. the
  PAM module — can connect, so a user process cannot answer on the daemon's
  behalf. Verify the peer with `SO_PEERCRED` anyway.
- **Nonce binding.** The module generates a nonce per request; the daemon's reply
  is single-use and bound to it with a short TTL. Without this, one gesture
  approving `run0 ls` can be consumed by a concurrent request for something else
   — a textbook confused deputy, and painful to retrofit.
- **Fail to password, never open.** Camera busy, daemon dead, ambient IR over the
  ceiling, timeout — all return `PAM_AUTHINFO_UNAVAIL` so the stack falls
  through to `system-auth`. With `sufficient`, failing closed costs the user
  nothing but a password prompt.
- **Rate limit.** After N failures, refuse for a cooldown, so a photo cannot be
  retried indefinitely.
- **Randomized challenge (recommended).** Because the dialog *tells the user what
  to perform*, the daemon can pick the pattern per request from a configured
  alphabet. A pre-recorded video of a fixed gesture then fails, which is cheap
  liveness that a static pattern cannot provide. Config as
  `pattern = "random"` with `length`, or an explicit fixed list.
- **Known limitation to write down:** a live presentation attack (a screen
  showing the requested gesture in IR) is not defended. The strobe and
  retroreflection raise the bar; they do not close it.

## Phase 2 — deferred: the live timeline

Not a custom polkit agent — that route is closed (only uid 0 can answer the
Authority, and the conversation is text-only). The way to draw a live preview on
GNOME is a **GNOME Shell extension**, which is where Howdy converged after
finding Mutter refuses windows above the lock shield. The daemon would publish
progress on a second socket or the session bus and the extension would render
it. Worth doing only once phase 1 proves the UX is worth the pixels.

## Verification

Marked with what has actually been run. `[x]` means run and passing; `[ ]` means
not attempted.

1. `[x]` **Unit**: gesture matcher port against the same cases as
   `bgate/gesture.py`'s `__main__` self-check; conformance test for crop and
   classifier against golden Python outputs.
2. **Daemon alone**: a CLI client that performs a request against a running
   daemon and prints the verdict — no PAM involved.
3. `[ ]` **PAM module in isolation**: create `/etc/pam.d/blinkergate-test` containing
   only the module, then `pamtester blinkergate-test nb authenticate`. This is
   where the module gets debugged; the real stack is never at risk. Then prove
   it **inside the sandbox** — the one assumption still unverified is that
   `connect()` to `/run/blinkergate/sock` succeeds under `ProtectSystem=strict`
   and `PrivateNetwork=yes`. Run the same test via a unit carrying the helper's
   sandbox settings before touching `/etc/pam.d/polkit-1`; if it fails, the fix
   is a `ReadWritePaths=/run/blinkergate` drop-in, still with no device access.
4. `[ ]` **Real stack, with a safety net**: keep a root shell open in another terminal
   for the whole session. Add `auth sufficient pam_blinkergate.so` to
   `/etc/pam.d/polkit-1` *above* `system-auth`, then run `run0 true` and confirm
   the GNOME dialog shows the instruction and that a correct gesture
   authenticates while a wrong one falls through to the password prompt.
5. `[ ]` **Failure modes explicitly**: stop the daemon, occupy the camera with
   `eyetrainer.py`, and unplug the model file — each must fall through to the
   password, never deny outright and never hang past the timeout.
6. `[~]` **Concurrency**: `bgatectl race` implements the second half of this and
   asserts the property directly, but has never been run against a live daemon.
   Fire two `run0` calls at once. Exactly one must get the
   gesture prompt; the other must fall through to the password. Then assert the
   property that matters — a single gesture must never satisfy two pending
   requests — by scripting two simultaneous requests against the daemon
   directly and confirming one is refused before any frame is captured.
7. `[ ]` **Consent granularity, eyes open**: run `run0 true`, note the dialog wording,
   then run `run0 true` again within a minute and confirm it does *not* prompt
   (that is `auth_admin_keep` working as designed). Install the `auth_admin`
   rules file and confirm the second call prompts again. Decide which behaviour
   you want shipped on the strength of having seen both.
8. `[ ]` **Regression against the sandbox**: `systemctl mask` the `howdy-git` drop-in
   (or uninstall it) and confirm blinkergate still authenticates. If it stops
   working, the camera has leaked into the PAM stack and the architecture has
   been violated somewhere.

## Useful context

`howdy-git r592` is installed here and its drop-in is what currently relaxes the
polkit sandbox — but Howdy is wired into `gdm-password` only, not `polkit-1`, so
**`run0` is password-only today**. Nothing to migrate, and nothing of Howdy's to
preserve; blinkergate should not inherit its dependency on that drop-in.


## Decisions taken while implementing

Deviations from the plan above, each with the reason, so they are not re-litigated
or accidentally reverted.

- **Three crates, not four.** `core` had one consumer.
- **YuNet is not ported.** It needs an ONNX runtime, and what it rescues is dark
  low-contrast frames rather than the tilted ones this gate sees. The mesh plus
  the rotation rescue plus the levelled refit are in. A frame nothing can locate
  falls through to the password.
- **The pixel model is not ported.** 44.2% at a 4px crop shift against 98.3%
  pooled; a `kind: "pixel"` export loads and then fails with that measurement.
- **`[replay]` config section, not a magic `device` value.** A device path and a
  directory of recordings are different things, and a daemon that silently
  accepted either would be one typo from authenticating against yesterday's
  frames.
- **A recording carries its own frame rate** in a `replay.toml` beside the frames,
  and one without it is refused. `long_min` separates a held closure from a blink
  by *duration*, so six shut frames are a held closure at 133ms and a blink at
  20ms. Nothing in the Rust assumes a frame rate any more.
- **No calibration phase.** The lit/dark split is learned from the frames as they
  arrive. A second up front cost about seven lit frames - a whole wink - and spent
  them *after* the dialog had said "blink". The strobe is only believed when the
  feed is plainly two populations (10 grey levels apart), so a camera whose
  emitter does not strobe keeps every frame.
- **`Model::parse` is split from `Model::load`.** Enrolment and `bgatectl check`
  legitimately hold a model that is not root-owned yet - they must be able to
  refuse a bad one *before* installing it.
- **The model file is trusted the same way the config is** (root-owned, not
  group/world-writable, not a symlink, via `trust::read_root_owned`). The plan
  said the model must be root-owned; nothing was enforcing it.
- **`SO_PEERCRED` via `rustix`**, because `std`'s `peer_cred` is still unstable
  (rust#42839) and hand-rolling the getsockopt means unsafe whose failure mode is
  reading something else as a uid.
- **Types carry the hazards** this codebase has already been bitten by: `Side` +
  `PerEye<T>` instead of `[T; 2]` (indexing a pair of eyes by number is how
  `img_left` came to mean the right of the picture), `Roll` in degrees with one
  conversion, `Turn { sin, cos }`, `Landmark`, `Grey`, `Brightness`/`Spread`,
  `Probability`, `Verdict` from `feed()`, and `Frame`/`Crop` as one `Raster`
  tagged differently - the classifier takes its pooling window from the raster it
  is handed, so a whole frame reaching it would read as a very large eye.

## Running it

Nothing here is committed - models and frames are yours.

```sh
uv run export_model.py                                     # model.json + golden data
uv run record_replay.py --pattern l,r --out /tmp/replay     # frames + replay.toml
cd rust && cargo test                                      # 28 unit + conformance
MEDIAPIPE_LIB=$PWD/../.venv/lib/python3.13/site-packages/mediapipe/tasks/c/libmediapipe.so \
  BGATE_REPLAY=/tmp/replay cargo test --test replay -- --ignored --nocapture
./target/debug/bgatectl check ../model.json                # would the daemon run it?
```

`MEDIAPIPE_LIB` avoids the `mediapipe` crate downloading its own 34MB copy; the
packaged daemon will need the same variable set in its unit.

`bgatectl` exit codes: 0 confirmed, 1 denied (it ran, the answer was no), 2 could
not ask.

## What is left

In dependency order. The VM harness comes first because it is what allows any of
the rest to be *run* without a password prompt.

1. **VM harness** - `packaging/test/run-vm.sh` + `e2e.sh`, on the pattern of
   virtme-ng: host userspace and host-built binaries exposed to the guest, so
   there is no rootfs to build. `vng --user root` gives root
   in a guest with the host filesystem overlaid, so `/etc`, `/var` and `/run`
   writes do not touch the host. The guest has no `/dev/video*`, which is what
   `[replay]` is for.
2. **`pam_blinkergate.so`** - `pam_sm_authenticate` resolves the user, connects,
   pushes the prompt through the conversation as `PAM_TEXT_INFO`, returns
   `PAM_SUCCESS` on a match and `PAM_IGNORE` on everything else. `pam_sm_setcred`
   a no-op returning `PAM_SUCCESS`. **Never write to stdout** and `catch_unwind`
   at every entry point.
3. **Verification step 3**, in the VM: `pamtester`, then the same test inside a
   unit carrying the helper's sandbox settings. This is the plan's one explicitly
   unverified assumption and the thing that broke Howdy on polkit 127.
4. **Enrolment and packaging** - `blinkergate-enroll`, the systemd unit (with
   `RuntimeDirectory=blinkergate` and `MEDIAPIPE_LIB`), `/etc/blinkergate/config.toml`,
   and where `libmediapipe.so` and `face_landmarker.task` live.
5. **Steps 4 to 8 on the real machine** - these need hands: performing an actual
   gesture at the camera, editing `/etc/pam.d/polkit-1` with a root shell held
   open, and the `auth_admin` vs `auth_admin_keep` decision, which is a policy
   choice rather than a code one.

Also outstanding: the README and `docs/` say nothing about the Rust half yet.
