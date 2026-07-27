//! Where frames come from.
//!
//! Windows Hello style cameras strobe their infrared emitter, so alternate frames
//! arrive unlit and useless. Dropping them is why the effective rate is half the
//! delivered one: the reference camera delivers 15 fps of the 30 it advertises,
//! leaving ~7.5 usable frames a second. That ceiling drives everything downstream -
//! a natural blink is ~150ms and lands in roughly *one* frame, so a gesture has to
//! be held rather than flicked.
//!
//! Two sources implement [`Frames`]. The camera is the real one; [`Replay`] reads a
//! directory of PGM files, because a virtual machine has no `/dev/video*` and the
//! end-to-end test needs frames that contain a real gesture.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::Deserialize;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;

use crate::error::{Error, Result};
use crate::frame::{Brightness, Frame, Size, Spread};

/// What one read off a frame source produced.
#[must_use]
pub enum Read {
    /// An emitter-lit frame, the only kind worth classifying.
    Lit(Frame),
    /// The emitter was dark for this one. Half of them are, by design - the dark
    /// frames are the ambient reference that makes subtraction possible.
    Dark,
    /// Nothing further will arrive: the device stopped delivering, or a replay ran
    /// out of frames. Distinct from `Dark` because waiting longer cannot help.
    Ended,
}

/// A source of frames, lit and unlit.
pub trait Frames {
    /// Block for the next frame. One read, one answer - a `Dark` is not retried
    /// here, so a caller counting reads can see the strobe.
    fn read(&mut self) -> Read;

    /// What the frames measure, for the geometry that has to divide by it.
    fn size(&self) -> Size;
}

/// An infrared camera, opened and configured but not yet streaming.
///
/// Streaming is a separate step because `v4l`'s stream borrows the device: keeping
/// both in one struct would be self-referential. It suits the daemon anyway, which
/// opens the camera inside one authentication and closes it again - there is no
/// always-on infrared camera here.
pub struct Camera {
    device: v4l::Device,
    path: PathBuf,
    size: Size,
}

impl Camera {
    /// Open the device and ask it for 8-bit grayscale.
    ///
    /// `GREY` is what the infrared sensor advertises; pointing this at the colour
    /// camera gets a format negotiation that succeeds and frames that are not
    /// infrared, which is why the config comment names `v4l2-ctl --list-devices`.
    pub fn open(path: &Path) -> Result<Camera> {
        let io = |source| Error::Io { path: path.into(), source };
        let device = v4l::Device::with_path(path).map_err(io)?;
        let mut format = device.format().map_err(io)?;
        format.fourcc = v4l::FourCC::new(b"GREY");
        let format = device.set_format(&format).map_err(io)?;
        if format.fourcc != v4l::FourCC::new(b"GREY") {
            return Err(Error::Camera {
                path: path.into(),
                why: format!("delivers {} rather than GREY", format.fourcc),
            });
        }
        let size = Size { width: format.width as usize, height: format.height as usize };
        Ok(Camera { device, path: path.to_path_buf(), size })
    }

    /// Start streaming.
    ///
    /// One buffer, not the driver's default of several: a queue deeper than one shows up
    /// as input lag rather than as dropped frames, and on the reference camera the
    /// default four cost ~264ms of it.
    pub fn stream(&self) -> Result<Strobe<'_>> {
        let io = |source| Error::Io { path: self.path.clone().into_boxed_path(), source };
        let stream =
            v4l::io::mmap::Stream::with_buffers(&self.device, v4l::buffer::Type::VideoCapture, 1)
                .map_err(io)?;
        Ok(Strobe { stream, size: self.size, seen: None })
    }
}

/// A streaming camera that can tell a lit frame from an unlit one.
pub struct Strobe<'a> {
    stream: v4l::io::mmap::Stream<'a>,
    size: Size,
    /// The brightness range delivered so far, which is where the cutoff comes from.
    /// `None` until the first frame arrives.
    seen: Option<Seen>,
}

/// The range of frame brightness a stream has delivered.
///
/// Learned as the frames arrive rather than in a calibration phase up front. A phase
/// would cost a second of every authentication - about seven lit frames, an entire
/// wink's worth - and it would spend that second *after* the dialog has already told the
/// user to blink, so the first thing they did would be thrown away.
///
/// The frames it learns from are also classified by it, which is safe in one direction
/// and not the other: with one frame seen the cutoff equals that frame, so it reads as
/// lit. A dark frame taken for lit costs one mesh pass that finds no face. A lit frame
/// taken for dark would cost a symbol of the gesture, and cannot happen.
#[derive(Debug, Clone, Copy)]
struct Seen {
    darkest: Brightness,
    lightest: Brightness,
}

/// The gap between the two populations of frames, below which there is only one.
///
/// Measured on the reference camera: unlit frames average about 5 against 40-95 for lit
/// ones, so a real strobe is unmistakable. A narrower spread than this is a camera whose
/// emitter simply does not strobe - and then the midpoint of the range would sit inside
/// the noise, throwing away half the frames for nothing.
const STROBE_GAP: Spread = Spread::levels(10.0);

impl Seen {
    fn first(mean: Brightness) -> Seen {
        Seen { darkest: mean, lightest: mean }
    }

    fn include(&mut self, mean: Brightness) {
        if mean < self.darkest {
            self.darkest = mean;
        }
        if mean > self.lightest {
            self.lightest = mean;
        }
    }

    /// The brightness below which a frame is unlit, or `None` if this feed holds only one
    /// population of frames and every one of them is usable.
    fn cutoff(&self) -> Option<Brightness> {
        let strobing = Brightness::spread(self.lightest, self.darkest) >= STROBE_GAP;
        strobing.then(|| Brightness::midpoint(self.darkest, self.lightest))
    }
}

impl Strobe<'_> {
    /// One frame, lit or not, or `None` if the read failed.
    fn grab(&mut self) -> Option<Frame> {
        let (bytes, meta) = self.stream.next().ok()?;
        let used = meta.bytesused as usize;
        // A short buffer is a partial frame; the geometry would read the rows below
        // it as the bottom of the face.
        if used < self.size.area() {
            return None;
        }
        Frame::from_bytes(self.size, &bytes[..self.size.area()])
    }
}

impl Frames for Strobe<'_> {
    fn read(&mut self) -> Read {
        let Some(frame) = self.grab() else { return Read::Ended };
        let mean = frame.brightness();
        let seen = self.seen.get_or_insert_with(|| Seen::first(mean));
        seen.include(mean);
        match seen.cutoff() {
            Some(cutoff) if mean < cutoff => Read::Dark,
            _ => Read::Lit(frame),
        }
    }

    fn size(&self) -> Size {
        self.size
    }
}

/// Frames read from a directory of PGM files, in filename order.
///
/// For the end-to-end test: a virtual machine has no `/dev/video*`, and a gesture is
/// the one thing a synthetic frame cannot contain. Record once on the host, replay
/// wherever the PAM stack is being exercised.
///
/// Every file counts as lit - a recording is of the frames the strobe already selected -
/// and they are handed out at the rate the recording itself declares.
pub struct Replay {
    files: std::vec::IntoIter<PathBuf>,
    size: Size,
    interval: Duration,
    next_at: Option<Instant>,
}

/// What a recording says about itself, read from `replay.toml` beside its frames.
///
/// The rate belongs to the recording rather than to the configuration, because it is
/// not a preference: `long_min` separates a held closure from a blink by *duration*, so
/// six shut frames are a held closure at 133ms and a blink at 20ms. Pacing someone
/// else's recording at your own rate silently changes what its frames mean.
///
/// Unknown keys are allowed: a recorder is expected to write provenance here - what
/// pattern the frames spell, where they came from - that nothing reads.
#[derive(Debug, Deserialize)]
struct Manifest {
    /// Seconds between frames, as measured by whatever recorded them.
    #[serde(deserialize_with = "crate::config::secs")]
    interval: Duration,
}

/// The file a recording declares itself in.
const MANIFEST: &str = "replay.toml";

impl Replay {
    pub fn open(dir: &Path) -> Result<Replay> {
        let io = |source| Error::Io { path: dir.into(), source };
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(io)?
            .filter_map(|entry| Some(entry.ok()?.path()))
            .filter(|p| p.extension().is_some_and(|e| e == "pgm"))
            .collect();
        files.sort();
        let first = files.first().ok_or_else(|| Error::Camera {
            path: dir.into(),
            why: "holds no .pgm frames".into(),
        })?;
        let size = read_pgm(first)?.size();
        let manifest = Manifest::read(dir)?;
        Ok(Replay { files: files.into_iter(), size, interval: manifest.interval, next_at: None })
    }
}

impl Manifest {
    fn read(dir: &Path) -> Result<Manifest> {
        let path = dir.join(MANIFEST);
        let text = std::fs::read_to_string(&path).map_err(|_| Error::Camera {
            path: dir.into(),
            // A recording with no rate cannot be replayed honestly, so this is a refusal
            // rather than a fall back to some other camera's rate.
            why: format!("has no {MANIFEST} saying what rate it was recorded at"),
        })?;
        toml::from_str(&text).map_err(|source| Error::Toml { path: path.into_boxed_path(), source })
    }
}

impl Frames for Replay {
    fn read(&mut self) -> Read {
        // Pace against when the last frame was *due*, not when it was handed over, so
        // the work done on it does not push the schedule out.
        let due = self.next_at.unwrap_or_else(Instant::now);
        if let Some(wait) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(wait);
        }
        self.next_at = Some(due + self.interval);
        let Some(path) = self.files.next() else { return Read::Ended };
        match read_pgm(&path) {
            Ok(frame) if frame.size() == self.size => Read::Lit(frame),
            // A frame of another size cannot be the same recording, and silently
            // cropping one is how a replay comes to prove nothing.
            _ => Read::Ended,
        }
    }

    fn size(&self) -> Size {
        self.size
    }
}

/// A binary PGM (`P5`), which is what `cv2.imwrite` writes for a grayscale frame and
/// the simplest thing that is still a real image file.
fn read_pgm(path: &Path) -> Result<Frame> {
    let bytes = std::fs::read(path).map_err(|source| Error::Io { path: path.into(), source })?;
    let bad = |why: String| Error::Camera { path: path.into(), why };

    let mut fields = Fields::new(&bytes);
    let mut field = || fields.next().ok_or_else(|| bad("truncated header".into()));
    if field()? != b"P5" {
        return Err(bad("not a binary PGM (P5)".into()));
    }
    let mut count = || {
        let field = field()?;
        number(field).ok_or_else(|| bad(format!("{:?} is not a number", as_text(field))))
    };
    let size = Size { width: count()?, height: count()? };
    let max = count()?;
    if max != 255 {
        return Err(bad(format!("maximum value {max}, not 8-bit")));
    }

    let raster = bytes
        .get(fields.raster_at()..)
        .and_then(|rest| rest.get(..size.area()))
        .ok_or_else(|| {
            // The failure that matters: a raster shorter than the header claims would
            // otherwise be read as a frame whose bottom rows are someone else's memory.
            let have = bytes.len().saturating_sub(fields.raster_at());
            bad(format!("{have} bytes of pixels for {}x{}", size.width, size.height))
        })?;
    Frame::from_bytes(size, raster).ok_or_else(|| bad("empty raster".into()))
}

/// A scanner over a PGM header's whitespace-separated fields.
///
/// Not `split_ascii_whitespace` over the whole file: the header is four fields and then
/// exactly one whitespace byte, after which everything is binary pixel data that would
/// happily parse as more fields. So this walks the header and remembers where it
/// stopped.
struct Fields<'a> {
    bytes: &'a [u8],
    /// One past the last field returned.
    at: usize,
}

impl<'a> Fields<'a> {
    fn new(bytes: &'a [u8]) -> Fields<'a> {
        Fields { bytes, at: 0 }
    }

    /// The next field, skipping whitespace and `#` comments, or `None` at the end.
    fn next(&mut self) -> Option<&'a [u8]> {
        loop {
            while self.bytes.get(self.at)?.is_ascii_whitespace() {
                self.at += 1;
            }
            if self.bytes[self.at] != b'#' {
                break;
            }
            while *self.bytes.get(self.at)? != b'\n' {
                self.at += 1;
            }
        }
        let start = self.at;
        while self.bytes.get(self.at).is_some_and(|b| !b.is_ascii_whitespace()) {
            self.at += 1;
        }
        Some(&self.bytes[start..self.at])
    }

    /// Where the pixels begin: the single whitespace byte that ends the header, and then
    /// no further skipping - a pixel whose value happens to be 32 is not a separator.
    fn raster_at(&self) -> usize {
        self.at + 1
    }
}

fn number(field: &[u8]) -> Option<usize> {
    as_text(field)?.parse().ok()
}

fn as_text(field: &[u8]) -> Option<&str> {
    std::str::from_utf8(field).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_strobe_is_learned_from_the_feed_and_only_when_it_is_there() {
        // One frame seen: the cutoff is that frame, so it reads lit. Throwing away the
        // first frame of every authentication would cost a symbol of the gesture.
        let mut seen = Seen::first(Brightness::level(60.0));
        assert!(seen.cutoff().is_none());

        // A dark frame arrives and there are two populations, 5 against 60 as measured.
        seen.include(Brightness::level(5.0));
        let cutoff = seen.cutoff().expect("5 against 60 is a strobe");
        assert!(Brightness::level(5.0) < cutoff && cutoff < Brightness::level(60.0));
        // And the range only ever widens, so a brighter frame does not lose the split.
        seen.include(Brightness::level(95.0));
        assert!(seen.cutoff().is_some());

        // A camera whose emitter does not strobe: a few levels of wobble is one
        // population, and splitting it would discard half of a perfectly good feed.
        let mut steady = Seen::first(Brightness::level(48.0));
        steady.include(Brightness::level(52.0));
        assert!(steady.cutoff().is_none(), "4 grey levels is noise, not a strobe");
    }

    #[test]
    fn reads_a_pgm_and_refuses_a_broken_one() {
        let dir = std::env::temp_dir().join("blinkergate-pgm-test");
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("000.pgm");

        // Header, a comment, then four pixels.
        std::fs::write(&path, b"P5\n# recorded by hand\n2 2\n255\n\x00\x40\x80\xff").unwrap();
        let frame = read_pgm(&path).unwrap();
        assert_eq!(frame.size(), Size { width: 2, height: 2 });
        assert_eq!(frame.px(), [0, 0x40, 0x80, 0xff].map(crate::frame::Grey::new));
        // (0 + 64 + 128 + 255) / 4, which is what the strobe compares to its cutoff.
        assert!(frame.brightness() > Brightness::level(111.7));
        assert!(frame.brightness() < Brightness::level(111.8));

        // A raster shorter than the header claims is the failure that matters: it
        // would otherwise be read as a frame whose bottom rows are someone else's
        // memory.
        std::fs::write(&path, b"P5\n2 2\n255\n\x00\x40").unwrap();
        assert!(read_pgm(&path).is_err());
        std::fs::write(&path, b"P2\n2 2\n255\n1 2 3 4").unwrap();
        assert!(read_pgm(&path).is_err());
        // A header that runs out, and one whose size is not a number.
        std::fs::write(&path, b"P5\n2\n").unwrap();
        assert!(read_pgm(&path).is_err());
        std::fs::write(&path, b"P5\ntwo 2\n255\n\x00\x00\x00\x00").unwrap();
        assert!(read_pgm(&path).is_err());
        // 16-bit PGM: a real format this cannot read, and reading it as 8-bit would
        // halve the picture rather than fail.
        std::fs::write(&path, b"P5\n2 2\n65535\n\x00\x00\x00\x00\x00\x00\x00\x00").unwrap();
        assert!(read_pgm(&path).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
