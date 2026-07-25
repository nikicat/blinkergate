"""Pick a Qt platform plugin that the installed OpenCV can actually load.

Import this **before** cv2.

mediapipe depends on opencv-contrib-python, whose wheel bundles its own Qt with
only ``libqxcb.so``. On a Wayland session ``QT_QPA_PLATFORM=wayland`` is commonly
exported, and OpenCV then aborts at the first ``imshow`` with "Could not find the
Qt platform plugin" rather than degrading gracefully. Falling back to xcb runs it
through XWayland, which works fine.

A distro OpenCV linked against system Qt does ship the Wayland plugin, so this
only intervenes when the plugin is genuinely absent.
"""

import importlib.util
import os
from pathlib import Path


def _bundled_platforms_dir():
    """Where the installed cv2 keeps its Qt platform plugins, without importing it."""
    spec = importlib.util.find_spec("cv2")
    if not spec or not spec.origin:
        return None
    path = Path(spec.origin).parent / "qt" / "plugins" / "platforms"
    return path if path.is_dir() else None


def apply():
    """Force xcb when the requested platform plugin is missing. Returns the
    platform in effect, or None if we left the environment untouched.

    The bundled Qt also warns about a missing font directory on every window.
    That is noise only - all of this app's text is drawn by OpenCV itself, not
    by Qt - and setting QT_QPA_FONTDIR does not silence it, so it is left alone.
    """
    requested = os.environ.get("QT_QPA_PLATFORM")
    wants_wayland = requested == "wayland" or (
        requested is None and os.environ.get("XDG_SESSION_TYPE") == "wayland")
    if not wants_wayland:
        return None

    plugins = _bundled_platforms_dir()
    if plugins is None:
        return None                                   # system Qt, leave it alone
    if (plugins / "libqwayland.so").exists():
        return None                                   # bundled Qt can do Wayland

    os.environ["QT_QPA_PLATFORM"] = "xcb"
    return "xcb"


applied = apply()
