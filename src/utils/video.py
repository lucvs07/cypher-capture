from __future__ import annotations

from pathlib import Path

import cv2


def resolve_source(source: int | str | Path) -> int | str:
    """Normalise a capture source for cv2.VideoCapture.

    Ints pass through, numeric strings (including negatives like "-1",
    OpenCV's auto-select index) become ints, anything else is treated as a
    file path / URL string.
    """
    if isinstance(source, int):
        return source
    s = str(source)
    try:
        return int(s)
    except ValueError:
        return s


def open_capture(source: int | str | Path) -> cv2.VideoCapture:
    """Open a video source, raising a clear error instead of silently
    yielding a capture that reads zero frames."""
    cap = cv2.VideoCapture(resolve_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source '{source}'")
    return cap
