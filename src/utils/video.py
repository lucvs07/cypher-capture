from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

_ROTATE_CODES = {
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def apply_rotation(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate a decoded frame clockwise by ``degrees`` (0/90/180/270).

    Some RTSP cameras are mounted physically turned (corridor-style
    install) and stream the raw sensor orientation with no rotation
    metadata — the scene comes in sideways while anything the camera
    itself burns into the frame (its own on-screen-display) stays upright
    in the *sensor's* coordinate frame, not the real world's. OpenCV can't
    detect this from the stream; it has to be told per camera.
    """
    if degrees == 0:
        return frame
    return cv2.rotate(frame, _ROTATE_CODES[degrees])


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


# Esquemas que representam captura contínua. Um arquivo de vídeo servido por
# HTTP cairia aqui por engano, mas isso não é um uso previsto do projeto — e o
# custo do erro é apenas tentar reconectar no fim, não corromper análise.
LIVE_SCHEMES = ("rtsp://", "rtsps://", "http://", "https://", "udp://", "tcp://")


def is_live_source(source: int | str | Path) -> bool:
    """Se a fonte é captura contínua (câmera) e não mídia finita (arquivo).

    Decide o comportamento de ``ResilientCapture``: ao vivo reconecta e
    descarta frame atrasado; arquivo é lido quadro a quadro e o fim é o fim.
    """
    resolvida = resolve_source(source)
    if isinstance(resolvida, int):
        return True
    return resolvida.lower().startswith(LIVE_SCHEMES)
