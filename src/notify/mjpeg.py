"""
Servidor MJPEG do frame anotado.

Escolhido em vez de WebRTC deliberadamente: entrega latência de cerca de um
frame com zero dependência nova e um <img src> no browser. WebRTC daria
sub-100 ms ao custo de aiortc ou um media server mais sinalização — semanas
de trabalho para um ganho imperceptível na demonstração.
"""
from __future__ import annotations

import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)

FRONTEIRA = "cypherframe"


class MjpegServer:
    """Publica o último frame anotado em multipart/x-mixed-replace."""

    def __init__(self, port: int = 8090, quality: int = 80) -> None:
        self.requested_port = port
        self.quality = quality
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._novo_frame = threading.Condition(self._lock)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def port(self) -> int:
        return self._server.server_address[1] if self._server else self.requested_port

    def publish(self, frame: np.ndarray) -> None:
        """Chamado da thread do frame. Uma atribuição sob lock — sem codificação."""
        with self._novo_frame:
            self._frame = frame
            self._novo_frame.notify_all()

    def has_frame(self) -> bool:
        with self._lock:
            return self._frame is not None

    def start(self) -> None:
        servidor_pai = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"

            def do_GET(self):  # noqa: N802 (assinatura da biblioteca padrão)
                if self.path.split("?")[0] != "/stream":
                    self.send_error(404, "Rota desconhecida")
                    return
                servidor_pai._servir_stream(self)

            def log_message(self, *args):
                pass   # o HUD do pipeline já ocupa o terminal

        self._server = ThreadingHTTPServer(("0.0.0.0", self.requested_port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever,
                                        name="mjpeg-server", daemon=True)
        self._thread.start()
        log.info("MJPEG servindo em http://0.0.0.0:%d/stream", self.port)

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _servir_stream(self, handler: BaseHTTPRequestHandler) -> None:
        handler.send_response(200)
        handler.send_header("Content-Type",
                            f"multipart/x-mixed-replace; boundary={FRONTEIRA}")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()

        ultimo = None
        try:
            while True:
                with self._novo_frame:
                    while self._frame is None or self._frame is ultimo:
                        if not self._novo_frame.wait(timeout=5.0):
                            break
                    if self._frame is None:
                        continue
                    ultimo = self._frame
                    frame = ultimo

                ok, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if not ok:
                    continue

                handler.wfile.write(f"--{FRONTEIRA}\r\n".encode())
                handler.wfile.write(b"Content-Type: image/jpeg\r\n")
                handler.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                handler.wfile.write(jpeg.tobytes())
                handler.wfile.write(b"\r\n")
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # Browser fechou a aba. O pipeline não precisa saber disso.
            pass
