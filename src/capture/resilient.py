"""
Abertura de vídeo que sobrevive a câmera real.

Três problemas que o laço de frames não deveria ter de conhecer:

  * a câmera cai — reconectar com backoff, sem derrubar o processo;
  * a câmera trava conectada sem enviar — ``read()`` bloqueia para sempre;
  * o processamento é mais lento que a fonte — o buffer cresce em silêncio até
    se estar analisando o passado.

Para fonte ao vivo, os três são resolvidos por uma thread de leitura que
guarda apenas o frame mais recente (ver ``_LiveReader``). Para arquivo, nada
disso se aplica: descartar frame mudaria o conteúdo analisado, e fim de
arquivo é fim.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol

import cv2
import numpy as np

from ..utils.video import is_live_source, resolve_source

log = logging.getLogger(__name__)

LARGURA_DEFAULT = 1280
ALTURA_DEFAULT = 720
FPS_DEFAULT = 15.0


class CaptureLike(Protocol):
    def isOpened(self) -> bool: ...
    def read(self) -> tuple[bool, Optional[np.ndarray]]: ...
    def release(self) -> None: ...
    def get(self, prop: int) -> float: ...
    def set(self, prop: int, value: float) -> bool: ...


class ResilientCapture:
    """Fonte de vídeo com comportamento distinto para arquivo e câmera."""

    def __init__(self, source,
                 factory: Optional[Callable[[], CaptureLike]] = None,
                 backoff_base: float = 0.5,
                 backoff_max: float = 10.0) -> None:
        self.source = source
        self.live = is_live_source(source)
        self.backoff_base = backoff_base
        self.backoff_max = backoff_max
        self._factory = factory or (lambda: cv2.VideoCapture(resolve_source(source)))

        self._cap: Optional[CaptureLike] = None
        self.state = "fechado"
        self.reconnects = 0
        self.frames_descartados = 0

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def open(self) -> "ResilientCapture":
        cap = self._factory()
        if not cap.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir a fonte de video '{self.source}'")
        self._cap = cap
        self.state = "conectado"
        return self

    def isOpened(self) -> bool:  # noqa: N802 (espelha a API do OpenCV)
        return self._cap is not None and self.state in ("conectado", "reconectando")

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.state != "encerrado":
            self.state = "fechado"

    # ------------------------------------------------------------------
    # Leitura
    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None

        ok, frame = self._cap.read()
        if ok:
            return True, frame

        # Arquivo: o fim é o fim. Reconectar aqui reiniciaria o clipe e
        # duplicaria a análise.
        self.state = "encerrado"
        return False, None

    # ------------------------------------------------------------------
    # Metadados
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._inteiro(cv2.CAP_PROP_FRAME_WIDTH, LARGURA_DEFAULT)

    @property
    def height(self) -> int:
        return self._inteiro(cv2.CAP_PROP_FRAME_HEIGHT, ALTURA_DEFAULT)

    @property
    def fps(self) -> float:
        if self._cap is None:
            return FPS_DEFAULT
        valor = self._cap.get(cv2.CAP_PROP_FPS)
        return valor if valor and valor > 0 else FPS_DEFAULT

    def _inteiro(self, prop: int, default: int) -> int:
        if self._cap is None:
            return default
        valor = int(self._cap.get(prop) or 0)
        return valor if valor > 0 else default
