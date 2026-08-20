"""
Abertura de vídeo que sobrevive a câmera real.

Três problemas que o laço de frames não deveria ter de conhecer:

  * a câmera cai — reconectar com backoff, sem derrubar o processo;
  * a câmera trava conectada sem enviar — ``read()`` bloqueia para sempre;
  * o processamento é mais lento que a fonte — o buffer cresce em silêncio até
    se estar analisando o passado.

Para fonte ao vivo, os três são resolvidos por uma thread de leitura que
guarda apenas o frame mais recente (ver ``_loop_leitura``). Para arquivo,
nada disso se aplica: descartar frame mudaria o conteúdo analisado, e fim de
arquivo é fim.
"""
from __future__ import annotations

import logging
import threading
import time
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

        self._parar: Optional[threading.Event] = None
        self._novo: Optional[threading.Condition] = None
        self._thread: Optional[threading.Thread] = None
        self._ultimo: Optional[np.ndarray] = None
        self._serie = 0
        self._serie_lida = 0

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    def open(self) -> "ResilientCapture":
        cap = self._factory()
        if not cap.isOpened():
            raise RuntimeError(f"Nao foi possivel abrir a fonte de video '{self.source}'")
        self._cap = cap
        self.state = "conectado"

        if self.live:
            self._preparar_ao_vivo(cap)
            self._parar = threading.Event()
            self._novo = threading.Condition()
            self._ultimo: Optional[np.ndarray] = None
            self._serie = 0
            self._serie_lida = 0
            self._thread = threading.Thread(target=self._loop_leitura,
                                            name="capture-reader", daemon=True)
            self._thread.start()
        return self

    def isOpened(self) -> bool:  # noqa: N802 (espelha a API do OpenCV)
        return self._cap is not None and self.state in ("conectado", "reconectando")

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self.live:
            return self._read_ao_vivo()
        return self._read_arquivo()

    def release(self) -> None:
        if self.live and getattr(self, "_parar", None) is not None:
            self._parar.set()
            with self._novo:
                self._novo.notify_all()
            if self._thread is not None:
                self._thread.join(timeout=2.0)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.state != "encerrado":
            self.state = "fechado"

    # ------------------------------------------------------------------
    # Arquivo
    # ------------------------------------------------------------------

    def _read_arquivo(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            return False, None
        ok, frame = self._cap.read()
        if ok:
            return True, frame
        # O fim é o fim. Reconectar reiniciaria o clipe e duplicaria a análise.
        self.state = "encerrado"
        return False, None

    # ------------------------------------------------------------------
    # Ao vivo
    # ------------------------------------------------------------------

    def _preparar_ao_vivo(self, cap: CaptureLike) -> None:
        # Best-effort: o backend FFmpeg costuma ignorar, e é justamente por
        # isso que a thread de leitura existe.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _read_ao_vivo(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._novo:
            if self._ultimo is None or self._serie == self._serie_lida:
                return False, None
            self._serie_lida = self._serie
            return True, self._ultimo

    def _loop_leitura(self) -> None:
        """Guarda apenas o frame mais recente.

        Descartar deliberadamente é a única correção confiável para a latência
        que cresce quando o processamento é mais lento que a fonte — e mantém
        o bloqueio de uma câmera muda isolado aqui, longe do laço.
        """
        espera = self.backoff_base
        while not self._parar.is_set():
            cap = self._cap
            if cap is None:
                self._reconectar(espera)
                espera = min(espera * 2, self.backoff_max)
                continue

            ok, frame = cap.read()
            if ok:
                espera = self.backoff_base
                with self._novo:
                    if self._ultimo is not None and self._serie != self._serie_lida:
                        self.frames_descartados += 1
                    self._ultimo = frame
                    self._serie += 1
                    self._novo.notify_all()
                continue

            if self._parar.is_set():
                return
            self._reconectar(espera)
            espera = min(espera * 2, self.backoff_max)

    def _reconectar(self, espera: float) -> None:
        self.state = "reconectando"
        log.warning("Fonte '%s' caiu; reconectando em %.1fs (reconexoes=%d)",
                    self.source, espera, self.reconnects + 1)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        time.sleep(espera)
        if self._parar.is_set():
            return

        self.reconnects += 1
        try:
            nova = self._factory()
        except Exception as falha:
            log.warning("Falha ao reabrir '%s': %s", self.source, falha)
            return
        if not nova.isOpened():
            return
        self._preparar_ao_vivo(nova)
        self._cap = nova
        self.state = "conectado"
        log.info("Fonte '%s' reconectada", self.source)

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
