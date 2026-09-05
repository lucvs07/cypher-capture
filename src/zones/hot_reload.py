"""
Hot-reload de zonas de risco (design doc de zonas de risco, §3.4/§7).

Sem isso, uma zona criada ou editada pela UI só valia a partir do próximo
boot do pipeline daquela câmera — registrado deliberadamente fora de escopo
na entrega original, movido pra dentro depois do teste manual ponta a ponta.

`ZoneHotReloader` roda a busca (`ZoneConfigClient.fetch`) periodicamente numa
thread separada, para não travar o loop de frames com uma chamada de rede
síncrona (timeout padrão de 5s do transporte real seria um stall real no
processamento). Só chama `on_zonas` quando a lista de zonas efetivamente
muda — comparação por igualdade estrutural de `Zone` (dataclass), então
ciclos sem mudança não recriam `ZoneManager`/`ZoneDetector` à toa.
"""
from __future__ import annotations

import logging
import threading
from typing import Callable, List, Optional, Protocol

from .manager import Zone

log = logging.getLogger(__name__)


class ZonasFetcher(Protocol):
    def fetch(self, camera_id: str, width: int, height: int,
              api_key: Optional[str] = None) -> Optional[List[Zone]]: ...


class ZoneHotReloader:
    def __init__(self, cliente: ZonasFetcher, camera_id: str, width: int, height: int,
                 on_zonas: Callable[[List[Zone]], None],
                 interval_seconds: float = 30.0,
                 api_key: Optional[str] = None) -> None:
        self.cliente = cliente
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.on_zonas = on_zonas
        self.interval_seconds = interval_seconds
        self.api_key = api_key
        self._ultima: Optional[List[Zone]] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def marcar_zonas_atuais(self, zonas: List[Zone]) -> None:
        """Registra o que o boot síncrono já aplicou, para o primeiro tick()
        em segundo plano não reaplicar (e reimprimir) a mesma lista."""
        self._ultima = zonas

    def tick(self) -> bool:
        """Busca zonas uma vez. Devolve True só quando aplicou uma mudança —
        unidade testável sem depender de threading real; start()/stop() só
        agendam chamadas a este método."""
        try:
            zonas = self.cliente.fetch(self.camera_id, self.width, self.height,
                                       api_key=self.api_key)
        except Exception:
            log.exception("Falha ao recarregar zonas de %s em segundo plano; "
                          "tentando de novo no proximo ciclo", self.camera_id)
            return False

        if zonas is None or zonas == self._ultima:
            return False
        self._ultima = zonas
        self.on_zonas(zonas)
        return True

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="zone-hot-reload")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            self.tick()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
