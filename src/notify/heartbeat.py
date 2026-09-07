"""
Envio periódico de heartbeat de câmera (design doc de gamificação E5, §2.1).

Mesma tolerância a falha de HttpIncidentSink (src/notify/sink.py): uma API
fora do ar não trava o pipeline — só loga e segue, tentando de novo apenas
depois do próximo intervalo inteiro (nunca a cada frame).
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Mapping, Optional, Protocol

from .sink import resolve_api_key

log = logging.getLogger(__name__)


class HeartbeatTransport(Protocol):
    def post(self, url: str, headers: Optional[dict] = None) -> None: ...


class RequestsHeartbeatTransport:
    """Transporte real. Importa requests preguiçosamente para manter os
    testes livres da dependência."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def post(self, url: str, headers: Optional[dict] = None) -> None:
        import requests
        resposta = requests.post(url, headers=headers, timeout=self.timeout)
        resposta.raise_for_status()


class HeartbeatSender:
    """Manda POST /cameras/{id}/heartbeat no máximo uma vez por
    ``interval_seconds``. ``enviar_se_necessario()`` é barato o bastante para
    ser chamado a cada frame — só há I/O de verdade quando o intervalo já
    passou.
    """

    def __init__(self, api_url: str, camera_id: str,
                 api_key_env: Optional[str] = None,
                 api_key: Optional[str] = None,
                 env: Optional[Mapping[str, str]] = None,
                 interval_seconds: float = 30.0,
                 transport: Optional[HeartbeatTransport] = None,
                 now: Optional[Callable[[], float]] = None) -> None:
        self.url = f"{api_url.rstrip('/')}/cameras/{camera_id}/heartbeat"
        chave = api_key if api_key is not None else resolve_api_key(
            api_key_env, env if env is not None else os.environ)
        self._headers: Optional[dict[str, str]] = (
            {"X-Cypher-Key": chave} if chave is not None else None)
        if self._headers is None:
            log.warning(
                "Heartbeat desativado para %s: sem api_key/api_key_env resolvido "
                "— a metrica de cobertura E5 desta camera ficara zerada.", self.url)

        self.interval = float(interval_seconds)
        if self.interval <= 0:
            log.warning(
                "Heartbeat desativado para %s: interval_seconds=%r nao e positivo.",
                self.url, interval_seconds)
            self._headers = None
        self.transport = transport or RequestsHeartbeatTransport()
        self._now = now or time.monotonic
        self._ultimo_envio: Optional[float] = None

    def enviar_se_necessario(self) -> None:
        agora = self._now()
        if self._ultimo_envio is not None and (agora - self._ultimo_envio) < self.interval:
            return
        self._ultimo_envio = agora

        if self._headers is None:
            return
        try:
            self.transport.post(self.url, headers=self._headers)
        except Exception as erro:
            log.warning("Heartbeat para %s falhou: %s", self.url, erro)
