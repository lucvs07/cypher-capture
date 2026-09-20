"""
Checklist global de EPI da empresa (achado de campo 2026-09-20).

Escopo default usado pela inferencia de ausencia (``epi_absence.py``) quando a
pessoa nao esta numa zona com ``required_ppe`` proprio — zona, quando resolve
algum item, tem prioridade sobre este checklist. Mesma escada de degradacao de
``config_client.py`` (API -> cache em disco -> None), replicada de proposito
em vez de compartilhada: e um endpoint global (sem camera_id), sem placeholder
de credencial pra expandir.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

log = logging.getLogger(__name__)


class EpiChecklistTransport(Protocol):
    def get_json(self, url: str, api_key: Optional[str] = None) -> tuple[int, dict]: ...


class RequestsEpiChecklistTransport:
    """Transporte real. Importa requests preguiçosamente para manter os
    testes livres da dependência."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def get_json(self, url: str, api_key: Optional[str] = None) -> tuple[int, dict]:
        import requests
        headers = {"X-Cypher-Key": api_key} if api_key else None
        resposta = requests.get(url, timeout=self.timeout, headers=headers)
        try:
            corpo = resposta.json()
        except ValueError:
            corpo = {}
        return resposta.status_code, corpo


class EpiChecklistClient:
    """1. API responde -> usa e grava o cache.
    2. API fora       -> usa o cache, se houver.
    3. sem cache       -> devolve None, e o chamador mantem o default do YAML.
    """

    def __init__(self, api_url: str,
                 cache_path: str = "outputs/epi-checklist-cache.json",
                 transport: Optional[EpiChecklistTransport] = None,
                 now: Optional[Callable[[], datetime]] = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.cache_path = Path(cache_path)
        self.transport = transport or RequestsEpiChecklistTransport()
        self._now = now or (lambda: datetime.now(timezone.utc))

    def fetch(self, api_key: Optional[str] = None) -> Optional[list[str]]:
        epis = self._da_api(api_key)
        if epis is not None:
            self._gravar_cache(epis)
            return epis
        return self._do_cache()

    def _da_api(self, api_key: Optional[str] = None) -> Optional[list[str]]:
        url = f"{self.api_url}/config/epis"
        try:
            status, corpo = self.transport.get_json(url, api_key)
        except Exception as falha:
            log.warning("Checklist global de EPI indisponivel na API: %s", falha)
            return None

        if status == 200:
            return list(corpo.get("epis") or [])

        log.warning("Checklist global de EPI: API respondeu %s", status)
        return None

    def _gravar_cache(self, epis: list[str]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(
                {"fetched_at": self._now().isoformat(), "epis": epis}, indent=2))
        except OSError as erro:
            log.warning("Nao foi possivel gravar o cache do checklist de EPI: %s", erro)

    def _do_cache(self) -> Optional[list[str]]:
        if not self.cache_path.exists():
            return None
        try:
            registro = json.loads(self.cache_path.read_text()) or {}
        except (OSError, json.JSONDecodeError):
            log.warning("Cache do checklist de EPI ilegivel, ignorando: %s", self.cache_path)
            return None
        epis = registro.get("epis")
        return list(epis) if epis is not None else None
