"""
Zonas de risco vindas do cadastro remoto (design doc de zonas de risco, §3.1).

Réplica deliberada do padrão de ``src/capture/config_client.py``
(``CameraConfigClient``), não uma abstração compartilhada — zona e câmera são
conceitos independentes o bastante para não forçar uma base comum. A
diferença central: aqui SEMPRE há conversão de coordenada (normalizado ->
pixel), porque o backend nunca abre a câmera e não tem como saber a
resolução real do frame; quem converte é este cliente, com a resolução que
o processo já tem em mãos depois de abrir a captura.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Protocol

from .manager import Zone

log = logging.getLogger(__name__)


class ZoneConfigTransport(Protocol):
    def get_json(self, url: str) -> tuple[int, object]: ...


class RequestsZoneConfigTransport:
    """Transporte real. Importa requests preguiçosamente para manter os
    testes livres da dependência."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout

    def get_json(self, url: str) -> tuple[int, object]:
        import requests
        resposta = requests.get(url, timeout=self.timeout)
        try:
            corpo = resposta.json()
        except ValueError:
            corpo = []
        return resposta.status_code, corpo


class ZoneConfigClient:
    """Escada de degradação da spec §3.1:

    1. API responde   -> converte normalizado->pixel com a resolução atual, grava cache normalizado
    2. API fora, cache -> converte o cache (normalizado) com a resolução atual
    3. sem cache      -> None; o chamador cai para o arquivo local (pixel absoluto, já existente hoje)

    Zero zonas em qualquer nível é aceito (spec §5) — nenhuma exceção
    própria, diferente de CameraConfigClient (zona não tem "desativada no
    cadastro" barrando o boot).
    """

    def __init__(self, api_url: str,
                 cache_path: str = "outputs/zones-config-cache.json",
                 transport: Optional[ZoneConfigTransport] = None,
                 now: Optional[Callable[[], datetime]] = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.cache_path = Path(cache_path)
        self.transport = transport or RequestsZoneConfigTransport()
        self._now = now or (lambda: datetime.now(timezone.utc))

    def fetch(self, camera_id: str, width: int, height: int) -> Optional[list[Zone]]:
        bruto = self._da_api(camera_id)
        if bruto is not None:
            self._gravar_cache(camera_id, bruto)
            return self._converter(bruto, width, height)

        bruto_cache = self._do_cache(camera_id)
        if bruto_cache is None:
            return None
        return self._converter(bruto_cache, width, height)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _da_api(self, camera_id: str) -> Optional[list[dict]]:
        url = f"{self.api_url}/cameras/{camera_id}/zones/config"
        try:
            status, corpo = self.transport.get_json(url)
        except Exception as falha:
            log.warning("Zonas de %s indisponiveis na API: %s", camera_id, falha)
            return None

        if status == 200:
            return corpo
        log.warning("Zonas de %s: API respondeu %s", camera_id, status)
        return None

    def _converter(self, bruto: list[dict], width: int, height: int) -> list[Zone]:
        zonas: list[Zone] = []
        for item in bruto:
            polygon_px = [(p["x"] * width, p["y"] * height) for p in item["polygon"]]
            zonas.append(Zone(
                zone_id=str(item["id"]),
                name=item["nome"],
                zone_type=item["zoneType"].lower(),
                polygon=polygon_px,
                severity=item["severity"],
                required_ppe=item.get("requiredPpe") or [],
            ))
        return zonas

    def _gravar_cache(self, camera_id: str, bruto: list[dict]) -> None:
        try:
            tudo = self._ler_cache_bruto()
            tudo[camera_id] = {"fetched_at": self._now().isoformat(), "zones": bruto}
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(tudo, indent=2))
        except OSError as erro:
            log.warning("Nao foi possivel gravar o cache de zonas: %s", erro)

    def _do_cache(self, camera_id: str) -> Optional[list[dict]]:
        registro = self._ler_cache_bruto().get(camera_id)
        if not registro:
            return None
        return registro.get("zones", [])

    def _ler_cache_bruto(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text()) or {}
        except (OSError, json.JSONDecodeError):
            log.warning("Cache de zonas ilegivel, ignorando: %s", self.cache_path)
            return {}
