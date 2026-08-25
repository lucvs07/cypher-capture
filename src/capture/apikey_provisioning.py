"""
Provisionamento automático da apiKey de ingestão da câmera.

Bootstrap só: loga como GESTOR uma vez (usuário/senha do `.env` do host,
nunca em texto puro no YAML), pede a rotação da apiKey da própria câmera via
``POST /cameras/{id}/apikey/regenerate``, e guarda o resultado num cache
local — dali em diante, `X-Cypher-Key` sai do cache, sem logar de novo.

Ao contrário de `CameraConfigClient` (config_client.py), aqui o cache é a
FONTE preferida, não o fallback: a apiKey não expira e regenerá-la a cada
boot rotacionaria à toa (invalidando a anterior sem necessidade). Só
provisiona de novo se o cache não tiver a câmera — o que acontece na
primeira execução, ou se alguém apagar o arquivo de cache manualmente.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Optional, Protocol

log = logging.getLogger(__name__)


class MissingGestorCredentialError(Exception):
    """CYPHER_GESTOR_USER/CYPHER_GESTOR_PASSWORD ausentes no ambiente."""

    def __init__(self) -> None:
        super().__init__(
            "CYPHER_GESTOR_USER e CYPHER_GESTOR_PASSWORD precisam estar "
            "definidos no ambiente (.env do host) para provisionar a apiKey "
            "automaticamente — ou configure notify.api_key_env com uma "
            "chave ja obtida manualmente."
        )


class ApiKeyProvisioningError(Exception):
    """Login ou regenerate falharam — credencial inválida, ou câmera inexistente."""


class ProvisioningTransport(Protocol):
    def login(self, api_url: str, username: str, password: str) -> None: ...
    def regenerate_api_key(self, api_url: str, camera_id: str) -> str: ...


class RequestsProvisioningTransport:
    """Transporte real. Uma única `requests.Session` mantém o cookie de
    sessão entre o login e o regenerate — importa `requests` preguiçosamente
    para manter os testes livres da dependência."""

    def __init__(self, timeout: float = 5.0) -> None:
        self.timeout = timeout
        self._session = None

    def login(self, api_url: str, username: str, password: str) -> None:
        import requests
        self._session = requests.Session()
        resposta = self._session.post(
            f"{api_url}/auth/login", json={"username": username, "password": password},
            timeout=self.timeout)
        if resposta.status_code != 200:
            raise ApiKeyProvisioningError(
                f"Login de provisionamento falhou ({resposta.status_code}) — "
                "confira CYPHER_GESTOR_USER/CYPHER_GESTOR_PASSWORD.")

    def regenerate_api_key(self, api_url: str, camera_id: str) -> str:
        if self._session is None:
            raise ApiKeyProvisioningError("login() precisa rodar antes de regenerate_api_key().")
        resposta = self._session.post(
            f"{api_url}/cameras/{camera_id}/apikey/regenerate", timeout=self.timeout)
        if resposta.status_code == 404:
            raise ApiKeyProvisioningError(f"Camera nao cadastrada: {camera_id}.")
        if resposta.status_code != 200:
            raise ApiKeyProvisioningError(
                f"Regenerate de apiKey falhou ({resposta.status_code}) para {camera_id}.")
        return resposta.json()["apiKey"]


class ApiKeyProvisioner:
    def __init__(self, api_url: str,
                 cache_path: str = "outputs/apikey-cache.json",
                 transport: Optional[ProvisioningTransport] = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.cache_path = Path(cache_path)
        self.transport = transport or RequestsProvisioningTransport()

    def obter(self, camera_id: str, env: Mapping[str, str]) -> str:
        """Cache-first: só chama a API se `camera_id` ainda não tiver chave gravada."""
        do_cache = self._ler_cache_bruto().get(camera_id)
        if do_cache is not None:
            return do_cache

        username = env.get("CYPHER_GESTOR_USER")
        password = env.get("CYPHER_GESTOR_PASSWORD")
        if not username or not password:
            raise MissingGestorCredentialError()

        self.transport.login(self.api_url, username, password)
        chave = self.transport.regenerate_api_key(self.api_url, camera_id)
        self._gravar_cache(camera_id, chave)
        return chave

    def _gravar_cache(self, camera_id: str, chave: str) -> None:
        try:
            tudo = self._ler_cache_bruto()
            tudo[camera_id] = chave
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(tudo, indent=2))
        except OSError as erro:
            log.warning("Nao foi possivel gravar o cache de apiKey: %s", erro)

    def _ler_cache_bruto(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text()) or {}
        except (OSError, json.JSONDecodeError):
            log.warning("Cache de apiKey ilegivel, ignorando: %s", self.cache_path)
            return {}
