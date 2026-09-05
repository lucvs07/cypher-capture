"""
De onde o pipeline tira sua fonte de captura.

O cadastro guarda a fonte com placeholders — ``rtsp://${U}:${P}@host/stream``
— e o segredo vive no ambiente do host que abre a câmera. Assim a senha nunca
atravessa a rede, nunca entra no banco e nunca aparece em backup nem em log
de requisição.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol

PLACEHOLDER = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class MissingCredentialError(Exception):
    """Uma variável referenciada pela fonte não existe no ambiente."""

    def __init__(self, variaveis: list[str]) -> None:
        self.variaveis = variaveis
        super().__init__(
            "Variaveis de ambiente ausentes para expandir a fonte da camera: "
            + ", ".join(variaveis)
            + ". Defina-as no .env do host (ou exporte antes de subir o pipeline)."
        )


def expand_placeholders(texto: Optional[str], env: Mapping[str, str]) -> Optional[str]:
    """Substitui ``${VAR}`` pelos valores de ``env``.

    Falha nomeando **todas** as variáveis ausentes: descobrir uma por execução
    é hostil quando faltam usuário e senha. Valor vazio conta como ausente,
    porque produziria uma URL sintaticamente válida e um erro de autenticação
    obscuro na câmera.
    """
    if not texto:
        return texto

    faltando: list[str] = []
    for nome in PLACEHOLDER.findall(texto):
        if not env.get(nome) and nome not in faltando:
            faltando.append(nome)
    if faltando:
        raise MissingCredentialError(faltando)

    return PLACEHOLDER.sub(lambda m: env[m.group(1)], texto)


log = logging.getLogger(__name__)


class CameraDisabledError(Exception):
    """A câmera existe no cadastro e está desativada."""


class CameraNotRegisteredError(Exception):
    """A câmera não existe no cadastro."""


class ConfigTransport(Protocol):
    def get_json(self, url: str, api_key: Optional[str] = None) -> tuple[int, dict]: ...


class RequestsConfigTransport:
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


@dataclass
class CameraConfig:
    """Config resolvida de uma câmera, com a procedência declarada."""

    codigo: str
    nome: str
    estacao_id: str
    fonte: Optional[str]
    stream_port: Optional[int]
    ativa: bool
    origem: str                      # "api" | "cache"
    idade_segundos: float = 0.0

    @property
    def do_cache(self) -> bool:
        return self.origem == "cache"


class CameraConfigClient:
    """Implementa a escada de degradação da spec §2.2.

    1. API responde  -> usa e grava o cache
    2. API fora      -> usa o cache, informando a idade dele
    3. sem cache     -> devolve None, e o chamador cai para a fonte local

    Duas respostas **não** degradam, porque são decisões do cadastro e não
    falhas de rede: 409 (desativada) e 404 (não cadastrada) abortam.
    """

    def __init__(self, api_url: str,
                 cache_path: str = "outputs/camera-config-cache.json",
                 transport: Optional[ConfigTransport] = None,
                 now: Optional[Callable[[], datetime]] = None) -> None:
        self.api_url = api_url.rstrip("/")
        self.cache_path = Path(cache_path)
        self.transport = transport or RequestsConfigTransport()
        self._now = now or (lambda: datetime.now(timezone.utc))

    def fetch(self, camera_id: str, env, api_key: Optional[str] = None) -> Optional[CameraConfig]:
        bruto = self._da_api(camera_id, api_key)
        if bruto is not None:
            self._gravar_cache(camera_id, bruto)
            return self._montar(bruto, env, origem="api", idade=0.0,
                                camera_id=camera_id)

        do_cache = self._do_cache(camera_id)
        if do_cache is None:
            return None
        bruto, idade = do_cache
        return self._montar(bruto, env, origem="cache", idade=idade,
                            camera_id=camera_id)

    # ------------------------------------------------------------------
    # Internos
    # ------------------------------------------------------------------

    def _da_api(self, camera_id: str, api_key: Optional[str] = None) -> Optional[dict]:
        url = f"{self.api_url}/cameras/{camera_id}/config"
        try:
            status, corpo = self.transport.get_json(url, api_key)
        except Exception as falha:
            log.warning("Config de %s indisponivel na API: %s", camera_id, falha)
            return None

        if status == 200:
            return corpo
        if status == 409:
            raise CameraDisabledError(
                corpo.get("erro") or f"Camera desativada no cadastro: {camera_id}")
        if status == 404:
            raise CameraNotRegisteredError(
                f"Camera nao cadastrada: {camera_id}. Cadastre-a ou use --source.")

        log.warning("Config de %s: API respondeu %s", camera_id, status)
        return None

    def _montar(self, bruto: dict, env, origem: str, idade: float,
                camera_id: str) -> CameraConfig:
        config = CameraConfig(
            codigo=bruto.get("codigo", ""),
            nome=bruto.get("nome", ""),
            estacao_id=bruto.get("estacaoId", ""),
            fonte=expand_placeholders(bruto.get("fonte"), env),
            stream_port=bruto.get("streamPort"),
            ativa=bool(bruto.get("ativa", True)),
            origem=origem,
            idade_segundos=idade,
        )
        if not config.ativa:
            # Não depende do backend responder 409: um "200 {ativa: false}"
            # (ou um registro em cache que ficou obsoleto) tem de abortar do
            # mesmo jeito, nunca virar uma CameraConfig usável.
            raise CameraDisabledError(
                f"Camera desativada no cadastro: {camera_id}")
        return config

    def _gravar_cache(self, camera_id: str, bruto: dict) -> None:
        # Guarda a fonte AINDA com placeholder: o cache é um arquivo em disco,
        # e gravar a senha expandida desfaria a razão do placeholder existir.
        try:
            tudo = self._ler_cache_bruto()
            tudo[camera_id] = {
                "fetched_at": self._now().isoformat(),
                "config": bruto,
            }
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(tudo, indent=2))
        except OSError as erro:
            log.warning("Nao foi possivel gravar o cache de config: %s", erro)

    def _do_cache(self, camera_id: str) -> Optional[tuple[dict, float]]:
        registro = self._ler_cache_bruto().get(camera_id)
        if not registro:
            return None
        try:
            gravado_em = datetime.fromisoformat(registro["fetched_at"])
            idade = (self._now() - gravado_em).total_seconds()
        except (KeyError, ValueError):
            idade = 0.0
        return registro.get("config", {}), idade

    def _ler_cache_bruto(self) -> dict:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text()) or {}
        except (OSError, json.JSONDecodeError):
            log.warning("Cache de config ilegivel, ignorando: %s", self.cache_path)
            return {}
