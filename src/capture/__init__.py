"""Obtenção da fonte de captura e abertura resiliente de vídeo (E2)."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import MutableMapping

import yaml

from .apikey_provisioning import (
    ApiKeyProvisioner,
    ApiKeyProvisioningError,
    MissingGestorCredentialError,
)
from .config_client import (
    CameraConfig,
    CameraConfigClient,
    CameraDisabledError,
    CameraNotRegisteredError,
    MissingCredentialError,
    expand_placeholders,
)
from .resilient import ResilientCapture


@dataclass
class CaptureConfig:
    """Seção `capture` do pose_config.yaml. Chaves desconhecidas são
    ignoradas, seguindo o contrato de PipelineConfig.from_yaml."""

    rtsp_transport: str = "tcp"
    read_timeout_ms: int = 5000
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 10.0
    cache_path: str = "outputs/camera-config-cache.json"
    zones_cache_path: str = "outputs/zones-config-cache.json"

    @classmethod
    def from_yaml(cls, path: str) -> "CaptureConfig":
        with open(path) as arquivo:
            bruto = yaml.safe_load(arquivo) or {}
        secao = bruto.get("capture") or {}
        conhecidos = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in secao.items() if k in conhecidos})


def aplicar_opcoes_ffmpeg(config: CaptureConfig,
                          env: MutableMapping[str, str]) -> None:
    """Define as opções do backend FFmpeg antes de qualquer captura abrir.

    TCP por padrão: UDP perde pacote sob carga e produz artefato que o
    detector interpreta como movimento — falso positivo de queda causado pela
    rede.

    O timeout de leitura se chama ``stimeout`` no FFmpeg antigo e ``timeout``
    no recente. As duas são declaradas: manter só uma faz a proteção não
    existir na outra versão, silenciosamente.
    """
    if env.get("OPENCV_FFMPEG_CAPTURE_OPTIONS"):
        # O operador sabe mais sobre a câmera dele do que este default.
        return

    micros = int(config.read_timeout_ms) * 1000
    env["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join([
        f"rtsp_transport;{config.rtsp_transport}",
        f"stimeout;{micros}",
        f"timeout;{micros}",
    ])


__all__ = [
    "ApiKeyProvisioner",
    "ApiKeyProvisioningError",
    "CameraConfig",
    "CameraConfigClient",
    "CameraDisabledError",
    "CameraNotRegisteredError",
    "CaptureConfig",
    "MissingCredentialError",
    "MissingGestorCredentialError",
    "ResilientCapture",
    "aplicar_opcoes_ffmpeg",
    "expand_placeholders",
]
