"""Emissão de incidentes e stream de vídeo ao vivo (Fase 3)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, fields
from typing import Optional

import yaml

from .heartbeat import HeartbeatSender
from .incident import ClosedReason, IncidentReport, POSTURE_TO_TYPE, build_incident_key
from .mjpeg import MjpegServer
from .sink import ConsoleIncidentSink, HttpIncidentSink, MissingApiKeyError
from .tracker import IncidentTracker, TrackerConfig


@dataclass
class NotifyConfig:
    """Seção `notify` do pose_config.yaml. Chaves desconhecidas são ignoradas,
    seguindo o mesmo contrato de PipelineConfig.from_yaml."""

    camera_id: str = "CAM-01"
    api_url: Optional[str] = None
    api_key_env: Optional[str] = None   # nome da env var; o valor nunca fica aqui
    min_severity: int = 2
    merge_window_seconds: float = 15.0
    evict_after_frames: int = 45
    min_track_frames: int = 15
    queue_size: int = 256
    spill_path: str = "outputs/incidents.jsonl"
    stream_port: int = 8090
    stream_enabled: bool = True
    heartbeat_interval_seconds: float = 30.0
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @property
    def enabled(self) -> bool:
        """Sem api_url, o pipeline se comporta exatamente como antes."""
        return bool(self.api_url)

    @classmethod
    def from_yaml(cls, path: str) -> "NotifyConfig":
        with open(path) as arquivo:
            bruto = yaml.safe_load(arquivo) or {}
        secao = bruto.get("notify") or {}
        conhecidos = {f.name for f in fields(cls)} - {"run_id"}
        return cls(**{k: v for k, v in secao.items() if k in conhecidos})


__all__ = [
    "ClosedReason",
    "ConsoleIncidentSink",
    "HeartbeatSender",
    "HttpIncidentSink",
    "IncidentReport",
    "IncidentTracker",
    "MissingApiKeyError",
    "MjpegServer",
    "NotifyConfig",
    "POSTURE_TO_TYPE",
    "TrackerConfig",
    "build_incident_key",
]
