"""
Contrato do incidente: o que sai daqui é exatamente o que a API recebe.

Este é o único módulo que conhece os nomes de campo do fio (camelCase); o
restante do pacote trabalha com o dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

# Categoria de postura -> tipo de incidente (spec §3.4).
# "normal" não aparece: não gera incidente.
POSTURE_TO_TYPE: dict[str, str] = {
    "immediate_risk": "FALL_RISK",
    "ergonomically_inadequate": "ERGONOMIC_RISK",
    "critical_zone": "ZONE_VIOLATION",
}


class ClosedReason:
    """Por que o episódio foi encerrado.

    TRACK_LOST e PIPELINE_STOPPED indicam duração de confiança menor: o risco
    pode não ter cessado, o sistema é que deixou de observar.
    """

    STATE_EXIT = "STATE_EXIT"
    ESCALATED = "ESCALATED"
    TRACK_LOST = "TRACK_LOST"
    PIPELINE_STOPPED = "PIPELINE_STOPPED"


def build_incident_key(camera_id: str, run_id: str, track_id: int,
                       detected_at: datetime) -> str:
    """Chave de idempotência.

    track_id é um id do ByteTrack que reinicia a cada execução do pipeline, por
    isso sozinho não identifica ninguém; run_id é o que a torna estável.
    O instante é normalizado em UTC para que a chave não dependa do fuso do
    host que a gerou.
    """
    instante = detected_at.astimezone(timezone.utc)
    carimbo = instante.strftime("%Y%m%dT%H%M%S") + f"{instante.microsecond // 1000:03d}Z"
    return f"{camera_id}:{run_id}:track-{track_id}:{carimbo}"


@dataclass
class IncidentReport:
    """Um episódio de risco, do alarme confirmado até o encerramento."""

    incident_key: str
    camera_id: str
    run_id: str
    track_id: int
    type: str
    severity: int
    reason: str
    posture_category: str
    zone_status: str
    detected_at: datetime
    reba_score: int = 1
    action_level: int = 1
    zone_name: Optional[str] = None
    epi_present: list[str] = field(default_factory=list)
    epi_missing: list[str] = field(default_factory=list)
    snapshot_url: Optional[str] = None
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    peak_severity: Optional[int] = None
    closed_reason: Optional[str] = None

    def open_payload(self) -> dict:
        """Corpo do POST de abertura. Nunca carrega dados de fechamento."""
        return {
            "incidentKey": self.incident_key,
            "cameraId": self.camera_id,
            "runId": self.run_id,
            "trackId": self.track_id,
            "type": self.type,
            "severity": self.severity,
            "reason": self.reason,
            "postureCategory": self.posture_category,
            "rebaScore": self.reba_score,
            "actionLevel": self.action_level,
            "zoneName": self.zone_name,
            "zoneStatus": self.zone_status,
            "epiPresent": list(self.epi_present),
            "epiMissing": list(self.epi_missing),
            "detectedAt": self.detected_at.isoformat(),
            "snapshotUrl": self.snapshot_url,
        }

    def close_payload(self) -> dict:
        """Corpo do PATCH.

        Só os campos preenchidos entram: o mesmo endpoint serve ao fechamento
        do episódio e à chegada tardia do snapshot, e enviar ``None`` apagaria
        o que já foi gravado.
        """
        payload: dict = {}
        if self.ended_at is not None:
            payload["endedAt"] = self.ended_at.isoformat()
        if self.duration_seconds is not None:
            payload["durationSeconds"] = self.duration_seconds
        if self.peak_severity is not None:
            payload["peakSeverity"] = self.peak_severity
        if self.closed_reason is not None:
            payload["closedReason"] = self.closed_reason
        if self.snapshot_url is not None:
            payload["snapshotUrl"] = self.snapshot_url
        return payload
