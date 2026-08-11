"""
Tradução de transições de estado em episódios.

A máquina de estados do classificador já existe e é testada; o que faltava era
expor suas transições como incidentes. Três coisas que este módulo faz e que
não são opcionais, porque foram medidas em cenario2.mp4 (spec §4.1.2):

  * janela de merge — sem ela, 2 quedas reais viram 10 incidentes;
  * eviction de track sumido — sem ela, 2 incidentes ficam abertos para sempre;
  * guard de idade do track — o ByteTrack produz tracks espúrios de poucos
    frames, e um deles abriria um FALL_RISK a 3 frames do fim do vídeo.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from ..fusion.engine import PersonEvent
from .incident import (
    POSTURE_TO_TYPE,
    ClosedReason,
    IncidentReport,
    build_incident_key,
)

Handler = Callable[[IncidentReport], None]

# Chave de um episódio em curso: o par (track, tipo). O mesmo track pode ter
# um FALL_RISK e um ERGONOMIC_RISK em momentos distintos.
IncidentKey = tuple[int, str]


@dataclass
class TrackerConfig:
    camera_id: str
    run_id: str
    min_severity: int = 2
    merge_window_seconds: float = 15.0
    evict_after_frames: int = 45
    min_track_frames: int = 15
    fps: float = 15.0


class IncidentTracker:
    """Converte eventos por frame em incidentes com início e fim."""

    def __init__(self, config: TrackerConfig,
                 on_open: Handler, on_close: Handler) -> None:
        self.config = config
        self._on_open = on_open
        self._on_close = on_close
        self._open: dict[IncidentKey, IncidentReport] = {}
        self._last_close: dict[IncidentKey, datetime] = {}
        # Relatórios já fechados que ainda estão dentro da janela de merge.
        self._merged: dict[IncidentKey, IncidentReport] = {}
        self._frames_seen: dict[int, int] = {}
        self._last_frame_seen: dict[int, int] = {}
        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Laço principal
    # ------------------------------------------------------------------

    def process(self, events: list[PersonEvent], now: datetime) -> None:
        self._frame_idx += 1

        for event in events:
            self._frames_seen[event.person_id] = self._frames_seen.get(event.person_id, 0) + 1
            self._last_frame_seen[event.person_id] = self._frame_idx
            self._handle(event, now)

        self._evict_lost_tracks(now)

    def flush(self, now: datetime) -> None:
        """Fecha tudo o que restou aberto. Chamado no encerramento do pipeline.

        Sem isto, toda execução deixa incidentes órfãos: um trabalhador que cai
        e é retirado do quadro ficaria no painel como FALL_RISK eternamente
        aberto, sem duração e sem severidade de pico.
        """
        for key in list(self._open):
            self._close(key, now, ClosedReason.PIPELINE_STOPPED)

    def open_incidents(self) -> dict[IncidentKey, IncidentReport]:
        return dict(self._open)

    # ------------------------------------------------------------------
    # Transições
    # ------------------------------------------------------------------

    def _handle(self, event: PersonEvent, now: datetime) -> None:
        atual = POSTURE_TO_TYPE.get(event.posture)
        anterior = POSTURE_TO_TYPE.get(event.previous_posture)

        if atual == anterior:
            if atual is not None:
                self._update_peak(event, atual)
            return

        if anterior is not None:
            motivo = ClosedReason.ESCALATED if atual else ClosedReason.STATE_EXIT
            self._close((event.person_id, anterior), now, motivo)

        if atual is not None:
            self._open_or_merge(event, atual, now)

    def _update_peak(self, event: PersonEvent, tipo: str) -> None:
        relatorio = self._open.get((event.person_id, tipo))
        if relatorio is None:
            return
        # A severidade do frame de abertura raramente é a pior do episódio.
        if event.severity > (relatorio.peak_severity or relatorio.severity):
            relatorio.peak_severity = event.severity

    def _open_or_merge(self, event: PersonEvent, tipo: str, now: datetime) -> None:
        if event.severity < self.config.min_severity:
            return
        if self._frames_seen.get(event.person_id, 0) < self.config.min_track_frames:
            return

        key = (event.person_id, tipo)

        fechado_em = self._last_close.get(key)
        if fechado_em is not None:
            desde = (now - fechado_em).total_seconds()
            if desde <= self.config.merge_window_seconds:
                # Reabre o mesmo episódio: o alarme oscilou, não recomeçou.
                anterior = self._merged.pop(key, None)
                if anterior is not None:
                    anterior.ended_at = None
                    anterior.duration_seconds = None
                    anterior.closed_reason = None
                    self._open[key] = anterior
                    self._update_peak(event, tipo)
                    return

        relatorio = IncidentReport(
            incident_key=build_incident_key(
                self.config.camera_id, self.config.run_id, event.person_id, now),
            camera_id=self.config.camera_id,
            run_id=self.config.run_id,
            track_id=event.person_id,
            type=tipo,
            severity=event.severity,
            reason=event.reason,
            posture_category=event.posture,
            zone_status=event.zone_status,
            zone_name=event.zone_name,
            reba_score=event.reba_score,
            action_level=event.action_level,
            epi_present=list(event.epi_present),
            epi_missing=list(event.epi_violations),
            detected_at=now,
            peak_severity=event.severity,
        )
        self._open[key] = relatorio
        self._on_open(relatorio)

    def _close(self, key: IncidentKey, now: datetime, motivo: str) -> None:
        relatorio = self._open.pop(key, None)
        if relatorio is None:
            return
        relatorio.ended_at = now
        relatorio.duration_seconds = (now - relatorio.detected_at).total_seconds()
        relatorio.closed_reason = motivo
        self._last_close[key] = now
        self._merged[key] = relatorio
        self._on_close(relatorio)

    def _evict_lost_tracks(self, now: datetime) -> None:
        """Fecha episódios de tracks que sumiram do quadro.

        As transições dependem de o classificador ser chamado de novo para
        aquele person_id. Se a pessoa sai do quadro ou o ByteTrack perde o id
        sob oclusão, isso nunca acontece.
        """
        limite = self._frame_idx - self.config.evict_after_frames
        for key in list(self._open):
            track_id, _ = key
            if self._last_frame_seen.get(track_id, 0) <= limite:
                self._close(key, now, ClosedReason.TRACK_LOST)
