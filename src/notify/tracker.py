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
from typing import Callable, Optional

from ..fusion.engine import PersonEvent
from .incident import (
    POSTURE_TO_TYPE,
    TIPO_EPI_VIOLATION,
    ClosedReason,
    IncidentReport,
    build_incident_key,
)

# Contrato do on_open: o handler precisa pegar snapshot_frame antes de
# retornar (copiar, codificar ou guardar a referencia num item proprio, como o
# HttpIncidentSink faz). Logo depois o tracker zera o campo no relatorio para
# nao prender uma imagem por incidente; um handler que enfileirasse o proprio
# relatorio e lesse o frame depois, num worker, receberia None.
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
    # None preserva o comportamento anterior (episodio fica aberto pra
    # sempre enquanto a violacao persiste) — testes de unidade do tracker
    # nao passam este parametro de proposito, pra nao terem que gerenciar
    # `now` avancando alem da janela em cada cenario que nao e sobre isto.
    # O pipeline real (NotifyConfig) liga com um default concreto.
    renotify_after_seconds: Optional[float] = None


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
        self._frame_atual = None    # frame anotado do frame em processamento
        # EPI e postura sao eixos independentes: a mesma pessoa pode estar,
        # no mesmo frame, com ZONE_VIOLATION (postura) E EPI_VIOLATION (epi)
        # abertos ao mesmo tempo — por isso o estado anterior de EPI nao cabe
        # em previous_posture/POSTURE_TO_TYPE, precisa do proprio dicionario.
        # Espelha "ha EPI_VIOLATION aberto para o track" (ver _handle_epi).
        self._epi_ativo: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Laço principal
    # ------------------------------------------------------------------

    def process(self, events: list[PersonEvent], now: datetime,
                frame=None) -> None:
        self._frame_idx += 1
        self._frame_atual = frame

        for event in events:
            self._frames_seen[event.person_id] = self._frames_seen.get(event.person_id, 0) + 1
            self._last_frame_seen[event.person_id] = self._frame_idx
            self._handle(event, now)

        self._evict_lost_tracks(now)
        self._expirar_janela_de_merge(now)

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
            if atual is not None and not self._talvez_renotificar(event, atual, now):
                self._update_peak(event, atual)
        else:
            if anterior is not None:
                motivo = ClosedReason.ESCALATED if atual else ClosedReason.STATE_EXIT
                self._close((event.person_id, anterior), now, motivo)

            if atual is not None:
                self._open_or_merge(event, atual, now)

        self._handle_epi(event, now)

    def _handle_epi(self, event: PersonEvent, now: datetime) -> None:
        """Eixo independente da postura (achado de campo 2026-09-20): EPI
        ausente confirmado abre/fecha seu proprio incidente, mesmo quando a
        postura da pessoa e normal — ate entao EPI so entrava como dado
        anexo (epiMissing) de um incidente ja aberto por outro motivo."""
        tem_violacao_agora = bool(event.epi_violations)
        # "Tinha antes" e ter um incidente de EPI aberto, nao ter visto a
        # violacao no frame anterior. Com a flag por frame, quem ja entrava
        # sem EPI nunca abria incidente: nos primeiros min_track_frames o
        # guard de _open_or_merge barrava a abertura, a flag virava True
        # mesmo assim e dali em diante so se atualizava o pico de um
        # incidente que nao existia. O mesmo com severidade abaixo de
        # min_severity no primeiro frame da violacao.
        key = (event.person_id, TIPO_EPI_VIOLATION)
        tinha_antes = key in self._open

        if tem_violacao_agora and tinha_antes:
            if not self._talvez_renotificar(event, TIPO_EPI_VIOLATION, now):
                self._update_peak(event, TIPO_EPI_VIOLATION)
        elif not tem_violacao_agora and tinha_antes:
            self._close(key, now, ClosedReason.STATE_EXIT)
        elif tem_violacao_agora:
            self._open_or_merge(event, TIPO_EPI_VIOLATION, now)
        self._epi_ativo[event.person_id] = key in self._open

    def _talvez_renotificar(self, event: PersonEvent, tipo: str, now: datetime) -> bool:
        """Fecha e reabre como incidente novo quando a mesma violacao
        persiste alem de renotify_after_seconds (achado de campo 2026-09-20:
        sem isto, uma violacao continua so gerava 1 alerta na vida inteira do
        episodio). Devolve True quando reabriu — o chamador nao deve tambem
        chamar _update_peak, porque o episodio que existia acabou de fechar."""
        limite = self.config.renotify_after_seconds
        if limite is None:
            return False
        relatorio = self._open.get((event.person_id, tipo))
        if relatorio is None:
            return False
        if (now - relatorio.detected_at).total_seconds() < limite:
            return False
        self._close((event.person_id, tipo), now, ClosedReason.RENOTIFIED)
        # A janela de merge reabriria o mesmo episodio (mesmo motivo de
        # existir) — pular a chave de merge deste fechamento evita isso.
        self._last_close.pop((event.person_id, tipo), None)
        self._merged.pop((event.person_id, tipo), None)
        self._open_or_merge(event, tipo, now)
        return True

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
            reason=event.reason or (
                # Postura normal (reason vazio) e o proprio motivo do
                # incidente ser EPI: usa os itens ausentes como reason em vez
                # de deixar o campo em branco no historico/dashboard.
                ", ".join(t.value for t in event.epi_violations)
                if tipo == TIPO_EPI_VIOLATION else event.reason
            ),
            posture_category=event.posture,
            zone_status=event.zone_status,
            zone_name=event.zone_name,
            reba_score=event.reba_score,
            action_level=event.action_level,
            epi_present=[t.value for t in event.epi_present],
            epi_missing=[t.value for t in event.epi_violations],
            detected_at=now,
            peak_severity=event.severity,
            snapshot_frame=self._frame_atual,
        )
        self._open[key] = relatorio
        try:
            self._on_open(relatorio)
        finally:
            # O sink ja pegou a referencia do frame em on_open (sincrono, ver o
            # contrato em Handler). Manter aqui prendia uma imagem inteira por
            # incidente enquanto o relatorio vivesse em _open/_merged — GBs
            # num processo que roda por dias.
            relatorio.snapshot_frame = None

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

        # Estado por track: o ByteTrack da id novo a cada pessoa que entra no
        # quadro, entao sem isto os dicionarios so crescem ao longo da execucao.
        for track_id in [t for t, visto in self._last_frame_seen.items() if visto <= limite]:
            del self._last_frame_seen[track_id]
            self._frames_seen.pop(track_id, None)
            self._epi_ativo.pop(track_id, None)

    def _expirar_janela_de_merge(self, now: datetime) -> None:
        """Esquece fechamentos cuja janela de merge ja passou.

        So servem para reabrir o mesmo episodio dentro da janela; depois
        dela, _open_or_merge nunca mais os consulta, e sem esta limpeza
        _merged/_last_close cresciam um item por incidente, para sempre.
        """
        janela = self.config.merge_window_seconds
        for key in [k for k, fechado_em in self._last_close.items()
                    if (now - fechado_em).total_seconds() > janela]:
            del self._last_close[key]
            self._merged.pop(key, None)
