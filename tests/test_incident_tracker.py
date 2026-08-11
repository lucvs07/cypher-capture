"""
Testes do ciclo de vida do incidente (src/notify/).

O que está sendo protegido: a tradução de transições da máquina de estados em
episódios com início e fim. Sem janela de merge, as 2 quedas reais de
cenario2.mp4 produzem 10 incidentes; sem eviction, 2 incidentes ficam abertos
para sempre. Ambos os números foram medidos, não estimados — ver a spec §4.1.2.

Nada aqui exige GPU ou modelo: os eventos são construídos à mão.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.fusion.engine import PersonEvent
from src.notify.incident import ClosedReason
from src.notify.tracker import IncidentTracker, TrackerConfig

BRT = timezone(timedelta(hours=-3))
T0 = datetime(2026, 8, 10, 14, 0, 0, tzinfo=BRT)


class Coletor:
    """Guarda o que o tracker abriu e fechou, na ordem."""

    def __init__(self) -> None:
        self.abertos: list = []
        self.fechados: list = []

    def on_open(self, relatorio):
        self.abertos.append(relatorio)

    def on_close(self, relatorio):
        self.fechados.append(relatorio)


def montar(**overrides):
    coletor = Coletor()
    config = TrackerConfig(camera_id="CAM-01", run_id="run-t", **overrides)
    return IncidentTracker(config, coletor.on_open, coletor.on_close), coletor


def evento(person_id: int, postura: str, anterior: str, severidade: int = 4) -> PersonEvent:
    return PersonEvent(
        person_id=person_id,
        posture=postura,
        previous_posture=anterior,
        severity=severidade,
        reason="fallen_sustained",
    )


def aquecer(tracker, person_id: int, frames: int, quando: datetime) -> None:
    """Dá vida suficiente ao track para passar pelo guard de min_track_frames."""
    for _ in range(frames):
        tracker.process([evento(person_id, "normal", "normal", severidade=0)], quando)


def test_evento_nasce_com_postura_anterior_normal():
    evento = PersonEvent(person_id=1)

    assert evento.previous_posture == "normal"


def test_postura_anterior_e_um_campo_independente_da_atual():
    evento = PersonEvent(person_id=1, posture="immediate_risk",
                         previous_posture="normal")

    assert evento.posture == "immediate_risk"
    assert evento.previous_posture == "normal"


def test_transicao_para_alarme_abre_incidente():
    tracker, coletor = montar()
    aquecer(tracker, 1, 20, T0)

    tracker.process([evento(1, "immediate_risk", "normal")], T0)

    assert len(coletor.abertos) == 1
    assert coletor.abertos[0].type == "FALL_RISK"
    assert coletor.abertos[0].severity == 4


def test_alarme_sustentado_nao_reabre_a_cada_frame():
    tracker, coletor = montar()
    aquecer(tracker, 1, 20, T0)
    tracker.process([evento(1, "immediate_risk", "normal")], T0)

    for i in range(30):
        tracker.process([evento(1, "immediate_risk", "immediate_risk")],
                        T0 + timedelta(seconds=i * 0.05))

    assert len(coletor.abertos) == 1
    assert coletor.fechados == []


def test_volta_ao_normal_fecha_com_duracao_e_pico():
    tracker, coletor = montar()
    aquecer(tracker, 1, 20, T0)
    tracker.process([evento(1, "immediate_risk", "normal", severidade=3)], T0)
    tracker.process([evento(1, "immediate_risk", "immediate_risk", severidade=4)],
                    T0 + timedelta(seconds=2))

    tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                    T0 + timedelta(seconds=5))

    assert len(coletor.fechados) == 1
    fechado = coletor.fechados[0]
    assert fechado.duration_seconds == pytest.approx(5.0)
    assert fechado.peak_severity == 4       # pico, não a severidade de abertura
    assert fechado.closed_reason == ClosedReason.STATE_EXIT


def test_reentrada_dentro_da_janela_reabre_o_mesmo_incidente():
    tracker, coletor = montar(merge_window_seconds=15.0)
    aquecer(tracker, 1, 20, T0)
    tracker.process([evento(1, "immediate_risk", "normal")], T0)
    tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                    T0 + timedelta(seconds=3))

    tracker.process([evento(1, "immediate_risk", "normal")], T0 + timedelta(seconds=10))

    assert len(coletor.abertos) == 1        # não abriu um segundo
    assert tracker.open_incidents()


def test_reentrada_fora_da_janela_abre_incidente_novo():
    tracker, coletor = montar(merge_window_seconds=15.0)
    aquecer(tracker, 1, 20, T0)
    tracker.process([evento(1, "immediate_risk", "normal")], T0)
    tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                    T0 + timedelta(seconds=3))

    tracker.process([evento(1, "immediate_risk", "normal")], T0 + timedelta(seconds=40))

    assert len(coletor.abertos) == 2


def test_escalacao_fecha_um_tipo_e_abre_o_outro():
    tracker, coletor = montar()
    aquecer(tracker, 1, 20, T0)
    tracker.process([evento(1, "ergonomically_inadequate", "normal", severidade=2)], T0)

    tracker.process([evento(1, "immediate_risk", "ergonomically_inadequate")],
                    T0 + timedelta(seconds=2))

    assert [r.type for r in coletor.abertos] == ["ERGONOMIC_RISK", "FALL_RISK"]
    assert coletor.fechados[0].type == "ERGONOMIC_RISK"
    assert coletor.fechados[0].closed_reason == ClosedReason.ESCALATED


def test_severidade_abaixo_do_limiar_nao_abre_incidente():
    tracker, coletor = montar(min_severity=3)
    aquecer(tracker, 1, 20, T0)

    tracker.process([evento(1, "ergonomically_inadequate", "normal", severidade=2)], T0)

    assert coletor.abertos == []


def test_track_muito_novo_nao_abre_incidente():
    tracker, coletor = montar(min_track_frames=15)

    # Só 3 frames de vida: o ByteTrack produz tracks espúrios curtos.
    aquecer(tracker, 4, 3, T0)
    tracker.process([evento(4, "immediate_risk", "normal")], T0)

    assert coletor.abertos == []


def test_incidentes_de_tracks_diferentes_sao_independentes():
    tracker, coletor = montar()
    aquecer(tracker, 1, 20, T0)
    aquecer(tracker, 2, 20, T0)

    tracker.process([
        evento(1, "immediate_risk", "normal"),
        evento(2, "ergonomically_inadequate", "normal", severidade=2),
    ], T0)

    assert len(coletor.abertos) == 2
    assert {r.track_id for r in coletor.abertos} == {1, 2}
