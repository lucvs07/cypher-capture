"""
Testes do ciclo de vida do incidente (src/notify/).

O que está sendo protegido: a tradução de transições da máquina de estados em
episódios com início e fim. Sem janela de merge, as 2 quedas reais de
cenario2.mp4 viram uma rajada de alertas; sem eviction, incidentes ficam
abertos para sempre. Os números vêm de medição, não de estimativa — ver a
spec §4.1.2.

O replay da fixture (transições medidas em cenario2.mp4) fecha em 2 incidentes
com a janela de merge e 8 sem ela. A execução real do pipeline chegou a 10
porque também contava episódios que os guards de idade de track e de
severidade agora barram antes de virarem incidente.

Nada aqui exige GPU ou modelo: os eventos são construídos à mão.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.fusion.engine import PersonEvent
from src.notify.incident import ClosedReason
from src.notify.tracker import IncidentTracker, TrackerConfig
from src.pose.epi_catalog import EpiTipo

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


def passar_janela(tracker, desde: datetime) -> None:
    """Fechamento so e enviado quando a janela de merge vence sem reabertura."""
    tracker.process([], desde + timedelta(seconds=tracker.config.merge_window_seconds + 1))


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
    assert coletor.fechados == []            # ainda pode reabrir na janela
    passar_janela(tracker, T0 + timedelta(seconds=5))

    assert len(coletor.fechados) == 1
    fechado = coletor.fechados[0]
    assert fechado.duration_seconds == pytest.approx(5.0)   # fim real, nao o envio
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
    passar_janela(tracker, T0 + timedelta(seconds=2))

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


# ----------------------------------------------------------------------
# Regressão sobre o log medido de cenario2.mp4
# ----------------------------------------------------------------------

FIXTURE = Path(__file__).parent / "fixtures" / "cenario2_transitions.json"


def _reproduzir_cenario2(**overrides):
    """Replica o log medido de cenario2.mp4 através do tracker.

    Cada frame do vídeo vira uma chamada a process(); os frames sem transição
    repetem a categoria vigente, que é o que o pipeline real produz.
    """
    dados = json.loads(FIXTURE.read_text())
    fps = dados["fps"]
    por_frame = {t["frame"]: t for t in dados["transitions"]}
    vida = {int(k): v for k, v in dados["vida_dos_tracks"].items()}

    parametros = dict(fps=fps, merge_window_seconds=15.0,
                      min_track_frames=15, evict_after_frames=45)
    parametros.update(overrides)     # o teste da janela zerada sobrescreve aqui
    tracker, coletor = montar(**parametros)

    # Frame em que cada track aparece pela primeira vez. A vida do track só diz
    # quantos frames ele durou, não onde começou; o primeiro frame é deduzido
    # dela e recuado até a primeira transição registrada, para que nenhuma
    # transição do log medido seja descartada por um track "ainda ausente".
    primeira_transicao = {}
    for t in dados["transitions"]:
        primeira_transicao.setdefault(t["track"], t["frame"])
    primeiro_frame = {
        tid: min(dados["total_frames"] - v + 1,
                 primeira_transicao.get(tid, dados["total_frames"]))
        for tid, v in vida.items()
    }
    categoria = {tid: "normal" for tid in vida}

    for frame in range(1, dados["total_frames"] + 1):
        agora = T0 + timedelta(seconds=frame / fps)
        eventos = []
        for tid in vida:
            if frame < primeiro_frame[tid]:
                continue
            anterior = categoria[tid]
            transicao = por_frame.get(frame)
            if transicao is not None and transicao["track"] == tid:
                categoria[tid] = transicao["to"]
            severidade = 0 if categoria[tid] == "normal" else 4
            eventos.append(evento(tid, categoria[tid], anterior, severidade))
        tracker.process(eventos, agora)

    tracker.flush(T0 + timedelta(seconds=dados["total_frames"] / fps))
    return tracker, coletor


def test_cenario2_produz_dois_incidentes_para_duas_quedas():
    """Trava o replay, não a contagem da execução ao vivo.

    Aqui as duas quedas caem dentro da janela de merge (intervalo de 11,6 s em
    tempo de vídeo) e viram um FALL_RISK só. A execução real do pipeline sobre
    o mesmo clipe produz DOIS FALL_RISK, porque a janela é medida em tempo de
    relógio e o processamento estica o intervalo para ~17,9 s. Ver
    docs/e1b-verificacao.md §1 — não é regressão, são critérios diferentes.
    """
    _, coletor = _reproduzir_cenario2()

    tipos = sorted(r.type for r in coletor.abertos)
    assert tipos == ["ERGONOMIC_RISK", "FALL_RISK"]


def test_cenario2_nao_deixa_incidente_orfao():
    tracker, coletor = _reproduzir_cenario2()

    assert tracker.open_incidents() == {}
    assert len(coletor.fechados) >= len(coletor.abertos)


def test_cenario2_fecha_o_ultimo_episodio_como_regresso_ao_normal():
    """Depois do gating temporal de queda, a corrida não termina mais com um
    alarme em aberto.

    A fixture antiga tinha um episódio espúrio de ``fall_velocity`` no fim do
    clipe (f1270–1328) que nunca voltava ao ``normal`` e só era fechado à força
    no ``flush`` — daí o ``PIPELINE_STOPPED``. O gating removeu esse falso
    "fall" trailing; agora o último episódio real (a 2ª queda, f1036–1263)
    volta ao ``normal`` bem antes do fim do vídeo e fecha normalmente por
    ``STATE_EXIT`` (a pessoa se recuperou e a janela de saída expirou).
    """
    _, coletor = _reproduzir_cenario2()

    ultimo = coletor.fechados[-1]
    assert ultimo.closed_reason == ClosedReason.STATE_EXIT


def test_sem_janela_de_merge_o_mesmo_clipe_explode_em_alertas():
    """Prova que a janela não é enfeite: sem ela, o painel vira rajada."""
    _, com_janela = _reproduzir_cenario2()
    _, sem_janela = _reproduzir_cenario2(merge_window_seconds=0.0)

    assert len(sem_janela.abertos) > len(com_janela.abertos)


def test_nenhum_track_curto_vira_incidente_no_replay():
    """O guard de ``min_track_frames`` barra tracks espúrios do ByteTrack.

    Na fixture nova, ``vida_dos_tracks`` traz o track 2 (1 frame), a
    pseudo-track ``-1`` (2 frames — id de fallback negativo que o estimador dá
    a detecções ainda não confirmadas) e o track 4 (12 frames), todos abaixo
    dos 15 frames de ``min_track_frames``. Nenhum deles pode abrir incidente;
    só o track 1 (1307 frames) tem vida suficiente.
    """
    _, coletor = _reproduzir_cenario2()

    dados = json.loads(FIXTURE.read_text())
    vida = {int(k): v for k, v in dados["vida_dos_tracks"].items()}
    curtos = {tid for tid, v in vida.items() if v < 15}

    assert curtos, "esperava ao menos um track curto na fixture"
    assert all(r.track_id not in curtos for r in coletor.abertos)


def test_eviction_fecha_track_que_some_do_quadro():
    tracker, coletor = montar(evict_after_frames=10)
    aquecer(tracker, 1, 20, T0)
    tracker.process([evento(1, "immediate_risk", "normal")], T0)

    # Ninguém mais reporta o track 1: a pessoa saiu do quadro.
    for i in range(15):
        tracker.process([], T0 + timedelta(seconds=1 + i * 0.05))
    passar_janela(tracker, T0 + timedelta(seconds=2))

    assert len(coletor.fechados) == 1
    assert coletor.fechados[0].closed_reason == ClosedReason.TRACK_LOST
    assert tracker.open_incidents() == {}


def test_flush_fecha_o_que_restou_aberto():
    tracker, coletor = montar()
    aquecer(tracker, 1, 20, T0)
    tracker.process([evento(1, "immediate_risk", "normal")], T0)

    tracker.flush(T0 + timedelta(seconds=4))

    assert len(coletor.fechados) == 1
    assert coletor.fechados[0].closed_reason == ClosedReason.PIPELINE_STOPPED
    assert coletor.fechados[0].duration_seconds == pytest.approx(4.0)


def evento_epi(person_id: int, epi_violations: list[EpiTipo], severidade: int = 2) -> PersonEvent:
    """Postura normal, fora de zona — o unico motivo de alerta e o EPI."""
    return PersonEvent(
        person_id=person_id,
        posture="normal",
        previous_posture="normal",
        severity=severidade,
        epi_violations=epi_violations,
        zone_status="outside",
    )


class TestEpiComoEixoIndependenteDaPostura:
    """Achado de campo 2026-09-20: ate entao EPI so entrava como dado anexo
    de um incidente ja aberto por outro motivo — postura normal + EPI
    ausente nunca abria incidente nenhum."""

    def test_epi_ausente_com_postura_normal_abre_incidente_proprio(self):
        tracker, coletor = montar()
        aquecer(tracker, 1, 20, T0)

        tracker.process([evento_epi(1, [EpiTipo.HELMET])], T0)

        assert len(coletor.abertos) == 1
        assert coletor.abertos[0].type == "EPI_VIOLATION"
        assert coletor.abertos[0].epi_missing == ["HELMET"]

    def test_track_que_ja_entra_sem_epi_abre_incidente_quando_passa_do_guard(self):
        """Achado 2026-09-22: a flag de EPI virava True no primeiro frame,
        enquanto min_track_frames ainda barrava a abertura, e o incidente
        nunca abria — 0 incidentes em 40 frames sem capacete."""
        tracker, coletor = montar(min_track_frames=15)

        for i in range(40):
            tracker.process([evento_epi(1, [EpiTipo.HELMET])],
                            T0 + timedelta(seconds=i * 0.1))

        assert len(coletor.abertos) == 1
        assert coletor.abertos[0].type == "EPI_VIOLATION"
        assert coletor.abertos[0].detected_at == T0 + timedelta(seconds=1.4)

    def test_epi_abaixo_do_limiar_abre_quando_a_severidade_sobe(self):
        tracker, coletor = montar(min_severity=3)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento_epi(1, [EpiTipo.HELMET], severidade=2)], T0)
        assert coletor.abertos == []

        tracker.process([evento_epi(1, [EpiTipo.HELMET], severidade=3)],
                        T0 + timedelta(seconds=1))

        assert len(coletor.abertos) == 1

    def test_epi_deixa_de_faltar_fecha_o_incidente(self):
        tracker, coletor = montar()
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento_epi(1, [EpiTipo.HELMET])], T0)

        tracker.process([evento_epi(1, [])], T0 + timedelta(seconds=1))
        passar_janela(tracker, T0 + timedelta(seconds=1))

        assert len(coletor.fechados) == 1
        assert coletor.fechados[0].closed_reason == ClosedReason.STATE_EXIT

    def test_epi_e_postura_abrem_incidentes_simultaneos_e_independentes(self):
        tracker, coletor = montar()
        aquecer(tracker, 1, 20, T0)

        evento_combinado = PersonEvent(
            person_id=1, posture="immediate_risk", previous_posture="normal",
            severity=4, reason="fallen_sustained", epi_violations=[EpiTipo.HELMET],
        )
        tracker.process([evento_combinado], T0)

        tipos_abertos = {r.type for r in coletor.abertos}
        assert tipos_abertos == {"FALL_RISK", "EPI_VIOLATION"}

    def test_epi_continuo_sem_renotify_configurado_atualiza_pico_sem_reabrir(self):
        tracker, coletor = montar()  # renotify_after_seconds=None por padrao no teste
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento_epi(1, [EpiTipo.HELMET], severidade=2)], T0)

        tracker.process([evento_epi(1, [EpiTipo.HELMET], severidade=3)],
                         T0 + timedelta(minutes=10))

        assert len(coletor.abertos) == 1
        assert len(coletor.fechados) == 0
        assert tracker.open_incidents()[(1, "EPI_VIOLATION")].peak_severity == 3


class TestRenotificacao:
    """Achado de campo 2026-09-20: violacao continua so gerava 1 alerta na
    vida inteira do episodio — fecha e reabre como incidente novo depois de
    renotify_after_seconds, pra reincidir no kiosk."""

    def test_violacao_persistente_alem_do_limite_fecha_e_reabre(self):
        tracker, coletor = montar(renotify_after_seconds=60.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento_epi(1, [EpiTipo.HELMET])], T0)

        tracker.process([evento_epi(1, [EpiTipo.HELMET])], T0 + timedelta(seconds=61))

        assert len(coletor.abertos) == 2
        assert len(coletor.fechados) == 1
        assert coletor.fechados[0].closed_reason == ClosedReason.RENOTIFIED
        assert coletor.abertos[0].incident_key != coletor.abertos[1].incident_key

    def test_violacao_persistente_antes_do_limite_nao_reabre(self):
        tracker, coletor = montar(renotify_after_seconds=60.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento_epi(1, [EpiTipo.HELMET])], T0)

        tracker.process([evento_epi(1, [EpiTipo.HELMET])], T0 + timedelta(seconds=30))

        assert len(coletor.abertos) == 1
        assert len(coletor.fechados) == 0

    def test_renotificacao_funciona_tambem_para_incidente_de_postura(self):
        tracker, coletor = montar(renotify_after_seconds=60.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "critical_zone", "normal", severidade=3)], T0)

        tracker.process([evento(1, "critical_zone", "critical_zone", severidade=3)],
                         T0 + timedelta(seconds=61))

        assert len(coletor.abertos) == 2
        assert coletor.fechados[0].closed_reason == ClosedReason.RENOTIFIED

    def test_renotificacao_desligada_por_padrao_nao_afeta_episodio_longo(self):
        tracker, coletor = montar()
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "critical_zone", "normal", severidade=3)], T0)

        tracker.process([evento(1, "critical_zone", "critical_zone", severidade=3)],
                         T0 + timedelta(hours=1))

        assert len(coletor.abertos) == 1
        assert len(coletor.fechados) == 0


class TestLiberaMemoria:
    """Achado 2026-09-22: cada incidente fechado ficava em _merged com o frame
    inteiro do snapshot, para sempre — num processo que roda por dias, GBs."""

    def test_relatorio_nao_segura_o_frame_depois_de_entregar_ao_sink(self):
        entregues = []
        config = TrackerConfig(camera_id="CAM-01", run_id="run-t")
        tracker = IncidentTracker(config,
                                  lambda r: entregues.append(r.snapshot_frame),
                                  lambda r: None)
        aquecer(tracker, 1, 20, T0)

        frame = object()
        tracker.process([evento(1, "immediate_risk", "normal")], T0, frame=frame)

        assert entregues == [frame]                  # o sink recebeu o frame
        (relatorio,) = tracker.open_incidents().values()
        assert relatorio.snapshot_frame is None      # o tracker nao segura mais

    def test_fechamento_e_esquecido_depois_da_janela_de_merge(self):
        tracker, _ = montar(merge_window_seconds=15.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))
        assert tracker._merged

        tracker.process([evento(1, "normal", "normal", severidade=0)],
                        T0 + timedelta(seconds=30))

        assert tracker._merged == {}
        assert tracker._last_close == {}

    def test_depois_da_janela_a_mesma_violacao_abre_incidente_novo(self):
        tracker, coletor = montar(merge_window_seconds=15.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))

        tracker.process([evento(1, "immediate_risk", "normal")], T0 + timedelta(seconds=30))

        assert len(coletor.abertos) == 2
        assert coletor.abertos[0].incident_key != coletor.abertos[1].incident_key

    def test_reabre_quando_volta_exatamente_no_limite_da_janela(self):
        tracker, coletor = montar(merge_window_seconds=15.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))

        tracker.process([evento(1, "immediate_risk", "normal")], T0 + timedelta(seconds=18))

        assert len(coletor.abertos) == 1

    def test_frame_e_liberado_mesmo_se_o_sink_falhar(self):
        def sink_que_falha(relatorio):
            raise RuntimeError("sink quebrou")

        config = TrackerConfig(camera_id="CAM-01", run_id="run-t")
        tracker = IncidentTracker(config, sink_que_falha, lambda r: None)
        aquecer(tracker, 1, 20, T0)

        with pytest.raises(RuntimeError):
            tracker.process([evento(1, "immediate_risk", "normal")], T0, frame=object())

        (relatorio,) = tracker.open_incidents().values()
        assert relatorio.snapshot_frame is None

    def test_fechamento_dentro_da_janela_continua_disponivel_para_reabrir(self):
        tracker, coletor = montar(merge_window_seconds=15.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))
        tracker.process([evento(1, "normal", "normal", severidade=0)],
                        T0 + timedelta(seconds=10))

        tracker.process([evento(1, "immediate_risk", "normal")], T0 + timedelta(seconds=12))

        assert len(coletor.abertos) == 1

    def test_estado_de_track_sumido_e_liberado(self):
        tracker, _ = montar(evict_after_frames=5)
        aquecer(tracker, 1, 20, T0)
        for _ in range(10):
            tracker.process([evento(2, "normal", "normal", severidade=0)], T0)

        assert 1 not in tracker._frames_seen
        assert 1 not in tracker._last_frame_seen
        assert 1 not in tracker._epi_ativo
        assert 2 in tracker._frames_seen


class TestFechamentoSoSaiDepoisDaJanela:
    """Achado 2026-09-22: a reabertura dentro da janela de merge acontecia so
    do lado do pipeline — o backend ja tinha recebido o fechamento e mostrava
    o incidente encerrado enquanto o risco continuava ativo."""

    def test_reabertura_na_janela_nunca_envia_o_fechamento_intermediario(self):
        tracker, coletor = montar(merge_window_seconds=15.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))
        tracker.process([evento(1, "immediate_risk", "normal")], T0 + timedelta(seconds=10))

        passar_janela(tracker, T0 + timedelta(seconds=10))

        assert len(coletor.abertos) == 1
        assert coletor.fechados == []            # o episodio continua aberto
        assert tracker.open_incidents()

    def test_episodio_reaberto_fecha_uma_vez_so_com_a_duracao_total(self):
        tracker, coletor = montar(merge_window_seconds=15.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))
        tracker.process([evento(1, "immediate_risk", "normal")], T0 + timedelta(seconds=10))
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=12))

        passar_janela(tracker, T0 + timedelta(seconds=12))

        assert len(coletor.fechados) == 1
        assert coletor.fechados[0].duration_seconds == pytest.approx(12.0)

    def test_flush_envia_fechamento_pendente_na_janela(self):
        tracker, coletor = montar(merge_window_seconds=15.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))

        tracker.flush(T0 + timedelta(seconds=4))

        assert len(coletor.fechados) == 1
        assert coletor.fechados[0].closed_reason == ClosedReason.STATE_EXIT
        assert coletor.fechados[0].duration_seconds == pytest.approx(3.0)

    def test_tick_envia_fechamento_mesmo_sem_frame(self):
        """Camera caiu logo depois do fechamento: sem frame, process() nao
        roda, e o fechamento ficava preso ate a camera voltar."""
        tracker, coletor = montar(merge_window_seconds=15.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))

        tracker.tick(T0 + timedelta(seconds=10))
        assert coletor.fechados == []

        tracker.tick(T0 + timedelta(seconds=19))
        assert len(coletor.fechados) == 1
        assert coletor.fechados[0].duration_seconds == pytest.approx(3.0)

    def test_tick_nao_dispara_eviction(self):
        tracker, coletor = montar(merge_window_seconds=15.0, evict_after_frames=5)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)

        for i in range(100):
            tracker.tick(T0 + timedelta(seconds=1 + i * 0.01))

        assert tracker.open_incidents()
        assert coletor.fechados == []

    def test_frame_depois_da_janela_envia_o_fechamento_antes_da_abertura_nova(self):
        ordem = []
        config = TrackerConfig(camera_id="CAM-01", run_id="run-t",
                               merge_window_seconds=15.0)
        tracker = IncidentTracker(config,
                                  lambda r: ordem.append(("open", r.incident_key)),
                                  lambda r: ordem.append(("close", r.incident_key)))
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))

        tracker.process([evento(1, "immediate_risk", "normal")], T0 + timedelta(seconds=30))

        assert [tipo for tipo, _ in ordem] == ["open", "close", "open"]
        assert ordem[1][1] == ordem[0][1]

    def test_track_perdido_fica_pendente_ate_a_janela_vencer(self):
        tracker, coletor = montar(merge_window_seconds=15.0, evict_after_frames=10)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)
        for i in range(15):
            tracker.process([], T0 + timedelta(seconds=1 + i * 0.05))

        assert coletor.fechados == []
        assert tracker.open_incidents() == {}

        passar_janela(tracker, T0 + timedelta(seconds=2))

        assert len(coletor.fechados) == 1
        assert coletor.fechados[0].closed_reason == ClosedReason.TRACK_LOST

    def test_reaberto_na_janela_e_depois_renotificado_fecha_uma_vez(self):
        tracker, coletor = montar(merge_window_seconds=15.0, renotify_after_seconds=60.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "critical_zone", "normal", severidade=3)], T0)
        tracker.process([evento(1, "normal", "critical_zone", severidade=0)],
                        T0 + timedelta(seconds=3))
        tracker.process([evento(1, "critical_zone", "normal", severidade=3)],
                        T0 + timedelta(seconds=10))

        tracker.process([evento(1, "critical_zone", "critical_zone", severidade=3)],
                        T0 + timedelta(seconds=61))
        passar_janela(tracker, T0 + timedelta(seconds=61))

        assert len(coletor.fechados) == 1
        assert coletor.fechados[0].closed_reason == ClosedReason.RENOTIFIED
        assert coletor.fechados[0].duration_seconds == pytest.approx(61.0)
        assert tracker._merged == {}
        assert tracker._last_close == {}

    def test_desescalada_na_janela_reabre_sem_enviar_o_escalated(self):
        tracker, coletor = montar(merge_window_seconds=15.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "ergonomically_inadequate", "normal", severidade=2)], T0)
        tracker.process([evento(1, "immediate_risk", "ergonomically_inadequate")],
                        T0 + timedelta(seconds=2))

        tracker.process([evento(1, "ergonomically_inadequate", "immediate_risk", severidade=2)],
                        T0 + timedelta(seconds=5))

        assert [r.type for r in coletor.abertos] == ["ERGONOMIC_RISK", "FALL_RISK"]
        assert all(r.type != "ERGONOMIC_RISK" for r in coletor.fechados)
        assert {t for _, t in tracker.open_incidents()} == {"ERGONOMIC_RISK"}

    def test_sem_janela_o_fechamento_sai_na_hora(self):
        tracker, coletor = montar(merge_window_seconds=0.0)
        aquecer(tracker, 1, 20, T0)
        tracker.process([evento(1, "immediate_risk", "normal")], T0)

        tracker.process([evento(1, "normal", "immediate_risk", severidade=0)],
                        T0 + timedelta(seconds=3))

        assert len(coletor.fechados) == 1
