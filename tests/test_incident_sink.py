"""
Testes do envio de incidentes (src/notify/sink.py).

O que está sendo protegido: que nenhum incidente grave se perca. A API pode
cair, a rede pode engasgar e a fila pode encher — em nenhum desses casos uma
queda pode desaparecer em silêncio.

Nenhum teste aqui toca a rede: o transporte é injetado.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone

import pytest

from src.notify.incident import ClosedReason, IncidentReport
from src.notify.sink import HttpIncidentSink

BRT = timezone(timedelta(hours=-3))
T0 = datetime(2026, 8, 10, 14, 0, 0, tzinfo=BRT)


@pytest.fixture(autouse=True)
def spill_isolado(tmp_path, monkeypatch):
    """Isola o spill padrão ("outputs/incidents.jsonl", relativo ao cwd).

    Sem isto, cada sink drena na subida o arquivo deixado pelo teste anterior —
    e o incidente de um teste reaparece nos posts de outro — além de sujar o
    outputs/ do repositório.
    """
    monkeypatch.chdir(tmp_path)


def relatorio(chave: str, severidade: int = 4) -> IncidentReport:
    return IncidentReport(
        incident_key=chave,
        camera_id="CAM-01",
        run_id="run-a",
        track_id=7,
        type="FALL_RISK",
        severity=severidade,
        reason="fallen_sustained",
        posture_category="immediate_risk",
        zone_status="outside",
        detected_at=T0,
        peak_severity=severidade,
    )


class TransporteFake:
    """Registra chamadas e falha sob demanda."""

    def __init__(self, falhas_ate: int = 0) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.patches: list[tuple[str, dict]] = []
        self.falhas_ate = falhas_ate
        self.tentativas = 0
        self.entregou = threading.Event()

    def post(self, url: str, payload: dict) -> None:
        self.tentativas += 1
        if self.tentativas <= self.falhas_ate:
            raise ConnectionError("API indisponivel")
        self.posts.append((url, payload))
        self.entregou.set()

    def patch(self, url: str, payload: dict) -> None:
        self.patches.append((url, payload))
        self.entregou.set()


def test_envia_abertura_para_o_endpoint_de_incidentes():
    transporte = TransporteFake()
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte)
    sink.start()
    try:
        sink.open(relatorio("k1"))
        assert transporte.entregou.wait(timeout=2)
    finally:
        sink.stop()

    url, payload = transporte.posts[0]
    assert url == "http://api/api/v1/incidents"
    assert payload["incidentKey"] == "k1"
    assert payload["severity"] == 4


def test_envia_fechamento_para_o_endpoint_da_chave():
    transporte = TransporteFake()
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte)
    sink.start()
    try:
        rel = relatorio("k1")
        rel.ended_at = T0 + timedelta(seconds=6)
        rel.duration_seconds = 6.0
        rel.closed_reason = ClosedReason.STATE_EXIT
        sink.close(rel)
        assert transporte.entregou.wait(timeout=2)
    finally:
        sink.stop()

    url, payload = transporte.patches[0]
    assert url == "http://api/api/v1/incidents/k1"
    assert payload["durationSeconds"] == 6.0


def test_nao_bloqueia_quem_chama():
    """O laço de frames tem 200ms de orcamento: enfileirar precisa ser barato."""
    class TransporteLento:
        def post(self, url, payload):
            threading.Event().wait(0.5)

        def patch(self, url, payload):
            pass

    sink = HttpIncidentSink("http://api/api/v1", transport=TransporteLento())
    sink.start()
    try:
        import time
        inicio = time.perf_counter()
        sink.open(relatorio("k1"))
        decorrido = time.perf_counter() - inicio
    finally:
        sink.stop(timeout=1.0)

    assert decorrido < 0.05


def test_retenta_ate_entregar():
    transporte = TransporteFake(falhas_ate=2)
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte,
                            backoff_base=0.01)
    sink.start()
    try:
        sink.open(relatorio("k1"))
        assert transporte.entregou.wait(timeout=3)
    finally:
        sink.stop()

    assert transporte.tentativas == 3
    assert len(transporte.posts) == 1


def test_grava_em_disco_quando_desiste(tmp_path):
    spill = tmp_path / "incidents.jsonl"
    transporte = TransporteFake(falhas_ate=99)
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte,
                            spill_path=str(spill), max_attempts=2,
                            backoff_base=0.01)
    sink.start()
    try:
        sink.open(relatorio("k1"))
        for _ in range(100):
            if spill.exists() and spill.read_text().strip():
                break
            threading.Event().wait(0.05)
    finally:
        sink.stop()

    linhas = [json.loads(l) for l in spill.read_text().splitlines() if l.strip()]
    assert linhas[0]["payload"]["incidentKey"] == "k1"
    assert linhas[0]["kind"] == "open"


def test_drena_o_spill_na_subida(tmp_path):
    spill = tmp_path / "incidents.jsonl"
    spill.write_text(json.dumps({
        "kind": "open",
        "payload": relatorio("k-antigo").open_payload(),
    }) + "\n")

    transporte = TransporteFake()
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte,
                            spill_path=str(spill))
    sink.start()
    try:
        assert transporte.entregou.wait(timeout=2)
    finally:
        sink.stop()

    assert transporte.posts[0][1]["incidentKey"] == "k-antigo"
    assert spill.read_text().strip() == ""


def test_fila_cheia_descarta_o_menos_severo_e_nunca_a_queda():
    class TransporteTravado:
        def __init__(self):
            self.liberar = threading.Event()
            self.recebidos = []

        def post(self, url, payload):
            self.liberar.wait(timeout=5)
            self.recebidos.append(payload)

        def patch(self, url, payload):
            pass

    transporte = TransporteTravado()
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte,
                            queue_size=3)
    sink.start()
    try:
        sink.open(relatorio("ocupa-worker", severidade=2))
        sink.open(relatorio("baixa-a", severidade=1))
        sink.open(relatorio("baixa-b", severidade=1))
        sink.open(relatorio("media", severidade=2))
        sink.open(relatorio("queda", severidade=4))   # estoura a fila

        assert sink.dropped() >= 1
        chaves = {item.incident_key for item in sink.snapshot()}
        assert "queda" in chaves
    finally:
        transporte.liberar.set()
        sink.stop()


def test_parada_entrega_o_que_ainda_esta_na_fila():
    """O fechamento final precisa chegar na API, nao no disco.

    run_pipeline chama tracker.flush() e sink.stop() em sequencia dentro do
    finally: os PATCH de PIPELINE_STOPPED entram na fila microssegundos antes
    da parada. Sem drenagem, todos eles iam para o spill e o painel ficava
    com o motivo de fechamento errado ate a proxima execucao.
    """
    transporte = TransporteFake()
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte)
    sink.start()
    sink.open(relatorio("fecha-no-fim"))
    sink.stop()                     # imediatamente depois, sem espera

    assert [p[1]["incidentKey"] for p in transporte.posts] == ["fecha-no-fim"]
    assert not __import__("pathlib").Path("outputs/incidents.jsonl").exists()


def test_parada_grava_em_disco_o_que_nao_deu_tempo_de_entregar():
    """stop() nao pode ser um ralo silencioso.

    O item que ja saiu da fila e estava a caminho quando a parada chegou
    precisa ir para o disco como qualquer outro — e reaparecer na proxima
    subida. Sem isto, encerrar o pipeline durante uma indisponibilidade da API
    perde exatamente o incidente que estava sendo retentado.
    """
    liberado = threading.Event()

    class TransporteQueTrava:
        def post(self, url, payload):
            liberado.wait(timeout=5)
            raise ConnectionError("API indisponivel")

        def patch(self, url, payload):
            pass

    sink = HttpIncidentSink("http://api/api/v1", transport=TransporteQueTrava(),
                            backoff_base=0.01)
    sink.start()
    sink.open(relatorio("em-voo"))
    # Deixa o worker tirar o item da fila e entrar no transporte.
    for _ in range(100):
        if sink.pending() == 0:
            break
        threading.Event().wait(0.01)

    liberado.set()
    sink.stop(timeout=2.0)

    spill = __import__("pathlib").Path("outputs/incidents.jsonl")
    gravados = [json.loads(l) for l in spill.read_text().splitlines() if l.strip()]
    assert [g["incident_key"] for g in gravados] == ["em-voo"]
