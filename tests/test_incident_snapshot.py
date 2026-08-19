"""
Testes do anexo tardio do snapshot.

O que está sendo protegido: que o upload nunca entre no caminho da
notificação. Um POST que espera centenas de milissegundos pelo Cloudinary
atrasa o alerta de queda pelo mesmo tanto — e é exatamente o alerta que não
pode esperar.
"""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from src.notify.incident import IncidentReport
from src.notify.sink import HttpIncidentSink

BRT = timezone(timedelta(hours=-3))
T0 = datetime(2026, 8, 10, 14, 0, 0, tzinfo=BRT)


@pytest.fixture(autouse=True)
def spill_isolado(tmp_path, monkeypatch):
    """Mesmo motivo de tests/test_incident_sink.py: o spill padrão é relativo
    ao cwd e vazaria entre testes e para o outputs/ do repositório."""
    monkeypatch.chdir(tmp_path)


def relatorio_com_frame(chave: str) -> IncidentReport:
    return IncidentReport(
        incident_key=chave,
        camera_id="CAM-01",
        run_id="run-a",
        track_id=7,
        type="FALL_RISK",
        severity=4,
        reason="fallen_sustained",
        posture_category="immediate_risk",
        zone_status="outside",
        detected_at=T0,
        peak_severity=4,
        snapshot_frame=np.zeros((48, 64, 3), dtype=np.uint8),
    )


class TransporteFake:
    def __init__(self) -> None:
        self.posts: list[tuple[str, dict]] = []
        self.patches: list[tuple[str, dict]] = []
        self.patch_chegou = threading.Event()

    def post(self, url, payload):
        self.posts.append((url, payload))

    def patch(self, url, payload):
        self.patches.append((url, payload))
        self.patch_chegou.set()


class UploaderFake:
    def __init__(self) -> None:
        self.chamadas: list[str] = []

    def upload(self, frame, public_id: str) -> str:
        self.chamadas.append(public_id)
        return f"https://cdn.exemplo/{public_id}.jpg"


def test_frame_nunca_entra_no_payload():
    payload = relatorio_com_frame("k1").open_payload()

    assert "snapshot_frame" not in payload
    assert "snapshotFrame" not in payload
    assert payload["snapshotUrl"] is None


def test_post_sai_sem_snapshot_e_patch_traz_a_url():
    transporte = TransporteFake()
    uploader = UploaderFake()
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte,
                            uploader=uploader)
    sink.start()
    try:
        sink.open(relatorio_com_frame("k1"))
        assert transporte.patch_chegou.wait(timeout=3)
    finally:
        sink.stop()

    # O POST foi primeiro e sem imagem.
    assert transporte.posts[0][1]["snapshotUrl"] is None
    # O PATCH veio depois, só com a URL.
    url, payload = transporte.patches[0]
    assert url == "http://api/api/v1/incidents/k1"
    assert payload == {"snapshotUrl": "https://cdn.exemplo/k1.jpg"}
    assert uploader.chamadas == ["k1"]


def test_falha_de_upload_nao_derruba_o_incidente():
    class UploaderQuebrado:
        def upload(self, frame, public_id):
            raise RuntimeError("Cloudinary fora do ar")

    transporte = TransporteFake()
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte,
                            uploader=UploaderQuebrado())
    sink.start()
    try:
        sink.open(relatorio_com_frame("k1"))
        import time
        time.sleep(0.3)
    finally:
        sink.stop()

    assert len(transporte.posts) == 1        # o incidente foi entregue
    assert transporte.patches == []          # só o snapshot se perdeu


def test_sem_uploader_configurado_nao_ha_patch():
    transporte = TransporteFake()
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte)
    sink.start()
    try:
        sink.open(relatorio_com_frame("k1"))
        import time
        time.sleep(0.2)
    finally:
        sink.stop()

    assert len(transporte.posts) == 1
    assert transporte.patches == []


def test_incidente_sem_frame_nao_tenta_upload():
    transporte = TransporteFake()
    uploader = UploaderFake()
    sink = HttpIncidentSink("http://api/api/v1", transport=transporte,
                            uploader=uploader)
    sink.start()
    try:
        rel = relatorio_com_frame("k1")
        rel.snapshot_frame = None
        sink.open(rel)
        import time
        time.sleep(0.2)
    finally:
        sink.stop()

    assert uploader.chamadas == []
