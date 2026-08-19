"""
Testes do contrato do incidente (src/notify/incident.py).

O que está sendo protegido: a chave de idempotência e o formato exato dos dois
payloads. Um erro aqui não quebra o pipeline — ele produz alerta duplicado ou
campo silenciosamente ausente no painel, que é pior porque não faz barulho.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.notify.incident import (
    POSTURE_TO_TYPE,
    ClosedReason,
    IncidentReport,
    build_incident_key,
)

BRT = timezone(timedelta(hours=-3))
INSTANTE = datetime(2026, 8, 10, 14, 31, 2, 482000, tzinfo=BRT)


def relatorio(**overrides) -> IncidentReport:
    base = dict(
        incident_key="CAM-01:run-a:track-7:20260810T173102482Z",
        camera_id="CAM-01",
        run_id="run-a",
        track_id=7,
        type="FALL_RISK",
        severity=4,
        reason="fallen_sustained",
        posture_category="immediate_risk",
        reba_score=11,
        action_level=4,
        zone_name=None,
        zone_status="outside",
        epi_present=["HELMET", "VEST"],
        epi_missing=[],
        detected_at=INSTANTE,
    )
    base.update(overrides)
    return IncidentReport(**base)


def test_categorias_mapeiam_para_tipos_de_incidente():
    assert POSTURE_TO_TYPE["immediate_risk"] == "FALL_RISK"
    assert POSTURE_TO_TYPE["ergonomically_inadequate"] == "ERGONOMIC_RISK"
    assert POSTURE_TO_TYPE["critical_zone"] == "ZONE_VIOLATION"


def test_categoria_normal_nao_tem_tipo():
    assert "normal" not in POSTURE_TO_TYPE


def test_chave_combina_camera_execucao_track_e_instante():
    chave = build_incident_key("CAM-01", "run-a", 7, INSTANTE)

    assert chave.startswith("CAM-01:run-a:track-7:")
    # Instante normalizado em UTC, sem separadores: 14:31:02.482-03:00 = 17:31:02.482Z
    assert chave.endswith(":20260810T173102482Z")


def test_chaves_de_execucoes_diferentes_nao_colidem():
    a = build_incident_key("CAM-01", "run-a", 7, INSTANTE)
    b = build_incident_key("CAM-01", "run-b", 7, INSTANTE)

    assert a != b


def test_payload_de_abertura_traz_o_contrato_completo():
    payload = relatorio().open_payload()

    assert payload["incidentKey"] == "CAM-01:run-a:track-7:20260810T173102482Z"
    assert payload["cameraId"] == "CAM-01"
    assert payload["runId"] == "run-a"
    assert payload["trackId"] == 7
    assert payload["type"] == "FALL_RISK"
    assert payload["severity"] == 4
    assert payload["reason"] == "fallen_sustained"
    assert payload["postureCategory"] == "immediate_risk"
    assert payload["rebaScore"] == 11
    assert payload["actionLevel"] == 4
    assert payload["zoneName"] is None
    assert payload["zoneStatus"] == "outside"
    assert payload["epiPresent"] == ["HELMET", "VEST"]
    assert payload["epiMissing"] == []
    assert payload["snapshotUrl"] is None
    assert payload["detectedAt"] == "2026-08-10T14:31:02.482000-03:00"


def test_payload_de_abertura_nao_leva_dados_de_fechamento():
    payload = relatorio(ended_at=INSTANTE, duration_seconds=6.6).open_payload()

    assert "endedAt" not in payload
    assert "durationSeconds" not in payload


def test_payload_de_fechamento_leva_apenas_o_que_esta_preenchido():
    rel = relatorio(
        ended_at=INSTANTE + timedelta(seconds=6.6),
        duration_seconds=6.6,
        peak_severity=4,
        closed_reason=ClosedReason.STATE_EXIT,
    )

    payload = rel.close_payload()

    assert payload["durationSeconds"] == 6.6
    assert payload["peakSeverity"] == 4
    assert payload["closedReason"] == "STATE_EXIT"
    assert payload["endedAt"].startswith("2026-08-10T14:31:09")
    assert "snapshotUrl" not in payload


def test_payload_de_fechamento_serve_ao_snapshot_atrasado():
    payload = relatorio(snapshot_url="https://cdn/snap.jpg").close_payload()

    assert payload == {"snapshotUrl": "https://cdn/snap.jpg"}
