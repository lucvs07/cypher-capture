"""
DecisionEngine.fuse: mescla labels de violacao explicitas (nunca geradas
hoje pelo detector, ver epi_catalog.py) com a inferencia de ausencia por
debounce (epi_missing_confirmed) — achado de campo 2026-09-20.
"""
from src.fusion.engine import DecisionEngine
from src.pose.classifier import PostureCategory, PostureState


def estado_normal() -> PostureState:
    return PostureState(category=PostureCategory.NORMAL, reason="")


class Reba:
    final_score = 1
    action_level = 1


def test_epi_missing_confirmed_entra_em_epi_violations():
    engine = DecisionEngine()

    event = engine.fuse(1, estado_normal(), Reba(), [], epi_labels=[],
                         epi_missing_confirmed=["no_helmet"])

    assert event.epi_violations == ["no_helmet"]


def test_epi_missing_confirmed_ausente_ou_none_nao_quebra():
    engine = DecisionEngine()

    event = engine.fuse(1, estado_normal(), Reba(), [], epi_labels=[])

    assert event.epi_violations == []


def test_severidade_sobe_quando_ha_epi_ausente_mesmo_com_postura_normal():
    engine = DecisionEngine()

    event = engine.fuse(1, estado_normal(), Reba(), [], epi_labels=[],
                         epi_missing_confirmed=["no_helmet"])

    assert event.severity >= 2


def test_epi_missing_confirmed_nao_duplica_com_label_explicita_igual():
    engine = DecisionEngine()

    event = engine.fuse(1, estado_normal(), Reba(), [], epi_labels=["no_helmet"],
                         epi_missing_confirmed=["no_helmet"])

    assert event.epi_violations == ["no_helmet"]
