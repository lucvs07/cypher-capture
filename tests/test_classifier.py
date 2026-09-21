"""
Unit tests for the postural state machine (src/pose/classifier.py).

Focus: the asymmetric temporal hysteresis that underwrites the Phase-2 recall
guarantee (immediate-risk enters fast, every alarm leaves slowly) and the four
posture categories. A silent regression here is exactly what would drop
immediate-risk recall below 0.90, so these are the highest-leverage tests.

API note
--------
The real signature is
    update(person_id, angles: JointAngles, reba: REBAScore, bbox,
           in_critical_zone=False, fall_risk=0.0) -> PostureState
(not the simplified ``update(person_id, reba_score=..., fall_risk=...)`` string
API sketched in the brief). Helpers below build the JointAngles / REBAScore /
bbox triples that drive each category.
"""
from __future__ import annotations

import pytest

from src.pose.angles import JointAngles
from src.pose.reba import REBAScore
from src.pose.classifier import (
    ClassifierThresholds,
    PostureCategory,
    PostureClassifier,
)

# Geometry helpers --------------------------------------------------------
TALL_BBOX = (0.0, 0.0, 50.0, 150.0)    # w/h = 0.33 -> upright
WIDE_BBOX = (0.0, 0.0, 150.0, 50.0)    # w/h = 3.0  -> fallen (aspect > 1.5)


def neutral_angles(pid: int = 1) -> JointAngles:
    """Angles that trip none of the rules (everything None, no rotation)."""
    return JointAngles(person_id=pid, trunk_rotation_proxy=0.0)


def reba(final: int = 1, pid: int = 1) -> REBAScore:
    return REBAScore(person_id=pid, final_score=final,
                     action_level=4 if final >= 8 else 1)


# Per-category single-frame drivers (raw rule, before hysteresis confirms it).
def drive_normal(clf, pid, **kw):
    return clf.update(pid, neutral_angles(pid), reba(1), TALL_BBOX, **kw)


def drive_ergonomic(clf, pid, **kw):
    # REBA at the risk threshold, nothing more severe present.
    return clf.update(pid, neutral_angles(pid), reba(8), TALL_BBOX, **kw)


def drive_fallen(clf, pid, **kw):
    return clf.update(pid, neutral_angles(pid), reba(1), WIDE_BBOX, **kw)


def drive_zone(clf, pid, **kw):
    kw.setdefault("in_critical_zone", True)
    return clf.update(pid, neutral_angles(pid), reba(1), TALL_BBOX, **kw)


# ---------------------------------------------------------------------------
# 1. Hysteresis — asymmetric entry/exit
# ---------------------------------------------------------------------------

class TestImmediateRiskEntry:
    def test_enter_requires_exactly_enter_frames_risk(self):
        clf = PostureClassifier(enter_frames_risk=3, enter_frames=5)
        pid = 1
        assert drive_fallen(clf, pid).category == PostureCategory.NORMAL   # f1
        assert drive_fallen(clf, pid).category == PostureCategory.NORMAL   # f2
        assert drive_fallen(clf, pid).category == PostureCategory.IMMEDIATE_RISK  # f3

    def test_default_config_enters_on_second_frame(self):
        clf = PostureClassifier()  # enter_frames_risk default = 2
        assert drive_fallen(clf, 1).category == PostureCategory.NORMAL
        assert drive_fallen(clf, 1).category == PostureCategory.IMMEDIATE_RISK

    def test_risk_enters_faster_than_ergonomic(self):
        # Same number of frames: risk confirms, ergonomic does not yet.
        risk = PostureClassifier(enter_frames=5, enter_frames_risk=2)
        for _ in range(2):
            r = drive_fallen(risk, 1)
        assert r.category == PostureCategory.IMMEDIATE_RISK

        ergo = PostureClassifier(enter_frames=5, enter_frames_risk=2)
        for _ in range(2):
            e = drive_ergonomic(ergo, 1)
        assert e.category == PostureCategory.NORMAL  # needs 5, only had 2


class TestImmediateRiskExit:
    def test_exit_requires_exit_frames_consecutive_normal(self):
        clf = PostureClassifier(enter_frames_risk=2, exit_frames=3)
        pid = 1
        drive_fallen(clf, pid)
        assert drive_fallen(clf, pid).category == PostureCategory.IMMEDIATE_RISK
        # now feed normal frames; must stay risky until the 3rd one
        assert drive_normal(clf, pid).category == PostureCategory.IMMEDIATE_RISK  # 1
        assert drive_normal(clf, pid).category == PostureCategory.IMMEDIATE_RISK  # 2
        assert drive_normal(clf, pid).category == PostureCategory.NORMAL          # 3

    def test_single_normal_frame_does_not_clear_alarm(self):
        clf = PostureClassifier(enter_frames_risk=2, exit_frames=15)
        drive_fallen(clf, 1)
        drive_fallen(clf, 1)
        assert drive_normal(clf, 1).category == PostureCategory.IMMEDIATE_RISK

    def test_intermittent_risk_resets_exit_counter(self):
        # A flicker back to risk during exit must restart the exit countdown,
        # keeping the alarm sticky (recall priority).
        clf = PostureClassifier(enter_frames_risk=2, exit_frames=3)
        drive_fallen(clf, 1)
        drive_fallen(clf, 1)
        drive_normal(clf, 1)   # exit counter = 1
        drive_normal(clf, 1)   # exit counter = 2
        drive_fallen(clf, 1)   # still risk, resets counter
        drive_normal(clf, 1)   # exit counter = 1 again
        drive_normal(clf, 1)   # 2
        assert clf.get_state(1).category == PostureCategory.IMMEDIATE_RISK
        assert drive_normal(clf, 1).category == PostureCategory.NORMAL  # 3 -> clears


class TestOscillation:
    def test_alternating_frames_never_confirm_entry(self):
        # Entry needs *consecutive* matching frames; a NORMAL frame pops the
        # pending counter, so an alternating signal can never enter an alarm.
        clf = PostureClassifier(enter_frames_risk=2, enter_frames=5)
        for _ in range(20):
            drive_fallen(clf, 1)
            s = drive_normal(clf, 1)
        assert s.category == PostureCategory.NORMAL


# ---------------------------------------------------------------------------
# 2. The four categories
# ---------------------------------------------------------------------------

class TestCategories:
    def test_normal(self):
        clf = PostureClassifier()
        for _ in range(10):
            s = drive_normal(clf, 1)
        assert s.category == PostureCategory.NORMAL

    def test_ergonomically_inadequate_via_reba(self):
        clf = PostureClassifier(enter_frames=3)
        for _ in range(3):
            s = drive_ergonomic(clf, 1)
        assert s.category == PostureCategory.ERGONOMICALLY_INADEQUATE
        assert s.reason.startswith("reba_")

    def test_ergonomically_inadequate_via_deep_squat(self):
        clf = PostureClassifier(enter_frames=3)
        ang = JointAngles(person_id=1, left_knee_flexion=45.0,
                          right_knee_flexion=45.0, trunk_rotation_proxy=0.0)
        for _ in range(3):
            s = clf.update(1, ang, reba(1), TALL_BBOX)
        assert s.category == PostureCategory.ERGONOMICALLY_INADEQUATE
        assert s.reason == "prolonged_squat"

    def test_immediate_risk_via_fallen_bbox(self):
        clf = PostureClassifier(enter_frames_risk=2)
        drive_fallen(clf, 1)
        s = drive_fallen(clf, 1)
        assert s.category == PostureCategory.IMMEDIATE_RISK
        assert s.reason == "person_fallen"

    def test_immediate_risk_via_fall_velocity(self):
        clf = PostureClassifier(enter_frames_risk=2)  # default fall_risk_min = 0.85
        clf.update(1, neutral_angles(), reba(1), TALL_BBOX, fall_risk=0.9)
        s = clf.update(1, neutral_angles(), reba(1), TALL_BBOX, fall_risk=0.9)
        # fall_velocity sozinho (sem geometria de queda) abre a janela de
        # confirmacao (design doc 2026-09-05, §2) em vez de confirmar direto —
        # ver TestFallConfirmation para o ciclo completo de confirmacao/descarte.
        assert s.category == PostureCategory.NORMAL

    def test_immediate_risk_via_horizontal_trunk(self):
        # Torso lying horizontal (trunk flexion ~ 90°) is fallen even with no
        # velocity and an upright-shaped bbox (velocity-independent cue).
        clf = PostureClassifier(enter_frames_risk=2)
        ang = JointAngles(person_id=1, trunk_flexion=80.0, trunk_rotation_proxy=0.0)
        clf.update(1, ang, reba(1), TALL_BBOX)
        s = clf.update(1, ang, reba(1), TALL_BBOX)
        assert s.category == PostureCategory.IMMEDIATE_RISK
        assert s.reason == "person_fallen"

    def test_immediate_risk_via_collapsed_vertical_extent(self):
        # Prone/inverted: nose and lowest foot at similar image height → small
        # vertical_extent → fallen, even with zero velocity and a narrow bbox.
        clf = PostureClassifier(enter_frames_risk=2)
        ang = JointAngles(person_id=1, vertical_extent=0.1, trunk_rotation_proxy=0.0)
        clf.update(1, ang, reba(1), TALL_BBOX)
        s = clf.update(1, ang, reba(1), TALL_BBOX)
        assert s.category == PostureCategory.IMMEDIATE_RISK
        assert s.reason == "person_fallen"

    def test_deep_forward_bend_is_not_fallen(self):
        # Toe-touch: head dips but feet stay planted far below → large
        # vertical_extent, sub-horizontal trunk → must NOT read as fallen.
        clf = PostureClassifier(enter_frames_risk=2)
        ang = JointAngles(person_id=1, trunk_flexion=40.0, vertical_extent=0.7,
                          trunk_rotation_proxy=0.0)
        for _ in range(4):
            s = clf.update(1, ang, reba(1), TALL_BBOX)
        assert s.category != PostureCategory.IMMEDIATE_RISK

    def test_immediate_risk_via_instability(self):
        clf = PostureClassifier(enter_frames_risk=2)
        ang = JointAngles(person_id=1, trunk_rotation_proxy=0.5)  # > 0.35
        clf.update(1, ang, reba(1), TALL_BBOX)
        s = clf.update(1, ang, reba(1), TALL_BBOX)
        assert s.category == PostureCategory.IMMEDIATE_RISK
        assert s.reason == "balance_loss_risk"

    def test_critical_zone(self):
        clf = PostureClassifier(enter_frames=3)
        for _ in range(3):
            s = drive_zone(clf, 1)
        assert s.category == PostureCategory.CRITICAL_ZONE
        assert s.reason == "critical_zone_entry"

    def test_immediate_risk_outranks_zone(self):
        # A fallen person inside a critical zone must read as IMMEDIATE_RISK.
        clf = PostureClassifier(enter_frames_risk=2)
        clf.update(1, neutral_angles(), reba(1), WIDE_BBOX, in_critical_zone=True)
        s = clf.update(1, neutral_angles(), reba(1), WIDE_BBOX, in_critical_zone=True)
        assert s.category == PostureCategory.IMMEDIATE_RISK


# ---------------------------------------------------------------------------
# 3. REBA threshold edge behaviour
# ---------------------------------------------------------------------------

class TestRebaThreshold:
    def test_score_7_is_not_ergonomic(self):
        clf = PostureClassifier(enter_frames=3)
        for _ in range(5):
            s = clf.update(1, neutral_angles(), reba(7), TALL_BBOX)
        assert s.category == PostureCategory.NORMAL

    def test_score_8_is_ergonomic(self):
        clf = PostureClassifier(enter_frames=3)
        for _ in range(3):
            s = clf.update(1, neutral_angles(), reba(8), TALL_BBOX)
        assert s.category == PostureCategory.ERGONOMICALLY_INADEQUATE

    def test_custom_threshold_is_honoured(self):
        th = ClassifierThresholds(reba_risk_score=6)
        clf = PostureClassifier(thresholds=th, enter_frames=2)
        for _ in range(2):
            s = clf.update(1, neutral_angles(), reba(6), TALL_BBOX)
        assert s.category == PostureCategory.ERGONOMICALLY_INADEQUATE


# ---------------------------------------------------------------------------
# 4. Multi-person isolation and tracking edge cases
# ---------------------------------------------------------------------------

class TestMultiPerson:
    def test_states_are_independent_per_person(self):
        clf = PostureClassifier(enter_frames_risk=2, enter_frames=3)
        # person 1 falls, person 2 stays normal
        drive_fallen(clf, 1)
        drive_fallen(clf, 1)
        drive_normal(clf, 2)
        assert clf.get_state(1).category == PostureCategory.IMMEDIATE_RISK
        assert clf.get_state(2).category == PostureCategory.NORMAL

    def test_two_people_distinct_alarms(self):
        clf = PostureClassifier(enter_frames_risk=2, enter_frames=3)
        for _ in range(3):
            drive_fallen(clf, 1)          # immediate risk
            drive_zone(clf, 2)            # critical zone
        assert clf.get_state(1).category == PostureCategory.IMMEDIATE_RISK
        assert clf.get_state(2).category == PostureCategory.CRITICAL_ZONE

    def test_new_track_id_starts_normal(self):
        # When the tracker assigns a fresh ID, that person starts from NORMAL
        # rather than inheriting another person's alarm state.
        clf = PostureClassifier(enter_frames_risk=2)
        drive_fallen(clf, 1)
        drive_fallen(clf, 1)
        assert clf.get_state(1).category == PostureCategory.IMMEDIATE_RISK
        assert clf.get_state(999).category == PostureCategory.NORMAL
        assert drive_fallen(clf, 999).category == PostureCategory.NORMAL  # 1st frame


class TestMissingKeypoints:
    def test_all_none_angles_do_not_crash_and_stay_normal(self):
        # Low-confidence keypoints surface as None angles; with an upright bbox
        # and no fall signal the person must remain NORMAL (no spurious alarm).
        clf = PostureClassifier()
        ang = JointAngles(person_id=1)  # every angle None, proxy None
        for _ in range(10):
            s = clf.update(1, ang, reba(1), TALL_BBOX)
        assert s.category == PostureCategory.NORMAL

    def test_fall_detected_from_bbox_even_without_keypoints(self):
        # Aspect-ratio fall detection must work on bbox alone (keypoints None).
        clf = PostureClassifier(enter_frames_risk=2)
        ang = JointAngles(person_id=1)
        clf.update(1, ang, reba(1), WIDE_BBOX)
        s = clf.update(1, ang, reba(1), WIDE_BBOX)
        assert s.category == PostureCategory.IMMEDIATE_RISK


class TestSustainedSeconds:
    def test_sustained_seconds_tracks_frames_over_fps(self):
        clf = PostureClassifier(fps=10.0, enter_frames=2)
        for _ in range(5):
            s = drive_ergonomic(clf, 1)
        # frames_in_state advanced; sustained_seconds = frames / fps
        assert s.sustained_seconds == pytest.approx(s.frames_in_state / 10.0)
        assert s.sustained_seconds > 0


class TestFallConfirmation:
    """Gating temporal de queda (design doc 2026-09-05, §2): fall_velocity só
    vira IMMEDIATE_RISK pra fora se terminar em person_fallen dentro da janela
    de confirmação — elimina o residuo de falso alarme em agachamentos rápidos
    (docs/metricas-dia2.md §5.2, precisão 0.418)."""

    def test_fall_velocity_confirmado_por_person_fallen_dentro_da_janela(self):
        clf = PostureClassifier(enter_frames_risk=2, fall_confirm_frames=5)
        pid = 1
        # 2 frames de fall_velocity (TALL_BBOX = sem cue geometrico) -> entra
        # em confirmacao, nao reporta IMMEDIATE_RISK ainda.
        assert clf.update(pid, neutral_angles(pid), reba(1), TALL_BBOX, fall_risk=0.9).category == PostureCategory.NORMAL
        assert clf.update(pid, neutral_angles(pid), reba(1), TALL_BBOX, fall_risk=0.9).category == PostureCategory.NORMAL
        # ainda dentro da janela (5 frames), person_fallen confirma via bbox largo.
        s = clf.update(pid, neutral_angles(pid), reba(1), WIDE_BBOX, fall_risk=0.9)
        assert s.category == PostureCategory.IMMEDIATE_RISK
        assert s.reason == "person_fallen"

    def test_fall_velocity_sem_confirmacao_nunca_vira_immediate_risk(self):
        clf = PostureClassifier(enter_frames_risk=2, fall_confirm_frames=3)
        pid = 1
        # 2 frames pra entrar em confirmacao, mais 3 frames de janela -- em
        # nenhum momento person_fallen aparece (agachamento rapido: so velocidade).
        estados = [
            clf.update(pid, neutral_angles(pid), reba(1), TALL_BBOX, fall_risk=0.9).category
            for _ in range(6)
        ]
        assert PostureCategory.IMMEDIATE_RISK not in estados

    def test_person_fallen_desde_o_primeiro_frame_confirma_sem_esperar_janela(self):
        clf = PostureClassifier(enter_frames_risk=2, fall_confirm_frames=10)
        pid = 1
        assert clf.update(pid, neutral_angles(pid), reba(1), WIDE_BBOX).category == PostureCategory.NORMAL
        s = clf.update(pid, neutral_angles(pid), reba(1), WIDE_BBOX)
        assert s.category == PostureCategory.IMMEDIATE_RISK
        assert s.reason == "person_fallen"

    def test_confirmacao_antecipada_nao_espera_o_resto_da_janela(self):
        clf = PostureClassifier(enter_frames_risk=2, fall_confirm_frames=10)
        pid = 1
        clf.update(pid, neutral_angles(pid), reba(1), TALL_BBOX, fall_risk=0.9)
        clf.update(pid, neutral_angles(pid), reba(1), TALL_BBOX, fall_risk=0.9)
        # so 1 frame dentro da janela de 10 antes de person_fallen confirmar --
        # nao precisa esperar os outros 9.
        s = clf.update(pid, neutral_angles(pid), reba(1), WIDE_BBOX, fall_risk=0.9)
        assert s.category == PostureCategory.IMMEDIATE_RISK
        assert s.reason == "person_fallen"
        assert s.frames_in_state == 1

    def test_janela_de_confirmacao_nao_bloqueia_balance_loss_risk(self):
        # Spec 2026-09-05 §2/§6: balance_loss_risk mantem o schedule normal
        # mesmo com uma janela de confirmacao de queda aberta.
        clf = PostureClassifier(enter_frames_risk=2, fall_confirm_frames=15)
        pid = 1
        # abre a janela de confirmacao via fall_velocity (2 frames).
        clf.update(pid, neutral_angles(pid), reba(1), TALL_BBOX, fall_risk=0.9)
        clf.update(pid, neutral_angles(pid), reba(1), TALL_BBOX, fall_risk=0.9)
        # agora aparece instabilidade (sem fall_risk) -> balance_loss_risk deve
        # confirmar em enter_frames_risk (2) frames, exatamente como no baseline.
        unstable = JointAngles(person_id=pid, trunk_rotation_proxy=0.5)
        assert clf.update(pid, unstable, reba(1), TALL_BBOX).category == PostureCategory.NORMAL
        s = clf.update(pid, unstable, reba(1), TALL_BBOX)
        assert s.category == PostureCategory.IMMEDIATE_RISK
        assert s.reason == "balance_loss_risk"


class TestGateDeZona:
    """Alerta ergonomico (REBA/postura sustentada) so vale dentro da zona
    configurada; queda e zona critica continuam ativas sempre (achado da
    auditoria: zona so alimentava um flag, nao restringia nada)."""

    def test_reba_alto_fora_da_zona_com_gate_desativado_nao_dispara(self):
        clf = PostureClassifier(enter_frames=1)
        estado = clf.update(1, neutral_angles(1), reba(9), TALL_BBOX,
                             ergonomic_eval_enabled=False)
        assert estado.category == PostureCategory.NORMAL

    def test_reba_alto_com_gate_ativado_dispara(self):
        clf = PostureClassifier(enter_frames=1)
        clf.update(1, neutral_angles(1), reba(9), TALL_BBOX, ergonomic_eval_enabled=True)
        estado = clf.update(1, neutral_angles(1), reba(9), TALL_BBOX,
                             ergonomic_eval_enabled=True)
        assert estado.category == PostureCategory.ERGONOMICALLY_INADEQUATE

    def test_queda_dispara_mesmo_com_gate_desativado(self):
        clf = PostureClassifier(enter_frames_risk=1)
        clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX, ergonomic_eval_enabled=False)
        estado = clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX,
                             ergonomic_eval_enabled=False)
        assert estado.category == PostureCategory.IMMEDIATE_RISK

    def test_gate_default_preserva_comportamento_atual(self):
        """Quem nao passa o kwarg novo continua avaliando ergonomia sempre
        (compatibilidade com todo o resto da suite que ja chama update sem ele)."""
        clf = PostureClassifier(enter_frames=1)
        clf.update(1, neutral_angles(1), reba(9), TALL_BBOX)
        estado = clf.update(1, neutral_angles(1), reba(9), TALL_BBOX)
        assert estado.category == PostureCategory.ERGONOMICALLY_INADEQUATE


class TestBboxTruncadoNaBorda:
    """Achado de campo (SPI 2026-09-16): pessoa colada na borda do frame gera
    bbox largo/baixo so pelo corte, nao pela postura — o cue de aspect ratio
    sozinho nao pode confiar nisso quando o bbox esta truncado."""

    def test_bbox_truncado_nao_dispara_fallen_por_aspecto(self):
        clf = PostureClassifier(enter_frames_risk=1)
        clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX, bbox_truncated=True)
        estado = clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX, bbox_truncated=True)
        assert estado.category == PostureCategory.NORMAL

    def test_bbox_nao_truncado_continua_disparando_fallen_por_aspecto(self):
        """Regressao: comportamento antigo preservado quando bbox_truncated=False (default)."""
        clf = PostureClassifier(enter_frames_risk=1)
        clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX)
        estado = clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX)
        assert estado.category == PostureCategory.IMMEDIATE_RISK
        assert estado.reason == "person_fallen"

    def test_bbox_truncado_mas_trunk_flexion_horizontal_ainda_dispara(self):
        """Queda real perto da borda ainda e pega pelos outros cues (keypoints),
        so o cue de aspect-ratio-puro e desativado."""
        clf = PostureClassifier(enter_frames_risk=1)
        deitado = JointAngles(person_id=1, trunk_flexion=80.0)
        clf.update(1, deitado, reba(1), TALL_BBOX, bbox_truncated=True)
        estado = clf.update(1, deitado, reba(1), TALL_BBOX, bbox_truncated=True)
        assert estado.category == PostureCategory.IMMEDIATE_RISK
        assert estado.reason == "person_fallen"


class TestCorroboracaoDeQueda:
    """Achado de campo (SPI 2026-09-16, camera FRESA2): pitch quase vertical
    faz o cue de bbox sozinho disparar em pessoa debrucada na mesa. Flag
    op-in por camera exige 2 cues elegiveis concordando em vez de 1."""

    def test_default_preserva_or_de_sempre(self):
        """fall_require_corroboration=False (default): 1 cue basta, igual hoje."""
        clf = PostureClassifier(enter_frames_risk=1)
        clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX)
        estado = clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX)
        assert estado.category == PostureCategory.IMMEDIATE_RISK

    def test_corroboracao_ligada_bbox_sozinho_nao_dispara(self):
        clf = PostureClassifier(enter_frames_risk=1, fall_require_corroboration=True)
        clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX)
        estado = clf.update(1, neutral_angles(1), reba(1), WIDE_BBOX)
        assert estado.category == PostureCategory.NORMAL

    def test_corroboracao_ligada_bbox_mais_trunk_flexion_dispara(self):
        clf = PostureClassifier(enter_frames_risk=1, fall_require_corroboration=True)
        deitado = JointAngles(person_id=1, trunk_flexion=80.0)
        clf.update(1, deitado, reba(1), WIDE_BBOX)
        estado = clf.update(1, deitado, reba(1), WIDE_BBOX)
        assert estado.category == PostureCategory.IMMEDIATE_RISK
        assert estado.reason == "person_fallen"

    def test_corroboracao_ligada_e_bbox_truncado_exige_os_2_cues_de_keypoint(self):
        """Bbox truncado tira 1 cue elegivel — com corroboracao ligada, os 2
        restantes (trunk_flexion e vertical_extent) precisam concordar."""
        clf = PostureClassifier(enter_frames_risk=1, fall_require_corroboration=True)
        so_trunk = JointAngles(person_id=1, trunk_flexion=80.0)  # vertical_extent None
        clf.update(1, so_trunk, reba(1), WIDE_BBOX, bbox_truncated=True)
        estado = clf.update(1, so_trunk, reba(1), WIDE_BBOX, bbox_truncated=True)
        assert estado.category == PostureCategory.NORMAL  # so 1 de 2 elegiveis
