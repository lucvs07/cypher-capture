"""
Rule-based postural classifier with temporal hysteresis.

Taxonomy:
  NORMAL                   — no ergonomic concern detected
  ERGONOMICALLY_INADEQUATE — sustained bad posture per NR-17 thresholds
  IMMEDIATE_RISK           — fall or collapse detected (recall-optimised)
  CRITICAL_ZONE            — person inside a demarcated restricted area
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .angles import JointAngles
from .reba import REBAScore


class PostureCategory(Enum):
    NORMAL = "normal"
    ERGONOMICALLY_INADEQUATE = "ergonomically_inadequate"
    IMMEDIATE_RISK = "immediate_risk"
    CRITICAL_ZONE = "critical_zone"


@dataclass
class ClassifierThresholds:
    """All rule thresholds; defaults match docs/taxonomy.md."""
    trunk_flexion_deg: float = 60.0      # forward lean beyond this = spine load risk
    spine_load_knee_deg: float = 150.0   # knee angle above this counts as "straight leg"
    deep_squat_knee_deg: float = 90.0    # knee interior angle below this = deep squat
    elevated_arm_deg: float = 90.0       # shoulder interior angle above this = arm raised past horizontal
    trunk_rotation_max: float = 0.35     # normalised shoulder/hip offset above this = instability
    fallen_aspect_ratio: float = 1.3     # bbox w/h above this = person horizontal
    reba_risk_score: int = 8             # REBA final score at/above this = ergonomic risk
    fall_risk_min: float = 0.85          # TemporalAnalyzer fall_risk_score threshold
                                         # (raised 0.7->0.85: the velocity signal is noisy and spikes
                                         #  during fast bends/squats; the velocity-independent cues
                                         #  below now carry recall, so this can be stricter)
    # Velocity-independent fallen cues (sustain the "on the ground" alarm even
    # when the person has stopped moving and the bbox is not wide):
    fallen_trunk_deg: float = 65.0       # trunk flexion (from vertical) above this = torso horizontal
    fallen_vertical_extent: float = 0.4  # vertical_extent (nose→lowest foot)/h BELOW this = body not upright


@dataclass
class PostureState:
    category: PostureCategory = PostureCategory.NORMAL
    reason: str = ""
    frames_in_state: int = 0
    sustained_seconds: float = 0.0


class PostureClassifier:
    """
    Per-person state machine with hysteresis.

    Hysteresis design: entering a high-severity state is fast (enter_frames),
    exiting requires more evidence (exit_frames). This trades a small increase
    in false positives for the recall guarantee required on IMMEDIATE_RISK.
    """

    def __init__(
        self,
        fps: float = 15.0,
        thresholds: Optional[ClassifierThresholds] = None,
        enter_frames: int = 5,
        exit_frames: int = 15,
        enter_frames_risk: int = 2,
        fall_confirm_frames: int = 15,
        fall_require_corroboration: bool = False,
    ) -> None:
        self.fps = fps
        self.t = thresholds or ClassifierThresholds()
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        # Achado SPI 2026-09-16: camera com pitch quase vertical (olhando de
        # cima pra mesa) faz o cue de bbox-aspect sozinho disparar em pessoa
        # so debrucada. Op-in por camera (default False = comportamento OR
        # de sempre, recall alto) — quando True, exige 2 dos cues ELEGIVEIS
        # (nao conta o cue de bbox quando ele esta suprimido por bbox_truncated).
        self.fall_require_corroboration = fall_require_corroboration
        # IMMEDIATE_RISK uses a lower entry threshold to maximise recall
        self.enter_frames_risk = enter_frames_risk
        # Gating temporal de queda (design doc 2026-09-05, §2): janela de
        # confirmacao antes de reportar IMMEDIATE_RISK via fall_velocity.
        self.fall_confirm_frames = fall_confirm_frames
        self._states: dict[int, PostureState] = {}
        self._pending: dict[int, tuple[PostureCategory, str, int]] = {}
        self._exit_counters: dict[int, int] = defaultdict(int)
        self._fall_pending_confirm: dict[int, int] = {}

    def update(
        self,
        person_id: int,
        angles: JointAngles,
        reba: REBAScore,
        bbox: tuple,
        in_critical_zone: bool = False,
        fall_risk: float = 0.0,
        ergonomic_eval_enabled: bool = True,
        bbox_truncated: bool = False,
    ) -> PostureState:
        raw_cat, raw_reason = self._evaluate_rules(
            angles, reba, bbox, in_critical_zone, fall_risk, ergonomic_eval_enabled,
            bbox_truncated,
        )
        state = self._apply_hysteresis(person_id, raw_cat, raw_reason)

        if person_id not in self._states:
            self._states[person_id] = PostureState()

        current = self._states[person_id]
        if state is not None:
            current = state
            self._states[person_id] = current
            self._exit_counters[person_id] = 0
        elif current.category != PostureCategory.NORMAL:
            if raw_cat == PostureCategory.NORMAL:
                self._exit_counters[person_id] += 1
                if self._exit_counters[person_id] >= self.exit_frames:
                    current = PostureState()
                    self._states[person_id] = current
                    self._exit_counters[person_id] = 0
            else:
                # exit requires *consecutive* normal frames
                self._exit_counters[person_id] = 0

        current.frames_in_state += 1
        current.sustained_seconds = current.frames_in_state / self.fps
        return current

    def get_state(self, person_id: int) -> PostureState:
        return self._states.get(person_id, PostureState())

    # ------------------------------------------------------------------
    # Rule evaluation
    # ------------------------------------------------------------------

    def _evaluate_rules(
        self,
        angles: JointAngles,
        reba: REBAScore,
        bbox: tuple,
        in_critical_zone: bool,
        fall_risk: float,
        ergonomic_eval_enabled: bool = True,
        bbox_truncated: bool = False,
    ) -> tuple[PostureCategory, str]:
        # Priority order: IMMEDIATE_RISK > CRITICAL_ZONE > ERGONOMICALLY_INADEQUATE

        if self._detect_fallen(bbox, angles, bbox_truncated):
            return PostureCategory.IMMEDIATE_RISK, "person_fallen"

        # Covers falls toward/away from the camera where the bbox never widens
        if fall_risk >= self.t.fall_risk_min:
            return PostureCategory.IMMEDIATE_RISK, "fall_velocity"

        if self._detect_instability(angles):
            return PostureCategory.IMMEDIATE_RISK, "balance_loss_risk"

        if in_critical_zone:
            return PostureCategory.CRITICAL_ZONE, "critical_zone_entry"

        # Gate de zona (spec RCC): alerta ergonomico so vale dentro de uma
        # zona configurada, quando a camera tem alguma zona cadastrada. Queda
        # e zona critica (acima) NUNCA sao gateados — sao checagens de
        # seguranca independentes de zona de trabalho estar configurada.
        if not ergonomic_eval_enabled:
            return PostureCategory.NORMAL, ""

        if self._detect_spine_load(angles):
            return PostureCategory.ERGONOMICALLY_INADEQUATE, "trunk_flexion_spine_load"

        if self._detect_prolonged_squat(angles):
            return PostureCategory.ERGONOMICALLY_INADEQUATE, "prolonged_squat"

        if self._detect_elevated_arm(angles):
            return PostureCategory.ERGONOMICALLY_INADEQUATE, "elevated_arm_work"

        if reba.final_score >= self.t.reba_risk_score:
            return PostureCategory.ERGONOMICALLY_INADEQUATE, f"reba_{reba.final_score}"

        return PostureCategory.NORMAL, ""

    def _detect_fallen(self, bbox: tuple, angles: JointAngles, bbox_truncated: bool = False) -> bool:
        """Person on the ground / horizontal.

        OR of complementary cues so different fall geometries are all caught
        (recall priority). Each cue covers a case the others miss:
          - bbox aspect: clear side-on lying (whole body wide).
          - trunk flexion ≈ horizontal: torso lying, seen from the side.
          - collapsed vertical extent: the nose→lowest-foot span shrinks when
            the body is horizontal/inverted, even prone toward the camera where
            the bbox never widens and there is no velocity. A forward bend keeps
            the feet planted far below the head, so its extent stays large —
            that is what stops a toe-touch from reading as a fall.
        The last two are velocity-independent, so a person who has fallen and
        gone still keeps the alarm asserted (combined with the sticky exit
        hysteresis) instead of decaying back to NORMAL.

        `bbox_truncated` (achado de campo, SPI 2026-09-16): quando a pessoa
        esta colada na borda do frame (perto demais da camera, corpo cortado),
        o bbox visivel fica largo e baixo por causa do corte, nao da postura —
        o cue de aspect ratio sozinho vira falso-positivo. Nesse caso ele e
        pulado; os outros dois cues (que dependem de keypoints, nao so do
        retangulo) continuam valendo e ainda pegam uma queda real perto da
        borda.

        `self.fall_require_corroboration` (achado de campo, SPI 2026-09-16,
        camera FRESA2): camera com pitch quase vertical faz o cue de bbox
        sozinho disparar numa pessoa so debrucada sobre a mesa. Quando True,
        exige 2 cues concordando em vez de qualquer 1 — mas conta so os cues
        ELEGIVEIS (com dado disponivel) nesse frame: se bbox esta truncado
        (so 2 cues elegiveis: trunk_flexion e vertical_extent), a exigencia
        vira "os 2 concordam", nao "2 de 3" — escolha deliberada, nao lacuna.
        Se so 1 cue estiver elegivel (oclusao), a queda nunca confirma com
        corroboracao ligada, mesmo que esse unico cue dispare — op-in por
        camera, troca recall por precisao, nunca ligado por default.
        """
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)

        cues_fired = 0
        cues_eligible = 0
        if not bbox_truncated:
            cues_eligible += 1
            if (w / h) > self.t.fallen_aspect_ratio:
                cues_fired += 1
        if angles.trunk_flexion is not None:
            cues_eligible += 1
            if angles.trunk_flexion > self.t.fallen_trunk_deg:
                cues_fired += 1
        if angles.vertical_extent is not None:
            cues_eligible += 1
            if angles.vertical_extent < self.t.fallen_vertical_extent:
                cues_fired += 1

        if not self.fall_require_corroboration:
            return cues_fired >= 1
        return cues_fired >= 2

    def _detect_instability(self, angles: JointAngles) -> bool:
        """Trunk rotation proxy above threshold suggests rapid unexpected twist."""
        return (
            angles.trunk_rotation_proxy is not None
            and angles.trunk_rotation_proxy > self.t.trunk_rotation_max
        )

    def _detect_spine_load(self, angles: JointAngles) -> bool:
        """Trunk flexion > threshold with legs mostly straight → lifting with bent back (NR-17)."""
        if angles.trunk_flexion is None:
            return False
        if angles.trunk_flexion <= self.t.trunk_flexion_deg:
            return False
        knees = [v for v in [angles.left_knee_flexion, angles.right_knee_flexion] if v is not None]
        if not knees:
            return False
        # knee_flexion angle close to 180° means leg is straight
        return max(knees) > self.t.spine_load_knee_deg

    def _detect_prolonged_squat(self, angles: JointAngles) -> bool:
        """Knee interior angle below threshold → deep squat."""
        knees = [v for v in [angles.left_knee_flexion, angles.right_knee_flexion] if v is not None]
        if not knees:
            return False
        return min(knees) < self.t.deep_squat_knee_deg

    def _detect_elevated_arm(self, angles: JointAngles) -> bool:
        """Upper arm above shoulder level for sustained work (NR-17 risk factor)."""
        elevations = [
            v for v in [angles.left_shoulder_elevation, angles.right_shoulder_elevation]
            if v is not None
        ]
        if not elevations:
            return False
        # shoulder_elevation = angle hip→shoulder→elbow: ~0° arm hanging at the
        # side, ~90° horizontal, ~180° overhead. An arm raised past (roughly)
        # the shoulder line means the angle is ABOVE the threshold.
        return any(e > self.t.elevated_arm_deg for e in elevations)

    # ------------------------------------------------------------------
    # Hysteresis
    # ------------------------------------------------------------------

    def _apply_hysteresis(
        self,
        person_id: int,
        cat: PostureCategory,
        reason: str,
    ) -> Optional[PostureState]:
        if cat == PostureCategory.NORMAL:
            # NORMAL is never "entered" through this path — leaving an alarm
            # state goes exclusively through the exit_frames counter in
            # update(), keeping alarms sticky (recall priority).
            self._pending.pop(person_id, None)
            self._fall_pending_confirm.pop(person_id, None)
            return None

        # Gating temporal de queda: uma confirmacao de fall_velocity ja esta
        # em andamento pra este track. So sai daqui confirmando IMMEDIATE_RISK
        # (via person_fallen) ou descartando o episodio (janela esgotada) —
        # nunca confirma via fall_velocity puro.
        # Narrowed per spec 2026-09-05 §2/§6: only fall_velocity/person_fallen
        # are gated here. Any other alarm (balance_loss_risk, critical_zone,
        # ...) aborts the pending fall episode and falls through to standard
        # hysteresis so the competing alarm surfaces exactly as on baseline.
        # Re-entry safety net: a transient NORMAL frame mid-window drops the
        # pending episode, but a genuine landing still re-triggers via the
        # normal person_fallen hysteresis (~enter_frames_risk delay). And a
        # persistent fall_risk re-opens a fresh window every cycle after a
        # discard, so fall_velocity alone can never surface publicly (intended).
        if person_id in self._fall_pending_confirm and reason in (
            "fall_velocity",
            "person_fallen",
        ):
            if reason == "person_fallen":
                frames_waited = self._fall_pending_confirm.pop(person_id)
                return PostureState(
                    category=PostureCategory.IMMEDIATE_RISK,
                    reason="person_fallen",
                    frames_in_state=frames_waited,
                )
            frames_waited = self._fall_pending_confirm[person_id] + 1
            if frames_waited >= self.fall_confirm_frames:
                # Janela esgotada sem confirmacao geometrica — descarta.
                # O episodio nunca chega a ser visivel fora do classificador
                # (spec 2026-09-05 §2: elimina o residuo de fall_velocity em
                # agachamentos rapidos, docs/metricas-dia2.md §5.2).
                self._fall_pending_confirm.pop(person_id)
            else:
                self._fall_pending_confirm[person_id] = frames_waited
            return None

        # A competing non-fall alarm arrived while a fall window was open:
        # abandon the pending episode and let it surface normally below.
        self._fall_pending_confirm.pop(person_id, None)

        pending = self._pending.get(person_id)
        enter_threshold = (
            self.enter_frames_risk
            if cat == PostureCategory.IMMEDIATE_RISK
            else self.enter_frames
        )

        if pending is None or pending[0] != cat:
            self._pending[person_id] = (cat, reason, 1)
            return None

        count = pending[2] + 1
        self._pending[person_id] = (cat, reason, count)

        current_cat = self._states.get(person_id, PostureState()).category
        if cat != current_cat and count >= enter_threshold:
            self._pending.pop(person_id)
            if cat == PostureCategory.IMMEDIATE_RISK and reason == "fall_velocity":
                # Nao confirma ainda -- abre a janela de confirmacao em vez
                # de reportar a transicao.
                self._fall_pending_confirm[person_id] = 0
                return None
            # The posture already persisted `count` raw frames before being
            # confirmed; start the counter there so sustained_seconds is honest
            # (update() increments once more right after).
            return PostureState(category=cat, reason=reason, frames_in_state=count - 1)

        return None
