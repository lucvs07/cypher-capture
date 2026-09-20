"""
Catálogo canônico de EPI — um único lugar que sabe traduzir entre as 3
convenções que existiam soltas no código antes disso (achado 2026-09-20):

  * classe do detector (o que ``models/exp.pt`` produz: ``HELMET``, ``Goggles``,
    ``GLOVE``, ``VEST``, ``boots``, ``MASK`` — maiúsculas inconsistentes,
    singular/plural inconsistente);
  * label de violação (o que ``DecisionEngine`` sempre esperou: ``no_helmet``,
    ``no_gloves``, ...) — nunca gerado de verdade até este catálogo existir,
    porque o detector só tem classes de presença;
  * nome livre em português usado em ``Zona.requiredPpe``/checklist do
    onboarding (``"capacete"``, ``"oculos"``) — nunca validado contra nada.

O nome do membro do enum (``HELMET``, ``GOGGLES``, ...) é o formato que
trafega entre backend e pipeline (JSON), meio-termo estável entre as duas
pontas — front traduz só para exibição.
"""
from __future__ import annotations

from enum import Enum


class EpiTipo(Enum):
    HELMET = "HELMET"
    GOGGLES = "GOGGLES"
    GLOVE = "GLOVE"
    VEST = "VEST"
    BOOTS = "BOOTS"
    MASK = "MASK"

    @property
    def detector_label(self) -> str:
        """Label exato que ``models/exp.pt`` emite para esta classe."""
        return _DETECTOR_LABELS[self]

    @property
    def violation_label(self) -> str:
        """Label que ``DecisionEngine.EPI_VIOLATION_LABELS`` reconhece."""
        return _VIOLATION_LABELS[self]

    @staticmethod
    def from_detector_label(label: str) -> "EpiTipo | None":
        return _POR_DETECTOR_LABEL.get(label)

    @staticmethod
    def from_wire_name(nome: str) -> "EpiTipo | None":
        """Resolve o nome que vem do cadastro (JSON), ignorando caixa —
        ``Zona.requiredPpe``/checklist da empresa trafegam como string solta,
        e um "helmet" minusculo digitado a mao nao pode virar item ignorado
        em silencio."""
        try:
            return EpiTipo[nome.strip().upper()]
        except KeyError:
            return None


_DETECTOR_LABELS: dict[EpiTipo, str] = {
    EpiTipo.HELMET: "HELMET",
    EpiTipo.GOGGLES: "Goggles",
    EpiTipo.GLOVE: "GLOVE",
    EpiTipo.VEST: "VEST",
    EpiTipo.BOOTS: "boots",
    EpiTipo.MASK: "MASK",
}

_VIOLATION_LABELS: dict[EpiTipo, str] = {
    EpiTipo.HELMET: "no_helmet",
    EpiTipo.GOGGLES: "no_goggle",
    EpiTipo.GLOVE: "no_gloves",
    EpiTipo.VEST: "no_vest",
    EpiTipo.BOOTS: "no_boots",
    EpiTipo.MASK: "no_mask",
}

_POR_DETECTOR_LABEL: dict[str, EpiTipo] = {v: k for k, v in _DETECTOR_LABELS.items()}
