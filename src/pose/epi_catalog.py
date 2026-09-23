"""
Catálogo canônico de EPI — o único lugar que conhece o nome cru que o modelo
de detecção emite.

Só existem aqui os EPIs que o modelo detecta de fato (``models/*.pt``:
``HELMET``, ``Goggles``, ``GLOVE``, ``VEST``, ``boots``, ``MASK`` — maiúsculas
e singular/plural inconsistentes). O label cru é convertido pra ``EpiTipo``
na borda de entrada (associação EPI -> pessoa); daí em diante o pipeline,
os incidentes e o payload pra API só falam ``EpiTipo``. Não há label de
violação (``no_*``): o modelo só tem classes de presença, e "ausente" é
sempre uma inferência temporal (``epi_absence.py``).

O nome do membro (``HELMET``, ``GOGGLES``, ...) é o formato que trafega entre
backend, front e pipeline (JSON) — ``cypher/.../model/EpiTipo.java`` e
``cypher-front/src/app/shared/epi-catalog.ts`` espelham este enum.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Iterable

log = logging.getLogger(__name__)


class EpiTipo(Enum):
    HELMET = "HELMET"
    GOGGLES = "GOGGLES"
    GLOVE = "GLOVE"
    VEST = "VEST"
    BOOTS = "BOOTS"
    MASK = "MASK"

    @property
    def detector_label(self) -> str:
        """Label exato que o modelo de detecção emite para esta classe."""
        return _DETECTOR_LABELS[self]

    @staticmethod
    def from_detector_label(label: str) -> "EpiTipo | None":
        return _POR_DETECTOR_LABEL.get(label)

    @staticmethod
    def from_wire_name(nome: object) -> "EpiTipo | None":
        """Resolve o nome que vem do cadastro (JSON), ignorando caixa e
        espaços — um "helmet" minúsculo de um ``zones.json`` local continua
        valendo. Qualquer outra coisa (nome desconhecido, ``None``, número)
        devolve ``None``: dado ruim de cadastro não derruba o pipeline."""
        if not isinstance(nome, str):
            return None
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

_POR_DETECTOR_LABEL: dict[str, EpiTipo] = {v: k for k, v in _DETECTOR_LABELS.items()}


def resolve_wire_names(nomes: Iterable[object] | None, origem: str) -> list[EpiTipo]:
    """Converte uma lista vinda do cadastro (zona ou checklist) pra
    ``EpiTipo``, sem duplicatas e na ordem original. Nome que não é EPI
    detectável é descartado com aviso — chamado uma vez por carga de
    configuração, nunca por frame."""
    tipos: list[EpiTipo] = []
    for nome in nomes or []:
        tipo = nome if isinstance(nome, EpiTipo) else EpiTipo.from_wire_name(nome)
        if tipo is None:
            log.warning("%s: EPI %r ignorado — o modelo não detecta esse item "
                        "(esperado um de %s)", origem, nome, [t.value for t in EpiTipo])
        elif tipo not in tipos:
            tipos.append(tipo)
    return tipos
