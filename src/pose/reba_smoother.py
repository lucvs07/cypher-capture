"""
Suavizacao temporal do score REBA final, por pessoa rastreada.

Achado da auditoria (2026-09-15): o REBA numerico era calculado cru a cada
frame - so a CATEGORIA discreta (normal/inadequado/risco/zona) tinha
histerese via PostureClassifier. Um unico frame de pose ruidosa podia gerar
um final_score espurio. Isto suaviza o numero em si, ignorando frames
indeterminados (ver REBAScore.indeterminate) em vez de deixa-los
contaminar a mediana com o score-1 de fallback.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

from .reba import REBAScore


class RebaSmoother:
    def __init__(self, window: int = 5) -> None:
        self.window = window
        self._history: dict[int, deque] = defaultdict(lambda: deque(maxlen=window))
        self._last_valid: dict[int, int] = {}

    def smooth(self, person_id: int, reba: REBAScore) -> int:
        """Mediana movel de final_score, pulando frames indeterminados.
        Sem nenhum frame valido ainda, devolve o proprio valor bruto."""
        if reba.indeterminate:
            return self._last_valid.get(person_id, reba.final_score)

        self._history[person_id].append(reba.final_score)
        self._last_valid[person_id] = reba.final_score
        return int(np.median(self._history[person_id]))
