"""
Fluxo de EPI ponta a ponta com os pesos reais: detector de EPI + pose +
associacao por pessoa + inferencia de ausencia + escalonamento por zona +
abertura de incidente. Compara com o golden de ``tests/fixtures/epi_flow/``
(ver ``tests/epi_flow_golden.py``).

Opt-in (``CYPHER_GOLDEN=1``): leva ~1 min em GPU e depende dos videos de
``data/validation/``, que nao sao versionados.
"""
import json
import os
from pathlib import Path

import pytest

from epi_flow_golden import (
    DEVICE, EPI_WEIGHTS, POSE_WEIGHTS, ROOT, SCENARIOS, golden_path, run_scenario, summarize,
)


def _pula_se_indisponivel(video: str) -> None:
    if os.environ.get("CYPHER_GOLDEN") != "1":
        pytest.skip("golden de EPI e opt-in: CYPHER_GOLDEN=1")
    for arquivo in (video, EPI_WEIGHTS, POSE_WEIGHTS):
        if not (ROOT / arquivo).exists():
            pytest.skip(f"{arquivo} ausente")
    if DEVICE == "cuda":
        import torch
        if not torch.cuda.is_available():
            pytest.skip("golden gravado em cuda")


@pytest.mark.parametrize("nome", sorted(SCENARIOS))
def test_golden_exercita_o_caminho_de_epi(nome):
    """Um golden vazio passaria 'identico' antes e depois sem provar nada."""
    resumo = summarize(json.loads(golden_path(nome).read_text()))
    assert resumo["with_epi_present"] > 0
    assert resumo["with_epi_missing"] > 0
    assert resumo["escalated"] > 0
    assert resumo["epi_incidents"] > 0


@pytest.mark.parametrize("nome", sorted(SCENARIOS))
def test_fluxo_de_epi_identico_ao_golden(nome, tmp_path: Path):
    _pula_se_indisponivel(SCENARIOS[nome])
    esperado = json.loads(golden_path(nome).read_text())
    obtido = run_scenario(SCENARIOS[nome], tmp_path)

    assert summarize(obtido) == summarize(esperado)
    assert obtido["incidents"] == esperado["incidents"]
    for f_obtido, f_esperado in zip(obtido["frames"], esperado["frames"], strict=True):
        assert f_obtido == f_esperado, f"divergiu no frame {f_esperado['frame']}"
