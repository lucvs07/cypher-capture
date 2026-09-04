"""
Teste de parse_args (src/orchestrator/__main__.py).

Não é praticável testar `main()` diretamente (login real, loop infinito,
subprocess real) — mesma situação de `run_pipeline.py`
(`tests/test_run_pipeline_config.py`). Só `parse_args` é testado aqui; o
resto é fiação de componentes já cobertos nas Tasks 1-4.
"""
from __future__ import annotations

from src.orchestrator.__main__ import parse_args


def test_parse_args_usa_os_defaults_documentados():
    args = parse_args(["--api-url", "http://localhost:8080/api/v1"])

    assert args.api_url == "http://localhost:8080/api/v1"
    assert args.intervalo == 30
    assert args.max_concorrentes == 2
    assert args.config == "config/pose_config.yaml"
    assert args.state_path == "outputs/orchestrator-state.json"


def test_parse_args_aceita_overrides():
    args = parse_args([
        "--api-url", "http://localhost:8080/api/v1",
        "--intervalo", "10",
        "--max-concorrentes", "5",
        "--config", "config/outro.yaml",
        "--state-path", "outputs/outro-estado.json",
    ])

    assert args.intervalo == 10
    assert args.max_concorrentes == 5
    assert args.config == "config/outro.yaml"
    assert args.state_path == "outputs/outro-estado.json"
