"""
Testes de PipelineConfig.from_yaml (src/fusion/pipeline.py) — cobre só o
campo fall_confirm_seconds (design doc 2026-09-05, §3). PipelineConfig nao
tinha teste dedicado antes desta entrega; nao criamos cobertura retroativa
pros demais campos, so pro que estamos adicionando.
"""
from __future__ import annotations

from src.fusion.pipeline import PipelineConfig


def test_fall_confirm_seconds_default_e_um_segundo():
    config = PipelineConfig()
    assert config.fall_confirm_seconds == 1.0


def test_fall_confirm_seconds_carregado_do_yaml(tmp_path):
    yaml_path = tmp_path / "pose_config.yaml"
    yaml_path.write_text(
        "classifier:\n"
        "  fall_confirm_seconds: 2.5\n"
    )

    config = PipelineConfig.from_yaml(str(yaml_path))

    assert config.fall_confirm_seconds == 2.5
