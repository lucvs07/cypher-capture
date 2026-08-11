"""
Testes da configuração de notificação.

O que está sendo protegido: que o pipeline continue rodando exatamente como
antes quando ninguém configurou a API — a compatibilidade para trás é o que
permite executar este plano sem quebrar a demonstração atual.
"""
from __future__ import annotations

import textwrap

from src.notify import NotifyConfig


def test_valores_padrao_desligam_a_notificacao(tmp_path):
    config = tmp_path / "vazio.yaml"
    config.write_text("pipeline:\n  device: cpu\n")

    notify = NotifyConfig.from_yaml(str(config))

    assert notify.api_url is None
    assert notify.enabled is False
    assert notify.camera_id == "CAM-01"
    assert notify.merge_window_seconds == 15.0
    assert notify.evict_after_frames == 45
    assert notify.min_track_frames == 15
    assert notify.stream_enabled is True
    assert notify.stream_port == 8090


def test_le_a_secao_notify(tmp_path):
    config = tmp_path / "cheio.yaml"
    config.write_text(textwrap.dedent("""
        notify:
          camera_id: CAM-03
          api_url: http://localhost:8080/api/v1
          min_severity: 3
          merge_window_seconds: 20
          evict_after_frames: 60
          min_track_frames: 10
          queue_size: 128
          spill_path: outputs/meu.jsonl
          stream_port: 8092
          stream_enabled: false
    """))

    notify = NotifyConfig.from_yaml(str(config))

    assert notify.enabled is True
    assert notify.camera_id == "CAM-03"
    assert notify.api_url == "http://localhost:8080/api/v1"
    assert notify.min_severity == 3
    assert notify.merge_window_seconds == 20
    assert notify.queue_size == 128
    assert notify.spill_path == "outputs/meu.jsonl"
    assert notify.stream_port == 8092
    assert notify.stream_enabled is False


def test_chaves_desconhecidas_sao_ignoradas(tmp_path):
    config = tmp_path / "extra.yaml"
    config.write_text("notify:\n  camera_id: CAM-09\n  invento: 42\n")

    notify = NotifyConfig.from_yaml(str(config))

    assert notify.camera_id == "CAM-09"


def test_cada_execucao_tem_run_id_proprio(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_text("notify:\n  camera_id: CAM-01\n")

    a = NotifyConfig.from_yaml(str(config))
    b = NotifyConfig.from_yaml(str(config))

    assert a.run_id != b.run_id
    assert len(a.run_id) >= 6
