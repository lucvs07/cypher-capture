"""
Entrypoint do orquestrador (design doc, §4): `python -m src.orchestrator`.

Fiação real dos três componentes (Tasks 1-4) — login, limpeza de órfãos do
boot anterior, e o loop de reconciliação. `main()` não é testado
diretamente (login real, loop infinito, subprocess real de verdade) — só
`parse_args` é coberto por teste.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from src.capture import CaptureConfig
from src.orchestrator.reconciler import Reconciler
from src.orchestrator.registry import ProcessRegistry
from src.orchestrator.session import OrchestratorSession, RequestsSessionTransport


class SubprocessLauncher:
    """Lançador real: um `python -m src.run_pipeline` por câmera, log
    redirecionado para um arquivo próprio (`outputs/orchestrator-logs/<id>.log`)."""

    def __init__(self, log_dir: str = "outputs/orchestrator-logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def iniciar(self, camera_id: str, api_url: str) -> subprocess.Popen:
        log_path = self.log_dir / f"{camera_id}.log"
        arquivo_log = open(log_path, "ab")
        return subprocess.Popen(
            [sys.executable, "-m", "src.run_pipeline",
             "--camera-id", camera_id, "--api-url", api_url, "--no-display"],
            stdout=arquivo_log, stderr=subprocess.STDOUT)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Orquestrador de pipelines por camera")
    p.add_argument("--api-url", required=True,
                   help="URL base da API do cypher para este tenant (ex.: http://localhost:8080/api/v1)")
    p.add_argument("--intervalo", type=int, default=30,
                   help="Segundos entre reconciliacoes (default: 30)")
    p.add_argument("--max-concorrentes", type=int, default=2,
                   help="Teto de pipelines simultaneos neste host (default: 2)")
    p.add_argument("--config", default="config/pose_config.yaml",
                   help="Arquivo de config de onde ler capture.backoff_base_seconds/backoff_max_seconds")
    p.add_argument("--state-path", default="outputs/orchestrator-state.json",
                   help="Onde persistir o mapa camera_id->pid entre reinicios do orquestrador")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    capture_config = CaptureConfig.from_yaml(args.config)

    sessao = OrchestratorSession(args.api_url, RequestsSessionTransport())
    sessao.login(os.environ)

    registry = ProcessRegistry(SubprocessLauncher(), state_path=args.state_path)
    registry.limpar_orfaos_do_boot_anterior()

    reconciler = Reconciler(
        sessao, registry, args.api_url, args.max_concorrentes,
        capture_config.backoff_base_seconds, capture_config.backoff_max_seconds)

    while True:
        reconciler.reconciliar_uma_vez()
        time.sleep(args.intervalo)


if __name__ == "__main__":
    main()
