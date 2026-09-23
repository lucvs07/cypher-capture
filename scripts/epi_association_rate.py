"""Taxa de associacao EPI -> pessoa em video real.

``DecisionEngine.associate_epis`` so atribui um EPI a uma pessoa quando o bbox
da pose cobre mais de ``--threshold`` da area do EPI. O bbox da pose e justo
ao corpo: capacete acima da cabeca e bota no pe podem ficar abaixo do limiar
e nao serem associados — e, com a ausencia inferida por debounce, EPI nao
associado vira "EPI ausente" falso. Este script mede isso por classe, sem
mudar nada no pipeline.

Uso:
    python scripts/epi_association_rate.py
    python scripts/epi_association_rate.py --videos data/validation/cenario4.mp4 --device cpu
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from ultralytics import YOLO

from src.fusion.engine import _containment

DEFAULT_VIDEOS = [
    "data/validation/cenario.mp4",
    "data/validation/cenario2.mp4",
    "data/validation/cenario3.mp4",
    "data/validation/cenario4.mp4",
    "data/validation/test_bus.mp4",
]
BUCKETS = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--videos", nargs="+", default=DEFAULT_VIDEOS)
    p.add_argument("--epi-model", default="models/exp-m4-1709-v4.pt")
    p.add_argument("--pose-model", default="yolo11s-pose.pt")
    p.add_argument("--device", default="cuda")
    p.add_argument("--conf-epi", type=float, default=0.5)
    p.add_argument("--conf-pose", type=float, default=0.5)
    p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    epi_model = YOLO(args.epi_model)
    pose_model = YOLO(args.pose_model)

    total: Counter = Counter()
    associadas: Counter = Counter()
    sem_pessoa: Counter = Counter()
    faixas: dict[str, Counter] = defaultdict(Counter)

    for video in args.videos:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            print(f"[skip] {video} nao abriu")
            continue
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            pessoas = [
                tuple(b.tolist())
                for r in pose_model(frame, conf=args.conf_pose, device=args.device, verbose=False)
                for b in r.boxes.xyxy
            ]
            for r in epi_model(frame, conf=args.conf_epi, device=args.device, verbose=False):
                for box, cls_id in zip(r.boxes.xyxy, r.boxes.cls):
                    label = r.names[int(cls_id)]
                    if label == "Person":
                        continue
                    total[label] += 1
                    if not pessoas:
                        sem_pessoa[label] += 1
                        continue
                    cobertura = max(_containment(p, tuple(box.tolist())) for p in pessoas)
                    if cobertura > args.threshold:
                        associadas[label] += 1
                    for lo, hi in BUCKETS:
                        if lo <= cobertura < hi:
                            faixas[label][f"{lo:.2f}-{min(hi, 1.0):.2f}"] += 1
        cap.release()

    print(f"\nlimiar de containment: > {args.threshold}   pesos: {args.epi_model}")
    print(f"{'classe':<10}{'deteccoes':>10}{'associadas':>12}{'taxa':>8}{'sem pessoa':>12}   cobertura (melhor pessoa)")
    for label in sorted(total):
        base = total[label] - sem_pessoa[label]
        taxa = associadas[label] / base if base else 0.0
        dist = "  ".join(f"{k}:{v}" for k, v in sorted(faixas[label].items()))
        print(f"{label:<10}{total[label]:>10}{associadas[label]:>12}{taxa:>8.1%}{sem_pessoa[label]:>12}   {dist}")


if __name__ == "__main__":
    main()
