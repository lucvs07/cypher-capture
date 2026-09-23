"""Benchmark de latencia e cobertura dos modelos candidatos de EPI.

Mede preprocess/inference/postprocess por frame em video real e conta deteccoes
por classe. Gera a evidencia de latencia exigida pela Fase 1 (<= 100 ms, >= 10 FPS)
e a base da decisao entre modelos.

Uso:
    python scripts/bench_models.py --device cuda
    python scripts/bench_models.py --device cpu --frames 30
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter
from pathlib import Path

import cv2
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops, get_num_params

DEFAULT_MODELS = [
    "models/exp-s3-1709.pt",
    "models/exp-m4-1709-v4.pt",
    "models/exp.pt",
]


def pct(values: list[float], p: float) -> float:
    return sorted(values)[max(0, int(p * len(values)) - 1)]


def load_frames(source: str, limit: int) -> list:
    cap = cv2.VideoCapture(source)
    frames = []
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit(f"nenhum frame lido de {source}")
    return frames


def bench(model_path: str, frames: list, device: str, imgsz: int, conf: float) -> dict:
    model = YOLO(model_path)
    params = get_num_params(model.model)
    try:
        flops = get_flops(model.model, imgsz=imgsz)
    except Exception:
        flops = float("nan")

    for frame in frames[:5]:  # warmup
        model.predict(frame, imgsz=imgsz, conf=conf, verbose=False, device=device)

    pre, inf, post, totals = [], [], [], []
    by_class: Counter[str] = Counter()
    confs: list[float] = []
    for frame in frames:
        result = model.predict(
            frame, imgsz=imgsz, conf=conf, verbose=False, device=device
        )[0]
        pre.append(result.speed["preprocess"])
        inf.append(result.speed["inference"])
        post.append(result.speed["postprocess"])
        totals.append(sum(result.speed.values()))
        for box in result.boxes:
            by_class[model.names[int(box.cls)]] += 1
            confs.append(float(box.conf))

    return {
        "model": model_path,
        "params": params,
        "gflops": round(flops, 1),
        "preprocess_ms": round(st.mean(pre), 1),
        "inference_ms": round(st.mean(inf), 1),
        "postprocess_ms": round(st.mean(post), 1),
        "total_mean_ms": round(st.mean(totals), 1),
        "total_p95_ms": round(pct(totals, 0.95), 1),
        "fps": round(1000 / st.mean(totals), 1),
        "detections": dict(sorted(by_class.items())),
        "detections_total": sum(by_class.values()),
        "conf_mean": round(st.mean(confs), 3) if confs else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--source", default="data/validation/cenario.mp4")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    ap.add_argument("--out", default="bench_models.json")
    args = ap.parse_args()

    frames = load_frames(args.source, args.frames)
    print(f"device={args.device}  frames={len(frames)}  imgsz={args.imgsz}  conf={args.conf}\n")

    rows = [bench(m, frames, args.device, args.imgsz, args.conf) for m in args.models]

    header = f"{'modelo':<28}{'GFLOPs':>8}{'inf(ms)':>9}{'total(ms)':>11}{'p95':>8}{'FPS':>7}{'dets':>7}{'conf':>7}"
    print(header)
    print("-" * len(header))
    for r in rows:
        name = Path(r["model"]).name
        print(
            f"{name:<28}{r['gflops']:>8.1f}{r['inference_ms']:>9.1f}"
            f"{r['total_mean_ms']:>11.1f}{r['total_p95_ms']:>8.1f}"
            f"{r['fps']:>7.1f}{r['detections_total']:>7d}{r['conf_mean']:>7.3f}"
        )

    print("\nDeteccoes por classe:")
    classes = sorted({c for r in rows for c in r["detections"]})
    print(f"{'classe':<12}" + "".join(f"{Path(r['model']).name:>26}" for r in rows))
    for c in classes:
        print(f"{c:<12}" + "".join(f"{r['detections'].get(c, 0):>26}" for r in rows))

    Path(args.out).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nJSON salvo em {args.out}")


if __name__ == "__main__":
    main()
