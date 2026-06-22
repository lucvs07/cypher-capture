# Cypher Capture — Industrial Vision AI

Real-time industrial safety monitoring for construction sites, built on [Ultralytics YOLO](https://docs.ultralytics.com/).

- **Phase 1 — PPE detection:** detects compliant and non-compliant Personal Protective Equipment from a webcam, video, or images, and optionally uploads violation frames to Cloudinary.
- **Phase 2 — Pose & behavioural analysis:** a unified pipeline that runs pose estimation (YOLO11-pose + ByteTrack) alongside PPE detection in the same loop, computes joint angles and a full **REBA** ergonomic score, classifies posture (normal / ergonomically inadequate / immediate risk / critical zone) with temporal hysteresis, and flags entry/approach to demarcated critical zones.

Normative references: **NR-17** (ergonomics), **NR-35** (work at height), **REBA** (Hignett & McAtamney, 2000). See [docs/taxonomy.md](docs/taxonomy.md) for the full posture taxonomy.

---

## Project structure

```
cypher-capture/
├── config/
│   ├── construction-ppe.yaml     # Phase 1 dataset YAML (paths + class names)
│   ├── pose_config.yaml          # Phase 2 single source of thresholds/params
│   └── zones/                    # zones.json (created by the annotator)
├── data/
│   ├── annotations/              # ground-truth JSON for evaluation
│   └── validation/               # local validation videos (git-ignored)
├── docs/
│   ├── taxonomy.md               # posture taxonomy, REBA, state machine
│   ├── auditoria-fase2-*.md      # technical audit
│   └── plano-testes-dia1.md      # day-1 test plan
├── models/
│   └── exp.pt                    # trained PPE detector (7 classes)
├── scripts/
│   ├── download_dataset.py       # fetch the Construction-PPE dataset
│   ├── annotate_zones.py         # draw critical-zone polygons
│   ├── annotate_gt.py            # label ground-truth posture per frame
│   ├── benchmark_pose.py         # pose-model latency benchmark
│   └── evaluate_pose.py          # PCK / accuracy / recall / latency harness
└── src/
    ├── detect.py                 # Phase 1 entry point (PPE only)
    ├── run_pipeline.py           # Phase 2 entry point (unified EPI + pose)
    ├── train.py                  # train the PPE detector
    ├── pose/                     # estimator, angles, REBA, classifier, temporal
    ├── zones/                    # zone manager + detector
    ├── fusion/                   # decision engine + unified pipeline
    └── utils/                    # video, visualization, upload helpers
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone git@github.com:lucvs07/cypher-capture.git
cd cypher-capture
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install --upgrade pip
```

### 2. Install PyTorch

The latency target (≤ 200 ms end-to-end) is only realistic on a **GPU**. Pick one:

```bash
# GPU (NVIDIA, CUDA 12.x) — recommended; required to meet the latency budget
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# CPU-only (works, but the unified pipeline will exceed the latency budget)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify the GPU is visible:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

The pose weights (`yolo11s-pose.pt`) download automatically on first run; to pre-fetch:

```bash
python -c "from ultralytics import YOLO; YOLO('yolo11s-pose.pt')"
```

### 4. (Phase 1 only) Configure Cloudinary

Only needed if you want Phase 1 to upload violation frames. Copy `.env.example` to `.env` and fill in the credentials. Otherwise pass `--no-upload`.

> **Always run commands from the project root** — model and config paths are relative to it.

---

## Models

| Model | File | Task | Notes |
|---|---|---|---|
| PPE detector | `models/exp.pt` | detect | 7 classes: `GLOVE, Goggles, HELMET, MASK, Person, VEST, boots` |
| Pose estimator | `yolo11s-pose.pt` | pose | auto-downloaded; 17 COCO keypoints |

> The PPE model emits **presence** classes only (no `no_*` violation classes), so the unified pipeline reports detected PPE but does not raise PPE-violation events. Inspect any weights with:
> `python -c "from ultralytics import YOLO; print(YOLO('models/exp.pt').names)"`

---

## Running — Phase 2 (unified EPI + pose)

The main entry point. Reads all parameters from `config/pose_config.yaml`.

```bash
# Webcam, with a live window (Linux: source 0 → /dev/video0)
python src/run_pipeline.py --source 0

# Video file
python src/run_pipeline.py --source data/validation/clip.mp4

# Headless (no GUI — servers / measurement)
python src/run_pipeline.py --source data/validation/clip.mp4 --no-display

# Save an annotated video
python src/run_pipeline.py --source data/validation/clip.mp4 --output outputs/captures/annot.mp4

# Use a critical-zone definition
python src/run_pipeline.py --source 0 --zones config/zones/zones.json

# A different config (e.g. one with device: cuda)
python src/run_pipeline.py --source 0 --config config/pose_config.yaml
```

| Flag | Default | Description |
|---|---|---|
| `--source` | — | webcam index, video path, or RTSP URL (required) |
| `--config` | `config/pose_config.yaml` | pipeline configuration |
| `--zones` | from config | critical-zone JSON override |
| `--output` | — | save annotated video to this path |
| `--no-display` | off | run without a GUI window |
| `--min-severity` | `2` | print events at or above this severity (0–4) |
| `--max-frames` | `0` | stop after N frames (0 = until source ends / `q`) |

Each frame prints a per-stage latency HUD (`pose | epi | analysis | overlay`), events at/above the severity threshold, and a final latency summary with the pass/fail verdict against the 200 ms budget. Press `q` to quit a windowed run.

**Selecting the device / model:** edit `config/pose_config.yaml` → `pipeline.device` (`cpu` | `cuda` | `mps`) and `pipeline.pose_model_path` (`yolo11n-pose.pt` for speed, `yolo11s-pose.pt` default, `yolo11m-pose.pt` for accuracy).

### Measured latency (this repo's reference numbers)

Same 5 s clip (3 people), `yolo11s-pose` + `exp.pt`:

| Device | p95 E2E | pose / frame | epi / frame | Verdict (≤ 200 ms) |
|---|---|---|---|---|
| CPU | ~333 ms | ~83 ms | ~204 ms | ❌ FAIL |
| GPU (RTX 3050) | ~44 ms | ~13 ms | ~28 ms | ✅ PASS |

---

## Running — Phase 1 (PPE detection only)

```bash
# Webcam (note the "usb" prefix here)
python src/detect.py --model models/exp.pt --source usb0 --no-upload

# Video / single image / directory of images
python src/detect.py --model models/exp.pt --source clip.mp4 --no-upload
python src/detect.py --model models/exp.pt --source image.jpg --no-upload
python src/detect.py --model models/exp.pt --source images/ --no-upload
```

| Flag | Default | Description |
|---|---|---|
| `--model` | — | path to YOLO weights (required) |
| `--source` | — | `usb0`, video, image, or directory (required) |
| `--thresh` | `0.5` | minimum confidence to display / flag |
| `--cooldown` | `5.0` | seconds between uploads on live streams |
| `--no-upload` | off | disable Cloudinary upload |

`detect.py` always opens a window (no headless mode).

---

## Critical zones

Draw zone polygons over a frame from the same camera/resolution you will run on:

```bash
python scripts/annotate_zones.py --source data/validation/clip.mp4 --output config/zones/zones.json
# left-click vertices → ENTER → enter name/type/severity → q to quit
```

A starting example lives at `config/zones/zones.example.json` (copy it to `config/zones/zones.json` and adjust the vertices to your resolution).

---

## Evaluation & benchmarking

**Pose-model latency in isolation:**

```bash
python scripts/benchmark_pose.py --source data/validation/clip.mp4 --frames 200 --track
```

**Posture metrics (PCK / accuracy / recall / false-alarm / latency).** First create ground truth, then evaluate:

```bash
python scripts/annotate_gt.py --video data/validation/clip.mp4 --output data/annotations/clip_gt.json
# label keys: n=normal e=ergonomic i=immediate_risk z=critical_zone | SPACE pause | . step | q finish

python scripts/evaluate_pose.py \
  --annotations data/annotations/clip_gt.json \
  --video data/validation/clip.mp4 \
  --zones config/zones/zones.json --device cuda
```

> Ground truth produced by `annotate_gt.py` is intended for the **classification** metrics (recall, accuracy, false alarms). Its keypoints come from the detector itself, so the reported PCK is self-referential — independent PCK requires manually-labelled keypoints.

---

## Training (Phase 1 detector)

```bash
python scripts/download_dataset.py          # fetch the Construction-PPE dataset
python src/train.py --model yolo11n.pt --epochs 100 --imgsz 640 --batch 16
cp outputs/runs/ppe_train/weights/best.pt models/exp.pt
```

---

## License

AGPL-3.0 — inherited from the Construction-PPE dataset.
