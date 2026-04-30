# Cypher Capture — PPE Detection

Real-time Personal Protective Equipment (PPE) detection for construction sites using [Ultralytics YOLO](https://docs.ultralytics.com/) and the [Construction-PPE dataset](https://docs.ultralytics.com/datasets/detect/construction-ppe/). The system detects compliant and non-compliant PPE usage simultaneously and optionally uploads violation frames to Cloudinary.

---

## Dataset

| Property | Value |
|---|---|
| Source | [Construction-PPE — Ultralytics](https://docs.ultralytics.com/datasets/detect/construction-ppe/) |
| License | AGPL-3.0 |
| Total images | 1,416 |
| Splits | Train: 1,132 · Val: 143 · Test: 141 |
| Classes | 11 |

### Classes

| ID | Label | Type |
|---|---|---|
| 0 | Gloves | Compliant |
| 1 | Goggles | Compliant |
| 2 | Helmet | Compliant |
| 3 | Person | Neutral |
| 4 | Vest | Compliant |
| 5 | Boots | Compliant |
| 6 | no_gloves | **Violation** |
| 7 | no_goggle | **Violation** |
| 8 | no_helmet | **Violation** |
| 9 | no_boots | **Violation** |
| 10 | none | **Violation** |

---

## Project Structure

```
cypher-capture/
├── config/
│   └── construction-ppe.yaml    # Dataset YAML (paths + class names)
├── data/                        # Dataset root (populated by download script)
├── models/                      # Trained weights (copy best.pt here)
├── notebooks/                   # Exploratory notebooks
├── outputs/
│   ├── captures/                # Saved inference frames
│   └── runs/                    # Training experiment logs and weights
├── scripts/
│   ├── download_dataset.py      # Fetch and extract the dataset
│   └── run.sh                   # Shell wrapper for quick inference
└── src/
    ├── detect.py                # Inference entry point
    ├── train.py                 # Training entry point
    └── utils/
        ├── upload.py            # Cloudinary upload helpers
        └── visualization.py     # Bounding box drawing with color coding
```

---

## Setup

### 1. Clone and create a virtual environment

```bash
git clone git@github.com:lucvs07/cypher-capture.git
cd cypher-capture
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your credentials:

```env
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

# Optional — receives a POST with violation snapshot URLs
API_ENDPOINT=http://localhost:3000/api/detections
```

---

## Download the Dataset

```bash
python scripts/download_dataset.py
```

This downloads the Construction-PPE zip (~178 MB), extracts it to `data/construction-ppe/`, and removes the archive.

---

## Training

Fine-tune a pre-trained YOLO model on the PPE dataset:

```bash
# Default: yolo11n.pt, 100 epochs, batch 16, imgsz 640
python src/train.py

# Custom settings
python src/train.py \
  --model yolo11s.pt \
  --epochs 150 \
  --batch 32 \
  --imgsz 640 \
  --patience 25 \
  --device 0 \
  --name ppe_v1
```

Training results (weights, metrics, confusion matrix) are saved under `outputs/runs/<name>/`.

After training, copy the best weights to `models/`:

```bash
cp outputs/runs/ppe_v1/weights/best.pt models/best.pt
```

---

## Inference

### Webcam (live stream)

```bash
python src/detect.py --model models/best.pt --source usb0
```

### Video file

```bash
python src/detect.py --model models/best.pt --source path/to/video.mp4 --thresh 0.45
```

### Single image or directory

```bash
python src/detect.py --model models/best.pt --source path/to/image.jpg
python src/detect.py --model models/best.pt --source path/to/images/
```

### Shell wrapper

```bash
# Uses MODEL / SOURCE / THRESH env vars (defaults to usb0, thresh 0.5)
bash scripts/run.sh

# Override via environment
MODEL=models/best.pt SOURCE=video.mp4 THRESH=0.4 bash scripts/run.sh
```

### Inference flags

| Flag | Default | Description |
|---|---|---|
| `--model` | — | Path to YOLO weights (required) |
| `--source` | — | `usb0`, video, image, or directory (required) |
| `--thresh` | `0.5` | Minimum confidence to display / flag a detection |
| `--cooldown` | `5.0` | Seconds between uploads for live streams |
| `--no-upload` | off | Disable Cloudinary upload entirely |

---

## Color coding

Bounding boxes are colored by PPE compliance status:

- **Green** — PPE item detected (compliant)
- **Red** — Missing PPE (violation)
- **Amber** — Person (neutral)

Violation frames are automatically uploaded to Cloudinary and optionally forwarded to an API endpoint.

---

## License

AGPL-3.0 — inherited from the Construction-PPE dataset.
