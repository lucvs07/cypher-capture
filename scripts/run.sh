#!/usr/bin/env bash
# Quick launcher for common inference tasks.
# Run from the project root: bash scripts/run.sh

set -euo pipefail

MODEL="${MODEL:-models/best.pt}"
SOURCE="${SOURCE:-usb0}"
THRESH="${THRESH:-0.5}"

if [[ ! -f "$MODEL" ]]; then
  echo "Model not found at $MODEL"
  echo "Either copy your weights there or set MODEL=path/to/weights.pt"
  exit 1
fi

python src/detect.py --model "$MODEL" --source "$SOURCE" --thresh "$THRESH" "$@"
