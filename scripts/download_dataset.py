"""
Download and extract the Construction-PPE dataset into data/.
Run from the project root: python scripts/download_dataset.py
"""

import os
import zipfile
import urllib.request
from pathlib import Path

DATASET_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip"
DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "construction-ppe.zip"
EXTRACT_DIR = DATA_DIR / "construction-ppe"


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100)
    bar = "#" * int(pct / 2)
    print(f"\r  [{bar:<50}] {pct:.1f}%", end="", flush=True)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if EXTRACT_DIR.exists():
        print(f"Dataset already extracted at {EXTRACT_DIR}. Nothing to do.")
        return

    print(f"Downloading Construction-PPE dataset (~178 MB)...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH, reporthook=_progress)
    print()

    print(f"Extracting to {EXTRACT_DIR}...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(DATA_DIR)

    ZIP_PATH.unlink()
    print("Done. Dataset ready at:", EXTRACT_DIR)


if __name__ == "__main__":
    main()
