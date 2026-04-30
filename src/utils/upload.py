"""Cloudinary upload helpers for detection snapshots."""

import os
import hashlib
import datetime
import tempfile
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import cloudinary
import cloudinary.uploader
from dotenv import load_dotenv

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True,
)


@dataclass
class DetectionSnapshot:
    """Immutable pair of frames captured at the exact same instant."""
    track_id: int
    timestamp: str
    frame_raw: np.ndarray
    frame_annotated: np.ndarray

    def __post_init__(self):
        assert self.frame_raw.shape == self.frame_annotated.shape, \
            "Raw and annotated frames must have identical dimensions."

    @property
    def uid(self) -> str:
        digest = hashlib.md5(self.frame_raw.tobytes()).hexdigest()[:8]
        return f"{self.timestamp}_{digest}"


def upload_snapshot(snapshot: DetectionSnapshot, folder: str = "cypher-capture") -> dict:
    """Upload raw and annotated frames and return their Cloudinary URLs."""
    results = {}
    for variant, frame in (("raw", snapshot.frame_raw), ("annotated", snapshot.frame_annotated)):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, frame)
            response = cloudinary.uploader.upload(
                tmp.name,
                folder=folder,
                public_id=f"{snapshot.uid}_{variant}",
            )
            results[variant] = response["secure_url"]
        os.unlink(tmp.name)
    return results


def build_snapshot(track_id: int, frame_raw: np.ndarray, frame_annotated: np.ndarray) -> DetectionSnapshot:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return DetectionSnapshot(
        track_id=track_id,
        timestamp=ts,
        frame_raw=frame_raw.copy(),
        frame_annotated=frame_annotated.copy(),
    )
