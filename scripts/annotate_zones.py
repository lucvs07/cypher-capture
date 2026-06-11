#!/usr/bin/env python3
"""
Interactive zone annotation tool.

Draw polygon zones over a camera frame with mouse clicks, then save them
to the zones JSON config for use by ZoneDetector at runtime.

Controls:
  Left-click  — add a polygon vertex
  ENTER       — close polygon and prompt for zone metadata
  R           — reset current polygon (discard pending points)
  Q           — quit and save all completed zones

Usage:
    python scripts/annotate_zones.py --source <video_path|0> [--output config/zones/zones.json]
"""
import argparse
import json
import uuid
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Video file or webcam index")
    p.add_argument("--output", default="config/zones/zones.json")
    return p.parse_args()


pending_points: list[tuple[int, int]] = []
saved_zones: list[dict] = []


def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        pending_points.append((x, y))


def prompt_zone_metadata(polygon: list[tuple[int, int]]) -> dict:
    name = input("  Zone name [Zone]: ").strip() or "Zone"
    raw_type = input("  Type — restricted / attention [restricted]: ").strip().lower()
    zone_type = raw_type if raw_type in ("restricted", "attention") else "restricted"
    raw_sev = input("  Severity 1–4 [3]: ").strip()
    severity = int(raw_sev) if raw_sev.isdigit() and 1 <= int(raw_sev) <= 4 else 3
    ppe_raw = input("  Required PPE (comma-separated, or blank): ").strip()
    ppe = [p.strip() for p in ppe_raw.split(",") if p.strip()] if ppe_raw else []
    return {
        "zone_id": str(uuid.uuid4())[:8],
        "name": name,
        "zone_type": zone_type,
        "polygon": polygon,
        "severity": severity,
        "required_ppe": ppe,
    }


def main() -> None:
    global pending_points, saved_zones
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"ERROR: could not read frame from '{args.source}'")
        return

    cv2.namedWindow("Zone Annotator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Zone Annotator", on_mouse)

    print("\nZone Annotator")
    print("  Left-click  → add vertex")
    print("  ENTER       → finish polygon & enter metadata")
    print("  R           → reset current polygon")
    print("  Q           → quit & save\n")

    while True:
        canvas = frame.copy()

        # Draw completed zones
        for z in saved_zones:
            pts = np.array(z["polygon"], dtype=np.int32)
            color = (0, 0, 200) if z["zone_type"] == "restricted" else (0, 140, 255)
            cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=2)
            cv2.putText(canvas, z["name"], pts[0].tolist(),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Draw pending polygon
        for pt in pending_points:
            cv2.circle(canvas, pt, 5, (0, 220, 220), -1)
        if len(pending_points) >= 2:
            cv2.polylines(
                canvas,
                [np.array(pending_points, dtype=np.int32)],
                isClosed=False,
                color=(0, 220, 220),
                thickness=2,
            )

        status = f"Zones: {len(saved_zones)}  |  Pending vertices: {len(pending_points)}"
        cv2.putText(canvas, status, (10, canvas.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Zone Annotator", canvas)
        key = cv2.waitKey(30) & 0xFF

        if key == 13 and len(pending_points) >= 3:  # ENTER
            print(f"\nPolygon with {len(pending_points)} vertices — enter zone metadata:")
            zone = prompt_zone_metadata(list(pending_points))
            saved_zones.append(zone)
            pending_points.clear()
            print(f"  Saved zone '{zone['name']}' ({zone['zone_type']}, severity {zone['severity']})")

        elif key == ord("r"):
            pending_points.clear()
            print("Pending polygon reset.")

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(saved_zones, f, indent=2)
    print(f"\nSaved {len(saved_zones)} zone(s) → {out_path}")


if __name__ == "__main__":
    main()
