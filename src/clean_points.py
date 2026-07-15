"""Select, clean, and interpolate fish tracks from raw detection JSON."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


Point = dict[str, Any]
Tracks = dict[str, list[Point]]


def select_best_detections(raw_data: dict[str, Any]) -> Tracks:
    """Choose the highest-confidence detection per fish in each frame."""
    tracks: Tracks = {}
    for frame in raw_data["frames"]:
        best_by_fish: dict[int, dict[str, Any]] = {}
        for detection in frame["detections"]:
            fish_id = int(detection["fish_id"])
            current = best_by_fish.get(fish_id)
            if current is None or detection["confidence"] > current["confidence"]:
                best_by_fish[fish_id] = detection

        for fish_id, detection in best_by_fish.items():
            x, y = detection["center"]
            tracks.setdefault(str(fish_id), []).append(
                {
                    "frame": int(frame["frame"]),
                    "x": float(x),
                    "y": float(y),
                    "confidence": float(detection["confidence"]),
                    "source": "detected",
                }
            )
    return tracks


def clean_track(
    points: list[Point], max_jump_per_frame: float, max_gap_frames: int
) -> list[Point]:
    """Drop implausible jumps and linearly interpolate short gaps."""
    if not points:
        return []

    points = sorted(points, key=lambda point: point["frame"])
    cleaned = [points[0]]

    for point in points[1:]:
        previous = cleaned[-1]
        frame_gap = point["frame"] - previous["frame"]
        if frame_gap <= 0:
            continue

        distance = math.hypot(point["x"] - previous["x"], point["y"] - previous["y"])
        if distance / frame_gap > max_jump_per_frame:
            continue

        if 1 < frame_gap <= max_gap_frames:
            for offset in range(1, frame_gap):
                fraction = offset / frame_gap
                cleaned.append(
                    {
                        "frame": previous["frame"] + offset,
                        "x": previous["x"] + (point["x"] - previous["x"]) * fraction,
                        "y": previous["y"] + (point["y"] - previous["y"]) * fraction,
                        "confidence": None,
                        "source": "interpolated",
                    }
                )
        cleaned.append(point)

    return cleaned


def clean_tracks(
    raw_data: dict[str, Any], max_jump_per_frame: float, max_gap_frames: int
) -> dict[str, Any]:
    if raw_data.get("data_type") != "raw_detections":
        raise ValueError("Input is not raw detection data")

    selected = select_best_detections(raw_data)
    tracks = {
        fish_id: clean_track(points, max_jump_per_frame, max_gap_frames)
        for fish_id, points in selected.items()
    }
    return {
        "schema_version": 1,
        "data_type": "cleaned_tracks",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "raw_detection_file": None,
            "video": raw_data["video"],
            "model": raw_data.get("model"),
        },
        "settings": {
            "selection_method": "highest_confidence_per_fish_per_frame",
            "cleaning_method": "jump_filter_and_linear_interpolation",
            "max_jump_per_frame": max_jump_per_frame,
            "max_gap_frames": max_gap_frames,
        },
        "tracks": tracks,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-jump-per-frame", type=float, default=30.0)
    parser.add_argument("--max-gap-frames", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_jump_per_frame <= 0:
        raise ValueError("--max-jump-per-frame must be positive")
    if args.max_gap_frames < 0:
        raise ValueError("--max-gap-frames cannot be negative")

    raw_data = json.loads(args.input.read_text(encoding="utf-8"))
    output = clean_tracks(raw_data, args.max_jump_per_frame, args.max_gap_frames)
    output["source"]["raw_detection_file"] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved cleaned tracks to {args.output}")


if __name__ == "__main__":
    main()
