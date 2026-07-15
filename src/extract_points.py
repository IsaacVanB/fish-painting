"""Extract raw YOLO detections from a video without modifying track data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from moviepy import VideoFileClip
from ultralytics import YOLO


def convert_mov_to_mp4(input_path: Path, output_path: Path | None = None) -> Path:
    """Convert a MOV video to MP4 and return the converted file path."""
    if input_path.suffix.lower() != ".mov":
        return input_path

    destination = output_path or input_path.with_suffix(".mp4")
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Converting {input_path} to {destination}")
    with VideoFileClip(str(input_path)) as clip:
        clip.write_videofile(str(destination), codec="libx264")
    return destination


def video_metadata(video_path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    metadata = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": capture.get(cv2.CAP_PROP_FPS),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    return metadata


def extract_detections(
    video_path: Path,
    model_path: Path,
    fish_ids: set[int] | None = None,
    confidence: float = 0.25,
) -> dict[str, Any]:
    """Run YOLO and retain every matching bounding box in every frame."""
    original_video_path = video_path
    video_path = convert_mov_to_mp4(video_path)
    metadata = video_metadata(video_path)
    model = YOLO(str(model_path))
    frames = []

    results = model(str(video_path), stream=True, conf=confidence, verbose=False)
    for frame_index, result in enumerate(results):
        detections = []
        boxes = result.boxes
        if boxes is not None and boxes.cls is not None:
            class_ids = boxes.cls.cpu().numpy().astype(int)
            confidences = boxes.conf.cpu().numpy()
            coordinates = boxes.xyxy.cpu().numpy()

            for class_id, score, box in zip(class_ids, confidences, coordinates):
                if fish_ids is not None and class_id not in fish_ids:
                    continue
                x1, y1, x2, y2 = (float(value) for value in box)
                detections.append(
                    {
                        "fish_id": int(class_id),
                        "confidence": float(score),
                        "bbox": [x1, y1, x2, y2],
                        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                    }
                )

        frames.append({"frame": frame_index, "detections": detections})

    metadata["processed_frame_count"] = len(frames)
    return {
        "schema_version": 1,
        "data_type": "raw_detections",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "video": {
            "path": str(video_path),
            "original_path": str(original_video_path),
            "converted_from_mov": video_path != original_video_path,
            **metadata,
        },
        "model": {"path": str(model_path)},
        "settings": {
            "fish_ids": sorted(fish_ids) if fish_ids is not None else None,
            "minimum_confidence": confidence,
        },
        "frames": frames,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: data/points/raw/<video name>.json).",
    )
    parser.add_argument("--fish-ids", nargs="+", type=int)
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum YOLO confidence to retain (default: 0.25).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be between 0 and 1")

    output = extract_detections(
        args.video, args.model, set(args.fish_ids) if args.fish_ids else None, args.confidence
    )
    destination = args.output or Path("data/points/raw") / f"{args.video.stem}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Saved raw detections to {destination}")


if __name__ == "__main__":
    main()
