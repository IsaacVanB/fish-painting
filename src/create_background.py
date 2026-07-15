"""Create a painted background from a still image or a frame from a video."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from .brush import draw_brush_mark
except ImportError:
    from brush import draw_brush_mark


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def read_source_frame(
    input_path: Path, frame_position: float
) -> tuple[np.ndarray, int | None, str]:
    if input_path.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not open image: {input_path}")
        return frame, None, "image"

    frame, frame_index = read_video_frame(input_path, frame_position)
    return frame, frame_index, "video"


def read_video_frame(video_path: Path, frame_position: float) -> tuple[np.ndarray, int]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise ValueError(f"Video has no readable frames: {video_path}")

    frame_index = round((frame_count - 1) * frame_position)
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise ValueError(f"Could not read frame {frame_index} from {video_path}")
    return frame, frame_index


def detections_for_frame(raw_data: dict[str, Any], frame_index: int) -> list[dict[str, Any]]:
    if raw_data.get("data_type") != "raw_detections":
        raise ValueError("Detection input is not raw detection data")
    for frame in raw_data["frames"]:
        if int(frame["frame"]) == frame_index:
            return frame["detections"]
    raise ValueError(f"Raw detection data does not contain frame {frame_index}")


def detect_fish(
    frame: np.ndarray, model_path: Path, confidence: float
) -> list[dict[str, Any]]:
    """Run YOLO on one image and return detections in the raw JSON shape."""
    from ultralytics import YOLO

    result = YOLO(str(model_path))(frame, conf=confidence, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or boxes.cls is None:
        return []

    class_ids = boxes.cls.cpu().numpy().astype(int)
    confidences = boxes.conf.cpu().numpy()
    coordinates = boxes.xyxy.cpu().numpy()
    detections = []
    for class_id, score, box in zip(class_ids, confidences, coordinates):
        x1, y1, x2, y2 = (float(value) for value in box)
        detections.append(
            {
                "fish_id": int(class_id),
                "confidence": float(score),
                "bbox": [x1, y1, x2, y2],
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            }
        )
    return detections


def remove_fish(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    box_padding: float = 0.15,
    inpaint_radius: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Mask padded fish boxes and fill them from surrounding image pixels."""
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for detection in detections:
        x1, y1, x2, y2 = (float(value) for value in detection["bbox"])
        box_width, box_height = x2 - x1, y2 - y1
        center = (round((x1 + x2) / 2), round((y1 + y2) / 2))
        axes = (
            max(1, round(box_width * (0.5 + box_padding))),
            max(1, round(box_height * (0.5 + box_padding))),
        )
        cv2.ellipse(mask, center, axes, 0, 0, 360, 255, thickness=-1)

    if not np.any(mask):
        return frame.copy(), mask
    return cv2.inpaint(frame, mask, inpaint_radius, cv2.INPAINT_TELEA), mask


def soften_color(color: np.ndarray, amount: float = 0.18) -> tuple[int, int, int]:
    """Lighten sampled colors slightly to keep fish marks visually dominant."""
    softened = color.astype(float) * (1 - amount) + 245 * amount
    return tuple(int(value) for value in np.clip(softened, 0, 255))


def render_painted_background(
    frame: np.ndarray,
    rng: random.Random,
    stroke_sizes: tuple[float, ...] = (96, 48, 24, 12),
    opacity: int = 85,
    scale: int = 2,
) -> Image.Image:
    """Render sampled image colors as coarse-to-fine, edge-aligned marks."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb_frame.shape[:2]
    canvas = Image.new("RGBA", (width * scale, height * scale), (195, 191, 171, 255))
    draw = ImageDraw.Draw(canvas, "RGBA")

    for stroke_size in sorted(stroke_sizes, reverse=True):
        blur_sigma = max(stroke_size / 5, 1)
        sampled = cv2.GaussianBlur(rgb_frame, (0, 0), blur_sigma)
        gray = cv2.cvtColor(sampled, cv2.COLOR_RGB2GRAY)
        gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        spacing = max(2, round(stroke_size * 0.55))

        positions = []
        for grid_y in range(spacing // 2, height, spacing):
            for grid_x in range(spacing // 2, width, spacing):
                jitter = spacing * 0.45
                x = min(width - 1, max(0, round(grid_x + rng.uniform(-jitter, jitter))))
                y = min(height - 1, max(0, round(grid_y + rng.uniform(-jitter, jitter))))
                positions.append((x, y))
        rng.shuffle(positions)

        for x, y in positions:
            tangent = math.atan2(float(gradient_y[y, x]), float(gradient_x[y, x]))
            rotation = tangent + math.pi / 2 + rng.uniform(-0.25, 0.25)
            mark_width = stroke_size * rng.uniform(0.8, 1.25) * scale
            mark_height = stroke_size * rng.uniform(0.22, 0.42) * scale
            color = (*soften_color(sampled[y, x]), rng.randint(round(opacity * 0.75), opacity))
            draw_brush_mark(
                draw,
                (x * scale, y * scale),
                (mark_width, mark_height),
                rotation,
                "rectangle",
                color,
                rng,
            )

    return canvas.resize((width, height), Image.Resampling.LANCZOS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "--video",
        dest="input_path",
        required=True,
        type=Path,
        help="Still image or video to paint; --video remains as a compatibility alias.",
    )
    parser.add_argument(
        "--detections",
        type=Path,
        help="Raw video detection JSON used to remove fish from the selected frame.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="YOLO model used to find and remove fish from a still image.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Minimum confidence for still-image YOLO detections (default: 0.25).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output PNG path (default: paintings/backgrounds/<input name>.png).",
    )
    parser.add_argument("--frame-position", type=float, default=0.5)
    parser.add_argument("--box-padding", type=float, default=0.15)
    parser.add_argument("--inpaint-radius", type=float, default=7.0)
    parser.add_argument("--stroke-sizes", nargs="+", type=float, default=(200, 120, 120, 60, 60, 30)) # use multiple passes to prevent blank spaces
    parser.add_argument("--opacity", type=int, default=75)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.frame_position <= 1:
        raise ValueError("--frame-position must be between 0 and 1")
    if args.box_padding < 0 or args.inpaint_radius <= 0:
        raise ValueError("Mask padding cannot be negative and inpaint radius must be positive")
    if not 1 <= args.opacity <= 255 or args.scale < 1:
        raise ValueError("Opacity must be 1-255 and scale must be at least 1")
    if any(size <= 0 for size in args.stroke_sizes):
        raise ValueError("All stroke sizes must be positive")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")
    if args.detections and args.model:
        raise ValueError("Use either --detections or --model, not both")

    frame, frame_index, source_type = read_source_frame(
        args.input_path, args.frame_position
    )
    detections = []
    if args.detections:
        if source_type == "image":
            raise ValueError("--detections contains video frame data and cannot be used with an image")
        raw_data = json.loads(args.detections.read_text(encoding="utf-8"))
        expected_size = (
            int(raw_data["video"]["width"]),
            int(raw_data["video"]["height"]),
        )
        actual_size = (frame.shape[1], frame.shape[0])
        if actual_size != expected_size:
            raise ValueError(
                f"Video frame is {actual_size}, but detection data expects {expected_size}"
            )
        detections = detections_for_frame(raw_data, frame_index)
    elif args.model:
        if source_type != "image":
            raise ValueError("--model is for still images; use --detections with video input")
        if not args.model.is_file():
            raise FileNotFoundError(f"Model not found: {args.model}")
        detections = detect_fish(frame, args.model, args.confidence)

    repaired_frame, _ = remove_fish(frame, detections, args.box_padding, args.inpaint_radius)
    image = render_painted_background(
        repaired_frame,
        random.Random(args.seed),
        tuple(args.stroke_sizes),
        args.opacity,
        args.scale,
    )
    destination = args.output or Path("paintings/backgrounds") / f"{args.input_path.stem}.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination)
    source_description = (
        f"frame {frame_index} from {args.input_path}"
        if frame_index is not None
        else str(args.input_path)
    )
    print(f"Saved background to {destination} using {source_description}; "
          f"removed {len(detections)} fish region(s)")


if __name__ == "__main__":
    main()
