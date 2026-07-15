"""Run fish detection, cleanup, background creation, and painting end to end."""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

from PIL import Image

try:
    from .clean_points import clean_tracks
    from .create_background import (
        detections_for_frame,
        read_video_frame,
        remove_fish,
        render_painted_background,
    )
    from .create_painting import FISH_COLORS, output_path, render_painting
    from .extract_points import convert_mov_to_mp4, extract_detections
except ImportError:
    from clean_points import clean_tracks
    from create_background import (
        detections_for_frame,
        read_video_frame,
        remove_fish,
        render_painted_background,
    )
    from create_painting import FISH_COLORS, output_path, render_painting
    from extract_points import convert_mov_to_mp4, extract_detections


def build_painting(args: argparse.Namespace, video_path: Path) -> Image.Image:
    """Run every stage in memory and return the final image."""
    print("Extracting fish detections...")
    raw_data = extract_detections(video_path, args.model, confidence=args.confidence)

    print("Creating painted background...")
    frame, frame_index = read_video_frame(video_path, args.frame_position)
    detections = detections_for_frame(raw_data, frame_index)
    repaired_frame, _ = remove_fish(
        frame, detections, args.box_padding, args.inpaint_radius
    )
    background = render_painted_background(
        repaired_frame,
        random.Random(args.seed),
        tuple(args.background_stroke_sizes),
        args.background_opacity,
        args.background_scale,
    )

    print("Cleaning fish tracks...")
    cleaned_data = clean_tracks(
        raw_data, args.max_jump_per_frame, args.max_gap_frames
    )
    fish_ids = args.fish_ids
    if fish_ids is None:
        fish_ids = sorted(
            int(fish_id)
            for fish_id, points in cleaned_data["tracks"].items()
            if points and int(fish_id) in FISH_COLORS
        )
    if not fish_ids:
        raise ValueError("No fish with configured color palettes were detected")

    print(f"Painting tracks for fish IDs: {fish_ids}")
    return render_painting(
        cleaned_data,
        fish_ids,
        background,
        random.Random(None if args.seed is None else args.seed + 1),
        args.marks_per_point,
        args.jitter_radius,
        tuple(args.alpha_range),
        tuple(args.size_range),
        tuple(args.shapes),
        args.painting_scale,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Final PNG or directory (default: paintings/fish_art_<datetime>.png).",
    )
    parser.add_argument("--fish-ids", nargs="+", type=int)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--max-jump-per-frame", type=float, default=30.0)
    parser.add_argument("--max-gap-frames", type=int, default=50)
    parser.add_argument("--frame-position", type=float, default=0.5)
    parser.add_argument("--box-padding", type=float, default=0.15)
    parser.add_argument("--inpaint-radius", type=float, default=7.0)
    parser.add_argument(
        "--background-stroke-sizes",
        nargs="+",
        type=float,
        default=(200, 120, 120, 60, 60, 30),
    )
    parser.add_argument("--background-opacity", type=int, default=75)
    parser.add_argument("--background-scale", type=int, default=2)
    parser.add_argument("--marks-per-point", type=int, default=15)
    parser.add_argument("--jitter-radius", type=float, default=12)
    parser.add_argument("--alpha-range", nargs=2, type=int, default=(155, 255))
    parser.add_argument("--size-range", nargs=2, type=float, default=(6, 20))
    parser.add_argument(
        "--shapes",
        nargs="+",
        choices=("rectangle", "circle", "triangle", "blob"),
        default=("rectangle",),
    )
    parser.add_argument("--painting-scale", type=int, default=4)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")
    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not 0 <= args.confidence <= 1 or not 0 <= args.frame_position <= 1:
        raise ValueError("Confidence and frame position must be between 0 and 1")
    if args.max_jump_per_frame <= 0 or args.max_gap_frames < 0:
        raise ValueError("Jump threshold must be positive and gap frames cannot be negative")
    if args.box_padding < 0 or args.inpaint_radius <= 0:
        raise ValueError("Mask padding cannot be negative and inpaint radius must be positive")
    if not 1 <= args.background_opacity <= 255:
        raise ValueError("Background opacity must be between 1 and 255")
    if args.background_scale < 1 or args.painting_scale < 1:
        raise ValueError("Rendering scales must be at least 1")
    if any(size <= 0 for size in args.background_stroke_sizes):
        raise ValueError("Background stroke sizes must be positive")


def main() -> None:
    args = parse_args()
    validate_args(args)
    destination = output_path(args.output or Path("paintings"))

    if args.video.suffix.lower() == ".mov":
        with tempfile.TemporaryDirectory(prefix="fish-art-") as temporary_directory:
            converted_path = Path(temporary_directory) / f"{args.video.stem}.mp4"
            convert_mov_to_mp4(args.video, converted_path)
            image = build_painting(args, converted_path)
    else:
        image = build_painting(args, args.video)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination)
    print(f"Saved final painting to {destination}")


if __name__ == "__main__":
    main()
