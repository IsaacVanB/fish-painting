"""Render a fish painting from cleaned track JSON."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


FISH_COLORS = {
    0: [(255, 238, 79), (253, 255, 142), (233, 136, 62)],
    1: [(226, 97, 47), (235, 131, 55), (235, 173, 48)],
    2: [(99, 156, 237), (21, 36, 64), (194, 179, 79)],
    3: [(247, 126, 55), (151, 113, 99), (205, 187, 230)],
    4: [(48, 56, 91), (232, 114, 30), (168, 182, 255)],
    5: [(16, 26, 23), (151, 224, 182), (213, 99, 26)],
}


def pick_weighted_color(colors: list[tuple[int, int, int]], rng: random.Random):
    if len(colors) != 3:
        raise ValueError("Each fish palette must contain exactly three colors")
    return rng.choices(colors, weights=[0.60, 0.25, 0.15], k=1)[0]


def polygon_for_shape(
    shape: str, width: float, height: float, rotation: float, rng: random.Random
) -> list[tuple[float, float]] | None:
    if shape == "rectangle":
        points = [(-width / 2, -height / 2), (width / 2, -height / 2),
                  (width / 2, height / 2), (-width / 2, height / 2)]
    elif shape == "triangle":
        points = [(0, -height / 2), (width / 2, height / 2), (-width / 2, height / 2)]
    elif shape == "blob":
        points = []
        for index in range(10):
            angle = 2 * math.pi * index / 10
            radius = width / 2 * (0.7 + 0.3 * rng.random())
            points.append((radius * math.cos(angle), radius * math.sin(angle)))
    else:
        return None

    cosine, sine = math.cos(rotation), math.sin(rotation)
    return [(x * cosine - y * sine, x * sine + y * cosine) for x, y in points]


def render_painting(
    tracks_data: dict[str, Any],
    fish_ids: list[int],
    background_path: Path | None,
    rng: random.Random,
    marks_per_point: int = 15,
    jitter_radius: float = 12,
    alpha_range: tuple[int, int] = (155, 255),
    size_range: tuple[float, float] = (6, 20),
    shape_types: tuple[str, ...] = ("rectangle",),
    scale: int = 4,
) -> Image.Image:
    if tracks_data.get("data_type") != "cleaned_tracks":
        raise ValueError("Input is not cleaned track data")

    video = tracks_data["source"]["video"]
    base_size = (int(video["width"]), int(video["height"]))
    if background_path:
        background = Image.open(background_path).convert("RGBA")
        if background.size != base_size:
            raise ValueError(
                f"Background is {background.size}, but video is {base_size}"
            )
    else:
        background = Image.new("RGBA", base_size, (255, 255, 255, 255))

    layer = Image.new("RGBA", (base_size[0] * scale, base_size[1] * scale), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    for fish_id in fish_ids:
        if fish_id not in FISH_COLORS:
            raise ValueError(f"No color palette is defined for fish {fish_id}")
        path = tracks_data["tracks"].get(str(fish_id), [])
        for index, point in enumerate(path):
            progress = index / max(len(path) - 1, 1)
            alpha_base = int(alpha_range[0] + (alpha_range[1] - alpha_range[0]) * progress)
            size_base = size_range[0] + (size_range[1] - size_range[0]) * progress

            for _ in range(marks_per_point):
                jitter_angle = rng.uniform(0, 2 * math.pi)
                jitter_distance = rng.uniform(0, jitter_radius)
                x = (point["x"] + jitter_distance * math.cos(jitter_angle)) * scale
                y = (point["y"] + jitter_distance * math.sin(jitter_angle)) * scale
                width = rng.uniform(size_base * 0.8, size_base * 1.2) * scale
                height = rng.uniform(size_base * 0.8, size_base * 1.2) * scale
                rotation = math.radians(rng.uniform(0, 360))
                alpha = rng.randint(int(alpha_base * 0.8), alpha_base)
                color = (*pick_weighted_color(FISH_COLORS[fish_id], rng), alpha)
                shape = rng.choice(shape_types)

                if shape == "circle":
                    draw.ellipse((x - width / 2, y - height / 2,
                                  x + width / 2, y + height / 2), fill=color)
                    continue

                polygon = polygon_for_shape(shape, width, height, rotation, rng)
                if polygon is None:
                    raise ValueError(f"Unsupported shape: {shape}")
                draw.polygon([(x + px, y + py) for px, py in polygon], fill=color)

    brush_layer = layer.resize(base_size, Image.Resampling.LANCZOS)
    result = background.copy()
    result.alpha_composite(brush_layer)
    return result


def output_path(path: Path) -> Path:
    if path.suffix.lower() == ".png":
        return path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path / f"fish_art_{timestamp}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--background", type=Path)
    parser.add_argument("--fish-ids", nargs="+", required=True, type=int)
    parser.add_argument("--marks-per-point", type=int, default=15)
    parser.add_argument("--jitter-radius", type=float, default=12)
    parser.add_argument("--alpha-range", nargs=2, type=int, default=(155, 255))
    parser.add_argument("--size-range", nargs=2, type=float, default=(6, 20))
    parser.add_argument(
        "--shapes", nargs="+", choices=("rectangle", "circle", "triangle", "blob"),
        default=("rectangle",)
    )
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tracks_data = json.loads(args.tracks.read_text(encoding="utf-8"))
    image = render_painting(
        tracks_data,
        args.fish_ids,
        args.background,
        random.Random(args.seed),
        args.marks_per_point,
        args.jitter_radius,
        tuple(args.alpha_range),
        tuple(args.size_range),
        tuple(args.shapes),
        args.scale,
    )
    destination = output_path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination)
    print(f"Saved painting to {destination}")


if __name__ == "__main__":
    main()
