"""Shared brush-mark geometry for backgrounds and fish paintings."""

from __future__ import annotations

import math
import random

from PIL import ImageDraw


def polygon_for_shape(
    shape: str, width: float, height: float, rotation: float, rng: random.Random
) -> list[tuple[float, float]] | None:
    if shape == "rectangle":
        points = [
            (-width / 2, -height / 2),
            (width / 2, -height / 2),
            (width / 2, height / 2),
            (-width / 2, height / 2),
        ]
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


def draw_brush_mark(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    size: tuple[float, float],
    rotation: float,
    shape: str,
    color: tuple[int, int, int, int],
    rng: random.Random,
) -> None:
    """Draw one brush mark using the project's supported shape primitives."""
    x, y = center
    width, height = size
    if shape == "circle":
        draw.ellipse(
            (x - width / 2, y - height / 2, x + width / 2, y + height / 2),
            fill=color,
        )
        return

    polygon = polygon_for_shape(shape, width, height, rotation, rng)
    if polygon is None:
        raise ValueError(f"Unsupported shape: {shape}")
    draw.polygon([(x + px, y + py) for px, py in polygon], fill=color)
