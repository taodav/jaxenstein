"""Converters for DeepMind Lab nav_maze .map files."""

from __future__ import annotations

import math
import re

from jes.objects import COLORED_WALL_SYMBOLS


DMLAB_CELL_SIZE = 100.0
_POINT_RE = re.compile(
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)"
)
_PROPERTY_RE = re.compile(r'"([^"]+)"\s+"([^"]*)"')
_DECAL_RE = re.compile(r"decal/lab_games/dec_img_style(\d+)_(\d+)")


def dmlab_map_to_ascii(map_text: str, *, cell_size: float = DMLAB_CELL_SIZE) -> str:
    """Convert a grid-aligned DMLab nav_maze .map into Jaxenstein ASCII."""

    entities = _top_level_blocks(map_text)
    if not entities:
        raise ValueError("expected at least one .map entity")

    worldspawn = entities[0]
    floor_cells: set[tuple[int, int]] = set()
    wall_symbols: dict[tuple[int, int], str] = {}

    for child in _child_blocks(worldspawn):
        if "patchDef2" in child and "decal/lab_games/" in child:
            bounds = _bounds_from_points(child, min_z=1.0)
            if bounds is None:
                continue
            texture = re.search(r"decal/lab_games/\S+", child)
            if texture is None:
                continue
            for pos in _wall_positions(bounds, cell_size):
                wall_symbols[pos] = dmlab_decal_symbol(texture.group(0))
        elif "_floor_" in child:
            bounds = _brush_bounds_from_planes(child)
            if bounds is None:
                continue
            floor_cells.update(_floor_cells(bounds, cell_size))
        elif "_wall_" in child:
            bounds = _brush_bounds_from_planes(child)
            if bounds is None:
                continue
            symbol = "4" if "wall_red" in child else "#"
            for pos in _wall_positions(bounds, cell_size):
                wall_symbols.setdefault(pos, symbol)

    starts: list[tuple[int, int]] = []
    goals: list[tuple[int, int]] = []
    apples: list[tuple[int, int]] = []
    for entity in entities[1:]:
        props = dict(_PROPERTY_RE.findall(entity))
        classname = props.get("classname")
        origin = props.get("origin")
        if origin is None:
            continue
        cell = _origin_cell(origin, cell_size)
        if classname == "info_player_start":
            starts.append(cell)
        elif classname == "goal":
            goals.append(cell)
        elif classname == "apple_reward":
            apples.append(cell)

    if not goals:
        goals = apples
    if not starts:
        raise ValueError("DMLab map did not contain any info_player_start entities")
    if not goals:
        raise ValueError("DMLab map did not contain any goal entities")

    cells = floor_cells | set(starts) | set(goals)
    max_col = max(col for col, _ in cells)
    max_row = max(row for _, row in cells)
    height = max(max_row * 2 + 3, max((pos[0] + 1 for pos in wall_symbols), default=0))
    width = max(max_col * 2 + 3, max((pos[1] + 1 for pos in wall_symbols), default=0))
    grid = [["#" for _ in range(width)] for _ in range(height)]

    for col, row in floor_cells:
        grid[2 * row + 1][2 * col + 1] = "."
    _open_floor_connectors(grid, floor_cells, wall_symbols)

    for (row, col), symbol in wall_symbols.items():
        if 0 <= row < height and 0 <= col < width:
            grid[row][col] = symbol
    for col, row in goals:
        grid[2 * row + 1][2 * col + 1] = "G"
    for col, row in starts:
        grid[2 * row + 1][2 * col + 1] = "S"

    return "\n".join("".join(row).rstrip() for row in grid)


def dmlab_decal_symbol(texture: str) -> str:
    match = _DECAL_RE.search(texture)
    if match is None:
        raise ValueError(f"unsupported DMLab decal texture {texture!r}")
    style = int(match.group(1))
    number = int(match.group(2))
    index = (style - 1) * 20 + (number - 1)
    return COLORED_WALL_SYMBOLS[index]


def _top_level_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    depth = 0
    start: int | None = None
    for index, char in enumerate(text):
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0 and start is not None:
                blocks.append(text[start : index + 1])
    return blocks


def _child_blocks(block: str) -> list[str]:
    blocks: list[str] = []
    depth = 0
    start: int | None = None
    for index, char in enumerate(block):
        if char == "{":
            if depth == 1:
                start = index
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 1 and start is not None:
                blocks.append(block[start : index + 1])
                start = None
    return blocks


def _bounds_from_points(
    block: str,
    *,
    min_z: float = -math.inf,
) -> tuple[float, float, float, float] | None:
    points = [
        (float(x), float(y))
        for x, y, z in _POINT_RE.findall(block)
        if float(z) > min_z
    ]
    if not points:
        return None
    xs, ys = zip(*points)
    return min(xs), max(xs), min(ys), max(ys)


def _brush_bounds_from_planes(block: str) -> tuple[float, float, float, float] | None:
    x_planes: list[float] = []
    y_planes: list[float] = []
    for line in block.splitlines():
        points = [
            (float(x), float(y))
            for x, y, _ in _POINT_RE.findall(line)[:3]
        ]
        if len(points) < 3:
            continue
        xs, ys = zip(*points)
        if max(xs) - min(xs) < 1.0e-5:
            x_planes.append(xs[0])
        if max(ys) - min(ys) < 1.0e-5:
            y_planes.append(ys[0])
    if len(x_planes) < 2 or len(y_planes) < 2:
        return None
    return min(x_planes), max(x_planes), min(y_planes), max(y_planes)


def _floor_cells(
    bounds: tuple[float, float, float, float],
    cell_size: float,
) -> set[tuple[int, int]]:
    x0, x1, y0, y1 = bounds
    return {
        (col, row)
        for col in _covered_cells(x0, x1, cell_size)
        for row in _covered_cells(y0, y1, cell_size)
    }


def _wall_positions(
    bounds: tuple[float, float, float, float],
    cell_size: float,
    *,
    thickness: float = 2.0,
) -> list[tuple[int, int]]:
    x0, x1, y0, y1 = bounds
    if x1 - x0 <= thickness and y1 - y0 > thickness:
        col = int(round(((x0 + x1) * 0.5) / cell_size)) * 2
        return [(2 * row + 1, col) for row in _covered_cells(y0, y1, cell_size)]
    if y1 - y0 <= thickness and x1 - x0 > thickness:
        row = int(round(((y0 + y1) * 0.5) / cell_size)) * 2
        return [(row, 2 * col + 1) for col in _covered_cells(x0, x1, cell_size)]
    return []


def _covered_cells(low: float, high: float, cell_size: float) -> range:
    start = int(math.floor((low + 1.0e-5) / cell_size))
    end = int(math.ceil((high - 1.0e-5) / cell_size))
    return range(start, end)


def _origin_cell(origin: str, cell_size: float) -> tuple[int, int]:
    x, y, *_ = (float(value) for value in origin.split())
    return int(x // cell_size), int(y // cell_size)


def _open_floor_connectors(
    grid: list[list[str]],
    floor_cells: set[tuple[int, int]],
    wall_symbols: dict[tuple[int, int], str],
) -> None:
    for col, row in floor_cells:
        east = (2 * row + 1, 2 * col + 2)
        south = (2 * row + 2, 2 * col + 1)
        if (col + 1, row) in floor_cells and east not in wall_symbols:
            grid[2 * row + 1][2 * col + 2] = "."
        if (col, row + 1) in floor_cells and south not in wall_symbols:
            grid[2 * row + 2][2 * col + 1] = "."
