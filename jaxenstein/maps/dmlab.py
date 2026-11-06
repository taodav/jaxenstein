"""Converters for DeepMind Lab nav_maze .map files.

DMLab rewards and resets

All six DMLab environments give +1 for each apple collected and +10 for reaching
the goal. Apples disappear after collection and reappear on reset; collecting
apples does not terminate the episode. The goal or time limit ends the episode.
Initial headings are uniform over the full circle.

| Size | Static apples | Static spawn locations | Random goal candidates | Random apples per episode |
| --- | ---: | ---: | ---: | ---: |
| 01 | 5 | 43 | 50 | 5 |
| 02 | 13 | 128 | 135 | 13 |
| 03 | 26 | 251 | 260 | 26 |

Static levels sample uniformly from their designated starts, including starts
that share an apple. Random levels choose a goal uniformly from the source map's
apple locations, choose a uniform start more than eight maze-grid edges from the
goal, and scatter apples uniformly without replacement over the remaining
locations. The apple count is `floor(0.1 * (candidate_count - 1) + 0.5)`.
The source map's single start is a placeholder relocated by the level script.

These rules follow DeepMind Lab's
[static factory](https://github.com/google-deepmind/lab/blob/b1db91af5b4d2f3a24466f4632a3e5e1b0829cca/game_scripts/factories/seek_avoid_factory.lua),
[random-goal factory](https://github.com/google-deepmind/lab/blob/b1db91af5b4d2f3a24466f4632a3e5e1b0829cca/game_scripts/factories/random_goal_factory.lua),
and [spawn code](https://github.com/google-deepmind/lab/blob/b1db91af5b4d2f3a24466f4632a3e5e1b0829cca/engine/code/game/g_client.c).
JAXenstein retains its own raycast geometry, movement, and rendering.

Factory overrides include `apple_reward=0.0`, `randomize_spawn_heading=False`,
and, for random levels, `scattered_reward_density=0.0`. Other environment IDs
retain fixed initial headings by default.

Apples are environment-defined ObjectSpec values from dmlab_object_types(),
with positions supplied separately from geometry. Override object_types to
change their properties or sprite. Objects absent from an instance's catalog
are never materialized or drawn. Static levels with objects=[] omit apples;
random levels with scattered_reward_density=0 build only goal properties and
textures, with no fruit sampling. dmlab_map_to_spec() preserves both geometry
and placements; dmlab_map_to_ascii() intentionally exports geometry only.

**Compatibility:** DMLab results from versions before this change used omitted
apples, fixed headings, and a fixed start in the random-goal levels. Those results
are a different task configuration. Setting only the overrides above does not
reproduce the old random-goal start/goal distribution. Existing checkpoints have
different object counts; start new episodes when upgrading.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from jaxenstein.objects import COLORED_WALL_SYMBOLS, ObjectPlacement, ObjectSpec
from jaxenstein.sprites import coordinates, rgba

DMLAB_CELL_SIZE = 100.0
_POINT_RE = re.compile(
    r"\(\s*([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)\s+"
    r"([-+]?\d+(?:\.\d+)?)"
)
_PROPERTY_RE = re.compile(r'"([^"]+)"\s+"([^"]*)"')
_DECAL_RE = re.compile(r"decal/lab_games/dec_img_style(\d+)_(\d+)")


def dmlab_map_to_ascii(map_text: str, *, cell_size: float = DMLAB_CELL_SIZE) -> str:
    """Convert a grid-aligned DMLab nav_maze .map into JAXenstein ASCII."""

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

    if not starts:
        raise ValueError("DMLab map did not contain any info_player_start entities")
    if not goals and not apples:
        raise ValueError("DMLab map did not contain any goal or apple entities")

    cells = floor_cells | set(starts) | set(goals) | set(apples)
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
    for col, row in apples:
        grid[2 * row + 1][2 * col + 1] = "."
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
        (float(x), float(y)) for x, y, z in _POINT_RE.findall(block) if float(z) > min_z
    ]
    if not points:
        return None
    xs, ys = zip(*points)
    return min(xs), max(xs), min(ys), max(ys)


def _brush_bounds_from_planes(block: str) -> tuple[float, float, float, float] | None:
    x_planes: list[float] = []
    y_planes: list[float] = []
    for line in block.splitlines():
        points = [(float(x), float(y)) for x, y, _ in _POINT_RE.findall(line)[:3]]
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


@dataclass(frozen=True)
class DMLabMap:
    ascii_map: str
    objects: tuple[ObjectPlacement, ...]


def dmlab_map_to_spec(map_text: str, *, cell_size: float = DMLAB_CELL_SIZE) -> DMLabMap:
    """Convert geometry plus separate apple placements, retaining spawn overlap."""
    objects = []
    for entity in _top_level_blocks(map_text)[1:]:
        props = dict(_PROPERTY_RE.findall(entity))
        if props.get("classname") == "apple_reward":
            col, row = _origin_cell(props["origin"], cell_size)
            objects.append(ObjectPlacement("apple", (2 * col + 1.5, 2 * row + 1.5)))
    return DMLabMap(dmlab_map_to_ascii(map_text, cell_size=cell_size), tuple(objects))


def apple_sprite():
    """Generate DMLab's sprite once on the CPU, only when the catalog uses it."""
    x, y = coordinates()
    body = (((x + 0.27) / 0.62) ** 2 + ((y - 0.12) / 0.70) ** 2 <= 1) | (
        ((x - 0.27) / 0.62) ** 2 + ((y - 0.12) / 0.70) ** 2 <= 1
    )
    body &= ~((np.abs(x) < 0.16) & (y < -0.42))
    stem = (np.abs(x + 0.03) < 0.09) & (y > -0.96) & (y < -0.38)
    leaf = ((x - 0.27) / 0.30) ** 2 + ((y + 0.72) / 0.14) ** 2 <= 1
    core = ((x + 0.32) / 0.18) ** 2 + ((y + 0.12) / 0.27) ** 2 <= 1
    image = rgba(body | stem | leaf, core, (214, 38, 36), (255, 126, 99))
    image[stem, :3] = (104, 62, 27)
    image[leaf, :3] = (64, 160, 48)
    return image


def dmlab_object_types(*, apple_reward=1.0):
    return {"apple": ObjectSpec(sprite=apple_sprite, reward=apple_reward)}
