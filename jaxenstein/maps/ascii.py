"""ASCII maze parsing utilities."""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from jaxenstein.objects import (
    DOOR_SYMBOLS,
    KEY_COLOR_NONE,
    OBJECT_NAMES_BY_SYMBOL,
    OBJECT_NONE,
    OBJECT_SYMBOLS,
    WALL_SYMBOLS,
    ObjectPlacement,
)


@dataclass(frozen=True)
class Maze:
    wall_grid: jax.Array
    color_grid: jax.Array
    spawn_xy: jax.Array
    spawn_xy_options: jax.Array
    spawn_count: jax.Array
    spawn_theta: jax.Array
    goal_xy: jax.Array
    object_xy: jax.Array
    object_type: jax.Array
    object_color: jax.Array
    door_grid: jax.Array
    object_names: tuple[str, ...] = ()
    object_initial_active: jax.Array | None = None


def _clean_ascii(ascii_maze: str) -> list[str]:
    body = textwrap.dedent(ascii_maze).strip("\n")
    rows = body.splitlines()
    if not rows:
        raise ValueError("maze must contain at least one row")
    if any(len(row) == 0 for row in rows):
        raise ValueError("maze rows must not be empty")
    return rows


def parse_ascii_maze(
    ascii_maze: str,
    *,
    wall_color_id: int = 1,
    require_goal: bool = True,
    objects: Sequence[ObjectPlacement] = (),
    object_symbols: Mapping[str, str] | None = None,
) -> Maze:
    """Parse an ASCII maze into JAX arrays.

    Cell centers use x/y coordinates where x is the column index + 0.5 and y
    is the row index + 0.5.
    """

    symbols = {**OBJECT_NAMES_BY_SYMBOL, **(object_symbols or {})}
    if any(
        len(symbol) != 1 or symbol in DOOR_SYMBOLS or symbol == "S"
        for symbol in symbols
    ):
        raise ValueError(
            "Object symbols must be single characters distinct from doors and S"
        )
    rows = _clean_ascii(ascii_maze)
    height = len(rows)
    width = max(len(row) for row in rows)

    wall_grid = np.ones((height, width), dtype=np.bool_)
    color_grid = np.full((height, width), wall_color_id, dtype=np.int32)
    door_grid = np.zeros((height, width), dtype=np.int32)
    spawn_xy_options: list[np.ndarray] = []
    goal_xy: np.ndarray | None = None
    object_xy: list[np.ndarray] = []
    object_type: list[int] = []
    object_color: list[int] = []
    object_names: list[str] = []
    object_initial_active: list[bool] = []

    allowed = {".", "S", " "} | set(WALL_SYMBOLS) | set(symbols) | set(DOOR_SYMBOLS)
    for row_idx, row in enumerate(rows):
        for col_idx, char in enumerate(row):
            if char not in allowed:
                raise ValueError(f"unsupported maze symbol {char!r}")

            if char in DOOR_SYMBOLS:
                wall_grid[row_idx, col_idx] = False
                color_grid[row_idx, col_idx] = 0
                door_grid[row_idx, col_idx] = DOOR_SYMBOLS[char]
                continue
            elif char in WALL_SYMBOLS and char not in symbols:
                color_grid[row_idx, col_idx] = WALL_SYMBOLS[char]
                continue

            wall_grid[row_idx, col_idx] = False
            color_grid[row_idx, col_idx] = 0
            xy = np.array([col_idx + 0.5, row_idx + 0.5], dtype=np.float32)

            if char == "S":
                spawn_xy_options.append(xy)
            if char in symbols:
                object_xy.append(xy)
                object_names.append(symbols[char])
                object_initial_active.append(True)

    for obj in objects:
        if not isinstance(obj, ObjectPlacement):
            raise TypeError("objects must contain ObjectPlacement values")
        xy = np.asarray(obj.pos, dtype=np.float32)
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError("Object positions must be finite (x, y) pairs")
        col, row = np.floor(xy).astype(int)
        if not (0 <= col < width and 0 <= row < height) or wall_grid[row, col]:
            raise ValueError(f"Object {obj.kind!r} must be placed on open floor")
        object_xy.append(xy)
        object_names.append(obj.kind)
        object_initial_active.append(bool(obj.active))

    order = sorted(
        range(len(object_xy)), key=lambda i: (object_xy[i][1], object_xy[i][0])
    )
    object_xy = [object_xy[i] for i in order]
    object_names = [object_names[i] for i in order]
    object_initial_active = [object_initial_active[i] for i in order]
    legacy = {
        OBJECT_NAMES_BY_SYMBOL[symbol]: value
        for symbol, value in OBJECT_SYMBOLS.items()
    }
    for name, xy in zip(object_names, object_xy):
        kind, color = legacy.get(name, (OBJECT_NONE, KEY_COLOR_NONE))
        object_type.append(kind)
        object_color.append(color)
        if name == "goal" and goal_xy is None:
            goal_xy = xy

    if not spawn_xy_options:
        raise ValueError("maze must contain at least one spawn")
    if goal_xy is None and require_goal:
        raise ValueError("maze must contain at least one goal")
    if goal_xy is None:
        goal_xy = np.zeros((2,), dtype=np.float32)
    if object_xy:
        object_xy_array = np.stack(object_xy)
    else:
        object_xy_array = np.zeros((0, 2), dtype=np.float32)

    spawn_xy_array = np.stack(spawn_xy_options)
    return Maze(
        wall_grid=jnp.asarray(wall_grid),
        color_grid=jnp.asarray(color_grid),
        spawn_xy=jnp.asarray(spawn_xy_array[0]),
        spawn_xy_options=jnp.asarray(spawn_xy_array),
        spawn_count=jnp.asarray(len(spawn_xy_options), dtype=jnp.int32),
        spawn_theta=jnp.asarray(0.0, dtype=jnp.float32),
        goal_xy=jnp.asarray(goal_xy),
        object_xy=jnp.asarray(object_xy_array, dtype=jnp.float32),
        object_type=jnp.asarray(object_type, dtype=jnp.int32),
        object_color=jnp.asarray(object_color, dtype=jnp.int32),
        door_grid=jnp.asarray(door_grid),
        object_names=tuple(object_names),
        object_initial_active=jnp.asarray(object_initial_active, dtype=jnp.bool_),
    )
