"""ASCII maze parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

from jaxenstein.objects import (
    DOOR_SYMBOLS,
    OBJECT_GOAL,
    OBJECT_SYMBOLS,
    WALL_SYMBOLS,
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


def _clean_ascii(ascii_maze: str) -> list[str]:
    body = textwrap.dedent(ascii_maze).strip("\n")
    rows = body.splitlines()
    if not rows:
        raise ValueError("maze must contain at least one row")
    if any(len(row) == 0 for row in rows):
        raise ValueError("maze rows must not be empty")
    return rows


def parse_ascii_maze(
    ascii_maze: str, *, wall_color_id: int = 1, require_goal: bool = True
) -> Maze:
    """Parse an ASCII maze into JAX arrays.

    Cell centers use x/y coordinates where x is the column index + 0.5 and y
    is the row index + 0.5.
    """

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

    allowed = (
        {".", "S", " "} | set(WALL_SYMBOLS) | set(OBJECT_SYMBOLS) | set(DOOR_SYMBOLS)
    )
    for row_idx, row in enumerate(rows):
        for col_idx, char in enumerate(row):
            if char not in allowed:
                raise ValueError(f"unsupported maze symbol {char!r}")

            if char in DOOR_SYMBOLS:
                wall_grid[row_idx, col_idx] = False
                color_grid[row_idx, col_idx] = 0
                door_grid[row_idx, col_idx] = DOOR_SYMBOLS[char]
                continue
            elif char in WALL_SYMBOLS:
                color_grid[row_idx, col_idx] = WALL_SYMBOLS[char]
                continue

            wall_grid[row_idx, col_idx] = False
            color_grid[row_idx, col_idx] = 0
            xy = np.array([col_idx + 0.5, row_idx + 0.5], dtype=np.float32)

            if char == "S":
                spawn_xy_options.append(xy)
            elif char in OBJECT_SYMBOLS:
                kind, color = OBJECT_SYMBOLS[char]
                object_xy.append(xy)
                object_type.append(kind)
                object_color.append(color)
                if kind == OBJECT_GOAL:
                    if goal_xy is None:
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
    )

