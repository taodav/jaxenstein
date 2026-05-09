"""ASCII maze parsing utilities."""

from __future__ import annotations

from dataclasses import dataclass
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

from jes.objects import (
    DOOR_SYMBOLS,
    KEY_COLOR_NONE,
    OBJECT_GOAL,
    OBJECT_NONE,
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


@dataclass(frozen=True)
class MazeBatch:
    wall_grids: jax.Array
    color_grids: jax.Array
    spawn_xy: jax.Array
    spawn_xy_options: jax.Array
    spawn_count: jax.Array
    spawn_theta: jax.Array
    goal_xy: jax.Array
    object_xy: jax.Array
    object_type: jax.Array
    object_color: jax.Array
    door_grids: jax.Array


def _clean_ascii(ascii_maze: str) -> list[str]:
    body = textwrap.dedent(ascii_maze).strip("\n")
    rows = body.splitlines()
    if not rows:
        raise ValueError("maze must contain at least one row")
    if any(len(row) == 0 for row in rows):
        raise ValueError("maze rows must not be empty")
    return rows


def parse_ascii_maze(ascii_maze: str, *, wall_color_id: int = 1) -> Maze:
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

            if char in WALL_SYMBOLS:
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
            elif char in DOOR_SYMBOLS:
                door_grid[row_idx, col_idx] = DOOR_SYMBOLS[char]

    if not spawn_xy_options:
        raise ValueError("maze must contain at least one spawn")
    if goal_xy is None:
        raise ValueError("maze must contain at least one goal")

    spawn_xy_array = np.stack(spawn_xy_options)
    return Maze(
        wall_grid=jnp.asarray(wall_grid),
        color_grid=jnp.asarray(color_grid),
        spawn_xy=jnp.asarray(spawn_xy_array[0]),
        spawn_xy_options=jnp.asarray(spawn_xy_array),
        spawn_count=jnp.asarray(len(spawn_xy_options), dtype=jnp.int32),
        spawn_theta=jnp.asarray(0.0, dtype=jnp.float32),
        goal_xy=jnp.asarray(goal_xy),
        object_xy=jnp.asarray(np.stack(object_xy), dtype=jnp.float32),
        object_type=jnp.asarray(object_type, dtype=jnp.int32),
        object_color=jnp.asarray(object_color, dtype=jnp.int32),
        door_grid=jnp.asarray(door_grid),
    )


def stack_mazes(mazes: list[Maze]) -> MazeBatch:
    """Stack mazes into a padded batch.

    Smaller maps are padded on the bottom/right with walls so dynamic maze
    indexing remains valid for JIT and vmap.
    """

    if not mazes:
        raise ValueError("expected at least one maze")

    max_h = max(int(maze.wall_grid.shape[0]) for maze in mazes)
    max_w = max(int(maze.wall_grid.shape[1]) for maze in mazes)
    max_objects = max(int(maze.object_type.shape[0]) for maze in mazes)
    max_spawns = max(int(maze.spawn_xy_options.shape[0]) for maze in mazes)

    wall_grids = []
    color_grids = []
    spawn_xy_options = []
    object_xy = []
    object_type = []
    object_color = []
    door_grids = []
    for maze in mazes:
        h, w = maze.wall_grid.shape
        pad_h = max_h - int(h)
        pad_w = max_w - int(w)
        wall_grids.append(
            jnp.pad(
                maze.wall_grid,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=True,
            )
        )
        color_grids.append(
            jnp.pad(
                maze.color_grid,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=1,
            )
        )
        door_grids.append(
            jnp.pad(
                maze.door_grid,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=KEY_COLOR_NONE,
            )
        )
        pad_spawns = max_spawns - int(maze.spawn_xy_options.shape[0])
        spawn_xy_options.append(
            jnp.pad(
                maze.spawn_xy_options,
                ((0, pad_spawns), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        )
        pad_objects = max_objects - int(maze.object_type.shape[0])
        object_xy.append(
            jnp.pad(
                maze.object_xy,
                ((0, pad_objects), (0, 0)),
                mode="constant",
                constant_values=0,
            )
        )
        object_type.append(
            jnp.pad(
                maze.object_type,
                (0, pad_objects),
                mode="constant",
                constant_values=OBJECT_NONE,
            )
        )
        object_color.append(
            jnp.pad(
                maze.object_color,
                (0, pad_objects),
                mode="constant",
                constant_values=KEY_COLOR_NONE,
            )
        )

    return MazeBatch(
        wall_grids=jnp.stack(wall_grids),
        color_grids=jnp.stack(color_grids),
        spawn_xy=jnp.stack([maze.spawn_xy for maze in mazes]),
        spawn_xy_options=jnp.stack(spawn_xy_options),
        spawn_count=jnp.asarray([maze.spawn_count for maze in mazes], dtype=jnp.int32),
        spawn_theta=jnp.stack([maze.spawn_theta for maze in mazes]),
        goal_xy=jnp.stack([maze.goal_xy for maze in mazes]),
        object_xy=jnp.stack(object_xy),
        object_type=jnp.stack(object_type),
        object_color=jnp.stack(object_color),
        door_grids=jnp.stack(door_grids),
    )


def parse_and_stack(ascii_mazes: list[str]) -> MazeBatch:
    return stack_mazes([parse_ascii_maze(ascii_maze) for ascii_maze in ascii_mazes])
