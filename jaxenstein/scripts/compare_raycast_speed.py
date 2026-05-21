"""Compare fixed-step ray marching with grid DDA raycasting.

Example:
    uv run python scripts/compare_raycast_speed.py --map my-way-home --widths 64 160 320
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import time

import jax
import jax.numpy as jnp

from jaxenstein.maps.ascii import parse_ascii_maze
from jaxenstein.maps import MAPS_BY_NAME
from jaxenstein.render import raycast_dda, raycast_fixed_step


RaycastFn = Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array],
]


def _block_until_ready(tree: object) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        leaf.block_until_ready()


def _time_first_call(fn: RaycastFn, args: tuple[jax.Array, ...]) -> float:
    start = time.perf_counter()
    result = fn(*args)
    _block_until_ready(result)
    return (time.perf_counter() - start) * 1000.0


def _time_repeated_calls(
    fn: RaycastFn,
    args: tuple[jax.Array, ...],
    *,
    repeats: int,
) -> float:
    start = time.perf_counter()
    for _ in range(repeats):
        result = fn(*args)
        _block_until_ready(result)
    return (time.perf_counter() - start) * 1000.0 / repeats


def _make_fixed_step_fn(
    *,
    img_w: int,
    max_depth: float,
    num_depth_samples: int,
) -> RaycastFn:
    return jax.jit(
        lambda pos, theta, wall_grid, color_grid: raycast_fixed_step(
            pos,
            theta,
            wall_grid,
            color_grid,
            img_w=img_w,
            max_depth=max_depth,
            num_depth_samples=num_depth_samples,
        )
    )


def _make_dda_fn(*, img_w: int, max_depth: float) -> RaycastFn:
    return jax.jit(
        lambda pos, theta, wall_grid, color_grid: raycast_dda(
            pos,
            theta,
            wall_grid,
            color_grid,
            img_w=img_w,
            max_depth=max_depth,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark JIT-compiled fixed-step and DDA raycasters."
    )
    parser.add_argument("--map", choices=sorted(MAPS_BY_NAME), default="my-way-home")
    parser.add_argument("--widths", type=int, nargs="+", default=[64, 160, 320])
    parser.add_argument("--max-depth", type=float, default=16.0)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--spawn-index", type=int, default=0)
    parser.add_argument("--theta", type=float, default=0.0)
    args = parser.parse_args()

    maze = parse_ascii_maze(MAPS_BY_NAME[args.map])
    spawn_count = int(maze.spawn_count)
    spawn_index = max(0, min(args.spawn_index, spawn_count - 1))
    pos = maze.spawn_xy_options[spawn_index]
    theta = maze.spawn_theta + jnp.asarray(args.theta, dtype=jnp.float32)
    raycast_args = (pos, theta, maze.wall_grid, maze.color_grid)

    print(
        f"map={args.map} spawns={spawn_count} spawn_index={spawn_index} "
        f"max_depth={args.max_depth:g} fixed_samples={args.samples} repeats={args.repeats}"
    )
    print(
        f"{'width':>6} {'fixed compile':>14} {'dda compile':>12} "
        f"{'fixed run':>11} {'dda run':>9} {'speedup':>8}"
    )
    print(
        f"{'':>6} {'ms':>14} {'ms':>12} {'ms/call':>11} {'ms/call':>9} {'x':>8}"
    )

    for width in args.widths:
        fixed_fn = _make_fixed_step_fn(
            img_w=width,
            max_depth=args.max_depth,
            num_depth_samples=args.samples,
        )
        dda_fn = _make_dda_fn(img_w=width, max_depth=args.max_depth)

        fixed_compile_ms = _time_first_call(fixed_fn, raycast_args)
        dda_compile_ms = _time_first_call(dda_fn, raycast_args)
        fixed_run_ms = _time_repeated_calls(
            fixed_fn,
            raycast_args,
            repeats=args.repeats,
        )
        dda_run_ms = _time_repeated_calls(
            dda_fn,
            raycast_args,
            repeats=args.repeats,
        )
        speedup = fixed_run_ms / dda_run_ms
        print(
            f"{width:6d} {fixed_compile_ms:14.2f} {dda_compile_ms:12.2f} "
            f"{fixed_run_ms:11.4f} {dda_run_ms:9.4f} {speedup:8.2f}"
        )


if __name__ == "__main__":
    main()
