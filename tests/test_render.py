import jax
import jax.numpy as jnp

from jes.ascii import parse_ascii_maze
from jes.maps import MAZE_SIMPLE
from jes.render import raycast_fixed_step, render_first_person


def test_raycast_depth_changes_with_rotation():
    maze = parse_ascii_maze(MAZE_SIMPLE)

    east_depth, _ = raycast_fixed_step(
        maze.spawn_xy, maze.spawn_theta, maze.wall_grid, maze.color_grid
    )
    west_depth, _ = raycast_fixed_step(
        maze.spawn_xy, jnp.asarray(jnp.pi), maze.wall_grid, maze.color_grid
    )

    assert east_depth[32] > west_depth[32]


def test_render_shape_dtype_and_nonuniform_pixels():
    maze = parse_ascii_maze(MAZE_SIMPLE)

    rgb = render_first_person(
        maze.spawn_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
        object_xy=maze.object_xy,
        object_type=maze.object_type,
        object_color=maze.object_color,
        object_active=jnp.ones_like(maze.object_type, dtype=jnp.bool_),
    )

    assert rgb.shape == (64, 64, 3)
    assert rgb.dtype == jnp.uint8
    assert int(jnp.unique(rgb.reshape((-1, 3)), axis=0).shape[0]) > 2


def test_render_object_color_defaults_when_omitted():
    maze = parse_ascii_maze(MAZE_SIMPLE)

    rgb = render_first_person(
        maze.spawn_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
        object_xy=maze.object_xy,
        object_type=maze.object_type,
        object_active=jnp.ones_like(maze.object_type, dtype=jnp.bool_),
    )

    assert rgb.shape == (64, 64, 3)


def test_object_sprite_is_visible_only_when_active_and_in_view():
    maze = parse_ascii_maze(MAZE_SIMPLE)
    object_active = jnp.ones_like(maze.object_type, dtype=jnp.bool_)

    facing_goal = render_first_person(
        maze.spawn_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
        object_xy=maze.object_xy,
        object_type=maze.object_type,
        object_color=maze.object_color,
        object_active=object_active,
    )
    facing_away = render_first_person(
        maze.spawn_xy,
        jnp.asarray(jnp.pi, dtype=jnp.float32),
        maze.wall_grid,
        maze.color_grid,
        object_xy=maze.object_xy,
        object_type=maze.object_type,
        object_color=maze.object_color,
        object_active=object_active,
    )
    inactive = render_first_person(
        maze.spawn_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
        object_xy=maze.object_xy,
        object_type=maze.object_type,
        object_color=maze.object_color,
        object_active=jnp.zeros_like(maze.object_type, dtype=jnp.bool_),
    )

    goal_pixels = (
        (facing_goal[..., 0] > 220)
        & (facing_goal[..., 1] > 180)
        & (facing_goal[..., 2] < 120)
    )
    away_goal_pixels = (
        (facing_away[..., 0] > 220)
        & (facing_away[..., 1] > 180)
        & (facing_away[..., 2] < 120)
    )
    inactive_goal_pixels = (
        (inactive[..., 0] > 220)
        & (inactive[..., 1] > 180)
        & (inactive[..., 2] < 120)
    )

    assert int(jnp.sum(goal_pixels)) > 0
    assert int(jnp.sum(away_goal_pixels)) == 0
    assert int(jnp.sum(inactive_goal_pixels)) == 0


def test_render_jit_and_vmap():
    maze = parse_ascii_maze(MAZE_SIMPLE)
    positions = jnp.stack([maze.spawn_xy, maze.spawn_xy])
    thetas = jnp.asarray([0.0, jnp.pi / 2.0], dtype=jnp.float32)
    wall_grids = jnp.stack([maze.wall_grid, maze.wall_grid])
    color_grids = jnp.stack([maze.color_grid, maze.color_grid])
    object_xy = jnp.stack([maze.object_xy, maze.object_xy])
    object_type = jnp.stack([maze.object_type, maze.object_type])
    object_color = jnp.stack([maze.object_color, maze.object_color])
    object_active = jnp.ones_like(object_type, dtype=jnp.bool_)

    rgb = jax.jit(render_first_person)(
        maze.spawn_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
        object_xy=maze.object_xy,
        object_type=maze.object_type,
        object_color=maze.object_color,
        object_active=jnp.ones_like(maze.object_type, dtype=jnp.bool_),
    )
    batched = jax.vmap(render_first_person)(positions, thetas, wall_grids, color_grids)
    batched_with_objects = jax.vmap(
        lambda pos, theta, wall_grid, color_grid, xy, kind, color, active: render_first_person(
            pos,
            theta,
            wall_grid,
            color_grid,
            object_xy=xy,
            object_type=kind,
            object_color=color,
            object_active=active,
        )
    )(
        positions,
        thetas,
        wall_grids,
        color_grids,
        object_xy,
        object_type,
        object_color,
        object_active,
    )

    assert rgb.shape == (64, 64, 3)
    assert batched.shape == (2, 64, 64, 3)
    assert batched_with_objects.shape == (2, 64, 64, 3)
