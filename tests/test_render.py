import jax
import jax.numpy as jnp

from jaxenstein.maps.ascii import parse_ascii_maze
from jaxenstein.maps import MAZE_SIMPLE
from jaxenstein.objects import (
    DOOR_UNLOCKED,
    KEY_COLOR_BLUE,
    KEY_COLOR_NONE,
    KEY_COLOR_RED,
    KEY_COLOR_YELLOW,
    OBJECT_MEDKIT,
    door_wall_color_id,
)
from jaxenstein.render import (
    raycast_dda,
    raycast_fixed_step,
    render_first_person,
)


def test_raycast_depth_changes_with_rotation():
    maze = parse_ascii_maze(MAZE_SIMPLE)

    east_depth, _ = raycast_fixed_step(
        maze.spawn_xy, maze.spawn_theta, maze.wall_grid, maze.color_grid
    )
    west_depth, _ = raycast_fixed_step(
        maze.spawn_xy, jnp.asarray(jnp.pi), maze.wall_grid, maze.color_grid
    )

    assert east_depth[32] > west_depth[32]


def test_raycast_dda_hits_exact_grid_boundary():
    maze = parse_ascii_maze(MAZE_SIMPLE)
    pos = jnp.asarray([1.3, 1.5], dtype=jnp.float32)

    depth, wall_ids = raycast_dda(
        pos,
        jnp.asarray(0.0, dtype=jnp.float32),
        maze.wall_grid,
        maze.color_grid,
        img_w=65,
    )

    assert jnp.isclose(depth[32], 6.7, atol=1.0e-5)
    assert int(wall_ids[32]) == 1


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


def test_floor_pattern_and_wall_height_scale_affect_render():
    maze = parse_ascii_maze(MAZE_SIMPLE)

    flat = render_first_person(
        maze.spawn_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
        wall_height_scale=1.0,
        floor_pattern=False,
    )
    styled = render_first_person(
        maze.spawn_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
    )

    assert styled.shape == flat.shape
    assert not jnp.array_equal(styled, flat)
    assert int(jnp.unique(styled[40:].reshape((-1, 3)), axis=0).shape[0]) > int(
        jnp.unique(flat[40:].reshape((-1, 3)), axis=0).shape[0]
    )


def test_custom_floor_checker_colors_affect_render():
    maze = parse_ascii_maze(MAZE_SIMPLE)
    dark = jnp.asarray([9, 19, 29], dtype=jnp.float32)
    light = jnp.asarray([79, 89, 99], dtype=jnp.float32)

    rgb = render_first_person(
        maze.spawn_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
        floor_checker_dark_rgb=dark,
        floor_checker_light_rgb=light,
    )
    floor_pixels = rgb[40:].reshape((-1, 3))

    assert bool(jnp.any(jnp.all(floor_pixels == dark.astype(jnp.uint8), axis=1)))
    assert bool(jnp.any(jnp.all(floor_pixels == light.astype(jnp.uint8), axis=1)))


def test_key_sprite_is_sparse_key_shape():
    maze = parse_ascii_maze(
        """
        ########
        #S.r..G#
        ########
        """
    )

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
    red_pixels = (rgb[..., 0] > 170) & (rgb[..., 1] < 130) & (rgb[..., 2] < 130)

    assert int(jnp.sum(red_pixels)) > 0
    assert int(jnp.sum(red_pixels)) < 220


def test_medkit_sprite_renders_white_pack_with_red_cross():
    maze = parse_ascii_maze(
        """
        ######
        #S..G#
        ######
        """
    )

    rgb = render_first_person(
        maze.spawn_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
        object_xy=jnp.asarray([[2.5, 1.5]], dtype=jnp.float32),
        object_type=jnp.asarray([OBJECT_MEDKIT], dtype=jnp.int32),
        object_color=jnp.asarray([KEY_COLOR_RED], dtype=jnp.int32),
        object_active=jnp.asarray([True]),
    )
    white_pixels = (rgb[..., 0] > 220) & (rgb[..., 1] > 220) & (rgb[..., 2] > 200)
    red_pixels = (rgb[..., 0] > 180) & (rgb[..., 1] < 80) & (rgb[..., 2] < 80)

    assert int(jnp.sum(white_pixels)) > 0
    assert int(jnp.sum(red_pixels)) > 0


def test_closed_doors_render_with_color_coded_wall_ids():
    unlocked_door = int(door_wall_color_id(jnp.asarray(DOOR_UNLOCKED)))
    red_door = int(door_wall_color_id(jnp.asarray(KEY_COLOR_RED)))
    blue_door = int(door_wall_color_id(jnp.asarray(KEY_COLOR_BLUE)))
    yellow_door = int(door_wall_color_id(jnp.asarray(KEY_COLOR_YELLOW)))

    assert unlocked_door != 0
    assert unlocked_door == blue_door
    assert red_door != blue_door
    assert red_door != yellow_door
    assert blue_door != yellow_door
    assert int(door_wall_color_id(jnp.asarray(KEY_COLOR_NONE))) == 0


def test_object_sprite_is_visible_only_when_active_and_in_view():
    maze = parse_ascii_maze(MAZE_SIMPLE)
    object_active = jnp.ones_like(maze.object_type, dtype=jnp.bool_)
    viewer_xy = jnp.asarray([5.5, 3.5], dtype=jnp.float32)

    facing_goal = render_first_person(
        viewer_xy,
        maze.spawn_theta,
        maze.wall_grid,
        maze.color_grid,
        object_xy=maze.object_xy,
        object_type=maze.object_type,
        object_color=maze.object_color,
        object_active=object_active,
    )
    facing_away = render_first_person(
        viewer_xy,
        jnp.asarray(jnp.pi, dtype=jnp.float32),
        maze.wall_grid,
        maze.color_grid,
        object_xy=maze.object_xy,
        object_type=maze.object_type,
        object_color=maze.object_color,
        object_active=object_active,
    )
    inactive = render_first_person(
        viewer_xy,
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
