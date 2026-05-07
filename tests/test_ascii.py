import jax.numpy as jnp

from jes.ascii import parse_ascii_maze, stack_mazes
from jes.objects import (
    KEY_COLOR_BLUE,
    KEY_COLOR_NONE,
    KEY_COLOR_RED,
    KEY_COLOR_YELLOW,
    OBJECT_GOAL,
    OBJECT_KEY,
    OBJECT_NONE,
)


def test_parse_ascii_maze_coordinates():
    maze = parse_ascii_maze(
        """
        #####
        #S.G#
        #####
        """
    )

    assert maze.wall_grid.shape == (3, 5)
    assert bool(maze.wall_grid[0, 0])
    assert not bool(maze.wall_grid[1, 1])
    assert jnp.allclose(maze.spawn_xy, jnp.asarray([1.5, 1.5]))
    assert jnp.allclose(maze.goal_xy, jnp.asarray([3.5, 1.5]))
    assert jnp.allclose(maze.object_xy, jnp.asarray([[3.5, 1.5]]))
    assert jnp.array_equal(maze.object_type, jnp.asarray([OBJECT_GOAL]))
    assert jnp.array_equal(maze.object_color, jnp.asarray([KEY_COLOR_YELLOW]))


def test_parse_ascii_maze_pickup_objects():
    maze = parse_ascii_maze(
        """
        #####
        #SKG#
        #####
        """
    )

    assert jnp.allclose(maze.object_xy, jnp.asarray([[2.5, 1.5], [3.5, 1.5]]))
    assert jnp.array_equal(maze.object_type, jnp.asarray([OBJECT_KEY, OBJECT_GOAL]))
    assert jnp.array_equal(
        maze.object_color, jnp.asarray([KEY_COLOR_RED, KEY_COLOR_YELLOW])
    )


def test_parse_ascii_maze_colored_keys_and_doors():
    maze = parse_ascii_maze(
        """
        #########
        #SrbRyYG#
        #########
        """
    )

    assert jnp.array_equal(
        maze.object_color,
        jnp.asarray(
            [KEY_COLOR_RED, KEY_COLOR_BLUE, KEY_COLOR_YELLOW, KEY_COLOR_YELLOW]
        ),
    )
    assert int(maze.door_grid[1, 4]) == KEY_COLOR_RED
    assert int(maze.door_grid[1, 6]) == KEY_COLOR_YELLOW


def test_stack_mazes_pads_with_walls():
    small = parse_ascii_maze(
        """
        #####
        #S.G#
        #####
        """
    )
    wide = parse_ascii_maze(
        """
        #######
        #S...G#
        #######
        """
    )

    batch = stack_mazes([small, wide])

    assert batch.wall_grids.shape == (2, 3, 7)
    assert bool(batch.wall_grids[0, 0, 6])
    assert int(batch.color_grids[0, 0, 6]) == 1
    assert batch.door_grids.shape == (2, 3, 7)
    assert batch.object_xy.shape == (2, 1, 2)
    assert batch.object_type.shape == (2, 1)
    assert batch.object_color.shape == (2, 1)
    assert int(batch.object_type[0, 0]) == OBJECT_GOAL


def test_stack_mazes_pads_objects_with_none():
    one_object = parse_ascii_maze(
        """
        #####
        #S.G#
        #####
        """
    )
    two_objects = parse_ascii_maze(
        """
        #####
        #SKG#
        #####
        """
    )

    batch = stack_mazes([one_object, two_objects])

    assert batch.object_xy.shape == (2, 2, 2)
    assert batch.object_type.shape == (2, 2)
    assert int(batch.object_type[0, 0]) == OBJECT_GOAL
    assert int(batch.object_type[0, 1]) == OBJECT_NONE
    assert int(batch.object_color[0, 1]) == KEY_COLOR_NONE
    assert jnp.array_equal(batch.object_type[1], jnp.asarray([OBJECT_KEY, OBJECT_GOAL]))
