import jax.numpy as jnp

from jes.ascii import parse_ascii_maze, stack_mazes
from jes.maps import MAZE_KEY_CORRIDOR, MAZE_MY_WAY_HOME, MAZE_MY_WAY_HOME_COLORLESS
from jes.objects import (
    DOOR_UNLOCKED,
    DOOR_UNLOCKED_YELLOW,
    KEY_COLOR_BLUE,
    KEY_COLOR_NONE,
    KEY_COLOR_RED,
    KEY_COLOR_YELLOW,
    OBJECT_GOAL,
    OBJECT_KEY,
    OBJECT_NONE,
)


def _reachable_cells(
    maze: list[str], start: tuple[int, int], passable: set[str] | None = None
) -> set[tuple[int, int]]:
    if passable is None:
        passable = {".", "S", "G"}
    frontier = [start]
    seen = {start}
    while frontier:
        row, col = frontier.pop()
        for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            next_cell = (row + drow, col + dcol)
            next_row, next_col = next_cell
            if next_cell in seen or maze[next_row][next_col] not in passable:
                continue
            seen.add(next_cell)
            frontier.append(next_cell)
    return seen


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
    assert jnp.allclose(maze.spawn_xy_options, jnp.asarray([[1.5, 1.5]]))
    assert int(maze.spawn_count) == 1
    assert jnp.allclose(maze.goal_xy, jnp.asarray([3.5, 1.5]))
    assert jnp.allclose(maze.object_xy, jnp.asarray([[3.5, 1.5]]))
    assert jnp.array_equal(maze.object_type, jnp.asarray([OBJECT_GOAL]))
    assert jnp.array_equal(maze.object_color, jnp.asarray([KEY_COLOR_YELLOW]))


def test_parse_ascii_maze_multiple_spawn_options():
    maze = parse_ascii_maze(
        """
        #######
        #S.S.G#
        #######
        """
    )

    assert jnp.allclose(maze.spawn_xy, jnp.asarray([1.5, 1.5]))
    assert jnp.allclose(
        maze.spawn_xy_options,
        jnp.asarray([[1.5, 1.5], [3.5, 1.5]]),
    )
    assert int(maze.spawn_count) == 2


def test_parse_ascii_maze_pickup_objects():
    maze = parse_ascii_maze(
        """
        #####
        #SrG#
        #####
        """
    )

    assert jnp.allclose(maze.object_xy, jnp.asarray([[2.5, 1.5], [3.5, 1.5]]))
    assert jnp.array_equal(maze.object_type, jnp.asarray([OBJECT_KEY, OBJECT_GOAL]))
    assert jnp.array_equal(
        maze.object_color, jnp.asarray([KEY_COLOR_RED, KEY_COLOR_YELLOW])
    )


def test_parse_ascii_maze_allows_optional_goal():
    no_goal = parse_ascii_maze(
        """
        #####
        #S..#
        #####
        """,
        require_goal=False,
    )

    assert no_goal.object_xy.shape == (0, 2)
    assert no_goal.object_type.shape == (0,)
    assert jnp.allclose(no_goal.goal_xy, jnp.asarray([0.0, 0.0]))


def test_parse_ascii_maze_multiple_goals():
    maze = parse_ascii_maze(
        """
        ######
        #SGG.#
        ######
        """
    )

    assert jnp.allclose(maze.goal_xy, jnp.asarray([2.5, 1.5]))
    assert jnp.array_equal(maze.object_type, jnp.asarray([OBJECT_GOAL, OBJECT_GOAL]))


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


def test_parse_ascii_maze_unlocked_door():
    maze = parse_ascii_maze(
        """
        #######
        #S"\\RG#
        #######
        """
    )

    assert int(maze.door_grid[1, 2]) == DOOR_UNLOCKED
    assert int(maze.door_grid[1, 3]) == DOOR_UNLOCKED_YELLOW
    assert int(maze.door_grid[1, 4]) == KEY_COLOR_RED


def test_parse_ascii_maze_colored_wall_symbols():
    maze = parse_ascii_maze(
        """
        11111
        1S.G1
        99999
        """
    )

    assert bool(maze.wall_grid[0, 0])
    assert bool(maze.wall_grid[2, 0])
    assert int(maze.color_grid[0, 0]) != int(maze.color_grid[2, 0])
    assert int(maze.color_grid[1, 1]) == 0


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
    assert batch.spawn_xy_options.shape == (2, 1, 2)
    assert jnp.array_equal(batch.spawn_count, jnp.asarray([1, 1], dtype=jnp.int32))
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
        #SrG#
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


def test_stack_mazes_pads_spawn_options():
    one_spawn = parse_ascii_maze(
        """
        #####
        #S.G#
        #####
        """
    )
    two_spawns = parse_ascii_maze(
        """
        #######
        #S.S.G#
        #######
        """
    )

    batch = stack_mazes([one_spawn, two_spawns])

    assert batch.spawn_xy_options.shape == (2, 2, 2)
    assert jnp.array_equal(batch.spawn_count, jnp.asarray([1, 2], dtype=jnp.int32))
    assert jnp.allclose(batch.spawn_xy_options[0, 0], jnp.asarray([1.5, 1.5]))
    assert jnp.allclose(batch.spawn_xy_options[0, 1], jnp.asarray([0.0, 0.0]))
    assert jnp.allclose(batch.spawn_xy_options[1, 1], jnp.asarray([3.5, 1.5]))


def test_key_corridor_requires_key_before_locked_goal():
    rows = MAZE_KEY_CORRIDOR.strip().splitlines()
    start = next(
        (row_idx, col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, char in enumerate(row)
        if char == "S"
    )
    key = next(
        (row_idx, col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, char in enumerate(row)
        if char == "r"
    )
    goal = next(
        (row_idx, col_idx)
        for row_idx, row in enumerate(rows)
        for col_idx, char in enumerate(row)
        if char == "G"
    )

    before_key = _reachable_cells(rows, start, {".", "S", "r", '"', "\\"})
    after_key = _reachable_cells(rows, key, {".", "S", "r", '"', "\\", "R", "G"})

    assert key in before_key
    assert goal not in before_key
    assert goal in after_key


def test_my_way_home_colored_variant_matches_colorless_topology():
    colored = parse_ascii_maze(MAZE_MY_WAY_HOME)
    colorless = parse_ascii_maze(MAZE_MY_WAY_HOME_COLORLESS)

    assert jnp.array_equal(colored.wall_grid, colorless.wall_grid)
    assert jnp.allclose(colored.spawn_xy_options, colorless.spawn_xy_options)
    assert jnp.allclose(colored.goal_xy, colorless.goal_xy)
    assert int(jnp.unique(colored.color_grid[colored.wall_grid]).shape[0]) > 1
    assert int(jnp.unique(colorless.color_grid[colorless.wall_grid]).shape[0]) == 1
