import jax.numpy as jnp

from jaxenstein.maps.ascii import parse_ascii_maze
from jaxenstein.maps import MAZE_KEY_CORRIDOR, MAZE_MY_WAY_HOME, MAZE_MY_WAY_HOME_COLORLESS
from jaxenstein.objects import (
    DOOR_UNLOCKED,
    DOOR_UNLOCKED_YELLOW,
    KEY_COLOR_BLUE,
    KEY_COLOR_RED,
    KEY_COLOR_YELLOW,
    OBJECT_GOAL,
    OBJECT_KEY,
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
