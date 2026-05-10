import jax.numpy as jnp

from jes.ascii import parse_ascii_maze
from jes.dmlab import dmlab_decal_symbol, dmlab_map_to_ascii
from jes.maps import (
    DMLAB_NAV_MAZE_RANDOM_GOAL_01,
    DMLAB_NAV_MAZE_RANDOM_GOAL_02,
    DMLAB_NAV_MAZE_RANDOM_GOAL_03,
    DMLAB_NAV_MAZE_STATIC_01,
    DMLAB_NAV_MAZE_STATIC_02,
    DMLAB_NAV_MAZE_STATIC_03,
    MAPS_BY_NAME,
)
from jes.objects import COLORED_WALL_SYMBOLS, OBJECT_GOAL, WALL_SYMBOLS


SAMPLE_DMLAB_MAP = """
{
  "classname" "worldspawn"
  {
    ( 0 0 0 ) ( 0 32 0 ) ( 0 0 32 ) map/lab_games/lg_style_01_floor_orange
    ( 200 0 0 ) ( 200 0 32 ) ( 200 32 0 ) map/lab_games/lg_style_01_floor_orange
    ( 0 0 0 ) ( 0 0 32 ) ( 32 0 0 ) map/lab_games/lg_style_01_floor_orange
    ( 0 100 0 ) ( 32 100 0 ) ( 0 100 32 ) map/lab_games/lg_style_01_floor_orange
  }
  {
    ( 0 0 0 ) ( 0 32 0 ) ( 0 0 32 ) map/lab_games/lg_style_01_wall_green
    ( 1 0 0 ) ( 1 0 32 ) ( 1 32 0 ) map/lab_games/lg_style_01_wall_green
    ( 0 0 0 ) ( 0 0 32 ) ( 32 0 0 ) map/lab_games/lg_style_01_wall_green
    ( 0 100 0 ) ( 32 100 0 ) ( 0 100 32 ) map/lab_games/lg_style_01_wall_green
  }
  {
    patchDef2
    {
      decal/lab_games/dec_img_style01_001
      ( 3 3 0 0 0 )
      (
        ( ( 0 20 20 0 1 ) ( 0 20 50 0 0.5 ) ( 0 20 80 0 0 ) )
        ( ( 0 50 20 0.5 1 ) ( 0 50 50 0.5 0.5 ) ( 0 50 80 0.5 0 ) )
        ( ( 0 80 20 1 1 ) ( 0 80 50 1 0.5 ) ( 0 80 80 1 0 ) )
      )
    }
  }
}
{
  "classname" "info_player_start"
  "origin" "50 50 30"
}
{
  "classname" "goal"
  "origin" "150 50 20"
}
"""


def _isolated_positions(ascii_map: str, symbol: str) -> list[tuple[int, int]]:
    rows = ascii_map.strip().splitlines()
    isolated = []
    for row_idx, row in enumerate(rows):
        for col_idx, char in enumerate(row):
            if char != symbol:
                continue
            open_neighbors = 0
            for drow, dcol in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                next_row = row_idx + drow
                next_col = col_idx + dcol
                if not (0 <= next_row < len(rows)):
                    continue
                if not (0 <= next_col < len(rows[next_row])):
                    continue
                if rows[next_row][next_col] not in WALL_SYMBOLS:
                    open_neighbors += 1
            if open_neighbors == 0:
                isolated.append((row_idx, col_idx))
    return isolated


def test_dmlab_decal_symbol_uses_stable_global_texture_index():
    assert (
        dmlab_decal_symbol("decal/lab_games/dec_img_style01_001")
        == COLORED_WALL_SYMBOLS[0]
    )
    assert (
        dmlab_decal_symbol("decal/lab_games/dec_img_style04_020")
        == COLORED_WALL_SYMBOLS[79]
    )


def test_dmlab_map_to_ascii_converts_entities_and_decal_wall_color():
    ascii_map = dmlab_map_to_ascii(SAMPLE_DMLAB_MAP)
    maze = parse_ascii_maze(ascii_map)

    assert ascii_map.splitlines()[1][0] == COLORED_WALL_SYMBOLS[0]
    assert int(maze.spawn_count) == 1
    assert jnp.allclose(maze.spawn_xy, jnp.asarray([1.5, 1.5]))
    assert jnp.allclose(maze.goal_xy, jnp.asarray([3.5, 1.5]))


def test_generated_dmlab_nav_maze_maps_parse_and_are_registered():
    dmlab_maps = {
        "dmlab-nav-maze-static-01": DMLAB_NAV_MAZE_STATIC_01,
        "dmlab-nav-maze-static-02": DMLAB_NAV_MAZE_STATIC_02,
        "dmlab-nav-maze-static-03": DMLAB_NAV_MAZE_STATIC_03,
        "dmlab-nav-maze-random-goal-01": DMLAB_NAV_MAZE_RANDOM_GOAL_01,
        "dmlab-nav-maze-random-goal-02": DMLAB_NAV_MAZE_RANDOM_GOAL_02,
        "dmlab-nav-maze-random-goal-03": DMLAB_NAV_MAZE_RANDOM_GOAL_03,
    }

    for name, ascii_map in dmlab_maps.items():
        maze = parse_ascii_maze(ascii_map)

        assert MAPS_BY_NAME[name] == ascii_map
        assert int(maze.spawn_count) >= 1
        assert bool(jnp.any(maze.object_type == OBJECT_GOAL))
        assert int(jnp.unique(maze.color_grid[maze.wall_grid]).shape[0]) > 1
        assert _isolated_positions(ascii_map, "S") == []


def test_dmlab_static_01_has_source_goal_and_multiple_starts():
    maze = parse_ascii_maze(DMLAB_NAV_MAZE_STATIC_01)

    assert int(maze.spawn_count) == 43
    assert int(jnp.sum(maze.object_type == OBJECT_GOAL)) == 1
    assert jnp.allclose(maze.goal_xy, jnp.asarray([9.5, 1.5]))


def test_dmlab_random_goal_01_uses_apple_rewards_as_goals():
    maze = parse_ascii_maze(DMLAB_NAV_MAZE_RANDOM_GOAL_01)

    assert int(maze.spawn_count) == 1
    assert int(jnp.sum(maze.object_type == OBJECT_GOAL)) == 49
