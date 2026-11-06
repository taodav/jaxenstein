import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxenstein import make_jaxenstein_env
from jaxenstein.maps import (
    DMLAB_NAV_MAZE_RANDOM_GOAL_01,
    DMLAB_NAV_MAZE_RANDOM_GOAL_02,
    DMLAB_NAV_MAZE_RANDOM_GOAL_03,
    DMLAB_NAV_MAZE_STATIC_01,
    DMLAB_NAV_MAZE_STATIC_02,
    DMLAB_NAV_MAZE_STATIC_03,
    MAP_RENDER_KWARGS_BY_NAME,
    MAPS_BY_NAME,
)
from jaxenstein.maps.ascii import parse_ascii_maze
from jaxenstein.maps.dmlab import (
    dmlab_decal_symbol,
    dmlab_map_to_ascii,
    dmlab_map_to_spec,
)
from jaxenstein.maps.dmlab_entities import DMLAB_OBJECTS_BY_NAME
from jaxenstein.objects import (
    COLORED_WALL_SYMBOLS,
    OBJECT_GOAL,
    WALL_SYMBOLS,
)

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


def test_four_connected_floor_tiles_do_not_create_a_pillar():
    room = (
        SAMPLE_DMLAB_MAP.replace("0 100 0", "0 200 0")
        .replace("0 100 32", "0 200 32")
        .replace("32 100 0", "32 200 0")
    )
    maze = parse_ascii_maze(dmlab_map_to_ascii(room))
    assert not bool(maze.wall_grid[2, 2])


def test_generated_dmlab_nav_maze_maps_parse_and_are_registered():
    dmlab_maps = {
        "dmlab-static-01": DMLAB_NAV_MAZE_STATIC_01,
        "dmlab-static-02": DMLAB_NAV_MAZE_STATIC_02,
        "dmlab-static-03": DMLAB_NAV_MAZE_STATIC_03,
        "dmlab-random-01": DMLAB_NAV_MAZE_RANDOM_GOAL_01,
        "dmlab-random-02": DMLAB_NAV_MAZE_RANDOM_GOAL_02,
        "dmlab-random-03": DMLAB_NAV_MAZE_RANDOM_GOAL_03,
    }

    for name, ascii_map in dmlab_maps.items():
        maze = parse_ascii_maze(
            ascii_map,
            require_goal=name.startswith("dmlab-static"),
            objects=DMLAB_OBJECTS_BY_NAME[name],
        )

        assert MAPS_BY_NAME[name] == ascii_map
        assert int(maze.spawn_count) >= 1
        assert ("goal" if "static" in name else "apple") in maze.object_names
        assert int(jnp.unique(maze.color_grid[maze.wall_grid]).shape[0]) > 1
        assert _isolated_positions(ascii_map, "S") == []


def test_dmlab_render_kwargs_match_maze_color_schemes():
    floor_by_index = {
        "01": [182, 154, 46],
        "02": [68, 132, 188],
        "03": [32, 122, 86],
    }

    for index, floor_rgb in floor_by_index.items():
        static = MAP_RENDER_KWARGS_BY_NAME[f"dmlab-static-{index}"]
        random_goal = MAP_RENDER_KWARGS_BY_NAME[f"dmlab-random-{index}"]

        assert static == random_goal
        assert static["floor_rgb"] == floor_rgb
        assert static["floor_checker_dark_rgb"] != static["floor_checker_light_rgb"]
        assert static["color_palette"][1] != [150, 150, 150]


def test_dmlab_static_01_has_source_goal_and_multiple_starts():
    maze = parse_ascii_maze(DMLAB_NAV_MAZE_STATIC_01)

    assert int(maze.spawn_count) == 43
    assert int(jnp.sum(maze.object_type == OBJECT_GOAL)) == 1
    assert jnp.allclose(maze.goal_xy, jnp.asarray([9.5, 1.5]))


def test_dmlab_random_goal_01_keeps_all_source_candidates():
    maze = parse_ascii_maze(
        DMLAB_NAV_MAZE_RANDOM_GOAL_01,
        require_goal=False,
        objects=DMLAB_OBJECTS_BY_NAME["dmlab-random-01"],
    )

    assert int(maze.spawn_count) == 1
    assert maze.object_names.count("apple") == 50


def test_dmlab_random_goal_env_activates_one_goal_per_episode():
    env = make_jaxenstein_env("dmlab-random-01", img_h=8, img_w=8)

    _, state = env.reset(jax.random.key(0))
    active_goals = state.object_active & (env._object_types(state) == OBJECT_GOAL)

    assert int(jnp.sum(active_goals)) == 1


def test_converter_preserves_apple_overlapping_spawn_and_standalone_apple():
    source = (
        SAMPLE_DMLAB_MAP
        + """
{
  "classname" "apple_reward"
  "origin" "50 50 20"
}
{
  "classname" "apple_reward"
  "origin" "250 50 20"
}
"""
    )
    converted = dmlab_map_to_spec(source)
    maze = parse_ascii_maze(converted.ascii_map, objects=converted.objects)
    assert converted.ascii_map.splitlines()[1][1] == "S"
    assert converted.ascii_map.splitlines()[1][5] == "."
    assert int(maze.spawn_count) == 1
    assert maze.object_names.count("apple") == 2
    assert int(jnp.sum(maze.object_type == OBJECT_GOAL)) == 1


SOURCE_ENTITIES = json.loads(
    (Path(__file__).parent / "fixtures" / "dmlab_entities.json").read_text()
)["maps"]


@pytest.mark.parametrize("env_id", SOURCE_ENTITIES)
def test_registered_maps_preserve_official_entities(env_id):
    maze = parse_ascii_maze(
        MAPS_BY_NAME[env_id], require_goal=False, objects=DMLAB_OBJECTS_BY_NAME[env_id]
    )
    source = SOURCE_ENTITIES[env_id]["locations"]

    def source_xy(kind):
        return sorted(tuple(2 * (v // 100) + 1.5 for v in xy) for xy in source[kind])

    def actual_xy(kind):
        return sorted(
            map(
                tuple, np.asarray(maze.object_xy[np.asarray(maze.object_names) == kind])
            )
        )

    assert sorted(map(tuple, np.asarray(maze.spawn_xy_options))) == source_xy(
        "info_player_start"
    )
    assert actual_xy("apple") == source_xy("apple_reward")
    assert actual_xy("goal") == source_xy("goal")


@pytest.mark.parametrize(
    "index,candidates,fruits", [("01", 50, 5), ("02", 135, 13), ("03", 260, 26)]
)
def test_random_goal_reset_matches_official_sampling_constraints(
    index, candidates, fruits
):
    env = make_jaxenstein_env(f"dmlab-random-{index}", img_h=8, img_w=8)
    keys = jax.random.split(jax.random.key(17), 64)
    states = jax.jit(jax.vmap(lambda key: env.reset(key)[1]))(keys)
    types = jax.vmap(env.object_type_ids)(states)
    assert states.object_active.shape == (64, candidates)
    assert np.all(
        np.sum(
            np.asarray(states.object_active & (types == env.object_catalog.id("goal"))),
            axis=1,
        )
        == 1
    )
    assert np.all(
        np.sum(
            np.asarray(
                states.object_active & (types == env.object_catalog.id("apple"))
            ),
            axis=1,
        )
        == fruits
    )
    assert len(np.unique(np.asarray(states.goal_xy), axis=0)) > 10
    assert len(np.unique(np.asarray(states.pos), axis=0)) > 10
    assert len(np.unique(np.asarray(states.theta))) == 64
    assert np.all(
        (np.asarray(states.theta) >= 0) & (np.asarray(states.theta) < 2 * np.pi)
    )
    expected = SOURCE_ENTITIES[f"dmlab-random-{index}"]["spawn_eligible_sha256"]
    assert (
        hashlib.sha256(np.asarray(env._spawn_eligible).tobytes()).hexdigest()
        == expected
    )
    candidate_xy = np.asarray(env.maze.object_xy)
    for pos, goal in zip(np.asarray(states.pos), np.asarray(states.goal_xy)):
        spawn_idx = np.flatnonzero(np.all(candidate_xy == pos, axis=1))[0]
        goal_idx = np.flatnonzero(np.all(candidate_xy == goal, axis=1))[0]
        assert bool(env._spawn_eligible[goal_idx, spawn_idx])
    repeated = jax.jit(lambda key: env.reset(key)[1])(keys[0])
    assert np.array_equal(repeated.object_active, states.object_active[0])
    assert np.array_equal(repeated.pos, states.pos[0])
    assert float(repeated.theta) == float(states.theta[0])


@pytest.mark.parametrize("index,apple_count", [("01", 5), ("02", 13), ("03", 26)])
def test_static_resets_activate_all_apples_and_randomize_heading(index, apple_count):
    env = make_jaxenstein_env(f"dmlab-static-{index}", img_h=8, img_w=8)
    keys = jax.random.split(jax.random.key(3), 32)
    states = jax.jit(jax.vmap(lambda key: env.reset(key)[1]))(keys)
    assert np.all(
        np.sum(
            np.asarray(
                states.object_active
                & (env._initial_type_ids == env.object_catalog.id("apple"))
            ),
            axis=1,
        )
        == apple_count
    )
    assert len(np.unique(np.asarray(states.theta))) == 32
    fixed = make_jaxenstein_env(
        f"dmlab-static-{index}", randomize_spawn_heading=False, img_h=8, img_w=8
    )
    _, fixed_state = fixed.reset(keys[0])
    assert float(fixed_state.theta) == 0.0
    assert np.array_equal(fixed_state.pos, states.pos[0])
