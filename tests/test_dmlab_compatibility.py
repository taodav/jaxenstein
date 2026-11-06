"""Contact and geometry checks against the pinned DeepMind Lab sources."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxenstein import ObjectPlacement, ObjectSpec, RayMazeEnv, make_jaxenstein_env
from jaxenstein.maps import MAPS_BY_NAME
from jaxenstein.maps.ascii import parse_ascii_maze
from jaxenstein.maps.dmlab import DMLAB_PICKUP_BOUNDS, dmlab_object_types
from jaxenstein.objects import compile_object_catalog
from jaxenstein.sprites import goal_sprite


@pytest.mark.parametrize("kind,reward", [("apple", 1), ("goal", 10)])
def test_dmlab_pickups_match_source_xy_contact_test(kind, reward):
    env = RayMazeEnv.from_ascii(
        "#########\n#S......#\n#.......#\n#.......#\n#.......#\n#########",
        objects=[ObjectPlacement(kind, (4.5, 3.5))],
        object_types=dmlab_object_types(),
        img_h=8,
        img_w=8,
    )
    _, state = env.reset(jax.random.key(1))
    # Box corners, asymmetric X limits, just inside/outside each edge, and
    # independence of heading. Distances above 0.35 catch the previous circle.
    offsets = np.array(
        [
            [-1, 0],
            [-1.0001, 0],
            [-0.9999, 0.7199],
            [-0.9999, -0.7199],
            [0.8799, 0.7199],
            [0.8799, -0.7199],
            [0.8801, 0],
            [0, 0.7201],
            [0, -0.7201],
            [0, 0.7199],
            [0, -0.7199],
            [0.5, 0.5],
            [0, 0],
        ],
        dtype=np.float32,
    )
    positions = offsets + np.asarray(env.maze.object_xy[0])
    source_offsets = (
        positions.astype(np.float64) - np.asarray(env.maze.object_xy[0])
    ) * 50
    expected = (
        (source_offsets[:, 0] >= -50)
        & (source_offsets[:, 0] <= 44)
        & (source_offsets[:, 1] >= -36)
        & (source_offsets[:, 1] <= 36)
    )

    @jax.jit
    def contact(positions, headings):
        return jax.vmap(lambda p, t: env.step(state.replace(pos=p, theta=t), 4)[2])(
            positions, headings
        )

    actual = contact(
        jnp.asarray(positions), jnp.linspace(0, 2 * jnp.pi, len(positions))
    )
    np.testing.assert_array_equal(actual, reward * expected)
    assert not env.object_catalog.has_radius_pickups


@pytest.mark.parametrize("env_id", ["dmlab-static-01", "dmlab-random-01"])
def test_factory_uses_bounds_for_both_apples_and_goals_and_keeps_reward_override(
    env_id,
):
    env = make_jaxenstein_env(env_id, apple_reward=2.5, goal_reward=7, img_h=8, img_w=8)
    catalog = env.object_catalog
    for name, reward in (("apple", 2.5), ("goal", 7)):
        idx = catalog.id(name)
        assert float(catalog.rewards[idx]) == reward
        np.testing.assert_allclose(catalog.pickup_mins[idx], DMLAB_PICKUP_BOUNDS[0])
        np.testing.assert_allclose(catalog.pickup_maxs[idx], DMLAB_PICKUP_BOUNDS[1])
    assert not catalog.has_radius_pickups


def test_mixed_circle_and_box_catalog_and_explicit_override():
    definitions = dmlab_object_types()
    definitions["small"] = replace(
        definitions["apple"], pickup_bounds=None, pickup_radius=0.25
    )
    env = RayMazeEnv.from_ascii(
        "#########\n#S......#\n#.......#\n#.......#\n#########",
        objects=[
            ObjectPlacement("apple", (3.5, 2.5)),
            ObjectPlacement("small", (6.5, 2.5)),
        ],
        object_types=definitions,
        img_h=8,
        img_w=8,
    )
    _, state = env.reset(jax.random.key(0))
    step = jax.jit(env.step)
    assert float(step(state.replace(pos=jnp.array([4.0, 2.5])), 4)[2]) == 1
    assert float(step(state.replace(pos=jnp.array([7.0, 2.5])), 4)[2]) == 0
    assert float(step(state.replace(pos=jnp.array([6.6, 2.5])), 4)[2]) == 1
    simple = make_jaxenstein_env(
        "simple", object_types={"unused": definitions["apple"]}
    )
    assert not simple.object_catalog.has_box_pickups


@pytest.mark.parametrize(
    "bounds", [((1, 0), (0, 1)), ((0, 0), (1, np.inf)), ((0,), (1,))]
)
def test_invalid_pickup_bounds_fail_at_construction(bounds):
    with pytest.raises(ValueError, match="pickup_bounds"):
        compile_object_catalog(
            ["x"], {"x": ObjectSpec(goal_sprite, pickup_bounds=bounds)}
        )


def test_random_maze_geometry_matches_every_official_lua_cell():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/dmlab_random_geometry.json").read_text()
    )
    for index, rows in fixture["entity_layers"].items():
        expected = np.array([[c == "*" for c in row] for row in rows])[::-1]
        maze = parse_ascii_maze(
            MAPS_BY_NAME[f"dmlab-random-{index}"], require_goal=False
        )
        np.testing.assert_array_equal(maze.wall_grid, expected)


@pytest.mark.parametrize(
    "env_id", ["simple", "dmlab-static-01", "dmlab-random-01", "health-gathering"]
)
def test_public_factory_can_construct_catalog_within_jit(env_id):
    make_jaxenstein_env(env_id, img_h=8, img_w=8)  # Import-time constants first.

    @jax.jit
    def reset(key):
        return make_jaxenstein_env(env_id, img_h=8, img_w=8).reset(key)

    obs, _ = reset(jax.random.key(0))
    assert obs.shape == (8, 8, 3)
