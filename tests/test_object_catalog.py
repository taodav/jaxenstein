import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxenstein import (
    ACTION_INTERACT,
    ObjectPlacement,
    ObjectSpec,
    RayMazeEnv,
    make_jaxenstein_env,
)
from jaxenstein.maps import dmlab


def square(color):
    image = np.empty((8, 8, 4), dtype=np.uint8)
    image[..., :3] = color
    image[..., 3] = 255
    return image


def forbidden_sprite():
    raise AssertionError("An unused sprite builder was called")


def test_unused_definitions_are_not_materialized_or_compiled():
    env = make_jaxenstein_env(
        "simple",
        img_h=8,
        img_w=8,
        object_types={
            "apple": ObjectSpec(forbidden_sprite),
            "unused": ObjectSpec(forbidden_sprite),
        },
    )
    assert env.object_catalog.names == ("goal",)
    assert env.object_catalog.sprites.shape[0] == 1
    obs, state = jax.jit(env.reset)(jax.random.key(0))
    jax.jit(env.step)(state, ACTION_INTERACT)
    assert obs.shape == (8, 8, 3)
    assert not hasattr(env, "apple_reward")
    assert env.object_catalog.rewards.shape == (1,)


@pytest.mark.parametrize(
    "env_id,kwargs",
    [
        ("simple", {}),
        ("my-way-home", {}),
        ("key-corridor", {}),
        ("health-gathering", {"max_medkits": 2, "initial_medkit_count": 1}),
        ("dmlab-static-01", {"objects": []}),
        ("dmlab-random-01", {"scattered_reward_density": 0.0}),
    ],
)
def test_instances_without_apples_never_build_apple_assets(monkeypatch, env_id, kwargs):
    monkeypatch.setattr(dmlab, "apple_sprite", forbidden_sprite)
    env = make_jaxenstein_env(env_id, img_h=8, img_w=8, **kwargs)
    assert "apple" not in env.object_catalog.names
    _, state = jax.jit(env.reset)(jax.random.key(0))
    _, _, _, _, info = jax.jit(env.step)(
        state, ACTION_INTERACT if env.num_actions == 5 else 0
    )
    assert "picked_apples" not in info


def test_custom_pickups_share_positions_and_apply_independent_properties():
    specs = {
        "coin": ObjectSpec(lambda: square((0, 200, 250)), reward=2.0),
        "checkpoint": ObjectSpec(
            lambda: square((200, 0, 250)), reward=0.5, consume=False
        ),
        "hazard": ObjectSpec(lambda: square((250, 50, 0)), reward=-3.0, terminal=True),
    }
    env = RayMazeEnv.from_ascii(
        "#######\n#S....#\n#######",
        objects=[
            ObjectPlacement("coin", (1.5, 1.5)),
            ObjectPlacement("checkpoint", (1.5, 1.5)),
            ObjectPlacement("hazard", (4.5, 1.5)),
        ],
        object_types=specs,
        img_h=8,
        img_w=8,
    )
    _, state = env.reset(jax.random.key(0))
    step = jax.jit(env.step)
    _, state, reward, done, info = step(state, ACTION_INTERACT)
    assert float(reward) == 2.5 and not bool(done)
    assert np.array_equal(info["picked_objects"], [True, True, False])
    assert np.array_equal(state.object_active, [False, True, True])
    _, state, reward, _, _ = step(state, ACTION_INTERACT)
    assert float(reward) == 0.5
    _, state, reward, done, info = step(
        state.replace(pos=jnp.array([4.5, 1.5])), ACTION_INTERACT
    )
    assert float(reward) == -3 and bool(done)
    assert not bool(info["reached_goal"])
    assert int(info["picked_type_id"]) == env.object_catalog.id("hazard")
    assert float(step(state, ACTION_INTERACT)[2]) == 0.0


def test_environment_local_symbols_and_custom_goal():
    env = RayMazeEnv.from_ascii(
        "#####\n#SD.#\n#####",
        object_symbols={"D": "exit"},
        object_types={
            "exit": ObjectSpec(
                lambda: square((0, 255, 255)),
                reward=7,
                terminal=True,
                is_goal=True,
                pickup_radius=0.1,
            )
        },
        img_h=8,
        img_w=8,
    )
    _, state = env.reset(jax.random.key(0))
    assert np.array_equal(state.goal_xy, [2.5, 1.5])
    # The custom radius is not overridden by the legacy goal_radius parameter.
    _, _, reward, done, _ = env.step(
        state.replace(pos=jnp.array([2.3, 1.5])), ACTION_INTERACT
    )
    assert float(reward) == 0 and not bool(done)
    _, _, reward, done, info = env.step(
        state.replace(pos=state.goal_xy), ACTION_INTERACT
    )
    assert float(reward) == 7 and bool(done) and bool(info["reached_goal"])


def test_zero_objects_support_jit_vmap_and_time_limit():
    env = RayMazeEnv.from_ascii(
        "#####\n#S..#\n#####", episode_horizon=1, img_h=8, img_w=8
    )
    assert env.object_catalog.names == ()
    assert env.object_catalog.sprites.shape[0] == 0
    keys = jax.random.split(jax.random.key(3), 3)
    obs, states = jax.jit(jax.vmap(env.reset))(keys)
    assert states.object_active.shape == (3, 0)
    _, states, rewards, dones, info = jax.jit(jax.vmap(env.step))(
        states, jnp.zeros(3, dtype=jnp.int32)
    )
    assert np.all(dones) and np.all(np.asarray(rewards) == 0)
    assert info["picked_objects"].shape == (3, 0)
    assert np.all(np.asarray(info["picked_type_id"]) == -1)
    assert obs.shape == (3, 8, 8, 3)


def test_rgba_catalog_can_render_new_objects_without_type_branches():
    red, blue = square((240, 0, 0)), square((0, 0, 240))
    red[2:6, 2:6, 3] = 0  # The farther object must show through the cutout.
    env = RayMazeEnv.from_ascii(
        "########\n#S.....#\n########",
        objects=[
            ObjectPlacement("red", (2.5, 1.5)),
            ObjectPlacement("blue", (3.5, 1.5)),
        ],
        object_types={"red": ObjectSpec(red), "blue": ObjectSpec(blue)},
        img_h=64,
        img_w=64,
    )
    obs, state = jax.jit(env.reset)(jax.random.key(0))
    assert np.any(np.all(np.asarray(obs) == [240, 0, 0], axis=-1))
    assert np.any(np.all(np.asarray(obs) == [0, 0, 240], axis=-1))
    hidden = env.render(state.replace(object_active=jnp.array([False, False])))
    assert not np.array_equal(obs, hidden)


@pytest.mark.parametrize(
    "objects,definitions,message",
    [
        ([ObjectPlacement("unknown", (1.5, 1.5))], {}, "Unknown object type"),
        (
            [ObjectPlacement("x", (0.5, 0.5))],
            {"x": ObjectSpec(lambda: square((0, 0, 0)))},
            "open floor",
        ),
        (
            [ObjectPlacement("x", (1.5, 1.5))],
            {"x": ObjectSpec(np.zeros((4, 4, 3), dtype=np.uint8))},
            "RGBA",
        ),
    ],
)
def test_invalid_object_configs_fail_before_compilation(objects, definitions, message):
    with pytest.raises(ValueError, match=message):
        RayMazeEnv.from_ascii(
            "#####\n#S..#\n#####", objects=objects, object_types=definitions
        )
