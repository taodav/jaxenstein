import jax
import jax.numpy as jnp

from jes.env import (
    ACTION_INTERACT,
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_TURN_LEFT,
    RayMazeEnv,
)
from jes.maps import MAZE_SIMPLE
from jes.objects import KEY_COLOR_BLUE, KEY_COLOR_RED, OBJECT_GOAL, OBJECT_KEY


def test_reset_returns_rgb_observation_and_spawn_state():
    env = RayMazeEnv.from_ascii([MAZE_SIMPLE])

    obs, state = env.reset(jax.random.key(0), 0)

    assert obs.shape == (64, 64, 3)
    assert obs.dtype == jnp.uint8
    assert jnp.allclose(state.pos, jnp.asarray([1.5, 1.5]))
    assert int(state.t) == 0
    assert not bool(state.done)
    assert jnp.array_equal(state.object_active, jnp.asarray([True]))
    assert jnp.array_equal(state.carried_keys, jnp.asarray([False, False, False, False]))


def test_forward_collision_noops_at_wall():
    env = RayMazeEnv.from_ascii([MAZE_SIMPLE])
    _, state = env.reset(jax.random.key(0), 0)
    wall_facing_state = state.replace(
        pos=jnp.asarray([1.05, 1.5], dtype=jnp.float32),
        theta=jnp.asarray(jnp.pi),
    )

    _, next_state, reward, done, _ = env.step(wall_facing_state, ACTION_MOVE_FORWARD)

    assert jnp.allclose(next_state.pos, wall_facing_state.pos)
    assert float(reward) == 0.0
    assert not bool(done)


def test_backward_movement_uses_collision():
    env = RayMazeEnv.from_ascii([MAZE_SIMPLE])
    _, state = env.reset(jax.random.key(0), 0)

    _, next_state, _, _, _ = env.step(state, ACTION_MOVE_BACKWARD)

    assert jnp.allclose(next_state.pos, jnp.asarray([1.35, 1.5], dtype=jnp.float32))


def test_sparse_goal_reward_and_done():
    env = RayMazeEnv.from_ascii([MAZE_SIMPLE])
    _, state = env.reset(jax.random.key(0), 0)
    near_goal = state.replace(pos=jnp.asarray([7.25, 1.5], dtype=jnp.float32))

    _, next_state, reward, done, info = env.step(near_goal, ACTION_MOVE_FORWARD)

    assert bool(done)
    assert bool(next_state.done)
    assert float(reward) == 1.0
    assert bool(info["reached_goal"])
    assert int(info["picked_object_type"]) == OBJECT_GOAL
    assert not bool(next_state.object_active[0])


def test_pickup_object_deactivates_without_done():
    env = RayMazeEnv.from_ascii(
        [
            """
            #####
            #SKG#
            #####
            """
        ]
    )
    _, state = env.reset(jax.random.key(0), 0)
    near_key = state.replace(pos=jnp.asarray([2.2, 1.5], dtype=jnp.float32))

    _, next_state, reward, done, info = env.step(near_key, ACTION_MOVE_FORWARD)

    assert float(reward) == 0.0
    assert not bool(done)
    assert not bool(next_state.object_active[0])
    assert bool(next_state.object_active[1])
    assert int(info["picked_object_type"]) == OBJECT_KEY
    assert int(info["picked_object_color"]) == KEY_COLOR_RED
    assert bool(next_state.carried_keys[KEY_COLOR_RED])


def test_interact_opens_matching_colored_door():
    env = RayMazeEnv.from_ascii(
        [
            """
            ######
            #SrRG#
            ######
            """
        ]
    )
    _, state = env.reset(jax.random.key(0), 0)
    near_key = state.replace(pos=jnp.asarray([2.2, 1.5], dtype=jnp.float32))

    _, key_state, _, _, _ = env.step(near_key, ACTION_MOVE_FORWARD)

    assert bool(key_state.carried_keys[KEY_COLOR_RED])

    at_closed_door = key_state.replace(pos=jnp.asarray([2.85, 1.5], dtype=jnp.float32))
    _, blocked_state, _, _, _ = env.step(at_closed_door, ACTION_MOVE_FORWARD)

    assert jnp.allclose(blocked_state.pos, at_closed_door.pos)
    assert not bool(blocked_state.door_open[1, 3])

    _, opened_state, reward, done, info = env.step(at_closed_door, ACTION_INTERACT)

    assert float(reward) == 0.0
    assert not bool(done)
    assert bool(info["opened_door"])
    assert int(info["opened_door_color"]) == KEY_COLOR_RED
    assert bool(opened_state.door_open[1, 3])

    _, passed_state, _, _, _ = env.step(opened_state, ACTION_MOVE_FORWARD)

    assert passed_state.pos[0] > opened_state.pos[0]


def test_interact_wrong_key_does_not_open_door():
    env = RayMazeEnv.from_ascii(
        [
            """
            ######
            #SbRG#
            ######
            """
        ]
    )
    _, state = env.reset(jax.random.key(0), 0)
    near_key = state.replace(pos=jnp.asarray([2.2, 1.5], dtype=jnp.float32))

    _, key_state, _, _, _ = env.step(near_key, ACTION_MOVE_FORWARD)
    at_closed_door = key_state.replace(pos=jnp.asarray([2.85, 1.5], dtype=jnp.float32))
    _, next_state, _, _, info = env.step(at_closed_door, ACTION_INTERACT)

    assert bool(key_state.carried_keys[KEY_COLOR_BLUE])
    assert not bool(key_state.carried_keys[KEY_COLOR_RED])
    assert not bool(info["opened_door"])
    assert not bool(next_state.door_open[1, 3])


def test_step_jit_and_vmap():
    env = RayMazeEnv.from_ascii([MAZE_SIMPLE])
    obs0, state = jax.jit(env.reset)(jax.random.key(0), jnp.asarray(0, dtype=jnp.int32))
    rendered0 = jax.jit(env.render)(state)

    assert obs0.shape == (64, 64, 3)
    assert jnp.array_equal(obs0, rendered0)

    obs1, state1, reward1, done1, info1 = jax.jit(env.step)(
        state, ACTION_MOVE_FORWARD
    )

    assert obs1.shape == (64, 64, 3)
    assert state1.pos.shape == (2,)
    assert reward1.shape == ()
    assert done1.shape == ()
    assert info1["cell_x"].shape == ()

    keys = jax.random.split(jax.random.key(0), 4)
    maze_ids = jnp.zeros((4,), dtype=jnp.int32)
    obs, states = jax.vmap(env.reset)(keys, maze_ids)

    assert obs.shape == (4, 64, 64, 3)
    rendered = jax.vmap(env.render)(states)
    assert jnp.array_equal(obs, rendered)

    actions = jnp.asarray(
        [ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, ACTION_MOVE_BACKWARD, ACTION_INTERACT],
        dtype=jnp.int32,
    )
    obs2, state2, reward2, done2, info2 = jax.vmap(env.step)(states, actions)

    assert obs2.shape == (4, 64, 64, 3)
    assert state2.pos.shape == (4, 2)
    assert state2.object_active.shape == (4, 1)
    assert state2.carried_keys.shape == (4, 4)
    assert state2.door_open.shape == (4, 5, 9)
    assert reward2.shape == (4,)
    assert done2.shape == (4,)
    assert info2["cell_x"].shape == (4,)
