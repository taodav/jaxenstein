import jax
import jax.numpy as jnp

from jes.env import (
    ACTION_INTERACT,
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_TURN_LEFT,
    RayMazeEnv,
)
from jes.maps import (
    DMLAB_NAV_MAZE_01_EPISODE_LENGTH_SECONDS,
    DMLAB_NAV_MAZE_01_MAX_STEPS,
    DMLAB_NAV_MAZE_02_EPISODE_LENGTH_SECONDS,
    DMLAB_NAV_MAZE_02_MAX_STEPS,
    DMLAB_NAV_MAZE_03_EPISODE_LENGTH_SECONDS,
    DMLAB_NAV_MAZE_03_MAX_STEPS,
    DMLAB_NAV_MAZE_HORIZON_FPS,
    DMLAB_NAV_MAZE_STATIC_01,
    DMLAB_NAV_MAZE_STATIC_02,
    DMLAB_NAV_MAZE_STATIC_03,
    MAP_DISCOUNT_GAMMAS_BY_NAME,
    MAP_EPISODE_HORIZONS_BY_NAME,
    MAP_REWARD_KWARGS_BY_NAME,
    MAPS_BY_NAME,
    MAZE_KEY_CORRIDOR,
    MAZE_MY_WAY_HOME,
    MAZE_SIMPLE,
    MINIGRID_KEY_CORRIDOR_S4R3_MAX_STEPS,
    VIZDOOM_MY_WAY_HOME_EPISODE_TIMEOUT,
)
from jes.objects import (
    DOOR_UNLOCKED,
    DOOR_UNLOCKED_YELLOW,
    KEY_COLOR_BLUE,
    KEY_COLOR_RED,
    OBJECT_GOAL,
    OBJECT_KEY,
)


def test_reset_returns_rgb_observation_and_spawn_state():
    env = RayMazeEnv.from_ascii([MAZE_SIMPLE])

    obs, state = env.reset(jax.random.key(0))

    assert obs.shape == (64, 64, 3)
    assert obs.dtype == jnp.uint8
    assert jnp.allclose(state.pos, jnp.asarray([1.5, 1.5]))
    assert jnp.allclose(state.goal_xy, jnp.asarray([7.5, 3.5]))
    assert int(state.t) == 0
    assert not bool(state.done)
    assert env.floor_pattern is True
    assert env.wall_height_scale == 1.35
    assert jnp.array_equal(state.object_active, jnp.asarray([True]))
    assert jnp.array_equal(
        state.carried_keys,
        jnp.asarray([False, False, False, False]),
    )


def test_custom_observation_resolution():
    env = RayMazeEnv.from_ascii([MAZE_SIMPLE], img_h=96, img_w=128)

    obs, state = env.reset(jax.random.key(0))
    rendered = env.render(state)

    assert env.observation_shape == (96, 128, 3)
    assert obs.shape == (96, 128, 3)
    assert rendered.shape == (96, 128, 3)
    assert obs.dtype == jnp.uint8


def test_custom_rewards_and_gamma():
    env = RayMazeEnv.from_ascii(
        [MAZE_SIMPLE],
        goal_reward=10.0,
        living_reward=-0.1,
        gamma=0.999,
    )
    _, state = env.reset(jax.random.key(0))

    _, _, reward, done, _ = env.step(state, ACTION_TURN_LEFT)
    near_goal = state.replace(pos=jnp.asarray([7.25, 3.5], dtype=jnp.float32))
    _, next_state, goal_reward, goal_done, _ = env.step(
        near_goal,
        ACTION_MOVE_FORWARD,
    )
    _, _, inactive_reward, _, _ = env.step(next_state, ACTION_TURN_LEFT)

    assert env.gamma == 0.999
    assert bool(jnp.isclose(reward, -0.1))
    assert not bool(done)
    assert bool(goal_done)
    assert bool(jnp.isclose(goal_reward, 9.9))
    assert bool(jnp.isclose(inactive_reward, 0.0))


def test_named_reward_and_gamma_specs_match_reference_tasks():
    assert MAP_REWARD_KWARGS_BY_NAME["my-way-home"] == {
        "goal_reward": 1.0,
        "living_reward": -0.0001,
    }
    assert MAP_REWARD_KWARGS_BY_NAME["dmlab-static-01"] == {
        "goal_reward": 10.0,
        "living_reward": 0.0,
    }
    assert MAP_REWARD_KWARGS_BY_NAME["dmlab-random-goal-03"] == {
        "goal_reward": 10.0,
        "living_reward": 0.0,
    }
    assert MAP_DISCOUNT_GAMMAS_BY_NAME["simple"] == 0.99
    assert MAP_DISCOUNT_GAMMAS_BY_NAME["key-door"] == 0.99
    assert MAP_DISCOUNT_GAMMAS_BY_NAME["my-way-home"] == 0.999
    assert MAP_DISCOUNT_GAMMAS_BY_NAME["dmlab-static-01"] == 0.999
    assert MAP_DISCOUNT_GAMMAS_BY_NAME["dmlab-static-02"] == 0.9995
    assert MAP_DISCOUNT_GAMMAS_BY_NAME["dmlab-static-03"] == 0.9999


def test_reset_samples_multiple_spawns_from_key():
    env = RayMazeEnv.from_ascii(
        [
            """
            #######
            #S.S.G#
            #######
            """
        ]
    )
    keys = jax.random.split(jax.random.key(0), 64)

    _, states = jax.vmap(env.reset)(keys)
    valid_spawns = jnp.asarray([[1.5, 1.5], [3.5, 1.5]], dtype=jnp.float32)
    is_valid_spawn = jnp.any(
        jnp.all(states.pos[:, None, :] == valid_spawns[None, :, :], axis=-1),
        axis=1,
    )
    unique_spawns = jnp.unique(states.pos, axis=0)

    assert bool(jnp.all(is_valid_spawn))
    assert int(unique_spawns.shape[0]) == 2


def test_reset_samples_map_when_vmapped():
    env = RayMazeEnv.from_ascii(
        [
            """
            #####
            #S.G#
            #####
            """,
            """
            #######
            #S...G#
            #######
            """,
        ]
    )
    keys = jax.random.split(jax.random.key(0), 64)

    _, states = jax.vmap(env.reset)(keys)

    assert states.maze_id.shape == (64,)
    assert int(jnp.unique(states.maze_id).shape[0]) == 2


def test_reset_samples_one_active_goal_from_multiple_candidates():
    env = RayMazeEnv.from_ascii(
        [
            """
            #########
            #S.G.G.#
            #########
            """
        ]
    )
    keys = jax.random.split(jax.random.key(0), 64)

    _, states = jax.vmap(env.reset)(keys)
    object_type = env.maze_batch.object_type[0]
    active_goals = states.object_active & (object_type[None, :] == OBJECT_GOAL)
    valid_goals = jnp.asarray([[3.5, 1.5], [5.5, 1.5]], dtype=jnp.float32)
    is_valid_goal = jnp.any(
        jnp.all(states.goal_xy[:, None, :] == valid_goals[None, :, :], axis=-1),
        axis=1,
    )

    assert bool(jnp.all(jnp.sum(active_goals, axis=1) == 1))
    assert bool(jnp.all(is_valid_goal))
    assert int(jnp.unique(states.goal_xy, axis=0).shape[0]) == 2


def test_inactive_goal_candidate_does_not_end_episode():
    env = RayMazeEnv.from_ascii(
        [
            """
            #########
            #S.G.G.#
            #########
            """
        ]
    )
    _, state = env.reset(jax.random.key(0))
    object_type = env.maze_batch.object_type[0]
    inactive_goal = (object_type == OBJECT_GOAL) & ~state.object_active
    inactive_goal_xy = jnp.sum(
        jnp.where(
            inactive_goal[:, None],
            env.maze_batch.object_xy[0],
            jnp.asarray(0.0, dtype=jnp.float32),
        ),
        axis=0,
    )

    _, next_state, reward, done, info = env.step(
        state.replace(pos=inactive_goal_xy),
        ACTION_TURN_LEFT,
    )

    assert float(reward) == 0.0
    assert not bool(done)
    assert not bool(next_state.done)
    assert not bool(info["reached_goal"])


def test_episode_horizons_can_vary_by_maze():
    env = RayMazeEnv.from_ascii(
        [
            """
            #####
            #S.G#
            #####
            """,
            """
            #####
            #S.G#
            #####
            """,
        ],
        episode_horizons=[1, 2],
    )
    _, state = env.reset(jax.random.key(0))
    short_state = state.replace(maze_id=jnp.asarray(0, dtype=jnp.int32))
    long_state = state.replace(maze_id=jnp.asarray(1, dtype=jnp.int32))

    _, short_state, _, short_done, _ = env.step(short_state, ACTION_TURN_LEFT)
    _, long_state, _, long_done, _ = env.step(long_state, ACTION_TURN_LEFT)
    _, long_state, _, long_done2, _ = env.step(long_state, ACTION_TURN_LEFT)

    assert bool(short_done)
    assert bool(short_state.done)
    assert not bool(long_done)
    assert bool(long_done2)
    assert bool(long_state.done)


def test_my_way_home_horizon_matches_vizdoom_timeout():
    env = RayMazeEnv.from_ascii(
        [MAZE_MY_WAY_HOME],
        episode_horizons=[
            MAP_EPISODE_HORIZONS_BY_NAME["my-way-home"],
        ],
    )

    assert VIZDOOM_MY_WAY_HOME_EPISODE_TIMEOUT == 2100
    assert int(env.episode_horizons[0]) == VIZDOOM_MY_WAY_HOME_EPISODE_TIMEOUT


def test_key_corridor_is_registered_with_minigrid_horizon():
    env = RayMazeEnv.from_ascii(
        [MAPS_BY_NAME["key-corridor"]],
        episode_horizons=[MAP_EPISODE_HORIZONS_BY_NAME["key-corridor"]],
    )
    door_grid = env.maze_batch.door_grids[0]

    assert MAPS_BY_NAME["key-corridor"] == MAZE_KEY_CORRIDOR
    assert MINIGRID_KEY_CORRIDOR_S4R3_MAX_STEPS == 480
    assert int(env.episode_horizons[0]) == MINIGRID_KEY_CORRIDOR_S4R3_MAX_STEPS
    assert int(jnp.sum(door_grid < 0)) == 5
    assert int(jnp.sum(door_grid == DOOR_UNLOCKED)) == 3
    assert int(jnp.sum(door_grid == DOOR_UNLOCKED_YELLOW)) == 2
    assert int(jnp.sum(door_grid == KEY_COLOR_RED)) == 1
    assert int(jnp.sum(env.maze_batch.object_type[0] == OBJECT_KEY)) == 1
    assert int(jnp.sum(env.maze_batch.object_type[0] == OBJECT_GOAL)) == 1


def test_dmlab_nav_maze_horizons_match_source_frame_steps():
    env = RayMazeEnv.from_ascii(
        [
            DMLAB_NAV_MAZE_STATIC_01,
            DMLAB_NAV_MAZE_STATIC_02,
            DMLAB_NAV_MAZE_STATIC_03,
        ],
        episode_horizons=[
            MAP_EPISODE_HORIZONS_BY_NAME["dmlab-static-01"],
            MAP_EPISODE_HORIZONS_BY_NAME["dmlab-static-02"],
            MAP_EPISODE_HORIZONS_BY_NAME["dmlab-static-03"],
        ],
    )

    assert DMLAB_NAV_MAZE_01_EPISODE_LENGTH_SECONDS == 60
    assert DMLAB_NAV_MAZE_02_EPISODE_LENGTH_SECONDS == 150
    assert DMLAB_NAV_MAZE_03_EPISODE_LENGTH_SECONDS == 300
    assert DMLAB_NAV_MAZE_HORIZON_FPS == 30
    assert DMLAB_NAV_MAZE_01_MAX_STEPS == 1800
    assert DMLAB_NAV_MAZE_02_MAX_STEPS == 4500
    assert DMLAB_NAV_MAZE_03_MAX_STEPS == 9000
    assert env.episode_horizons.tolist() == [1800, 4500, 9000]
    assert MAP_EPISODE_HORIZONS_BY_NAME["dmlab-random-goal-01"] == 1800
    assert MAP_EPISODE_HORIZONS_BY_NAME["dmlab-random-goal-02"] == 4500
    assert MAP_EPISODE_HORIZONS_BY_NAME["dmlab-random-goal-03"] == 9000


def test_forward_collision_noops_at_wall():
    env = RayMazeEnv.from_ascii([MAZE_SIMPLE])
    _, state = env.reset(jax.random.key(0))
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
    _, state = env.reset(jax.random.key(0))

    _, next_state, _, _, _ = env.step(state, ACTION_MOVE_BACKWARD)

    assert jnp.allclose(next_state.pos, jnp.asarray([1.35, 1.5], dtype=jnp.float32))


def test_sparse_goal_reward_and_done():
    env = RayMazeEnv.from_ascii([MAZE_SIMPLE])
    _, state = env.reset(jax.random.key(0))
    near_goal = state.replace(pos=jnp.asarray([7.25, 3.5], dtype=jnp.float32))

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
    _, state = env.reset(jax.random.key(0))
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
    _, state = env.reset(jax.random.key(0))
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


def test_interact_opens_unlocked_door_without_key():
    env = RayMazeEnv.from_ascii(
        [
            """
            ######
            #S".G#
            ######
            """
        ]
    )
    _, state = env.reset(jax.random.key(0))

    _, opened_state, _, _, info = env.step(state, ACTION_INTERACT)

    assert bool(info["opened_door"])
    assert int(info["opened_door_color"]) == DOOR_UNLOCKED
    assert bool(opened_state.door_open[1, 2])
    assert not bool(jnp.any(opened_state.carried_keys))


def test_key_corridor_locked_door_requires_matching_key():
    env = RayMazeEnv.from_ascii([MAZE_KEY_CORRIDOR])
    _, state = env.reset(jax.random.key(0))
    near_locked_door = state.replace(pos=jnp.asarray([5.2, 4.5], dtype=jnp.float32))

    _, blocked_state, _, _, blocked_info = env.step(near_locked_door, ACTION_INTERACT)

    carried_red_key = state.carried_keys.at[KEY_COLOR_RED].set(True)
    _, opened_state, _, _, opened_info = env.step(
        near_locked_door.replace(carried_keys=carried_red_key),
        ACTION_INTERACT,
    )

    assert not bool(blocked_info["opened_door"])
    assert not bool(blocked_state.door_open[4, 6])
    assert bool(opened_info["opened_door"])
    assert int(opened_info["opened_door_color"]) == KEY_COLOR_RED
    assert bool(opened_state.door_open[4, 6])


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
    _, state = env.reset(jax.random.key(0))
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
    obs0, state = jax.jit(env.reset)(jax.random.key(0))
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
    obs, states = jax.vmap(env.reset)(keys)

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
