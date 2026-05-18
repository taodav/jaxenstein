import jax
import jax.numpy as jnp

from jes.env import ACTION_MOVE_FORWARD, ACTION_TURN_LEFT, RayMazeEnv
from jes.health_gathering import (
    HEALTH_GATHERING_NUM_ACTIONS,
    HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB,
    HEALTH_GATHERING_FLOOR_RGB,
    HealthGatheringEnv,
    HealthGatheringParams,
)
from jes.maps import (
    HEALTH_GATHERING_EPISODE_HORIZONS_BY_NAME,
    HEALTH_GATHERING_MAPS_BY_NAME,
    HEALTH_GATHERING_RENDER_KWARGS_BY_NAME,
    HEALTH_GATHERING_REWARD_KWARGS_BY_NAME,
    MAZE_HEALTH_GATHERING,
    MAZE_SIMPLE,
    HEALTH_GATHERING_ACID_DAMAGE,
    HEALTH_GATHERING_ACID_DAMAGE_INTERVAL,
    HEALTH_GATHERING_DEATH_PENALTY,
    HEALTH_GATHERING_EPISODE_TIMEOUT,
    HEALTH_GATHERING_INITIAL_HEALTH,
    HEALTH_GATHERING_INITIAL_MEDKITS,
    HEALTH_GATHERING_LIVING_REWARD,
    HEALTH_GATHERING_MAX_MEDKITS,
    HEALTH_GATHERING_MEDKIT_HEAL,
    HEALTH_GATHERING_MEDKIT_SPAWN_INTERVAL,
    HEALTH_GATHERING_WALL_RGB,
)


def test_health_gathering_reset_uses_separate_health_state_and_defaults():
    env = HealthGatheringEnv.from_ascii([MAZE_HEALTH_GATHERING])
    nav_env = RayMazeEnv.from_ascii([MAZE_SIMPLE])

    obs, state = env.reset(jax.random.key(0))
    _, nav_state = nav_env.reset(jax.random.key(0))

    assert obs.shape == (64, 64, 3)
    assert obs.dtype == jnp.uint8
    assert env.num_actions == HEALTH_GATHERING_NUM_ACTIONS
    assert int(env.episode_horizons[0]) == HEALTH_GATHERING_EPISODE_TIMEOUT
    assert float(state.health) == HEALTH_GATHERING_INITIAL_HEALTH
    assert int(jnp.sum(state.medkit_active)) == HEALTH_GATHERING_INITIAL_MEDKITS
    assert state.medkit_xy.shape == (HEALTH_GATHERING_MAX_MEDKITS, 2)
    assert not hasattr(nav_state, "health")
    assert not hasattr(state, "carried_keys")

    assert HEALTH_GATHERING_MAPS_BY_NAME["health-gathering"] == MAZE_HEALTH_GATHERING
    assert HEALTH_GATHERING_EPISODE_HORIZONS_BY_NAME["health-gathering"] == 2100
    assert HEALTH_GATHERING_REWARD_KWARGS_BY_NAME["health-gathering"] == {
        "living_reward": HEALTH_GATHERING_LIVING_REWARD,
        "death_penalty": HEALTH_GATHERING_DEATH_PENALTY,
    }
    assert HEALTH_GATHERING_RENDER_KWARGS_BY_NAME["health-gathering"][
        "floor_rgb"
    ] == HEALTH_GATHERING_FLOOR_RGB.astype(int).tolist()
    assert HEALTH_GATHERING_RENDER_KWARGS_BY_NAME["health-gathering"][
        "floor_checker_light_rgb"
    ] == HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB.astype(int).tolist()
    wall_rgb = HEALTH_GATHERING_RENDER_KWARGS_BY_NAME["health-gathering"][
        "color_palette"
    ][1]
    assert wall_rgb == HEALTH_GATHERING_WALL_RGB
    assert wall_rgb[0] == wall_rgb[1]
    assert wall_rgb[1] >= wall_rgb[2]


def test_health_gathering_medkit_pickup_heals_and_deactivates():
    env = HealthGatheringEnv.from_ascii([MAZE_HEALTH_GATHERING])
    _, state = env.reset(jax.random.key(0))
    only_first_medkit_active = jnp.arange(env.max_medkits) == 0
    damaged = state.replace(
        pos=state.medkit_xy[0],
        health=jnp.asarray(50.0, dtype=jnp.float32),
        medkit_active=only_first_medkit_active,
    )

    _, next_state, reward, done, info = env.step(damaged, ACTION_TURN_LEFT)

    assert bool(info["picked_medkit"])
    assert not bool(next_state.medkit_active[0])
    assert bool(
        jnp.isclose(
            next_state.health,
            50.0 + HEALTH_GATHERING_MEDKIT_HEAL,
        )
    )
    assert float(reward) == HEALTH_GATHERING_LIVING_REWARD
    assert not bool(done)


def test_health_gathering_acid_damage_can_kill_with_death_penalty():
    params = HealthGatheringParams(
        acid_damage=jnp.asarray(8.0, dtype=jnp.float32),
        acid_damage_interval=jnp.asarray(1, dtype=jnp.int32),
    )
    env = HealthGatheringEnv.from_ascii(
        [MAZE_HEALTH_GATHERING],
        params=params,
        initial_medkit_count=0,
        max_medkits=4,
    )
    _, state = env.reset(jax.random.key(0))
    fragile = state.replace(health=jnp.asarray(8.0, dtype=jnp.float32))

    _, next_state, reward, done, info = env.step(fragile, ACTION_TURN_LEFT)

    assert bool(done)
    assert bool(next_state.done)
    assert bool(info["died"])
    assert float(next_state.health) == 0.0
    assert float(info["acid_damage"]) == 8.0
    assert float(reward) == -HEALTH_GATHERING_DEATH_PENALTY


def test_health_gathering_spawns_new_medkit_every_interval():
    params = HealthGatheringParams(acid_damage=jnp.asarray(0.0, dtype=jnp.float32))
    env = HealthGatheringEnv.from_ascii(
        [MAZE_HEALTH_GATHERING],
        params=params,
        initial_medkit_count=0,
        medkit_spawn_interval=2,
        max_medkits=4,
    )
    _, state = env.reset(jax.random.key(0))

    _, state, _, _, info1 = env.step(state, ACTION_TURN_LEFT)
    _, state, _, _, info2 = env.step(state, ACTION_TURN_LEFT)

    assert not bool(info1["spawned_medkit"])
    assert bool(info2["spawned_medkit"])
    assert int(jnp.sum(state.medkit_active)) == 1
    assert int(state.next_medkit_slot) == 1


def test_health_gathering_movement_collision_and_periodic_damage_defaults():
    env = HealthGatheringEnv.from_ascii(
        [MAZE_HEALTH_GATHERING],
        initial_medkit_count=0,
        max_medkits=4,
    )
    _, state = env.reset(jax.random.key(0))
    wall_facing_state = state.replace(
        pos=jnp.asarray([1.05, 1.5], dtype=jnp.float32),
        theta=jnp.asarray(jnp.pi, dtype=jnp.float32),
    )
    near_damage_tick = state.replace(
        t=jnp.asarray(HEALTH_GATHERING_ACID_DAMAGE_INTERVAL - 1),
        medkit_active=jnp.zeros((env.max_medkits,), dtype=jnp.bool_),
    )

    _, blocked_state, _, _, _ = env.step(wall_facing_state, ACTION_MOVE_FORWARD)
    _, damaged_state, _, _, info = env.step(near_damage_tick, ACTION_TURN_LEFT)

    assert jnp.allclose(blocked_state.pos, wall_facing_state.pos)
    assert float(info["acid_damage"]) == HEALTH_GATHERING_ACID_DAMAGE
    assert bool(
        jnp.isclose(
            damaged_state.health,
            HEALTH_GATHERING_INITIAL_HEALTH
            - HEALTH_GATHERING_ACID_DAMAGE,
        )
    )
    assert HEALTH_GATHERING_MEDKIT_SPAWN_INTERVAL == 30


def test_health_gathering_step_jit_and_vmap():
    env = HealthGatheringEnv.from_ascii(
        [MAZE_HEALTH_GATHERING],
        initial_medkit_count=2,
        max_medkits=8,
    )
    obs0, state = jax.jit(env.reset)(jax.random.key(0))
    rendered0 = jax.jit(env.render)(state)

    assert obs0.shape == (64, 64, 3)
    assert jnp.array_equal(obs0, rendered0)

    obs1, state1, reward1, done1, info1 = jax.jit(env.step)(
        state, ACTION_MOVE_FORWARD
    )

    assert obs1.shape == (64, 64, 3)
    assert state1.health.shape == ()
    assert reward1.shape == ()
    assert done1.shape == ()
    assert info1["health"].shape == ()

    keys = jax.random.split(jax.random.key(0), 4)
    obs, states = jax.vmap(env.reset)(keys)
    actions = jnp.asarray([0, 1, 2, 2], dtype=jnp.int32)
    obs2, state2, reward2, done2, info2 = jax.vmap(env.step)(states, actions)

    assert obs.shape == (4, 64, 64, 3)
    assert obs2.shape == (4, 64, 64, 3)
    assert state2.health.shape == (4,)
    assert state2.medkit_xy.shape == (4, 8, 2)
    assert reward2.shape == (4,)
    assert done2.shape == (4,)
    assert info2["health"].shape == (4,)
