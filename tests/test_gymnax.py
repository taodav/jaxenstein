import jax
import jax.numpy as jnp
import pytest

from jaxenstein import (
    ACTION_MOVE_FORWARD,
    JAXENSTEIN_ENV_IDS,
    make_jaxenstein_gymnax_env,
)


@pytest.mark.parametrize("env_id", JAXENSTEIN_ENV_IDS)
def test_make_jaxenstein_gymnax_env_supports_public_ids(env_id):
    env, params = make_jaxenstein_gymnax_env(env_id, img_h=16, img_w=20)

    obs, state = env.reset(jax.random.key(0), params)
    action = jnp.asarray(0, dtype=jnp.int32)
    obs, state, reward, done, info = env.step(jax.random.key(2), state, action, params)

    assert env.name == env_id
    assert env.default_params is params
    assert env.action_space(params).n == env.num_actions
    assert env.observation_space(params).shape == (16, 20, 3)
    assert obs.shape == (16, 20, 3)
    assert obs.dtype == jnp.uint8
    assert reward.shape == ()
    assert done.shape == ()
    assert info["discount"].shape == ()
    assert hasattr(state, "done")


@pytest.mark.parametrize("env_id", ("simple", "health-gathering"))
def test_jaxenstein_gymnax_env_jit_reset_and_step(env_id):
    kwargs = {"initial_medkit_count": 0, "max_medkits": 4} if env_id == "health-gathering" else {}
    env, params = make_jaxenstein_gymnax_env(
        env_id,
        img_h=16,
        img_w=16,
        **kwargs,
    )
    key, reset_key, step_key = jax.random.split(jax.random.key(0), 3)
    del key

    obs, state = jax.jit(env.reset)(reset_key, params)
    obs, state, reward, done, info = jax.jit(env.step)(
        step_key, state, ACTION_MOVE_FORWARD, params
    )

    assert obs.shape == (16, 16, 3)
    assert reward.shape == ()
    assert done.shape == ()
    assert info["discount"].shape == ()


def test_jaxenstein_gymnax_env_vmap_reset_and_step():
    env, params = make_jaxenstein_gymnax_env("simple", img_h=16, img_w=16)
    reset_keys = jax.random.split(jax.random.key(0), 4)
    step_keys = jax.random.split(jax.random.key(1), 4)
    actions = jnp.full((4,), ACTION_MOVE_FORWARD, dtype=jnp.int32)

    obs, states = jax.vmap(lambda key: env.reset(key, params))(reset_keys)
    obs, states, reward, done, info = jax.vmap(
        lambda key, state, action: env.step(key, state, action, params)
    )(step_keys, states, actions)

    assert obs.shape == (4, 16, 16, 3)
    assert states.pos.shape == (4, 2)
    assert reward.shape == (4,)
    assert done.shape == (4,)
    assert info["discount"].shape == (4,)


def test_jaxenstein_gymnax_env_auto_resets_terminal_state():
    env, params = make_jaxenstein_gymnax_env("simple", img_h=16, img_w=16)
    _, state = env.reset(jax.random.key(0), params)
    terminal_next_step = state.replace(t=env.native_env.episode_horizon - 1)

    _, next_state, _, done, _ = env.step(
        jax.random.key(1), terminal_next_step, ACTION_MOVE_FORWARD, params
    )

    assert bool(done)
    assert int(next_state.t) == 0
    assert not bool(next_state.done)


def test_health_gathering_gymnax_params_are_passed_to_reset():
    env, params = make_jaxenstein_gymnax_env(
        "health-gathering",
        img_h=16,
        img_w=16,
        initial_health=jnp.asarray(42.0, dtype=jnp.float32),
        initial_medkit_count=0,
        max_medkits=4,
    )

    _, state = env.reset(jax.random.key(0), params)

    assert float(state.health) == 42.0
