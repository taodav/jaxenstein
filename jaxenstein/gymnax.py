"""Gymnax-compatible wrappers for JAXenstein environments."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from gymnax.environments import spaces
from gymnax.environments.environment import Environment

from jaxenstein.env import EnvParams, RayMazeEnv, State
from jaxenstein.factory import make_jaxenstein_env
from jaxenstein.maps.health_gathering import (
    HealthGatheringEnv,
    HealthGatheringParams,
    HealthGatheringState,
)
from jaxenstein.objects import NUM_KEY_COLORS


class _KeySpace:
    def sample(self, key: jax.Array) -> jax.Array:
        return key

    def contains(self, x: jax.Array) -> jax.Array:
        del x
        return jnp.asarray(True)


class _BaseGymnaxEnv(Environment):
    def __init__(self, env_id: str, native_env: RayMazeEnv | HealthGatheringEnv):
        self.env_id = env_id
        self.native_env = native_env

    @property
    def name(self) -> str:
        return self.env_id

    @property
    def default_params(self) -> EnvParams | HealthGatheringParams:
        return self.native_env.params

    @property
    def num_actions(self) -> int:
        return self.native_env.num_actions

    def get_obs(
        self,
        state: State | HealthGatheringState,
        params: EnvParams | HealthGatheringParams | None = None,
        key: jax.Array | None = None,
    ) -> jax.Array:
        del params, key
        return self.native_env.render(state)

    def is_terminal(
        self,
        state: State | HealthGatheringState,
        params: EnvParams | HealthGatheringParams,
    ) -> jax.Array:
        del params
        return state.done

    def discount(
        self,
        state: State | HealthGatheringState,
        params: EnvParams | HealthGatheringParams,
    ) -> jax.Array:
        return jnp.where(
            self.is_terminal(state, params),
            0.0,
            jnp.asarray(self.native_env.gamma, dtype=jnp.float32),
        )

    def action_space(self, params: Any = None) -> spaces.Discrete:
        del params
        return spaces.Discrete(self.num_actions)

    def observation_space(self, params: Any = None) -> spaces.Box:
        del params
        return spaces.Box(
            low=0,
            high=255,
            shape=self.native_env.observation_shape,
            dtype=jnp.uint8,
        )


class JAXensteinGymnaxEnv(_BaseGymnaxEnv):
    """Gymnax adapter for navigation environments."""

    native_env: RayMazeEnv

    def __init__(self, env_id: str, native_env: RayMazeEnv):
        super().__init__(env_id, native_env)

    @property
    def default_params(self) -> EnvParams:
        return self.native_env.params

    def reset_env(self, key: jax.Array, params: EnvParams) -> tuple[jax.Array, State]:
        return self.native_env.reset(key, params)

    def step_env(
        self,
        key: jax.Array,
        state: State,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, State, jax.Array, jax.Array, dict[Any, Any]]:
        del key
        obs, next_state, reward, done, info = self.native_env.step(
            state, action, params
        )
        return obs, next_state, reward, done, {
            **info,
            "discount": self.discount(next_state, params),
        }

    def state_space(self, params: EnvParams | None = None) -> spaces.Dict:
        del params
        return spaces.Dict(
            {
                "pos": _float_space((2,)),
                "theta": spaces.Box(0.0, 2.0 * jnp.pi, (), dtype=jnp.float32),
                "t": _int_space(()),
                "done": _bool_space(()),
                "goal_xy": _float_space((2,)),
                "object_active": _bool_space(self.native_env.maze.object_type.shape),
                "carried_keys": _bool_space((NUM_KEY_COLORS,)),
                "door_open": _bool_space(self.native_env.maze.door_grid.shape),
            }
        )


class HealthGatheringGymnaxEnv(_BaseGymnaxEnv):
    """Gymnax adapter for Health Gathering."""

    native_env: HealthGatheringEnv

    def __init__(self, env_id: str, native_env: HealthGatheringEnv):
        super().__init__(env_id, native_env)

    @property
    def default_params(self) -> HealthGatheringParams:
        return self.native_env.params

    def reset_env(
        self, key: jax.Array, params: HealthGatheringParams
    ) -> tuple[jax.Array, HealthGatheringState]:
        return self.native_env.reset(key, params)

    def step_env(
        self,
        key: jax.Array,
        state: HealthGatheringState,
        action: int | float | jax.Array,
        params: HealthGatheringParams,
    ) -> tuple[
        jax.Array,
        HealthGatheringState,
        jax.Array,
        jax.Array,
        dict[Any, Any],
    ]:
        del key
        obs, next_state, reward, done, info = self.native_env.step(
            state, action, params
        )
        return obs, next_state, reward, done, {
            **info,
            "discount": self.discount(next_state, params),
        }

    def state_space(self, params: HealthGatheringParams | None = None) -> spaces.Dict:
        params = self.default_params if params is None else params
        return spaces.Dict(
            {
                "pos": _float_space((2,)),
                "theta": spaces.Box(0.0, 2.0 * jnp.pi, (), dtype=jnp.float32),
                "t": _int_space(()),
                "done": _bool_space(()),
                "health": spaces.Box(
                    0.0, params.max_health, (), dtype=jnp.float32
                ),
                "medkit_xy": _float_space((self.native_env.max_medkits, 2)),
                "medkit_active": _bool_space((self.native_env.max_medkits,)),
                "next_medkit_slot": spaces.Discrete(self.native_env.max_medkits),
                "rng_key": _KeySpace(),
            }
        )


def make_jaxenstein_gymnax_env(
    env_id: str, **kwargs: Any
) -> tuple[JAXensteinGymnaxEnv | HealthGatheringGymnaxEnv, EnvParams | HealthGatheringParams]:
    """Create a Gymnax-compatible JAXenstein environment and default params."""

    native_env = make_jaxenstein_env(env_id, **kwargs)
    if isinstance(native_env, HealthGatheringEnv):
        env = HealthGatheringGymnaxEnv(env_id, native_env)
    else:
        env = JAXensteinGymnaxEnv(env_id, native_env)
    return env, env.default_params


def _float_space(shape: tuple[int, ...]) -> spaces.Box:
    return spaces.Box(-jnp.inf, jnp.inf, shape, dtype=jnp.float32)


def _int_space(shape: tuple[int, ...]) -> spaces.Box:
    return spaces.Box(
        jnp.iinfo(jnp.int32).min,
        jnp.iinfo(jnp.int32).max,
        shape,
        dtype=jnp.int32,
    )


def _bool_space(shape: tuple[int, ...]) -> spaces.Box:
    return spaces.Box(False, True, shape, dtype=jnp.bool_)
