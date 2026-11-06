"""Environment factories and registry IDs."""

from __future__ import annotations

from typing import Any

import jax

from jaxenstein.env import RayMazeEnv
from jaxenstein.maps import (
    DEFAULT_DISCOUNT_GAMMA,
    HEALTH_GATHERING_EPISODE_HORIZONS_BY_NAME,
    HEALTH_GATHERING_MAPS_BY_NAME,
    HEALTH_GATHERING_RENDER_KWARGS_BY_NAME,
    MAP_DISCOUNT_GAMMAS_BY_NAME,
    MAP_EPISODE_HORIZONS_BY_NAME,
    MAP_RENDER_KWARGS_BY_NAME,
    MAP_REWARD_KWARGS_BY_NAME,
    MAPS_BY_NAME,
)
from jaxenstein.maps.dmlab_random_goal import DMLabRandomGoalEnv
from jaxenstein.maps.health_gathering import (
    HealthGatheringEnv,
    HealthGatheringParams,
)

JAXENSTEIN_ENV_IDS = tuple(sorted((*MAPS_BY_NAME, *HEALTH_GATHERING_MAPS_BY_NAME)))
_HEALTH_GATHERING_PARAM_NAMES = frozenset(HealthGatheringParams.__dataclass_fields__)


def make_jaxenstein_env(env_id: str, **kwargs: Any) -> RayMazeEnv | HealthGatheringEnv:
    """Create a JAXenstein environment from a registered environment ID."""

    # Catalog construction validates static arrays on the CPU. Training code
    # may call this factory while tracing its enclosing JIT/vmap function.
    with jax.ensure_compile_time_eval():
        return _make_jaxenstein_env(env_id, **kwargs)


def _make_jaxenstein_env(env_id: str, **kwargs: Any) -> RayMazeEnv | HealthGatheringEnv:

    if env_id in HEALTH_GATHERING_MAPS_BY_NAME:
        params = kwargs.pop("params", None)
        params = _apply_health_gathering_param_overrides(params, kwargs)
        env_kwargs = {
            "episode_horizon": HEALTH_GATHERING_EPISODE_HORIZONS_BY_NAME.get(env_id),
            **HEALTH_GATHERING_RENDER_KWARGS_BY_NAME.get(env_id, {}),
            **kwargs,
        }
        if params is not None:
            env_kwargs["params"] = params
        return HealthGatheringEnv.from_ascii(
            HEALTH_GATHERING_MAPS_BY_NAME[env_id],
            **env_kwargs,
        )

    if env_id in MAPS_BY_NAME:
        env_kwargs = {
            "randomize_spawn_heading": env_id.startswith("dmlab-"),
            "episode_horizon": MAP_EPISODE_HORIZONS_BY_NAME.get(env_id),
            **MAP_REWARD_KWARGS_BY_NAME.get(env_id, {}),
            **MAP_RENDER_KWARGS_BY_NAME.get(env_id, {}),
            "gamma": MAP_DISCOUNT_GAMMAS_BY_NAME.get(env_id, DEFAULT_DISCOUNT_GAMMA),
            **kwargs,
        }
        if env_id.startswith("dmlab-"):
            from jaxenstein.maps.dmlab import dmlab_object_types
            from jaxenstein.maps.dmlab_entities import DMLAB_OBJECTS_BY_NAME

            defaults = dmlab_object_types(
                apple_reward=env_kwargs.pop("apple_reward", 1.0),
                goal_reward=env_kwargs.get("goal_reward", 1.0),
            )
            env_kwargs["object_types"] = {
                **defaults,
                **(env_kwargs.get("object_types") or {}),
            }
            env_kwargs.setdefault("objects", DMLAB_OBJECTS_BY_NAME[env_id])
        env_cls = (
            DMLabRandomGoalEnv if env_id.startswith("dmlab-random-") else RayMazeEnv
        )
        return env_cls.from_ascii(MAPS_BY_NAME[env_id], **env_kwargs)

    raise ValueError(
        f"unknown env_id {env_id!r}; expected one of {list(JAXENSTEIN_ENV_IDS)}"
    )


def _apply_health_gathering_param_overrides(
    params: HealthGatheringParams | None,
    kwargs: dict[str, Any],
) -> HealthGatheringParams | None:
    param_kwargs = {}
    for name in tuple(kwargs):
        if name in _HEALTH_GATHERING_PARAM_NAMES:
            param_kwargs[name] = kwargs.pop(name)

    if not param_kwargs:
        return params

    base_params = HealthGatheringParams() if params is None else params
    return base_params.replace(**param_kwargs)
