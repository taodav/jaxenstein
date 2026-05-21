"""Public Jaxenstein API."""

from __future__ import annotations

from typing import Any

from jaxenstein.env import (
    ACTION_INTERACT,
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
    NUM_ACTIONS,
    EnvParams,
    RayMazeEnv,
    State,
)
from jaxenstein.factory import JAXENSTEIN_ENV_IDS, make_jaxenstein_env
from jaxenstein.maps.health_gathering import (
    HEALTH_GATHERING_ACTION_MOVE_FORWARD,
    HEALTH_GATHERING_ACTION_TURN_LEFT,
    HEALTH_GATHERING_ACTION_TURN_RIGHT,
    HEALTH_GATHERING_NUM_ACTIONS,
    HealthGatheringEnv,
    HealthGatheringParams,
    HealthGatheringState,
)


def make_jaxenstein_gymnax_env(env_id: str, **kwargs: Any) -> Any:
    """Create a Gymnax-compatible Jaxenstein environment and default params."""

    from jaxenstein.gymnax import make_jaxenstein_gymnax_env as _make_gymnax_env

    return _make_gymnax_env(env_id, **kwargs)


__all__ = [
    "ACTION_INTERACT",
    "ACTION_MOVE_BACKWARD",
    "ACTION_MOVE_FORWARD",
    "ACTION_TURN_LEFT",
    "ACTION_TURN_RIGHT",
    "EnvParams",
    "HEALTH_GATHERING_ACTION_MOVE_FORWARD",
    "HEALTH_GATHERING_ACTION_TURN_LEFT",
    "HEALTH_GATHERING_ACTION_TURN_RIGHT",
    "HEALTH_GATHERING_NUM_ACTIONS",
    "HealthGatheringEnv",
    "HealthGatheringParams",
    "HealthGatheringState",
    "JAXENSTEIN_ENV_IDS",
    "NUM_ACTIONS",
    "RayMazeEnv",
    "State",
    "make_jaxenstein_env",
    "make_jaxenstein_gymnax_env",
]
