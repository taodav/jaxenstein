import pytest

from jaxenstein import (
    JAXENSTEIN_ENV_IDS,
    HealthGatheringEnv,
    RayMazeEnv,
    make_jaxenstein_env,
)
from jaxenstein.maps import (
    HEALTH_GATHERING_EPISODE_HORIZONS_BY_NAME,
    MAP_DISCOUNT_GAMMAS_BY_NAME,
    MAP_EPISODE_HORIZONS_BY_NAME,
)


README_ENV_IDS = (
    "simple",
    "key-door",
    "key-corridor",
    "health-gathering",
    "my-way-home",
    "my-way-home-colorless",
    "dmlab-static-01",
    "dmlab-static-02",
    "dmlab-static-03",
    "dmlab-random-01",
    "dmlab-random-02",
    "dmlab-random-03",
)


def test_env_ids_match_readme_ids():
    assert JAXENSTEIN_ENV_IDS == tuple(sorted(README_ENV_IDS))


@pytest.mark.parametrize("env_id", README_ENV_IDS)
def test_make_jaxenstein_env_supports_readme_ids(env_id):
    env = make_jaxenstein_env(env_id)

    if env_id == "health-gathering":
        assert isinstance(env, HealthGatheringEnv)
        assert int(env.episode_horizon) == HEALTH_GATHERING_EPISODE_HORIZONS_BY_NAME[
            env_id
        ]
    else:
        assert isinstance(env, RayMazeEnv)
        assert env.gamma == MAP_DISCOUNT_GAMMAS_BY_NAME[env_id]
        if env_id in MAP_EPISODE_HORIZONS_BY_NAME:
            assert int(env.episode_horizon) == MAP_EPISODE_HORIZONS_BY_NAME[env_id]


def test_make_jaxenstein_env_applies_overrides_after_registry_defaults():
    env = make_jaxenstein_env(
        "dmlab-static-02",
        img_h=32,
        img_w=48,
        gamma=0.5,
        goal_reward=3.0,
    )

    assert isinstance(env, RayMazeEnv)
    assert env.observation_shape == (32, 48, 3)
    assert env.gamma == 0.5
    assert float(env.goal_reward) == 3.0


def test_make_jaxenstein_env_rejects_unknown_ids():
    with pytest.raises(ValueError, match="unknown env_id"):
        make_jaxenstein_env("missing")
