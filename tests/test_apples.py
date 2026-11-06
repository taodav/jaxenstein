import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxenstein import ACTION_INTERACT, RayMazeEnv, make_jaxenstein_env
from jaxenstein.maps.ascii import parse_ascii_maze
from jaxenstein.objects import KEY_COLOR_YELLOW, OBJECT_APPLE, OBJECT_GOAL
from jaxenstein.render import render_billboard_sprites


def test_spawn_can_share_an_apple_without_changing_wall_symbols():
    maze = parse_ascii_maze("########\n#K.D.AG#\n########")
    assert int(maze.spawn_count) == 1
    assert np.array_equal(maze.spawn_xy, [1.5, 1.5])
    assert np.array_equal(
        maze.object_xy[maze.object_type == OBJECT_APPLE], [[1.5, 1.5], [3.5, 1.5]]
    )
    assert bool(maze.wall_grid[1, 5])  # A remains a colored wall.


def test_apple_reward_is_once_per_episode_and_does_not_terminate():
    env = RayMazeEnv.from_ascii(
        "######\n#KD.G#\n######",
        img_h=8,
        img_w=8,
        goal_reward=10,
        apple_reward=1,
        living_reward=-0.1,
    )
    _, state = env.reset(jax.random.key(0))
    step = jax.jit(env.step)
    _, state, reward, done, info = step(state, ACTION_INTERACT)
    assert float(reward) == pytest.approx(0.9)
    assert int(info["picked_apples"]) == 1
    assert not bool(done)
    assert not bool(info["reached_goal"])
    assert not bool(state.object_active[0])
    _, state, reward, _, _ = step(state, ACTION_INTERACT)
    assert float(reward) == pytest.approx(-0.1)
    _, state, reward, done, _ = step(
        state.replace(pos=jnp.array([2.5, 1.5])), ACTION_INTERACT
    )
    assert float(reward) == pytest.approx(0.9)
    assert not bool(done)  # Collecting every apple does not end the level.
    _, state, reward, done, info = step(
        state.replace(pos=state.goal_xy), ACTION_INTERACT
    )
    assert float(reward) == pytest.approx(9.9)
    assert bool(done) and bool(info["reached_goal"])
    _, _, reward, _, info = step(state, ACTION_INTERACT)
    assert float(reward) == 0.0
    assert int(info["picked_apples"]) == 0
    _, fresh = env.reset(jax.random.key(0))
    assert np.all(fresh.object_active)


def test_batched_random_map_apples_and_goal_have_distinct_rewards():
    env = make_jaxenstein_env("dmlab-random-01", img_h=8, img_w=8)
    keys = jax.random.split(jax.random.key(8), 4)
    states = jax.jit(jax.vmap(lambda key: env.reset(key)[1]))(keys)
    types = jax.vmap(env._object_types)(states)
    colors = jax.vmap(env._object_colors)(states)
    assert np.all(
        np.asarray(colors)[np.asarray(types) == OBJECT_GOAL] == KEY_COLOR_YELLOW
    )
    apple_idx = jnp.argmax(states.object_active & (types == OBJECT_APPLE), axis=1)
    on_apples = states.replace(pos=env.maze.object_xy[apple_idx])
    step = jax.jit(jax.vmap(env.step))
    _, after, rewards, dones, infos = step(on_apples, jnp.full((4,), ACTION_INTERACT))
    assert np.all(np.asarray(rewards) == 1)
    assert not np.any(dones)
    assert np.all(np.asarray(infos["picked_apples"]) == 1)
    _, _, rewards, _, _ = step(after, jnp.full((4,), ACTION_INTERACT))
    assert np.all(np.asarray(rewards) == 0)
    _, _, rewards, dones, infos = step(
        after.replace(pos=after.goal_xy), jnp.full((4,), ACTION_INTERACT)
    )
    assert np.all(np.asarray(rewards) == 10)
    assert np.all(dones)
    assert np.all(infos["reached_goal"])


def test_apple_sprite_has_fruit_leaf_stem_and_respects_visibility():
    background = jnp.zeros((64, 64, 3))

    @jax.jit
    def draw(active, theta, wall_depth):
        return render_billboard_sprites(
            background,
            jnp.array([0.0, 0.0]),
            theta,
            jnp.array([[1.5, 0.0]]),
            jnp.array([OBJECT_APPLE]),
            jnp.array([1]),
            jnp.array([active]),
            jnp.full((64,), wall_depth),
            img_h=64,
            img_w=64,
            fov=jnp.pi / 3,
        )

    visible = np.asarray(draw(True, 0.0, 10.0))
    for color in ([214, 38, 36], [255, 126, 99], [64, 160, 48], [104, 62, 27]):
        assert np.any(np.all(visible == color, axis=-1))
    assert np.array_equal(draw(False, 0.0, 10.0), background)
    assert np.array_equal(draw(True, jnp.pi, 10.0), background)
    assert np.array_equal(draw(True, 0.0, 1.0), background)


def test_non_dmlab_defaults_keep_fixed_heading_and_custom_apple_reward():
    env = make_jaxenstein_env("simple", img_h=8, img_w=8)
    _, state = env.reset(jax.random.key(1))
    assert float(state.theta) == 0.0
    custom = RayMazeEnv.from_ascii(
        "#####\n#K.G#\n#####", apple_reward=2.5, img_h=8, img_w=8
    )
    _, state = custom.reset(jax.random.key(1))
    assert float(custom.step(state, ACTION_INTERACT)[2]) == 2.5


@pytest.mark.parametrize("density,expected", [(0.0, 0), (1.0, 49)])
def test_random_fruit_density_boundary(density, expected):
    env = make_jaxenstein_env(
        "dmlab-random-01", scattered_reward_density=density, img_h=8, img_w=8
    )
    _, state = env.reset(jax.random.key(2))
    assert (
        int(jnp.sum(state.object_active & (env._object_types(state) == OBJECT_APPLE)))
        == expected
    )
