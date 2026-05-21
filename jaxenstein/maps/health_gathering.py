"""Health Gathering environment."""

from __future__ import annotations

from collections.abc import Sequence

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from jaxenstein.maps.ascii import Maze, parse_ascii_maze
from jaxenstein.maps import (
    HEALTH_GATHERING_ACID_DAMAGE,
    HEALTH_GATHERING_ACID_DAMAGE_INTERVAL,
    HEALTH_GATHERING_DEATH_PENALTY,
    HEALTH_GATHERING_EPISODE_TIMEOUT,
    HEALTH_GATHERING_INITIAL_HEALTH,
    HEALTH_GATHERING_INITIAL_MEDKITS,
    HEALTH_GATHERING_LIVING_REWARD,
    HEALTH_GATHERING_MAX_HEALTH,
    HEALTH_GATHERING_MAX_MEDKITS,
    HEALTH_GATHERING_MEDKIT_HEAL,
    HEALTH_GATHERING_MEDKIT_SPAWN_INTERVAL,
    HEALTH_GATHERING_WALL_RGB as _WALL_RGB,
    HEALTH_GATHERING_FLOOR_RGB as _FLOOR_RGB,
    HEALTH_GATHERING_FLOOR_CHECKER_DARK_RGB as _FLOOR_CHECKER_DARK_RGB,
    HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB as _FLOOR_CHECKER_LIGHT_RGB,
)
from jaxenstein.objects import KEY_COLOR_RED, OBJECT_MEDKIT, object_pickup_radius
from jaxenstein.render import (
    DEFAULT_WALL_PALETTE,
    render_first_person,
)


HEALTH_GATHERING_ACTION_TURN_LEFT = 0
HEALTH_GATHERING_ACTION_TURN_RIGHT = 1
HEALTH_GATHERING_ACTION_MOVE_FORWARD = 2
HEALTH_GATHERING_NUM_ACTIONS = 3

HEALTH_GATHERING_WALL_PALETTE = DEFAULT_WALL_PALETTE.at[1].set(
    jnp.asarray(_WALL_RGB, dtype=jnp.float32)
)
HEALTH_GATHERING_FLOOR_RGB = jnp.asarray(_FLOOR_RGB, dtype=jnp.float32)
HEALTH_GATHERING_FLOOR_CHECKER_DARK_RGB = jnp.asarray(
    _FLOOR_CHECKER_DARK_RGB, dtype=jnp.float32
)
HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB = jnp.asarray(
    _FLOOR_CHECKER_LIGHT_RGB, dtype=jnp.float32
)


@struct.dataclass
class HealthGatheringState:
    pos: jax.Array
    theta: jax.Array
    t: jax.Array
    done: jax.Array
    health: jax.Array
    medkit_xy: jax.Array
    medkit_active: jax.Array
    next_medkit_slot: jax.Array
    rng_key: jax.Array


@struct.dataclass
class HealthGatheringParams:
    turn_angle: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(jnp.pi / 12.0, dtype=jnp.float32)
    )
    move_speed: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(0.15, dtype=jnp.float32)
    )
    initial_health: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            HEALTH_GATHERING_INITIAL_HEALTH, dtype=jnp.float32
        )
    )
    max_health: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            HEALTH_GATHERING_MAX_HEALTH, dtype=jnp.float32
        )
    )
    medkit_heal: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            HEALTH_GATHERING_MEDKIT_HEAL, dtype=jnp.float32
        )
    )
    medkit_radius: jax.Array = struct.field(
        default_factory=lambda: object_pickup_radius(
            jnp.asarray(OBJECT_MEDKIT, dtype=jnp.int32)
        )
    )
    acid_damage: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            HEALTH_GATHERING_ACID_DAMAGE, dtype=jnp.float32
        )
    )
    acid_damage_interval: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            HEALTH_GATHERING_ACID_DAMAGE_INTERVAL, dtype=jnp.int32
        )
    )
    living_reward: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            HEALTH_GATHERING_LIVING_REWARD, dtype=jnp.float32
        )
    )
    death_penalty: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            HEALTH_GATHERING_DEATH_PENALTY, dtype=jnp.float32
        )
    )


class HealthGatheringEnv:
    """A JAX-native Health Gathering task."""

    def __init__(
        self,
        maze: Maze,
        *,
        params: HealthGatheringParams | None = None,
        episode_horizon: jax.Array | int | None = None,
        initial_medkit_count: int = HEALTH_GATHERING_INITIAL_MEDKITS,
        medkit_spawn_interval: int = HEALTH_GATHERING_MEDKIT_SPAWN_INTERVAL,
        max_medkits: int = HEALTH_GATHERING_MAX_MEDKITS,
        img_h: int = 64,
        img_w: int = 64,
        fov: float = jnp.pi / 3.0,
        max_depth: float = 16.0,
        num_depth_samples: int = 128,
        color_palette: Sequence[
            Sequence[float]
        ] | jax.Array = HEALTH_GATHERING_WALL_PALETTE,
        floor_rgb: Sequence[float] | jax.Array = HEALTH_GATHERING_FLOOR_RGB,
        floor_checker_dark_rgb: Sequence[
            float
        ] | jax.Array = HEALTH_GATHERING_FLOOR_CHECKER_DARK_RGB,
        floor_checker_light_rgb: Sequence[
            float
        ] | jax.Array = HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB,
        wall_height_scale: float = 1.35,
        floor_pattern: bool = True,
        gamma: float = 0.99,
    ):
        if initial_medkit_count < 0:
            raise ValueError("initial_medkit_count must be nonnegative")
        if medkit_spawn_interval < 1:
            raise ValueError("medkit_spawn_interval must be at least 1")
        if max_medkits <= initial_medkit_count:
            raise ValueError("max_medkits must be > initial_medkit_count")

        self.maze = maze
        self.params = HealthGatheringParams() if params is None else params
        self.initial_medkit_count = int(initial_medkit_count)
        self.medkit_spawn_interval = int(medkit_spawn_interval)
        self.max_medkits = int(max_medkits)
        self.img_h = img_h
        self.img_w = img_w
        self.fov = fov
        self.max_depth = max_depth
        self.num_depth_samples = num_depth_samples
        self.color_palette = jnp.asarray(color_palette, dtype=jnp.float32)
        self.floor_rgb = jnp.asarray(floor_rgb, dtype=jnp.float32)
        self.floor_checker_dark_rgb = jnp.asarray(
            floor_checker_dark_rgb, dtype=jnp.float32
        )
        self.floor_checker_light_rgb = jnp.asarray(
            floor_checker_light_rgb, dtype=jnp.float32
        )
        self.wall_height_scale = wall_height_scale
        self.floor_pattern = floor_pattern
        self.gamma = float(gamma)
        self.episode_horizon = _episode_horizon(
            episode_horizon,
            default_horizon=jnp.asarray(
                HEALTH_GATHERING_EPISODE_TIMEOUT, dtype=jnp.int32
            ),
        )
        self.floor_xy_options, self.floor_count = _floor_options(maze)
        self._medkit_type = jnp.full(
            (self.max_medkits,), OBJECT_MEDKIT, dtype=jnp.int32
        )
        self._medkit_color = jnp.full(
            (self.max_medkits,), KEY_COLOR_RED, dtype=jnp.int32
        )

    @classmethod
    def from_ascii(cls, ascii_maze: str, **kwargs) -> "HealthGatheringEnv":
        return cls(parse_ascii_maze(ascii_maze, require_goal=False), **kwargs)

    @property
    def num_actions(self) -> int:
        return HEALTH_GATHERING_NUM_ACTIONS

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.img_h, self.img_w, 3)

    def reset(
        self, key: jax.Array, params: HealthGatheringParams | None = None
    ) -> tuple[jax.Array, HealthGatheringState]:
        params = self.params if params is None else params
        spawn_key, medkit_key, state_key = jax.random.split(key, 3)
        spawn_idx = jax.random.randint(
            spawn_key,
            shape=(),
            minval=0,
            maxval=self.maze.spawn_count,
            dtype=jnp.int32,
        )
        medkit_keys = jax.random.split(medkit_key, self.max_medkits)
        medkit_xy = jax.vmap(self._sample_floor_xy)(medkit_keys)
        medkit_active = jnp.arange(self.max_medkits) < self.initial_medkit_count
        state = HealthGatheringState(
            pos=self.maze.spawn_xy_options[spawn_idx],
            theta=self.maze.spawn_theta,
            t=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
            health=params.initial_health,
            medkit_xy=medkit_xy,
            medkit_active=medkit_active,
            next_medkit_slot=jnp.asarray(self.initial_medkit_count, dtype=jnp.int32),
            rng_key=state_key,
        )
        return self.render(state), state

    def step(
        self,
        state: HealthGatheringState,
        action: jax.Array | int,
        params: HealthGatheringParams | None = None,
    ) -> tuple[
        jax.Array,
        HealthGatheringState,
        jax.Array,
        jax.Array,
        dict[str, jax.Array],
    ]:
        params = self.params if params is None else params
        action = jnp.asarray(action, dtype=jnp.int32)
        active = ~state.done

        turn_delta = jnp.where(
            action == HEALTH_GATHERING_ACTION_TURN_LEFT,
            -params.turn_angle,
            jnp.where(
                action == HEALTH_GATHERING_ACTION_TURN_RIGHT,
                params.turn_angle,
                0.0,
            ),
        )
        theta = jnp.mod(state.theta + jnp.where(active, turn_delta, 0.0), 2.0 * jnp.pi)

        forward = jnp.asarray([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.float32)
        move_direction = jnp.where(
            action == HEALTH_GATHERING_ACTION_MOVE_FORWARD, 1.0, 0.0
        )
        proposed_pos = state.pos + jnp.where(
            active,
            move_direction * params.move_speed * forward,
            jnp.zeros((2,), dtype=jnp.float32),
        )
        wall_grid = self.maze.wall_grid
        blocked = _is_wall_at(wall_grid, proposed_pos)
        pos = jnp.where(blocked, state.pos, proposed_pos)

        medkit_distances = jnp.linalg.norm(state.medkit_xy - pos[None, :], axis=-1)
        picked_medkits = active & state.medkit_active & (
            medkit_distances < params.medkit_radius
        )
        picked_medkit = jnp.any(picked_medkits)
        medkit_active = state.medkit_active & ~picked_medkits
        health_after_pickup = jnp.minimum(
            state.health + jnp.where(picked_medkit, params.medkit_heal, 0.0),
            params.max_health,
        )

        t = state.t + jnp.where(active, 1, 0).astype(jnp.int32)
        acid_damage_interval = jnp.maximum(params.acid_damage_interval, 1)
        takes_acid_damage = (
            active
            & (params.acid_damage > 0.0)
            & (jnp.mod(t, acid_damage_interval) == 0)
        )
        health = jnp.maximum(
            health_after_pickup
            - jnp.where(takes_acid_damage, params.acid_damage, 0.0),
            0.0,
        )
        died = active & (health <= 0.0)
        done = state.done | died | (t >= self.episode_horizon)

        rng_key, spawn_key = jax.random.split(state.rng_key)
        spawn_medkit = (
            active
            & ~done
            & (
                jnp.mod(
                    t,
                    jnp.asarray(self.medkit_spawn_interval, dtype=jnp.int32),
                )
                == 0
            )
        )
        spawn_xy = self._sample_floor_xy(spawn_key)
        slot = state.next_medkit_slot
        medkit_xy = state.medkit_xy.at[slot].set(
            jnp.where(spawn_medkit, spawn_xy, state.medkit_xy[slot])
        )
        medkit_active = medkit_active.at[slot].set(
            jnp.where(spawn_medkit, True, medkit_active[slot])
        )
        next_medkit_slot = jnp.where(
            spawn_medkit,
            jnp.mod(slot + 1, self.max_medkits),
            slot,
        )
        rng_key = jnp.where(active, rng_key, state.rng_key)
        reward = jnp.where(
            active,
            jnp.where(died, -params.death_penalty, params.living_reward),
            0.0,
        ).astype(jnp.float32)

        new_state = HealthGatheringState(
            pos=pos,
            theta=theta,
            t=t,
            done=done,
            health=health,
            medkit_xy=medkit_xy,
            medkit_active=medkit_active,
            next_medkit_slot=next_medkit_slot,
            rng_key=rng_key,
        )
        info = {
            "x": pos[0],
            "y": pos[1],
            "theta": theta,
            "cell_x": jnp.floor(pos[0]).astype(jnp.int32),
            "cell_y": jnp.floor(pos[1]).astype(jnp.int32),
            "health": health,
            "picked_medkit": picked_medkit,
            "spawned_medkit": spawn_medkit,
            "acid_damage": jnp.where(takes_acid_damage, params.acid_damage, 0.0),
            "died": died,
        }
        return self.render(new_state), new_state, reward, done, info

    def render(self, state: HealthGatheringState) -> jax.Array:
        return render_first_person(
            state.pos,
            state.theta,
            self.maze.wall_grid,
            self.maze.color_grid,
            img_h=self.img_h,
            img_w=self.img_w,
            fov=self.fov,
            max_depth=self.max_depth,
            num_depth_samples=self.num_depth_samples,
            color_palette=self.color_palette,
            floor_rgb=self.floor_rgb,
            floor_checker_dark_rgb=self.floor_checker_dark_rgb,
            floor_checker_light_rgb=self.floor_checker_light_rgb,
            object_xy=state.medkit_xy,
            object_type=self._medkit_type,
            object_color=self._medkit_color,
            object_active=state.medkit_active,
            wall_height_scale=self.wall_height_scale,
            floor_pattern=self.floor_pattern,
        )

    def _sample_floor_xy(self, key: jax.Array) -> jax.Array:
        cell_key, jitter_key = jax.random.split(key)
        option_idx = jax.random.randint(
            cell_key,
            shape=(),
            minval=0,
            maxval=self.floor_count,
            dtype=jnp.int32,
        )
        cell_xy = self.floor_xy_options[option_idx]
        jitter = jax.random.uniform(
            jitter_key,
            shape=(2,),
            minval=-0.42,
            maxval=0.42,
            dtype=jnp.float32,
        )
        return cell_xy + jitter


def _floor_options(maze: Maze) -> tuple[jax.Array, jax.Array]:
    wall_grid = np.asarray(maze.wall_grid)
    open_y, open_x = np.nonzero(~wall_grid)
    if open_x.size == 0:
        raise ValueError("health gathering maps must contain at least one floor cell")
    xy = np.stack([open_x + 0.5, open_y + 0.5], axis=-1).astype(np.float32)
    return jnp.asarray(xy), jnp.asarray(xy.shape[0], dtype=jnp.int32)


def _is_wall_at(wall_grid: jax.Array, xy: jax.Array) -> jax.Array:
    h, w = wall_grid.shape
    cell_x = jnp.floor(xy[0]).astype(jnp.int32)
    cell_y = jnp.floor(xy[1]).astype(jnp.int32)
    inside = (cell_x >= 0) & (cell_x < w) & (cell_y >= 0) & (cell_y < h)
    clipped_x = jnp.clip(cell_x, 0, w - 1)
    clipped_y = jnp.clip(cell_y, 0, h - 1)
    return wall_grid[clipped_y, clipped_x] | ~inside


def _episode_horizon(
    episode_horizon: jax.Array | int | None,
    *,
    default_horizon: jax.Array,
) -> jax.Array:
    if episode_horizon is None:
        return jnp.asarray(default_horizon, dtype=jnp.int32)

    horizon = jnp.asarray(episode_horizon, dtype=jnp.int32)
    if horizon.ndim != 0:
        raise ValueError(f"episode_horizon must be scalar, got {horizon.shape}")
    return horizon
