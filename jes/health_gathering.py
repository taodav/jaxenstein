"""ViZDoom-style Health Gathering environment."""

from __future__ import annotations

from collections.abc import Sequence

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from jes.ascii import Maze, MazeBatch, parse_and_stack, stack_mazes
from jes.maps import (
    VIZDOOM_HEALTH_GATHERING_ACID_DAMAGE,
    VIZDOOM_HEALTH_GATHERING_ACID_DAMAGE_INTERVAL,
    VIZDOOM_HEALTH_GATHERING_DEATH_PENALTY,
    VIZDOOM_HEALTH_GATHERING_EPISODE_TIMEOUT,
    VIZDOOM_HEALTH_GATHERING_INITIAL_HEALTH,
    VIZDOOM_HEALTH_GATHERING_INITIAL_MEDKITS,
    VIZDOOM_HEALTH_GATHERING_LIVING_REWARD,
    VIZDOOM_HEALTH_GATHERING_MAX_HEALTH,
    VIZDOOM_HEALTH_GATHERING_MAX_MEDKITS,
    VIZDOOM_HEALTH_GATHERING_MEDKIT_HEAL,
    VIZDOOM_HEALTH_GATHERING_MEDKIT_SPAWN_INTERVAL,
    VIZDOOM_HEALTH_GATHERING_WALL_RGB,
    VIZDOOM_HEALTH_GATHERING_FLOOR_RGB,
    VIZDOOM_HEALTH_GATHERING_FLOOR_CHECKER_DARK_RGB,
    VIZDOOM_HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB,
)
from jes.objects import KEY_COLOR_RED, OBJECT_MEDKIT, object_pickup_radius
from jes.render import (
    DEFAULT_WALL_PALETTE,
    render_first_person,
)


HEALTH_GATHERING_ACTION_TURN_LEFT = 0
HEALTH_GATHERING_ACTION_TURN_RIGHT = 1
HEALTH_GATHERING_ACTION_MOVE_FORWARD = 2
HEALTH_GATHERING_NUM_ACTIONS = 3

HEALTH_GATHERING_WALL_PALETTE = DEFAULT_WALL_PALETTE.at[1].set(
    jnp.asarray(VIZDOOM_HEALTH_GATHERING_WALL_RGB, dtype=jnp.float32)
)
HEALTH_GATHERING_FLOOR_RGB = jnp.asarray(
    VIZDOOM_HEALTH_GATHERING_FLOOR_RGB, dtype=jnp.float32
)
HEALTH_GATHERING_FLOOR_CHECKER_DARK_RGB = jnp.asarray(
    VIZDOOM_HEALTH_GATHERING_FLOOR_CHECKER_DARK_RGB, dtype=jnp.float32
)
HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB = jnp.asarray(
    VIZDOOM_HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB, dtype=jnp.float32
)


@struct.dataclass
class HealthGatheringState:
    pos: jax.Array
    theta: jax.Array
    t: jax.Array
    done: jax.Array
    maze_id: jax.Array
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
            VIZDOOM_HEALTH_GATHERING_INITIAL_HEALTH, dtype=jnp.float32
        )
    )
    max_health: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            VIZDOOM_HEALTH_GATHERING_MAX_HEALTH, dtype=jnp.float32
        )
    )
    medkit_heal: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            VIZDOOM_HEALTH_GATHERING_MEDKIT_HEAL, dtype=jnp.float32
        )
    )
    medkit_radius: jax.Array = struct.field(
        default_factory=lambda: object_pickup_radius(
            jnp.asarray(OBJECT_MEDKIT, dtype=jnp.int32)
        )
    )
    acid_damage: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            VIZDOOM_HEALTH_GATHERING_ACID_DAMAGE, dtype=jnp.float32
        )
    )
    acid_damage_interval: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            VIZDOOM_HEALTH_GATHERING_ACID_DAMAGE_INTERVAL, dtype=jnp.int32
        )
    )
    living_reward: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            VIZDOOM_HEALTH_GATHERING_LIVING_REWARD, dtype=jnp.float32
        )
    )
    death_penalty: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(
            VIZDOOM_HEALTH_GATHERING_DEATH_PENALTY, dtype=jnp.float32
        )
    )


class HealthGatheringEnv:
    """A JAX-native approximation of ViZDoom Health Gathering."""

    def __init__(
        self,
        maze_batch: MazeBatch,
        *,
        params: HealthGatheringParams | None = None,
        episode_horizons: Sequence[int] | jax.Array | int | None = None,
        initial_medkit_count: int = VIZDOOM_HEALTH_GATHERING_INITIAL_MEDKITS,
        medkit_spawn_interval: int = VIZDOOM_HEALTH_GATHERING_MEDKIT_SPAWN_INTERVAL,
        max_medkits: int = VIZDOOM_HEALTH_GATHERING_MAX_MEDKITS,
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

        self.maze_batch = maze_batch
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
        self.episode_horizons = _episode_horizon_array(
            episode_horizons,
            num_mazes=int(maze_batch.wall_grids.shape[0]),
            default_horizon=jnp.asarray(
                VIZDOOM_HEALTH_GATHERING_EPISODE_TIMEOUT, dtype=jnp.int32
            ),
        )
        self.floor_xy_options, self.floor_counts = _floor_options(maze_batch)
        self._medkit_type = jnp.full(
            (self.max_medkits,), OBJECT_MEDKIT, dtype=jnp.int32
        )
        self._medkit_color = jnp.full(
            (self.max_medkits,), KEY_COLOR_RED, dtype=jnp.int32
        )

    @classmethod
    def from_mazes(cls, mazes: Sequence[Maze], **kwargs) -> "HealthGatheringEnv":
        return cls(stack_mazes(list(mazes)), **kwargs)

    @classmethod
    def from_ascii(
        cls, ascii_mazes: Sequence[str], **kwargs
    ) -> "HealthGatheringEnv":
        return cls(parse_and_stack(list(ascii_mazes), require_goal=False), **kwargs)

    @property
    def num_mazes(self) -> int:
        return int(self.maze_batch.wall_grids.shape[0])

    @property
    def num_actions(self) -> int:
        return HEALTH_GATHERING_NUM_ACTIONS

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.img_h, self.img_w, 3)

    def reset(self, key: jax.Array) -> tuple[jax.Array, HealthGatheringState]:
        if self.num_mazes == 1:
            maze_id = jnp.asarray(0, dtype=jnp.int32)
        else:
            maze_key, key = jax.random.split(key)
            maze_id = jax.random.randint(
                maze_key,
                shape=(),
                minval=0,
                maxval=self.num_mazes,
                dtype=jnp.int32,
            )

        spawn_key, medkit_key, state_key = jax.random.split(key, 3)
        spawn_idx = jax.random.randint(
            spawn_key,
            shape=(),
            minval=0,
            maxval=self.maze_batch.spawn_count[maze_id],
            dtype=jnp.int32,
        )
        medkit_keys = jax.random.split(medkit_key, self.max_medkits)
        medkit_xy = jax.vmap(lambda k: self._sample_floor_xy(k, maze_id))(medkit_keys)
        medkit_active = jnp.arange(self.max_medkits) < self.initial_medkit_count
        state = HealthGatheringState(
            pos=self.maze_batch.spawn_xy_options[maze_id, spawn_idx],
            theta=self.maze_batch.spawn_theta[maze_id],
            t=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
            maze_id=maze_id,
            health=self.params.initial_health,
            medkit_xy=medkit_xy,
            medkit_active=medkit_active,
            next_medkit_slot=jnp.asarray(self.initial_medkit_count, dtype=jnp.int32),
            rng_key=state_key,
        )
        return self.render(state), state

    def step(
        self, state: HealthGatheringState, action: jax.Array | int
    ) -> tuple[
        jax.Array,
        HealthGatheringState,
        jax.Array,
        jax.Array,
        dict[str, jax.Array],
    ]:
        action = jnp.asarray(action, dtype=jnp.int32)
        active = ~state.done

        turn_delta = jnp.where(
            action == HEALTH_GATHERING_ACTION_TURN_LEFT,
            -self.params.turn_angle,
            jnp.where(
                action == HEALTH_GATHERING_ACTION_TURN_RIGHT,
                self.params.turn_angle,
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
            move_direction * self.params.move_speed * forward,
            jnp.zeros((2,), dtype=jnp.float32),
        )
        wall_grid = self.maze_batch.wall_grids[state.maze_id]
        blocked = _is_wall_at(wall_grid, proposed_pos)
        pos = jnp.where(blocked, state.pos, proposed_pos)

        medkit_distances = jnp.linalg.norm(state.medkit_xy - pos[None, :], axis=-1)
        picked_medkits = active & state.medkit_active & (
            medkit_distances < self.params.medkit_radius
        )
        picked_medkit = jnp.any(picked_medkits)
        medkit_active = state.medkit_active & ~picked_medkits
        health_after_pickup = jnp.minimum(
            state.health + jnp.where(picked_medkit, self.params.medkit_heal, 0.0),
            self.params.max_health,
        )

        t = state.t + jnp.where(active, 1, 0).astype(jnp.int32)
        acid_damage_interval = jnp.maximum(self.params.acid_damage_interval, 1)
        takes_acid_damage = (
            active
            & (self.params.acid_damage > 0.0)
            & (jnp.mod(t, acid_damage_interval) == 0)
        )
        health = jnp.maximum(
            health_after_pickup
            - jnp.where(takes_acid_damage, self.params.acid_damage, 0.0),
            0.0,
        )
        died = active & (health <= 0.0)
        episode_horizon = self.episode_horizons[state.maze_id]
        done = state.done | died | (t >= episode_horizon)

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
        spawn_xy = self._sample_floor_xy(spawn_key, state.maze_id)
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
            jnp.where(died, -self.params.death_penalty, self.params.living_reward),
            0.0,
        ).astype(jnp.float32)

        new_state = HealthGatheringState(
            pos=pos,
            theta=theta,
            t=t,
            done=done,
            maze_id=state.maze_id,
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
            "acid_damage": jnp.where(takes_acid_damage, self.params.acid_damage, 0.0),
            "died": died,
            "maze_id": state.maze_id,
        }
        return self.render(new_state), new_state, reward, done, info

    def render(self, state: HealthGatheringState) -> jax.Array:
        return render_first_person(
            state.pos,
            state.theta,
            self.maze_batch.wall_grids[state.maze_id],
            self.maze_batch.color_grids[state.maze_id],
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

    def _sample_floor_xy(self, key: jax.Array, maze_id: jax.Array) -> jax.Array:
        cell_key, jitter_key = jax.random.split(key)
        option_idx = jax.random.randint(
            cell_key,
            shape=(),
            minval=0,
            maxval=self.floor_counts[maze_id],
            dtype=jnp.int32,
        )
        cell_xy = self.floor_xy_options[maze_id, option_idx]
        jitter = jax.random.uniform(
            jitter_key,
            shape=(2,),
            minval=-0.42,
            maxval=0.42,
            dtype=jnp.float32,
        )
        return cell_xy + jitter


def _floor_options(maze_batch: MazeBatch) -> tuple[jax.Array, jax.Array]:
    wall_grids = np.asarray(maze_batch.wall_grids)
    options: list[np.ndarray] = []
    counts: list[int] = []
    for wall_grid in wall_grids:
        open_y, open_x = np.nonzero(~wall_grid)
        if open_x.size == 0:
            raise ValueError(
                "health gathering maps must contain at least one floor cell"
            )
        xy = np.stack([open_x + 0.5, open_y + 0.5], axis=-1).astype(np.float32)
        options.append(xy)
        counts.append(int(xy.shape[0]))

    max_options = max(counts)
    padded = []
    for xy in options:
        pad = max_options - int(xy.shape[0])
        padded.append(np.pad(xy, ((0, pad), (0, 0)), constant_values=0.0))
    return jnp.asarray(np.stack(padded)), jnp.asarray(counts, dtype=jnp.int32)


def _is_wall_at(wall_grid: jax.Array, xy: jax.Array) -> jax.Array:
    h, w = wall_grid.shape
    cell_x = jnp.floor(xy[0]).astype(jnp.int32)
    cell_y = jnp.floor(xy[1]).astype(jnp.int32)
    inside = (cell_x >= 0) & (cell_x < w) & (cell_y >= 0) & (cell_y < h)
    clipped_x = jnp.clip(cell_x, 0, w - 1)
    clipped_y = jnp.clip(cell_y, 0, h - 1)
    return wall_grid[clipped_y, clipped_x] | ~inside


def _episode_horizon_array(
    episode_horizons: Sequence[int] | jax.Array | int | None,
    *,
    num_mazes: int,
    default_horizon: jax.Array,
) -> jax.Array:
    if episode_horizons is None:
        return jnp.full((num_mazes,), default_horizon, dtype=jnp.int32)

    horizons = jnp.asarray(episode_horizons, dtype=jnp.int32)
    if horizons.ndim == 0:
        return jnp.full((num_mazes,), horizons, dtype=jnp.int32)
    if horizons.shape != (num_mazes,):
        raise ValueError(
            f"episode_horizons must be scalar or have shape ({num_mazes},), "
            f"got {horizons.shape}"
        )
    return horizons
