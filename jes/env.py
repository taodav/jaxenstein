"""JAX-native first-person maze environment."""

from __future__ import annotations

from collections.abc import Sequence

from flax import struct
import jax
import jax.numpy as jnp

from jes.ascii import Maze, MazeBatch, parse_and_stack, stack_mazes
from jes.objects import (
    KEY_COLOR_NONE,
    NUM_KEY_COLORS,
    OBJECT_GOAL,
    OBJECT_KEY,
    OBJECT_NONE,
    door_wall_color_id,
    object_pickup_radius,
)
from jes.render import render_first_person


ACTION_TURN_LEFT = 0
ACTION_TURN_RIGHT = 1
ACTION_MOVE_FORWARD = 2
ACTION_MOVE_BACKWARD = 3
ACTION_INTERACT = 4
NUM_ACTIONS = 5


@struct.dataclass
class State:
    pos: jax.Array
    theta: jax.Array
    t: jax.Array
    done: jax.Array
    maze_id: jax.Array
    object_active: jax.Array
    carried_keys: jax.Array
    door_open: jax.Array


@struct.dataclass
class EnvParams:
    turn_angle: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(jnp.pi / 12.0, dtype=jnp.float32)
    )
    move_speed: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(0.15, dtype=jnp.float32)
    )
    goal_radius: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(0.35, dtype=jnp.float32)
    )
    horizon: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(300, dtype=jnp.int32)
    )
    interact_distance: jax.Array = struct.field(
        default_factory=lambda: jnp.asarray(0.9, dtype=jnp.float32)
    )


class RayMazeEnv:
    """A small stationary-action first-person maze environment."""

    def __init__(
        self,
        maze_batch: MazeBatch,
        *,
        params: EnvParams | None = None,
        episode_horizons: Sequence[int] | jax.Array | int | None = None,
        img_h: int = 64,
        img_w: int = 64,
        fov: float = jnp.pi / 3.0,
        max_depth: float = 16.0,
        num_depth_samples: int = 128,
        wall_height_scale: float = 1.0,
        floor_pattern: bool = False,
    ):
        self.maze_batch = maze_batch
        self.params = EnvParams() if params is None else params
        self.img_h = img_h
        self.img_w = img_w
        self.fov = fov
        self.max_depth = max_depth
        self.num_depth_samples = num_depth_samples
        self.wall_height_scale = wall_height_scale
        self.floor_pattern = floor_pattern
        self.episode_horizons = _episode_horizon_array(
            episode_horizons,
            num_mazes=int(maze_batch.wall_grids.shape[0]),
            default_horizon=self.params.horizon,
        )

    @classmethod
    def from_mazes(cls, mazes: Sequence[Maze], **kwargs) -> "RayMazeEnv":
        return cls(stack_mazes(list(mazes)), **kwargs)

    @classmethod
    def from_ascii(cls, ascii_mazes: Sequence[str], **kwargs) -> "RayMazeEnv":
        return cls(parse_and_stack(list(ascii_mazes)), **kwargs)

    @property
    def num_mazes(self) -> int:
        return int(self.maze_batch.wall_grids.shape[0])

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.img_h, self.img_w, 3)

    def reset(self, key: jax.Array, maze_id: jax.Array | int = 0) -> tuple[jax.Array, State]:
        maze_id = jnp.asarray(maze_id, dtype=jnp.int32)
        spawn_idx = jax.random.randint(
            key,
            shape=(),
            minval=0,
            maxval=self.maze_batch.spawn_count[maze_id],
            dtype=jnp.int32,
        )
        state = State(
            pos=self.maze_batch.spawn_xy_options[maze_id, spawn_idx],
            theta=self.maze_batch.spawn_theta[maze_id],
            t=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
            maze_id=maze_id,
            object_active=self.maze_batch.object_type[maze_id] != OBJECT_NONE,
            carried_keys=jnp.zeros((NUM_KEY_COLORS,), dtype=jnp.bool_),
            door_open=jnp.zeros_like(self.maze_batch.door_grids[maze_id], dtype=jnp.bool_),
        )
        return self.render(state), state

    def step(
        self, state: State, action: jax.Array | int
    ) -> tuple[jax.Array, State, jax.Array, jax.Array, dict[str, jax.Array]]:
        action = jnp.asarray(action, dtype=jnp.int32)
        active = ~state.done

        turn_delta = jnp.where(
            action == ACTION_TURN_LEFT,
            -self.params.turn_angle,
            jnp.where(action == ACTION_TURN_RIGHT, self.params.turn_angle, 0.0),
        )
        theta = jnp.mod(state.theta + jnp.where(active, turn_delta, 0.0), 2.0 * jnp.pi)

        forward = jnp.asarray([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.float32)
        move_direction = jnp.where(
            action == ACTION_MOVE_FORWARD,
            1.0,
            jnp.where(action == ACTION_MOVE_BACKWARD, -1.0, 0.0),
        )
        proposed_pos = state.pos + jnp.where(
            active,
            move_direction * self.params.move_speed * forward,
            jnp.zeros((2,), dtype=jnp.float32),
        )

        wall_grid = _solid_grid(
            self.maze_batch.wall_grids[state.maze_id],
            self.maze_batch.door_grids[state.maze_id],
            state.door_open,
        )
        blocked = _is_wall_at(wall_grid, proposed_pos)
        pos = jnp.where(blocked, state.pos, proposed_pos)

        object_xy = self.maze_batch.object_xy[state.maze_id]
        object_type = self.maze_batch.object_type[state.maze_id]
        object_color = self.maze_batch.object_color[state.maze_id]
        object_distances = jnp.linalg.norm(object_xy - pos[None, :], axis=-1)
        pickup_radius = jnp.where(
            object_type == OBJECT_GOAL,
            self.params.goal_radius,
            object_pickup_radius(object_type),
        )
        picked_objects = (
            active
            & state.object_active
            & (object_type != OBJECT_NONE)
            & (object_distances < pickup_radius)
        )
        object_active = state.object_active & ~picked_objects
        key_colors = jnp.arange(NUM_KEY_COLORS, dtype=jnp.int32)
        picked_key_colors = jnp.where(
            picked_objects & (object_type == OBJECT_KEY), object_color, KEY_COLOR_NONE
        )
        picked_key_mask = jnp.any(
            picked_key_colors[:, None] == key_colors[None, :], axis=0
        ) & (key_colors != KEY_COLOR_NONE)
        carried_keys = state.carried_keys | picked_key_mask
        reached_goal = jnp.any(picked_objects & (object_type == OBJECT_GOAL))
        door_open, opened_door, opened_door_color = _interact_with_door(
            self.maze_batch.door_grids[state.maze_id],
            state.door_open,
            pos,
            theta,
            carried_keys,
            (action == ACTION_INTERACT) & active,
            self.params.interact_distance,
        )

        goal_xy = self.maze_batch.goal_xy[state.maze_id]
        distance_to_goal = jnp.linalg.norm(pos - goal_xy)
        t = state.t + jnp.where(active, 1, 0).astype(jnp.int32)
        episode_horizon = self.episode_horizons[state.maze_id]
        done = state.done | reached_goal | (t >= episode_horizon)
        reward = jnp.where(active & reached_goal, 1.0, 0.0).astype(jnp.float32)

        new_state = State(
            pos=pos,
            theta=theta,
            t=t,
            done=done,
            maze_id=state.maze_id,
            object_active=object_active,
            carried_keys=carried_keys,
            door_open=door_open,
        )
        info = {
            "x": pos[0],
            "y": pos[1],
            "theta": theta,
            "cell_x": jnp.floor(pos[0]).astype(jnp.int32),
            "cell_y": jnp.floor(pos[1]).astype(jnp.int32),
            "reached_goal": reached_goal,
            "picked_object_type": jnp.max(jnp.where(picked_objects, object_type, 0)),
            "picked_object_color": jnp.max(jnp.where(picked_objects, object_color, 0)),
            "carried_keys": carried_keys,
            "opened_door": opened_door,
            "opened_door_color": opened_door_color,
            "distance_to_goal": distance_to_goal,
            "maze_id": state.maze_id,
        }
        return self.render(new_state), new_state, reward, done, info

    def render(self, state: State) -> jax.Array:
        return render_first_person(
            state.pos,
            state.theta,
            _solid_grid(
                self.maze_batch.wall_grids[state.maze_id],
                self.maze_batch.door_grids[state.maze_id],
                state.door_open,
            ),
            _render_color_grid(
                self.maze_batch.color_grids[state.maze_id],
                self.maze_batch.door_grids[state.maze_id],
                state.door_open,
            ),
            img_h=self.img_h,
            img_w=self.img_w,
            fov=self.fov,
            max_depth=self.max_depth,
            num_depth_samples=self.num_depth_samples,
            object_xy=self.maze_batch.object_xy[state.maze_id],
            object_type=self.maze_batch.object_type[state.maze_id],
            object_color=self.maze_batch.object_color[state.maze_id],
            object_active=state.object_active,
            wall_height_scale=self.wall_height_scale,
            floor_pattern=self.floor_pattern,
        )


def _is_wall_at(wall_grid: jax.Array, xy: jax.Array) -> jax.Array:
    h, w = wall_grid.shape
    cell_x = jnp.floor(xy[0]).astype(jnp.int32)
    cell_y = jnp.floor(xy[1]).astype(jnp.int32)
    inside = (cell_x >= 0) & (cell_x < w) & (cell_y >= 0) & (cell_y < h)
    clipped_x = jnp.clip(cell_x, 0, w - 1)
    clipped_y = jnp.clip(cell_y, 0, h - 1)
    return wall_grid[clipped_y, clipped_x] | ~inside


def _closed_door_mask(door_grid: jax.Array, door_open: jax.Array) -> jax.Array:
    return (door_grid != KEY_COLOR_NONE) & ~door_open


def _solid_grid(
    wall_grid: jax.Array, door_grid: jax.Array, door_open: jax.Array
) -> jax.Array:
    return wall_grid | _closed_door_mask(door_grid, door_open)


def _render_color_grid(
    color_grid: jax.Array, door_grid: jax.Array, door_open: jax.Array
) -> jax.Array:
    closed = _closed_door_mask(door_grid, door_open)
    return jnp.where(closed, door_wall_color_id(door_grid), color_grid)


def _interact_with_door(
    door_grid: jax.Array,
    door_open: jax.Array,
    pos: jax.Array,
    theta: jax.Array,
    carried_keys: jax.Array,
    should_interact: jax.Array,
    interact_distance: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    h, w = door_grid.shape
    forward = jnp.asarray([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.float32)
    target = pos + interact_distance * forward
    cell_x = jnp.floor(target[0]).astype(jnp.int32)
    cell_y = jnp.floor(target[1]).astype(jnp.int32)
    inside = (cell_x >= 0) & (cell_x < w) & (cell_y >= 0) & (cell_y < h)
    clipped_x = jnp.clip(cell_x, 0, w - 1)
    clipped_y = jnp.clip(cell_y, 0, h - 1)

    door_color = door_grid[clipped_y, clipped_x]
    has_key = jnp.take(carried_keys, door_color, mode="clip")
    opened_door = (
        should_interact
        & inside
        & (door_color != KEY_COLOR_NONE)
        & ~door_open[clipped_y, clipped_x]
        & has_key
    )

    new_cell_open = door_open[clipped_y, clipped_x] | opened_door
    new_door_open = door_open.at[clipped_y, clipped_x].set(new_cell_open)
    return new_door_open, opened_door, jnp.where(opened_door, door_color, KEY_COLOR_NONE)


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
