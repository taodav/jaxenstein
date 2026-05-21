"""JAX-native first-person maze environment."""

from __future__ import annotations

from collections.abc import Sequence

from flax import struct
import jax
import jax.numpy as jnp

from jaxenstein.maps.ascii import Maze, parse_ascii_maze
from jaxenstein.objects import (
    DOOR_LOCKED_WALL_COLOR_OFFSET,
    KEY_COLOR_NONE,
    NUM_KEY_COLORS,
    OBJECT_GOAL,
    OBJECT_KEY,
    OBJECT_NONE,
    door_wall_color_id,
    object_pickup_radius,
)
from jaxenstein.render import (
    DEFAULT_WALL_PALETTE,
    FLOOR_CHECKER_DARK_RGB,
    FLOOR_CHECKER_LIGHT_RGB,
    FLOOR_RGB,
    render_first_person,
)


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
    goal_xy: jax.Array
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
        maze: Maze,
        *,
        params: EnvParams | None = None,
        episode_horizon: jax.Array | int | None = None,
        img_h: int = 64,
        img_w: int = 64,
        fov: float = jnp.pi / 3.0,
        max_depth: float = 16.0,
        num_depth_samples: int = 128,
        color_palette: Sequence[Sequence[float]] | jax.Array = DEFAULT_WALL_PALETTE,
        floor_rgb: Sequence[float] | jax.Array = FLOOR_RGB,
        floor_checker_dark_rgb: Sequence[float] | jax.Array = FLOOR_CHECKER_DARK_RGB,
        floor_checker_light_rgb: Sequence[float] | jax.Array = FLOOR_CHECKER_LIGHT_RGB,
        wall_height_scale: float = 1.35,
        floor_pattern: bool = True,
        goal_reward: float | jax.Array = 1.0,
        living_reward: float | jax.Array = 0.0,
        gamma: float = 0.99,
    ):
        self.maze = maze
        self.params = EnvParams() if params is None else params
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
        self.goal_reward = jnp.asarray(goal_reward, dtype=jnp.float32)
        self.living_reward = jnp.asarray(living_reward, dtype=jnp.float32)
        self.gamma = float(gamma)
        self.episode_horizon = _episode_horizon(
            episode_horizon,
            default_horizon=self.params.horizon,
        )

    @classmethod
    def from_ascii(cls, ascii_maze: str, **kwargs) -> "RayMazeEnv":
        return cls(parse_ascii_maze(ascii_maze), **kwargs)

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return (self.img_h, self.img_w, 3)

    def reset(
        self, key: jax.Array, params: EnvParams | None = None
    ) -> tuple[jax.Array, State]:
        del params
        spawn_key, goal_key = jax.random.split(key)
        spawn_idx = jax.random.randint(
            spawn_key,
            shape=(),
            minval=0,
            maxval=self.maze.spawn_count,
            dtype=jnp.int32,
        )
        object_type = self.maze.object_type
        goal_mask = object_type == OBJECT_GOAL
        goal_order = jnp.cumsum(goal_mask.astype(jnp.int32)) - 1
        goal_idx = jax.random.randint(
            goal_key,
            shape=(),
            minval=0,
            maxval=jnp.sum(goal_mask.astype(jnp.int32)),
            dtype=jnp.int32,
        )
        selected_goal = goal_mask & (goal_order == goal_idx)
        object_active = (object_type != OBJECT_NONE) & (
            (object_type != OBJECT_GOAL) | selected_goal
        )
        goal_xy = jnp.sum(
            jnp.where(
                selected_goal[:, None],
                self.maze.object_xy,
                jnp.asarray(0.0, dtype=jnp.float32),
            ),
            axis=0,
        )
        state = State(
            pos=self.maze.spawn_xy_options[spawn_idx],
            theta=self.maze.spawn_theta,
            t=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
            goal_xy=goal_xy,
            object_active=object_active,
            carried_keys=jnp.zeros((NUM_KEY_COLORS,), dtype=jnp.bool_),
            door_open=jnp.zeros_like(self.maze.door_grid, dtype=jnp.bool_),
        )
        return self.render(state), state

    def step(
        self,
        state: State,
        action: jax.Array | int,
        params: EnvParams | None = None,
    ) -> tuple[jax.Array, State, jax.Array, jax.Array, dict[str, jax.Array]]:
        params = self.params if params is None else params
        action = jnp.asarray(action, dtype=jnp.int32)
        active = ~state.done

        turn_delta = jnp.where(
            action == ACTION_TURN_LEFT,
            -params.turn_angle,
            jnp.where(action == ACTION_TURN_RIGHT, params.turn_angle, 0.0),
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
            move_direction * params.move_speed * forward,
            jnp.zeros((2,), dtype=jnp.float32),
        )

        wall_grid = _solid_grid(
            self.maze.wall_grid,
            self.maze.door_grid,
            state.door_open,
        )
        blocked = _is_wall_at(wall_grid, proposed_pos)
        pos = jnp.where(blocked, state.pos, proposed_pos)

        object_xy = self.maze.object_xy
        object_type = self.maze.object_type
        object_color = self.maze.object_color
        object_distances = jnp.linalg.norm(object_xy - pos[None, :], axis=-1)
        pickup_radius = jnp.where(
            object_type == OBJECT_GOAL,
            params.goal_radius,
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
            self.maze.door_grid,
            state.door_open,
            pos,
            theta,
            carried_keys,
            (action == ACTION_INTERACT) & active,
            params.interact_distance,
        )

        distance_to_goal = jnp.linalg.norm(pos - state.goal_xy)
        t = state.t + jnp.where(active, 1, 0).astype(jnp.int32)
        done = state.done | reached_goal | (t >= self.episode_horizon)
        reward = jnp.where(
            active,
            self.living_reward + jnp.where(reached_goal, self.goal_reward, 0.0),
            0.0,
        ).astype(jnp.float32)

        new_state = State(
            pos=pos,
            theta=theta,
            t=t,
            done=done,
            goal_xy=state.goal_xy,
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
        }
        return self.render(new_state), new_state, reward, done, info

    def render(self, state: State) -> jax.Array:
        return render_first_person(
            state.pos,
            state.theta,
            _solid_grid(
                self.maze.wall_grid,
                self.maze.door_grid,
                state.door_open,
            ),
            _render_color_grid(
                self.maze.color_grid,
                self.maze.door_grid,
                state.door_open,
            ),
            img_h=self.img_h,
            img_w=self.img_w,
            fov=self.fov,
            max_depth=self.max_depth,
            num_depth_samples=self.num_depth_samples,
            color_palette=self.color_palette,
            floor_rgb=self.floor_rgb,
            floor_checker_dark_rgb=self.floor_checker_dark_rgb,
            floor_checker_light_rgb=self.floor_checker_light_rgb,
            object_xy=self.maze.object_xy,
            object_type=self.maze.object_type,
            object_color=self.maze.object_color,
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
    door_color = door_wall_color_id(door_grid)
    locked_door = door_grid > KEY_COLOR_NONE
    door_color = jnp.where(
        locked_door,
        door_color + DOOR_LOCKED_WALL_COLOR_OFFSET,
        door_color,
    )
    return jnp.where(closed, door_color, color_grid)


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
    is_unlocked_door = door_color < KEY_COLOR_NONE
    door_key_color = jnp.maximum(door_color, KEY_COLOR_NONE)
    has_key = jnp.take(carried_keys, door_key_color, mode="clip")
    opened_door = (
        should_interact
        & inside
        & (door_color != KEY_COLOR_NONE)
        & ~door_open[clipped_y, clipped_x]
        & (is_unlocked_door | has_key)
    )

    new_cell_open = door_open[clipped_y, clipped_x] | opened_door
    new_door_open = door_open.at[clipped_y, clipped_x].set(new_cell_open)
    return new_door_open, opened_door, jnp.where(opened_door, door_color, KEY_COLOR_NONE)


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
