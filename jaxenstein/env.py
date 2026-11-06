"""JAX-native first-person maze environment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import jaxenstein._struct as struct
from jaxenstein.maps.ascii import Maze, parse_ascii_maze
from jaxenstein.objects import (
    DOOR_LOCKED_WALL_COLOR_OFFSET,
    KEY_COLOR_NONE,
    KEY_COLOR_YELLOW,
    NUM_KEY_COLORS,
    OBJECT_GOAL,
    OBJECT_KEY,
    OBJECT_NONE,
    ObjectPlacement,
    ObjectSpec,
    compile_object_catalog,
    door_wall_color_id,
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
        randomize_spawn_heading: bool = False,
        object_types: Mapping[str, ObjectSpec] | None = None,
        possible_object_types: Sequence[str] = (),
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
        self.randomize_spawn_heading = bool(randomize_spawn_heading)
        names = maze.object_names
        if not names and len(maze.object_type):
            # Compatibility for externally constructed Maze values.
            key_names = {1: "red_key", 2: "blue_key", 3: "yellow_key"}
            names = tuple(
                "goal" if int(kind) == OBJECT_GOAL else key_names[int(color)]
                for kind, color in zip(
                    np.asarray(maze.object_type), np.asarray(maze.object_color)
                )
            )
        self.object_catalog = compile_object_catalog(
            (*names, *possible_object_types),
            object_types,
            goal_reward=float(goal_reward),
        )
        self._initial_type_ids = jnp.asarray(
            [self.object_catalog.id(name) for name in names], dtype=jnp.int32
        )
        self._initial_active = (
            jnp.ones((len(names),), dtype=jnp.bool_)
            if maze.object_initial_active is None
            else maze.object_initial_active
        )
        self._has_objects = bool(names)
        self._has_doors = bool(np.any(np.asarray(maze.door_grid)))
        self._goal_candidates = (
            self.object_catalog.is_goal[self._initial_type_ids]
            if names
            else jnp.zeros((0,), dtype=jnp.bool_)
        )
        self._has_goal_candidates = bool(np.any(np.asarray(self._goal_candidates)))
        self._param_goal_radius = jnp.asarray(
            [
                name == "goal" and (object_types is None or name not in object_types)
                for name in self.object_catalog.names
            ]
        )
        self._uses_param_goal_radius = bool(np.any(np.asarray(self._param_goal_radius)))
        self._legacy_types = jnp.where(
            self.object_catalog.is_goal,
            OBJECT_GOAL,
            jnp.where(self.object_catalog.inventory_keys != 0, OBJECT_KEY, OBJECT_NONE),
        )
        self._legacy_colors = jnp.where(
            self.object_catalog.is_goal,
            KEY_COLOR_YELLOW,
            self.object_catalog.inventory_keys,
        )
        self.episode_horizon = _episode_horizon(
            episode_horizon,
            default_horizon=self.params.horizon,
        )

    @classmethod
    def from_ascii(
        cls,
        ascii_maze: str,
        *,
        objects: Sequence[ObjectPlacement] = (),
        object_symbols: Mapping[str, str] | None = None,
        **kwargs,
    ) -> "RayMazeEnv":
        return cls(
            parse_ascii_maze(
                ascii_maze,
                objects=objects,
                object_symbols=object_symbols,
                require_goal=False,
            ),
            **kwargs,
        )

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
        state = self._reset_state(key)
        return self.render(state), state

    def _spawn_heading(self, key: jax.Array) -> jax.Array:
        if self.randomize_spawn_heading:
            return jax.random.uniform(key, (), maxval=2.0 * jnp.pi)
        return self.maze.spawn_theta

    def object_type_ids(self, state: State) -> jax.Array:
        """Environment-local catalog IDs, one per object slot."""
        return self._initial_type_ids

    def _object_types(self, state: State) -> jax.Array:
        return self._legacy_types[self.object_type_ids(state)]

    def _object_colors(self, state: State) -> jax.Array:
        return self._legacy_colors[self.object_type_ids(state)]

    def _reset_state(self, key: jax.Array) -> State:
        spawn_key, goal_key = jax.random.split(key)
        spawn_idx = jax.random.randint(
            spawn_key,
            shape=(),
            minval=0,
            maxval=self.maze.spawn_count,
            dtype=jnp.int32,
        )
        if self._has_goal_candidates:
            goal_mask = self._goal_candidates & self._initial_active
            goal_order = jnp.cumsum(goal_mask.astype(jnp.int32)) - 1
            goal_idx = jax.random.randint(
                goal_key, (), 0, jnp.maximum(jnp.sum(goal_mask), 1)
            )
            selected_goal = goal_mask & (goal_order == goal_idx)
            object_active = self._initial_active & (
                ~self._goal_candidates | selected_goal
            )
            goal_xy = jnp.sum(
                jnp.where(selected_goal[:, None], self.maze.object_xy, 0.0), axis=0
            )
        else:
            object_active = self._initial_active
            goal_xy = jnp.zeros((2,), dtype=jnp.float32)
        return State(
            pos=self.maze.spawn_xy_options[spawn_idx],
            theta=self._spawn_heading(jax.random.fold_in(key, 2)),
            t=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
            goal_xy=goal_xy,
            object_active=object_active,
            carried_keys=jnp.zeros((NUM_KEY_COLORS,), dtype=jnp.bool_),
            door_open=jnp.zeros_like(self.maze.door_grid, dtype=jnp.bool_),
        )

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

        catalog = self.object_catalog
        type_ids = self.object_type_ids(state)
        if self._has_objects:
            offsets = pos[None, :] - self.maze.object_xy
            if catalog.has_radius_pickups:
                radii = catalog.pickup_radii[type_ids]
                if self._uses_param_goal_radius:
                    radii = jnp.where(
                        self._param_goal_radius[type_ids], params.goal_radius, radii
                    )
                in_range = jnp.linalg.norm(offsets, axis=-1) < radii
            if catalog.has_box_pickups:
                in_box = jnp.all(
                    (offsets >= catalog.pickup_mins[type_ids])
                    & (offsets <= catalog.pickup_maxs[type_ids]),
                    axis=-1,
                )
                in_range = (
                    jnp.where(catalog.box_pickups[type_ids], in_box, in_range)
                    if catalog.has_radius_pickups
                    else in_box
                )
            picked_objects = active & state.object_active & in_range
            object_active = state.object_active & ~(
                picked_objects & catalog.consume[type_ids]
            )
            object_reward = jnp.sum(
                jnp.where(picked_objects, catalog.rewards[type_ids], 0.0)
            )
            terminal_pickup = jnp.any(picked_objects & catalog.terminal[type_ids])
            reached_goal = jnp.any(picked_objects & catalog.is_goal[type_ids])
            picked_type_id = jnp.max(
                jnp.where(picked_objects, type_ids, -1), initial=-1
            )
            picked_object_type = jnp.max(
                jnp.where(picked_objects, self._legacy_types[type_ids], 0), initial=0
            )
            picked_object_color = jnp.max(
                jnp.where(picked_objects, self._legacy_colors[type_ids], 0), initial=0
            )
        else:
            picked_objects = jnp.zeros((0,), dtype=jnp.bool_)
            object_active = state.object_active
            object_reward = jnp.asarray(0.0, dtype=jnp.float32)
            terminal_pickup = reached_goal = jnp.asarray(False)
            picked_type_id = jnp.asarray(-1, dtype=jnp.int32)
            picked_object_type = picked_object_color = jnp.asarray(0, dtype=jnp.int32)
        carried_keys = state.carried_keys
        if catalog.has_keys:
            key_colors = jnp.arange(NUM_KEY_COLORS, dtype=jnp.int32)
            picked_keys = jnp.where(
                picked_objects, catalog.inventory_keys[type_ids], KEY_COLOR_NONE
            )
            key_mask = jnp.any(picked_keys[:, None] == key_colors[None, :], axis=0)
            carried_keys = carried_keys | (key_mask & (key_colors != KEY_COLOR_NONE))
        if self._has_doors:
            door_open, opened_door, opened_door_color = _interact_with_door(
                self.maze.door_grid,
                state.door_open,
                pos,
                theta,
                carried_keys,
                (action == ACTION_INTERACT) & active,
                params.interact_distance,
            )
        else:
            door_open = state.door_open
            opened_door = jnp.asarray(False)
            opened_door_color = jnp.asarray(KEY_COLOR_NONE, dtype=jnp.int32)

        distance_to_goal = jnp.linalg.norm(pos - state.goal_xy)
        t = state.t + jnp.where(active, 1, 0).astype(jnp.int32)
        done = state.done | terminal_pickup | (t >= self.episode_horizon)
        reward = jnp.where(
            active,
            self.living_reward + object_reward,
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
            "picked_objects": picked_objects,
            "picked_type_id": picked_type_id,
            "picked_object_type": picked_object_type,
            "picked_object_color": picked_object_color,
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
            object_sprite=self.object_type_ids(state),
            sprite_atlas=self.object_catalog.sprites,
            sprite_width_scales=self.object_catalog.width_scales,
            sprite_height_scales=self.object_catalog.height_scales,
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
    return (
        new_door_open,
        opened_door,
        jnp.where(opened_door, door_color, KEY_COLOR_NONE),
    )


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
