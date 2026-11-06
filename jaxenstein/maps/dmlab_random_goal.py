"""Episode sampling for DeepMind Lab's nav_maze_random_goal levels."""

from __future__ import annotations

from collections import deque
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from jaxenstein.env import RayMazeEnv, State
from jaxenstein.maps.ascii import Maze, parse_ascii_maze
from jaxenstein.objects import NUM_KEY_COLORS


class DMLabRandomGoalEnv(RayMazeEnv):
    """Sample a goal, scattered apples, and a start more than eight cells away.

    Source: google-deepmind/lab game_scripts/factories/random_goal_factory.lua.
    The source .map's apples mark all possible goal/fruit locations; its single
    player start is relocated by Lua at every reset.
    """

    def __init__(self, maze: Maze, *, scattered_reward_density: float = 0.1, **kwargs):
        if not 0.0 <= scattered_reward_density <= 1.0:
            raise ValueError("scattered_reward_density must be between zero and one")
        if any(name != "apple" for name in maze.object_names):
            raise ValueError("random-goal maps must contain only apple candidates")
        self._candidate_count = len(maze.object_type)
        if self._candidate_count < 2:
            raise ValueError("random-goal maps require at least two candidates")
        self._fruit_count = int(
            scattered_reward_density * (self._candidate_count - 1) + 0.5
        )
        self.scattered_reward_density = float(scattered_reward_density)
        self._spawn_eligible = jnp.asarray(_eligible_starts(maze))
        if self._fruit_count == 0:
            # These slots can only be goals. Do not build/transfer an unused
            # fruit sprite or carry fruit properties into the compiled program.
            maze = replace(maze, object_names=("goal",) * self._candidate_count)
        possible = tuple(kwargs.pop("possible_object_types", ()))
        super().__init__(maze, possible_object_types=(*possible, "goal"), **kwargs)
        self._goal_type_id = self.object_catalog.id("goal")
        self._fruit_type_id = (
            self.object_catalog.id("apple") if self._fruit_count else self._goal_type_id
        )

    @classmethod
    def from_ascii(
        cls, ascii_maze: str, *, objects=(), object_symbols=None, **kwargs
    ) -> "DMLabRandomGoalEnv":
        return cls(
            parse_ascii_maze(
                ascii_maze,
                require_goal=False,
                objects=objects,
                object_symbols=object_symbols,
            ),
            **kwargs,
        )

    def _reset_state(self, key: jax.Array) -> State:
        goal_key, fruit_key, spawn_key, heading_key = jax.random.split(key, 4)
        goal_idx = jax.random.randint(goal_key, (), 0, self._candidate_count)
        if self._fruit_count:
            # Uniformly sample a fixed-size subset excluding the goal.
            order = jax.random.permutation(fruit_key, self._candidate_count)
            ranks = jnp.argsort(order)
            ranks = ranks - (ranks > ranks[goal_idx])
            fruit_mask = (ranks < self._fruit_count) & (
                jnp.arange(self._candidate_count) != goal_idx
            )
        else:
            fruit_mask = jnp.zeros((self._candidate_count,), dtype=jnp.bool_)
        eligible = self._spawn_eligible[goal_idx]
        spawn_rank = jax.random.randint(spawn_key, (), 0, jnp.sum(eligible))
        spawn_idx = jnp.argmax(eligible & (jnp.cumsum(eligible) == spawn_rank + 1))
        return State(
            pos=self.maze.object_xy[spawn_idx],
            theta=self._spawn_heading(heading_key),
            t=jnp.asarray(0, dtype=jnp.int32),
            done=jnp.asarray(False),
            goal_xy=self.maze.object_xy[goal_idx],
            object_active=fruit_mask | (jnp.arange(self._candidate_count) == goal_idx),
            carried_keys=jnp.zeros((NUM_KEY_COLORS,), dtype=jnp.bool_),
            door_open=jnp.zeros_like(self.maze.door_grid, dtype=jnp.bool_),
        )

    def object_type_ids(self, state: State) -> jax.Array:
        if not self._fruit_count:
            return self._initial_type_ids
        at_goal = jnp.all(self.maze.object_xy == state.goal_xy, axis=-1)
        return jnp.where(at_goal, self._goal_type_id, self._fruit_type_id)


def _eligible_starts(maze: Maze) -> np.ndarray:
    """Precompute graph distances once, outside the JIT-compiled reset path."""
    walls = np.asarray(maze.wall_grid)
    cells = [(int(y), int(x)) for x, y in np.asarray(maze.object_xy)]
    eligible = np.zeros((len(cells), len(cells)), dtype=bool)
    for goal_idx, goal in enumerate(cells):
        distances = {goal: 0}
        frontier = deque([goal])
        while frontier:
            row, col = frontier.popleft()
            for cell in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                r, c = cell
                if (
                    0 <= r < walls.shape[0]
                    and 0 <= c < walls.shape[1]
                    and not walls[r, c]
                    and cell not in distances
                ):
                    distances[cell] = distances[(row, col)] + 1
                    frontier.append(cell)
        if any(cell not in distances for cell in cells):
            raise ValueError("random-goal candidates must be connected")
        eligible[goal_idx] = [distances[cell] > 8 for cell in cells]
    if not np.all(eligible.any(axis=1)):
        raise ValueError("every goal needs a start more than eight cells away")
    return eligible
