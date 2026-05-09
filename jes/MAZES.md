# ASCII Maze Guide

This document describes how to define a custom Jaxenstein maze and initialize a `RayMazeEnv` from it.

## Minimal Map

```python
from jes import RayMazeEnv

MY_MAZE = """
#########
#S.....G#
#.#####.#
#.......#
#########
"""

env = RayMazeEnv.from_ascii([MY_MAZE])
```

Every maze must contain at least one spawn and at least one goal. If a map has
multiple `S` cells, `RayMazeEnv.reset` samples uniformly from them using the
reset key.

## Coordinates

ASCII rows and columns map to continuous `x, y` positions. Cell centers are:

```python
x = col + 0.5
y = row + 0.5
```

The parser converts `S`, `G`, keys, doors, `.`, and spaces into open floor.
`#` and colored wall symbols remain static walls.

## Symbol Table

| Symbol | Meaning |
| --- | --- |
| `#` | default static wall |
| `1`-`9`, generated symbols | colored static wall |
| `.` | floor |
| space | floor |
| `S` | spawn candidate |
| `G` | yellow goal object |
| `K` | red key |
| `r` | red key |
| `b` | blue key |
| `y` | yellow key |
| `D` | red door |
| `R` | red door |
| `B` | blue door |
| `Y` | yellow door |

Keys are billboard pickup objects. Doors are grid cells: closed doors block movement and render as paneled colored wall slices; open doors become passable.

## Actions

```python
from jes import (
    ACTION_INTERACT,
    ACTION_MOVE_BACKWARD,
    ACTION_MOVE_FORWARD,
    ACTION_TURN_LEFT,
    ACTION_TURN_RIGHT,
)
```

| Constant | Value | Behavior |
| --- | ---: | --- |
| `ACTION_TURN_LEFT` | `0` | rotate left |
| `ACTION_TURN_RIGHT` | `1` | rotate right |
| `ACTION_MOVE_FORWARD` | `2` | move forward with collision |
| `ACTION_MOVE_BACKWARD` | `3` | move backward with collision |
| `ACTION_INTERACT` | `4` | open the door in front of the agent if the matching key is carried |

## Key-Door Example

```python
KEY_DOOR_MAZE = """
###########
#S.r.R...G#
###########
"""
```

In this map, the agent starts at `S`, picks up the red key `r`, interacts with the red door `R`, and then can reach the goal `G`.

Interactive play:

```bash
uv run python play.py --maze key-door
```

## ViZDoom My Way Home

`jes.maps.MAZE_MY_WAY_HOME` is a rasterized version of ViZDoom's bundled My Way Home map. The footprint is converted from the UDMF WAD at 32 Doom units per ASCII cell, with the green vest represented by `G` and the original 17 spawn map points represented by `S`.

The default `my-way-home` variant uses digit wall symbols to approximate the original WAD texture colors. `my-way-home-colorless` uses the same topology with default `#` walls. The renderer uses a checker floor pattern and taller walls by default for all maps.

ViZDoom's bundled `my_way_home.cfg` sets `episode_timeout = 2100`. Jaxenstein mirrors that in `MAP_EPISODE_HORIZONS_BY_NAME`, and `play.py` passes that horizon automatically.

```bash
uv run python play.py --maze my-way-home
uv run python play.py --maze my-way-home-colorless
```

## DeepMind Lab Nav Mazes

`jes.dmlab.dmlab_map_to_ascii` converts grid-aligned DMLab `nav_maze` `.map`
files into ASCII maps. `info_player_start` entities become `S`, explicit
`goal` entities become `G`, and decal image walls become stable colored wall
symbols instead of textured patches. Random-goal maps use `apple_reward`
entities as goals.

```bash
uv run python scripts/convert_dmlab_map.py path/to/nav_maze_static_01.map
uv run python play.py --maze dmlab-nav-maze-static-01
```

## Custom Map Example

```python
import jax
import jax.numpy as jnp

from jes import ACTION_INTERACT, ACTION_MOVE_FORWARD, RayMazeEnv

CUSTOM = """
#############
#S...b.B...G#
#.#########.#
#...........#
#############
"""

env = RayMazeEnv.from_ascii([CUSTOM])
obs, state = env.reset(jax.random.key(0), jnp.asarray(0, dtype=jnp.int32))

obs, state, reward, done, info = env.step(state, ACTION_MOVE_FORWARD)
obs, state, reward, done, info = env.step(state, ACTION_INTERACT)
```

Useful state and info fields:

| Field | Meaning |
| --- | --- |
| `state.pos` | continuous `x, y` position |
| `state.theta` | heading in radians |
| `state.object_active` | active/inactive mask for pickup objects |
| `state.carried_keys` | boolean inventory indexed by key color |
| `state.door_open` | boolean grid showing opened doors |
| `info["picked_object_type"]` | object type picked up on this step, or `0` |
| `info["picked_object_color"]` | object color picked up on this step, or `0` |
| `info["opened_door"]` | whether interact opened a door |
| `info["opened_door_color"]` | door color opened on this step, or `0` |

## Batching

`RayMazeEnv.from_ascii([...])` accepts multiple maps. Maps may have different sizes and different object counts; smaller maps are padded with walls and missing objects are padded with `OBJECT_NONE`.

```python
import jax
import jax.numpy as jnp

from jes import ACTION_MOVE_FORWARD, RayMazeEnv
from jes.maps import MAZE_KEY_DOOR, MAZE_SIMPLE

env = RayMazeEnv.from_ascii([MAZE_SIMPLE, MAZE_KEY_DOOR])

keys = jax.random.split(jax.random.key(0), 2)
maze_ids = jnp.asarray([0, 1], dtype=jnp.int32)
obs, states = jax.vmap(env.reset)(keys, maze_ids)

actions = jnp.asarray([ACTION_MOVE_FORWARD, ACTION_MOVE_FORWARD], dtype=jnp.int32)
obs, states, reward, done, info = jax.vmap(env.step)(states, actions)
```
