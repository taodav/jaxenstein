# ASCII Maze Guide

This document describes how to define a custom Jaxenstein maze and initialize a `RayMazeEnv` from it.

## Minimal Map

```python
from jes import RayMazeEnv

MY_MAZE = """
#########
#S......#
#.#####.#
#......G#
#########
"""

env = RayMazeEnv.from_ascii([MY_MAZE])
```

Every maze must contain at least one spawn and at least one goal. If a map has
multiple `S` or `G` cells, `RayMazeEnv.reset` samples one spawn and one active
goal uniformly from those candidates using the reset key.

## Coordinates

ASCII rows and columns map to continuous `x, y` positions. Cell centers are:

```python
x = col + 0.5
y = row + 0.5
```

The parser converts `S`, `G`, keys, doors, `.`, and spaces into open floor.
`#` and colored wall symbols remain static walls. Rows may be ragged; shorter
rows are padded with walls when maps are stacked into a batch.

## Symbol Table

| Symbol | Meaning |
| --- | --- |
| `#` | default static wall |
| `1`-`9`, generated DMLab symbols | colored static wall |
| `.` | floor |
| space | floor |
| `S` | spawn candidate |
| `G` | yellow goal candidate |
| `K` | red key |
| `r` | red key |
| `b` | blue key |
| `y` | yellow key |
| `"` | blue unlocked door |
| `\` | yellow unlocked door |
| `D` | red-locked door |
| `R` | red-locked door |
| `B` | blue-locked door |
| `Y` | yellow-locked door |

## Map Semantics

Static walls are impassable forever. Closed doors are also impassable, but
`ACTION_INTERACT` can open the door in front of the agent. Unlocked doors
(`"`, `\`) open without a key. Locked doors open only when the carried key
color matches the door's encoded color. Closed doors render as colored panels:
unlocked doors use their cue color, and locked doors use the color of the
required key.

Keys are billboard pickup objects. Walking near a key picks it up and records
the color in `state.carried_keys`. Goals are billboard pickup objects too:
walking near the active goal gives reward `1.0` and ends the episode. If a map
contains multiple `G` symbols, reset samples exactly one active goal candidate.

Objects are stored in row-major ASCII order. Padded object slots use
`OBJECT_NONE` when maps with different object counts are batched together.

## ViZDoom Health Gathering

`HealthGatheringEnv` implements the regular ViZDoom Health Gathering scenario
as a separate environment class. Its state includes `health`, `medkit_xy`,
`medkit_active`, and medkit spawn bookkeeping instead of adding those fields to
the navigation `State`.

The bundled ViZDoom scenario uses a rectangular 1216x1216 Doom-unit room with
gray `GSTONE1` walls, a green `NUKAGE1` acidic floor, the player at the center,
16 initial medkits, one new medkit every 30 tics, +1 living reward, 100 death
penalty, and 2100 tic timeout. Jaxenstein mirrors those values in
`MAZE_HEALTH_GATHERING` and `HealthGatheringEnv`.

```python
import jax

from jes import ACTION_MOVE_FORWARD, HealthGatheringEnv
from jes.maps import MAZE_HEALTH_GATHERING

env = HealthGatheringEnv.from_ascii([MAZE_HEALTH_GATHERING])
obs, state = env.reset(jax.random.key(0))
obs, state, reward, done, info = env.step(state, ACTION_MOVE_FORWARD)
```

Interactive play:

```bash
uv run python play.py --maze health-gathering
```

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
| `ACTION_INTERACT` | `4` | open the door in front of the agent if it is unlocked or the matching key is carried |

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

## MiniGrid KeyCorridor

`jes.maps.MAZE_KEY_CORRIDOR` is a fixed MiniGrid
`MiniGrid-KeyCorridorS4R3-v0`-style map. It uses a 3-by-3 room grid with
2-by-2 room interiors, a connected middle hallway, colored unlocked doors to
side rooms, one red key, and one red-locked door guarding the goal. This mirrors
the source structure: the middle column is connected as a hallway, a locked
door is placed on the left wall of a right-column room, the target is behind
that door, the matching key is placed in a left-column room, and extra unlocked
doors connect the remaining rooms. MiniGrid's default `max_steps` for this
configuration is
`30 * room_size**2 = 480`, mirrored by `MAP_EPISODE_HORIZONS_BY_NAME`.

```bash
uv run python play.py --maze key-corridor
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
entities as `G` candidates, with one active goal sampled at reset.

The DMLab Lua levels set `episodeLengthSeconds` to 60, 150, and 300 for
indices 01, 02, and 03. Using a Doom-like 30 Hz step budget, Jaxenstein maps
those to horizons of 1800, 4500, and 9000 for the static and random-goal
variants.

```bash
uv run python scripts/convert_dmlab_map.py path/to/nav_maze_static_01.map
uv run python play.py --maze dmlab-static-01
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
obs, state = env.reset(jax.random.key(0))

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
| `info["opened_door_color"]` | opened door color; negative values are unlocked colored doors, positive values are key-locked doors, and `0` means no door opened |

## Batching

`RayMazeEnv.from_ascii([...])` accepts multiple maps. Maps may have different sizes and different object counts; smaller maps are padded with walls and missing objects are padded with `OBJECT_NONE`.

```python
import jax
import jax.numpy as jnp

from jes import ACTION_MOVE_FORWARD, RayMazeEnv
from jes.maps import MAZE_KEY_DOOR, MAZE_SIMPLE

env = RayMazeEnv.from_ascii([MAZE_SIMPLE, MAZE_KEY_DOOR])

keys = jax.random.split(jax.random.key(0), 2)
obs, states = jax.vmap(env.reset)(keys)

actions = jnp.asarray([ACTION_MOVE_FORWARD, ACTION_MOVE_FORWARD], dtype=jnp.int32)
obs, states, reward, done, info = jax.vmap(env.step)(states, actions)
```
