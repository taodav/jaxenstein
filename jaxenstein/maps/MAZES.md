# ASCII Maze Guide

This document describes how to define a custom JAXenstein maze and initialize a `RayMazeEnv` from it.

## Minimal Map

```python
from jaxenstein import RayMazeEnv

MY_MAZE = """
#########
#S......#
#.#####.#
#......G#
#########
"""

env = RayMazeEnv.from_ascii(MY_MAZE)
```

Every maze must contain at least one spawn. If a map has multiple `S` or `G`
cells, `RayMazeEnv.reset` samples one spawn and one active goal uniformly from
those candidates using the reset key. Goals are optional for tasks that end
through another object effect or the time limit.

## Coordinates

ASCII rows and columns map to continuous `x, y` positions. Cell centers are:

```python
x = col + 0.5
y = row + 0.5
```

The parser converts `S`, `G`, keys, doors, `.`, and spaces into open floor.
`#` and colored wall symbols remain static walls. Rows may be ragged; shorter
rows are padded with walls.

## Symbol Table

| Symbol | Meaning |
| --- | --- |
| `#` | default static wall |
| `1`-`9`, generated DMLab symbols | colored static wall |
| `.` | floor |
| space | floor |
| `S` | spawn candidate |
| `G` | yellow goal candidate |
| `r` | red key |
| `b` | blue key |
| `y` | yellow key |
| `"` | blue unlocked door |
| `\` | yellow unlocked door |
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

Objects are stored in row-major ASCII order.

## Environment-defined objects

Each environment has its own object catalog. Define types with `ObjectSpec`,
then place any number of instances with `ObjectPlacement`. Positions are world
(x, y) coordinates; cells are centered at (column + 0.5, row + 0.5).
Object placements are separate from geometry, so multiple objects can share a
cell or a spawn. `G` and colored-key symbols remain convenience defaults.

```python
import numpy as np
from jaxenstein import ObjectSpec, ObjectPlacement, RayMazeEnv

# Or load a PNG as uint8 RGBA. Alpha uses cutouts: 0 or 255.
def gem_sprite():
    texture = np.zeros((16, 16, 4), dtype=np.uint8)
    texture[3:13, 3:13] = [40, 200, 255, 255]
    return texture

env = RayMazeEnv.from_ascii(
    "#######\n#S...G#\n#######",
    objects=[ObjectPlacement("gem", (2.5, 1.5)),
             ObjectPlacement("gem", (3.5, 1.5))],
    object_types={"gem": ObjectSpec(sprite=gem_sprite, reward=2.0)},
)
```

An `ObjectSpec` defines `reward`, `pickup_radius`, `terminal`, `consume`,
`is_goal`, `inventory_key`, `width_scale`, `height_scale`, and a sprite. Sprites
are uint8 RGBA arrays or zero-argument CPU builders, evaluated once during
construction. The renderer samples an atlas containing only types referenced
by that environment's object slots or `possible_object_types`. Unused
configuration entries are ignored; their sprite builders are never called.
No object-specific shape functions run inside JIT-compiled rendering.

Optional `pickup_bounds=((min_x, min_y), (max_x, max_y))` replaces the circular
radius with an inclusive rectangular contact region. Bounds are world-axis
offsets of the player from the object, independent of heading. Catalogs using
only circles do not execute rectangular contact tests, and vice versa. DMLab
apples and goals use the source engine's XY bounds, scaled to maze coordinates.

`terminal=True` ends an episode on collection; `is_goal=True` separately marks
navigation goals. Reset activates one eligible goal candidate, all other
initially active objects, and preserves inactive slots. `consume=False` keeps
an object active, so collection events and rewards can repeat on contact.
`inventory_key` grants a built-in key color. For additional effects, environment
subclasses can use the per-slot `info["picked_objects"]` mask. Catalog IDs in
`info["picked_type_id"]` are local to `env.object_catalog.names` (-1 if none).
Multiple simultaneous pickups are represented by the mask.

A custom `object_symbols={"D": "gem"}` legend can place named objects directly
in ASCII. It applies only to that parse and overrides colored-wall glyphs;
spawn and door symbols cannot be overridden. Explicit placements avoid glyph
conflicts. Layouts without objects or goals are supported by `RayMazeEnv` and
end at the time limit. The lower-level parser defaults to requiring a goal;
pass `require_goal=False` when inspecting goal-less layouts directly.

Object slot counts stay fixed throughout a compiled run. Set a placement's
`active=False` to reserve a slot and change its active flag in an environment's
reset/step logic when needed. A subclass that changes slot types can override
`object_type_ids(state)` and declare the additional possible types at
construction. Batch different counts only after padding to compatible shapes.

DMLab supplies its apple definition in [dmlab.py](dmlab.py) and placements in
[dmlab_entities.py](dmlab_entities.py); apples are not built-in engine types.
The random-goal environment with `scattered_reward_density=0` builds only a
goal catalog and omits fruit sampling. Static levels can omit apples with
`objects=[]`. See [dmlab.py](dmlab.py) for source reward and reset rules.

## Health Gathering

`HealthGatheringEnv` implements the health-survival task as a separate
environment class. Its state includes `health`, `medkit_xy`, `medkit_active`,
and medkit spawn bookkeeping instead of adding those fields to the navigation
`State`.

The default task uses a square acidic room with gray walls, a green floor, the
player at the center, 16 initial medkits, one new medkit every 30 steps, +1
living reward, 100 death penalty, and a 2100-step timeout.

```python
import jax

from jaxenstein import HEALTH_GATHERING_ACTION_MOVE_FORWARD, make_jaxenstein_env

env = make_jaxenstein_env("health-gathering")
obs, state = env.reset(jax.random.key(0))
obs, state, reward, done, info = env.step(
    state, HEALTH_GATHERING_ACTION_MOVE_FORWARD
)
```

Interactive play:

```bash
uv run python play.py --maze health-gathering
```

## Actions

```python
from jaxenstein import (
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

`jaxenstein.maps.MAZE_KEY_CORRIDOR` is a fixed MiniGrid
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

## My Way Home

`jaxenstein.maps.MAZE_MY_WAY_HOME` is a large fixed map with 17 spawn points
represented by `S` cells and one goal represented by `G`.

The default `my-way-home` variant uses digit wall symbols for colored wall
regions. `my-way-home-colorless` uses the same topology with default `#` walls.
The renderer uses a checker floor pattern and taller walls by default for all
maps.

`MAP_EPISODE_HORIZONS_BY_NAME` gives My Way Home a 2100-step timeout, and
`play.py` passes that horizon automatically.

```bash
uv run python play.py --maze my-way-home
uv run python play.py --maze my-way-home-colorless
```

## DeepMind Lab Nav Mazes

`jaxenstein.maps.dmlab.dmlab_map_to_ascii` converts grid-aligned DMLab `nav_maze` `.map`
files into geometry-only ASCII. Player starts become `S`, goals become `G`, and
decal image walls become stable colored wall symbols. Use `dmlab_map_to_spec`
for the complete conversion, which includes separate object placements and
preserves spawn/object overlap. The CLI exports this complete spec as JSON by
default (`--format ascii` explicitly requests geometry only). In random-goal
levels, the factory samples the candidate roles and relocates the start using
the official Lua distance rule.

The DMLab Lua levels set `episodeLengthSeconds` to 60, 150, and 300 for
indices 01, 02, and 03. Using a Doom-like 30 Hz step budget, JAXenstein maps
those to horizons of 1800, 4500, and 9000 for the static and random
variants.

```bash
uv run python scripts/convert_dmlab_map.py path/to/nav_maze_static_01.map
uv run python play.py --maze dmlab-static-01
```

## Custom Map Example

```python
import jax
import jax.numpy as jnp

from jaxenstein import ACTION_INTERACT, ACTION_MOVE_FORWARD, RayMazeEnv

CUSTOM = """
#############
#S...b.B...G#
#.#########.#
#...........#
#############
"""

env = RayMazeEnv.from_ascii(CUSTOM)
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

## Vectorization

Use `jax.vmap` to run multiple instances of the same environment.

```python
import jax
import jax.numpy as jnp

from jaxenstein import ACTION_MOVE_FORWARD, RayMazeEnv
from jaxenstein.maps import MAZE_SIMPLE

env = RayMazeEnv.from_ascii(MAZE_SIMPLE)

keys = jax.random.split(jax.random.key(0), 2)
obs, states = jax.vmap(env.reset)(keys)

actions = jnp.asarray([ACTION_MOVE_FORWARD, ACTION_MOVE_FORWARD], dtype=jnp.int32)
obs, states, reward, done, info = jax.vmap(env.step)(states, actions)
```
