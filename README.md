# JAXENSTEIN

A small Wolfenstein-style first-person maze renderer and environment in JAX.

The current environment supports:

- ASCII-authored mazes
- first-person RGB observations
- DDA grid raycasting
- billboard object sprites
- colored keys and locked or unlocked doors
- sparse goal reward
- JIT and vmap-friendly reset, step, and render paths
- an interactive Tk player

## Setup

Use `uv` from the repository root:

```bash
uv sync --extra dev
```

## Environments

Pass these IDs to `play.py --maze <id>` or look them up in
`jes.maps.MAPS_BY_NAME`.

| Group | Environment | ID | Description |
| --- | --- | --- | --- |
| Basic | Simple maze | `simple` | Small navigation task with one spawn, one goal, and interior walls for movement and collision checks. |
| Basic | Key-door maze | `key-door` | Linear key-door task where the agent must collect the red key, open the red door, and reach the goal. |
| MiniGrid | KeyCorridorS4R3 | `key-corridor` | Fixed MiniGrid KeyCorridorS4R3-style map with a red key, colored unlocked doors, a red-locked door to the goal, and a 480-step horizon. |
| ViZDoom | My Way Home | `my-way-home` | Large ViZDoom My Way Home conversion with 17 spawn candidates, colored wall regions, a single green-armor goal, and a 2100-step timeout. |
| ViZDoom | My Way Home, colorless walls | `my-way-home-colorless` | Same My Way Home topology and timeout, but with default colorless walls. |
| DeepMind Lab static-goal | Static 01 | `dmlab-static-01` | Small converted DMLab nav maze with a fixed goal, sampled spawn candidates, decal-derived wall colors, and an 1800-step horizon. |
| DeepMind Lab static-goal | Static 02 | `dmlab-static-02` | Medium converted DMLab nav maze with a fixed goal, sampled spawn candidates, decal-derived wall colors, and a 4500-step horizon. |
| DeepMind Lab static-goal | Static 03 | `dmlab-static-03` | Large converted DMLab nav maze with a fixed goal, sampled spawn candidates, decal-derived wall colors, and a 9000-step horizon. |
| DeepMind Lab random-goal | Random Goal 01 | `dmlab-random-goal-01` | Small converted DMLab nav maze where one apple-reward goal candidate is sampled active at reset, with an 1800-step horizon. |
| DeepMind Lab random-goal | Random Goal 02 | `dmlab-random-goal-02` | Medium converted DMLab nav maze with one active sampled goal, decal-derived wall colors, and a 4500-step horizon. |
| DeepMind Lab random-goal | Random Goal 03 | `dmlab-random-goal-03` | Large converted DMLab nav maze with one active sampled goal, dense landmark wall colors, and a 9000-step horizon. |

## Play

```bash
uv run python play.py
```

Controls:

| Key | Action |
| --- | --- |
| `W` | move forward |
| `S` | move backward |
| `A` | turn left |
| `D` | turn right |
| `Space` | interact |
| `R` | reset |
| `Q` / `Escape` | quit |

Play the key-door demo:

```bash
uv run python play.py --maze key-door
```

Play the ViZDoom My Way Home maze conversion:

```bash
uv run python play.py --maze my-way-home
```

Play a DeepMind Lab nav maze conversion:

```bash
uv run python play.py --maze dmlab-static-01
```

The DMLab source levels set `episodeLengthSeconds` to 60, 150, and 300 for
maze indices 01, 02, and 03. Using a Doom-like 30 Hz step budget, Jaxenstein
uses horizons of 1800, 4500, and 9000 for both static and random-goal
variants.

All maps render with a higher wall scale and checker floor by default. This map also uses colored wall symbols and the original ViZDoom `episode_timeout` of 2100 steps. The same topology is available without wall colors (for a harder memory testing environment):

```bash
uv run python play.py --maze my-way-home-colorless
```

Record a GIF of the trajectory:

```bash
uv run python play.py --maze key-door --record runs/key-door.gif
```

Useful display options:

```bash
uv run python play.py --resolution 128 --scale 4 --tick-ms 30
uv run python play.py --resolution 320x240 --scale 2
```

## Python Usage

```python
import jax
import jax.numpy as jnp

from jes import ACTION_MOVE_FORWARD, RayMazeEnv
from jes.maps import MAZE_SIMPLE

env = RayMazeEnv.from_ascii([MAZE_SIMPLE], img_h=128, img_w=128)
obs, state = env.reset(jax.random.key(0), jnp.asarray(0, dtype=jnp.int32))
obs, state, reward, done, info = env.step(state, ACTION_MOVE_FORWARD)

print(obs.shape)  # (64, 64, 3)
print(obs.dtype)  # uint8
```

The core environment APIs are compatible with JAX transforms:

```python
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_render = jax.jit(env.render)

keys = jax.random.split(jax.random.key(0), 8)
maze_ids = jnp.zeros((8,), dtype=jnp.int32)
obs, states = jax.vmap(env.reset)(keys, maze_ids)

actions = jnp.full((8,), ACTION_MOVE_FORWARD, dtype=jnp.int32)
obs, states, reward, done, info = jax.vmap(env.step)(states, actions)
```

## ASCII Mazes

Maps are plain strings. Example:

```python
MAZE_KEY_DOOR = """
###########
#S.r.R...G#
###########
"""
```

`S` marks a spawn candidate, `G` marks a goal candidate, lowercase symbols are colored keys, uppercase door symbols encode the required key, `"`, and `\` are colored unlocked doors, and digit or generated symbols are colored walls. Multiple `S` and `G` cells are sampled uniformly on reset. See [jes/MAZES.md](jes/MAZES.md) for the full ASCII map semantics and examples.

## Tests

```bash
env JAX_PLATFORMS=cpu uv run pytest
```
