# Jaxenstein

A small Wolfenstein-style first-person maze renderer and environment in JAX.

The current environment supports:

- ASCII-authored mazes
- first-person RGB observations
- DDA grid raycasting
- billboard object sprites
- colored keys and colored doors
- sparse goal reward
- JIT and vmap-friendly reset, step, and render paths
- an interactive Tk player

## Setup

Use `uv` from the repository root:

```bash
uv sync --extra dev
```

For local CPU-only JAX runs, set:

```bash
export JAX_PLATFORMS=cpu
```

You can also prefix individual commands:

```bash
env JAX_PLATFORMS=cpu uv run pytest
```

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
uv run python play.py --maze dmlab-nav-maze-static-01
```

All maps render with a higher wall scale and checker floor by default. This map also uses colored wall symbols and the original ViZDoom `episode_timeout` of 2100 steps. The same topology is available without wall colors:

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

`S` marks a spawn candidate, `G` is the goal, lowercase symbols are colored keys, uppercase colored symbols are doors, and digit symbols are colored walls. Multiple `S` cells are sampled uniformly on reset. See [jes/MAZES.md](jes/MAZES.md) for the full symbol table and examples for creating custom maps.

## Tests

```bash
env JAX_PLATFORMS=cpu uv run pytest
```
