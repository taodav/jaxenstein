# JAXENSTEIN

First-person maze environments in JAX.

Features: ASCII maps, RGB raycast observations, billboard sprites, colored keys
and doors, sparse goal rewards, and JIT/vmap-friendly environment steps.

## Install

`pip` is enough for normal installs:

```bash
pip install git+https://github.com/taodav/jaxenstein.git
```

For local development and scripts:

```bash
git clone https://github.com/taodav/jaxenstein.git
cd jaxenstein
pip install -e ".[dev]"
```

## Environments

Use an ID with `play.py --maze {ID}` or `jes.maps.MAPS_BY_NAME`.

| Environment | ID | Description |
| --- | --- | --- |
| Simple | `simple` | Small navigation task with one start and one goal. |
| Key-door | `key-door` | Collect a red key, open a red door, reach the goal. |
| MiniGrid KeyCorridorS4R3 | `key-corridor` | 3-by-3 room key corridor with colored doors and a locked goal room. |
| ViZDoom My Way Home | `my-way-home` | Large maze with many starts, colored walls, and one goal. |
| DMLab static goal | `dmlab-static-{01,02,03}` | Fixed-goal mazes; `01` small, `02` medium, `03` large. |
| DMLab random goal | `dmlab-random-goal-{01,02,03}` | Same sizes; one goal candidate is active each episode. |

## Scripts

### `play.py`

```bash
python play.py --maze key-door
```

Options:

| Option | Meaning |
| --- | --- |
| `--maze {ID}` | Environment ID. |
| `--resolution N` | Square render resolution. |
| `--resolution WIDTHxHEIGHT` | Rectangular render resolution. |
| `--scale N` | Integer display scale. |
| `--tick-ms MS` | Delay between held-key steps. |
| `--record [PATH]` | Save a GIF. Default path is `trajectory.gif`. |

Controls: `W/S` move, `A/D` turn, `Space` interact, `R` reset,
`Q` or `Escape` quit.

### `scripts/compare_raycast_speed.py`

```bash
python scripts/compare_raycast_speed.py --map my-way-home --widths 64 160 320
```

Options: `--map`, `--widths`, `--max-depth`, `--samples`, `--repeats`,
`--spawn-index`, `--theta`.

### `scripts/convert_dmlab_map.py`

```bash
python scripts/convert_dmlab_map.py path/to/nav_maze_static_01.map
```

Prints the Jaxenstein ASCII map for a DMLab `.map` file.

## Python Usage

```python
import jax
import jax.numpy as jnp

from jes import ACTION_MOVE_FORWARD, RayMazeEnv
from jes.maps import MAZE_SIMPLE

env = RayMazeEnv.from_ascii([MAZE_SIMPLE], img_h=128, img_w=128)
obs, state = env.reset(jax.random.key(0))
obs, state, reward, done, info = env.step(state, ACTION_MOVE_FORWARD)

print(obs.shape)  # (128, 128, 3)
print(obs.dtype)  # uint8
```

The core API works with JAX transforms:

```python
keys = jax.random.split(jax.random.key(0), 8)
obs, states = jax.vmap(env.reset)(keys)

actions = jnp.full((8,), ACTION_MOVE_FORWARD, dtype=jnp.int32)
obs, states, reward, done, info = jax.vmap(env.step)(states, actions)
```

If an environment batch contains multiple maps, `reset(key)` samples one from
the key.

## ASCII Maps

```python
MAZE_KEY_DOOR = """
###########
#S.r.R...G#
###########
"""
```

Common symbols: `S` start, `G` goal, lowercase keys, uppercase locked doors,
`"` and `\` unlocked colored doors, `#` walls, `.` floor. Multiple `S` and
`G` cells are sampled uniformly on reset. See [jes/MAZES.md](jes/MAZES.md).

## Tests

```bash
JAX_PLATFORMS=cpu pytest
```
