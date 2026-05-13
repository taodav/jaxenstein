<p align="center">
 <img width="80%" src="images/jaxenstein.png" />
</p>

# JAXenstein

First-person maze environments in JAX.

Features: ASCII maps, RGB raycast observations, billboard sprites, colored keys
and doors, sparse goal rewards, ViZDoom-style health survival, and
JIT/vmap-friendly environment steps.

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

Use an ID with `play.py --maze {ID}`. Navigation maps live in
`jes.maps.MAPS_BY_NAME`; Health Gathering maps live in
`jes.maps.HEALTH_GATHERING_MAPS_BY_NAME`.

| Group | Environment | ID | Description |
| --- | --- | --- | --- |
| Basic | Simple | `simple` | Small navigation task with one start and one goal. |
| Basic | Key-door | `key-door` | Collect a red key, open a red door, reach the goal. |
| MiniGrid | KeyCorridorS4R3 | `key-corridor` | 3-by-3 room key corridor with colored doors and a locked goal room. |
| ViZDoom | Health Gathering | `health-gathering` | Survive an acidic room by collecting medkits. |
| ViZDoom | My Way Home | `my-way-home` | Large maze with many starts, colored walls, and one goal. |
| DMLab | Static goal | `dmlab-static-{01,02,03}` | Fixed-goal mazes; `01` small, `02` medium, `03` large. |
| DMLab | Random goal | `dmlab-random-goal-{01,02,03}` | Same sizes; one goal candidate is active each episode. |

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

Navigation controls: `W/S` move, `A/D` turn, `Space` interact, `R` reset,
`Q` or `Escape` quit. Health Gathering uses the ViZDoom action set: `W` move
forward and `A/D` turn.

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

Health Gathering has a separate state with a `health` variable and medkit slots,
so navigation tasks keep their smaller `State`:

```python
import jax

from jes import ACTION_MOVE_FORWARD, HealthGatheringEnv
from jes.maps import MAZE_HEALTH_GATHERING

env = HealthGatheringEnv.from_ascii([MAZE_HEALTH_GATHERING])
obs, state = env.reset(jax.random.key(0))
obs, state, reward, done, info = env.step(state, ACTION_MOVE_FORWARD)

print(state.health)
```

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
