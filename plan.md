# Goal

Build a JAX-native environment:

[
\texttt{ASCII maze} \rightarrow \texttt{JAX grid} \rightarrow \texttt{first-person RGB observation}
]

with an API like:

```python
state = env.reset(key, maze_id)
obs, state, reward, done, info = env.step(state, action)
```

where `obs` is a true perspective first-person image, not a top-down egocentric crop.

The first version should support only:

```text
- static walls
- static goal
- first-person maze traversal
- fixed action space
- pure JAX reset / step / render
- vmappable batch execution
```

No enemies, no doors, no keys, no textures yet.

---

# Version 0 design

## 1. ASCII maze format

Start with something like:

```text
#################
#S....#.........#
#.###.#.#######.#
#...#.#.......#.#
###.#.#######.#.#
#...#.....#...#.#
#.#######.#.###.#
#.........#....G#
#################
```

Suggested symbols:

| Symbol | Meaning               |
| ------ | --------------------- |
| `#`    | wall                  |
| `.`    | empty floor           |
| `S`    | spawn                 |
| `G`    | goal                  |
| space  | empty floor, optional |

Later you can add:

| Symbol             | Meaning                          |
| ------------------ | -------------------------------- |
| `A`, `B`, `C`      | different wall colors/textures   |
| `0`, `1`, `2`, `3` | spawn with fixed initial heading |
| `K`                | key                              |
| `D`                | door                             |
| `R`, `B`, `Y`      | colored goals/cues               |

But for v0, only `#`, `.`, `S`, `G`.

The ASCII parser does **not** need to be JAX-jitted. It can run once in Python and convert maps into JAX arrays.

Recommended parsed representation:

```python
@dataclass
class Maze:
    wall_grid: jax.Array      # [H, W], bool
    color_grid: jax.Array     # [H, W], int32 wall/color id
    spawn_xy: jax.Array       # [2], float32
    spawn_theta: float
    goal_xy: jax.Array        # [2], float32
```

For cells, use coordinates where cell centers are at:

```python
x = col + 0.5
y = row + 0.5
```

This makes collision and raycasting cleaner.

---

# 2. Environment state

Use a small immutable state object:

```python
@dataclass
class State:
    pos: jax.Array       # [2], float32, x/y position
    theta: jax.Array     # scalar angle in radians
    t: jax.Array         # int32 step count
    done: jax.Array      # bool
    maze_id: jax.Array   # int32, if batching over a fixed set of mazes
```

For the first version, I would not put the maze itself inside the state. Instead, keep a static batch of mazes in the env object:

```python
wall_grids: [N, H, W]
color_grids: [N, H, W]
spawn_xy: [N, 2]
goal_xy: [N, 2]
```

Then `maze_id` indexes into these.

This makes it easy to swap in different ASCII maps while still supporting JIT/vmap.

---

# 3. Action space

Use a stationary discrete action space:

```python
0 = turn left
1 = turn right
2 = move forward
3 = move backward
4 = strafe left
5 = strafe right
```

For the very first prototype, even this is enough:

```python
0 = turn left
1 = turn right
2 = move forward
```

I would include backward/strafe only if you want a more FPS-like feel.

Parameters:

```python
turn_angle = pi / 12      # 15 degrees
move_speed = 0.15         # cell units per step
agent_radius = 0.15       # optional, can start with point agent
horizon = 300
```

Movement direction:

```python
forward = [cos(theta), sin(theta)]
right = [cos(theta + pi / 2), sin(theta + pi / 2)]
```

Collision v0 can be simple:

```python
new_pos = proposed_pos
if proposed_pos is inside wall:
    new_pos = old_pos
```

Later, add sliding collision, radius checks, and separate x/y collision.

---

# 4. Reward and termination

For v0:

```python
reward = 1.0 if distance(pos, goal_xy) < goal_radius else 0.0
done = reached_goal or t >= horizon
```

Use sparse reward only.

For exploration diagnostics, keep privileged info:

```python
info = {
    "x": pos[0],
    "y": pos[1],
    "theta": theta,
    "cell_x": floor(pos[0]),
    "cell_y": floor(pos[1]),
    "reached_goal": reached_goal,
}
```

This gives you true latent visitation counts for evaluation.

---

# 5. Renderer overview

The renderer should do classic Wolfenstein-style raycasting.

Input:

```python
pos: [2]
theta: scalar
wall_grid: [H, W]
color_grid: [H, W]
```

Output:

```python
rgb: [img_h, img_w, 3]
```

Recommended initial settings:

```python
img_h = 64
img_w = 64
fov = pi / 3       # 60 degrees
max_depth = 20.0
```

For each image column, cast a ray.

Ray angles:

```python
camera_x = linspace(-1, 1, img_w)
ray_angle = theta + atan(camera_x * tan(fov / 2))
ray_dir = [cos(ray_angle), sin(ray_angle)]
```

Then for each ray, find the first wall intersection. This gives:

```python
distance[col]
wall_id[col]
hit_side[col]        # whether horizontal or vertical wall face was hit
hit_fraction[col]    # later used for textures
```

Then convert distances into vertical wall heights:

```python
corrected_distance = distance * cos(ray_angle - theta)
wall_height = img_h / corrected_distance
```

The cosine correction avoids fisheye distortion.

For each pixel row, decide whether it is ceiling, wall, or floor:

```python
wall_top = img_h / 2 - wall_height / 2
wall_bottom = img_h / 2 + wall_height / 2

if row < wall_top:
    ceiling
elif row > wall_bottom:
    floor
else:
    wall
```

That gives you a simple first-person RGB renderer.

---

# 6. Raycasting implementation choice

There are two viable ways.

## Option A: fixed-step ray marching

This is the easiest to write.

For each ray, sample points:

```python
p_k = pos + depth_k * ray_dir
```

where:

```python
depth_k = k * step_size
```

Then find the first `k` where `p_k` lands inside a wall cell.

Advantages:

```text
- very simple
- easy to vmap
- easy to JIT
- no tricky control flow
```

Disadvantages:

```text
- approximate intersections
- possible aliasing artifacts
- more samples needed for clean walls
```

For a first implementation, this is probably enough.

Example parameters:

```python
num_depth_samples = 128
max_depth = 16.0
step_size = max_depth / num_depth_samples
```

This means per observation you check:

```python
64 columns × 128 samples = 8192 grid lookups
```

That is totally reasonable in JAX, especially batched.

I would start here.

---

## Option B: DDA grid traversal

This is the classic Wolfenstein method.

It steps from grid cell to grid cell along the ray and finds the exact first wall. It is faster and cleaner than fixed-step ray marching, but slightly more annoying to implement in JAX.

Advantages:

```text
- exact cell intersections
- sharper rendering
- fewer steps
- better texture coordinates
```

Disadvantages:

```text
- more complex
- requires lax.while_loop or fixed-length lax.scan
```

I would implement DDA after the environment already works.

My recommended path:

```text
v0: fixed-step ray marching
v1: DDA raycasting
v2: textured DDA walls
```

---

# 7. Minimal renderer v0

For the first renderer, use distance-shaded flat colors.

Wall color:

```python
base_wall_color = color_palette[wall_id]
shade = 1.0 / (1.0 + 0.1 * distance ** 2)
wall_rgb = base_wall_color * shade
```

Floor and ceiling:

```python
ceiling_rgb = [70, 70, 90]
floor_rgb = [45, 45, 45]
```

Maybe add simple horizon shading later.

This is already enough to look like a 1990s raycast FPS.

---

# 8. ASCII map swapping

I would define maps as Python strings:

```python
MAZE_SIMPLE = """
#########
#S.....G#
#.#####.#
#.......#
#########
"""

MAZE_ALIASED_LOOP = """
###############
#S....#.......#
#.###.#.#####.#
#...#.#.....#.#
###.#.#####.#.#
#...#.......#.#
#.###########.#
#...........G.#
###############
"""
```

Then a loader:

```python
mazes = [
    parse_ascii_maze(MAZE_SIMPLE),
    parse_ascii_maze(MAZE_ALIASED_LOOP),
    parse_ascii_maze(MAZE_LONG_CORRIDOR),
]
env = RayMazeEnv(mazes)
```

Important constraint: if you want to stack maps into JAX arrays, they should all have the same height and width. For convenience, either:

1. enforce same shape, or
2. pad smaller maps with walls.

I would pad with walls.

---

# 9. Suggested code organization

Something like:

```text
jax_raymaze/
  __init__.py
  ascii.py          # parse ASCII maps into arrays
  env.py            # reset, step, reward, termination
  render.py         # raycaster
  maps.py           # predefined ASCII maps
  test_render.py    # save image/grid sanity checks
  test_env.py       # collision/reward tests
```

Core modules:

## `ascii.py`

Responsible for:

```python
parse_ascii_maze(ascii_str) -> Maze
stack_mazes(list[Maze]) -> MazeBatch
```

## `env.py`

Responsible for:

```python
reset(key, maze_id=None)
step(state, action)
```

## `render.py`

Responsible for:

```python
render_first_person(pos, theta, wall_grid, color_grid) -> rgb
raycast_fixed_step(pos, theta, wall_grid, color_grid) -> distances, wall_ids
```

Keep rendering independent from environment stepping. That makes debugging much easier.

---

# 10. Testing plan

I would test in this order.

## Test 1: ASCII parsing

Given:

```text
#####
#S.G#
#####
```

Check:

```python
wall_grid.shape == (3, 5)
spawn_xy == [1.5, 1.5]
goal_xy == [3.5, 1.5]
```

## Test 2: collision

From spawn, repeatedly move forward into a wall.

Check:

```python
pos does not enter wall
```

## Test 3: reward

Teleport or step near goal.

Check:

```python
reward == 1
done == True
```

## Test 4: renderer smoke test

Render a straight corridor. Save image. Confirm visually:

```text
- walls on left/right
- floor below
- ceiling above
- far wall in center
```

## Test 5: rotation sanity

Render the same position at four headings:

```python
theta = 0, pi/2, pi, 3pi/2
```

Confirm the view changes correctly.

## Test 6: JIT and vmap

Make sure this works:

```python
jit_step = jax.jit(env.step)
batched_step = jax.vmap(env.step)
```

The renderer should also work under:

```python
jax.jit(render_first_person)
jax.vmap(render_first_person)
```

---

# 11. Milestones

## Milestone 1: Traversal without rendering

Build the gridworld dynamics first.

Deliverable:

```text
ASCII maze -> reset -> step -> collision -> reward
```

Use a top-down debug plot only for development.

---

## Milestone 2: Depth renderer

Before RGB, render a vector:

```python
depth_obs: [num_rays]
```

This is the easiest sanity check.

The agent should see small distances when facing a wall and large distances down a corridor.

This is also a useful non-pixel observation baseline.

---

## Milestone 3: RGB raycast renderer

Convert depth rays into vertical wall slices.

Deliverable:

```python
obs: [64, 64, 3]
```

At this point, you have the core Wolfenstein environment.

---

## Milestone 4: Batched JAX env

Support:

```python
state = vmap(reset)(keys, maze_ids)
obs, state, reward, done, info = vmap(step)(state, actions)
```

This is where the environment becomes useful for PPO/RND.

---

## Milestone 5: Maze suite

Add a small set of hand-designed ASCII layouts:

```text
1. simple corridor
2. T-maze
3. aliased loop
4. four identical rooms
5. long corridor with repeated junctions
6. sparse goal maze
```

These should be chosen to test specific failure modes.

---

# 12. First benchmark maps I would make

## A. Straight corridor

Purpose: renderer sanity check.

```text
###########
#S.......G#
###########
```

## B. T-maze

Purpose: simple partial observability / memory.

```text
###########
#....G....#
#####.#####
#####.#####
#####S#####
###########
```

## C. Aliased loop

Purpose: observation aliasing.

```text
#############
#S....#.....#
#.##..#.##..#
#.#...#.#...#
#.#.###.#.###
#.#.....#..G#
#############
```

## D. Repeated rooms

Purpose: many latent states with similar observations.

```text
#################
#S..#...#...#...#
#...#...#...#...#
#...............#
#...#...#...#...#
#...#...#...#..G#
#################
```

## E. Spiral maze

Purpose: long-horizon sparse reward.

```text
###############
#S............#
#############.#
#.............#
#.#############
#.............#
#############.#
#G............#
###############
```

You can later add procedural generators, but I would start hand-designed. For the paper, hand-designed environments are also easier to explain.

---

# 13. Important design choices for your exploration project

## Keep actions stationary

Unlike Battleship, invalid actions should be no-ops, not removed.

So even if the agent bumps into a wall:

```python
action = move_forward
transition = no-op
```

The action space remains fixed.

This matters because your exploration method should not get help from action masking.

---

## Keep latent state available for analysis

Even though the agent sees only pixels, your environment should expose privileged diagnostics:

```python
latent_cell = (floor(x), floor(y), discretized_theta)
latent_pose_count
room_id
distance_to_goal
```

This lets you directly evaluate whether RND prediction error correlates with true latent novelty.

---

## Include wall-bump nuisance trajectories

Do not prevent repeated wall bumps. They are a useful test.

A bad recurrent memory may treat:

```text
walk into wall, walk into wall, walk into wall
```

as highly novel because the action sequence continues changing hidden state.

A good memory/exploration bonus should ideally learn that these histories are low-value nuisance trajectories.

This is directly relevant to your tangent-memory story.

---

## Separate map identity from exploration

Initially, I would train and evaluate on fixed maps. Then add random maps.

Suggested progression:

```text
1. fixed map, fixed spawn
2. fixed map, random spawn
3. small set of fixed maps
4. procedural maps from same distribution
5. held-out procedural maps
```

For the first paper result, `1–3` may already be enough.

---

# 14. What I would implement first

The absolute first working version should be:

```python
class RayMazeEnv:
    def reset(self, key, maze_id):
        ...

    def step(self, state, action):
        ...

    def render(self, state):
        ...
```

with:

```text
- fixed-step ray marching
- 64 × 64 RGB observations
- 3 actions: left, right, forward
- ASCII maps
- sparse goal reward
- no procedural generation
- no textures
- no keys
- no doors
```

That gives you a clean minimum viable environment.

Then the first experimental comparison can be:

```text
1. PPO sparse reward only
2. PPO + frame RND
3. PPO + feedforward random target over obs/action
4. PPO + frozen GRU RND over obs/action history
5. PPO + tangent memory RND over obs/action history
```

Main metrics:

```text
- goal success
- true latent cell coverage
- true latent pose coverage
- wall-bump rate
- RND prediction error vs visitation count
- coverage of aliased states with identical-looking observations
```

---

# 15. My recommended development order

I would do it exactly in this order:

```text
1. ASCII parser
2. non-rendered grid dynamics
3. top-down debug visualization, only for you
4. depth raycaster
5. RGB Wolfenstein renderer
6. jit single env
7. vmap many envs
8. fixed benchmark maps
9. PPO integration
10. RND integration
11. DDA raycaster upgrade
12. colored walls / texture ids
13. key-door or cue-goal variants
```

The biggest thing is to avoid overbuilding. The first version does not need realistic 3D. It needs to be a **first-person, visually aliased, stationary-action, JAX-native POMDP**. A simple Wolfenstein renderer is almost perfectly matched to that.
