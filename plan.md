Given the current repo already has ASCII maps, RGB raycast observations, billboard sprites, colored keys/doors, sparse rewards, ViZDoom-style health survival, and JIT/vmap-friendly steps, I would treat the DOOM-style extension as a **v2 rendering/environment**, not a rewrite of the environment API. The current layout is nicely modular already: `jes/ascii.py`, `jes/env.py`, `jes/render.py`, `jes/objects.py`, `jes/maps.py`, plus DMLab conversion scripts and playable environments. ([GitHub][1])

# Goal

Add a **2.5D sector engine** to JAXenstein:

```text
current engine:
  grid cells are either wall or floor

v2 engine:
  cells belong to sectors
  sectors have floor/ceiling heights, colors, materials, and traversal rules
```

This should unlock:

```text
- rooms with different floor heights
- ceiling height variation
- stairs
- pits
- low walls
- windows
- raised platforms
- overhang-like visual effects, if carefully constrained
- true vertical projection
- pitch/look up/down
```

I would **not** aim for full DOOM compatibility. The target should be:

> DOOM-style 2.5D rendering and traversal, while staying grid-aligned, ASCII-authorable, JAX-native, fast, and compatible with vmap/jit.

# High-level design

Right now, your ASCII semantics are basically:

```python
wall_grid: [H, W] bool
color_grid: [H, W] int
door_grid: [H, W] int
object_xy: [N, 2]
object_type: [N]
object_color: [N]
```

The parser already maps ASCII rows/cols to continuous (x,y) cell centers and pads ragged maps with walls when stacking batches, which is exactly the right foundation for keeping v2 maps JIT-friendly. ([GitHub][2])

The sector extension should add:

```python
sector_grid: [H, W] int32          # sector id per cell
sector_floor_z: [S] float32
sector_ceil_z: [S] float32
sector_floor_color: [S] int32
sector_ceil_color: [S] int32
sector_light: [S] float32
sector_flags: [S] int32
```

Then static walls become a special case:

```python
solid_grid: [H, W] bool
```

or:

```python
sector_grid[y, x] = SECTOR_SOLID
```

I would prefer **separate `solid_grid` plus `sector_grid`** because it keeps collision and DDA logic clearer.

# Core principle

Do not replace your current renderer immediately.

Add a parallel renderer:

```python
render_first_person(...)          # current Wolfenstein grid renderer
render_sector_first_person(...)   # new DOOM-lite renderer
```

and a parallel environment class:

```python
RayMazeEnv          # current
SectorMazeEnv       # v2
```

The v2 parser can reuse most of the existing ASCII parser, but should produce a richer `SectorMaze` / `SectorMazeBatch`.

# Proposed data structures

## `jes/sector.py`

Add a new file:

```python
@dataclass(frozen=True)
class SectorMaze:
    wall_grid: jax.Array          # [H, W], bool
    color_grid: jax.Array         # [H, W], int32, for blocking walls
    sector_grid: jax.Array        # [H, W], int32

    sector_floor_z: jax.Array     # [S], float32
    sector_ceil_z: jax.Array      # [S], float32
    sector_floor_color: jax.Array # [S], int32
    sector_ceil_color: jax.Array  # [S], int32
    sector_light: jax.Array       # [S], float32
    sector_flags: jax.Array       # [S], int32

    spawn_xy: jax.Array
    spawn_xy_options: jax.Array
    spawn_count: jax.Array
    spawn_theta: jax.Array
    goal_xy: jax.Array

    object_xy: jax.Array
    object_type: jax.Array
    object_color: jax.Array
    door_grid: jax.Array
```

And:

```python
@dataclass(frozen=True)
class SectorMazeBatch:
    wall_grids: jax.Array
    color_grids: jax.Array
    sector_grids: jax.Array

    sector_floor_z: jax.Array
    sector_ceil_z: jax.Array
    sector_floor_color: jax.Array
    sector_ceil_color: jax.Array
    sector_light: jax.Array
    sector_flags: jax.Array

    spawn_xy: jax.Array
    spawn_xy_options: jax.Array
    spawn_count: jax.Array
    spawn_theta: jax.Array
    goal_xy: jax.Array

    object_xy: jax.Array
    object_type: jax.Array
    object_color: jax.Array
    door_grids: jax.Array
```

For batching, pad sectors just like you already pad objects and spawns.

# State extension

Your current navigation state can probably remain almost unchanged, except for vertical/camera info:

```python
@dataclass(frozen=True)
class SectorState:
    pos: jax.Array          # [2]
    theta: jax.Array        # scalar
    pitch: jax.Array        # scalar or pixel offset
    camera_z: jax.Array     # scalar
    sector_id: jax.Array    # current sector
    t: jax.Array
    done: jax.Array
    maze_id: jax.Array
    carried_keys: jax.Array
    object_active: jax.Array
    door_open: jax.Array
```

But I would **not** make `camera_z` an independent physics variable yet. Start with:

```python
camera_z = floor_z[current_sector] + eye_height
```

where:

```python
eye_height = 0.5
```

This gives you stairs and raised floors without gravity, jumping, or falling.

# Rendering plan

Your current `render_first_person` already does DDA raycasting, distance correction, wall projection, checker floor rendering, door panel patterns, and billboard sprite compositing. That means v2 should reuse the same outer structure: one ray per column, DDA through the grid, then project vertical surfaces into image columns. ([GitHub][3])

The key change is that a ray no longer only asks:

```python
did I hit a wall?
```

It asks:

```python
what vertical segment should I draw at this grid boundary?
```

## Current wall projection

Current style:

```python
wall_height = img_h * wall_height_scale / corrected_dist
wall_top = img_h / 2 - wall_height / 2
wall_bottom = img_h / 2 + wall_height / 2
```

## Sector projection

New style:

```python
y_top = horizon - focal * (z_top - camera_z) / corrected_dist
y_bottom = horizon - focal * (z_bottom - camera_z) / corrected_dist
```

This one formula is the core of the whole extension.

For ordinary full-height walls:

```python
z_bottom = sector_floor_z[current_sector]
z_top = sector_ceil_z[current_sector]
```

For a step up:

```python
z_bottom = current_floor_z
z_top = next_floor_z
```

For a low wall:

```python
z_bottom = current_floor_z
z_top = current_floor_z + low_wall_height
```

For a window:

```python
lower segment: z_bottom = floor_z, z_top = window_bottom
upper segment: z_bottom = window_top, z_top = ceil_z
middle is transparent
```

# DDA extension

I would create a new DDA function that returns more than one hit per ray.

Current DDA stops at the first blocked cell. For sectors, a ray may pass through many cells and encounter several visible vertical boundaries.

So instead of:

```python
distances: [W]
wall_ids: [W]
```

return fixed-size per-column hit buffers:

```python
hit_dist: [W, K]
hit_kind: [W, K]
hit_color: [W, K]
hit_z0: [W, K]
hit_z1: [W, K]
hit_sector_front: [W, K]
hit_sector_back: [W, K]
hit_active: [W, K]
```

where (K) is a small static maximum number of visible segments per ray, for example:

```python
max_hits_per_ray = 8
```

This is very JAX-compatible because shapes stay static.

## What counts as a hit?

During DDA, each step crosses from cell A to cell B.

Let:

```python
a = current_cell
b = next_cell
sector_a = sector_grid[a]
sector_b = sector_grid[b]
```

You emit a visible boundary if any of these are true:

```text
1. next cell is outside map
2. next cell is solid wall
3. sector_b floor is higher than sector_a floor
4. sector_b ceiling is lower than sector_a ceiling
5. sector_a and sector_b have different wall/material boundary
6. there is a door/window boundary between them
```

The simplest v0 sector logic should only include cases 1–4.

# Sector renderer v0

I would implement the first sector renderer with these constraints:

```text
- no transparent windows yet
- no portals yet
- no overlapping geometry
- no slopes
- no moving platforms
- no vertical sprite z yet
- one vertical segment per crossed boundary
```

Supported geometry:

```text
- full walls
- open sectors with floor/ceiling heights
- stairs as adjacent sectors with different floor heights
- pits as lower sectors
- low walls as explicit blocking cells with short z extent
```

The rendering algorithm:

```python
hits = raycast_sector_dda(...)

rgb = draw_ceiling_and_floor(...)
rgb = draw_hit_segments_back_to_front_or_front_to_back(...)
rgb = render_billboard_sprites(...)
```

For simplicity, draw wall segments **front-to-back** with an occlusion buffer:

```python
column_y_min_drawn: [W]
column_y_max_drawn: [W]
```

But the easiest implementation is probably:

```python
for k in reversed(range(max_hits_per_ray)):
    draw farther hits first
```

Because (K) is small and static, `lax.fori_loop` over hits is fine.

# Floor and ceiling rendering

Current floor checker rendering assumes a flat floor at a fixed height. For sector floors, floor casting should sample the floor height and color/material of the cell hit by a ray through each pixel.

I would not do full sector floor casting in v0.

Instead, use a staged approach:

## v0: Keep old floor/ceiling

Use current checker/floor rendering, plus wall/step segments. This will look imperfect but lets you debug vertical segments.

## v1: Per-sector flat floor/ceiling bands

When a ray enters a sector, draw floor/ceiling color based on the closest visible sector. Approximate but cheap.

## v2: True floor casting

For each pixel below horizon, compute intersection with floor plane:

```python
depth = (camera_z - floor_z) * focal / (row - horizon)
world_xy = pos + ray_dir * depth
cell = floor(world_xy)
sector = sector_grid[cell]
color = sector_floor_color[sector]
```

The tricky part is that `floor_z` depends on the sector you land in, which depends on `world_xy`, which depends on `floor_z`. For flat floors with modest height variation, one lookup pass is usually good enough:

```python
assume floor_z = current_floor_z
compute world_xy
lookup target sector
recompute depth using target floor_z
compute final world_xy
lookup final target sector
```

That is still vectorizable.

# ASCII authoring plan

I would avoid cramming all sector metadata into one ASCII map. Keep your current map style for geometry, and add optional metadata layers.

## Minimal v2 syntax

```python
MAZE = """
###########
#S....a..G#
#.....a...#
#..bbb....#
###########
"""

SECTORS = {
    ".": SectorDef(floor_z=0.0, ceil_z=1.0, floor_color=0, ceil_color=0),
    "a": SectorDef(floor_z=0.25, ceil_z=1.25, floor_color=1, ceil_color=0),
    "b": SectorDef(floor_z=0.50, ceil_z=1.50, floor_color=2, ceil_color=0),
}
```

But this conflicts with your current use of letters for objects/doors. So I would use a **separate sector layer**:

```python
LAYOUT = """
###########
#S.......G#
#.........#
#.........#
###########
"""

SECTOR_LAYER = """
###########
#000001111#
#000001111#
#000222111#
###########
"""
```

Then:

```python
SECTOR_DEFS = {
    "0": SectorDef(floor_z=0.0, ceil_z=1.0),
    "1": SectorDef(floor_z=0.25, ceil_z=1.25),
    "2": SectorDef(floor_z=0.50, ceil_z=1.50),
}
```

This preserves your existing symbol table for keys, doors, goals, walls, and DMLab symbols. The current docs already define a clear ASCII symbol table for walls, floor, spawn, goals, keys, and doors, so a second layer is cleaner than overloading the primary map. ([GitHub][2])

# Parser API

Add:

```python
SectorMazeEnv.from_ascii(
    ascii_mazes,
    sector_layers=None,
    sector_defs=None,
    ...
)
```

Examples:

```python
env = SectorMazeEnv.from_ascii(
    [MAZE_STAIRS],
    sector_layers=[SECTOR_STAIRS],
    sector_defs=SECTOR_DEFS,
)
```

For backward compatibility:

```python
SectorMazeEnv.from_ascii([old_maze])
```

should produce a single sector:

```python
floor_z = 0.0
ceil_z = 1.0
```

That way all existing maps can be run through the sector renderer.

# Traversal rules

Keep dynamics simple.

At each move proposal:

```python
old_cell = floor(old_pos)
new_cell = floor(new_pos)

old_sector = sector_grid[old_cell]
new_sector = sector_grid[new_cell]
```

Allow movement iff:

```python
not wall_grid[new_cell]
and not closed_door[new_cell]
and abs(floor_z[new_sector] - floor_z[old_sector]) <= max_step_height
and camera_z + head_margin <= ceil_z[new_sector]
```

Initial values:

```python
max_step_height = 0.35
eye_height = 0.5
agent_height = 0.8
head_margin = 0.05
```

No gravity initially. If the next sector is lower, the agent instantly snaps to it. This is not realistic, but it is clean for RL.

# Pitch / look up-down

Add pitch as a rendering-only variable first:

```python
state.pitch: float32
```

Actions:

```python
ACTION_LOOK_UP
ACTION_LOOK_DOWN
ACTION_CENTER_VIEW
```

But I might not add pitch actions to the default RL action space immediately. For experiments, I would expose pitch in play mode first and keep the RL action space unchanged unless the environment needs it.

Projection:

```python
horizon = img_h / 2 + pitch_scale * jnp.tan(pitch)
```

Clamp:

```python
pitch = clip(pitch, -max_pitch, max_pitch)
```

Use:

```python
max_pitch = pi / 6
```

# Doors and windows

Your current engine already has door semantics: closed doors are impassable, `ACTION_INTERACT` can open the door in front of the agent, unlocked doors open without a key, and locked doors require matching key color. Keys/goals are billboard pickup objects. ([GitHub][2])

For sectors, I would preserve that behavior, but reinterpret doors as **boundaries** rather than cells eventually.

## v0: cell doors

Keep current door grid behavior:

```python
door cell is blocking until opened
```

This is easy and compatible.

## v1: boundary doors

Represent doors between cells:

```python
door_x_edges: [H, W + 1]
door_y_edges: [H + 1, W]
```

This lets doors exist between sectors without consuming a whole cell.

Do not do boundary doors first. It will add a lot of parser and interaction complexity.

## Windows

Add later as boundary features:

```python
boundary_lower_z
boundary_upper_z
boundary_material
boundary_flags
```

A window is just a boundary with visible lower and upper wall segments and a transparent middle.

# JAX implementation notes

The sector renderer should preserve static shapes.

Recommended constants:

```python
max_steps = int(max_depth * 2) + 4
max_hits_per_ray = 8
max_sectors = fixed per batch
```

Avoid Python lists in jitted render paths. Store everything as arrays.

Use fixed loops:

```python
jax.lax.fori_loop(0, max_steps, dda_body, state)
jax.lax.fori_loop(0, max_hits_per_ray, draw_body, image)
```

For hit-buffer insertion, keep:

```python
hit_count: [W]
```

and insert with masks:

```python
slot = hit_count
should_insert = active & visible_boundary & (slot < max_hits_per_ray)
```

Then use `jnp.where` to update each slot. Since dynamic indexing into `[W, K]` can be annoying, a simple way is to update all K slots with:

```python
slot_mask = jnp.arange(K)[None, :] == hit_count[:, None]
```

Then:

```python
hit_dist = jnp.where(slot_mask & should_insert[:, None], new_dist[:, None], hit_dist)
```

# File-by-file implementation plan

## 1. `jes/sector.py`

Add:

```python
SectorDef
SectorMaze
SectorMazeBatch
DEFAULT_SECTOR_DEF
```

Maybe:

```python
SECTOR_FLAG_DAMAGE
SECTOR_FLAG_LAVA
SECTOR_FLAG_GOAL
SECTOR_FLAG_LOW_CEILING
```

But do not use many flags at first.

## 2. `jes/sector_ascii.py`

Add:

```python
parse_sector_ascii_maze(...)
stack_sector_mazes(...)
parse_and_stack_sector(...)
```

Reuse logic from `ascii.py`, but add `sector_layer`.

Important validation:

```text
- layout and sector_layer must have same shape after cleaning
- wall cells may have sector id but it is ignored
- every non-wall traversable cell must have valid sector id
- sector defs are padded across batch
```

## 3. `jes/sector_render.py`

Add:

```python
raycast_sector_dda(...)
render_sector_first_person(...)
project_z_segment(...)
draw_vertical_segments(...)
```

Do not modify `render.py` too much. Once stable, you can deduplicate shared helpers.

## 4. `jes/sector_env.py`

Add:

```python
SectorMazeEnv
SectorState
```

Start by copying `RayMazeEnv` and changing:

```text
- reset initializes sector_id and camera_z
- step uses floor_z/ceil_z traversal tests
- render calls render_sector_first_person
```

## 5. `jes/sector_maps.py`

Add test maps:

```python
MAZE_SECTOR_STAIRS
MAZE_SECTOR_PIT
MAZE_SECTOR_LOW_WALL
MAZE_SECTOR_WINDOW
MAZE_SECTOR_MULTIROOM_HEIGHTS
```

## 6. `play.py`

Add a flag:

```bash
python play.py --maze sector-stairs --engine sector
```

or infer engine from environment registry.

Add optional controls:

```text
Up/Down arrows: look up/down
C: center view
```

# Milestones

## Milestone 1: Sector data model, no visual change

Goal: old maps run through `SectorMazeEnv` and look identical to the current renderer.

Implement:

```text
- single default sector
- same walls
- same doors
- same objects
- same reward
```

Success criterion:

```text
RayMazeEnv(simple) and SectorMazeEnv(simple) produce visually similar observations.
```

## Milestone 2: True vertical projection

Replace centered wall projection with:

```python
project(z_bottom, z_top, camera_z, dist)
```

Still use single-height sectors.

Success criterion:

```text
old maps still look correct.
```

## Milestone 3: Stairs / raised platforms

Add `sector_layer` and sector floor heights.

Example:

```text
layout:
###########
#S.......G#
#.........#
###########

sectors:
###########
#000111222#
#000111222#
###########
```

Success criterion:

```text
agent can move from 0 -> 1 -> 2 if step height allows;
walls/steps render with visible vertical risers.
```

## Milestone 4: Ceiling heights

Support different `ceil_z`.

Success criterion:

```text
low-ceiling room visibly changes wall top projection;
collision prevents entering a sector whose ceiling is too low.
```

## Milestone 5: Pitch

Add pitch to state and renderer.

Success criterion:

```text
looking up/down shifts horizon and changes projected walls/floors consistently.
```

## Milestone 6: Floor casting over sector heights

Add approximate sector-aware floor/ceiling colors.

Success criterion:

```text
raised platforms and pits have correct floor colors;
sector transitions are visually legible.
```

## Milestone 7: Windows / low walls

Add boundary/partial-height wall features.

I would implement low walls as special cells first:

```python
low_wall_grid: [H, W] bool
low_wall_z_top: [H, W]
```

Then later convert to boundary walls.

Success criterion:

```text
agent cannot pass through low wall;
agent can see over it.
```

# Suggested example maps

## 1. Stairs

```python
LAYOUT = """
#############
#S.........G#
#...........#
#############
"""

SECTORS = """
#############
#00011122222#
#00011122222#
#############
"""
```

Sector defs:

```python
0: floor_z=0.0, ceil_z=1.2
1: floor_z=0.2, ceil_z=1.4
2: floor_z=0.4, ceil_z=1.6
```

## 2. Pit

```python
LAYOUT = """
#############
#S.........G#
#...........#
#############
"""

SECTORS = """
#############
#00011100000#
#00011100000#
#############
"""
```

Sector `1` has `floor_z=-0.4`.

## 3. Low-ceiling tunnel

```python
SECTORS = """
#############
#0001112222G#
#00011122222#
#############
"""
```

Sector `1` has a low ceiling:

```python
floor_z=0.0
ceil_z=0.75
```

This is useful for testing visual ambiguity and traversal constraints.

## 4. Raised observation platform

Use a high sector in the middle of a room, with a goal visible over low walls. This could become a neat memory/exploration benchmark.

# What not to add yet

I would delay:

```text
- true polygon sectors
- arbitrary angled walls
- sloped floors
- elevators
- moving platforms
- gravity/jumping
- projectiles
- full Doom WAD parsing
- boundary doors
- recursive portal rendering
```

Those are tempting, but they risk turning this into an engine project instead of a research environment library.

# Research-facing value

This extension gives you a stronger benchmark suite without losing interpretability. You can create environments where the same (x,y) corridor layout has different latent vertical structure, or where first-person observations become more/less aliased depending on height and pitch.

For recurrent exploration, the interesting new knobs are:

```text
- partial observability from vertical occlusion
- revisitation at different heights
- stair/pit bottlenecks
- low-wall visibility without traversability
- goal visibility before reachability
- maps where local visual novelty and latent progress diverge
```

That last one is especially useful for your project. A goal can be visible from a raised platform or behind a low wall, but the agent still needs a history-dependent route to actually reach it. That makes the exploration signal less about “seeing new pixels” and more about **remembering traversable structure**.

# My recommended PR sequence

I would split this into small PRs:

```text
PR 1: SectorMaze dataclasses and parser with default single-sector compatibility
PR 2: SectorMazeEnv that reproduces current RayMazeEnv behavior
PR 3: z-aware wall projection in sector renderer
PR 4: sector floor heights + stairs traversal
PR 5: pitch/head-tilt rendering
PR 6: sector-aware floor/ceiling coloring
PR 7: low walls and windows
PR 8: sector benchmark maps and README docs
```

The key is to keep every PR playable. After PR 2, you should already be able to run:

```bash
python play.py --maze simple --engine sector
```

After PR 4:

```bash
python play.py --maze sector-stairs --engine sector
```

After PR 5:

```bash
python play.py --maze sector-stairs --engine sector --look
```

I’d make the v2 engine feel boringly incremental at first. The payoff comes once the sector maps start producing traversal problems that the current flat Wolfenstein renderer cannot express.

[1]: https://github.com/taodav/jaxenstein "GitHub - taodav/jaxenstein · GitHub"
[2]: https://github.com/taodav/jaxenstein/blob/main/jes/MAZES.md "jaxenstein/jes/MAZES.md at main · taodav/jaxenstein · GitHub"
[3]: https://raw.githubusercontent.com/taodav/jaxenstein/main/jes/render.py "raw.githubusercontent.com"
