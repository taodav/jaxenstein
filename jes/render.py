"""First-person raycasting renderer."""

from __future__ import annotations

import colorsys

import jax
import jax.numpy as jnp

from jes.objects import (
    DOOR_LOCKED_WALL_COLOR_OFFSET,
    DOOR_PANEL_WALL_COLOR_IDS,
    KEY_COLOR_RED,
    KEY_COLOR_YELLOW,
    OBJECT_GOAL,
    OBJECT_KEY,
    OBJECT_NONE,
    OBJECT_CORE_PALETTE_BY_COLOR,
    OBJECT_EDGE_PALETTE_BY_COLOR,
)


def _make_wall_palette(size: int = 96) -> jax.Array:
    base = [
        [150, 150, 150],
        [176, 124, 78],
        [74, 166, 104],
        [78, 120, 206],
        [236, 44, 36],
        [48, 126, 255],
        [255, 212, 38],
        [138, 64, 216],
        [0, 168, 180],
        [230, 96, 32],
        [214, 56, 126],
        [112, 196, 48],
        [96, 76, 42],
        [232, 232, 220],
        [34, 72, 160],
        [184, 80, 46],
    ]
    # Extra colors cover generated DMLab decal wall ids.
    extra = []
    for index in range(size - len(base)):
        hue = (0.07 + index * 0.61803398875) % 1.0
        saturation = 0.72 + 0.18 * ((index % 3) / 2.0)
        value = 0.78 + 0.14 * ((index + 1) % 2)
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        extra.append(
            [round(channel * 255) for channel in rgb]
        )
    return jnp.asarray(base + extra, dtype=jnp.float32)


DEFAULT_WALL_PALETTE = _make_wall_palette()
CEILING_RGB = jnp.asarray([184, 218, 238], dtype=jnp.float32)
FLOOR_RGB = jnp.asarray([54, 47, 38], dtype=jnp.float32)
FLOOR_CHECKER_DARK_RGB = jnp.asarray([48, 42, 34], dtype=jnp.float32)
FLOOR_CHECKER_LIGHT_RGB = jnp.asarray([126, 104, 70], dtype=jnp.float32)


def _lookup_grid(grid: jax.Array, cell_x: jax.Array, cell_y: jax.Array) -> jax.Array:
    h, w = grid.shape
    clipped_x = jnp.clip(cell_x, 0, w - 1)
    clipped_y = jnp.clip(cell_y, 0, h - 1)
    return grid[clipped_y, clipped_x]


def raycast_fixed_step(
    pos: jax.Array,
    theta: jax.Array,
    wall_grid: jax.Array,
    color_grid: jax.Array,
    *,
    img_w: int = 64,
    fov: float = jnp.pi / 3.0,
    max_depth: float = 16.0,
    num_depth_samples: int = 128,
) -> tuple[jax.Array, jax.Array]:
    """Cast one ray per image column and return hit distance and wall id."""

    camera_x = jnp.linspace(-1.0, 1.0, img_w, dtype=jnp.float32)
    ray_angles = theta + jnp.arctan(camera_x * jnp.tan(fov / 2.0))
    ray_dirs = jnp.stack([jnp.cos(ray_angles), jnp.sin(ray_angles)], axis=-1)

    depths = jnp.linspace(
        max_depth / num_depth_samples,
        max_depth,
        num_depth_samples,
        dtype=jnp.float32,
    )
    points = pos[None, None, :] + ray_dirs[:, None, :] * depths[None, :, None]
    cell_x = jnp.floor(points[..., 0]).astype(jnp.int32)
    cell_y = jnp.floor(points[..., 1]).astype(jnp.int32)

    h, w = wall_grid.shape
    inside = (cell_x >= 0) & (cell_x < w) & (cell_y >= 0) & (cell_y < h)
    wall = _lookup_grid(wall_grid, cell_x, cell_y) | ~inside

    hit_any = jnp.any(wall, axis=1)
    first_hit = jnp.argmax(wall.astype(jnp.int32), axis=1)
    distances = jnp.take(depths, first_hit)
    distances = jnp.where(hit_any, distances, max_depth)

    hit_x = jnp.take_along_axis(cell_x, first_hit[:, None], axis=1)[:, 0]
    hit_y = jnp.take_along_axis(cell_y, first_hit[:, None], axis=1)[:, 0]
    wall_ids = _lookup_grid(color_grid, hit_x, hit_y).astype(jnp.int32)
    wall_ids = jnp.where(hit_any, wall_ids, 0)
    return distances, wall_ids


def raycast_dda(
    pos: jax.Array,
    theta: jax.Array,
    wall_grid: jax.Array,
    color_grid: jax.Array,
    *,
    img_w: int = 64,
    fov: float = jnp.pi / 3.0,
    max_depth: float = 16.0,
    max_steps: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Cast one ray per image column with exact grid-cell intersections."""

    if max_steps is None:
        max_steps = int(max_depth * 2.0) + 4

    ray_angles = _ray_angles(theta, img_w, fov)
    ray_dirs = jnp.stack([jnp.cos(ray_angles), jnp.sin(ray_angles)], axis=-1)
    ray_dir_x = ray_dirs[:, 0]
    ray_dir_y = ray_dirs[:, 1]

    cell_x = jnp.floor(pos[0]).astype(jnp.int32) + jnp.zeros((img_w,), dtype=jnp.int32)
    cell_y = jnp.floor(pos[1]).astype(jnp.int32) + jnp.zeros((img_w,), dtype=jnp.int32)
    step_x = jnp.where(ray_dir_x < 0.0, -1, 1).astype(jnp.int32)
    step_y = jnp.where(ray_dir_y < 0.0, -1, 1).astype(jnp.int32)

    delta_x = jnp.where(
        jnp.abs(ray_dir_x) > 1.0e-6,
        jnp.abs(1.0 / ray_dir_x),
        jnp.inf,
    )
    delta_y = jnp.where(
        jnp.abs(ray_dir_y) > 1.0e-6,
        jnp.abs(1.0 / ray_dir_y),
        jnp.inf,
    )
    side_dist_x = jnp.where(
        ray_dir_x < 0.0,
        (pos[0] - cell_x.astype(jnp.float32)) * delta_x,
        (cell_x.astype(jnp.float32) + 1.0 - pos[0]) * delta_x,
    )
    side_dist_y = jnp.where(
        ray_dir_y < 0.0,
        (pos[1] - cell_y.astype(jnp.float32)) * delta_y,
        (cell_y.astype(jnp.float32) + 1.0 - pos[1]) * delta_y,
    )

    hit = jnp.zeros((img_w,), dtype=jnp.bool_)
    hit_dist = jnp.full((img_w,), max_depth, dtype=jnp.float32)
    hit_x = cell_x
    hit_y = cell_y
    h, w = wall_grid.shape

    def body(
        _: int,
        state: tuple[
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
            jax.Array,
        ],
    ) -> tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ]:
        cell_x, cell_y, side_dist_x, side_dist_y, hit, hit_dist, hit_x, hit_y = state
        along_x = side_dist_x < side_dist_y
        next_dist = jnp.where(along_x, side_dist_x, side_dist_y)
        active = (~hit) & (next_dist <= max_depth)

        next_cell_x = cell_x + jnp.where(along_x, step_x, 0)
        next_cell_y = cell_y + jnp.where(along_x, 0, step_y)
        inside = (
            (next_cell_x >= 0)
            & (next_cell_x < w)
            & (next_cell_y >= 0)
            & (next_cell_y < h)
        )
        blocked = _lookup_grid(wall_grid, next_cell_x, next_cell_y) | ~inside
        new_hit = active & blocked

        cell_x = jnp.where(active, next_cell_x, cell_x)
        cell_y = jnp.where(active, next_cell_y, cell_y)
        side_dist_x = jnp.where(active & along_x, side_dist_x + delta_x, side_dist_x)
        side_dist_y = jnp.where(active & ~along_x, side_dist_y + delta_y, side_dist_y)
        hit_dist = jnp.where(new_hit, next_dist, hit_dist)
        hit_x = jnp.where(new_hit, next_cell_x, hit_x)
        hit_y = jnp.where(new_hit, next_cell_y, hit_y)
        hit = hit | new_hit
        return cell_x, cell_y, side_dist_x, side_dist_y, hit, hit_dist, hit_x, hit_y

    _, _, _, _, hit, distances, hit_x, hit_y = jax.lax.fori_loop(
        0,
        max_steps,
        body,
        (cell_x, cell_y, side_dist_x, side_dist_y, hit, hit_dist, hit_x, hit_y),
    )
    wall_ids = _lookup_grid(color_grid, hit_x, hit_y).astype(jnp.int32)
    wall_ids = jnp.where(hit, wall_ids, 0)
    distances = jnp.where(hit, distances, max_depth)
    return distances, wall_ids


def _ray_angles(theta: jax.Array, img_w: int, fov: float) -> jax.Array:
    camera_x = jnp.linspace(-1.0, 1.0, img_w, dtype=jnp.float32)
    return theta + jnp.arctan(camera_x * jnp.tan(fov / 2.0))


def render_first_person(
    pos: jax.Array,
    theta: jax.Array,
    wall_grid: jax.Array,
    color_grid: jax.Array,
    *,
    img_h: int = 64,
    img_w: int = 64,
    fov: float = jnp.pi / 3.0,
    max_depth: float = 16.0,
    num_depth_samples: int = 128,
    object_xy: jax.Array | None = None,
    object_type: jax.Array | None = None,
    object_color: jax.Array | None = None,
    object_active: jax.Array | None = None,
    goal_xy: jax.Array | None = None,
    color_palette: jax.Array = DEFAULT_WALL_PALETTE,
    wall_height_scale: float = 1.35,
    floor_pattern: bool = True,
) -> jax.Array:
    """Render a Wolfenstein-style RGB observation as uint8 [H, W, 3]."""

    distances, wall_ids = raycast_dda(
        pos,
        theta,
        wall_grid,
        color_grid,
        img_w=img_w,
        fov=fov,
        max_depth=max_depth,
    )
    locked_door_cols = wall_ids >= DOOR_LOCKED_WALL_COLOR_OFFSET
    wall_ids = jnp.where(
        locked_door_cols,
        wall_ids - DOOR_LOCKED_WALL_COLOR_OFFSET,
        wall_ids,
    )
    ray_angles = _ray_angles(theta, img_w, fov)
    corrected = jnp.maximum(distances * jnp.cos(ray_angles - theta), 1.0e-3)
    wall_height = img_h * wall_height_scale / corrected
    wall_top = img_h / 2.0 - wall_height / 2.0
    wall_bottom = img_h / 2.0 + wall_height / 2.0

    rows = jnp.arange(img_h, dtype=jnp.float32)[:, None]
    wall_mask = (rows >= wall_top[None, :]) & (rows <= wall_bottom[None, :])
    ceiling_mask = rows < wall_top[None, :]

    base_wall = jnp.take(color_palette, wall_ids, axis=0, mode="clip")
    shade = 0.35 + 0.65 / (1.0 + 0.05 * distances**2)
    wall_rgb = base_wall * shade[:, None]

    rgb = jnp.where(
        ceiling_mask[..., None],
        CEILING_RGB,
        jnp.where(wall_mask[..., None], wall_rgb[None, :, :], FLOOR_RGB),
    )
    rgb = jax.lax.cond(
        jnp.asarray(floor_pattern),
        lambda image: _apply_floor_checker_pattern(
            image,
            pos,
            theta,
            ray_angles,
            wall_mask,
            ceiling_mask,
            img_h=img_h,
            wall_height_scale=wall_height_scale,
        ),
        lambda image: image,
        rgb,
    )
    rgb = _apply_door_panel_pattern(
        rgb,
        wall_ids,
        wall_mask,
        img_w=img_w,
    )
    if object_xy is None and goal_xy is not None:
        object_xy = goal_xy[None, :]
        object_type = jnp.asarray([OBJECT_GOAL], dtype=jnp.int32)
        object_color = jnp.asarray([KEY_COLOR_YELLOW], dtype=jnp.int32)
        object_active = jnp.asarray([True])

    if object_xy is not None:
        if object_type is None:
            raise ValueError("object_type is required when object_xy is provided")
        if object_color is None:
            object_color = jnp.full_like(object_type, KEY_COLOR_RED)
        if object_active is None:
            object_active = object_type != OBJECT_NONE
        rgb = render_billboard_sprites(
            rgb,
            pos,
            theta,
            object_xy,
            object_type,
            object_color,
            object_active,
            corrected,
            img_h=img_h,
            img_w=img_w,
            fov=fov,
        )
    return jnp.clip(rgb, 0, 255).astype(jnp.uint8)


def _apply_floor_checker_pattern(
    rgb: jax.Array,
    pos: jax.Array,
    theta: jax.Array,
    ray_angles: jax.Array,
    wall_mask: jax.Array,
    ceiling_mask: jax.Array,
    *,
    img_h: int,
    wall_height_scale: float,
    checker_size: float = 1.0,
) -> jax.Array:
    rows = jnp.arange(img_h, dtype=jnp.float32)[:, None]
    below_horizon = rows > (img_h / 2.0)
    floor_mask = below_horizon & ~wall_mask & ~ceiling_mask

    vertical = jnp.maximum(2.0 * rows - img_h, 1.0)
    perp_depth = (img_h * wall_height_scale) / vertical
    ray_depth = perp_depth / jnp.maximum(jnp.cos(ray_angles - theta)[None, :], 1.0e-3)
    floor_x = pos[0] + jnp.cos(ray_angles)[None, :] * ray_depth
    floor_y = pos[1] + jnp.sin(ray_angles)[None, :] * ray_depth
    checker = jnp.mod(
        jnp.floor(floor_x / checker_size) + jnp.floor(floor_y / checker_size),
        2.0,
    )
    floor_rgb = jnp.where(
        checker[..., None] < 1.0,
        FLOOR_CHECKER_DARK_RGB,
        FLOOR_CHECKER_LIGHT_RGB,
    )
    return jnp.where(floor_mask[..., None], floor_rgb, rgb)


def _apply_door_panel_pattern(
    rgb: jax.Array,
    wall_ids: jax.Array,
    wall_mask: jax.Array,
    *,
    img_w: int,
) -> jax.Array:
    door_cols = jnp.any(
        wall_ids[:, None] == DOOR_PANEL_WALL_COLOR_IDS[None, :], axis=1
    )

    cols = jnp.arange(img_w, dtype=jnp.float32)[None, :]
    vertical_panel = jnp.mod(cols + 2.0, 9.0) < 2.0
    door_pixels = wall_mask & door_cols[None, :]
    panel_pixels = door_pixels & vertical_panel
    return jnp.where(panel_pixels[..., None], rgb * 0.42 + 28.0, rgb)


def render_billboard_sprites(
    rgb: jax.Array,
    pos: jax.Array,
    theta: jax.Array,
    object_xy: jax.Array,
    object_type: jax.Array,
    object_color: jax.Array,
    object_active: jax.Array,
    wall_depth: jax.Array,
    *,
    img_h: int,
    img_w: int,
    fov: float,
) -> jax.Array:
    """Composite camera-facing object sprites over an RGB raycast image."""

    rel = object_xy - pos[None, :]
    forward = jnp.asarray([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.float32)
    right = jnp.asarray([-jnp.sin(theta), jnp.cos(theta)], dtype=jnp.float32)
    forward_depth = rel @ forward
    right_depth = rel @ right

    half_fov_tan = jnp.tan(fov / 2.0)
    screen_x = right_depth / jnp.maximum(forward_depth * half_fov_tan, 1.0e-3)
    center_col = (screen_x + 1.0) * 0.5 * (img_w - 1)
    center_row = img_h * 0.50
    sprite_h = jnp.clip(img_h * 0.85 / jnp.maximum(forward_depth, 1.0e-3), 3.0, img_h)
    sprite_w = jnp.clip(sprite_h * 0.58, 3.0, img_w)

    rows = jnp.arange(img_h, dtype=jnp.float32)[None, :, None]
    cols = jnp.arange(img_w, dtype=jnp.float32)[None, None, :]
    dx = (cols - center_col[:, None, None]) / jnp.maximum(
        sprite_w[:, None, None] * 0.5, 1.0
    )
    dy = (rows - center_row) / jnp.maximum(sprite_h[:, None, None] * 0.5, 1.0)
    radius_sq = dx**2 + dy**2

    goal_mask = radius_sq <= 1.0
    goal_core = (radius_sq <= 0.32) | (jnp.abs(dx) < 0.16)

    head_dist = (dx + 0.48) ** 2 + (dy + 0.02) ** 2
    key_head = (head_dist <= 0.32**2) & (head_dist >= 0.17**2)
    key_head_core = (head_dist <= 0.26**2) & (head_dist >= 0.21**2)
    key_shaft = (dx > -0.22) & (dx < 0.56) & (jnp.abs(dy + 0.02) < 0.085)
    key_shaft_core = (dx > -0.16) & (dx < 0.50) & (jnp.abs(dy + 0.02) < 0.040)
    key_bit = (dx > 0.32) & (dx < 0.62) & (dy > 0.02) & (dy < 0.21)
    key_bit_notch = (dx > 0.44) & (dx < 0.53) & (dy > 0.11) & (dy < 0.23)
    key_bit_core = (dx > 0.38) & (dx < 0.57) & (dy > 0.05) & (dy < 0.16)
    key_mask = key_head | key_shaft | (key_bit & ~key_bit_notch)
    key_core = key_head_core | key_shaft_core | key_bit_core

    is_key = object_type[:, None, None] == OBJECT_KEY
    object_mask = jnp.where(is_key, key_mask, goal_mask)
    core_mask = jnp.where(is_key, key_core, goal_core)
    visible = (
        object_active
        & (object_type != OBJECT_NONE)
        & (forward_depth > 0.05)
        & (jnp.abs(screen_x) < 1.35)
    )
    not_occluded = forward_depth[:, None, None] < (wall_depth[None, None, :] - 0.05)
    sprite_mask = object_mask & visible[:, None, None] & not_occluded

    sprite_depth = jnp.where(sprite_mask, forward_depth[:, None, None], jnp.inf)
    nearest_idx = jnp.argmin(sprite_depth, axis=0)
    nearest_depth = jnp.take_along_axis(sprite_depth, nearest_idx[None, :, :], axis=0)[0]
    has_sprite = jnp.isfinite(nearest_depth)

    nearest_type = jnp.take(object_type, nearest_idx, mode="clip")
    nearest_core = jnp.take_along_axis(core_mask, nearest_idx[None, :, :], axis=0)[0]
    nearest_color = jnp.take(object_color, nearest_idx, mode="clip")
    edge_rgb = jnp.take(OBJECT_EDGE_PALETTE_BY_COLOR, nearest_color, axis=0, mode="clip")
    core_rgb = jnp.take(OBJECT_CORE_PALETTE_BY_COLOR, nearest_color, axis=0, mode="clip")
    key_edge_rgb = jnp.asarray([44, 28, 24], dtype=jnp.float32)
    edge_rgb = jnp.where(nearest_type[..., None] == OBJECT_KEY, key_edge_rgb, edge_rgb)
    sprite_rgb = jnp.where(nearest_core[..., None], core_rgb, edge_rgb)
    return jnp.where(has_sprite[..., None], sprite_rgb, rgb)
