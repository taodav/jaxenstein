"""Fixed-step first-person raycasting renderer."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jes.objects import (
    DOOR_WALL_COLOR_IDS,
    KEY_COLOR_RED,
    KEY_COLOR_YELLOW,
    OBJECT_GOAL,
    OBJECT_KEY,
    OBJECT_NONE,
    OBJECT_CORE_PALETTE_BY_COLOR,
    OBJECT_EDGE_PALETTE_BY_COLOR,
)


DEFAULT_WALL_PALETTE = jnp.asarray(
    [
        [150, 150, 150],
        [126, 132, 144],
        [72, 128, 96],
        [72, 96, 160],
        [216, 56, 48],
        [56, 112, 224],
        [224, 192, 48],
    ],
    dtype=jnp.float32,
)
CEILING_RGB = jnp.asarray([70, 70, 90], dtype=jnp.float32)
FLOOR_RGB = jnp.asarray([45, 45, 45], dtype=jnp.float32)


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
) -> jax.Array:
    """Render a Wolfenstein-style RGB observation as uint8 [H, W, 3]."""

    distances, wall_ids = raycast_fixed_step(
        pos,
        theta,
        wall_grid,
        color_grid,
        img_w=img_w,
        fov=fov,
        max_depth=max_depth,
        num_depth_samples=num_depth_samples,
    )
    ray_angles = _ray_angles(theta, img_w, fov)
    corrected = jnp.maximum(distances * jnp.cos(ray_angles - theta), 1.0e-3)
    wall_height = img_h / corrected
    wall_top = img_h / 2.0 - wall_height / 2.0
    wall_bottom = img_h / 2.0 + wall_height / 2.0

    rows = jnp.arange(img_h, dtype=jnp.float32)[:, None]
    wall_mask = (rows >= wall_top[None, :]) & (rows <= wall_bottom[None, :])
    ceiling_mask = rows < wall_top[None, :]

    base_wall = jnp.take(color_palette, wall_ids, axis=0, mode="clip")
    shade = 1.0 / (1.0 + 0.1 * distances**2)
    wall_rgb = base_wall * shade[:, None]

    rgb = jnp.where(
        ceiling_mask[..., None],
        CEILING_RGB,
        jnp.where(wall_mask[..., None], wall_rgb[None, :, :], FLOOR_RGB),
    )
    rgb = _apply_door_panel_pattern(
        rgb,
        wall_ids,
        wall_mask,
        wall_top,
        wall_bottom,
        img_h=img_h,
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


def _apply_door_panel_pattern(
    rgb: jax.Array,
    wall_ids: jax.Array,
    wall_mask: jax.Array,
    wall_top: jax.Array,
    wall_bottom: jax.Array,
    *,
    img_h: int,
    img_w: int,
) -> jax.Array:
    door_ids = DOOR_WALL_COLOR_IDS[1:]
    door_cols = jnp.any(wall_ids[:, None] == door_ids[None, :], axis=1)

    rows = jnp.arange(img_h, dtype=jnp.float32)[:, None]
    cols = jnp.arange(img_w, dtype=jnp.float32)[None, :]
    wall_mid = (wall_top + wall_bottom) * 0.5

    vertical_panel = jnp.mod(cols + 2.0, 9.0) < 2.0
    horizontal_panel = jnp.abs(rows - wall_mid[None, :]) < 1.5
    door_pixels = wall_mask & door_cols[None, :]
    panel_pixels = door_pixels & (vertical_panel | horizontal_panel)
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

    head_dist = (dx + 0.42) ** 2 + (dy + 0.04) ** 2
    key_head = (head_dist <= 0.30**2) & (head_dist >= 0.13**2)
    key_head_core = (head_dist <= 0.24**2) & (head_dist >= 0.17**2)
    key_shaft = (dx > -0.16) & (dx < 0.60) & (jnp.abs(dy + 0.04) < 0.10)
    key_shaft_core = (dx > -0.10) & (dx < 0.54) & (jnp.abs(dy + 0.04) < 0.055)
    key_tooth_a = (dx > 0.30) & (dx < 0.46) & (dy > 0.02) & (dy < 0.32)
    key_tooth_b = (dx > 0.48) & (dx < 0.64) & (dy > 0.02) & (dy < 0.24)
    key_tooth_core = (dx > 0.34) & (dx < 0.60) & (dy > 0.04) & (dy < 0.21)
    key_mask = key_head | key_shaft | key_tooth_a | key_tooth_b
    key_core = key_head_core | key_shaft_core | key_tooth_core

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
    sprite_rgb = jnp.where(nearest_core[..., None], core_rgb, edge_rgb)
    return jnp.where(has_sprite[..., None], sprite_rgb, rgb)
