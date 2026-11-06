"""First-person raycasting renderer."""

from __future__ import annotations

import colorsys

import jax
import jax.numpy as jnp

from jaxenstein.objects import (
    DOOR_LOCKED_WALL_COLOR_OFFSET,
    DOOR_PANEL_WALL_COLOR_IDS,
    KEY_COLOR_RED,
    KEY_COLOR_YELLOW,
    OBJECT_GOAL,
    OBJECT_NONE,
)
from jaxenstein.sprites import legacy_sprite_atlas


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
        extra.append([round(channel * 255) for channel in rgb])
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
    object_sprite: jax.Array | None = None,
    sprite_atlas: jax.Array | None = None,
    sprite_width_scales: jax.Array | None = None,
    sprite_height_scales: jax.Array | None = None,
    goal_xy: jax.Array | None = None,
    color_palette: jax.Array = DEFAULT_WALL_PALETTE,
    floor_rgb: jax.Array = FLOOR_RGB,
    floor_checker_dark_rgb: jax.Array = FLOOR_CHECKER_DARK_RGB,
    floor_checker_light_rgb: jax.Array = FLOOR_CHECKER_LIGHT_RGB,
    wall_height_scale: float = 1.35,
    floor_pattern: bool = True,
) -> jax.Array:
    """Render a Wolfenstein-style RGB observation as uint8 [H, W, 3]."""

    color_palette = jnp.asarray(color_palette, dtype=jnp.float32)
    floor_rgb = jnp.asarray(floor_rgb, dtype=jnp.float32)
    floor_checker_dark_rgb = jnp.asarray(floor_checker_dark_rgb, dtype=jnp.float32)
    floor_checker_light_rgb = jnp.asarray(floor_checker_light_rgb, dtype=jnp.float32)

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
        jnp.where(wall_mask[..., None], wall_rgb[None, :, :], floor_rgb),
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
            floor_checker_dark_rgb=floor_checker_dark_rgb,
            floor_checker_light_rgb=floor_checker_light_rgb,
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
        if object_sprite is None and object_type is None:
            raise ValueError(
                "object_sprite or legacy object_type is required with object_xy"
            )
        if object_color is None and object_type is not None:
            object_color = jnp.full_like(object_type, KEY_COLOR_RED)
        if object_active is None:
            object_active = (
                jnp.ones(object_xy.shape[0], dtype=jnp.bool_)
                if object_sprite is not None
                else object_type != OBJECT_NONE
            )
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
            object_sprite=object_sprite,
            sprite_atlas=sprite_atlas,
            sprite_width_scales=sprite_width_scales,
            sprite_height_scales=sprite_height_scales,
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
    floor_checker_dark_rgb: jax.Array,
    floor_checker_light_rgb: jax.Array,
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
        floor_checker_dark_rgb,
        floor_checker_light_rgb,
    )
    return jnp.where(floor_mask[..., None], floor_rgb, rgb)


def _apply_door_panel_pattern(
    rgb: jax.Array,
    wall_ids: jax.Array,
    wall_mask: jax.Array,
    *,
    img_w: int,
) -> jax.Array:
    door_cols = jnp.any(wall_ids[:, None] == DOOR_PANEL_WALL_COLOR_IDS[None, :], axis=1)

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
    object_type: jax.Array | None,
    object_color: jax.Array | None,
    object_active: jax.Array,
    wall_depth: jax.Array,
    *,
    img_h: int,
    img_w: int,
    fov: float,
    object_sprite: jax.Array | None = None,
    sprite_atlas: jax.Array | None = None,
    sprite_width_scales: jax.Array | None = None,
    sprite_height_scales: jax.Array | None = None,
) -> jax.Array:
    """Composite cutout RGBA textures without any object-specific drawing."""
    count = object_xy.shape[0]
    if count == 0:
        return rgb
    if sprite_atlas is None:
        if object_type is None or object_color is None:
            raise ValueError(
                "Supply an atlas and sprite IDs, or legacy type/color arrays"
            )
        sprite_atlas = jnp.asarray(legacy_sprite_atlas())
        object_sprite = object_type * 4 + object_color
    if object_sprite is None:
        raise ValueError("object_sprite is required with sprite_atlas")
    rel = object_xy - pos[None, :]
    forward = jnp.asarray([jnp.cos(theta), jnp.sin(theta)], dtype=jnp.float32)
    right = jnp.asarray([-jnp.sin(theta), jnp.cos(theta)], dtype=jnp.float32)
    forward_depth = rel @ forward
    right_depth = rel @ right
    screen_x = right_depth / jnp.maximum(forward_depth * jnp.tan(fov / 2.0), 1.0e-3)
    center_col = (screen_x + 1.0) * 0.5 * (img_w - 1)
    width_scale = (
        0.58 if sprite_width_scales is None else sprite_width_scales[object_sprite]
    )
    height_scale = (
        0.85 if sprite_height_scales is None else sprite_height_scales[object_sprite]
    )
    sprite_h = jnp.clip(
        img_h * height_scale / jnp.maximum(forward_depth, 1.0e-3), 3.0, img_h
    )
    sprite_w = jnp.clip(sprite_h * width_scale, 3.0, img_w)
    rows = jnp.arange(img_h, dtype=jnp.float32)[None, :, None]
    cols = jnp.arange(img_w, dtype=jnp.float32)[None, None, :]
    dx = (cols - center_col[:, None, None]) / jnp.maximum(
        sprite_w[:, None, None] * 0.5, 1.0
    )
    dy = (rows - img_h * 0.5) / jnp.maximum(sprite_h[:, None, None] * 0.5, 1.0)
    texture_h, texture_w = sprite_atlas.shape[1:3]
    u = jnp.clip(
        jnp.floor((dx + 1) * 0.5 * texture_w).astype(jnp.int32), 0, texture_w - 1
    )
    v = jnp.clip(
        jnp.floor((dy + 1) * 0.5 * texture_h).astype(jnp.int32), 0, texture_h - 1
    )
    alpha = sprite_atlas[..., 3][object_sprite[:, None, None], v, u]
    visible = object_active & (forward_depth > 0.05) & (jnp.abs(screen_x) < 1.35)
    not_occluded = forward_depth[:, None, None] < (wall_depth[None, None, :] - 0.05)
    mask = (
        (jnp.abs(dx) <= 1)
        & (jnp.abs(dy) <= 1)
        & (alpha != 0)
        & visible[:, None, None]
        & not_occluded
    )
    depths = jnp.where(mask, forward_depth[:, None, None], jnp.inf)
    nearest = jnp.argmin(depths, axis=0)
    nearest_depth = jnp.take_along_axis(depths, nearest[None], axis=0)[0]
    nearest_u = jnp.take_along_axis(
        jnp.broadcast_to(u, mask.shape), nearest[None], axis=0
    )[0]
    nearest_v = jnp.take_along_axis(
        jnp.broadcast_to(v, mask.shape), nearest[None], axis=0
    )[0]
    sprite_rgb = sprite_atlas[object_sprite[nearest], nearest_v, nearest_u, :3]
    return jnp.where(jnp.isfinite(nearest_depth)[..., None], sprite_rgb, rgb)
