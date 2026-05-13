"""Object type constants for maze pickups and billboard sprites."""

from __future__ import annotations

import jax
import jax.numpy as jnp


OBJECT_NONE = 0
OBJECT_GOAL = 1
OBJECT_KEY = 2
OBJECT_MEDKIT = 3

KEY_COLOR_NONE = 0
KEY_COLOR_RED = 1
KEY_COLOR_BLUE = 2
KEY_COLOR_YELLOW = 3
NUM_KEY_COLORS = 4
DOOR_UNLOCKED = -KEY_COLOR_BLUE
DOOR_UNLOCKED_BLUE = -KEY_COLOR_BLUE
DOOR_UNLOCKED_YELLOW = -KEY_COLOR_YELLOW

OBJECT_SYMBOLS = {
    "G": (OBJECT_GOAL, KEY_COLOR_YELLOW),
    "K": (OBJECT_KEY, KEY_COLOR_RED),
    "r": (OBJECT_KEY, KEY_COLOR_RED),
    "b": (OBJECT_KEY, KEY_COLOR_BLUE),
    "y": (OBJECT_KEY, KEY_COLOR_YELLOW),
}

DOOR_SYMBOLS = {
    '"': DOOR_UNLOCKED,
    "\\": DOOR_UNLOCKED_YELLOW,
    "D": KEY_COLOR_RED,
    "R": KEY_COLOR_RED,
    "B": KEY_COLOR_BLUE,
    "Y": KEY_COLOR_YELLOW,
}

COLORED_WALL_SYMBOLS = (
    "1234567890!$%&()*+,-/:;<=>?@[]^_`{|}~'"
    "acdefghijklmnopqstuvwxzACEFHIJLMNOPQTUVWXZ"
)
WALL_SYMBOLS = {"#": 1} | {
    symbol: 7 + index for index, symbol in enumerate(COLORED_WALL_SYMBOLS)
}

OBJECT_CORE_PALETTE_BY_COLOR = jnp.asarray(
    [
        [0, 0, 0],
        [255, 72, 64],
        [80, 160, 255],
        [255, 220, 48],
    ],
    dtype=jnp.float32,
)
OBJECT_EDGE_PALETTE_BY_COLOR = jnp.asarray(
    [
        [0, 0, 0],
        [140, 28, 32],
        [32, 80, 190],
        [48, 220, 96],
    ],
    dtype=jnp.float32,
)
OBJECT_PICKUP_RADIUS = jnp.asarray([0.0, 0.35, 0.35, 0.60], dtype=jnp.float32)
DOOR_WALL_COLOR_IDS = jnp.asarray([0, 4, 5, 6], dtype=jnp.int32)
DOOR_PANEL_WALL_COLOR_IDS = jnp.asarray([4, 5, 6], dtype=jnp.int32)
DOOR_LOCKED_WALL_COLOR_OFFSET = 100


def object_pickup_radius(object_type: jax.Array) -> jax.Array:
    return jnp.take(OBJECT_PICKUP_RADIUS, object_type, mode="clip")


def door_wall_color_id(key_color: jax.Array) -> jax.Array:
    door_color = jnp.abs(key_color)
    color_id = jnp.take(DOOR_WALL_COLOR_IDS, door_color, mode="clip")
    return jnp.where(key_color == KEY_COLOR_NONE, 0, color_id)
