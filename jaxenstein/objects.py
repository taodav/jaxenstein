"""Environment-local object definitions and compatibility constants."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from jaxenstein.sprites import goal_sprite, key_sprite

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
    "r": (OBJECT_KEY, KEY_COLOR_RED),
    "b": (OBJECT_KEY, KEY_COLOR_BLUE),
    "y": (OBJECT_KEY, KEY_COLOR_YELLOW),
}

OBJECT_NAMES_BY_SYMBOL = {
    "G": "goal",
    "r": "red_key",
    "b": "blue_key",
    "y": "yellow_key",
}

DOOR_SYMBOLS = {
    '"': DOOR_UNLOCKED,
    "\\": DOOR_UNLOCKED_YELLOW,
    "R": KEY_COLOR_RED,
    "B": KEY_COLOR_BLUE,
    "Y": KEY_COLOR_YELLOW,
}

COLORED_WALL_SYMBOLS = (
    "1234567890!$%&()*+,-/:;<=>?@[]^_`{|}~'acdefghijklmnopqstuvwxzACEFHIJLMNOPQTUVWXZ"
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


@dataclass(frozen=True)
class ObjectSpec:
    """Properties shared by instances of one environment-defined object.

    Sprite data is uint8 [H, W, 4] RGBA, or a zero-argument CPU-side builder.
    Builders for unused definitions are never called. consume=False permits
    repeated collection events; custom effects can use info["picked_objects"].
    """

    sprite: np.ndarray | Callable[[], np.ndarray]
    reward: float = 0.0
    pickup_radius: float = 0.35
    terminal: bool = False
    consume: bool = True
    is_goal: bool = False
    inventory_key: int = KEY_COLOR_NONE
    width_scale: float = 0.58
    height_scale: float = 0.85


@dataclass(frozen=True)
class ObjectPlacement:
    """One object slot at an (x, y) world position, separate from map geometry."""

    kind: str
    pos: tuple[float, float]
    active: bool = True


@dataclass(frozen=True)
class ObjectCatalog:
    names: tuple[str, ...]
    sprites: jax.Array
    rewards: jax.Array
    pickup_radii: jax.Array
    terminal: jax.Array
    consume: jax.Array
    is_goal: jax.Array
    inventory_keys: jax.Array
    width_scales: jax.Array
    height_scales: jax.Array
    has_goals: bool
    has_keys: bool

    def id(self, name: str) -> int:
        return self.names.index(name)


def default_object_spec(name: str, *, goal_reward: float = 1.0) -> ObjectSpec:
    if name == "goal":
        return ObjectSpec(goal_sprite, reward=goal_reward, terminal=True, is_goal=True)
    keys = {
        "red_key": KEY_COLOR_RED,
        "blue_key": KEY_COLOR_BLUE,
        "yellow_key": KEY_COLOR_YELLOW,
    }
    if name in keys:
        return ObjectSpec(partial(key_sprite, keys[name]), inventory_key=keys[name])
    raise ValueError(f"Unknown object type {name!r}; provide it in object_types")


def compile_object_catalog(
    names: Sequence[str],
    object_types: Mapping[str, ObjectSpec] | None = None,
    *,
    goal_reward: float = 1.0,
) -> ObjectCatalog:
    """Materialize only the types referenced by this environment instance."""
    names = tuple(dict.fromkeys(names))
    definitions = {} if object_types is None else object_types
    specs, sprites = [], []
    for name in names:
        spec = (
            definitions[name]
            if name in definitions
            else default_object_spec(name, goal_reward=goal_reward)
        )
        if not isinstance(spec, ObjectSpec):
            raise TypeError(f"Object definition {name!r} must be an ObjectSpec")
        if (
            not np.isfinite(spec.reward)
            or not np.isfinite(spec.pickup_radius)
            or spec.pickup_radius < 0
        ):
            raise ValueError(
                "Object rewards must be finite and pickup radii finite and nonnegative"
            )
        if not 0 <= spec.inventory_key < NUM_KEY_COLORS:
            raise ValueError("inventory_key must be a supported key color")
        if (
            not np.isfinite(spec.width_scale)
            or not np.isfinite(spec.height_scale)
            or min(spec.width_scale, spec.height_scale) <= 0
        ):
            raise ValueError("Sprite scales must be positive and finite")
        sprite = np.asarray(spec.sprite() if callable(spec.sprite) else spec.sprite)
        if (
            sprite.ndim != 3
            or sprite.shape[-1] != 4
            or min(sprite.shape[:2]) < 1
            or sprite.dtype != np.uint8
        ):
            raise ValueError("Sprites must be nonempty uint8 [H, W, 4] RGBA arrays")
        if not np.all((sprite[..., 3] == 0) | (sprite[..., 3] == 255)):
            raise ValueError("Sprite alpha must be 0 or 255 (cutout transparency)")
        specs.append(spec)
        sprites.append(sprite)
    # Normalize texture resolution once on the CPU; no per-type drawing runs
    # in reset/step/render. An empty catalog has no dummy object or sprite.
    height = max((s.shape[0] for s in sprites), default=1)
    width = max((s.shape[1] for s in sprites), default=1)
    atlas = np.empty((len(sprites), height, width, 4), dtype=np.uint8)
    for i, sprite in enumerate(sprites):
        ys = np.minimum(
            np.arange(height) * sprite.shape[0] // height, sprite.shape[0] - 1
        )
        xs = np.minimum(
            np.arange(width) * sprite.shape[1] // width, sprite.shape[1] - 1
        )
        atlas[i] = sprite[ys[:, None], xs[None, :]]

    def array(field, dtype):
        return jnp.asarray([getattr(spec, field) for spec in specs], dtype=dtype)

    return ObjectCatalog(
        names=names,
        sprites=jnp.asarray(atlas),
        rewards=array("reward", jnp.float32),
        pickup_radii=array("pickup_radius", jnp.float32),
        terminal=array("terminal", jnp.bool_),
        consume=array("consume", jnp.bool_),
        is_goal=array("is_goal", jnp.bool_),
        inventory_keys=array("inventory_key", jnp.int32),
        width_scales=array("width_scale", jnp.float32),
        height_scales=array("height_scale", jnp.float32),
        has_goals=any(s.is_goal for s in specs),
        has_keys=any(s.inventory_key != 0 for s in specs),
    )
