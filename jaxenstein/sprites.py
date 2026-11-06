"""CPU-side sprite builders, called once when constructing an object catalog."""

from functools import lru_cache

import numpy as np

CORE_COLORS = ((0, 0, 0), (255, 72, 64), (80, 160, 255), (255, 220, 48))
EDGE_COLORS = ((0, 0, 0), (140, 28, 32), (32, 80, 190), (48, 220, 96))
SPRITE_RESOLUTION = 64


def coordinates(size=SPRITE_RESOLUTION):
    axis = (np.arange(size, dtype=np.float32) + 0.5) * (2.0 / size) - 1.0
    return np.meshgrid(axis, axis)


def rgba(mask, core, edge_color, core_color):
    image = np.zeros((*mask.shape, 4), dtype=np.uint8)
    image[..., :3] = np.where(core[..., None], core_color, edge_color)
    image[..., 3] = np.where(mask, 255, 0)
    return image


def goal_sprite(color=3):
    x, y = coordinates()
    radius = x * x + y * y
    return rgba(
        radius <= 1,
        (radius <= 0.32) | (np.abs(x) < 0.16),
        EDGE_COLORS[color],
        CORE_COLORS[color],
    )


def key_sprite(color=1):
    x, y = coordinates()
    head_dist = (x + 0.48) ** 2 + (y + 0.02) ** 2
    head = (head_dist <= 0.32**2) & (head_dist >= 0.17**2)
    head_core = (head_dist <= 0.26**2) & (head_dist >= 0.21**2)
    shaft = (x > -0.22) & (x < 0.56) & (np.abs(y + 0.02) < 0.085)
    shaft_core = (x > -0.16) & (x < 0.50) & (np.abs(y + 0.02) < 0.040)
    bit = (x > 0.32) & (x < 0.62) & (y > 0.02) & (y < 0.21)
    notch = (x > 0.44) & (x < 0.53) & (y > 0.11) & (y < 0.23)
    bit_core = (x > 0.38) & (x < 0.57) & (y > 0.05) & (y < 0.16)
    return rgba(
        head | shaft | (bit & ~notch),
        head_core | shaft_core | bit_core,
        (44, 28, 24),
        CORE_COLORS[color],
    )


def medkit_sprite():
    x, y = coordinates()
    mask = (np.abs(x) <= 0.78) & (np.abs(y) <= 0.52)
    core = ((np.abs(x) <= 0.15) & (np.abs(y) <= 0.42)) | (
        (np.abs(y) <= 0.15) & (np.abs(x) <= 0.48)
    )
    return rgba(mask, core, (238, 238, 222), (214, 28, 28))


@lru_cache(maxsize=1)
def legacy_sprite_atlas():
    """Compatibility for direct renderer calls using the original numeric IDs.

    Environment instances supply their own compact atlas instead. No
    environment-specific objects are registered or generated here.
    """
    blank = np.zeros((SPRITE_RESOLUTION, SPRITE_RESOLUTION, 4), dtype=np.uint8)
    return np.stack(
        [blank] * 4
        + [goal_sprite(color) for color in range(4)]
        + [key_sprite(color) for color in range(4)]
        + [medkit_sprite()] * 4
    )
