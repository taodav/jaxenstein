"""Hand-authored benchmark maps."""

MAZE_SIMPLE = """
#########
#S.....G#
#.#####.#
#.......#
#########
"""

MAZE_KEY_DOOR = """
###########
#S.r.R...G#
###########
"""

MAZE_STRAIGHT_CORRIDOR = """
###########
#S.......G#
###########
"""

MAZE_T = """
###########
#....G....#
#####.#####
#####.#####
#####S#####
###########
"""

MAPS_BY_NAME = {
    "simple": MAZE_SIMPLE,
    "key-door": MAZE_KEY_DOOR,
}

DEFAULT_MAZES = [MAZE_SIMPLE]
