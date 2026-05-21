"""Map definitions and map-specific defaults."""

MAZE_SIMPLE = """
#########
#S......#
#.#####.#
#......G#
#########
"""

MAZE_KEY_DOOR = """
###########
#S.r.R...G#
###########
"""

MINIGRID_KEY_CORRIDOR_S4R3_MAX_STEPS = 30 * 4**2

MAZE_KEY_CORRIDOR = r"""
##########
#..#.."..#
#r."..#..#
####..####
#..\S.RG.#
#..#..#..#
####..####
#.."..\..#
#..#..#..#
##########
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

MY_WAY_HOME_EPISODE_TIMEOUT = 2100
HEALTH_GATHERING_EPISODE_TIMEOUT = 2100
HEALTH_GATHERING_LIVING_REWARD = 1.0
HEALTH_GATHERING_DEATH_PENALTY = 100.0
HEALTH_GATHERING_INITIAL_HEALTH = 100.0
HEALTH_GATHERING_MAX_HEALTH = 100.0
HEALTH_GATHERING_MEDKIT_HEAL = 25.0
HEALTH_GATHERING_INITIAL_MEDKITS = 16
HEALTH_GATHERING_MEDKIT_SPAWN_INTERVAL = 30
HEALTH_GATHERING_ACID_DAMAGE = 8.0
HEALTH_GATHERING_ACID_DAMAGE_INTERVAL = 35
HEALTH_GATHERING_MAX_MEDKITS = 96
HEALTH_GATHERING_WALL_RGB = [118, 118, 112]
HEALTH_GATHERING_FLOOR_RGB = [32, 112, 28]
HEALTH_GATHERING_FLOOR_CHECKER_DARK_RGB = [22, 76, 26]
HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB = [76, 150, 44]


def _rectangle_map(width: int, height: int, spawn_col: int, spawn_row: int) -> str:
    rows = []
    for row_idx in range(height):
        if row_idx == 0 or row_idx == height - 1:
            rows.append("#" * width)
            continue

        row = ["#"] + ["."] * (width - 2) + ["#"]
        if row_idx == spawn_row:
            row[spawn_col] = "S"
        rows.append("".join(row))
    return "\n".join(rows)


# The health-gathering room uses a 38-cell play area plus a one-cell wall
# border, with the player spawning at the center.
MAZE_HEALTH_GATHERING = _rectangle_map(width=40, height=40, spawn_col=19, spawn_row=19)

# Convert DMLab seconds to a Doom-like 30 Hz step budget.
DMLAB_NAV_MAZE_HORIZON_FPS = 30
DMLAB_NAV_MAZE_01_EPISODE_LENGTH_SECONDS = 60
DMLAB_NAV_MAZE_02_EPISODE_LENGTH_SECONDS = 150
DMLAB_NAV_MAZE_03_EPISODE_LENGTH_SECONDS = 300
DMLAB_NAV_MAZE_01_MAX_STEPS = (
    DMLAB_NAV_MAZE_01_EPISODE_LENGTH_SECONDS * DMLAB_NAV_MAZE_HORIZON_FPS
)
DMLAB_NAV_MAZE_02_MAX_STEPS = (
    DMLAB_NAV_MAZE_02_EPISODE_LENGTH_SECONDS * DMLAB_NAV_MAZE_HORIZON_FPS
)
DMLAB_NAV_MAZE_03_MAX_STEPS = (
    DMLAB_NAV_MAZE_03_EPISODE_LENGTH_SECONDS * DMLAB_NAV_MAZE_HORIZON_FPS
)

DEFAULT_GOAL_REWARD = 1.0
DEFAULT_LIVING_REWARD = 0.0
DEFAULT_DISCOUNT_GAMMA = 0.99
MY_WAY_HOME_GOAL_REWARD = 1.0
MY_WAY_HOME_LIVING_REWARD = -0.0001
MY_WAY_HOME_DISCOUNT_GAMMA = 0.999
DMLAB_GOAL_REWARD = 10.0
DMLAB_NAV_MAZE_01_DISCOUNT_GAMMA = 0.999
DMLAB_NAV_MAZE_02_DISCOUNT_GAMMA = 0.9995
DMLAB_NAV_MAZE_03_DISCOUNT_GAMMA = 0.9999


def _dmlab_wall_palette(
    primary_wall: list[int],
    accents: list[list[int]],
    *,
    size: int = 96,
) -> list[list[int]]:
    palette = [[150, 150, 150] for _ in range(size)]
    palette[1] = primary_wall
    for color_id in range(7, size):
        palette[color_id] = accents[(color_id - 7) % len(accents)]
    return palette


DMLAB_RENDER_KWARGS_BY_INDEX = {
    "01": {
        "color_palette": _dmlab_wall_palette(
            [68, 112, 76],
            [
                [62, 108, 72],
                [82, 134, 86],
                [102, 150, 94],
                [118, 92, 42],
                [156, 48, 42],
                [96, 52, 38],
                [204, 184, 70],
            ],
        ),
        "floor_rgb": [182, 154, 46],
        "floor_checker_dark_rgb": [146, 124, 38],
        "floor_checker_light_rgb": [214, 184, 62],
    },
    "02": {
        "color_palette": _dmlab_wall_palette(
            [132, 116, 36],
            [
                [126, 110, 34],
                [154, 132, 42],
                [184, 154, 48],
                [88, 106, 92],
                [54, 116, 178],
                [42, 84, 132],
                [206, 62, 72],
            ],
        ),
        "floor_rgb": [68, 132, 188],
        "floor_checker_dark_rgb": [52, 102, 150],
        "floor_checker_light_rgb": [92, 164, 216],
    },
    "03": {
        "color_palette": _dmlab_wall_palette(
            [166, 88, 24],
            [
                [170, 86, 22],
                [202, 112, 30],
                [126, 72, 26],
                [40, 116, 82],
                [28, 92, 66],
                [126, 148, 152],
                [172, 188, 190],
                [138, 38, 34],
            ],
        ),
        "floor_rgb": [32, 122, 86],
        "floor_checker_dark_rgb": [24, 92, 66],
        "floor_checker_light_rgb": [50, 154, 110],
    },
}

# My Way Home footprint with 17 spawn map points represented as multiple S
# cells and sampled uniformly by RayMazeEnv.reset. Digits are static wall cells
# with stable palette colors.
MAZE_MY_WAY_HOME = """
99999999999999999999999999999999
91333333.....35.....599997.....9
91133333.....99.....999999.....9
91113333..S..S...S.....S....S..9
91111333.......................9
91111133.....35.....599997.....9
911111133.S93355.S955599777.S979
911111122..92244..944448888..989
9.....12.....24.....444888.....9
9.....99.....99.....444888.....9
9..S......S..S...S..444888..S..9
9...................444888.....9
9.....12.....24.....444888.....9
911111122..9224444444448889...89
911111129..9924444444448899...99
911111669.S9966444444449999...99
911116666..9666644444449999.G.99
91116666.....66664444499999...99
91166666.....6666644499999999999
91666666..S..6666664499999999999
96666666.....6666666999999999999
96666666.....6666666999999999999
966666666..966666999999999999999
966666669.S999999999999999999999
966666699..999999999999999999999
966669999.....999999999999999999
969999999...S.999999999999999999
99999999999999999999999999999999
"""

MAZE_MY_WAY_HOME_COLORLESS = """
################################
########.....##.....######.....#
########.....##.....######.....#
########..S..S...S.....S....S..#
########.......................#
########.....##.....######.....#
#########.S#####.S#########.S###
#########..#####..#########..###
#.....##.....##.....######.....#
#.....##.....##.....######.....#
#..S......S..S...S..######..S..#
#...................######.....#
#.....##.....##.....######.....#
#########..################...##
#########..################...##
#########.S################...##
#########..################.G.##
########.....##############...##
########.....###################
########..S..###################
########.....###################
########.....###################
#########..#####################
#########.S#####################
#########..#####################
#########.....##################
#########...S.##################
################################
"""

# DeepMind Lab nav_maze maps converted from .map files. Decal textures are
# represented as stable colored wall symbols instead of textured patches.
DMLAB_NAV_MAZE_STATIC_01 = """
#####+###############
#S.S;....G#S.S.S.S.S#
#.#.###.#.#.#4#####.#
#S#S.S!.8.xS#S.S.ShS#
#.###.#.#j#[###.#.#.#
#S.S#...#S.S.S4S#S_S#
#.#.#t###.#.#.#.#.#.#
#S/S#S.S#S.S.S.S#S.S#
#.#.#.#.#4#.#w#.#N###
#S#S.S#S.S.S.S.S.S.S#
#####################
"""

DMLAB_NAV_MAZE_STATIC_02 = """
#########<#####################
#SHS.S.S#S.S.S.S.S.S#S.S.S.S.S#
#.#.###.#.#####.###.#.#'#####.#
#S#S;S#S#S.S.S#S.SeS#S#S.S.S.S#
#.#.#.#.#####.###.#.#.#.#c#t###
zS#S.SMS.S.S.S.S#S#S.SQS.S#...#
#.#.#.#######{###.#g#####.#.#.#
#S.S#S.S.S.S.S.S#SoS.S.S.S..#.j
#.###.#.#.#.#.#.#.#.#.#.###)#.#
#S.S.S.S.S.S.S.S#SXS.S.S#S.S#G#
###.#.#.#.#.#.#.#.#.#.#.#.###.#
#S#S#S.S.S.S.S.SCS[S.S.S#Sx...2
#.#.#.#.#.#.#.#.#.###P###.#####
4S.S#S.S.S.S.S.S#S#S.S.SfS.S.S#
#.#w#L###,###.#E#.#.###.###+#.#
#S#S.S.S.S.S.S=S0S.S#S#S.S.S7S#
#.#######.#h#.#.###T#.###.###.#
#S.S.S.S.S#S.S.S.S.S.S.S#S.S.S#
#############################9#
"""

DMLAB_NAV_MAZE_STATIC_03 = """
#######A#l###########`###################
#S#S.S.S#S.S.S.S.S#S.S.S.S#S.S.S.S#S.S.S#
#.#.#$#.#.#######.#.#####.#.#'#9#.#.###.#
>S#S.S#S#S#S.S.S#S.S#S.S|S.S#S.S.S{S#S#S#
#.###.#.#.###.#.#######.#####.#:#.#.#.#.#
#SNS.S#S#S.S#SdS.S.S.S.S.S.S#S.S#S#S#SPS#
#.#.###.###.#####.#####.#.###[#.###.#.#.#
#S.S#S#S.S.S}S.S#S#S.S.S.S.ScS.S7S.S#S#S#
#.###.#.###U#.#.#.#.#.#.#.#.#.###.#1#.#.#
#SoS.S#S.S.S.S!S#S#S.S.S.S.S#S.S.S.S.S#S#
#.###.#####.#E#.#.#.#.#.#.#.#####w#####.#
#SaS.S+S.S.S.S#S.SWS.S.S.S.S#S.S.S#S.S#S#
#.#.#.#.#.#.#.###.#.#.#.#.#.###.###.#.#.#
#S#S,S.S.S.S.S#SZS.S.S.S.S.S#S.S)S.S#S.S?
#.#.#.#.#.#.#.#.#.#Q#########.###.#^###.#
#S#SmS.S.S.S.S2S.S.S.S.S.S.S.S3S.S#S.S.S#
#.#.#I#.#.#.#.#.#f###z###.#&###.#/#q#F#.#
#S#S.S]S.S.S.S#S.S.S.S.S.S.S.S#S.S.S.STS#
#.#.#.#.#.#.#.#4#.#.#.#.#.#.#.#.#####.#.#
#S#S#S#S.S.S.S#S.S#S.S.S.S.S.StS.S.S#S#S#
#.###.#.#.#.#.#.###.#.#.#.#.#.#.#####.#.#
#S.S.SXS.S.S.S#S#S#S.S.S.S.S.S.S#S.S#S#S#
#############.#.#.#.#.#.#.#.#.###.#.#.#.#
%S.S.S.S.S.S#S.S.S#S.S.S.S.S.S#S.S#S*S#S#
#.#####e###.#####(#C#########.#.#~#.#.#.#
#S.S.S.SsS.........G.........S.S=S.S.SnS#
#############O###;#################g#####
"""

DMLAB_NAV_MAZE_RANDOM_GOAL_01 = """
###>###############k#
#S.G.G.G.G.G.G.G.G.G#
###d#.#.#####_###`#.#
QG.G.GpG.GIGjG.G#G#G#
#.#####.#.#.#.#.#.#.#
#G6G.G#G.G4G.G.G#G#G#
#.#.#.#.#.#.#.#.#.#.#
#G#GsGVG.G4G.G.G#G#G#
#.#.#.#4#4#.#{#8#.#.#
#G.G#G.G#G.G.G.G.G.G#
#####################
"""

DMLAB_NAV_MAZE_RANDOM_GOAL_02 = """
#######q#####U#5###############
#S.G.G.G#G.G.G.G#G.G.G.G.G.G.G#
###.#@#.#8###.#.#.#######.###?#
#G.GFG.G.G.G.G#G#G#G.G.GMG.G.G#
#.###C#.#######.#.#&#.#/#s###.#
#G#G.G.G#G.G.G.G#G.G.G#G.G.GOG#
#.#.#.#.#.#########.#.#.#.#.#.#
#G.G.G.G#G#G.G.G.G.G.G#G.G.G*G_
#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
#G#G.G.G#G.G.G.G.G.GWGEG.G.G3G#
#.#h#.#|###.#.#.#.#.#.#.#.#.###
#G.G.G#G.G#G.G.G.G.GQGeG.G.G#G#
#L#.###.#.#.#.#.#.#.#.#####.#.#
#G.G#G.G(G#G.G.G.G.GkG.G#G.G.G#
#.###.###.#############.###%#.#
#G.G#G.G#G.G.G.G.G.G.GZG2G.G#G#
#.#.###.#########V#.###.#.#.#.#
#G#G.G.G.G.G.G.G.G#G.G.G.G#G.G#
###:#####################X###`#
"""

DMLAB_NAV_MAZE_RANDOM_GOAL_03 = """
###############0#####d#####,#############
cS.G.G#G.G.G#G.G.G#G#G.G.G.G.G.G#G.G.G#G#
#.###.#.#q#.#.###.#.#.#.#####`#.#.###.#.#
#G.G#G.G#G.G.G#G.G.G#G#G.G.G.G#G.G#G#G#G#
###.#.#####/#######5#.#.#.#.#.#####.#.#.#
#G.G*G.G.G.G.G.G.G.G#G#G.G.G.G#G.G#G#G#G#
#.###.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
#G.GkG8G.G.G.G.G.G.G#G#G.G.G.G#G#G.GWG#G#
#.#.#-#.#.#.#.#.#.#.#.#.#.#.#.#.#}###.#.#
#G#G.GUG.G.G.G.G.G.G#G#G.G.G.G#G.G.G#G.G#
#####.#.#.#.#.#.#.#.#.#.#.#.#.#.###.###.#
#G.G:GeG.G.G.G.G.G.G#G#G.G.G.G#GZG#G.G.G#
#.#.#.#?#7#####n#####.#.#######.#.#N#####
#G#G.G.G.G.G#G.G.G.G.G#G.G.G.G#G.G.G.G.G#
#.###J###O#%###########.###############.#
#G.G.G#G.G.G.G.G.GvG.G.GmG.G.G.G.G.G.G&G#
#.###.#.#.#.#.#.#.#.#E#.#.#.#.#.#.#.#.#.#
IG.G#G#G.G.G.G.G.G#G.G9G#G.G.G.G.G.G.G#G#
#.#<#.#.#.#.#.#.#.###.#.#.#.#.#.#.#.#.#.#
#GCG.GiG.G.G.G.G.G#G.G#G#G.G.G.G.G.G.G#G#
###.#.#.#.#.#.#.#.#.###.#.#.#.#.#.#.#.#.#
#G.G#G.G.G.G.G.G.G#GAG>GHG.G.G.G.G.G.GlG#
#.#x#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
#G[G.G6G.G.G.G.G.G|G#G.G{G.G.G.G.G.G.GaG#
#.#######@#########.#####)#####.#######.#
#G.G.G.G.G.G.G.G.G.G.G.G.G.G.G.G.G.G.G.G#
###############^###P#################z#T#
"""

DMLAB_NAV_MAZE_01 = DMLAB_NAV_MAZE_STATIC_01


MAPS_BY_NAME = {
    "simple": MAZE_SIMPLE,
    "key-door": MAZE_KEY_DOOR,
    "key-corridor": MAZE_KEY_CORRIDOR,
    "my-way-home": MAZE_MY_WAY_HOME,
    "my-way-home-colorless": MAZE_MY_WAY_HOME_COLORLESS,
    "dmlab-static-01": DMLAB_NAV_MAZE_STATIC_01,
    "dmlab-static-02": DMLAB_NAV_MAZE_STATIC_02,
    "dmlab-static-03": DMLAB_NAV_MAZE_STATIC_03,
    "dmlab-random-01": DMLAB_NAV_MAZE_RANDOM_GOAL_01,
    "dmlab-random-02": DMLAB_NAV_MAZE_RANDOM_GOAL_02,
    "dmlab-random-03": DMLAB_NAV_MAZE_RANDOM_GOAL_03,
}

HEALTH_GATHERING_MAPS_BY_NAME = {
    "health-gathering": MAZE_HEALTH_GATHERING,
}

HEALTH_GATHERING_REWARD_KWARGS_BY_NAME = {
    "health-gathering": {
        "living_reward": HEALTH_GATHERING_LIVING_REWARD,
        "death_penalty": HEALTH_GATHERING_DEATH_PENALTY,
    },
}

MAP_REWARD_KWARGS_BY_NAME = {
    "simple": {
        "goal_reward": DEFAULT_GOAL_REWARD,
        "living_reward": DEFAULT_LIVING_REWARD,
    },
    "key-door": {
        "goal_reward": DEFAULT_GOAL_REWARD,
        "living_reward": DEFAULT_LIVING_REWARD,
    },
    "key-corridor": {
        "goal_reward": DEFAULT_GOAL_REWARD,
        "living_reward": DEFAULT_LIVING_REWARD,
    },
    "my-way-home": {
        "goal_reward": MY_WAY_HOME_GOAL_REWARD,
        "living_reward": MY_WAY_HOME_LIVING_REWARD,
    },
    "my-way-home-colorless": {
        "goal_reward": MY_WAY_HOME_GOAL_REWARD,
        "living_reward": MY_WAY_HOME_LIVING_REWARD,
    },
    "dmlab-static-01": {
        "goal_reward": DMLAB_GOAL_REWARD,
        "living_reward": DEFAULT_LIVING_REWARD,
    },
    "dmlab-static-02": {
        "goal_reward": DMLAB_GOAL_REWARD,
        "living_reward": DEFAULT_LIVING_REWARD,
    },
    "dmlab-static-03": {
        "goal_reward": DMLAB_GOAL_REWARD,
        "living_reward": DEFAULT_LIVING_REWARD,
    },
    "dmlab-random-01": {
        "goal_reward": DMLAB_GOAL_REWARD,
        "living_reward": DEFAULT_LIVING_REWARD,
    },
    "dmlab-random-02": {
        "goal_reward": DMLAB_GOAL_REWARD,
        "living_reward": DEFAULT_LIVING_REWARD,
    },
    "dmlab-random-03": {
        "goal_reward": DMLAB_GOAL_REWARD,
        "living_reward": DEFAULT_LIVING_REWARD,
    },
}

MAP_DISCOUNT_GAMMAS_BY_NAME = {
    "simple": DEFAULT_DISCOUNT_GAMMA,
    "key-door": DEFAULT_DISCOUNT_GAMMA,
    "key-corridor": DEFAULT_DISCOUNT_GAMMA,
    "my-way-home": MY_WAY_HOME_DISCOUNT_GAMMA,
    "my-way-home-colorless": MY_WAY_HOME_DISCOUNT_GAMMA,
    "dmlab-static-01": DMLAB_NAV_MAZE_01_DISCOUNT_GAMMA,
    "dmlab-random-01": DMLAB_NAV_MAZE_01_DISCOUNT_GAMMA,
    "dmlab-static-02": DMLAB_NAV_MAZE_02_DISCOUNT_GAMMA,
    "dmlab-random-02": DMLAB_NAV_MAZE_02_DISCOUNT_GAMMA,
    "dmlab-static-03": DMLAB_NAV_MAZE_03_DISCOUNT_GAMMA,
    "dmlab-random-03": DMLAB_NAV_MAZE_03_DISCOUNT_GAMMA,
}

MAP_EPISODE_HORIZONS_BY_NAME = {
    "key-corridor": MINIGRID_KEY_CORRIDOR_S4R3_MAX_STEPS,
    "my-way-home": MY_WAY_HOME_EPISODE_TIMEOUT,
    "my-way-home-colorless": MY_WAY_HOME_EPISODE_TIMEOUT,
    "dmlab-static-01": DMLAB_NAV_MAZE_01_MAX_STEPS,
    "dmlab-static-02": DMLAB_NAV_MAZE_02_MAX_STEPS,
    "dmlab-static-03": DMLAB_NAV_MAZE_03_MAX_STEPS,
    "dmlab-random-01": DMLAB_NAV_MAZE_01_MAX_STEPS,
    "dmlab-random-02": DMLAB_NAV_MAZE_02_MAX_STEPS,
    "dmlab-random-03": DMLAB_NAV_MAZE_03_MAX_STEPS,
}

HEALTH_GATHERING_EPISODE_HORIZONS_BY_NAME = {
    "health-gathering": HEALTH_GATHERING_EPISODE_TIMEOUT,
}

MAP_RENDER_KWARGS_BY_NAME = {
    "dmlab-static-01": DMLAB_RENDER_KWARGS_BY_INDEX["01"],
    "dmlab-random-01": DMLAB_RENDER_KWARGS_BY_INDEX["01"],
    "dmlab-static-02": DMLAB_RENDER_KWARGS_BY_INDEX["02"],
    "dmlab-random-02": DMLAB_RENDER_KWARGS_BY_INDEX["02"],
    "dmlab-static-03": DMLAB_RENDER_KWARGS_BY_INDEX["03"],
    "dmlab-random-03": DMLAB_RENDER_KWARGS_BY_INDEX["03"],
}

HEALTH_GATHERING_RENDER_KWARGS_BY_NAME = {
    "health-gathering": {
        "color_palette": [
            [150, 150, 150],
            HEALTH_GATHERING_WALL_RGB,
            [74, 166, 104],
            [78, 120, 206],
            [236, 44, 36],
            [48, 126, 255],
            [255, 212, 38],
        ],
        "floor_rgb": HEALTH_GATHERING_FLOOR_RGB,
        "floor_checker_dark_rgb": HEALTH_GATHERING_FLOOR_CHECKER_DARK_RGB,
        "floor_checker_light_rgb": HEALTH_GATHERING_FLOOR_CHECKER_LIGHT_RGB,
    },
}

DEFAULT_MAZES = [MAZE_SIMPLE]
