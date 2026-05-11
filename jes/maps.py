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

VIZDOOM_MY_WAY_HOME_EPISODE_TIMEOUT = 2100

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

# ViZDoom's My Way Home footprint, rasterized from the bundled UDMF WAD at
# 32 Doom units per cell. The 17 original spawn map points are represented as
# multiple S cells and are sampled uniformly by RayMazeEnv.reset. Digits are
# static wall cells colored from nearby UDMF linedef textures.
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
    "dmlab-random-goal-01": DMLAB_NAV_MAZE_RANDOM_GOAL_01,
    "dmlab-random-goal-02": DMLAB_NAV_MAZE_RANDOM_GOAL_02,
    "dmlab-random-goal-03": DMLAB_NAV_MAZE_RANDOM_GOAL_03,
}

MAP_EPISODE_HORIZONS_BY_NAME = {
    "key-corridor": MINIGRID_KEY_CORRIDOR_S4R3_MAX_STEPS,
    "my-way-home": VIZDOOM_MY_WAY_HOME_EPISODE_TIMEOUT,
    "my-way-home-colorless": VIZDOOM_MY_WAY_HOME_EPISODE_TIMEOUT,
    "dmlab-static-01": DMLAB_NAV_MAZE_01_MAX_STEPS,
    "dmlab-static-02": DMLAB_NAV_MAZE_02_MAX_STEPS,
    "dmlab-static-03": DMLAB_NAV_MAZE_03_MAX_STEPS,
    "dmlab-random-goal-01": DMLAB_NAV_MAZE_01_MAX_STEPS,
    "dmlab-random-goal-02": DMLAB_NAV_MAZE_02_MAX_STEPS,
    "dmlab-random-goal-03": DMLAB_NAV_MAZE_03_MAX_STEPS,
}

MAP_RENDER_KWARGS_BY_NAME = {
}

DEFAULT_MAZES = [MAZE_SIMPLE]
