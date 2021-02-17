# FROM THE OP PAPER-ISH
MINI_BATCH_SIZE = 32
MEMORY_SIZE = 10**6
BUFFER_SIZE = 100
LHIST = 4
GAMMA = 0.99
UPDATE_FREQ_ONlINE = 4
UPDATE_TARGET = 2500 # This was 10**4 but is measured in actor steps, so it's divided update_freq_online
TEST_FREQ = 5*10**4 # Measure in updates
TEST_STEPS = 10**4
LEARNING_RATE = 0.00025
G_MOMENTUM = 0.95
EPSILON_INIT = 1.0
EPSILON_FINAL = 0.1
EPSILON_TEST = 0.05
EPSILON_LIFE = 10**6
REPLAY_START = 5*10**4
NO_OP_MAX = 30
UPDATES = 5*10**6
CLIP_REWARD = 1.0
CLIP_ERROR = 1.0
# MISC
PLAY_STEPS = 3000
BUFFER_SAMPLES = 20
CROP = (0, -1)
FRAMESIZE = [84,84]
FRAMESIZETP = (84,84)
#DROPS = [0.0,0.15,0.1,0.0]
DROPS = [0.0, 0.0, 0.0, 0.0]

Games = ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider',  'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber',  'demon_attack', 'double_dunk',
    'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'pong',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 
    'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor',  'zaxxon']

GamesExtras = ['defender','phoenix','berzerk','skiing','yars_revenge','solaris','pitfall',]

ACTION_MEANING = {
    0: "NOOP",
    1: "FIRE",
    2: "UP",
    3: "RIGHT",
    4: "LEFT",
    5: "DOWN",
    6: "UPRIGHT",
    7: "UPLEFT",
    8: "DOWNRIGHT",
    9: "DOWNLEFT",
    10: "UPFIRE",
    11: "RIGHTFIRE",
    12: "LEFTFIRE",
    13: "DOWNFIRE",
    14: "UPRIGHTFIRE",
    15: "UPLEFTFIRE",
    16: "DOWNRIGHTFIRE",
    17: "DOWNLEFTFIRE",
}