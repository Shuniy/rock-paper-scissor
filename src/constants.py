from enum import Enum

import numpy as np

SEED = 42
RNG = np.random.default_rng(SEED)

class ACTION(str, Enum):
    ROCK = 'Rock'
    PAPER = 'Paper'
    SCISSORS = 'Scissors'

ACTIONS = [action.value for action in ACTION]
