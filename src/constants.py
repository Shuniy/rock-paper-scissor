from enum import Enum
import torch
import numpy as np

SEED = 42
RNG = np.random.default_rng(SEED)

class ACTION(str, Enum):
    ROCK = 'Rock'
    PAPER = 'Paper'
    SCISSORS = 'Scissors'

ACTIONS = [action.value for action in ACTION]

PAYOFF_TENSOR = torch.tensor(
    [[0.0, -1.0, 1.0],  # Rock
     [1.0, 0.0, -1.0],  # Paper
     [-1.0, 1.0, 0.0]]  # Scissors
)