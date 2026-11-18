import numpy as np
from src.constants import ACTIONS

class RockPaperScissors:
    def __init__(self):
        self.actions = ACTIONS
        self.payoff_matrix = np.array([
            [0, -1, 1],   # Rock
            [1, 0, -1],   # Paper
            [-1, 1, 0]    # Scissors
        ])

    def play_round(self, action_a, action_b):
        reward_a = self.payoff_matrix[self.actions.index(action_a), self.actions.index(action_b)]
        reward_b = self.payoff_matrix[self.actions.index(action_b), self.actions.index(action_a)]
        return reward_a, reward_b
