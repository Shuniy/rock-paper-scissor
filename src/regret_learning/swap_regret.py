from src.game import Game
from src.player import Player
from src.constants import ACTIONS, ACTION
import numpy as np

class SwapRegretGame:
    def __init__(self, player_a_name="Agent A", player_b_name="Agent B"):
        self.player_a = Player(player_a_name)
        self.player_b = Player(player_b_name)
        self.game = Game(self.player_a, self.player_b)
        self.learning_rate = 0.1
        # Four rounds of observed actions
        self.history_a = np.array([ACTION.ROCK, ACTION.ROCK, ACTION.PAPER, ACTION.PAPER])
        self.history_b = np.array([ACTION.SCISSORS, ACTION.PAPER, ACTION.SCISSORS, ACTION.SCISSORS])

        self.fixed_rewards_a = np.zeros(len(ACTIONS))
        self.fixed_rewards_b = np.zeros(len(ACTIONS))

    def play(self, rounds=100, verbose=False):
        for a, b in zip(self.history_a, self.history_b):
            self.player_a.action = a
            self.player_b.action = b
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()

        print(f"Total reward for {self.player_a.name} after {rounds} rounds: {self.player_a.score}")
        print(f"Total reward for {self.player_b.name} after {rounds} rounds: {self.player_b.score}")

        for action in ACTIONS:
            for a, b in zip(self.history_a, self.history_b):
                self.fixed_rewards_a[ACTIONS.index(action)] += self.game.rock_paper_scissors.play_round(action, b)[0]
                self.fixed_rewards_b[ACTIONS.index(action)] += self.game.rock_paper_scissors.play_round(a, action)[1]

        best_fixed_reward_a = np.max(self.fixed_rewards_a)
        best_fixed_reward_b = np.max(self.fixed_rewards_b)
        external_regret_a = best_fixed_reward_a - self.player_a.score
        external_regret_b = best_fixed_reward_b - self.player_b.score
        print(f"External regret for {self.player_a.name}: {external_regret_a}")
        print(f"External regret for {self.player_b.name}: {external_regret_b}")