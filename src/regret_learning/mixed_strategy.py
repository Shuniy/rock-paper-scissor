from src.game import Game
from src.player import Player
from src.constants import ACTIONS
from src.rock_paper_scissors import RockPaperScissors
import numpy as np

class MixedStrategyGame:
    def __init__(self, player_a_name="Player A", player_b_name="Player B"):
        self.player_a = Player(player_a_name)
        self.player_b = Player(player_b_name)
        self.game = Game(self.player_a, self.player_b)

    def reset(self):
        self.player_a.score = 0
        self.player_b.score = 0

    def play_b_uniformly(self, rounds=100, verbose=False):
        for _ in range(rounds):
            strategy_a = [0.5, 0.3, 0.2]  # Example mixed strategy for Player A
            strategy_b = [1/3, 1/3, 1/3]  # Example mixed strategy for Player B
            self.player_a.action = self.player_a.choose_action(probabilities=strategy_a)
            self.player_b.action = self.player_b.choose_action(probabilities=strategy_b)
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            if verbose:
                print(f"Round {round_number}: {self.player_a.name} chose {action_a}, {self.player_b.name} chose {action_b}. Rewards: {self.player_a.name}: {reward_a}, {self.player_b.name}: {reward_b}")
        print(f"Total reward for {self.player_a.name} after {rounds} rounds: {self.player_a.score}")
        print(f"Total reward for {self.player_b.name} after {rounds} rounds: {self.player_b.score}")

    def play(self, rounds=100, verbose=False):
        for _ in range(rounds):
            strategy_a = [0.8, 0.1, 0.1]  # Example mixed strategy for Player A
            expected_rewards = np.matmul(strategy_a, RockPaperScissors().payoff_matrix)  # Example mixed strategy for Player B based on Player A's strategy
            best_response_index = np.argmin(expected_rewards)  # Player B chooses the action that minimizes Player A's expected reward
            self.player_a.action = self.player_a.choose_action(probabilities=strategy_a)
            self.player_b.action = self.player_b.choose_action(action=ACTIONS[best_response_index])
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            if verbose:
                print("Best response for Player B based on Player A's strategy:", ACTIONS[best_response_index])
                print(f"Round {round_number}: {self.player_a.name} chose {action_a}, {self.player_b.name} chose {action_b}. Rewards: {self.player_a.name}: {reward_a}, {self.player_b.name}: {reward_b}")
        print(f"Total reward for {self.player_a.name} after {rounds} rounds: {self.player_a.score}")
        print(f"Total reward for {self.player_b.name} after {rounds} rounds: {self.player_b.score}")