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
        self.history_a = [ACTION.ROCK, ACTION.ROCK, ACTION.PAPER, ACTION.PAPER]
        self.history_b = [ACTION.SCISSORS, ACTION.PAPER, ACTION.SCISSORS, ACTION.SCISSORS]

        self.fixed_rewards_a = np.zeros(len(ACTIONS))
        self.fixed_rewards_b = np.zeros(len(ACTIONS))

    def play_history(self, rounds=100, verbose=False):
        """Replay the four recorded rounds. Keep rounds for existing callers."""
        for a, b in zip(self.history_a, self.history_b):
            self.player_a.action = a
            self.player_b.action = b
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            if verbose:
                print(f"Round {round_number}: {self.player_a.name} chose {a}, {self.player_b.name} chose {b}.")

        print(f"Total reward for {self.player_a.name} after {self.game.rounds_played} rounds: {self.player_a.score}")
        print(f"Total reward for {self.player_b.name} after {self.game.rounds_played} rounds: {self.player_b.score}")

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

        swap_regret_a = 0
        swap_regret_b = 0
        for i, action in enumerate(ACTIONS):
            replacement_rewards_a = np.zeros(len(ACTIONS))
            replacement_rewards_b = np.zeros(len(ACTIONS))
            original_rewards_a = 0
            original_rewards_b = 0
            for a, b in zip(self.history_a, self.history_b):
                if a == action:
                    original_rewards_a += self.game.rock_paper_scissors.play_round(a, b)[0]
                    for replacement_action in ACTIONS:
                        replacement_rewards_a[ACTIONS.index(replacement_action)] += self.game.rock_paper_scissors.play_round(replacement_action, b)[0]
                if b == action:
                    original_rewards_b += self.game.rock_paper_scissors.play_round(a, b)[1]
                    for replacement_action in ACTIONS:
                        replacement_rewards_b[ACTIONS.index(replacement_action)] += self.game.rock_paper_scissors.play_round(a, replacement_action)[1]
            best_replacement_reward_a = np.max(replacement_rewards_a)
            best_replacement_reward_b = np.max(replacement_rewards_b)
            improvement_a = best_replacement_reward_a - original_rewards_a
            improvement_b = best_replacement_reward_b - original_rewards_b
            swap_regret_a += improvement_a
            swap_regret_b += improvement_b
            print(f"Swap regret for {self.player_a.name} when replacing {action}: {improvement_a}")
            print(f"Swap regret for {self.player_b.name} when replacing {action}: {improvement_b}")
        print(f"Total swap regret for {self.player_a.name}: {swap_regret_a}")
        print(f"Total swap regret for {self.player_b.name}: {swap_regret_b}")
