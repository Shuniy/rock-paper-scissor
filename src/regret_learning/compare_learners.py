from typing import Literal

from src.game import Game
from src.player import Player
from src.constants import ACTIONS, ACTION
from src.rock_paper_scissors import RockPaperScissors
import numpy as np

class CompareLearners:
    def __init__(self, player_a_name = "alice", player_b_name = "bob"):
        self.player_a = Player(player_a_name)
        self.player_b = Player(player_b_name)
        self.game = Game(self.player_a, self.player_b)
        self.learning_rate = 0.1  # Example learning rate for Hedge algorithm
        self.payoff_matrix_a = RockPaperScissors().payoff_matrix
        self.payoff_matrix_b = -self.payoff_matrix_a.T
        self.cumulative_rewards_a = np.zeros(len(ACTIONS))
        self.cumulative_rewards_b = np.zeros(len(ACTIONS))

        self.game = Game(self.player_a, self.player_b)
        
        self.swap_weights_a = np.zeros((len(ACTIONS), len(ACTIONS)))  # Swap weights for Player A
        self.swap_weights_b = np.zeros((len(ACTIONS), len(ACTIONS)))  # Swap weights for Player B

        self.hedge_weights_a = np.ones(len(ACTIONS))  # Hedge weights for Player A
        self.hedge_weights_b = np.ones(len(ACTIONS))  # Hedge weights for Player B

    @staticmethod
    def results(rounds=100, verbose=False):
        # Separate, fresh players and game for Hedge
        hedge_experiment = CompareLearners()

        # Separate, fresh players and game for SwapMW
        swap_experiment = CompareLearners()

        return {
            "hedge": hedge_experiment.play(
                algorithm="hedge",
                rounds=rounds,
                verbose=verbose
            ),
            "swap": swap_experiment.play(
                algorithm="swap",
                rounds=rounds,
                verbose=verbose
            )
        }

    def play(self, algorithm: Literal["hedge", "swap"], rounds=100, verbose=False):
        if algorithm == "hedge":
            for step in range(rounds):
                probabilities_a = self.hedge_weights_a / sum(self.hedge_weights_a)
                probabilities_b = self.hedge_weights_b / sum(self.hedge_weights_b)

                self.player_a.action = self.player_a.choose_action(probabilities=probabilities_a)
                self.player_b.action = self.player_b.choose_action(probabilities=probabilities_b)

                round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
                all_rewards_a = self.payoff_matrix_a[:, ACTIONS.index(action_b)]
                all_rewards_b = self.payoff_matrix_b[:, ACTIONS.index(action_a)]

                self.cumulative_rewards_a += all_rewards_a
                self.cumulative_rewards_b += all_rewards_b

                # Update weights based on rewards
                self.hedge_weights_a *= np.exp(self.learning_rate * all_rewards_a)
                self.hedge_weights_b *= np.exp(self.learning_rate * all_rewards_b)

                if verbose:
                    if step + 1 in [rounds // 10 * i for i in range(1, 11)]:   # Print every 10% of the rounds
                        print(f"Round {round_number}: {self.player_a.name} chose {action_a}, {self.player_b.name} chose {action_b}. Rewards: {self.player_a.name}: {reward_a}, {self.player_b.name}: {reward_b}")
            best_action_index_a = np.argmax(self.cumulative_rewards_a)
            best_action_index_b = np.argmax(self.cumulative_rewards_b)
            best_action_a = ACTIONS[best_action_index_a]
            best_action_b = ACTIONS[best_action_index_b]
            best_cumulative_reward_a = self.cumulative_rewards_a[best_action_index_a]
            best_cumulative_reward_b = self.cumulative_rewards_b[best_action_index_b]
            print(f"Best action for {self.player_a.name}: {best_action_a} with cumulative reward: {best_cumulative_reward_a}")
            print(f"Best action for {self.player_b.name}: {best_action_b} with cumulative reward: {best_cumulative_reward_b}")
            regret_a = best_cumulative_reward_a - self.player_a.score
            regret_b = best_cumulative_reward_b - self.player_b.score
            print(f"Regret for {self.player_a.name}: {regret_a}")
            print(f"Regret for {self.player_b.name}: {regret_b}")
            average_regret_a = regret_a / rounds
            average_regret_b = regret_b / rounds
            print(f"Average regret for {self.player_a.name}: {average_regret_a}")
            print(f"Average regret for {self.player_b.name}: {average_regret_b}")
            print(f"Total reward for {self.player_a.name} after {rounds} rounds: {self.player_a.score}")
            print(f"Total reward for {self.player_b.name} after {rounds} rounds: {self.player_b.score}")
            return {
                "player_a": {
                    "score": self.player_a.score,
                    "best_action": best_action_a,
                    "best_cumulative_reward": best_cumulative_reward_a,
                    "regret": regret_a,
                    "average_regret": average_regret_a,
                    "total_reward": self.player_a.score
                },
                "player_b": {
                    "score": self.player_b.score,
                    "best_action": best_action_b,
                    "best_cumulative_reward": best_cumulative_reward_b,
                    "regret": regret_b,
                    "average_regret": average_regret_b,
                    "total_reward": self.player_b.score
                },
                "weight_a": self.hedge_weights_a,
                "weight_b": self.hedge_weights_b
            }
        elif algorithm == "swap":
            history_a = []
            history_b = []

            for step in range(rounds):
                transition_matrix_a = self.get_transition_matrix(self.swap_weights_a)
                transition_matrix_b = self.get_transition_matrix(self.swap_weights_b)

                probabilities_a = self.get_stationary_distribution(transition_matrix_a, rounds=rounds)
                probabilities_b = self.get_stationary_distribution(transition_matrix_b, rounds=rounds)

                self.player_a.action = self.player_a.choose_action(probabilities=probabilities_a)
                self.player_b.action = self.player_b.choose_action(probabilities=probabilities_b)

                round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()

                history_a.append(action_a)
                history_b.append(action_b)

                rewards_a = self.payoff_matrix_a[:, ACTIONS.index(action_b)]
                rewards_b = self.payoff_matrix_b[:, ACTIONS.index(action_a)]

                self.swap_weights_a += self.learning_rate * probabilities_a[:, np.newaxis] * rewards_a[np.newaxis, :]
                self.swap_weights_b += self.learning_rate * probabilities_b[:, np.newaxis] * rewards_b[np.newaxis, :]
              
                if verbose:
                    if step + 1 in [rounds // 10 * i for i in range(1, 11)]:
                        print(f"After {step + 1} rounds:")
                        print("Stationary distribution for Player A:", np.round(probabilities_a, 3))
                        print("Transition matrix for Player A:", np.round(transition_matrix_a, 3))
                        print("Stationary distribution for Player B:", np.round(probabilities_b, 3))
                        print("Transition matrix for Player B:", np.round(transition_matrix_b, 3))
            transition_matrix_a = self.get_transition_matrix(self.swap_weights_a)
            transition_matrix_b = self.get_transition_matrix(self.swap_weights_b)
            probabilities_a = self.get_stationary_distribution(self.get_transition_matrix(self.swap_weights_a), rounds=rounds)
            probabilities_b = self.get_stationary_distribution(self.get_transition_matrix(self.swap_weights_b), rounds=rounds)
            print(f"Final stationary distribution for {self.player_a.name}: {np.round(probabilities_a, 3)}")
            print(f"Final stationary distribution for {self.player_b.name}: {np.round(probabilities_b, 3)}")
            print(f"Final transition matrix for {self.player_a.name}: {np.round(transition_matrix_a, 3)}")
            print(f"Final transition matrix for {self.player_b.name}: {np.round(transition_matrix_b, 3)}")
            history_a = np.array([ACTIONS.index(action) for action in history_a])
            history_b = np.array([ACTIONS.index(action) for action in history_b])
            fixed_rewards_a = np.zeros(len(ACTIONS))
            fixed_rewards_b = np.zeros(len(ACTIONS))
            for action in range(len(ACTIONS)):
                fixed_rewards_a[action] = np.sum(self.payoff_matrix_a[action, history_b])
                fixed_rewards_b[action] = np.sum(self.payoff_matrix_b[action, history_a])
            best_action_index_a = np.argmax(fixed_rewards_a)
            best_action_index_b = np.argmax(fixed_rewards_b)
            best_action_a = ACTIONS[best_action_index_a]
            best_action_b = ACTIONS[best_action_index_b]
            external_regret_a = fixed_rewards_a[best_action_index_a] - self.player_a.score
            external_regret_b = fixed_rewards_b[best_action_index_b] - self.player_b.score
            average_external_regret_a = external_regret_a / rounds
            average_external_regret_b = external_regret_b / rounds
            print(f"Best fixed action for {self.player_a.name}: {best_action_a} with cumulative reward: {fixed_rewards_a[best_action_index_a]}")
            print(f"Best fixed action for {self.player_b.name}: {best_action_b} with cumulative reward: {fixed_rewards_b[best_action_index_b]}")
            print(f"External regret for {self.player_a.name}: {external_regret_a}")
            print(f"External regret for {self.player_b.name}: {external_regret_b}")
            print(f"Average external regret for {self.player_a.name}: {average_external_regret_a}")
            print(f"Average external regret for {self.player_b.name}: {average_external_regret_b}")
            swap_regret_a = 0
            swap_regret_b = 0
            for i, action in enumerate(ACTIONS):
                replacement_reward_a = np.zeros(len(ACTIONS))
                replacement_reward_b = np.zeros(len(ACTIONS))
                original_reward_a = 0
                original_reward_b = 0
                for a, b in zip(history_a, history_b):
                    if a == i:
                        original_reward_a += self.payoff_matrix_a[a, b]
                        for replacement_action in range(len(ACTIONS)):
                            replacement_reward_a[replacement_action] += self.payoff_matrix_a[replacement_action, b]
                    if b == i:
                        original_reward_b += self.payoff_matrix_b[b, a]
                        for replacement_action in range(len(ACTIONS)):
                            replacement_reward_b[replacement_action] += self.payoff_matrix_b[replacement_action, a]
                best_replacement_reward_a = np.max(replacement_reward_a)
                best_replacement_reward_b = np.max(replacement_reward_b)
                improvement_a = best_replacement_reward_a - original_reward_a
                improvement_b = best_replacement_reward_b - original_reward_b
                swap_regret_a += improvement_a
                swap_regret_b += improvement_b
            return {
                "player_a": {
                    "score": self.player_a.score,
                    "best_fixed_action": best_action_a,
                    "best_fixed_cumulative_reward": fixed_rewards_a[best_action_index_a],
                    "external_regret": external_regret_a,
                    "average_external_regret": average_external_regret_a,
                    "swap_regret": swap_regret_a
                },
                "player_b": {
                    "score": self.player_b.score,
                    "best_fixed_action": best_action_b,
                    "best_fixed_cumulative_reward": fixed_rewards_b[best_action_index_b],
                    "external_regret": external_regret_b,
                    "average_external_regret": average_external_regret_b,
                    "swap_regret": swap_regret_b
                },
                "weight_a": self.swap_weights_a,
                "weight_b": self.swap_weights_b,
                "probabilities_a": probabilities_a,
                "probabilities_b": probabilities_b,
                "transition_matrix_a": transition_matrix_a,
                "transition_matrix_b": transition_matrix_b
            }

    def get_transition_matrix(self, log_weights):
        shifted = log_weights - np.max(log_weights, axis = 1, keepdims=True)
        exp_weights = np.exp(shifted)
        transition_matrix = exp_weights / np.sum(exp_weights, axis=1, keepdims=True)
        return transition_matrix

    def get_stationary_distribution(self, P, rounds):
        pi = np.ones(P.shape[0]) / P.shape[0]  # Start with uniform distribution
        for _ in range(rounds):
            pi = pi @ P  # Update the distribution by multiplying with the transition matrix
        return pi
    