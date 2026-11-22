from src.game import Game
from src.player import Player
from src.constants import ACTIONS, ACTION
import numpy as np

class SwapLearnerGame:
    def __init__(self, player_a_name = "Alice", player_b_name = "Bob"):
        self.player_a = Player(player_a_name)
        self.player_b = Player(player_b_name)
        self.game = Game(self.player_a, self.player_b)
        self.weights_a = np.zeros((len(ACTIONS), len(ACTIONS)))  # Weights for Player A's actions
        self.weights_b = np.zeros((len(ACTIONS), len(ACTIONS)))  # Weights for Player B's actions
        self.learning_rate = 0.1  # Learning rate
        self.payoff_matrix = self.game.rock_paper_scissors.payoff_matrix  # Payoff matrix for the game
        self.history_a = []  # To store the history of Player A's actions
        self.history_b = []  # To store the history of Player B's actions

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


    def experiment(self, rounds=100, verbose=False):
        # 1. Calculate current policy
        P = self.get_transition_matrix(self.weights_a)
        pi = self.get_stationary_distribution(P, rounds)
        print("Stationary distribution for Player A:", pi)
        # 2. B chooses Paper
        action_b = ACTION.PAPER  # Player B plays fixed action Paper
        # 3. Rewards for all possible A actions
        rewards_a = self.payoff_matrix[:, ACTIONS.index(action_b)]
        print("Rewards for Player A's actions against Paper:", rewards_a)
        # 4. Update every row's log-weights
        for i in range(len(ACTIONS)):
            for j in range(len(ACTIONS)):
                self.weights_a[i][j] += self.learning_rate * pi[i] * rewards_a[j]
        # 5. Recalculate the transition matrix
        P = self.get_transition_matrix(self.weights_a)
        # 6. Recalculate Agent A's strategy
        pi = self.get_stationary_distribution(P, rounds)
        print("Updated transition matrix for Player A after learning:", np.round(P, 3))
        print("Updated probabilities for Player A's actions after learning:", np.round(pi, 3))

    def experiment_5_round(self):
        for round in range(5):
            print(f"\nRound {round + 1}:")
            self.experiment(rounds=100, verbose=True)
        P = self.get_transition_matrix(self.weights_a)
        pi = self.get_stationary_distribution(P, 100)
        print("\nFinal probabilities for Player A's actions after 5 rounds of learning:", np.round(pi, 3))
        print("Final transition matrix for Player A after 5 rounds of learning:", np.round(P, 3))

    def play_fixed_b(self, action=ACTION.PAPER, rounds=100, verbose=False):
        for step in range(rounds):
            # step 1: Calculate current policy
            P_a = self.get_transition_matrix(self.weights_a)
            P_b = self.get_transition_matrix(self.weights_b)
            # step 2: Calculate stationary distributions for both players
            pi_a = self.get_stationary_distribution(P_a, rounds)
            pi_b = self.get_stationary_distribution(P_b, rounds)
            # step 3: Sample actions for both players based on their stationary distributions
            action_a = np.random.choice(ACTIONS, p=pi_a)
            # step 4: Player B plays fixed action
            action_b = action  # Player B plays fixed action
            # step 5: Play the round and get rewards
            self.player_a.action = action_a
            self.player_b.action = action_b
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            # step 6: store actions in history
            self.history_a.append(action_a)
            self.history_b.append(action_b)
            # step 7: reward for all possible actions of Player A against Player B's action
            rewards_a = self.payoff_matrix[:, ACTIONS.index(action_b)]
            # step 8: Update every row's log-weights for Player A
            self.weights_a += self.learning_rate * pi_a[:, np.newaxis] * rewards_a[np.newaxis, :]
            if step + 1 in [rounds // 10 * i for i in range(1, 11)]:
                print(f"After {step + 1} rounds:")
                print("Stationary distribution for Player A:", np.round(pi_a, 3))
                print("Transition matrix for Player A:", np.round(P_a, 3))
        # step 9: Final stationary distribution and transition matrix for Player A after all rounds
        P_a = self.get_transition_matrix(self.weights_a)
        pi_a = self.get_stationary_distribution(P_a, rounds)
        print("\nFinal stationary distribution for Player A after all rounds:", np.round(pi_a, 3))
        print("Final transition matrix for Player A after all rounds:", np.round(P_a, 3))
        # step 10: calculate external regret for Player A
        history_a = np.array([ACTIONS.index(action) for action in self.history_a])
        history_b = np.array([ACTIONS.index(action) for action in self.history_b])
        fixed_rewards_a = np.zeros(len(ACTIONS))
        for i, action in enumerate(ACTIONS):
            fixed_rewards_a[i] = np.sum(self.payoff_matrix[i, history_b])
        external_regret_a = np.max(fixed_rewards_a) - self.player_a.score
        print("\nExternal regret for Player A after all rounds:", external_regret_a)

        # step swap regret for Player A
        swap_regret_a = 0
        for i, action in enumerate(ACTIONS):
            replacement_rewards_a = np.zeros(len(ACTIONS))
            original_rewards_a = 0
            for j, action_j in enumerate(history_a):
                if history_a[j] == i:
                    original_rewards_a += self.payoff_matrix[action_j, history_b[j]]
                    for replacement_action in range(len(ACTIONS)):
                        replacement_rewards_a[replacement_action] += self.payoff_matrix[replacement_action, history_b[j]]
            best_replacement_reward_a = np.max(replacement_rewards_a)
            improvement_a = best_replacement_reward_a - original_rewards_a
            swap_regret_a += improvement_a
        print("\nSwap regret for Player A after all rounds:", swap_regret_a)
        print("\nAverage swap regret for Player A after all rounds:", swap_regret_a / rounds)

    def play(self, rounds=100, verbose=False):
        for step in range(rounds):
            # step 1: Calculate current policy
            P_a = self.get_transition_matrix(self.weights_a)
            P_b = self.get_transition_matrix(self.weights_b)
            # step 2: Calculate stationary distributions for both players
            pi_a = self.get_stationary_distribution(P_a, rounds)
            pi_b = self.get_stationary_distribution(P_b, rounds)
            # step 3: Sample actions for both players based on their stationary distributions
            action_a = np.random.choice(ACTIONS, p=pi_a)
            action_b = np.random.choice(ACTIONS, p=pi_b)
            # step 4: Play the round and get rewards
            self.player_a.action = action_a
            self.player_b.action = action_b
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            # step 5: store actions in history
            self.history_a.append(action_a)
            self.history_b.append(action_b)
            # step 6: reward for all possible actions of Player A and Player B against Player A's and B's action
            rewards_a = self.payoff_matrix[:, ACTIONS.index(action_b)]
            # step 7: Update every row's log-weights for Player A
            self.weights_a += self.learning_rate * pi_a[:, np.newaxis] * rewards_a[np.newaxis, :]
            # step 8: reward for all possible actions of Player B against Player A's action
            rewards_b = -self.payoff_matrix[ACTIONS.index(action_a), :]
            # step 9: Update every row's log-weights for Player B
            self.weights_b += self.learning_rate * pi_b[:, np.newaxis] * rewards_b[np.newaxis, :]
            if step + 1 in [rounds // 10 * i for i in range(1, 11)]:
                print(f"After {step + 1} rounds:")
                print("Stationary distribution for Player A:", np.round(pi_a, 3))
                print("Transition matrix for Player A:", np.round(P_a, 3))
                print("Stationary distribution for Player B:", np.round(pi_b, 3))
                print("Transition matrix for Player B:", np.round(P_b, 3))

        # step 10: Final stationary distribution and transition matrix for Player A and Player B after all rounds
        P_a = self.get_transition_matrix(self.weights_a)
        pi_a = self.get_stationary_distribution(P_a, rounds)
        P_b = self.get_transition_matrix(self.weights_b)
        pi_b = self.get_stationary_distribution(P_b, rounds)
        print("\nFinal stationary distribution for Player A after all rounds:", np.round(pi_a, 3))
        print("Final transition matrix for Player A after all rounds:", np.round(P_a, 3))
        print("Final stationary distribution for Player B after all rounds:", np.round(pi_b, 3))
        print("Final transition matrix for Player B after all rounds:", np.round(P_b, 3))
        #step 11: calculate external regret for Player A and Player B
        history_a = np.array([ACTIONS.index(action) for action in self.history_a])
        history_b = np.array([ACTIONS.index(action) for action in self.history_b])
        fixed_rewards_a = np.zeros(len(ACTIONS))
        fixed_rewards_b = np.zeros(len(ACTIONS))
        for i, action in enumerate(ACTIONS):
            fixed_rewards_a[i] = np.sum(self.payoff_matrix[i, history_b])
            fixed_rewards_b[i] = -np.sum(self.payoff_matrix[history_a, i])
        external_regret_a = np.max(fixed_rewards_a) - self.player_a.score
        external_regret_b = np.max(fixed_rewards_b) - self.player_b.score
        print("\nExternal regret for Player A after all rounds:", external_regret_a)
        print("External regret for Player B after all rounds:", external_regret_b)
        # step 12: calculate swap regret for Player A and Player B
        swap_regret_a = 0
        swap_regret_b = 0
        for i, action in enumerate(ACTIONS):
            replacement_rewards_a = np.zeros(len(ACTIONS))
            original_rewards_a = 0
            replacement_rewards_b = np.zeros(len(ACTIONS))
            original_rewards_b = 0
            for a, b in zip(history_a, history_b):
                if a == i:
                    original_rewards_a += self.payoff_matrix[a, b]
                    for replacement_action in range(len(ACTIONS)):
                        replacement_rewards_a[replacement_action] += self.payoff_matrix[replacement_action, b]
                if b == i:
                    original_rewards_b -= self.payoff_matrix[a, b]
                    for replacement_action in range(len(ACTIONS)):
                        replacement_rewards_b[replacement_action] -= self.payoff_matrix[a, replacement_action]
            best_replacement_reward_a = np.max(replacement_rewards_a)
            improvement_a = best_replacement_reward_a - original_rewards_a
            swap_regret_a += improvement_a
            best_replacement_reward_b = np.max(replacement_rewards_b)
            improvement_b = best_replacement_reward_b - original_rewards_b
            swap_regret_b += improvement_b
        print("\nSwap regret for Player A after all rounds:", swap_regret_a)
        print("Swap regret for Player B after all rounds:", swap_regret_b)
            