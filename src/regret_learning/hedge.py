from src.game import Game
from src.player import Player
from src.constants import ACTIONS, ACTION
from src.rock_paper_scissors import RockPaperScissors
import numpy as np

class HedgeGame:
    def __init__(self, player_a_name="Player A", player_b_name="Player B"):
        self.player_a = Player(player_a_name)
        self.player_b = Player(player_b_name)
        self.game = Game(self.player_a, self.player_b)
        self.learning_rate = 0.1  # Example learning rate for Hedge algorithm
        self.strategy_a_weights = [1.0, 1.0, 1.0]  # Example mixed strategy for Player A
        self.strategy_b_weights = [1.0, 1.0, 1.0]  # Example mixed strategy for Player B
        self.cumulative_rewards_a = np.zeros(len(ACTIONS))
        self.cumulative_rewards_b = np.zeros(len(ACTIONS))

    def experiment_b(self, first_action, second_action, rounds=100, verbose=False, change_action_after=50):
        all_rewards = RockPaperScissors().payoff_matrix
        for step in range(rounds):
            probabilities_a = np.array(self.strategy_a_weights) / sum(self.strategy_a_weights)
            self.player_a.action = self.player_a.choose_action(probabilities=probabilities_a)
            player_b_fixed_action = first_action if step < change_action_after else second_action # player b always chooses the specified action
            self.player_b.action = self.player_b.choose_action(action=player_b_fixed_action)
            all_rewards = RockPaperScissors().payoff_matrix
            # if player b chooses action, then all the rewards for player a are the corresponding row in the payoff matrix
            all_rewards_for_a = all_rewards[:, ACTIONS.index(player_b_fixed_action)]
            self.cumulative_rewards_a += all_rewards_for_a
            # play the game
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            # update the weights for player a based on the reward received
            self.strategy_a_weights *= np.exp(self.learning_rate * all_rewards_for_a)
            if verbose:
                if step + 1 in [rounds // 10 * i for i in range(1, 11)]:  # Print every 10% of the rounds
                    print(f"Round {round_number}: {self.player_a.name} chose {action_a}, {self.player_b.name} chose {action_b}. Rewards: {self.player_a.name}: {reward_a}, {self.player_b.name}: {reward_b}")

        # First 50 rounds: Agent B plays Paper (1)
        best_first_half = np.max(all_rewards[:, 1]) * 50

        # Next 50 rounds: Agent B plays Rock (0)
        best_second_half = np.max(all_rewards[:, 0]) * 50

        # Best strategy allowed to change once
        best_dynamic_reward = best_first_half + best_second_half

        dynamic_regret = best_dynamic_reward - self.player_a.score

        print("\nDynamic Regret Analysis:")
        print("Best reward in first half:", best_first_half)
        print("Best reward in second half:", best_second_half)
        print("Best dynamic reward:", best_dynamic_reward)
        print("Hedge reward:", self.player_a.score)
        print("Dynamic regret:", dynamic_regret)
        # best action for player a is the action with the highest cumulative reward
        best_action_index = np.argmax(self.cumulative_rewards_a)
        best_action = ACTIONS[best_action_index]
        best_cumulative_reward = self.cumulative_rewards_a[best_action_index]
        print(f"Best action for {self.player_a.name} after {rounds} rounds: {best_action} with cumulative reward {self.cumulative_rewards_a[best_action_index]}")
        # regret for player a is the difference between the best action and the action actually taken
        regret_a = best_cumulative_reward - self.player_a.score
        print(f"Regret for {self.player_a.name} after {rounds} rounds: {regret_a}")
        average_regret_a = regret_a / rounds
        print(f"Average regret for {self.player_a.name} after {rounds} rounds: {average_regret_a}")
        print(f"Total reward for {self.player_a.name} after {rounds} rounds: {self.player_a.score}")
        print(f"Total reward for {self.player_b.name} after {rounds} rounds: {self.player_b.score}")
    

    def experiment_a(self, first_action, second_action, rounds=100, verbose=False, change_action_after=50):
        for step in range(rounds):
            probabilities_a = np.array(self.strategy_a_weights) / sum(self.strategy_a_weights)
            self.player_a.action = self.player_a.choose_action(probabilities=probabilities_a)
            player_b_fixed_action = first_action if step < change_action_after else second_action # player b always chooses the specified action
            self.player_b.action = self.player_b.choose_action(action=player_b_fixed_action)
            all_rewards = RockPaperScissors().payoff_matrix
            # if player b chooses action, then all the rewards for player a are the corresponding row in the payoff matrix
            all_rewards_for_a = all_rewards[:, ACTIONS.index(player_b_fixed_action)]
            self.cumulative_rewards_a += all_rewards_for_a
            # play the game
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            # update the weights for player a based on the reward received
            self.strategy_a_weights *= np.exp(self.learning_rate * all_rewards_for_a)
            if verbose:
                if step + 1 in [rounds // 10 * i for i in range(1, 11)]:  # Print every 10% of the rounds
                    print(f"Round {round_number}: {self.player_a.name} chose {action_a}, {self.player_b.name} chose {action_b}. Rewards: {self.player_a.name}: {reward_a}, {self.player_b.name}: {reward_b}")

        # best action for player a is the action with the highest cumulative reward
        best_action_index = np.argmax(self.cumulative_rewards_a)
        best_action = ACTIONS[best_action_index]
        best_cumulative_reward = self.cumulative_rewards_a[best_action_index]
        print(f"Best action for {self.player_a.name} after {rounds} rounds: {best_action} with cumulative reward {self.cumulative_rewards_a[best_action_index]}")
        # regret for player a is the difference between the best action and the action actually taken
        regret_a = best_cumulative_reward - self.player_a.score
        print(f"Regret for {self.player_a.name} after {rounds} rounds: {regret_a}")
        average_regret_a = regret_a / rounds
        print(f"Average regret for {self.player_a.name} after {rounds} rounds: {average_regret_a}")
        print(f"Total reward for {self.player_a.name} after {rounds} rounds: {self.player_a.score}")
        print(f"Total reward for {self.player_b.name} after {rounds} rounds: {self.player_b.score}")

    def play_fixed_b(self, action, rounds=100, verbose=False):
        for step in range(rounds):
            probabilities_a = np.array(self.strategy_a_weights) / sum(self.strategy_a_weights)
            self.player_a.action = self.player_a.choose_action(probabilities=probabilities_a)
            player_b_fixed_action = action # player b always chooses the specified action
            self.player_b.action = self.player_b.choose_action(action=player_b_fixed_action)
            all_rewards = RockPaperScissors().payoff_matrix
            # if player b chooses action, then all the rewards for player a are the corresponding row in the payoff matrix
            all_rewards_for_a = all_rewards[:, ACTIONS.index(player_b_fixed_action)]
            self.cumulative_rewards_a += all_rewards_for_a
            # play the game
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            # update the weights for player a based on the reward received
            self.strategy_a_weights *= np.exp(self.learning_rate * all_rewards_for_a)
            if verbose:
                if step + 1 in [rounds // 10 * i for i in range(1, 11)]:   # Print every 10% of the rounds
                    print(f"Round {round_number}: {self.player_a.name} chose {action_a}, {self.player_b.name} chose {action_b}. Rewards: {self.player_a.name}: {reward_a}, {self.player_b.name}: {reward_b}")

        # best action for player a is the action with the highest cumulative reward
        best_action_index = np.argmax(self.cumulative_rewards_a)
        best_action = ACTIONS[best_action_index]
        best_cumulative_reward = self.cumulative_rewards_a[best_action_index]
        print(f"Best action for {self.player_a.name} after {rounds} rounds: {best_action} with cumulative reward {self.cumulative_rewards_a[best_action_index]}")
        # regret for player a is the difference between the best action and the action actually taken
        regret_a = best_cumulative_reward - self.player_a.score
        print(f"Regret for {self.player_a.name} after {rounds} rounds: {regret_a}")
        average_regret_a = regret_a / rounds
        print(f"Average regret for {self.player_a.name} after {rounds} rounds: {average_regret_a}")
        print(f"Total reward for {self.player_a.name} after {rounds} rounds: {self.player_a.score}")
        print(f"Total reward for {self.player_b.name} after {rounds} rounds: {self.player_b.score}")

    def play(self, rounds=100, verbose=False):
        for step in range(rounds):
            probabilities_a = np.array(self.strategy_a_weights) / sum(self.strategy_a_weights)
            probabilities_b = np.array(self.strategy_b_weights) / sum(self.strategy_b_weights)
            self.player_a.action = self.player_a.choose_action(probabilities=probabilities_a)
            self.player_b.action = self.player_b.choose_action(probabilities=probabilities_b)
            all_rewards = RockPaperScissors().payoff_matrix
            # if player b chooses action, then all the rewards for player a are the corresponding row in the payoff matrix
            all_rewards_for_a = all_rewards[:, ACTIONS.index(self.player_b.action)]
            all_rewards_for_b = all_rewards[ACTIONS.index(self.player_a.action), :]
            self.cumulative_rewards_a += all_rewards_for_a
            self.cumulative_rewards_b += all_rewards_for_b
            # play the game
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            # update the weights for player a based on the reward received
            self.strategy_a_weights *= np.exp(self.learning_rate * all_rewards_for_a)
            self.strategy_b_weights *= np.exp(self.learning_rate * all_rewards_for_b)
            if verbose:
                if step + 1 in [rounds // 10 * i for i in range(1, 11)]:   # Print every 10% of the rounds
                    print(f"Round {round_number}: {self.player_a.name} chose {action_a}, {self.player_b.name} chose {action_b}. Rewards: {self.player_a.name}: {reward_a}, {self.player_b.name}: {reward_b}")
    
        # best action for player a is the action with the highest cumulative reward
        best_action_index_a = np.argmax(self.cumulative_rewards_a)
        best_action = ACTIONS[best_action_index_a]
        best_cumulative_reward_a = self.cumulative_rewards_a[best_action_index_a]
        print(f"Best action for {self.player_a.name} after {rounds} rounds: {best_action} with cumulative reward {self.cumulative_rewards_a[best_action_index_a]}")
        # regret for player a is the difference between the best action and the action actually taken
        regret_a = best_cumulative_reward_a - self.player_a.score
        print(f"Regret for {self.player_a.name} after {rounds} rounds: {regret_a}")
        average_regret_a = regret_a / rounds
        print(f"Average regret for {self.player_a.name} after {rounds} rounds: {average_regret_a}")
        print(f"Total reward for {self.player_a.name} after {rounds} rounds: {self.player_a.score}")
        
        # best action for player b is the action with the highest cumulative reward
        best_action_index_b = np.argmax(self.cumulative_rewards_b)
        best_action_b = ACTIONS[best_action_index_b]
        best_cumulative_reward_b = self.cumulative_rewards_b[best_action_index_b]
        print(f"Best action for {self.player_b.name} after {rounds} rounds: {best_action_b} with cumulative reward {self.cumulative_rewards_b[best_action_index_b]}")
        # regret for player b is the difference between the best action and the action actually taken
        regret_b = best_cumulative_reward_b - self.player_b.score
        print(f"Regret for {self.player_b.name} after {rounds} rounds: {regret_b}")
        average_regret_b = regret_b / rounds
        print(f"Average regret for {self.player_b.name} after {rounds} rounds: {average_regret_b}")
        print(f"Total reward for {self.player_b.name} after {rounds} rounds: {self.player_b.score}")
