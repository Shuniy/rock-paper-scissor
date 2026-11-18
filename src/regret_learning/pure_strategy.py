from src.game import Game
from src.player import Player
from src.constants import ACTION

class PureStrategyGame:
    def __init__(self, player_a_name="Player A", player_b_name="Player B"):
        self.player_a = Player(player_a_name)
        self.player_b = Player(player_b_name)
        self.game = Game(self.player_a, self.player_b)

    def play(self, rounds=100, verbose=False):
        for _ in range(rounds):
            self.player_b.action = self.player_b.choose_action() # Player B chooses an action randomly
            self.player_a.action = self.player_a.choose_action(action=ACTION.ROCK)  # Example: Player A always chooses ROCK
            round_number, action_a, action_b, reward_a, reward_b = self.game.play_round()
            if verbose:
                print(f"Round {round_number}: {self.player_a.name} chose {action_a}, {self.player_b.name} chose {action_b}. Rewards: {self.player_a.name}: {reward_a}, {self.player_b.name}: {reward_b}")
        print(f"Total reward for {self.player_a.name} after {rounds} rounds: {self.player_a.score}")
        print(f"Total reward for {self.player_b.name} after {rounds} rounds: {self.player_b.score}")