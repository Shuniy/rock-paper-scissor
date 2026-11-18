from src.player import Player
from src.rock_paper_scissors import RockPaperScissors

class Game:
    def __init__(self, player_a, player_b):
        self.player_a = player_a
        self.player_b = player_b
        self.rounds_played = 0
        self.reward = {
            self.player_a.name: 0,
            self.player_b.name: 0
        }
        self.rock_paper_scissors = RockPaperScissors()

    def play_round(self):
        action_a = self.player_a.action
        action_b = self.player_b.action
        reward_a, reward_b = self.rock_paper_scissors.play_round(action_a, action_b)
        self.player_a.score += reward_a
        self.player_b.score += reward_b
        self.rounds_played += 1
        return self.rounds_played, action_a, action_b, reward_a, reward_b

    def results(self):
        return {
            self.player_a.name: self.player_a.score,
            self.player_b.name: self.player_b.score
        }
