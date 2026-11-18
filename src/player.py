from src.constants import RNG, ACTIONS, ACTION

class Player:
    def __init__(self, name):
        self.name = name
        self.score = 0
        self.actions = ACTIONS
        self.action = None

    def choose_action(self, action=None, probabilities=None):
        if action is not None:
            return action
        return RNG.choice(self.actions, p=probabilities)

