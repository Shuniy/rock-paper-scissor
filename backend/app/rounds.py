"""Score completed decisions using a fixed recorded opponent sequence."""
import numpy as np

from src.constants import ACTIONS
from src.rock_paper_scissors import RockPaperScissors

from .catalog import Round

RPS = RockPaperScissors().payoff_matrix


def window(items, size):
    return items[-size:] if size else []


def draw(policy, uniform):
    return min(int(np.searchsorted(np.cumsum(policy), uniform, side="right")), len(policy) - 1)


class Score:
    def __init__(self, run_id, experiment_id, condition, payoff_a=RPS, payoff_b=None,
                 actions_a=ACTIONS, actions_b=ACTIONS):
        self.ids = dict(run_id=run_id, experiment_id=experiment_id, condition=condition)
        self.payoff_a = payoff_a
        self.payoff_b = -payoff_a if payoff_b is None else payoff_b
        self.actions_a = actions_a
        self.actions_b = actions_b
        self.fixed = np.zeros(len(actions_a))
        self.gains = np.zeros((len(actions_a), len(actions_a)))
        self.counts = np.zeros(len(actions_a))
        self.cumulative_a = self.cumulative_b = 0.0
        self.round = 0

    def record(self, a, b, policy=None, phase=None, history=None, detail=None):
        counterfactual = self.payoff_a[:, b]
        reward_a = float(counterfactual[a])
        reward_b = float(self.payoff_b[a, b])
        self.cumulative_a += reward_a
        self.cumulative_b += reward_b
        self.fixed += counterfactual
        self.gains[a] += counterfactual - reward_a
        self.round += 1
        self.counts[a] += 1
        return Round(**self.ids, round=self.round, a=self.actions_a[a], b=self.actions_b[b],
                     reward_a=reward_a, reward_b=reward_b, cumulative_a=self.cumulative_a,
                     cumulative_b=self.cumulative_b, average_a=self.cumulative_a / self.round,
                     average_b=self.cumulative_b / self.round, external=float(self.fixed.max() - self.cumulative_a),
                     swap=float(self.gains.max(axis=1).sum()), frequency=(self.counts / self.round).tolist(),
                     policy=policy.tolist() if policy is not None else None, phase=phase, history=history,
                     detail=detail or {})
