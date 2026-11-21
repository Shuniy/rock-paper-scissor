"""Project 2: black-box response modelling against a continuously training RPS transformer.

Put beside online_transformer_rps_model_mismatch.py.
B observes only prior B actions and sampled A actions; B never reads A's model,
parameters, policy, learning rate, objective, or rewards counterfactuals.
A's true policy is accessed only by the experiment/evaluator, and by the
explicitly labelled ORACLE diagnostic (not a black-box competitor).
"""
from itertools import product
import numpy as np
import torch
from src.neural.online_transformer_rps_model_mismatch import (
    ACTIONS, HISTORY, LR, OnlineTransformer, PAYOFF_B, update_a,
)

torch.set_num_threads(1)
PAYOFF_B_NP = PAYOFF_B.numpy()
WARMUP = 20
EVAL_ROUNDS = 40
SEEDS = range(10)
EXPLORATION_RATE = 0.10


class BlackBoxPredictor:
    """Recency-weighted estimates of P(A action | last 0/1/2 B actions)."""

    def __init__(self, forgetting=0.94, smoothing=3.0):
        self.evidence = {}   # suffix -> decayed action counts
        self.forgetting = forgetting
        self.smoothing = smoothing

    def update(self, b_history, observed_a):
        # The only label B uses is the ACTUAL sampled action of A.
        for counts in self.evidence.values():
            counts *= self.forgetting
        for k in range(3):
            key = tuple(b_history[-k:]) if k else ()
            if key not in self.evidence:
                self.evidence[key] = np.zeros(3, dtype=float)
            self.evidence[key][observed_a] += 1.0

    def predict(self, b_history):
        # Global smoothed estimate, then progressively context-specific.
        counts = self.evidence.get((), np.zeros(3))
        p = (counts + 1.0) / (counts.sum() + 3.0)
        for k in (1, 2):
            key = tuple(b_history[-k:])
            counts = self.evidence.get(key, np.zeros(3))
            # Shrink sparse conditional estimates toward their parent estimate.
            p = (counts + self.smoothing * p) / (
                counts.sum() + self.smoothing
            )
        return p


def choose_blackbox_b(estimator, b_history, strategy):
    if strategy == 'blackbox_myopic':
        return int(np.argmax(PAYOFF_B_NP @ estimator.predict(b_history)))
    horizon = {'blackbox_plan2': 2, 'blackbox_plan3': 3}[strategy]
    best_first, best_value = 0, -np.inf
    for plan in product(range(3), repeat=horizon):
        simulated_history = list(b_history)
        value = 0.0
        for b in plan:
            p_pred = estimator.predict(simulated_history)
            value += float(PAYOFF_B_NP[b] @ p_pred)
            simulated_history.append(b)
        if value > best_value:
            best_first, best_value = plan[0], value
    return int(best_first)


def run(strategy, lambda_swap, seed):
    torch.manual_seed(seed)
    agent_a = OnlineTransformer()
    agent_a.train()
    rng = np.random.default_rng(seed)
    history = rng.integers(0, 3, size=HISTORY).tolist()
    all_rounds = WARMUP + EVAL_ROUNDS
    a_draws = rng.random(all_rounds)
    # Shared random exploration choices across all strategies for a seed.
    explore_draws = rng.random(all_rounds)
    b_random_actions = rng.integers(0, 3, size=all_rounds)
    estimator = BlackBoxPredictor()
    memory = []

    expected_b = realized_b = pred_l1 = 0.0
    action_counts = np.zeros(3, dtype=int)

    for t in range(all_rounds):
        # EVALUATOR accesses the true policy; this array is NOT passed to
        # black-box decision-making or to the learned response estimator.
        with torch.no_grad():
            true_p = agent_a(torch.tensor([history], dtype=torch.long))[0].numpy().copy()
        predicted_p = estimator.predict(history)

        if t < WARMUP or strategy == 'uniform':
            b = int(b_random_actions[t])
        elif strategy == 'oracle_myopic':  # diagnostic only
            b = int(np.argmax(PAYOFF_B_NP @ true_p))
        elif explore_draws[t] < EXPLORATION_RATE:
            b = int(b_random_actions[t])
        else:
            b = choose_blackbox_b(estimator, history, strategy)

        a = min(int(np.searchsorted(np.cumsum(true_p), a_draws[t])), 2)

        if t >= WARMUP:
            action_counts[b] += 1
            expected_b += float(PAYOFF_B_NP[b] @ true_p)
            realized_b += float(PAYOFF_B_NP[b, a])
            pred_l1 += float(np.abs(predicted_p - true_p).sum() / 2)

        # Agent B observes the sampled A action, not A's true policy.
        estimator.update(history, a)
        # Agent A learns online using its true objective and true learning rate.
        h_tensor = torch.tensor(history, dtype=torch.long)
        memory = update_a(agent_a, memory, h_tensor, b, lambda_swap, learning_rate=LR)
        history = (history + [b])[-HISTORY:]

    return dict(
        expected_b=expected_b / EVAL_ROUNDS,
        realized_b=realized_b / EVAL_ROUNDS,
        prediction_tv=pred_l1 / EVAL_ROUNDS,
        counts=action_counts,
    )


def blackbox_experiment():
    strategies = [
        'uniform', 'blackbox_myopic', 'blackbox_plan2',
        'blackbox_plan3', 'oracle_myopic',
    ]
    print(f'Black-box opponent: {WARMUP} shared exploration rounds + '
          f'{EVAL_ROUNDS} evaluation rounds; {len(SEEDS)} seeds')
    print('B sees only action pairs. Oracle-myopic is a privileged diagnostic.')
    for lam in (0.0, 1.0):
        print(f'\n=== A objective: lambda={lam} ===')
        print(f'{"B strategy":<20} {"B expected":>11} {"B realized":>11} {"pred TV":>9}')
        for name in strategies:
            outcomes = [run(name, lam, s) for s in SEEDS]
            expected = np.mean([r['expected_b'] for r in outcomes])
            realized = np.mean([r['realized_b'] for r in outcomes])
            tv = np.mean([r['prediction_tv'] for r in outcomes])
            print(f'{name:<20} {expected:11.3f} {realized:11.3f} {tv:9.3f}')
    print('\nPrediction TV = mean total-variation distance between B\'s '
          'estimated A policy and the real A policy (evaluator only).')
    print('Planning assumes a fixed estimated response model during rollouts; '
          'this is behaviour-aware, not yet full identification of gradient updates.')
