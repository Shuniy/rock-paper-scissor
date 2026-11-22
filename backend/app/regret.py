"""Round adapters for the research multiplicative-weights experiments."""
import numpy as np

from src.regret_learning.swap_learner import SwapLearnerGame

from .catalog import Event
from .rounds import ACTIONS, RPS, Score, draw


def hedge_policy(log_weights):
    weights = np.exp(log_weights - np.max(log_weights))
    return weights / weights.sum()


def stationary(matrix):
    # Markov-chain tree sums give the stationary mass for each root.
    # Compute products in log space so tiny transition probabilities remain usable.
    log_p = np.log(np.maximum(matrix, np.finfo(float).tiny))
    roots = np.zeros(3)
    for i in range(3):
        j, k = [x for x in range(3) if x != i]
        roots[i] = np.logaddexp.reduce([
            log_p[j, i] + log_p[k, i], log_p[j, k] + log_p[k, i], log_p[k, j] + log_p[j, i],
        ])
    return hedge_policy(roots)


def simulate(run_id, exp, p, conditions):
    a_draws = np.random.default_rng(p["seed"]).random(p["rounds"])
    rng_b = np.random.default_rng(p["seed"] + 1)
    b_draws = rng_b.random(p["rounds"])
    strategy = p["strategy"]
    rate = p["learning_rate"]
    for condition in conditions:
        swap_learner = SwapLearnerGame() if condition == "SwapMW" else None
        weights = swap_learner.weights_a if swap_learner else np.zeros(3)
        weights_b = np.zeros(3)
        score = Score(run_id, exp, condition)
        public = []
        for t, (random_a, random_b) in enumerate(zip(a_draws, b_draws)):
            detail = {"strategy": strategy, "pre_weights": np.exp(weights).tolist(),
                      "pre_log_weights": weights.tolist(), "learning_rate": rate}
            if swap_learner:
                matrix = swap_learner.get_transition_matrix(weights)
                policy = stationary(matrix)
                detail["transition"] = matrix.tolist()
            else:
                policy = hedge_policy(weights)
            if strategy == "hedge":
                b = draw(hedge_policy(weights_b), random_b)
            elif strategy == "fixed":
                b = ACTIONS.index(p["fixed_action"])
            elif strategy == "switch":
                b = 1 if t < p["rounds"] // 2 else 0
            else:
                b = int(random_b * 3)

            a = draw(policy, random_a)
            reward_vector = RPS[:, b]
            increment = rate * reward_vector
            if swap_learner:
                increment = policy[:, None] * increment[None, :]
            weights += increment
            if strategy == "hedge":
                weights_b -= rate * RPS[a, :]
            next_policy = (stationary(swap_learner.get_transition_matrix(weights))
                           if swap_learner else hedge_policy(weights))
            detail.update(counterfactual_rewards=reward_vector.tolist(), post_weights=np.exp(weights).tolist(),
                          post_log_weights=weights.tolist(), next_policy=next_policy.tolist(),
                          explanation=("Each row gets a full-information update scaled by its stationary source probability."
                                       if swap_learner else
                                       "Actions with higher counterfactual payoffs gain weight before the next normalization."))
            if strategy == "switch":
                detail["dynamic_regret"] = float(t + 1 - score.cumulative_a - RPS[a, b])
            row = score.record(a, b, policy=policy, phase=ACTIONS[b] if strategy == "switch" else None,
                               history=public[-8:], detail=detail)
            public.append({"a": row.a, "b": row.b})
            yield Event(type="round", data=row.model_dump(exclude_none=True))
