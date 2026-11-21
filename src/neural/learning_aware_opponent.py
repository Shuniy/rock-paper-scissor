
from itertools import product
import numpy as np

ACTIONS_A = ["X", "Y"]
ACTIONS_B = ["Bait", "Exploit", "Safe"]

# Rows: A's actions
# Columns: B's actions

A_PAYOFF = np.array([
    [1.0, -1.0, 0.0],
    [0.0,  1.0, 0.0]
])

B_PAYOFF = np.array([
    [-0.1,  3.0, 0.2],
    [-0.1, -3.0, 0.2]
])

ETA = 0.5
HORIZON = 3
ROUNDS = 30


# Agent A's current policy
def policy(log_weights):

    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)

    return weights / weights.sum()


# Agent B chooses an action
def choose_b(log_weights, method):

    # Immediate best response
    if method == "myopic":

        p_a = policy(log_weights)

        return int(np.argmax(p_a @ B_PAYOFF))

    if method != "planner":
        raise ValueError(method)

    # Learning-aware planning
    best_value = float("-inf")
    best_first_action = None

    for plan in product(range(3), repeat=HORIZON):

        simulated_weights = log_weights.copy()
        total_expected_reward = 0.0

        for b in plan:

            p_a = policy(simulated_weights)

            # B's expected immediate reward
            total_expected_reward += float(
                p_a @ B_PAYOFF[:, b]
            )

            # Simulate how A learns from B's action
            simulated_weights += (
                ETA * A_PAYOFF[:, b]
            )

        if total_expected_reward > best_value:

            best_value = total_expected_reward
            best_first_action = plan[0]

    return best_first_action


# Run one episode
def play(method, seed=42):

    rng = np.random.default_rng(seed)

    # A starts with equal preferences
    log_weights = np.zeros(2)

    total_expected_b = 0.0
    total_realized_b = 0.0

    print(f"\n{method.upper()}")

    for t in range(ROUNDS):

        p_a = policy(log_weights)

        # B observes A's policy
        b = choose_b(log_weights, method)

        # A samples its action
        a = int(rng.choice(2, p=p_a))

        # Record B's rewards
        total_expected_b += float(
            p_a @ B_PAYOFF[:, b]
        )

        total_realized_b += B_PAYOFF[a, b]

        if t < 10:
            print(
                f"Round {t + 1}: "
                f"P(A=X)={p_a[0]:.3f}, "
                f"B={ACTIONS_B[b]}, "
                f"A={ACTIONS_A[a]}"
            )

        # A learns from the observed B action
        log_weights += ETA * A_PAYOFF[:, b]

    print(
        "B average expected reward:",
        round(total_expected_b / ROUNDS, 3)
    )

    print(
        "B average realized reward:",
        round(total_realized_b / ROUNDS, 3)
    )


def play_experiment():
    print("\nLearning-Aware Opponent Experiment")
    play("myopic")
    play("planner")
