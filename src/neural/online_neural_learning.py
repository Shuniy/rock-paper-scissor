
import copy
from itertools import product

import numpy as np
import torch
from torch import nn

torch.set_num_threads(2)

A_PAYOFF = torch.tensor([
    [1., -1., 0.],
    [0.,  1., 0.]
])

B_PAYOFF = torch.tensor([
    [-0.1,  3., 0.2],
    [-0.1, -3., 0.2]
])

B_ACTIONS = ["Bait", "Exploit", "Safe"]

LR = 0.3
LOOKAHEAD = 3
ROUNDS = 30


# STEP 1: Agent A is a neural network
class NeuralLearner(nn.Module):

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(1, 8),
            nn.Tanh(),
            nn.Linear(8, 2)
        )

        # Start with equal action probabilities
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self):

        x = torch.ones(1, 1)

        logits = self.layers(x)

        return logits.softmax(dim=-1)[0]


# STEP 2: Agent A learns after observing B
def update_a(model, b):

    p = model()

    # Maximize A's expected reward
    loss = -(p @ A_PAYOFF[:, b])

    model.zero_grad()
    loss.backward()

    # Online gradient-descent update
    with torch.no_grad():
        for param in model.parameters():
            param -= LR * param.grad


# STEP 3: Agent B chooses a strategy
def choose_b(model, method):

    if method == "myopic":

        with torch.no_grad():
            p = model()
            return int((p @ B_PAYOFF).argmax())

    if method != "planner":
        raise ValueError(method)

    best_value = float("-inf")
    best_first = None

    # Try all 3^LOOKAHEAD possible action plans
    for plan in product(range(3), repeat=LOOKAHEAD):

        # Don't modify the actual neural agent
        simulated = copy.deepcopy(model)

        value = 0.0

        for b in plan:

            with torch.no_grad():

                # B's expected reward in this round
                value += float(
                    simulated() @ B_PAYOFF[:, b]
                )

            # Predict how A's neural weights change
            update_a(simulated, b)

        if value > best_value:
            best_value = value
            best_first = plan[0]

    return best_first


# STEP 4: Play a complete episode
def play(method, seed=42):

    torch.manual_seed(seed)

    model = NeuralLearner()

    rng = np.random.default_rng(seed)
    random_draws = rng.random(ROUNDS)

    expected_total = 0.0
    realized_total = 0.0

    print(f"\n{method.upper()}")

    for t in range(ROUNDS):

        with torch.no_grad():
            p = model().numpy()

        # B chooses its action
        b = choose_b(model, method)

        # A samples its action
        a = 0 if random_draws[t] < p[0] else 1

        expected_total += float(
            p @ B_PAYOFF[:, b].numpy()
        )

        realized_total += float(B_PAYOFF[a, b])

        if t < 10:
            print(
                f"Round {t + 1}: "
                f"P(X)={p[0]:.3f}, "
                f"B={B_ACTIONS[b]}, "
                f"A={'X' if a == 0 else 'Y'}"
            )

        # Crucial: actual neural-network training
        update_a(model, b)

    print(
        "Avg expected B reward:",
        round(expected_total / ROUNDS, 3)
    )

    print(
        "Avg realized B reward:",
        round(realized_total / ROUNDS, 3)
    )


def play_experiment_online_learning_aware():
    print("\nOnline Learning-Aware Experiment")
    play("myopic")
    play("planner")
