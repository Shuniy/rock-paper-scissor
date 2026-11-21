"""RPS: learning-aware planning against an ONLINE-learning transformer.

A sees the last HISTORY B actions and chooses Rock / Paper / Scissors.
A updates its neural weights after each round with full-information reward
and an optional differentiable swap-regret regularizer.
B knows A's current network and update rule: compare myopic vs lookahead.

This is a controlled simulation; the results do not establish no-swap-regret
or a real-world attack on a pretrained LLM.
"""

import copy
from itertools import product

import numpy as np
import torch
from torch import nn


torch.set_num_threads(1)

ACTIONS = ["Rock", "Paper", "Scissors"]
PAYOFF_A = torch.tensor([
    [0.0, -1.0,  1.0],
    [1.0,  0.0, -1.0],
    [-1.0, 1.0,  0.0],
])  # rows: A actions, columns: B actions
PAYOFF_B = -PAYOFF_A.T  # rows: B actions, columns: A actions

HISTORY = 4  # how many previous B moves A observes
WINDOW = 6   # number of past observations used per gradient step
ROUNDS = 20
HORIZON = 2  # B considers 3**HORIZON action sequences
LR = 0.24


class OnlineTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 8
        self.token = nn.Embedding(3, d)
        self.position = nn.Embedding(HISTORY, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d,
            nhead=2,
            dim_feedforward=16,
            dropout=0.0,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.output = nn.Linear(d, 3)  # A: Rock, Paper, Scissors

        # Make A initially uniform for every history.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(HISTORY, HISTORY, dtype=torch.bool), diagonal=1),
        )

    def forward(self, histories):
        positions = torch.arange(HISTORY, device=histories.device)
        x = self.token(histories) + self.position(positions)
        h = self.encoder(x, mask=self.causal_mask)
        return self.output(h[:, -1, :]).softmax(dim=-1)


def update_a(model, memory, history, b, lambda_swap):
    """A's actual online update (also used for B's simulated rollouts)."""
    memory = (memory + [(history.clone(), int(b))])[-WINDOW:]
    contexts = torch.stack([h for h, _ in memory])
    b_actions = torch.tensor([act for _, act in memory], dtype=torch.long)

    p = model(contexts)                 # [window, A actions]
    rewards = PAYOFF_A[:, b_actions].T  # [window, A actions]
    expected_reward = (p * rewards).sum(dim=-1).mean()

    # delta[t, source, destination] = gain from replacing source with dest
    delta = rewards[:, None, :] - rewards[:, :, None]
    swap_gains = (p[:, :, None] * delta).mean(dim=0)
    swap_surrogate = swap_gains.max(dim=-1).values.sum()

    loss = -expected_reward + lambda_swap * swap_surrogate
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter -= LR * parameter.grad
    return memory


@torch.no_grad()
def b_expected_rewards(model, history):
    p = model(history.unsqueeze(0))[0]
    return PAYOFF_B @ p  # B rewards for Rock / Paper / Scissors


def choose_b(model, history, memory, method, lambda_swap):
    if method == "myopic":
        return int(b_expected_rewards(model, history).argmax())
    if method != "planner":
        raise ValueError(f"Unknown B strategy: {method}")

    best_value, best_first = -float("inf"), None

    for plan in product(range(3), repeat=HORIZON):
        simulated_model = copy.deepcopy(model)
        simulated_history = history.clone()
        simulated_memory = list(memory)
        value = 0.0

        for b in plan:
            with torch.no_grad():
                value += float(b_expected_rewards(simulated_model, simulated_history)[b])

            # Predict A's gradient update caused by this hypothetical B move.
            simulated_memory = update_a(
                simulated_model, simulated_memory,
                simulated_history, b, lambda_swap,
            )

            # A also observes B's action as its new history token.
            simulated_history = torch.cat(
                (simulated_history[1:], torch.tensor([b], dtype=torch.long))
            )

        if value > best_value:
            best_value, best_first = value, plan[0]

    # Receding-horizon planning: execute only the first move; replan next round.
    return int(best_first)


def measured_regrets(a_actions, b_actions):
    a = torch.tensor(a_actions, dtype=torch.long)
    b = torch.tensor(b_actions, dtype=torch.long)
    reward_matrix = PAYOFF_A[:, b].T  # [round, candidate A action]
    realized_a = reward_matrix[torch.arange(len(a)), a].sum()

    external = (reward_matrix.sum(dim=0).max() - realized_a) / len(a)

    swap_total = 0.0
    for source in range(3):
        selected = a == source
        original = reward_matrix[selected, source].sum()
        best_replacement = reward_matrix[selected].sum(dim=0).max()
        swap_total += float(best_replacement - original)

    swap = swap_total / len(a)
    assert swap >= max(0.0, float(external)) - 1e-6
    return float(external), swap


def run(method, lambda_swap, seed=42, verbose=False):
    torch.manual_seed(seed)  # identical initial transformer for each comparison
    model = OnlineTransformer()
    model.train()

    # Fixed, seeded initial history; identical for myopic vs planner.
    rng = np.random.default_rng(seed)
    history = torch.tensor(rng.integers(0, 3, size=HISTORY), dtype=torch.long)
    draws = rng.random(ROUNDS)  # matched A-action sampling randomness

    memory = []
    expected_b, realized_b = 0.0, 0.0
    a_actions, b_actions = [], []

    if verbose:
        print(f"\n{method.upper()} / lambda={lambda_swap}")
        print("Initial B history:", [ACTIONS[int(x)] for x in history])

    for t in range(ROUNDS):
        with torch.no_grad():
            p = model(history.unsqueeze(0))[0].numpy().copy()

        b = choose_b(model, history, memory, method, lambda_swap)
        a = min(int(np.searchsorted(np.cumsum(p), draws[t], side="left")), 2)

        expected_b += float((PAYOFF_B @ torch.tensor(p))[b])
        realized_b += float(PAYOFF_B[b, a])
        a_actions.append(a)
        b_actions.append(b)

        if verbose and t < 10:
            print(
                f"Round {t + 1:2d}: "
                f"P(A)={np.round(p, 3)}, "
                f"B={ACTIONS[b]:8s}, A={ACTIONS[a]}"
            )

        # Actual update: same update_a used in every candidate plan.
        memory = update_a(model, memory, history, b, lambda_swap)
        history = torch.cat((history[1:], torch.tensor([b], dtype=torch.long)))

    external, swap = measured_regrets(a_actions, b_actions)
    return {
        "expected_b": expected_b / ROUNDS,
        "realized_b": realized_b / ROUNDS,
        "ext_a": external,
        "swap_a": swap,
        "b_counts": [b_actions.count(i) for i in range(3)],
    }


def experiment_strategic_online_transformer_planner():
    seeds = [42, 43, 44]
    for lam in [0.0, 1.0]:
        for method in ["myopic", "planner"]:
            outcomes = [run(method, lam, seed=s, verbose=(s == seeds[0])) for s in seeds]
            print("Mean over matched seeds:", ", ".join(
                f"{field}={np.mean([result[field] for result in outcomes]):.3f}"
                for field in ["expected_b", "realized_b", "ext_a", "swap_a"]
            ))
            print("First seed B action counts:", dict(zip(ACTIONS, outcomes[0]["b_counts"])))
