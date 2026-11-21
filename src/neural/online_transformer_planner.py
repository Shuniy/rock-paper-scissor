"""Learning-aware planner against an ONLINE-updated, history-conditioned transformer.

A sees previous B actions and updates with full-information expected reward.
B knows A's update rule and compares myopic exploitation with lookahead.
Optional differentiable swap-regret regularization applies to replayed observations.
"""
import copy
from itertools import product
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)

A_PAYOFF = torch.tensor([[1., -1., 0.], [0., 1., 0.]])  # [A, B]
B_PAYOFF = torch.tensor([[-0.1, 3., 0.2], [-0.1, -3., 0.2]])
B_NAMES = ["Bait", "Exploit", "Safe"]
HISTORY = 4
WINDOW = 6
ROUNDS = 20
HORIZON = 2
LR = 0.24

class OnlineTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 8
        self.token = nn.Embedding(3, d)
        self.position = nn.Embedding(HISTORY, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=2, dim_feedforward=16,
            dropout=0., batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.output = nn.Linear(d, 2)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, histories):
        pos = torch.arange(HISTORY, device=histories.device)
        x = self.token(histories) + self.position(pos)
        mask = torch.triu(torch.ones(HISTORY, HISTORY, dtype=torch.bool), diagonal=1)
        h = self.encoder(x, mask=mask)
        return self.output(h[:, -1]).softmax(-1)


def update_a(model, memory, history, b, lambda_swap):
    """One online gradient update using the latest WINDOW observations."""
    memory = (memory + [(history.clone(), b)])[-WINDOW:]
    contexts = torch.stack([h for h, _ in memory])
    b_actions = torch.tensor([b_ for _, b_ in memory], dtype=torch.long)
    p = model(contexts)                         # [W, 2]
    r = A_PAYOFF[:, b_actions].T                # [W, 2]
    reward = (p * r).sum(-1).mean()

    # swap[source, destination] = E[p(source) * (r(dest) - r(source))]
    delta = r[:, None, :] - r[:, :, None]     # [W, 2, 2]
    swap_gains = (p[:, :, None] * delta).mean(0)
    swap_surrogate = swap_gains.max(-1).values.sum()

    loss = -reward + lambda_swap * swap_surrogate
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter -= LR * parameter.grad
    return memory


def choose_b(model, history, memory, method, lambda_swap):
    with torch.no_grad():
        p = model(history[None])[0]
        immediate = p @ B_PAYOFF
    if method == "myopic":
        return int(immediate.argmax())

    best_value, best_action = float("-inf"), None
    for plan in product(range(3), repeat=HORIZON):
        simulated = copy.deepcopy(model)
        simulated_history = history.clone()
        simulated_memory = list(memory)
        value = 0.
        for b in plan:
            with torch.no_grad():
                value += float(simulated(simulated_history[None])[0] @ B_PAYOFF[:, b])
            simulated_memory = update_a(
                simulated, simulated_memory, simulated_history, b, lambda_swap)
            simulated_history = torch.cat((simulated_history[1:], torch.tensor([b])))
        if value > best_value:
            best_value, best_action = value, plan[0]
    return best_action


def run(method, lambda_swap, seed=42, verbose=False):
    torch.manual_seed(seed)                     # matched initial A weights per seed
    model = OnlineTransformer()
    history = torch.full((HISTORY,), 2, dtype=torch.long)  # start with Safe tokens
    memory = []
    rng = np.random.default_rng(seed)
    expected_b, realized_b = 0., 0.
    chosen_a, chosen_b = [], []
    if verbose:
        print(f"\n{method.upper()} / lambda={lambda_swap}")
    for t in range(ROUNDS):
        with torch.no_grad():
            p = model(history[None])[0].numpy()
        b = choose_b(model, history, memory, method, lambda_swap)
        a = int(rng.choice(2, p=p))
        expected_b += float(p @ B_PAYOFF[:, b].numpy())
        realized_b += float(B_PAYOFF[a, b])
        chosen_a.append(a)
        chosen_b.append(b)
        if verbose and t < 10:
            print(f"Round {t+1:2d}: P(X)={p[0]:.3f}, B={B_NAMES[b]:7s}, A={'X' if a == 0 else 'Y'}")
        memory = update_a(model, memory, history, b, lambda_swap)
        history = torch.cat((history[1:], torch.tensor([b])))

    # Actual sampled-action swap regret for A vs the recorded B history.
    a = torch.tensor(chosen_a)
    b = torch.tensor(chosen_b)
    r = A_PAYOFF[:, b].T
    actual_a_reward = r[torch.arange(ROUNDS), a].sum()
    external = (r.sum(0).max() - actual_a_reward) / ROUNDS
    regret_swap = 0.
    for source in range(2):
        chosen_rounds = a == source
        source_reward = r[chosen_rounds, source].sum()
        replacement_reward = r[chosen_rounds].sum(0).max()
        regret_swap += float(replacement_reward - source_reward)
    return dict(expected_b=expected_b / ROUNDS,
                realized_b=realized_b / ROUNDS,
                ext_a=float(external), swap_a=regret_swap / ROUNDS,
                bait=chosen_b.count(0), exploit=chosen_b.count(1), safe=chosen_b.count(2))


def experiment_online_transformer_planner():
    for lam in [0.0, 1.0]:
        for method in ["myopic", "planner"]:
            first = run(method, lam, verbose=True)
            results = [first] + [run(method, lam, seed=s) for s in [43, 44]]
            fields = ["expected_b", "realized_b", "ext_a", "swap_a"]
            means = {key: np.mean([r[key] for r in results]) for key in fields}
            print("Mean over 3 matched seeds:",
                  ", ".join(f"{key}={means[key]:.3f}" for key in fields))
            print(f"First seed B actions: Bait={first['bait']}, Exploit={first['exploit']}, Safe={first['safe']}")
