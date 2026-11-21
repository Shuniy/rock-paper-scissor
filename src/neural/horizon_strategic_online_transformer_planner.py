"""Project 2, Step 3: Compare RPS planning horizons against online-learning A.

Requires: torch, numpy. All baselines retrain A online from identical
initial seeds; the payoff is zero-sum. Planner simulates exact updates.
"""
import copy
from itertools import product
import numpy as np
import torch
from torch import nn

torch.set_num_threads(1)
ACTIONS = ['Rock', 'Paper', 'Scissors']
PAYOFF_A = torch.tensor([[0., -1., 1.], [1., 0., -1.], [-1., 1., 0.]])
PAYOFF_B = -PAYOFF_A.T
HISTORY = 4
WINDOW = 6
ROUNDS = 20
LR = 0.24
SEEDS = [42, 43, 44, 45, 46]

class OnlineTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        d = 8
        self.token = nn.Embedding(3, d)
        self.position = nn.Embedding(HISTORY, d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=2, dim_feedforward=16,
            dropout=0.0, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.output = nn.Linear(d, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer('causal_mask', torch.triu(
            torch.ones(HISTORY, HISTORY, dtype=torch.bool), diagonal=1
        ))

    def forward(self, histories):
        positions = torch.arange(HISTORY, device=histories.device)
        x = self.token(histories) + self.position(positions)
        h = self.encoder(x, mask=self.causal_mask)
        return self.output(h[:, -1, :]).softmax(dim=-1)


def update_a(model, memory, history, b, lambda_swap):
    memory = (memory + [(history.clone(), int(b))])[-WINDOW:]
    contexts = torch.stack([h for h, _ in memory])
    b_actions = torch.tensor([a for _, a in memory], dtype=torch.long)
    p = model(contexts)
    rewards = PAYOFF_A[:, b_actions].T
    expected_reward = (p * rewards).sum(dim=-1).mean()
    delta = rewards[:, None, :] - rewards[:, :, None]
    swap_gains = (p[:, :, None] * delta).mean(dim=0)
    swap_surrogate = swap_gains.max(dim=-1).values.sum()
    loss = -expected_reward + lambda_swap * swap_surrogate
    model.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            param -= LR * param.grad
    return memory


@torch.no_grad()
def b_expected_rewards(model, history):
    p = model(history.unsqueeze(0))[0]
    return PAYOFF_B @ p


def choose_b(model, history, memory, strategy, lambda_swap):
    if strategy == 'myopic':
        return int(b_expected_rewards(model, history).argmax())

    if not strategy.startswith('plan'):
        raise ValueError(strategy)
    horizon = int(strategy.removeprefix('plan'))
    if horizon < 1:
        raise ValueError(horizon)

    best_value, best_first = -float('inf'), 0
    for plan in product(range(3), repeat=horizon):
        simulated_model = copy.deepcopy(model)
        simulated_history = history.clone()
        simulated_memory = list(memory)
        value = 0.0
        for b in plan:
            with torch.no_grad():
                value += float(b_expected_rewards(simulated_model, simulated_history)[b])
            simulated_memory = update_a(
                simulated_model, simulated_memory, simulated_history,
                b, lambda_swap
            )
            simulated_history = torch.cat((
                simulated_history[1:], torch.tensor([b], dtype=torch.long)
            ))
        if value > best_value:
            best_value, best_first = value, plan[0]
    return int(best_first)


def measured_regrets(a_actions, b_actions):
    a = torch.tensor(a_actions, dtype=torch.long)
    b = torch.tensor(b_actions, dtype=torch.long)
    rewards = PAYOFF_A[:, b].T
    actual = rewards[torch.arange(len(a)), a].sum()
    external = float(rewards.sum(0).max() - actual) / len(a)
    swap = 0.0
    for source in range(3):
        selected = a == source
        original = rewards[selected, source].sum()
        alternative = rewards[selected].sum(0).max()
        swap += float(alternative - original)
    swap /= len(a)
    assert swap >= max(0.0, external) - 1e-6
    return external, swap


def run(strategy, lambda_swap, seed):
    torch.manual_seed(seed)
    model = OnlineTransformer()
    model.train()

    # A has identical initial history and random numbers across strategies.
    rng = np.random.default_rng(seed)
    history = torch.tensor(rng.integers(0, 3, size=HISTORY), dtype=torch.long)
    a_draws = rng.random(ROUNDS)
    b_rng = np.random.default_rng(seed + 100000)

    memory = []
    expected_b = 0.0
    realized_b = 0.0
    entropies = []
    a_actions, b_actions = [], []
    for t in range(ROUNDS):
        with torch.no_grad():
            p = model(history.unsqueeze(0))[0].numpy().copy()

        if strategy == 'uniform':
            b = int(b_rng.integers(0, 3))
        elif strategy.startswith('fixed_'):
            b = ACTIONS.index(strategy.removeprefix('fixed_'))
        else:
            b = choose_b(model, history, memory, strategy, lambda_swap)

        a = min(int(np.searchsorted(np.cumsum(p), a_draws[t])), 2)
        expected_b += float((PAYOFF_B @ torch.tensor(p))[b])
        realized_b += float(PAYOFF_B[b, a])
        entropies.append(float(-np.sum(p * np.log(np.clip(p, 1e-12, 1))) / np.log(3)))
        a_actions.append(a)
        b_actions.append(b)
        memory = update_a(model, memory, history, b, lambda_swap)
        history = torch.cat((history[1:], torch.tensor([b], dtype=torch.long)))

    external, swap = measured_regrets(a_actions, b_actions)
    return dict(
        expected_b=expected_b / ROUNDS,
        realized_b=realized_b / ROUNDS,
        ext_a=external,
        swap_a=swap,
        a_entropy=np.mean(entropies),
    )


def experiment_horizon_strategic_online_transformer_planner():
    strategies = [
        'uniform', 'fixed_Rock', 'fixed_Paper', 'fixed_Scissors',
        'myopic', 'plan2', 'plan3'
    ]
    for lam in [0.0, 1.0]:
        print(f'\n=== Agent A: lambda={lam}; {len(SEEDS)} seeds, {ROUNDS} rounds ===')
        print(f'{"B strategy":<16} {"B reward":>9} {"B realized":>11} {"A ext":>8} {"A swap":>8} {"A entropy":>10}')
        for strategy in strategies:
            results = [run(strategy, lam, seed) for seed in SEEDS]
            def avg(key):
                return np.mean([result[key] for result in results])
            print(
                f'{strategy:<16} {avg("expected_b"):9.3f} '
                f'{avg("realized_b"):11.3f} {avg("ext_a"):8.3f} '
                f'{avg("swap_a"):8.3f} {avg("a_entropy"):10.3f}'
            )
    print('\nA entropy is normalized: 1 = uniform, 0 = deterministic.')
    print('Use 20+ seeds for stronger comparisons after a quick smoke test.')