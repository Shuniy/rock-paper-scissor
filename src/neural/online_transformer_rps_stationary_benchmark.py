"""Project 2, Step 4: Test model-mismatch in learning-aware RPS planning.

Requires: torch, numpy. Agent A learns online at the TRUE LR and objective.
Agent B may plan with an incorrect assumed LR or swap coefficient.
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


def update_a(model, memory, history, b, lambda_swap, learning_rate=LR):
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
            param -= learning_rate * param.grad
    return memory


@torch.no_grad()
def b_expected_rewards(model, history):
    p = model(history.unsqueeze(0))[0]
    return PAYOFF_B @ p


def choose_b(model, history, memory, strategy, lambda_swap, assumed_lr=None, assumed_lambda=None):
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
                b,
                lambda_swap if assumed_lambda is None else assumed_lambda,
                learning_rate=LR if assumed_lr is None else assumed_lr
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



# Project 2, Step 5: Compare learning-aware planning with a stationary mixed commitment.
# We select a stationary distribution on separate tuning seeds, then evaluate
# the selected commitment and adaptive strategies on held-out seeds.

GRID_DENOMINATOR = 4  # probabilities in steps of 0.25 (15 candidate mixtures)
TUNE_SEEDS = list(range(200, 205))  # 5 seeds, used ONLY for choosing the mixture
TEST_SEEDS = list(range(42, 47))  # match earlier planner results; disjoint from tuning


def run(strategy, lambda_swap, seed, mix=None):
    torch.manual_seed(seed)
    model = OnlineTransformer()
    model.train()

    rng = np.random.default_rng(seed)
    history = torch.tensor(rng.integers(0, 3, size=HISTORY), dtype=torch.long)
    a_draws = rng.random(ROUNDS)
    # Common B draws for all fixed mixtures on the same seed
    b_draws = np.random.default_rng(seed + 100000).random(ROUNDS)

    memory = []
    expected_b = 0.0
    realized_b = 0.0
    entropies = []
    a_actions, b_actions = [], []

    if strategy == 'mixed':
        if mix is None or len(mix) != 3 or not np.isclose(sum(mix), 1.0):
            raise ValueError('mix must be three probabilities summing to 1')
        mix_cdf = np.cumsum(np.array(mix, dtype=float))

    for t in range(ROUNDS):
        with torch.no_grad():
            p = model(history.unsqueeze(0))[0].numpy().copy()

        if strategy == 'mixed':
            b = min(int(np.searchsorted(mix_cdf, b_draws[t])), 2)
        else:
            b = choose_b(model, history, memory, strategy, lambda_swap)

        a = min(int(np.searchsorted(np.cumsum(p), a_draws[t])), 2)
        expected_b += float((PAYOFF_B @ torch.tensor(p))[b])
        realized_b += float(PAYOFF_B[b, a])
        entropies.append(float(-np.sum(p * np.log(np.clip(p, 1e-12, 1))) / np.log(3)))
        a_actions.append(a)
        b_actions.append(b)

        # Crucial: A always updates its real weights, regardless of B strategy
        memory = update_a(model, memory, history, b, lambda_swap, learning_rate=LR)
        history = torch.cat((history[1:], torch.tensor([b], dtype=torch.long)))

    external, swap = measured_regrets(a_actions, b_actions)
    return dict(
        expected_b=expected_b / ROUNDS,
        realized_b=realized_b / ROUNDS,
        ext_a=external,
        swap_a=swap,
        a_entropy=float(np.mean(entropies))
    )


def candidate_mixtures(denominator=GRID_DENOMINATOR):
    for rock in range(denominator + 1):
        for paper in range(denominator - rock + 1):
            scissors = denominator - rock - paper
            yield (rock / denominator, paper / denominator, scissors / denominator)


def mean_metric(results, key):
    return float(np.mean([r[key] for r in results]))


def online_transformer_rps_stationary_benchmark():
    print('Stationary mixed commitments vs learning-aware RPS planning')
    print(f'{len(TUNE_SEEDS)} tuning seeds, {len(TEST_SEEDS)} held-out seeds; '
          f'{ROUNDS} rounds/episode, LR={LR}')
    print('Mixture search grid: increments of 0.25; stationary B samples iid each round.')

    for lam in [0.0, 1.0]:
        candidates = list(candidate_mixtures())
        selection_scores = []
        for q in candidates:
            scores = [run('mixed', lam, s, mix=q)['expected_b'] for s in TUNE_SEEDS]
            selection_scores.append(float(np.mean(scores)))
        best_idx = int(np.argmax(selection_scores))
        q_star = candidates[best_idx]

        print(f'\n=== Actual A lambda={lam} ===')
        print(f'Best mixture on tuning seeds: {dict(zip(ACTIONS, q_star))}')
        print(f'Tuning expected B reward: {selection_scores[best_idx]:.3f}')

        competitors = {
            'uniform_mix': ('mixed', (1/3, 1/3, 1/3)),
            'best_grid_mix': ('mixed', q_star),
            'myopic': ('myopic', None),
        }
        print(f'{"B strategy":<16} {"B expected":>11} {"B realized":>11} '
              f'{"A ext":>9} {"A swap":>9} {"A entropy":>10}')

        results = {}
        for label, (strategy, mix) in competitors.items():
            records = [run(strategy, lam, seed, mix=mix) for seed in TEST_SEEDS]
            results[label] = records
            print(
                f'{label:<16} {mean_metric(records, "expected_b"):11.3f} '
                f'{mean_metric(records, "realized_b"):11.3f} '
                f'{mean_metric(records, "ext_a"):9.3f} '
                f'{mean_metric(records, "swap_a"):9.3f} '
                f'{mean_metric(records, "a_entropy"):10.3f}'
            )

        # Use previous exact Planner-3 results on the same seeds as a reference.
        old_planner = 0.306 if lam == 0.0 else 0.519
        print(f'Previous Planner-3 B expected on these same five seeds: {old_planner:.3f}')
        print(f'Planner-3 minus held-out best-grid mixture: '
              f'{old_planner - mean_metric(results["best_grid_mix"], "expected_b"):+.3f}')

    print('\nNote: best_grid_mix is optimal only among 15 candidate distributions '
          'on the tuning seeds, NOT globally optimal stationary play.')
    print('The paired differences are descriptive; short episodes and few seeds '
          'do not establish statistical significance.')
