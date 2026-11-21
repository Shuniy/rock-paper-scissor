"""Black-box learning-dynamics model for an online RPS Transformer.

Run beside online_transformer_rps_model_mismatch.py. B never receives the real
A model, weights, gradients, learning rate, objective, or probability vector.
B trains a GRU on past observed (B action, A sampled action) interactions.
Evaluation uses fresh Agent-A seeds that were not used for model fitting.
A future A action in a plan is approximated by the surrogate's most likely
sampled outcome: this is an approximation, not exact belief-state planning.
"""
from itertools import product
import numpy as np
import torch
from torch import nn
from src.neural.online_transformer_rps_model_mismatch import (
    HISTORY, LR, OnlineTransformer, PAYOFF_B, update_a,
)

torch.set_num_threads(1)
PAYOFF_B_NP = PAYOFF_B.numpy()
NUM_TRAIN_EPISODES = 64     # Raise later to 200+ for broader coverage.
TRAIN_ROUNDS = 25
EVAL_ROUNDS = 25
EVAL_SEEDS = range(900, 905)
TRAIN_EPOCHS = 25


class BehaviourGRU(nn.Module):
    """Sequence-to-distribution emulator of A using observable action pairs."""

    def __init__(self):
        super().__init__()
        # Tokens 0..8 encode past pairs (B,A); 9..11 encode pre-game B history.
        self.embed = nn.Embedding(12, 16)
        self.gru = nn.GRU(16, 32, batch_first=True)
        self.head = nn.Linear(32, 3)

    def forward(self, observed_tokens):
        hidden, _ = self.gru(self.embed(observed_tokens))
        return self.head(hidden)


# A and B are simultaneously selected, so the current B action never appears
# in the input for predicting the current A action.
def episode_tokens(initial_history, b_actions, a_actions):
    assert len(b_actions) == len(a_actions)
    prefix = [9 + int(b) for b in initial_history]
    past_pairs = [3 * int(b) + int(a)
                  for b, a in zip(b_actions[:-1], a_actions[:-1])]
    return prefix + past_pairs


def collect_training_episode(seed, lam, rounds=TRAIN_ROUNDS):
    # The data collection process is black-box: only sampled A actions are saved.
    torch.manual_seed(seed)
    a_model = OnlineTransformer().train()
    rng = np.random.default_rng(seed + 100000)
    init_history = rng.integers(0, 3, size=HISTORY).tolist()
    b_history = init_history.copy()
    memory = []
    a_actions, b_actions = [], []
    mode = seed % 5
    fixed_b = int(rng.integers(0, 3))
    for t in range(rounds):
        with torch.no_grad():
            true_p = a_model(torch.tensor([b_history], dtype=torch.long))[0].numpy().copy()

        # Diverse exploration policies, all chosen without viewing true_p.
        if mode == 0:
            b = int(rng.integers(0, 3))
        elif mode == 1:
            b = fixed_b if rng.random() > 0.15 else int(rng.integers(0, 3))
        elif mode == 2:
            b = (fixed_b + t // 8) % 3
        elif mode == 3:
            b = (fixed_b + t) % 3
        else:
            counts = np.bincount(a_actions[-8:], minlength=3).astype(float)
            estimate = (counts + 1.0) / (counts.sum() + 3.0)
            b = int(np.argmax(PAYOFF_B_NP @ estimate))
            if rng.random() < 0.2:
                b = int(rng.integers(0, 3))

        a = int(rng.choice(3, p=true_p))
        a_actions.append(a)
        b_actions.append(b)
        memory = update_a(
            a_model, memory, torch.tensor(b_history, dtype=torch.long),
            b, lam, learning_rate=LR
        )
        b_history = (b_history + [b])[-HISTORY:]

    tokens = episode_tokens(init_history, b_actions, a_actions)
    return torch.tensor(tokens, dtype=torch.long), torch.tensor(a_actions, dtype=torch.long)


def fit_surrogate():
    print('Collecting black-box training trajectories...')
    trajectories = []
    for i in range(NUM_TRAIN_EPISODES):
        lam = (0.0, 1.0)[i % 2]
        trajectories.append(collect_training_episode(10000 + i, lam))

    # Training across BOTH objectives: B is not given A's true lambda.
    torch.manual_seed(555)
    emulator = BehaviourGRU()
    optim = torch.optim.Adam(emulator.parameters(), lr=0.003)
    for epoch in range(1, TRAIN_EPOCHS + 1):
        permutation = torch.randperm(len(trajectories)).tolist()
        total_loss = 0.0
        for start in range(0, len(trajectories), 24):
            batch = [trajectories[j] for j in permutation[start:start + 24]]
            x = torch.stack([e[0] for e in batch])  # [batch, H+T-1]
            targets = torch.stack([e[1] for e in batch])
            logits = emulator(x)[:, HISTORY - 1:, :]  # T next-A predictions
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, 3), targets.reshape(-1)
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_loss += loss.detach().item() * len(batch)
        if epoch in (1, 10, 20, TRAIN_EPOCHS):
            print(f'Epoch {epoch:2d}: observed-action NLL={total_loss / len(trajectories):.3f}')
    emulator.eval()
    return emulator


@torch.no_grad()
def predict_a(emulator, tokens):
    x = torch.tensor(tokens, dtype=torch.long).unsqueeze(0)
    return emulator(x)[0, -1].softmax(-1).numpy()


def choose_b_blackbox(emulator, tokens, strategy):
    if strategy == 'learned_myopic':
        return int(np.argmax(PAYOFF_B_NP @ predict_a(emulator, tokens)))

    horizon = {'learned_plan2': 2, 'learned_plan3': 3}[strategy]
    best_score = -float('inf')
    best_first = 0
    for plan in product(range(3), repeat=horizon):
        hypothetical = tokens.copy()
        total = 0.0
        for b in plan:
            p = predict_a(emulator, hypothetical)
            total += float(PAYOFF_B_NP[b] @ p)
            # Approximate an unobserved future A action. True rollout requires
            # marginalizing over its 3 possible outcomes at every step.
            a_guess = int(np.argmax(p))
            hypothetical.append(3 * b + a_guess)
        if total > best_score:
            best_score = total
            best_first = plan[0]
    return int(best_first)


def evaluate_episode(emulator, lam, seed, strategy):
    torch.manual_seed(seed)
    a_model = OnlineTransformer().train()
    rng = np.random.default_rng(seed + 100000)
    initial_history = rng.integers(0, 3, size=HISTORY).tolist()
    history = initial_history.copy()
    tokens = [9 + b for b in initial_history]
    memory = []
    a_draws = rng.random(EVAL_ROUNDS)
    b_draws = rng.integers(0, 3, size=EVAL_ROUNDS)
    total_expected = 0.0
    total_realized = 0.0
    total_tv = 0.0

    for t in range(EVAL_ROUNDS):
        with torch.no_grad():
            true_p = a_model(torch.tensor([history], dtype=torch.long))[0].numpy().copy()
        estimated_p = predict_a(emulator, tokens)

        if strategy == 'uniform':
            b = int(b_draws[t])
        elif strategy == 'oracle_myopic':
            # Evaluator-only upper-information benchmark, NOT black-box B.
            b = int(np.argmax(PAYOFF_B_NP @ true_p))
        else:
            b = choose_b_blackbox(emulator, tokens, strategy)

        a = min(int(np.searchsorted(np.cumsum(true_p), a_draws[t])), 2)
        total_expected += float(PAYOFF_B_NP[b] @ true_p)
        total_realized += float(PAYOFF_B_NP[b, a])
        total_tv += 0.5 * float(np.abs(true_p - estimated_p).sum())

        # B only adds its observed pair to the surrogate's context.
        tokens.append(3 * b + a)
        memory = update_a(
            a_model, memory, torch.tensor(history, dtype=torch.long),
            b, lam, learning_rate=LR
        )
        history = (history + [b])[-HISTORY:]

    return np.array([total_expected, total_realized, total_tv]) / EVAL_ROUNDS


def rps_blackbox_dynamics():
    emulator = fit_surrogate()
    strategies = [
        'uniform', 'learned_myopic', 'learned_plan2',
        'learned_plan3', 'oracle_myopic'
    ]
    print('\nTRAINED SURROGATE: action-pair GRU; no real A probabilities used for fitting.')
    print('Planner rollouts use MAP future A actions, not exact stochastic expectation.')
    for lam in (0.0, 1.0):
        print(f'\nActual A lambda={lam}; {len(EVAL_SEEDS)} held-out seeds; '
              f'{EVAL_ROUNDS} rounds')
        print(f'{"B strategy":<18} {"B expected":>10} {"B realized":>12} {"pred TV":>10}')
        for strategy in strategies:
            result = np.mean([evaluate_episode(emulator, lam, s, strategy)
                              for s in EVAL_SEEDS], axis=0)
            print(f'{strategy:<18} {result[0]:10.3f} {result[1]:12.3f} {result[2]:10.3f}')
