"""Measure neural weight-update effect, holding A's next input history fixed.

Place next to online_transformer_rps_model_mismatch.py.
Run: python3 rps_learning_update_ablation.py
"""
import numpy as np
import torch

from src.neural.online_transformer_rps_model_mismatch import (
    HISTORY, ROUNDS, PAYOFF_B, OnlineTransformer,
    b_expected_rewards, choose_b, update_a,
)


def run_ablation(lam, strategy, seed=42):
    torch.manual_seed(seed)
    a = OnlineTransformer()
    a.train()
    rng = np.random.default_rng(seed)
    history = torch.tensor(rng.integers(0, 3, size=HISTORY), dtype=torch.long)
    memory = []
    changes, shifts, records = 0, [], []

    for step in range(ROUNDS):
        # Choose B's move from A's current learned state.
        b = choose_b(a, history, memory, strategy, lam)
        next_history = torch.cat([history[1:], torch.tensor([b])])

        # Counterfactual: same next history, but do NOT change neural weights.
        with torch.no_grad():
            p_frozen = a(next_history.unsqueeze(0))[0].detach().clone()

        # Actual online gradient update after observing B's move.
        memory = update_a(a, memory, history, b, lam)
        with torch.no_grad():
            p_updated = a(next_history.unsqueeze(0))[0].detach().clone()

        # Total variation distance: how much just the weight update changed P(A).
        tv = float((p_updated - p_frozen).abs().sum() / 2)
        shifts.append(tv)

        # Would B's next best response differ because of A's weight update?
        best_frozen = int((PAYOFF_B @ p_frozen).argmax())
        best_updated = int((PAYOFF_B @ p_updated).argmax())
        changes += (best_frozen != best_updated)
        if step < 4:
            records.append((step + 1, b, p_frozen.numpy(), p_updated.numpy(), tv))
        history = next_history

    return float(np.mean(shifts)), changes / ROUNDS, records


def rps_learning_update_ablation():
    for lam in [0.0, 1.0]:
        print(f'\nA objective lambda={lam}')
        for strategy in ['myopic', 'plan3']:
            avg, disagreement, examples = run_ablation(lam, strategy)
            print(f'  B={strategy}: average policy shift from weight update={avg:.3f}; '
                  f'next-best-response changes={disagreement:.0%}')
            if strategy == 'plan3':
                for step, b, frozen, updated, tv in examples:
                    print(f'    round {step}: B={b}, history-only={frozen.round(3)}, '
                          f'after gradient={updated.round(3)}, shift={tv:.3f}')
    print('\nThese are counterfactual differences at the same next history, '
          'not evidence of a planner advantage or of statistical significance.')
