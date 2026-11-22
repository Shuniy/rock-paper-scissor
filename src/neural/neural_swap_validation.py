"""Step 2: paired-seed evaluation of reward vs swap-regularized online transformer.

Smoke test:
    SEEDS=2 STRATEGIES=uniform,myopic python3 -m src.neural.neural_swap_validation

Main test:
    SEEDS=20 STRATEGIES=uniform,myopic,plan2 python3 -m src.neural.neural_swap_validation

Metrics are per-round averages over each full episode. Bootstrap CIs use
seed pairs, not individual game rounds, as independent observational units.
"""
import csv
import os
from pathlib import Path

import numpy as np
import src.neural.online_transformer_rps_model_mismatch as core

SEEDS = int(os.getenv('SEEDS', '10'))
BASE_SEED = int(os.getenv('BASE_SEED', '1001'))
STRATEGIES = [x.strip() for x in os.getenv('STRATEGIES', 'uniform,myopic,plan2').split(',') if x.strip()]
OUTPUT = Path("outputs", "neural_swap_results")
BOOTSTRAPS = 3000


def interval(values, rng):
    """Percentile bootstrap interval for mean over independent run seeds."""
    x = np.asarray(values, dtype=float)
    samples = x[rng.integers(0, len(x), size=(BOOTSTRAPS, len(x)))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def fmt(metric, rows):
    return float(np.mean([r[metric] for r in rows]))

def neural_swap_validation():
    if SEEDS < 2:
        raise ValueError('Use at least two seeds. Prefer >=20 for interpretation.')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results = []
    seeds = list(range(BASE_SEED, BASE_SEED + SEEDS))
    for strategy in STRATEGIES:
        if strategy not in ('uniform', 'myopic', 'plan1', 'plan2', 'plan3'):
            raise ValueError(f'Unsupported strategy: {strategy}')
        for seed in seeds:
            for lam in (0.0, 1.0):
                r = core.run(strategy, lam, seed)
                results.append({'strategy': strategy, 'seed': seed, 'lambda_swap': lam,
                                **{k: float(v) for k, v in r.items()}})
        print(f'Finished {strategy} ({SEEDS} matched initialization seeds)', flush=True)

    with (OUTPUT / 'runs.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    rng = np.random.default_rng(54321)
    for strategy in STRATEGIES:
        print(f'\n=== B STRATEGY: {strategy} ===')
        by = {
            lam: {r['seed']: r for r in results if r['strategy'] == strategy and r['lambda_swap'] == lam}
            for lam in (0.0, 1.0)
        }
        print(f'{"A learning":<17} {"B expected":>11} {"B realized":>11} {"A external":>11} {"A swap":>9} {"A entropy":>10}')
        for lam in (0.0, 1.0):
            rs = list(by[lam].values())
            print(f'{"Reward" if lam==0 else "Reward+swap":<17} '
                  f'{fmt("expected_b",rs):11.3f} {fmt("realized_b",rs):11.3f} '
                  f'{fmt("ext_a",rs):11.3f} {fmt("swap_a",rs):9.3f} '
                  f'{fmt("a_entropy",rs):10.3f}')
        print('Paired lambda=1 minus lambda=0 (95% seed bootstrap CI):')
        for metric in ('expected_b', 'swap_a', 'a_entropy'):
            differences = [by[1.0][s][metric] - by[0.0][s][metric] for s in seeds]
            lo, hi = interval(differences, rng)
            print(f'  {metric:<12}: {np.mean(differences):+.3f}  [{lo:+.3f}, {hi:+.3f}]')

    if 'myopic' in STRATEGIES:
        for comparator in ('plan2', 'plan3'):
            if comparator not in STRATEGIES:
                continue
            print(f'\n=== Planner advantage: {comparator} minus myopic (B expected reward) ===')
            for lam in (0.0, 1.0):
                def lookup(strategy, seed):
                    return next(r for r in results if r['strategy'] == strategy and r['seed'] == seed and r['lambda_swap'] == lam)
                differences = [lookup(comparator, s)['expected_b'] - lookup('myopic', s)['expected_b'] for s in seeds]
                lo, hi = interval(differences, rng)
                print(f'  lambda={lam:.0f}: {np.mean(differences):+.3f} [{lo:+.3f}, {hi:+.3f}]')
    print(f'\nIndividual runs saved at: {OUTPUT / "runs.csv"}')
    print('The lambda treatments change B and A trajectories; pairing matches initial seeds, not opponent histories.')
    print('Swap loss is a short-memory surrogate, NOT a no-swap-regret guarantee.')
    print('Regret here uses fixed recorded opponent actions (not causal policy regret).')


if __name__ == "__main__":
    neural_swap_validation()
