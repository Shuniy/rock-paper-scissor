"""Compare black-box RPS baselines with a deliberately simple bait/switch policy.

Run in the same directory as rps_ollama_baseline.py:
    OLLAMA_MODEL=llama3.1:8b SEEDS=3 ROUNDS=18 python3 rps_ollama_bait_switch.py

Only A's past observed moves are used to choose B's current move.
This is a deterministic bait/switch *baseline*, not an optimized planner.
"""
import os
import math
import numpy as np

import src.llm.rps_ollama_baseline as base

SEEDS = int(os.environ.get("SEEDS", "3"))
ROUNDS = int(os.environ.get("ROUNDS", "18"))

# Play Rock for four rounds, then Scissors for two, repeat.
# Rock may encourage A to choose Paper; Scissors then exploits Paper.
BAIT_ROUNDS = 4
EXPLOIT_ROUNDS = 2


def choose_b(history, strategy, rng):
    if strategy == "bait_switch":
        phase = len(history) % (BAIT_ROUNDS + EXPLOIT_ROUNDS)
        return 0 if phase < BAIT_ROUNDS else 2
    return base.choose_b(history, strategy, rng)


def run_episode(strategy, use_history, seed, verbose=False):
    rng = np.random.default_rng(seed)
    history = []
    invalid_count = 0
    for t in range(ROUNDS):
        # Both decide using only information from earlier rounds.
        b = choose_b(history, strategy, rng)
        a, raw, invalid = base.choose_a(history, use_history, rng)
        invalid_count += int(invalid)
        reward_a = int(base.PAYOFF_A[a, b])
        history.append((a, b, reward_a))
        if verbose and t < 12:
            print(f"  {t+1:2d}: A={base.ACTIONS[a]:8s} B={base.ACTIONS[b]:8s}  A reward={reward_a:+d}")
    b_rewards = [-r for _, _, r in history]
    a_counts = np.bincount([a for a, _, _ in history], minlength=3)
    p_a = a_counts / len(history)
    entropy = -sum(float(p) * math.log(float(p)) for p in p_a if p > 0) / math.log(3)
    return {
        "b_reward": float(np.mean(b_rewards)),
        "a_entropy": entropy,
        "invalid": invalid_count,
        "a_counts": a_counts,
    }


def rps_ollama_bait_switch():
    if SEEDS < 1 or ROUNDS < 1:
        raise ValueError("SEEDS and ROUNDS must be positive")
    print(f"Ollama model: {base.MODEL_ID}; {SEEDS} seeds; {ROUNDS} rounds per episode")
    print("Bait/switch means Rock x4, Scissors x2; it is NOT an optimal planner.\n")
    settings = [
        ("uniform", True),
        ("adaptive", True),
        ("bait_switch", True),
        ("adaptive", False),
    ]
    for strategy, use_history in settings:
        results = [
            run_episode(strategy, use_history, seed=42 + k, verbose=(k == 0))
            for k in range(SEEDS)
        ]
        rewards = np.array([r["b_reward"] for r in results])
        print(f"\n{strategy}, A history={use_history}: ")
        print(f"  B reward per round: {rewards.mean():+.3f} (across-seed SD={rewards.std(ddof=1):.3f})" if SEEDS > 1 else f"  B reward per round: {rewards.mean():+.3f}")
        print(f"  A normalized action entropy: {np.mean([r['a_entropy'] for r in results]):.3f}")
        print(f"  Invalid outputs: {sum(r['invalid'] for r in results)}/{SEEDS*ROUNDS}")
        print(f"  First-seed A action counts: {dict(zip(base.ACTIONS, results[0]['a_counts'].tolist()))}")

