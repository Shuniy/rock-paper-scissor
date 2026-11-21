"""Controlled RPS baiting ablation against a locally served Ollama LLM.

Run beside rps_ollama_baseline.py:
  OLLAMA_MODEL=llama3.2 SEEDS=3 ROUNDS=18 WARMUP=2 python3 rps_ollama_bait_causality.py

All strategies use the same seed-specific uniform warmup to start each comparison.
After that, compare constant Scissors vs Rock-bait-then-Scissors, plus the
previous observed-action adaptive heuristic. No strategy observes A's current
move, hidden policy, or neural parameters. The LLM is NOT weight-updated.
"""
import math
import os
import numpy as np
import src.llm.rps_ollama_baseline as base

SEEDS = int(os.getenv("SEEDS", "3"))
ROUNDS = int(os.getenv("ROUNDS", "18"))
WARMUP = int(os.getenv("WARMUP", "2"))
STRATEGIES = ("always_scissors", "bait2_then_scissors", "bait4_then_scissors", "adaptive")


def choose_b(strategy, history, trial_round, rng):
    if strategy == "always_scissors":
        return 2, "exploit"
    if strategy.startswith("bait"):
        bait_rounds = 2 if strategy.startswith("bait2") else 4
        return (0, "bait") if trial_round < bait_rounds else (2, "exploit")
    if strategy == "adaptive":
        return base.choose_b(history, "adaptive", rng), "adaptive"
    raise ValueError(strategy)


def run_episode(strategy, seed, verbose=False):
    rng = np.random.default_rng(seed)
    history = []
    invalid_count = 0
    eval_actions_a = []
    eval_rewards_b = []
    post_actions_a = []
    bait_rewards = []
    exploit_rewards = []

    for t in range(ROUNDS + WARMUP):
        if t < WARMUP:
            b = int(rng.integers(3))
            phase = "shared_warmup"
        else:
            b, phase = choose_b(strategy, history, t - WARMUP, rng)

        a, _raw, invalid = base.choose_a(history, use_history=True, rng=rng)
        invalid_count += int(invalid)
        reward_a = int(base.PAYOFF_A[a, b])
        reward_b = -reward_a
        history.append((a, b, reward_a))

        if t >= WARMUP:
            eval_actions_a.append(a)
            eval_rewards_b.append(reward_b)
            if phase == "bait":
                bait_rewards.append(reward_b)
            if phase == "exploit":
                exploit_rewards.append(reward_b)
            # A common late window for all strategies, so bait4 has completed.
            if t >= WARMUP + 4:
                post_actions_a.append(a)

        if verbose and t < WARMUP + 8:
            print(f"  {t+1:2d}  B={base.ACTIONS[b]:8} A={base.ACTIONS[a]:8} "
                  f"B reward={reward_b:+d} ({phase})")

    counts = np.bincount(eval_actions_a, minlength=3)
    p = counts / len(eval_actions_a)
    entropy = -sum(float(q) * math.log(float(q)) for q in p if q > 0) / math.log(3)

    return {
        "reward": float(np.mean(eval_rewards_b)),
        "post_paper_rate": float(np.mean(np.array(post_actions_a) == 1)),
        "entropy": entropy,
        "invalid": invalid_count,
        "counts": counts,
        "bait_n": len(bait_rewards),
        "bait_reward": sum(bait_rewards),
        "exploit_n": len(exploit_rewards),
        "exploit_reward": sum(exploit_rewards),
    }


def rps_ollama_bait_causality():
    if SEEDS < 1 or ROUNDS < 5 or WARMUP < 0:
        raise ValueError("Use SEEDS >= 1, ROUNDS >= 5 and WARMUP >= 0")
    print(f"Ollama model: {base.MODEL_ID}; SEEDS={SEEDS}; evaluation ROUNDS={ROUNDS}; shared WARMUP={WARMUP}")
    print("Post-bait Paper rate uses evaluation rounds 5 onward for every strategy.")
    print("NOTE: zero-temperature deterministic LLM output can make seeds highly correlated.\n")
    print(f"{'Strategy':24} {'B reward':>9} {'A Paper post':>13} {'A entropy':>10} {'invalid':>9}")

    for strategy in STRATEGIES:
        results = [run_episode(strategy, 42 + i, verbose=(i == 0)) for i in range(SEEDS)]
        print(f"{strategy:24} {np.mean([x['reward'] for x in results]):+9.3f} "
              f"{np.mean([x['post_paper_rate'] for x in results]):13.3f} "
              f"{np.mean([x['entropy'] for x in results]):10.3f} "
              f"{sum(x['invalid'] for x in results):4d}/{SEEDS*(ROUNDS+WARMUP)}")
        if strategy.startswith("bait"):
            print("    Bait: rounds={}, reward={}; exploit: rounds={}, reward={}".format(
                sum(x['bait_n'] for x in results),
                sum(x['bait_reward'] for x in results),
                sum(x['exploit_n'] for x in results),
                sum(x['exploit_reward'] for x in results),
            ))
        print("    First-seed A counts:", dict(zip(base.ACTIONS, results[0]['counts'].tolist())))

