"""Compare fixed vs observation-triggered bait/switch against an Ollama RPS LLM.

Run next to rps_ollama_baseline.py:
    OLLAMA_MODEL=llama3.2:3b SEEDS=3 ROUNDS=18 python3 rps_ollama_adaptive_bait.py

B sees only *past* A and B actions, never A's current choice, weights, or token probabilities.
This is a causal heuristic, NOT a trained learning-aware planner.
"""
import math
import os
import numpy as np
import src.llm.rps_ollama_baseline as base

SEEDS = int(os.environ.get("SEEDS", "3"))
ROUNDS = int(os.environ.get("ROUNDS", "18"))
BAIT_MAX = int(os.environ.get("BAIT_MAX", "4"))


class ReactiveBaitSwitch:
    """Bait with Rock, exploit with Scissors once A has shown a Paper habit.

    During exploitation, if fewer than 2 of the last 3 A actions are Paper,
    return to baiting. B only uses observed actions from completed rounds.
    """
    def __init__(self):
        self.phase = "bait"
        self.bait_count = 0

    def select(self, history):
        a_recent = [a for a, _, _ in history[-3:]]
        two_recent_papers = (
            len(history) >= 2 and
            history[-1][0] == 1 and history[-2][0] == 1
        )

        if self.phase == "bait" and (two_recent_papers or self.bait_count >= BAIT_MAX):
            self.phase = "exploit"
        elif self.phase == "exploit" and len(a_recent) == 3 and a_recent.count(1) < 2:
            self.phase = "bait"
            self.bait_count = 0

        if self.phase == "bait":
            self.bait_count += 1
            return 0, "bait"           # Rock
        return 2, "exploit"             # Scissors


def play_episode(strategy, seed, verbose=False):
    rng = np.random.default_rng(seed)
    history = []
    reactive = ReactiveBaitSwitch()
    invalid_count = 0
    bait_rewards, exploit_rewards = [], []

    for t in range(ROUNDS):
        if strategy == "reactive_bait":
            b, phase = reactive.select(history)
        elif strategy == "fixed_bait":
            b = 0 if (t % 6) < 4 else 2
            phase = "bait" if b == 0 else "exploit"
        elif strategy == "adaptive":
            b = base.choose_b(history, "adaptive", rng)
            phase = "adaptive"
        elif strategy == "uniform":
            b = base.choose_b(history, "uniform", rng)
            phase = "uniform"
        else:
            raise ValueError(strategy)

        # A has no access to B's hidden current move.
        a, raw, invalid = base.choose_a(history, use_history=True, rng=rng)
        invalid_count += int(invalid)
        reward_a = int(base.PAYOFF_A[a, b])
        reward_b = -reward_a
        history.append((a, b, reward_a))
        if phase == "bait":
            bait_rewards.append(reward_b)
        elif phase == "exploit":
            exploit_rewards.append(reward_b)
        if verbose and t < 12:
            print(
                f"  {t + 1:2d}: B={base.ACTIONS[b]:8} "
                f"A={base.ACTIONS[a]:8} "
                f"B reward={reward_b:+d} ({phase})"
            )

    actions = np.array([a for a, _, _ in history], dtype=int)
    counts = np.bincount(actions, minlength=3)
    p = counts / ROUNDS
    entropy = -sum(float(x) * math.log(float(x)) for x in p if x > 0) / math.log(3)
    return {
        "reward": -sum(r for _, _, r in history) / ROUNDS,
        "entropy": entropy,
        "invalid": invalid_count,
        "counts": counts,
        "bait_n": len(bait_rewards),
        "bait_total": sum(bait_rewards),
        "exploit_n": len(exploit_rewards),
        "exploit_total": sum(exploit_rewards),
    }


def rps_ollama_adaptive_bait():
    if min(SEEDS, ROUNDS, BAIT_MAX) < 1:
        raise ValueError("SEEDS, ROUNDS, and BAIT_MAX must all be positive")
    print(f"Ollama model: {base.MODEL_ID}; seeds={SEEDS}, rounds={ROUNDS}")
    print("Reactive B uses observed past actions, not LLM weights or probabilities.\n")
    for strategy in ["uniform", "fixed_bait", "reactive_bait", "adaptive"]:
        print(f"\n=== {strategy} ===")
        results = [play_episode(strategy, 42 + s, verbose=(s == 0)) for s in range(SEEDS)]
        rewards = np.array([v["reward"] for v in results])
        print(f"B mean reward/round: {rewards.mean():+.3f}")
        if SEEDS > 1:
            print(f"Across-seed SD: {rewards.std(ddof=1):.3f}")
        print(f"A normalized entropy: {np.mean([v['entropy'] for v in results]):.3f}")
        print(f"Invalid outputs: {sum(v['invalid'] for v in results)}/{SEEDS*ROUNDS}")
        print(f"First-seed A counts: {dict(zip(base.ACTIONS, results[0]['counts'].tolist()))}")
        if strategy.endswith("bait"):
            total_bait_n = sum(v["bait_n"] for v in results)
            total_exploit_n = sum(v["exploit_n"] for v in results)
            print(f"B bait rounds: {total_bait_n}, total reward: {sum(v['bait_total'] for v in results):+d}")
            print(f"B exploit rounds: {total_exploit_n}, total reward: {sum(v['exploit_total'] for v in results):+d}")

