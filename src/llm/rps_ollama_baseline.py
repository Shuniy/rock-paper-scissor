"""RPS experiment using an already installed Ollama LLM (no HF download).

Usage:
    ollama ls
    OLLAMA_MODEL=llama3.2:3b python3 rps_ollama_baseline.py

Use a model name from `ollama ls`. Ollama must be running locally.
"""
import json
import os
import re
from urllib import request

import numpy as np

MODEL_ID = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat")
ACTIONS = ("Rock", "Paper", "Scissors")
PAYOFF_A = np.array([
    [0, -1, 1],
    [1, 0, -1],
    [-1, 1, 0],
], dtype=int)
ROUNDS = 12
HISTORY_WINDOW = 8
SEED = 42


def choose_b(history, strategy, rng):
    """B acts on public past actions, never on A's current hidden move."""
    if strategy == "uniform" or not history:
        return int(rng.integers(3))
    if strategy != "adaptive":
        raise ValueError(f"Unknown strategy: {strategy}")

    counts = np.ones(3, dtype=float)  # smoothing for small samples
    for a, _, _ in history[-HISTORY_WINDOW:]:
        counts[a] += 1
    p_a = counts / counts.sum()
    # A rows, B columns: negative of A's expected payoff
    b_expected = -(p_a @ PAYOFF_A)
    return int(b_expected.argmax())


def build_messages(history, use_history):
    if use_history and history:
        lines = []
        for a, b, reward_a in history[-HISTORY_WINDOW:]:
            lines.append(
                f"You: {ACTIONS[a]}, Opponent: {ACTIONS[b]}, "
                f"Your reward: {reward_a:+d}"
            )
        previous = "\n".join(lines)
    else:
        previous = "No prior rounds shown."

    return [
        {
            "role": "system",
            "content": (
                "You are player A in repeated simultaneous Rock-Paper-Scissors. "
                "Rock beats Scissors; Paper beats Rock; Scissors beats Paper. "
                "Win=+1, draw=0, loss=-1. Your objective is to maximize "
                "your total score. Choose your next action based only on the "
                "public history. Output exactly one word: Rock, Paper, or Scissors."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Previous rounds (oldest to newest):\n{previous}\n\n"
                "The opponent's current move is hidden. "
                "Your next move (one word only):"
            ),
        },
    ]


def choose_a(history, use_history, rng):
    messages = build_messages(history, use_history)
    payload = {
        "model": MODEL_ID,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 12},
        "keep_alive": "10m",
    }
    req = request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=120) as response:
            data = json.load(response)
        text = data["message"]["content"].strip()
    except Exception as exc:
        raise RuntimeError(
            f"Could not call Ollama at {OLLAMA_URL} with model {MODEL_ID!r}. "
            "Check that Ollama is running and that this model appears in `ollama ls`."
        ) from exc

    match = re.search(r"\b(rock|paper|scissors)\b", text, re.I)
    if match:
        return ACTIONS.index(match.group(1).capitalize()), text, False
    return int(rng.integers(3)), text, True


def play(strategy, use_history, seed=SEED):
    rng = np.random.default_rng(seed)
    history = []
    invalid = 0
    for round_idx in range(ROUNDS):
        # Both players choose based on PAST public moves only.
        b = choose_b(history, strategy, rng)
        a, raw, was_invalid = choose_a(history, use_history, rng)
        invalid += int(was_invalid)
        reward_a = int(PAYOFF_A[a, b])
        history.append((a, b, reward_a))
        print(
            f"{round_idx + 1:2d} | "
            f"A={ACTIONS[a]:8s} B={ACTIONS[b]:8s} "
            f"A reward={reward_a:+d}"
            + (f" [invalid generated text: {raw!r}]" if was_invalid else "")
        )

    reward_a_total = sum(r for _, _, r in history)
    reward_b_total = -reward_a_total
    print(
        f"RESULT: B mean realized reward={reward_b_total / ROUNDS:+.3f}; "
        f"A mean realized reward={reward_a_total / ROUNDS:+.3f}; "
        f"invalid LLM outputs={invalid}/{ROUNDS}\n"
    )



def rps_pretrained_llm_baseline():
    print(f"Using Ollama model: {MODEL_ID} via {OLLAMA_URL}")

    experiments = [
        ("uniform", True),
        ("adaptive", True),
        ("adaptive", False),
    ]
    for strategy, use_history in experiments:
        print(
            f"\n=== B={strategy}; A sees history={use_history}; "
            f"seed={SEED} ==="
        )
        play(strategy, use_history)
