"""Step 1: controlled, matched-prefix LLM RPS comparisons.

Local Ollama only, no model downloads or fine-tuning.

Quick smoke test:
    MODELS=llama3.1:8b PREFIXES=1 ROUNDS=8 python3 step1_controlled_ollama_rps.py

Both models:
    MODELS=llama3.1:8b,qwen3.5:latest PREFIXES=4 ROUNDS=18 python3 step1_controlled_ollama_rps.py

Requirements: numpy, running local Ollama with model(s) already installed.
All B decisions use only past public actions, never A's current action.
"""
import csv
import json
import math
import os
import re
from pathlib import Path
from urllib import request

import numpy as np

ACTIONS = ('Rock', 'Paper', 'Scissors')
PAYOFF_A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]], dtype=int)
MODELS = [s.strip() for s in os.getenv('MODELS', 'llama3.1:8b,qwen3.5:latest').split(',') if s.strip()]
PREFIXES = int(os.getenv('PREFIXES', '2'))
ROUNDS = int(os.getenv('ROUNDS', '12'))
WARMUP = int(os.getenv('WARMUP', '3'))
OLLAMA_URL = os.getenv('OLLAMA_URL', 'http://localhost:11434/api/chat')
OUTPUT = Path("outputs", 'ollama_rps_results')
STRATEGIES = ('uniform', 'always_scissors', 'bait2_scissors', 'bait4_scissors', 'adaptive')
HISTORY_WINDOW = 8


def make_messages(history, window=HISTORY_WINDOW):
    lines = [
        f'You: {ACTIONS[a]}, Opponent: {ACTIONS[b]}, Your reward: {r:+d}'
        for a, b, r in (history[-window:] if window else ())
    ]
    previous = '\n'.join(lines) if lines else 'No prior rounds shown.'
    return [
        {'role': 'system', 'content': (
            'You are player A in repeated simultaneous Rock-Paper-Scissors. '
            'Rock beats Scissors; Paper beats Rock; Scissors beats Paper. '
            'Win=+1, draw=0, loss=-1. Your goal is to maximize total reward. '
            'Only earlier completed rounds are visible. The current opponent action is hidden. '
            'Output exactly one word: Rock, Paper, or Scissors.'
        )},
        {'role': 'user', 'content':
            f'Previous rounds (oldest to newest):\n{previous}\n\n'
            'The opponent has not revealed its next move. Your action (one word only):'
        },
    ]


def ask_llm(model, history):
    payload = {
        'model': model,
        'messages': make_messages(history),
        'stream': False,
        'think': False,
        'options': {'temperature': 0, 'num_predict': 24, 'seed': 42},
        'keep_alive': '10m',
    }
    req = request.Request(
        OLLAMA_URL, data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}, method='POST'
    )
    try:
        with request.urlopen(req, timeout=180) as response:
            data = json.load(response)
    except Exception as exc:
        raise RuntimeError(
            f'Ollama request failed (model={model}, url={OLLAMA_URL}). '
            'Check `ollama ls` and that the server is running.'
        ) from exc
    raw = data.get('message', {}).get('content', '').strip()
    return parse_action(raw)


def parse_action(raw):
    words = re.findall(r'\b(rock|paper|scissors)\b', raw, flags=re.IGNORECASE)
    # Strict validation: accept a single RPS action; never silently read a multi-action list.
    if len(words) == 1:
        return ACTIONS.index(words[0].capitalize()), raw, False
    # Explicitly mark invalid instead of counting it as good model behaviour.
    return 0, raw, True


def choose_b(strategy, history, eval_round, rng):
    if strategy == 'uniform':
        return int(rng.integers(3)), 'uniform'
    if strategy == 'always_scissors':
        return 2, 'exploit'
    if strategy.startswith('bait'):
        n = 2 if strategy.startswith('bait2') else 4
        return (0, 'bait') if eval_round < n else (2, 'exploit')
    if strategy == 'adaptive':
        counts = np.ones(3)
        for a, _, _ in history[-HISTORY_WINDOW:]:
            counts[a] += 1
        p = counts / counts.sum()
        return int((-(p @ PAYOFF_A)).argmax()), 'adaptive'
    raise ValueError(strategy)


def warm_up(model, prefix_id):
    # Distinct initial public histories, reused across every strategy for this model/prefix.
    rng = np.random.default_rng(14003 + prefix_id * 101)
    history, rows = [], []
    for t in range(WARMUP):
        b = int(rng.integers(3))
        a, raw, invalid = ask_llm(model, history)
        r = int(PAYOFF_A[a, b])
        history.append((a, b, r))
        rows.append({'round': t + 1, 'a': ACTIONS[a], 'b': ACTIONS[b],
                     'reward_b': -r, 'invalid': invalid, 'raw': raw})
    return history, rows


def run(model, prefix_id, strategy, initial_history):
    history = list(initial_history)
    rng = np.random.default_rng(8021 + prefix_id * 101)
    trace = []
    for t in range(ROUNDS):
        b, phase = choose_b(strategy, history, t, rng)
        a, raw, invalid = ask_llm(model, history)
        reward_a = int(PAYOFF_A[a, b])
        history.append((a, b, reward_a))
        trace.append({
            'model': model, 'prefix': prefix_id, 'strategy': strategy,
            'round': t + 1, 'phase': phase, 'a': ACTIONS[a], 'b': ACTIONS[b],
            'reward_b': -reward_a, 'invalid': int(invalid), 'raw': raw,
        })
    a = [ACTIONS.index(x['a']) for x in trace]
    p = np.bincount(a, minlength=3) / ROUNDS
    entropy = -sum(float(q * math.log(q)) for q in p if q > 0) / math.log(3)
    # Common comparison window after all fixed bait schedules could finish.
    paper_post = float(np.mean(np.array(a[4:]) == 1))
    return trace, {
        'model': model, 'prefix': prefix_id, 'strategy': strategy,
        'b_reward': float(np.mean([x['reward_b'] for x in trace])),
        'paper_post': paper_post, 'entropy': entropy,
        'invalid': sum(x['invalid'] for x in trace),
    }


def save_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)



def rps_controlled_ollama():
    if PREFIXES < 1 or ROUNDS < 6 or WARMUP < 1:
        raise ValueError('Require PREFIXES >=1, ROUNDS >=6, WARMUP >=1')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_rounds, all_episodes = [], []
    n_calls = len(MODELS) * PREFIXES * (WARMUP + len(STRATEGIES) * ROUNDS)
    print(f'Models: {MODELS}; prefixes={PREFIXES}, warmup={WARMUP}, rounds={ROUNDS}')
    print(f'Ollama calls planned: {n_calls}. Temperature=0, think=False.')
    for model in MODELS:
        print(f'\n=== MODEL: {model} ===', flush=True)
        for prefix in range(PREFIXES):
            initial, warmup_rows = warm_up(model, prefix)
            if any(x['invalid'] for x in warmup_rows):
                print(f'WARNING: invalid warmup response for prefix {prefix}.')
            print('Shared warmup', prefix, [(x['a'], x['b']) for x in warmup_rows], flush=True)
            for strategy in STRATEGIES:
                rounds, result = run(model, prefix, strategy, initial)
                all_rounds.extend(rounds)
                all_episodes.append(result)
                print(f'  {strategy:<19} reward={result["b_reward"]:+.3f} '
                      f'Paper_post={result["paper_post"]:.3f} '
                      f'invalid={result["invalid"]}', flush=True)
            save_csv(OUTPUT / 'rounds.csv', all_rounds)
            save_csv(OUTPUT / 'episodes.csv', all_episodes)
        print(f'\nSummary for {model}:')
        for strategy in STRATEGIES:
            subset = [r for r in all_episodes if r['model'] == model and r['strategy'] == strategy]
            print(f'  {strategy:<19} reward={np.mean([r["b_reward"] for r in subset]):+.3f} '
                  f'Paper_post={np.mean([r["paper_post"] for r in subset]):.3f}')
        ref = {r['prefix']: r for r in all_episodes if r['model'] == model and r['strategy'] == 'always_scissors'}
        for strategy in ('bait2_scissors', 'bait4_scissors', 'adaptive'):
            rows = [r for r in all_episodes if r['model'] == model and r['strategy'] == strategy]
            dr = np.mean([r['b_reward'] - ref[r['prefix']]['b_reward'] for r in rows])
            dp = np.mean([r['paper_post'] - ref[r['prefix']]['paper_post'] for r in rows])
            print(f'  Paired {strategy} vs always_scissors: '
                  f'delta_B_reward={dr:+.3f}, delta_Paper_post={dp:+.3f}')
    print(f'\nSaved: {OUTPUT / "rounds.csv"} and {OUTPUT / "episodes.csv"}')
    print('Matched deterministic prefixes are not independent random samples of LLM behaviour.')
    print('Bait causal claims require replication across varied prefixes/prompts and controls.')

