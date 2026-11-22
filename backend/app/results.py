"""Local result storage, CSV exports, and adapters for the original research CSVs."""
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .rounds import ACTIONS, Score

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def csv_text(rows):
    if not rows:
        return ""
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict))
                         else _csv_scalar(value) for key, value in row.items()})
    return stream.getvalue()


def _csv_scalar(value):
    # Raw model output and model names may start with spreadsheet formula characters.
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def mean(items):
    return float(np.mean(items)) if items else None


def summarize(events):
    conditions = defaultdict(list)
    for event in events:
        if event["type"] == "round":
            row = event["data"]
            conditions[row["condition"]].append(row)
    summaries = []
    for name, rows in conditions.items():
        last = rows[-1]
        metric = dict(condition=name, rounds=len(rows), cumulative_a=last["cumulative_a"],
                      cumulative_b=last["cumulative_b"], average_a=last["average_a"],
                      average_b=last["average_b"], external_per_round=last["external"] / len(rows),
                      swap_per_round=last["swap"] / len(rows))
        for key in ("expected_b", "expected_a", "prediction_tv", "policy_entropy"):
            values = [row["detail"][key] for row in rows if key in row["detail"]]
            if values:
                metric[key] = mean(values)
        for key in ("prediction_accuracy", "swap_surrogate"):
            if key in last["detail"]:
                metric[key] = last["detail"][key]
        if "invalid" in last["detail"]:
            marker = last["detail"]
            metric.update(model=marker["model"], strategy=marker["strategy"])
            if marker.get("prefix") is not None:
                metric["prefix"] = marker["prefix"]
            frequencies = last["frequency"]
            metric.update(invalid=sum(row["detail"]["invalid"] for row in rows),
                          action_entropy=-sum(q * np.log(q) for q in frequencies if q) / np.log(3),
                          paper_post=mean([row["a"] == "Paper" for row in rows[4:]]),
                          bait_b=mean([row["reward_b"] for row in rows if row.get("phase") == "bait"]),
                          exploit_b=mean([row["reward_b"] for row in rows if row.get("phase") == "exploit"]))
        if "validation_seed" in last["detail"]:
            metric.update(seed=last["detail"]["validation_seed"], strategy=last["detail"]["validation_strategy"],
                          lambda_swap=last["detail"]["lambda_swap"])
        summaries.append(metric)
    return summaries


def paired_ollama(summary, reward_key):
    baseline = {(r["model"], r["prefix"]): r for r in summary if r["strategy"] == "always_scissors"}
    paired = []
    for row in summary:
        if row["strategy"] == "always_scissors":
            continue
        control = baseline.get((row["model"], row["prefix"]))
        if control is not None and row["paper_post"] is not None and control["paper_post"] is not None:
            paired.append(dict(model=row["model"], prefix=row["prefix"], strategy=row["strategy"],
                               b_reward_delta=row[reward_key] - control[reward_key],
                               paper_post_delta=row["paper_post"] - control["paper_post"]))
    return paired


def validation_pairs(summary, target_rounds):
    return paired_neural([
        dict(seed=r["seed"], strategy=r["strategy"], lambda_swap=r["lambda_swap"], expected_b=r["expected_b"],
             swap_a=r["swap_per_round"], a_entropy=r["policy_entropy"])
        for r in summary if "seed" in r and r["rounds"] == target_rounds
    ])


def paired_neural(rows):
    rng = np.random.default_rng(54321)
    summaries = []
    for strategy in dict.fromkeys(row["strategy"] for row in rows):
        grouped = {(row["seed"], row["lambda_swap"]): row for row in rows if row["strategy"] == strategy}
        seeds = sorted({seed for seed, weight in grouped if weight == 0 and (seed, 1) in grouped})
        if len(seeds) < 2:
            continue
        result = {"strategy": strategy, "pairs": len(seeds)}
        for key in ("expected_b", "swap_a", "a_entropy"):
            delta = np.array([grouped[(seed, 1)][key] - grouped[(seed, 0)][key] for seed in seeds])
            bootstrap = delta[rng.integers(len(seeds), size=(3000, len(seeds)))].mean(axis=1)
            lo, hi = np.quantile(bootstrap, [.025, .975])
            result.update({f"{key}_delta": float(delta.mean()), f"{key}_low": float(lo), f"{key}_high": float(hi)})
        summaries.append(result)
    return summaries


def historical(dataset):
    if dataset == "neural-validation":
        source = "neural_swap_results/runs.csv"
        rows = [{key: value if key == "strategy" else int(value) if key == "seed" else float(value)
                 for key, value in row.items()} for row in read_csv(OUTPUTS / source)]
        return {"id": dataset, "sources": [source], "aggregate_only": True, "rows": rows,
                "paired": paired_neural(rows)}
    sources = ["ollama_rps_results/episodes.csv", "ollama_rps_results/rounds.csv"]
    episodes = [{**r, "prefix": int(r["prefix"]), "b_reward": float(r["b_reward"]),
                 "paper_post": float(r["paper_post"]), "entropy": float(r["entropy"]),
                 "invalid": int(r["invalid"])} for r in read_csv(OUTPUTS / sources[0])]
    groups = defaultdict(list)
    for row in read_csv(OUTPUTS / sources[1]):
        groups[(row["model"], int(row["prefix"]), row["strategy"])].append(row)
    runs = []
    for (model, prefix, strategy), rows in groups.items():
        condition = f"{model} · Prefix {prefix} · {strategy}"
        score = Score(dataset, "l4", condition)
        rounds = []
        for r in sorted(rows, key=lambda row: int(row["round"])):
            data = score.record(ACTIONS.index(r["a"]), ACTIONS.index(r["b"]), phase=r["phase"],
                                detail=dict(raw=r["raw"], invalid=bool(int(r["invalid"])), strategy=strategy,
                                            model=model, prefix=prefix, recorded_round=int(r["round"])))
            if data.round != int(r["round"]) or data.reward_b != int(r["reward_b"]):
                raise ValueError(f"Recorded round or payoff mismatch in {condition}, round {r['round']}.")
            rounds.append(data.model_dump(exclude_none=True))
        runs.append(dict(condition=condition, model=model, prefix=prefix, strategy=strategy, rounds=rounds))
    return {"id": dataset, "sources": sources, "aggregate_only": False, "rows": episodes, "runs": runs,
            "paired": paired_ollama(episodes, "b_reward")}


DATASETS = [
    {"id": "ollama-controlled", "title": "Controlled Ollama Experiments", "category": "llm",
     "description": "Recorded decisions, rewards, and raw responses from matched-prefix experiments.",
     "files": ["ollama_rps_results/rounds.csv", "ollama_rps_results/episodes.csv"]},
    {"id": "neural-validation", "title": "Paired Neural Swap Validation", "category": "neural",
     "description": "Per-seed results for reward-only and swap-regularized online learning.",
     "files": ["neural_swap_results/runs.csv"]},
]
