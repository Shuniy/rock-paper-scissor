"""Registry dispatch. Iterators yield observable milestones at cancellation boundaries."""
from .catalog import CATALOG, LLM_OPPONENTS, VALIDATION_STRATEGIES

ASSUMPTIONS = {"exact": "Exact update", "half": "Half learning rate", "frozen": "Frozen approximation",
               "wrong_objective": "Wrong objective", "myopic": "Myopic"}


def models_for(p):
    return [p[key] for key in ("model", "model_secondary") if key in p]


def plan(exp, p):
    if exp.startswith("r"):
        return {"r1": ["Hedge"], "r2": ["SwapMW"], "r3": ["Hedge", "SwapMW"]}[exp]
    if exp.startswith("n"):
        return {"n1": ["MLP"], "n2": ["Transformer", "MLP"] if p.get("architecture") == "compare" else ["Transformer"],
                "n3": ["Reward only"], "n4": ["Reward only", f"Swap λ={p.get('lambda_swap')}"]}[exp]
    if exp in {"s1", "s2"}:
        if p["strategy"] == "compare":
            return ["Planner", "Myopic"]
        return [p["strategy"].capitalize()]
    if exp == "s3":
        if p["strategy"] == "compare":
            return ["Myopic", "Planner-1", "Planner-2", "Planner-3"]
        return [p["strategy"].capitalize()]
    if exp == "s4":
        return list(ASSUMPTIONS.values()) if p["assumption"] == "compare" else [ASSUMPTIONS[p["assumption"]]]
    if exp == "s5":
        field = next(f for f in CATALOG[exp]["fields"] if f["key"] == "strategy")
        return [field["options"][p["strategy"]]]
    if exp == "s6":
        methods = VALIDATION_STRATEGIES if p["strategy"] == "all" else [p["strategy"]]
        return [(f"{VALIDATION_STRATEGIES[method]} · seed {seed} · λ={weight}", method, seed, weight)
                for method in methods for seed in range(p["seed"], p["seed"] + p["seed_count"])
                for weight in (0, 1)]
    if exp == "l2":
        return [(p["model"], 0, [("History shown", p["strategy"]), ("No history", p["strategy"])])]
    if exp == "l4":
        models = models_for(p)
        return [(model, prefix, [
            ((f"{model} · " if len(models) > 1 else "") + f"Prefix {prefix + 1} · {name}", key)
            for key, name in LLM_OPPONENTS.items()
        ]) for model in models for prefix in range(p["prefixes"])]
    return [(p["model"], 0, [(LLM_OPPONENTS[p["strategy"]], p["strategy"])])]


def describe(exp, p, conditions):
    llm = exp.startswith("l")
    if llm:
        names = [label for _, _, group in conditions for label, _ in group]
    else:
        names = [label for label, _, _, _ in conditions] if exp == "s6" else conditions
    total = len(names) * p["rounds"]
    return names, total, total + len(conditions) * p.get("warmup", 0) if llm else None


def execute(run_id, exp, parameters, conditions):
    if exp.startswith("r"):
        from .regret import simulate
    elif exp.startswith("n"):
        import torch

        from .neural import simulate
        with torch.random.fork_rng(devices=[]):
            yield from simulate(run_id, exp, parameters, conditions)
        return
    elif exp.startswith("s"):
        import torch

        from .strategic import online_rps, toy, validation
        simulate = {"s1": toy, "s2": toy, "s6": validation}.get(exp, online_rps)
        with torch.random.fork_rng(devices=[]):
            yield from simulate(run_id, exp, parameters, conditions)
        return
    else:
        from .llm import simulate
    yield from simulate(run_id, exp, parameters, conditions)
