"""Experiment metadata drives the configuration forms and server validation."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RPS_OPPONENTS = {
    "fixed": "Fixed action", "uniform": "Uniform random", "switch": "Switch Paper → Rock",
    "hedge": "Hedge learner",
}
LLM_OPPONENTS = {
    "uniform": "Uniform random", "always_scissors": "Always Scissors",
    "bait2_scissors": "Bait 2 → Scissors", "bait4_scissors": "Bait 4 → Scissors",
    "adaptive": "Adaptive frequency response",
}


def number(key, label, default, minimum, maximum, step=1, hint=""):
    return dict(key=key, label=label, default=default, min=minimum, max=maximum, step=step,
                kind="number", hint=hint)


def select(key, label, default, options, hint=""):
    return dict(key=key, label=label, default=default, options=options, kind="select", hint=hint)


SEED = number("seed", "Random seed", 42, 0, 2147483647, hint="Controls initialization and sampling.")
ETA = number("learning_rate", "Learning rate η", .1, .001, 1, .001)
LAMBDA = number("lambda_swap", "Swap coefficient λ", 1, 0, 5, .1)
HORIZON = select("horizon", "Planning horizon", "3", {"1": "1 round", "2": "2 rounds", "3": "3 rounds"},
                 "Future decisions evaluated at each turn.")
TOY = select("strategy", "Opponent strategy", "compare", {"compare": "Myopic vs planner", "myopic": "Myopic",
             "planner": "Learning-aware planner"})
ONLINE = select("online", "Online learning", "on", {"on": "Enabled", "off": "Frozen weights"})
INITIAL = dict(key="initial_history", label="Initial history", default="", kind="text",
               hint="Optional. Four space-separated RPS moves, oldest first.")
NEURAL_RATE = number("learning_rate", "Learning rate", .003, .0001, .1, .0001)
TRAIN = number("training_steps", "Training iterations", 60, 1, 500, hint="Optimizer steps per model.")
MODEL = dict(key="model", label="Installed Ollama model", kind="model", default="")
SECOND_MODEL = dict(key="model_secondary", label="Second model (optional)", kind="optional_model", default="",
                    hint="Each model generates its own warm-up histories.")
VALIDATION_STRATEGIES = {"uniform": "Uniform", "myopic": "Myopic", "plan2": "Planner-2"}
WINDOW = number("history_window", "History window", 8, 0, 30, hint="Completed rounds included in A’s prompt.")
TEMP = number("temperature", "Temperature", 0, 0, 2, .1)


def experiment(id, category, title, description, fields, rounds=30, maximum=200, game="rps", minimum=1):
    round_field = number("rounds", "Evaluation rounds", rounds, minimum, maximum,
                         hint="Scored decisions per condition.")
    return dict(id=id, category=category, title=title, description=description, game=game,
                fields=[round_field, *([] if any(f["key"] == "seed" for f in fields) else [SEED]), *fields])


def regret(id, title, description):
    return experiment(id, "regret", title, description, [
        select("strategy", "Opponent strategy", "switch", RPS_OPPONENTS),
        {**select("fixed_action", "Fixed action", "Paper", {a: a for a in ("Rock", "Paper", "Scissors")}),
         "when": {"strategy": ["fixed"]}}, ETA,
    ], rounds=60, maximum=500)


def neural(id, title, description, extra=()):
    return experiment(id, "neural", title, description, [
        TRAIN, NEURAL_RATE, number("history_window", "History window", 5, 1, 12,
                                   hint="Previous opponent moves per decision."), *extra,
    ], rounds=30, maximum=150)


CATALOG = {item["id"]: item for item in [
    regret("r1", "External Regret / Hedge", "Watch multiplicative weights adapt against a repeated opponent."),
    regret("r2", "Swap-Regret Learner", "Explore row learners, replacement probabilities, and stationary play."),
    regret("r3", "Hedge vs. SwapMW", "Compare two regret learners with matched seeds and evaluation lengths."),
    neural("n1", "Neural Opponent Prediction", "Train an MLP to predict the next move from recent opponent history."),
    neural("n2", "Transformer Prediction", "Compare the tiny Transformer and MLP on identical held-out examples.", [
        select("architecture", "Architecture", "compare", {"compare": "Transformer vs MLP", "transformer": "Transformer"}),
    ]),
    neural("n3", "Reward-Trained Transformer", "Train a policy directly on the expected payoff of its actions."),
    neural("n4", "Swap-Regularized Transformer", "Compare reward-only training with an expected-occupancy swap penalty.",
           [number("lambda_swap", "Comparison coefficient λ", 1, .1, 5, .1),
            select("evaluation_opponent", "Held-out opponent", "switch", {"fixed": "Fixed preference",
                   "switch": "Switching preference", "cycle": "Cycling preference", "uniform": "Uniform random"})]),
    experiment("s1", "strategy", "Bait / Exploit / Safe", "See when accepting an immediate cost can shape future play.",
               [TOY, {**HORIZON, "when": {"strategy": ["compare", "planner"]}}, {**ETA, "default": .5}],
               game="bait"),
    experiment("s2", "strategy", "Neural Learning-Aware Opponent", "Plan over copied neural weights in the constructed game.",
               [TOY, {**HORIZON, "when": {"strategy": ["compare", "planner"]}},
                {**ETA, "default": .3}, ONLINE], maximum=100, game="bait"),
    experiment("s3", "strategy", "Planning Horizon", "Compare one-, two-, and three-step white-box planners in RPS.",
               [select("strategy", "Opponent strategy", "compare", {"compare": "Compare all horizons",
                       "planner": "Planner", "myopic": "Myopic", "uniform": "Uniform"}),
                {**HORIZON, "when": {"strategy": ["planner"]}}, {**ETA, "default": .24}, LAMBDA, ONLINE, INITIAL],
               rounds=20, maximum=60),
    experiment("s4", "strategy", "Model Mismatch", "Test how assumptions about A’s updating affect B’s decisions.",
               [select("assumption", "Opponent model", "compare", {"compare": "Compare assumptions",
                       "exact": "Exact update", "half": "Half learning rate", "frozen": "Frozen approximation",
                       "wrong_objective": "Wrong swap objective", "myopic": "Myopic"}),
                {**HORIZON, "when": {"assumption": ["compare", "exact", "half", "frozen", "wrong_objective"]}},
                {**ETA, "default": .24}, LAMBDA, INITIAL], rounds=20, maximum=60),
    experiment("s5", "strategy", "Black-Box Opponent", "Predict A using observed moves and compare with a privileged oracle.",
               [select("strategy", "Opponent strategy", "frequency", {"uniform": "Uniform baseline",
                       "heuristic": "Global action-frequency heuristic", "frequency": "Contextual frequency predictor",
                       "frequency_plan": "Frequency planner", "gru": "GRU predictor", "gru_plan": "GRU MAP planner",
                       "oracle": "Privileged myopic oracle"}),
                {**HORIZON, "when": {"strategy": ["frequency_plan", "gru_plan"]}},
                number("warmup", "Exploration rounds", 10, 0, 40, hint="Unscored public rounds before evaluation."),
                {**TRAIN, "default": 12, "max": 50, "when": {"strategy": ["gru", "gru_plan"]}}, LAMBDA,
                {**ETA, "default": .24}], rounds=20, maximum=80),
    experiment("s6", "strategy", "Paired Neural Swap Validation", "Repeat reward-only and swap-regularized online RPS across matched initialization seeds.", [
        dict(SEED, label="First initialization seed", default=1001),
        number("seed_count", "Initialization seed pairs", 3, 2, 20),
        select("strategy", "Opponent strategy", "all", {"all": "All recorded strategies", **VALIDATION_STRATEGIES}),
    ], rounds=20, maximum=60),
    experiment("l1", "llm", "LLM Baseline", "Play repeated RPS with a locally installed model and inspect its raw responses.",
               [MODEL, WINDOW, TEMP, select("strategy", "Opponent strategy", "uniform", LLM_OPPONENTS)],
               rounds=12, maximum=60),
    experiment("l2", "llm", "History vs. No History", "Run the same frozen model with and without completed rounds in its prompt.",
               [MODEL, {**WINDOW, "min": 1}, TEMP, select("strategy", "Opponent strategy", "adaptive", LLM_OPPONENTS)],
               rounds=12, maximum=40),
    experiment("l3", "llm", "Strategic Baiting", "Measure what B earns during baiting and exploitation, and how often A later plays Paper.",
               [MODEL, WINDOW, TEMP, select("strategy", "Opponent strategy", "bait4_scissors", LLM_OPPONENTS)],
               rounds=12, maximum=60, minimum=5),
    experiment("l4", "llm", "Controlled Comparisons", "Branch five opponent strategies from each shared warm-up history.",
               [MODEL, SECOND_MODEL, {**WINDOW, "min": 1}, TEMP, number("prefixes", "Matched prefixes", 1, 1, 4),
                number("warmup", "Warm-up rounds", 3, 1, 12, hint="Played once per prefix and reused across all five strategies.")],
               rounds=8, maximum=30, minimum=5),
]}


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    experiment_id: str
    parameters: dict[str, str | int | float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_parameters(self):
        experiment = CATALOG.get(self.experiment_id)
        if experiment is None:
            raise ValueError("Unknown experiment.")
        fields = {f["key"]: f for f in experiment["fields"]}
        unknown = self.parameters.keys() - fields.keys()
        if unknown:
            raise ValueError(f"Unsupported parameters: {', '.join(sorted(unknown))}")
        resolved = {k: self.parameters.get(k, f["default"]) for k, f in fields.items()}
        validated = {}
        for key, value in resolved.items():
            field = fields[key]
            if any(resolved[k] not in allowed for k, allowed in field.get("when", {}).items()):
                continue
            if field["kind"] == "number":
                whole = isinstance(field["step"], int)
                if (not isinstance(value, (int, float)) or not field["min"] <= value <= field["max"]
                        or (whole and int(value) != value)):
                    raise ValueError(f"{field['label']} must be {'a whole number' if whole else 'a number'} "
                                     f"from {field['min']} to {field['max']}.")
                value = int(value) if whole else float(value)
            elif field["kind"] == "select":
                if not isinstance(value, str) or value not in field["options"]:
                    raise ValueError(f"Invalid {field['label'].lower()}.")
            elif field["kind"] == "text":
                if not isinstance(value, str) or (value.strip() and (
                    len(value.split()) != 4 or set(value.title().split()) - {"Rock", "Paper", "Scissors"}
                )):
                    raise ValueError("Initial history must contain four RPS moves.")
                value = " ".join(value.title().split())
            elif field["kind"] == "optional_model" and value == "":
                continue
            elif not isinstance(value, str) or not value.strip() or len(value) > 200:
                raise ValueError("Select an installed Ollama model.")
            validated[key] = value
        if validated.get("model_secondary") == validated.get("model") and "model_secondary" in validated:
            raise ValueError("Choose two different Ollama models.")
        self.parameters = validated
        return self


class Round(BaseModel):
    run_id: str
    experiment_id: str
    condition: str
    round: int
    a: str
    b: str
    reward_a: float
    reward_b: float
    cumulative_a: float
    cumulative_b: float
    average_a: float
    average_b: float
    external: float
    swap: float
    frequency: list[float]
    policy: list[float] | None = None
    phase: str | None = None
    history: list | None = None
    detail: dict = Field(default_factory=dict)


class Event(BaseModel):
    type: Literal["started", "training", "warmup", "round", "completed", "cancelled", "failed"]
    data: dict
