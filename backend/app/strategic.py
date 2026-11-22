"""Adapt research planners and online learners to cancellable round events."""
import numpy as np
import torch
from torch.nn import functional as F

from src.neural import learning_aware_opponent as bait
from src.neural import online_neural_learning as neural_bait
from src.neural import online_transformer_rps_model_mismatch as online
from src.neural import blackbox_opponent as frequency
from src.neural import rps_blackbox_dynamics as gru

from .catalog import Event, VALIDATION_STRATEGIES
from .rounds import ACTIONS, RPS, Score, draw


def toy(run_id, exp, p, conditions):
    neural = exp == "s2"
    rate = p["learning_rate"]
    for condition in conditions:
        torch.manual_seed(p["seed"])
        learner = neural_bait.NeuralLearner() if neural else np.zeros(2)
        rng = np.random.default_rng(p["seed"])
        score = Score(run_id, exp, condition, bait.A_PAYOFF, bait.B_PAYOFF, bait.ACTIONS_A, bait.ACTIONS_B)
        method = "myopic" if condition == "Myopic" else "planner"
        horizon = int(p.get("horizon", 1))
        updating = p.get("online") != "off"
        public = []
        for random_a in rng.random(p["rounds"]):
            if neural:
                with torch.no_grad():
                    policy = learner().numpy().copy()
                b, plan, value = neural_bait.choose_b(learner, method, horizon, rate if updating else 0,
                                                       return_plan=True)
            else:
                policy = bait.policy(learner)
                b, plan, value = bait.choose_b(learner, method, horizon, rate, return_plan=True)
            a = draw(policy, random_a)
            vector = bait.A_PAYOFF[:, b]
            detail = dict(strategy=condition, expected_a=float(policy @ vector),
                          expected_b=float(policy @ bait.B_PAYOFF[:, b]),
                          predicted_return=value, plan=[bait.ACTIONS_B[x] for x in plan],
                          counterfactual_rewards=vector.tolist(), online_learning=updating,
                          explanation="B evaluates expected payoff along its chosen plan. A updates after the actual move is revealed."
                          if updating else "A’s weights are frozen, so the planner rolls out an unchanged policy.")
            if neural:
                if updating:
                    detail["training_loss"] = neural_bait.update_a(learner, b, rate)
                with torch.no_grad():
                    detail["next_policy"] = learner().tolist()
            else:
                learner += rate * vector
                detail["post_log_weights"] = learner.tolist()
                detail["next_policy"] = bait.policy(learner).tolist()
            row = score.record(a, b, policy=policy, phase=bait.ACTIONS_B[b].lower(),
                               history=public[-8:], detail=detail)
            public.append({"a": row.a, "b": row.b})
            yield Event(type="round", data=row.model_dump(exclude_none=True))


def online_rps(run_id, exp, p, conditions, validation_case=None):
    rate, weight, seed = p["learning_rate"], p["lambda_swap"], p["seed"]
    horizon = int(p.get("horizon", 1))
    blackbox = exp == "s5"
    strategy = p.get("strategy")
    learning = p.get("online") != "off"
    warmup = p.get("warmup", 0)
    emulator = None
    if strategy in {"gru", "gru_plan"}:
        collection = []
        for i in range(12):
            # Disjoint from the live evaluation seed, alternating both objectives.
            collection.append(gru.collect_training_episode(2**31 + i, i % 2, 16, learning_rate=rate))
            yield Event(type="training", data=dict(condition="GRU trajectories", step=i + 1, total=12))
        torch.manual_seed(seed)
        emulator = gru.BehaviourGRU()
        optimizer = torch.optim.Adam(emulator.parameters(), lr=.003)
        xs, ys = (torch.stack(values) for values in zip(*collection))
        for i in range(p["training_steps"]):
            logits = emulator(xs)[:, online.HISTORY - 1:]
            loss = F.cross_entropy(logits.reshape(-1, 3), ys.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            yield Event(type="training", data=dict(condition="GRU fit", step=i + 1,
                                                    total=p["training_steps"], loss=loss.item()))
        emulator.eval()

    for condition in conditions:
        torch.manual_seed(seed)
        model = online.OnlineTransformer()
        rng = np.random.default_rng(seed)
        initial = p.get("initial_history", "").split()
        context = torch.tensor([ACTIONS.index(x) for x in initial]) if initial else torch.tensor(
            rng.integers(0, 3, size=online.HISTORY), dtype=torch.long)
        tokens = [9 + int(b) for b in context]
        a_draws = rng.random(warmup + p["rounds"])
        b_rng = np.random.default_rng(seed + 100000)
        memory, public = [], []
        estimator = frequency.BlackBoxPredictor() if blackbox else None
        score = Score(run_id, exp, validation_case["label"] if validation_case else condition)
        for t, random_a in enumerate(a_draws):
            with torch.no_grad():
                probs = model(context.unsqueeze(0))[0].numpy().copy()
            past = [int(x) for x in context]
            plan, value = None, None
            prediction = None
            detail = dict(strategy=condition, online_learning=learning,
                          explanation="A observes B’s completed move and updates on its recent observation buffer."
                          if learning else "Only the history window changes. A’s weights stay frozen.")
            if validation_case:
                detail.update(validation_seed=validation_case["seed"], validation_strategy=validation_case["strategy"],
                              lambda_swap=weight)
            if blackbox:
                if emulator is not None:
                    prediction = gru.predict_a(emulator, tokens)
                elif strategy == "heuristic":
                    prediction = estimator.predict([])
                elif strategy in {"frequency", "frequency_plan"}:
                    prediction = estimator.predict(past)
                if t < warmup or strategy == "uniform":
                    b = int(b_rng.integers(3))
                elif strategy == "oracle":
                    b = int((-(probs @ RPS)).argmax())
                elif strategy == "frequency_plan" and horizon > 1:
                    b, plan, value = frequency.choose_blackbox_b(estimator, past, f"blackbox_plan{horizon}", True)
                elif strategy == "gru_plan" and horizon > 1:
                    b, plan, value = gru.choose_b_blackbox(emulator, tokens, f"learned_plan{horizon}", True)
                else:
                    b = int((-(prediction @ RPS)).argmax())
                detail["visibility"] = ("Privileged policy access" if strategy == "oracle" and t >= warmup else
                                        "Completed public actions")
            elif condition == "Uniform":
                b = int(b_rng.integers(3))
            else:
                if condition == "Myopic":
                    method = "myopic"
                elif condition.startswith("Planner-"):
                    method = f"plan{condition[-1]}"
                else:
                    method = f"plan{horizon}"
                assumed_rate = rate if learning else 0
                if condition == "Half learning rate":
                    assumed_rate *= .5
                elif condition == "Frozen approximation":
                    assumed_rate = 0
                assumed_weight = float(weight == 0) if condition == "Wrong objective" else weight
                b, plan, value = online.choose_b(model, context, memory, method, weight,
                                                 assumed_rate, assumed_weight, return_plan=True)
                detail.update(assumed_learning_rate=assumed_rate, assumed_swap_coefficient=assumed_weight)
            if plan is not None:
                detail.update(plan=[ACTIONS[x] for x in plan], predicted_return=value)

            a = draw(probs, random_a)
            detail.update(expected_a=float(probs @ RPS[:, b]), expected_b=float(-(probs @ RPS[:, b])),
                          policy_entropy=float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum() / np.log(3)))
            if prediction is not None:
                detail.update(prediction=prediction.tolist(), prediction_tv=float(np.abs(prediction - probs).sum() / 2))
            if estimator is not None:
                estimator.update(past, a)
                tokens.append(3 * b + a)
                detail["public_history"] = public.copy()
            public.append({"a": ACTIONS[a], "b": ACTIONS[b]})
            next_context = torch.cat((context[1:], torch.tensor([b])))
            if exp in {"s3", "s4"} and learning:
                with torch.no_grad():
                    before_update = model(next_context.unsqueeze(0))[0].numpy()
                detail["next_policy_frozen"] = before_update.tolist()
            if learning:
                memory, metrics = online.update_a(model, memory, context, b, weight, rate, return_metrics=True)
                detail.update(metrics)
            with torch.no_grad():
                detail["next_policy"] = model(next_context.unsqueeze(0))[0].tolist()
            context = next_context

            if t < warmup:
                yield Event(type="warmup", data=dict(condition=condition, step=t + 1, total=warmup,
                                                      a=ACTIONS[a], b=ACTIONS[b]))
            else:
                row = score.record(a, b, policy=probs, history=[ACTIONS[x] for x in past], detail=detail)
                yield Event(type="round", data=row.model_dump(exclude_none=True))


def validation(run_id, exp, p, cases):
    for label, strategy, seed, weight in cases:
        case = {"label": label, "strategy": strategy, "seed": seed}
        params = {**p, "seed": seed, "lambda_swap": weight, "learning_rate": online.LR}
        yield from online_rps(run_id, exp, params, [VALIDATION_STRATEGIES[strategy]], case)
