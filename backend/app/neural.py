"""Small CPU training runs with held-out evaluation and matched conditions."""
import numpy as np
import torch
from torch.nn import functional as F

from src.constants import PAYOFF_TENSOR
from src.neural.batch_generator import make_batch
from src.neural.simple_opponent_predictor import SimpleOpponentPredictor
from src.neural.swap_regret_transformer import calculate_metrics, generate_episodes
from src.neural.tiny_transformer import TinyTransformer

from .catalog import Event
from .rounds import ACTIONS, RPS, Score, draw


def simulate(run_id, exp, p, conditions):
    seed, size, rounds = p["seed"], p["history_window"], p["rounds"]
    eval_rng = torch.Generator().manual_seed(seed + 1)
    supervised = exp in {"n1", "n2"}
    episode = exp == "n4"
    if episode:
        eval_h, eval_b = generate_episodes(4, size, rounds, eval_rng)
        selected = ("fixed", "switch", "cycle", "uniform").index(p["evaluation_opponent"])
        eval_h, eval_b = eval_h[selected], eval_b[selected]
    else:
        eval_h, eval_b = make_batch(rounds, size, eval_rng)
    draws = np.random.default_rng(seed).random(rounds)
    for condition in conditions:
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed + 2)
        model = SimpleOpponentPredictor(size) if condition == "MLP" else TinyTransformer(size)
        optimizer = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
        weight = p["lambda_swap"] if condition.startswith("Swap") else 0.0
        steps = p["training_steps"]
        for step in range(1, steps + 1):
            stage = {}
            if episode:
                histories, targets = generate_episodes(12, size, 20, rng)
                probabilities = model(histories.flatten(end_dim=1)).softmax(-1).reshape(12, 20, 3)
                reward, surrogate = calculate_metrics(probabilities, targets)
                loss = -reward + weight * surrogate
                stage = {"expected_a": reward.item(), "swap_surrogate": surrogate.item()}
            else:
                histories, targets = make_batch(64, size, rng)
                logits = model(histories)
                if supervised:
                    loss = F.cross_entropy(logits, targets)
                else:
                    expected = (logits.softmax(-1) * PAYOFF_TENSOR.T[targets]).sum(-1).mean()
                    loss = -expected
                    stage = {"expected_a": expected.item()}
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            yield Event(type="training", data=dict(condition=condition, step=step, total=steps,
                        loss=loss.item(), **stage))

        model.eval()
        with torch.no_grad():
            probabilities = model(eval_h).softmax(-1)
        score = Score(run_id, exp, condition)
        correct = 0
        for t in range(rounds):
            b = int(eval_b[t])
            probs = probabilities[t].numpy()
            detail = {"strategy": "Synthetic held-out opponent", "online_learning": False,
                      "explanation": ("Counter the predicted distribution by selecting the highest expected RPS payoff."
                                      if supervised else "Sample A’s learned policy. Model weights stay fixed during evaluation.")}
            if episode:
                detail["strategy"] = p["evaluation_opponent"]
            if supervised:
                correct += int(probs.argmax() == b)
                a = int((RPS @ probs).argmax())
                policy = np.eye(3)[a]
                detail.update(prediction=probs.tolist(), prediction_accuracy=correct / (t + 1),
                              predicted_action=ACTIONS[probs.argmax()], predicted_payoff=float((RPS @ probs)[a]))
            else:
                a = draw(probs, draws[t])
                policy = probs
                detail["expected_a"] = float(probs @ RPS[:, b])
                detail["policy_entropy"] = float(-(probs * np.log(np.clip(probs, 1e-12, 1))).sum() / np.log(3))
                if episode:
                    with torch.no_grad():
                        _, surrogate = calculate_metrics(probabilities[None, :t + 1], eval_b[None, :t + 1])
                    detail["swap_surrogate"] = surrogate.item()
            row = score.record(a, b, policy=policy, history=[ACTIONS[x] for x in eval_h[t]], detail=detail)
            yield Event(type="round", data=row.model_dump(exclude_none=True))
