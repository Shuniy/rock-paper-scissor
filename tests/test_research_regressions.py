import hashlib
import io
import json
from itertools import product

import numpy as np
import pytest
import torch

from src.constants import ACTIONS, PAYOFF_TENSOR
from src.llm import rps_ollama_baseline
from src.neural import (
    rps_blackbox_dynamics,
    strategic_swap_regret_transformer,
    swap_regret_transformer,
)
from src.neural.neural_agent import SimpleOpponentPredictorGame
from src.neural.regret_neural_agent import RegretOpponentPredictorGame
from src.player import Player
from src.regret_learning.compare_learners import CompareLearners
from src.regret_learning.swap_learner import SwapLearnerGame
from src.regret_learning.swap_regret import SwapRegretGame
from src.rock_paper_scissors import RockPaperScissors


def test_blackbox_gru_evaluation(monkeypatch):
    monkeypatch.setattr(rps_blackbox_dynamics, "EVAL_ROUNDS", 2)
    emulator = rps_blackbox_dynamics.BehaviourGRU().eval()
    metrics = rps_blackbox_dynamics.evaluate_episode(emulator, 1.0, 42, "learned_plan2")
    assert metrics.shape == (3,)
    assert np.isfinite(metrics).all()


@pytest.mark.parametrize("module", [swap_regret_transformer, strategic_swap_regret_transformer])
def test_transformer_treatments_use_matched_weights_and_batches(module, monkeypatch):
    model_class = module.TinyTransformer
    generate = module.generate_episodes
    initial_weights = []
    batches = []

    def make_model():
        model = model_class()
        initial_weights.append(torch.nn.utils.parameters_to_vector(model.parameters()).detach())
        return model

    def record_batch(*args, **kwargs):
        tensors = generate(*args, **kwargs)
        batches.append(tuple(hashlib.sha256(t.numpy().tobytes()).digest() for t in tensors))
        return tensors

    monkeypatch.setattr(module, "TinyTransformer", make_model)
    monkeypatch.setattr(module, "generate_episodes", record_batch)
    reward_model = module.train_model(0.0)
    module.evaluate(reward_model)
    reward_batches = batches.copy()
    batches.clear()
    swap_model = module.train_model(1.0)
    module.evaluate(swap_model)

    assert torch.equal(*initial_weights)
    assert len(batches) == 151  # 150 optimizer steps and one held-out batch
    assert reward_batches == batches
    assert any(not torch.equal(a, b) for a, b in zip(
        reward_model.parameters(), swap_model.parameters()
    ))


def test_stationary_distribution_for_slow_mixing_softmax():
    expected = np.array([0.1, 0.2, 0.7])
    transition = (1 - 1e-7) * np.eye(3) + 1e-7 * expected
    for learner in (SwapLearnerGame(), CompareLearners()):
        P = learner.get_transition_matrix(np.log(transition))
        pi = learner.get_stationary_distribution(P, rounds=100)
        assert (pi >= 0).all()
        np.testing.assert_allclose(pi.sum(), 1.0)
        np.testing.assert_allclose(pi @ P, pi, atol=1e-12, rtol=0)
        np.testing.assert_allclose(pi, expected, atol=1e-10, rtol=0)


def test_self_play_comparison_matches_draws_and_is_reproducible(monkeypatch):
    choose_action = Player.choose_action
    draws = []

    def record_draw(player, *, probabilities, rng):
        draws.append(rng.bit_generator.state)
        return choose_action(player, probabilities=probabilities, rng=rng)

    monkeypatch.setattr(Player, "choose_action", record_draw)
    rounds = 40
    first = CompareLearners.results(rounds=rounds, seed=17)
    assert draws[:2 * rounds] == draws[2 * rounds:]
    second = CompareLearners.results(rounds=rounds, seed=17)
    np.testing.assert_equal(first, second)
    for result in first.values():
        assert result["player_a"]["score"] + result["player_b"]["score"] == 0
    for player in ("player_a", "player_b"):
        stats = first["swap"][player]
        assert stats["swap_regret"] >= max(0, stats["external_regret"])


def test_rps_payoffs_are_zero_sum_and_follow_action_order():
    game = RockPaperScissors()
    np.testing.assert_array_equal(game.payoff_matrix, PAYOFF_TENSOR.numpy())
    for a, b in product(ACTIONS, repeat=2):
        reward_a, reward_b = game.play_round(a, b)
        assert reward_a == -reward_b
    for winner, loser in (("Rock", "Scissors"), ("Paper", "Rock"), ("Scissors", "Paper")):
        assert game.play_round(winner, loser) == (1, -1)


def test_historical_swap_regret_reports_four_numbered_rounds(capsys):
    SwapRegretGame().play_history(rounds=100, verbose=True)
    output = capsys.readouterr().out
    assert [line.split(":")[0] for line in output.splitlines() if line.startswith("Round ")] == [
        "Round 1", "Round 2", "Round 3", "Round 4"
    ]
    assert output.count("after 4 rounds") == 2


@pytest.mark.parametrize("raw, expected", [
    ("**pApEr**", 1), ("rock or paper", None), ("rock rock", None), ("", None),
])
def test_ollama_parsing_flags_ambiguous_responses(monkeypatch, raw, expected):
    response = json.dumps({"message": {"content": raw}}).encode()
    monkeypatch.setattr(rps_ollama_baseline.request, "urlopen",
                        lambda *args, **kwargs: io.BytesIO(response))
    seed = 17
    action, text, invalid = rps_ollama_baseline.choose_a([], False, np.random.default_rng(seed))
    fallback = np.random.default_rng(seed).integers(3)
    assert text == raw
    assert action == (fallback if expected is None else expected)
    assert invalid is (expected is None)


@pytest.mark.parametrize("wrapper", [SimpleOpponentPredictorGame, RegretOpponentPredictorGame])
def test_neural_wrapper_preserves_pytorch_modes_and_can_fit(wrapper):
    agent = wrapper()
    assert agent.eval() is agent
    assert not agent.training and not agent.model.training
    assert agent.train() is agent
    assert agent.training and agent.model.training
    weights = torch.nn.utils.parameters_to_vector(agent.parameters()).detach()
    agent.fit(epochs=1)
    assert not torch.equal(weights, torch.nn.utils.parameters_to_vector(agent.parameters()))
