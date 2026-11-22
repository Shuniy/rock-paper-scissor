import itertools

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from backend.app.catalog import CATALOG, RunRequest
from backend.app.regret import hedge_policy, stationary
from backend.app.results import summarize, validation_pairs
from backend.app.rounds import RPS, Score, window
from backend.app.simulations import describe, execute, plan
from src.constants import ACTIONS
from src.neural import online_transformer_rps_model_mismatch as reference
from src.neural.learning_aware_opponent import A_PAYOFF, B_PAYOFF
from src.regret_learning.hedge import HedgeGame
from src.regret_learning.swap_learner import SwapLearnerGame
from src.rock_paper_scissors import RockPaperScissors


def run_events(exp, **parameters):
    p = RunRequest(experiment_id=exp, parameters=parameters).parameters
    return list(execute("test", exp, p, plan(exp, p)))


def traces(events):
    return [e.data for e in events if e.type == "round"]


def test_payoff_matrix_full_cross_product():
    game = RockPaperScissors()
    for a, b in itertools.product(range(3), repeat=2):
        actual = game.play_round(ACTIONS[a], ACTIONS[b])
        expected = 0 if a == b else 1 if (a - b) % 3 == 1 else -1
        assert actual == (expected, -expected)
        assert actual == (RPS[a, b], -RPS[a, b])


def test_regrets_equal_brute_force_mapping_over_each_prefix():
    game = Score("test", "r1", "Hedge")
    pairs = [(0, 2), (1, 0), (2, 1), (1, 2), (0, 1)]
    for t, (a, b) in enumerate(pairs, start=1):
        row = game.record(a, b)
        used = pairs[:t]
        actual = sum(RPS[i, j] for i, j in used)
        external = max(sum(RPS[k, j] for _, j in used) for k in range(3)) - actual
        swap = max(sum(RPS[mapping[i], j] for i, j in used)
                   for mapping in itertools.product(range(3), repeat=3)) - actual
        assert (row.external, row.swap) == (external, swap)
        assert row.swap >= max(0, row.external)
        if t == 3:
            assert row.external == -3


def test_stationary_solution_with_nonuniform_transition_rows():
    learner = SwapLearnerGame()
    weights = np.array([[3., -1., 0.], [1., 1., 2.], [0., 1., -5.]])
    matrix = learner.get_transition_matrix(weights)
    pi = stationary(matrix)
    np.testing.assert_allclose(matrix.sum(axis=1), [1, 1, 1])
    np.testing.assert_allclose(pi.sum(), 1)
    np.testing.assert_allclose(pi @ matrix, pi, atol=1e-12)
    np.testing.assert_allclose(learner.get_stationary_distribution(matrix, 1000), pi, atol=1e-12)


def test_stationary_solution_avoids_underflow_near_identity():
    matrix = np.eye(3) + np.full((3, 3), 1e-200)
    matrix /= matrix.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(stationary(matrix), np.full(3, 1 / 3))


def test_hedge_update_matches_exp_formula_and_next_pre_policy():
    rows = traces(run_events("r1", rounds=3, strategy="fixed", fixed_action="Paper", learning_rate=.2))
    np.testing.assert_allclose(rows[0]["detail"]["post_weights"], np.exp(.2 * RPS[:, 1]))
    np.testing.assert_allclose(rows[1]["policy"], hedge_policy(.2 * RPS[:, 1]))
    np.testing.assert_allclose(rows[0]["detail"]["next_policy"], rows[1]["policy"])


def test_original_hedge_dynamic_comparator_uses_configured_switch(capsys):
    HedgeGame().experiment_b("Rock", "Scissors", rounds=4, change_action_after=1)
    output = capsys.readouterr().out
    assert "Best reward in first phase: 1" in output
    assert "Best reward in second phase: 3" in output
    assert "Best dynamic reward: 4" in output


def test_hedge_opponent_uses_b_payoff_after_reveal():
    seed = 7
    rows = traces(run_events("r1", rounds=10, strategy="hedge", seed=seed, learning_rate=.4))
    weights = np.zeros(3)
    for row, draw in zip(rows, np.random.default_rng(seed + 1).random(10)):
        b = int(np.searchsorted(np.cumsum(hedge_policy(weights)), draw, side="right"))
        assert row["b"] == ACTIONS[b]
        weights -= .4 * RPS[ACTIONS.index(row["a"]), :]


def test_swap_update_scales_each_row_by_stationary_mass():
    rows = traces(run_events("r2", rounds=3, strategy="fixed", fixed_action="Rock", learning_rate=.3))
    expected = .3 * np.outer(rows[0]["policy"], RPS[:, 0])
    np.testing.assert_allclose(rows[0]["detail"]["post_log_weights"], expected)
    np.testing.assert_allclose(rows[0]["detail"]["next_policy"], rows[1]["policy"])


def test_matched_regret_conditions_precommit_same_uniform_moves():
    rows = traces(run_events("r3", rounds=10, seed=12, strategy="uniform"))
    assert [r["b"] for r in rows[:10]] == [r["b"] for r in rows[10:]]
    assert rows == traces(run_events("r3", rounds=10, seed=12, strategy="uniform"))
    assert rows != traces(run_events("r3", rounds=10, seed=13, strategy="uniform"))


@pytest.mark.parametrize("experiment,params", [
    ("nope", {}), ("r1", {"rounds": 0}), ("r1", {"rounds": 4.5}), ("r1", {"rounds": True}),
    ("r1", {"learning_rate": float("nan")}), ("r1", {"missing_field": 1}),
    ("l1", {"model": ""}), ("l4", {"model": "llama", "model_secondary": "llama"}),
    ("s3", {"initial_history": "rock unknown"}), ("s6", {"seed_count": 1}),
])
def test_config_validation(experiment, params):
    with pytest.raises(ValidationError):
        RunRequest(experiment_id=experiment, parameters=params)


def test_valid_config_resolves_defaults_and_hides_irrelevant_inputs():
    request = RunRequest(experiment_id="s3", parameters={"strategy": "myopic", "initial_history": " rock Paper ROCK scissors "})
    assert request.parameters["rounds"] == 20
    assert request.parameters["initial_history"] == "Rock Paper Rock Scissors"
    assert "horizon" not in request.parameters


def test_window_boundaries():
    values = [1, 2, 3]
    assert window(values, 0) == []
    assert window(values, 1) == [3]
    assert window(values, 10) == values


@pytest.mark.parametrize("exp", [key for key in CATALOG if not key.startswith("l")])
def test_every_cpu_experiment_generates_valid_rounds(exp):
    options = {f["key"]: 1 for f in CATALOG[exp]["fields"] if f["key"] in {"rounds", "training_steps"}}
    if exp == "s6":
        options.update(seed_count=2, strategy="uniform")
    params = RunRequest(experiment_id=exp, parameters=options).parameters
    conditions = plan(exp, params)
    labels, total, _ = describe(exp, params, conditions)
    rows = traces(execute("test", exp, params, conditions))
    assert len(rows) == total
    assert list(dict.fromkeys(r["condition"] for r in rows)) == labels
    assert all(row["round"] == 1 for row in rows)
    assert all(sum(row["policy"]) == pytest.approx(1) for row in rows)


def test_matched_neural_inputs_and_seed_reproducibility():
    kwargs = {"rounds": 4, "seed": 66, "history_window": 2, "training_steps": 3}
    rows = traces(run_events("n2", **kwargs))
    assert rows == traces(run_events("n2", **kwargs))
    assert [(r["history"], r["b"]) for r in rows[:4]] == [(r["history"], r["b"]) for r in rows[4:]]
    assert all(len(r["history"]) == 2 and len(r["detail"]["prediction"]) == 3 for r in rows)


def test_bait_game_matrices_and_planning_advantage():
    rows = traces(run_events("s1", rounds=30))
    planner, myopic = rows[:30], rows[30:]
    assert planner[0]["b"] == "Bait"
    assert any(r["b"] == "Exploit" for r in planner)
    assert all(r["b"] == "Safe" for r in myopic)
    assert myopic[-1]["average_b"] == pytest.approx(.2)
    assert np.mean([r["detail"]["expected_b"] for r in planner]) == pytest.approx(.693, abs=.001)
    for row in rows:
        a, b = ["X", "Y"].index(row["a"]), ["Bait", "Exploit", "Safe"].index(row["b"])
        assert row["reward_a"] == A_PAYOFF[a, b]
        assert row["reward_b"] == B_PAYOFF[a, b]


def test_white_box_planning_leaves_real_model_and_history_untouched():
    model = reference.OnlineTransformer()
    state = {k: v.clone() for k, v in model.state_dict().items()}
    context = torch.tensor([0, 1, 2, 0])
    initial = context.clone()
    action, sequence, _ = reference.choose_b(model, context, [], "plan2", 1, return_plan=True)
    assert action == sequence[0] and len(sequence) == 2
    assert torch.equal(context, initial)
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(param.grad is None for param in model.parameters())


def test_online_history_cannot_include_current_move():
    rows = traces(run_events("s3", rounds=4, strategy="uniform", initial_history="Rock Rock Rock Rock"))
    assert rows[0]["history"] == ["Rock"] * 4
    for previous, current in zip(rows, rows[1:]):
        assert current["history"] == previous["history"][1:] + [previous["b"]]


@pytest.mark.parametrize("strategy", ["frequency", "frequency_plan", "gru", "gru_plan", "oracle"])
def test_blackbox_strategies_and_warmup_boundaries(strategy):
    events = run_events("s5", rounds=2, strategy=strategy, warmup=2, training_steps=1, horizon="2")
    assert [e.data["step"] for e in events if e.type == "warmup"] == [1, 2]
    rows = traces(events)
    assert len(rows) == 2 and rows[0]["round"] == 1
    assert len(rows[0]["detail"]["public_history"]) == 2
    if strategy != "oracle":
        assert rows[0]["detail"]["visibility"] == "Completed public actions"
        assert 0 <= rows[0]["detail"]["prediction_tv"] <= 1


def test_live_validation_matches_original_reference():
    events = run_events("s6", seed_count=2, strategy="myopic")
    rows = summarize([event.model_dump() for event in events])
    assert len(rows) == 4
    for row in rows:
        baseline = reference.run(row["strategy"], row["lambda_swap"], row["seed"])
        for key, original in [("expected_b", "expected_b"), ("average_b", "realized_b"),
                              ("swap_per_round", "swap_a"), ("policy_entropy", "a_entropy")]:
            assert row[key] == pytest.approx(baseline[original], abs=2e-7)
    assert validation_pairs(rows, 20)[0]["pairs"] == 2
