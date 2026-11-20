# Rock Paper Scissors: Learning Strategies

A Python playground for studying zero-sum games, regret learning, and neural strategies through Rock Paper Scissors. The project has progressed from basic game simulation to adaptive learners, opponent prediction, and direct reward optimization.

## Work completed so far

### Game foundation

- Implemented players, action selection, repeated rounds, and cumulative scores.
- Defined a shared action order: Rock (`0`), Paper (`1`), Scissors (`2`).
- Added NumPy and PyTorch payoff representations, with rewards of `+1` for a win, `0` for a draw, and `-1` for a loss.
- Set a shared random seed (`42`) for NumPy and neural experiments.

The payoff matrix below gives Player A's reward; rows are A's actions and columns are B's actions. Player B receives the opposite reward.

| A / B | Rock | Paper | Scissors |
| --- | ---: | ---: | ---: |
| Rock | 0 | -1 | 1 |
| Paper | 1 | 0 | -1 |
| Scissors | -1 | 1 | 0 |

### Strategy and regret experiments

- Added baseline zero-sum, pure-strategy, and mixed-strategy simulations.
- Implemented Hedge with multiplicative weight updates, including fixed opponents, opponents that switch actions, and two learning players.
- Added comparisons against the best fixed action, cumulative reward, and average regret; switching-opponent experiments also examine dynamic regret.
- Implemented history-based external and swap regret calculations.
- Added a swap learner using action transition matrices and approximate stationary distributions, with fixed-opponent and self-play experiments.
- Added a Hedge versus swap-learner comparison that runs each algorithm in a separate game and reports scores and regret statistics.

### Neural opponent prediction

- Added synthetic training and test batches of five-move histories and a next-action target. Each sample has a preferred action: the generator follows it with probability `0.85` and otherwise draws a random action, which can also match the preference.
- Implemented an MLP with action embeddings and a small Transformer with token and position embeddings.
- Added supervised training with Adam and cross-entropy loss, plus held-out next-action accuracy reporting.
- Converted predicted opponent probabilities into expected rewards for A, then selected the action with the highest expected reward.
- Added example predictions for histories containing five Rocks, five Papers, or five Scissors.

### Direct reward optimization and fixes

- Added a separate neural training path that interprets model outputs as A's strategy and minimizes negative expected payoff against B's observed action.
- Added expected-reward evaluation and strategy probability reporting for this path. Despite its module name, this objective optimizes immediate expected reward; it does not explicitly minimize cumulative regret.
- Added explicit training and evaluation mode changes in the neural helpers.
- Resolved the integer-versus-floating-point payoff multiplication error; the shared PyTorch payoff tensor now contains floating-point values.

## Setup

From the project root, create an environment and install the dependencies listed in [requirements.txt](./requirements.txt):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies are NumPy, PyTorch, and torchvision. The project has been exercised with Python 3.10; dependencies are currently unpinned.

## Run the experiments

```bash
python main.py
```

By default, [main.py](./main.py) runs three neural experiments:

1. Supervised MLP opponent prediction for 100 training epochs.
2. Supervised Transformer opponent prediction for 100 training epochs.
3. Transformer strategy learning through direct reward optimization for 500 training epochs.

Each run prints its training loss, evaluation metrics, and decisions for the three constant-action histories. In these neural helpers, the `rounds` argument controls training epochs.

To run the baseline and regret-learning experiment suite separately:

```bash
python -c 'from main import checking_regret_learning; checking_regret_learning()'
```

To compare Hedge and swap learning directly:

```bash
python -c 'from src.regret_learning.compare_learners import CompareLearners; print(CompareLearners.results(rounds=1000, verbose=False))'
```

## Code guide

| Location | Purpose |
| --- | --- |
| [main.py](./main.py) | Experiment entry points |
| [src/constants.py](./src/constants.py) | Actions, seed, and PyTorch payoff tensor |
| [src/rock_paper_scissors.py](./src/rock_paper_scissors.py) | Payoff matrix and round rewards |
| [src/player.py](./src/player.py), [src/game.py](./src/game.py) | Players, action sampling, rounds, and scores |
| [src/regret_learning/](./src/regret_learning/) | Baselines, Hedge, swap learning, and regret comparisons |
| [src/llm/batch_generator.py](./src/llm/batch_generator.py) | Synthetic opponent histories |
| [src/llm/simple_opponent_predictor.py](./src/llm/simple_opponent_predictor.py) | MLP model |
| [src/llm/tiny_transformer.py](./src/llm/tiny_transformer.py) | Small Transformer model |
| [src/llm/neural_agent.py](./src/llm/neural_agent.py), [src/llm/simple_neural_train_eval_predict.py](./src/llm/simple_neural_train_eval_predict.py) | Supervised training, evaluation, and best responses |
| [src/llm/regret_neural_agent.py](./src/llm/regret_neural_agent.py), [src/llm/regret_train_eval_predict.py](./src/llm/regret_train_eval_predict.py) | Direct reward training, evaluation, and strategy output |

## Current scope

The neural models are small networks trained locally on synthetic data. The `src/llm` directory does not currently integrate a pretrained language model or an external LLM service.

Neural training reuses one generated training batch across epochs and evaluates on a separate generated batch. The MLP expects five-move histories, and the prediction printing helpers assume the three constant-action examples used by the entry point. Results are exploratory; the repository does not currently include an automated test suite, saved model checkpoints, or a benchmark report.
