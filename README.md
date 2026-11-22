# Rock–Paper–Scissors Simulation Lab

Run repeated games in a local web app to see how agents learn, respond to opponents, and score over time. The lab includes zero-sum rock–paper–scissors (RPS) and a small Bait / Exploit / Safe game designed to show how an opponent can influence an online learner's future choices.

- **Agent A** is the learner or policy under evaluation. It can use multiplicative weights, a neural network, or a local language model prompted with game history.
- **Agent B** is the opponent. It can follow a fixed or random strategy, learn from play, maximize immediate payoff, or plan around predicted changes in A's policy.

## Features

- Inspect Hedge and SwapMW updates, action probabilities, and external and swap regret.
- Train an MLP or a small Transformer to predict moves, or train a Transformer policy on expected reward with an optional swap-regret penalty.
- Compare learning-aware planners across horizons and assumptions about A. Test white-box access to A's policy and updates or black-box estimates from public moves.
- Play against frozen local Ollama models and compare prompt histories, baiting schedules, and adaptive responses.
- Track progress, pause or step through rounds, inspect actions and payoffs, compare compatible runs, and export CSV or JSON results.

## Getting started

Use Python 3.10+ and Node.js 20.19+ in the 20.x line or Node.js 22.12+. The small PyTorch models run on CPU. Ollama is optional and is only needed to run new LLM simulations.

From the repository root, install the dependencies and start both services.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend ci
./dev.sh
```

Open [the simulation lab](http://127.0.0.1:5173/). Choose **Run Simulation** or an experiment in the sidebar, set the evaluation length, seed, and mode-specific options, then start the run. Scrub the timeline or select a round to inspect it. **Replay Research Results** opens the bundled Ollama rounds, neural seed summaries, and saved local runs. **Experiment Comparison** overlays compatible completed local runs.

To run the backend and frontend separately, use these commands in two terminals. Vite proxies API requests to port 8000 by default. Set `LAB_API_TARGET` when starting Vite if the API is listening elsewhere. [Development guide](docs/development.md)

```bash
# Terminal 1
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 5

# Terminal 2
npm --prefix frontend run dev
```

### Enable Ollama

Start the Ollama app or run `ollama serve` in a separate terminal. Use `ollama list` to check which models are installed. Select an installed model in the lab. The default chat endpoint is `http://127.0.0.1:11434/api/chat`. Set `OLLAMA_URL` before starting the backend to use a different chat endpoint. The bundled results can be replayed without Ollama.

## Simulation modes

| Mode | Configurations and what to inspect |
| --- | --- |
| **Regret learning** | Run Hedge, SwapMW, or both against fixed, uniform, Paper-to-Rock switching, or Hedge opponents. Inspect counterfactual rewards, weights, the swap transition matrix, and how cumulative regret changes. The comparison shares sampling draws across conditions. A learning opponent can still generate different move sequences. |
| **Neural learning** | Train a next-move MLP or Transformer, compare them on the same held-out examples, or optimize a Transformer directly for reward. The swap-regularized comparison pairs reward-only training with a configurable penalty and evaluates on a noisy fixed-preference, switching, cycling, or uniform opponent. The fitted models stay frozen during evaluation. |
| **Learning-aware strategy** | Compare myopic play and planning against Hedge or a small online neural policy in Bait / Exploit / Safe. In RPS, compare one- to three-round horizons, freeze A's updates, or vary B's assumptions about A's learning rate and objective. Black-box options include global and contextual action counts, a GRU trained on simulated action pairs, and planners that use these estimates. A privileged myopic oracle provides a reference. Paired neural validation runs the two training objectives across matched seeds. |
| **LLM simulations** | Run a baseline, compare prompts with and without completed rounds, or inspect fixed baiting and adaptive frequency responses. The controlled mode branches five B strategies from the same warm-up history for each model and prefix. Inspect raw responses, invalid-output flags, phase payoffs, and later Paper frequency. |

The catalog lists all 17 experiments and their options. [Experiment catalog](docs/experiments.md)

## How it works

### Games and payoffs

Both agents select a move before either current move is revealed. In RPS, A scores `+1` for a win, `−1` for a loss, and `0` for a draw. B receives the opposite payoff. A's matrix uses Rock, Paper, Scissors in that order for its rows and columns. Code that indexes B's own action first uses `PAYOFF_B = -PAYOFF_A.T`.

```text
             B: Rock  Paper  Scissors
A: Rock            0     -1        +1
A: Paper          +1      0        -1
A: Scissors       -1     +1         0
```

Bait / Exploit / Safe is a general-sum game. In the table, each cell gives the payoff pair `(A, B)`. Bait costs B `0.1` and makes X favorable to A. Exploit rewards B when A chooses X, while Safe pays B `0.2` regardless of A's move.

| A action | Bait | Exploit | Safe |
| --- | --- | --- | --- |
| X | `(+1, −0.1)` | `(−1, +3)` | `(0, +0.2)` |
| Y | `(0, −0.1)` | `(+1, −3)` | `(0, +0.2)` |

### Learning and planning

- **Hedge** normalizes action weights to get its policy. After observing B's move, each action's weight is multiplied by $\exp(\eta r_t(i))$, where $r_t(i)$ is the payoff that action $i$ would have earned. The implementation stores weights in log space.
- **SwapMW** uses a Blum–Mansour-style construction. Three row learners form a transition matrix $P$ of replacement probabilities. A plays its stationary distribution $\pi P=\pi$ and scales each row's update by the probability of its source action.
- **Neural prediction** uses cross-entropy to fit B's next action from its past moves. The MLP flattens learned 8-dimensional action embeddings. The Transformer uses learned action and position embeddings and one causal encoder layer with four attention heads. A chooses the best RPS response to the predicted distribution. The noisy-favourite generator plays its favourite with effective probability $0.85+0.15/3=0.90$.
- **Neural policy training** outputs A's action distribution $\pi_\theta(\cdot\mid h_t)$. The loss is negative expected payoff plus an optional penalty. For each source action $i$, the penalty takes the best replacement gain across destinations $j$, weighted by the probability of having played $i$. The objective averages over $T$ observations.

$$
\begin{aligned}
\widehat R_{ij} &= \frac{1}{T}\sum_{t=1}^{T}\pi_\theta(i\mid h_t)
  \bigl[u_A(j,b_t)-u_A(i,b_t)\bigr], \\
\mathcal L &= -\frac{1}{T}\sum_{t=1}^{T}\sum_i\pi_\theta(i\mid h_t)u_A(i,b_t)
  +\lambda\sum_i\max_j\widehat R_{ij}.
\end{aligned}
$$

- **Online RPS** uses four prior B moves as the policy input. With learning enabled, A updates its weights after each completed round using up to six recent observations. Myopic B maximizes immediate expected payoff. White-box planners score candidate B move sequences on copies of A, simulate its updates using the assumed learning rate and objective, and play the first move of the best sequence. Black-box planners estimate A from completed actions. The live GRU planner rolls forward using the most probable predicted A move at each step.
- **Ollama inference** puts completed moves and A's rewards into a text prompt. A history window of zero omits previous rounds. The model weights stay frozen throughout play. The shared parser accepts a response containing exactly one recognized action word, case-insensitively. It flags other responses and scores Rock as the fallback. The adaptive opponent uses up to eight completed A moves, independent of A's prompt window.

### Regret and simulation lifecycle

For sampled A actions $a_t$ and recorded B actions $b_t$, cumulative external regret compares A's score with the best fixed action. Cumulative swap regret lets a fixed replacement rule choose a different destination for each source action.

$$
\begin{aligned}
R_T^{\mathrm{ext}} &= \max_j\sum_{t=1}^{T}u_A(j,b_t)-\sum_{t=1}^{T}u_A(a_t,b_t), \\
R_T^{\mathrm{swap}} &= \sum_i\max_j\sum_{t:a_t=i}\bigl[u_A(j,b_t)-u_A(i,b_t)\bigr].
\end{aligned}
$$

Realized external regret can be negative. Swap regret is nonnegative because identity replacements are allowed, and it is at least as large as external regret because constant replacements are allowed. Divide by $T$ for per-round regret. Dynamic regret uses a time-varying comparator, such as the action that beats B in each phase of a fixed switching schedule.

Each condition initializes its own policy and history. Training and warm-up events appear separately from scored decisions. During scoring, **realized payoff** comes from the sampled moves and **expected payoff** weights the possible outcomes by the pre-action policy, where available. The backend streams events to the playback controls, charts, and inspector. **Stop** requests cancellation, which can take until the current unit of work finishes.

## Architecture and implementation

| Path | Role |
| --- | --- |
| [`src/`](src/) | Python implementations and command-line experiments. NumPy handles game scoring and sampling. PyTorch provides MLP, Transformer, and GRU models. |
| [`backend/app/`](backend/app/) | FastAPI catalog, simulation worker, scoring, event stream, run storage, and exports. The Ollama adapter discovers installed models and calls the local chat API. |
| [`frontend/src/`](frontend/src/) | React and TypeScript app served by Vite, with Recharts for run metrics and comparisons. |
| [`outputs/`](outputs/) | Bundled experiment CSVs and local run output under `outputs/lab_runs/`. |
| [`docs/`](docs/) | Setup, experiment configurations, and API documentation. |

FastAPI uses one background worker to run simulations and stores terminal runs as JSON. The browser consumes server-sent events for training, warm-up, round, and status updates. The [architecture and API guide](docs/architecture.md) describes the endpoints and event schema. The local server also exposes [interactive API documentation](http://127.0.0.1:8000/docs).

## Results and observations

### Regret learning and held-out neural policies

- In a 100-round Hedge example against Paper followed by Rock, A earned `47`. The best fixed action earned `50`, giving external regret `3`. The phase-switching comparator earned `100`, giving dynamic regret `53`.
- The early MLP and Transformer runs reported **88.67%** and **81.25%** held-out next-move accuracy, respectively, on different test sets. The synthetic opponent's stable favourite makes action counts informative. The live prediction comparison uses a shared held-out set.
- A reward-trained Transformer reached **+0.7812 expected reward per decision** on the test set after 500 training steps. Its final training loss was `−0.8867`, consistent with minimizing negative payoff.
- In one held-out swap comparison, adding $\lambda=1$ changed expected reward from `0.5640` to `0.5465` and the surrogate from `0.0791` to `0.0816`. A separate sampled-action run reduced realized swap regret from `0.0867` to `0.0763`, with realized reward decreasing from `0.5523` to `0.5433`. The two runs showed different effects on the surrogate.

### Learning-aware opponents

- **Bait / Exploit / Safe.** Over 30 rounds, a three-step planner against Hedge increased A's probability of X from `0.500` to about `0.924` by round 6 before exploiting it. B averaged `0.693` expected and `0.430` realized payoff per round. Myopic B chose Safe and averaged `0.200` expected payoff. With a small online neural policy, the planner averaged `0.646` expected payoff. With an online Transformer over three matched seeds, it averaged `0.355` at $\lambda=0$ and `0.438` at $\lambda=1$, with the same `0.200` myopic baseline.
- **Frozen-policy RPS.** Myopic and multi-step B each earned `0.9101` expected payoff against the reward-trained Transformer and `0.9590` against the swap-regularized Transformer. A's decisions depended on move history while its weights stayed fixed.
- **White-box RPS.** Across five seeds and 20 rounds per run, the three-step planner earned `0.306` expected B payoff per round against reward-only A, compared with `0.265` myopic and `0.251` for two steps. With swap regularization, myopic earned `0.547`, compared with `0.519` for three steps and `0.429` for two. The best of 15 stationary mixtures on the tuning grid, `[0.25, 0.25, 0.50]`, earned `0.063` for each objective on five held-out seeds.
- **Update assumptions.** In the same five-seed RPS configuration, simulating frozen weights yielded `0.219` and `0.464` expected B payoff for $\lambda=0$ and $\lambda=1$, respectively, below the exact-update planner. Some mismatched assumptions scored higher than the exact model, including the wrong-objective planner at `0.341` and `0.543`. A separate ablation held the next input history fixed. Gradient updates moved A's probabilities by `0.085–0.197` in average total-variation distance across the tested conditions.
- **Black-box RPS.** With 20 exploration rounds and 40 scored rounds over ten seeds, the frequency-based myopic policy earned `0.238` and `0.249` expected B payoff against $\lambda=0$ and $\lambda=1$, respectively. The privileged oracle earned `0.410` and `0.613`. In a separate GRU evaluation on five held-out seeds of 25 rounds each, two-step most-probable-action planning raised expected B payoff from `0.209` to `0.290` against $\lambda=1$. Probability-weighted two- and three-step rollouts gave `0.233` and `0.042` in that condition. The rollouts weight futures under the learned GRU model, so prediction errors can accumulate with depth.

### Paired online neural validation

The archived RPS comparison used ten matched seeds (1001–1010), 20 rounds per run, a learning rate of `0.24`, and a six-observation update buffer. Pairs share initialization, starting history, and A's random draws. Adaptive B can take different actions across the two objectives. The table gives per-round averages, with entropy normalized from 0 (deterministic) to 1 (uniform). [Per-seed data](outputs/neural_swap_results/runs.csv)

| B strategy | $\lambda$ | Expected B payoff | Realized B payoff | A external regret | A swap regret | A policy entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform | 0 | −0.006 | 0.040 | 0.210 | 0.315 | 0.584 |
| Uniform | 1 | 0.021 | 0.040 | 0.210 | 0.300 | 0.383 |
| Myopic | 0 | 0.250 | 0.240 | 0.320 | 0.575 | 0.909 |
| Myopic | 1 | 0.499 | 0.535 | 0.635 | 1.015 | 0.707 |
| Planner-2 | 0 | 0.292 | 0.250 | 0.340 | 0.600 | 0.866 |
| Planner-2 | 1 | 0.491 | 0.480 | 0.610 | 0.960 | 0.677 |

Swap regularization concentrated A's action probabilities and increased both expected B payoff and A's realized swap regret against the two adaptive opponents. Against uniform B, realized payoff was unchanged and the estimated changes in expected B payoff and A swap regret were small. The table reports paired changes for $\lambda=1$ minus $\lambda=0$, with 95% percentile intervals from 3,000 seed-level bootstrap resamples.

| B strategy | Change in expected B payoff | Change in A realized swap regret | Change in A policy entropy |
| --- | ---: | ---: | ---: |
| Uniform | +0.027 [−0.009, +0.063] | −0.015 [−0.145, +0.115] | −0.202 [−0.250, −0.160] |
| Myopic | +0.248 [+0.206, +0.290] | +0.440 [+0.220, +0.690] | −0.202 [−0.239, −0.168] |
| Planner-2 | +0.199 [+0.152, +0.244] | +0.360 [+0.150, +0.565] | −0.189 [−0.229, −0.150] |

The Planner-2 minus myopic difference in expected B payoff was `+0.042 [−0.001, +0.084]` at $\lambda=0$ and `−0.008 [−0.057, +0.037]` at $\lambda=1$. Both intervals include zero.

### Local LLM play

In an early 12-round `llama3.2` example, adaptive B averaged `+0.917` per round with history hidden, when A played Rock every round, and `+0.333` with history shown. In a separate 18-round comparison across three seeds, fixed periodic baiting averaged `−0.278`, reactive baiting `+0.167`, and the adaptive action-frequency response `+0.444`.

The archived comparison used `llama3.1:8b` and `qwen3.5:latest`, two prefixes per model, three shared warm-up rounds per prefix, and five B strategies with 12 scored rounds each. It used temperature `0`, sent up to eight prior rounds, and requested thinking to be disabled. The archive contains 20 episodes and 240 scored rounds with zero invalid responses. The full run used 252 model calls, including 12 for warm-up. Later Paper frequency uses scored rounds 5–12. [Episode data](outputs/ollama_rps_results/episodes.csv) · [Round data](outputs/ollama_rps_results/rounds.csv)

| B strategy | Llama B payoff/round | Llama later Paper | Qwen B payoff/round | Qwen later Paper |
| --- | ---: | ---: | ---: | ---: |
| Uniform | −0.292 | 37.5% | +0.250 | 12.5% |
| Always Scissors | −0.250 | 25.0% | 0.000 | 6.25% |
| Rock ×2 → Scissors | −0.208 | 25.0% | +0.125 | 0.0% |
| Rock ×4 → Scissors | −0.083 | 37.5% | +0.292 | 12.5% |
| Adaptive frequency response | −0.375 | 31.25% | +0.542 | 31.25% |

Compared with Always Scissors, four rounds of Rock improved B's payoff against Llama by `0.167` per round and increased later Paper frequency by `12.5` percentage points. The game still averaged a loss for B. Against Qwen, four rounds of Rock improved B payoff by `0.292` per round and later Paper frequency by `6.25` points. B already averaged `+0.750` during those Rock rounds and earned `+0.0625` during the Scissors phase. The higher Qwen score was largely earned during the bait phase itself.

## Running experiments and tests

### Command-line experiments

Run these commands from the repository root after installing the backend dependencies. The two archive reproduction commands replace the bundled CSVs. Use a separate checkout to retain both the bundled and newly generated results. The command-line experiments also include the stationary-mixture benchmark, frozen-policy planning, and probability-weighted GRU rollouts.

```bash
# Re-run the archived Ollama configuration with both models installed
MODELS=llama3.1:8b,qwen3.5:latest PREFIXES=2 WARMUP=3 ROUNDS=12 \
  .venv/bin/python -m src.llm.controlled_ollama_rps

# Re-run the archived online neural comparison
BASE_SEED=1001 SEEDS=10 STRATEGIES=uniform,myopic,plan2 \
  .venv/bin/python -m src.neural.neural_swap_validation

# Run the stationary-mixture benchmark
.venv/bin/python - <<'PY'
from src.neural.online_transformer_rps_stationary_benchmark import online_transformer_rps_stationary_benchmark

online_transformer_rps_stationary_benchmark()
PY
```

The neural validation also runs via `.venv/bin/python main.py`. Set `SEEDS=20` for a larger comparison. In the web app, select ten seed pairs, first seed 1001, 20 rounds, and all strategies in **Paired Neural Swap Validation** to match the archive configuration. [Command-line entry point](main.py) · [Validation script](src/neural/neural_swap_validation.py) · [Controlled Ollama script](src/llm/controlled_ollama_rps.py)

| Experiment | Implementation |
| --- | --- |
| Hedge and SwapMW self-play | [Compare learners](src/regret_learning/compare_learners.py) |
| Prediction, reward training, and the swap objective | [MLP](src/neural/simple_opponent_predictor.py) · [Tiny Transformer](src/neural/tiny_transformer.py) · [Reward training](src/neural/regret_train_eval_predict.py) · [Swap training](src/neural/swap_regret_transformer.py) |
| Bait / Exploit / Safe | [Hedge](src/neural/learning_aware_opponent.py) · [Online neural policy](src/neural/online_neural_learning.py) · [Online Transformer](src/neural/online_transformer_planner.py) |
| RPS planning, mismatch, and baselines | [Frozen-policy planning](src/neural/strategic_swap_regret_transformer.py) · [Online planning](src/neural/strategic_online_transformer_planner.py) · [Horizons](src/neural/horizon_strategic_online_transformer_planner.py) · [Model mismatch](src/neural/online_transformer_rps_model_mismatch.py) · [Update ablation](src/neural/rps_learning_update_ablation.py) · [Stationary benchmark](src/neural/online_transformer_rps_stationary_benchmark.py) |
| Black-box RPS | [Frequency model](src/neural/blackbox_opponent.py) · [GRU and most-probable-action rollouts](src/neural/rps_blackbox_dynamics.py) · [Probability-weighted rollouts](src/neural/rps_blackbox_expected_rollouts.py) |
| Early Ollama comparisons | [History baseline](src/llm/rps_ollama_baseline.py) · [Fixed baiting](src/llm/rps_ollama_bait_switch.py) · [Reactive baiting](src/llm/rps_ollama_adaptive_bait.py) · [Always Scissors control](src/llm/rps_ollama_bait_causality.py) |

### Tests and build

```bash
.venv/bin/python -m pip install -r backend/requirements-dev.txt
.venv/bin/python -m pytest -q backend/tests tests
npm --prefix frontend test
npm --prefix frontend run build
```

Tests mock Ollama responses. The frontend build is written to `frontend/dist/`. [Development and testing guide](docs/development.md)

### Archives and exports

| File | Contents |
| --- | --- |
| [Ollama episodes](outputs/ollama_rps_results/episodes.csv) | 20 episode summaries by model, prefix, and B strategy |
| [Ollama rounds](outputs/ollama_rps_results/rounds.csv) | 240 scored rounds with actions, phases, rewards, and raw responses |
| [Neural validation runs](outputs/neural_swap_results/runs.csv) | 60 per-seed summaries with payoff, regret, and entropy |

Live runs save JSON under `outputs/lab_runs/`. Download a scored-round CSV or a full JSON export from the run page. The archived neural validation contains summaries only. Run it in the lab to inspect new round traces.

## Limitations

- The differentiable swap penalty is a finite-window, expected-occupancy surrogate with no formal no-swap-regret guarantee. Its value and the regret measured from sampled actions can move in different directions.
- Hindsight regret holds the recorded B actions fixed. A reactive B could follow a different trajectory if A's earlier actions changed.
- Planning performance depends on the game, the horizon, and the quality and visibility of B's model of A. Individual runs and the tested seed sets support comparisons under their particular configurations.
- Ollama is inference-only. The archived comparisons cover short episodes from two warm-up prefixes per model. Separate model histories, prompt sensitivity, and existing action biases limit what later Paper frequencies establish about baiting.
