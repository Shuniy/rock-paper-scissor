# Experiment catalog

Open a category in the lab to run these experiments. Each experiment includes a component diagram, configuration inputs, algorithm, equations, worked example, and guidance for interpreting the actual results.

| ID | Experiment | Live conditions and what to inspect |
|---|---|---|
| R1 | External Regret / Hedge | Fixed, uniform, switching, or Hedge B. Inspect counterfactual payoffs and multiplicative weights. |
| R2 | Swap-Regret Learner | The same B choices. Inspect three row learners, the transition matrix, and its stationary distribution. |
| R3 | Hedge vs. SwapMW | Matched draws and length. A learning B can generate different opponent paths across branches. |
| N1 | Neural Opponent Prediction | Train an MLP on noisy-favourite data. Inspect predicted B probabilities and A’s best response. |
| N2 | Transformer Prediction | Train the Transformer alone or compare it with the MLP on an identical held-out set. |
| N3 | Reward-Trained Transformer | Train A directly on differentiable expected payoff. The fitted model stays frozen during evaluation. |
| N4 | Swap-Regularized Transformer | Compare λ=0 with a configurable swap penalty. Choose one held-out opponent type and compare expected-occupancy and sampled regret metrics. |
| S1 | Bait / Exploit / Safe | Myopic versus planning B in the constructed game. Compare phase choices, changes in P(X), and B payoffs. |
| S2 | Neural Learning-Aware Opponent | A small online network in the constructed game. B plans on model copies. |
| S3 | Planning Horizon | Compare horizons 1–3 in online RPS with myopic and uniform baselines available. |
| S4 | Model Mismatch | Exact, half-rate, frozen approximation, wrong objective, and myopic conditions. |
| S5 | Black-Box Opponent | Uniform, global frequency heuristic, contextual counts, GRU prediction, planning, and a privileged oracle. The light GRU demonstration trains on disjoint simulated trajectories. |
| S6 | Paired Neural Swap Validation | Match initial seeds for λ=0 and λ=1. Compute per-seed metrics and paired bootstraps. The live default is three seeds. Choose ten seeds, first seed 1001, 20 rounds, and all strategies to match the archived protocol. |
| L1 | LLM Baseline | Run an installed Ollama model in repeated RPS. Inspect the prompt window, response, and invalid flag. |
| L2 | History vs. No History | Compare prompts with past completed rounds against a blank history while keeping decoding settings fixed. |
| L3 | Strategic Baiting | Compare phase metrics, later Paper frequency, entropy, and invalid output counts across choices of B. |
| L4 | Controlled Comparisons | Run five opponent conditions per matched prefix across one or two installed models. Each model and prefix receives its own shared warm-up. |

Ollama uses a frozen model. The controlled research parser accepts a single recognized action word. It marks other responses invalid and scores Rock as a fallback. A history window of zero sends no completed rounds to A. The adaptive B policy uses up to eight completed A moves, independent of A’s prompt setting.

## Recorded results

| Dataset | Recorded detail | Source |
|---|---|---|
| Controlled Ollama | 20 evaluated episodes and 240 evaluated rounds. Explore each model, prefix, strategy, phase, and raw response. | `outputs/ollama_rps_results/episodes.csv` and `rounds.csv` |
| Paired neural validation | 60 per-seed runs. Inspect strategy, λ, expected B reward, realized B reward, A’s regret, entropy, and 95% paired bootstrap intervals. | `outputs/neural_swap_results/runs.csv` |

The original neural validation archive has summary metrics only. Live validation can produce new round traces. Early reported examples and ablations that lack CSVs are described in the [research record](../README.md).

The original command-line research scripts also cover the stationary-mixture grid search, the frozen-policy planning comparison, and probability-weighted GRU rollouts. Their reported tables are in the research README. The live black-box adapter uses the frequency and GRU MAP constructions, and the white-box inspector shows an individual round’s same-history gradient ablation.

## Read the metrics

RPS payoffs sum to zero and range from −1 to +1 per agent. The constructed Bait / Exploit / Safe game has its own two payoff matrices and its own scale. External regret compares the sampled sequence with one fixed A action. Swap regret allows a separate replacement for each source action. Both comparators hold the observed B sequence fixed. An adaptive B could behave differently if A’s earlier actual moves changed.

Expected reward weights payoffs by the pre-action distribution. Realized reward uses the sampled action. A plan’s predicted return can span several future decisions. The differentiable swap surrogate weights counterfactual improvements by expected action occupancy. Measured swap regret instead uses the actions actually sampled.

The findings preserve the exploratory limits of the research. The constructed game demonstrated a learning-aware mechanism. Neural planning gains were conditional, and the final tested swap penalty increased exploitability under adaptive opponents. Ollama showed history sensitivity and action biases. The experiments did not establish a general causal and profitable baiting mechanism or train a pretrained LLM with a swap objective.
