# Learning-Aware Strategies and Swap-Regret Objectives for Neural and LLM Agents

> **A hands-on preliminary research project in game theory, online learning, neural networks, Transformers, strategic planning, and black-box LLM agents.**
>
> **Study completed:** 21 November 2026  
> **Status:** Exploratory prototype / educational research record — **not** a peer-reviewed study or a demonstrated theoretical guarantee.  
> **Primary environment:** Python, NumPy, PyTorch, synthetic games, and local Ollama models.

## At a glance

We investigated a repeated-game setting in which:

- **Agent A (the learner)** chooses actions and may adapt using Hedge, a neural-network update, a Transformer update, or the history in an LLM prompt.
- **Agent B (the strategist)** chooses actions to maximize its own reward. B may act myopically, plan against predicted future learning, or infer A's behaviour from black-box observations.
- **Research hypothesis:** Can B gain by influencing what A learns, and can a swap-regret-based training objective make A less exploitable?

**Main outcome:** We demonstrated intentional manipulation in a *constructed* Bait/Exploit online-learning game; built reward- and swap-regularized Transformers; tested white-box and black-box planning in rock–paper–scissors (RPS); and ran controlled games with locally hosted LLMs. We **did not** establish reliable planning superiority in RPS, causal bait-induced LLM manipulation, or a robust protective effect from the implemented swap-regret surrogate. In the final online-neural validation, swap regularization actually **increased** exploitability under the tested adaptive opponents.

### Scope and evidence

| Component | What was achieved | What was **not** established |
|---|---|---|
| Regret learning | Implemented and understood Hedge, external regret, swap regret, and a transition-matrix swap learner | A formal no-regret proof for our implementations |
| Project 1: neural/swap objectives | Trained MLP and Transformer policies; optimized a differentiable swap-gain surrogate | A proven no-swap-regret learner, or swap-regret fine-tuning of a pretrained LLM |
| Project 2: learning-aware strategies | Demonstrated planning over online updates, model-mismatch tests, black-box prediction and RPS exploration | Consistent superiority of planning over myopic exploitation in RPS |
| Pretrained LLM experiments | Tested history-conditioned RPS play via Ollama and controlled baiting comparisons | A causally identified, profitable induced LLM preference change |

---

## Table of contents

1. [Research motivation and questions](#1-research-motivation-and-questions)
2. [Foundations: repeated games and regret](#2-foundations-repeated-games-and-regret)
3. [Project 1: neural learning and swap-regret objectives](#3-project-1-neural-learning-and-swap-regret-objectives)
4. [Project 2: learning-aware strategic opponents](#4-project-2-learning-aware-strategic-opponents)
5. [Experiments with pretrained LLMs](#5-experiments-with-pretrained-llms)
6. [Final controlled evaluations](#6-final-controlled-evaluations)
7. [What the project answered](#7-what-the-project-answered)
8. [Limitations and research interpretation](#8-limitations-and-research-interpretation)
9. [Reproducing and navigating the code](#9-reproducing-and-navigating-the-code)
10. [Complete questions and answers](#10-complete-questions-and-answers)

---

## 1. Research motivation and questions

### Original idea

Imagine two agents playing repeatedly:

- A learns that a particular action is profitable, and begins playing it more often.
- B realizes that **its own actions influence A's future learning**.
- Instead of maximizing this round's payoff, B might temporarily sacrifice reward to make A prefer a target action, then exploit that preference later.

For example, B might repeatedly play Rock to encourage A to play Paper, then switch to Scissors. The strategic question is not just *“What beats A now?”*, but *“What should B do now to influence what A will do later?”*

We used a small project, rather than aiming to solve the full open research problem, to study:

1. **Learning:** How do classical regret learners and neural/Transformer agents change their strategies?
2. **Manipulation:** Can a strategic opponent profit from anticipating those changes?
3. **Swap regret:** Does penalizing profitable action replacements affect the learner's performance or vulnerability?
4. **Black-box settings:** What remains possible when B cannot inspect A's weights or update rule?
5. **LLMs:** Do analogous effects appear when a pretrained LLM chooses actions from natural-language game history?

### Core distinctions

| Term | Meaning in this project |
|---|---|
| **External regret** | Hindsight advantage of the best *single fixed action* over the learner's actual decisions |
| **Swap regret** | Hindsight advantage of replacing each chosen action according to a fixed mapping from source actions to replacement actions |
| **Myopic B** | Chooses the action with greatest immediate expected reward |
| **Learning-aware B** | Accounts for how today's action changes A's future policy, including its online learning updates when modelled |
| **White-box B** | Can inspect A's policy and, in planner experiments, simulate the learner's update rule |
| **Black-box B** | Uses public observations (past actions and sometimes rewards), not A's parameters or gradients |
| **Parameter learning** | Gradient-based changes to model weights during training or online gameplay |
| **In-context adaptation** | Different decisions from the *same frozen model* because its prompt/history changes |
| **Swap-regret surrogate** | Our differentiable training penalty; not equivalent to a proved no-swap-regret guarantee |

---

## 2. Foundations: repeated games and regret

### Step F1 — Define the RPS environment

We used actions indexed as `Rock=0`, `Paper=1`, `Scissors=2` and the payoff matrix for A:

```python
PAYOFF_A = [
    [ 0, -1, +1],  # A chooses Rock
    [+1,  0, -1],  # A chooses Paper
    [-1, +1,  0],  # A chooses Scissors
]
```

Rows index A's action, columns index B's action. A wins `+1`, loses `−1`, or draws `0`.

RPS is **zero-sum**: `u_B(b,a) = −u_A(a,b)`. To give B a matrix with its *own* actions as rows, we used:

```python
payoff_matrix_b = -payoff_matrix_a.T
```

This corrects an important indexing/sign issue in the original two-player regret implementation. In standard RPS the two matrices happen to be numerically identical because the matrix is antisymmetric; their *index meanings* are different.

**Questions answered:** What is a payoff matrix? Why use `−A.T` for B? Why must A's and B's scores sum to zero? How do probabilities form a mixed strategy?

### Step F2 — Hedge / multiplicative weights

Hedge keeps a weight for each action, converts weights to a probability distribution, and updates them after receiving full-information rewards:

$$
 w_{t+1}(i)=w_t(i)\exp(\eta r_t(i)), \qquad
 \pi_t(i)=\frac{w_t(i)}{\sum_j w_t(j)}.
$$

Against a fixed Paper-playing opponent, for example, Scissors becomes more likely because it earns `+1` against Paper.

We also considered a switching opponent: Paper for the first half, then Rock. This illustrated that *the best fixed action in hindsight* is a different comparison from *the best action allowed to switch over time* (dynamic regret). In an illustrative 100-round Hedge run, the learner earned `47`, the best fixed action earned `50` (external regret `3`), and an ideal phase-switching action sequence earned `100` (dynamic regret `53`). These are different comparator classes.

**Questions answered:** How does Hedge learn without gradients? Why can a low external regret coexist with high dynamic regret? What does the learning rate change?

### Step F3 — External regret

For A choosing `a_t` against observed B actions `b_t`, cumulative external regret is:

$$
 R_T^{\mathrm{ext}}
 = \max_{j}\sum_{t=1}^{T}u_A(j,b_t)
   - \sum_{t=1}^{T}u_A(a_t,b_t).
$$

We can also report `R_T / T` as average regret per round.

**Important nuance:** *Realized* external regret can be negative: a history-dependent or randomly sampled sequence can outperform every fixed action in hindsight. That is not a calculation error.

### Step F4 — Swap regret and the transition-matrix learner

A swap mapping `φ` specifies which action to substitute whenever A originally chooses each source action. Its hindsight advantage is:

$$
 R_T^{\mathrm{swap}}
 = \max_{\phi}\sum_{t=1}^{T}
    \bigl[u_A(\phi(a_t),b_t)-u_A(a_t,b_t)\bigr].
$$

For three RPS actions, we can calculate it as:

$$
 R_T^{\mathrm{swap}}
 = \sum_{i\in\{R,P,S\}}
    \max_j\sum_{t:a_t=i}
    \bigl[u_A(j,b_t)-u_A(i,b_t)\bigr].
$$

The **identity replacement** is always allowed, so realized swap regret is nonnegative and at least as large as realized external regret.

We implemented a Blum–Mansour-style construction with three row learners. Each row has a probability distribution over action replacements; together they form a `3 × 3` transition matrix `P`. The played strategy is a stationary distribution satisfying `πP = π`.

An illustrative fixed-Paper experiment moved A's distribution toward Scissors; a separate 100-round example reported both external and swap regret of `21`, because the optimal substitutions effectively pointed toward the same winning action. Equality in one example **does not** imply the two regrets are generally identical.

**Questions answered:** How is swap regret different from external regret? What is a transition matrix? What is its stationary distribution? Why can individual row learners have different weights?

### Step F5 — Compare Hedge vs. SwapMW, with both agents learning

We corrected the player-B payoff indexing and then inspected a 100-round comparison:

| Algorithm | Player | Actual reward | Best fixed reward | External regret | Swap regret |
|---|---|---:|---:|---:|---:|
| Hedge | A | +4 | +5 | 1 | Not recorded in this run |
| Hedge | B | −4 | +9 | 13 | Not recorded in this run |
| SwapMW | A | −1 | +6 | 7 | 13 |
| SwapMW | B | +1 | +7 | 6 | 13 |

We also inspected Hedge's final action weights, SwapMW's row log-weights, transition matrices, and stationary action probabilities. The stationary distributions were approximately `π_A=[0.402, 0.280, 0.317]` and `π_B=[0.413, 0.319, 0.268]` in that SwapMW run.

**Observation:** Both algorithms generated valid zero-sum payoffs. Neither can be declared superior from these runs: Hedge played against Hedge while SwapMW played against SwapMW, creating **different opponent trajectories**. Low finite-horizon regret in an individual run is not a proof of asymptotic no-regret.

**Code artifact:** `compare_learners_payoffs.py`, with changes documented in `compare_learners_payoffs.patch`. This class relies on the original project's `src/` modules.

---

## 3. Project 1: neural learning and swap-regret objectives

**Goal:** Build an agent that learns from examples, then from game rewards, and finally from a combined reward-and-swap-regret objective.

### Step P1.1 — MLP predicts an opponent's next move

We generated synthetic opponents with a preferred RPS action. The generator selected the favourite `85%` of the time and otherwise sampled randomly; therefore the favourite was played with effective probability `0.85 + 0.15/3 = 0.90`.

Architecture: `5 previous actions → learned 8-D token embeddings → flatten (40 features) → MLP → three action logits`.

| Observed outcome | Value |
|---|---:|
| Final reported training loss | `0.3319` |
| Held-out prediction accuracy | **88.67%** |
| Random uniform-guess baseline | `33.33%` |

For five Rocks, the model predicted B's `[Rock, Paper, Scissors]` probabilities as approximately `[0.893, 0.048, 0.059]`; A's correct best response was **Paper**. For five Papers, A selected Scissors; for five Scissors, A selected Rock.

**Answer:** The MLP learned correlations in synthetic action histories through backpropagation. It did not yet learn strategy directly from a reward function.

### Step P1.2 — Replace the MLP with a tiny Transformer

Architecture: token embeddings + learned position embeddings + a causal single-layer Transformer encoder (four attention heads) + prediction head.

| Model | Reported final training loss | Reported held-out accuracy |
|---|---:|---:|
| MLP | 0.3319 | **88.67%** |
| Tiny Transformer | 0.2582 | 81.25% |

The Transformer still selected the correct counteraction after five identical opponent actions. Its lower *reported training-batch* loss did not imply higher test accuracy. The runs did not use an identical held-out dataset and were not a controlled architecture benchmark.

**Concept learned:** Token embeddings represent *what* action was played; positional embeddings represent *where* it occurred; self-attention learns how to combine information from different positions. Positional embeddings do not automatically force recent actions to matter more.

**Why attention had little evident benefit:** The original synthetic opponent's favourite did not depend on action order. A bag of action counts is already highly informative.

### Step P1.3 — Train the Transformer directly for reward

We changed the output interpretation: it now represented **A's mixed strategy**, not predictions of B's next action. Using the full-information RPS payoff for B's observed action, we minimized negative expected reward:

$$
 \mathcal{L}_{\mathrm{reward}}
 =-\mathbb{E}_{a\sim\pi_\theta(\cdot\mid h)}[u_A(a,b)].
$$

The reported 500-step training run reached:

| Metric | Value |
|---|---:|
| Final training loss | `−0.8867` |
| Test expected reward per decision | **`+0.7812`** |

The policy became nearly deterministic: after repeated Rock, it chose Paper; after repeated Paper, Scissors; after repeated Scissors, Rock. Negative training loss is valid because the objective is **negative reward**.

**Question answered:** Does the neural model need to predict B explicitly? **No.** It can be optimized directly to select high-reward A actions using differentiable expected payoff.

### Step P1.4 — Add a differentiable swap-regret surrogate

We generated short episodes (`T=30`) containing fixed-preference, switching, cycling, and random opponents. At each round the Transformer saw only B's previous `H=5` actions, and output A's action distribution.

For a source action `i` and destination action `j` we used the expected-occupancy improvement:

$$
 \widehat R_{ij}
 =\frac{1}{T}\sum_t\pi_\theta(i\mid h_t)
   [u_A(j,b_t)-u_A(i,b_t)],
 \qquad
 \mathcal{L}_{\mathrm{swap}}=\sum_i\max_j\widehat R_{ij}.
$$

The combined loss was:

$$
 \mathcal{L}_{\mathrm{total}}
 =-\mathbb{E}[\mathrm{reward}]
  +\lambda\mathcal{L}_{\mathrm{swap}}.
$$

We compared `λ=0` (reward-only) with `λ=1` (reward plus swap surrogate).

**First exploratory evaluation:**

| Model | Expected reward/round | Swap surrogate/round |
|---|---:|---:|
| `λ=0` | **0.5640** | **0.0791** |
| `λ=1` | 0.5465 | 0.0816 |

In this run, adding the penalty did **not** improve held-out results.

**Separate subsequent run with sampled-action evaluation:**

| Model | Expected reward | Realized reward | Swap surrogate | Realized swap regret | Realized external regret |
|---|---:|---:|---:|---:|---:|
| `λ=0` | **0.5523** | **0.5523** | 0.0851 | 0.0867 | −0.1827 |
| `λ=1` | 0.5429 | 0.5433 | **0.0725** | **0.0763** | −0.1737 |

Here the regularized model reduced sampled-action swap regret by about **12%**, with about **1.6%** less realized reward. These are *separate experimental runs*, so their raw metrics should not be combined as though measured in a single trial.

**Answer:** The swap objective was differentiable and sometimes lowered regret on a synthetic test set. It was **not** a demonstrated no-swap-regret guarantee, and the improvement was not consistent across experiments.

### Project 1 takeaway

We built the intended learning pipeline:

`Hedge/SwapMW → MLP → causal Transformer → reward-trained policy → differentiable swap-regularized policy → sampled-action regret evaluation`.

The distinction between **training loss**, **expected swap surrogate**, and **realized swap regret** was central. A lower training surrogate does not automatically imply lower regret against an adaptive adversary.

---

## 4. Project 2: learning-aware strategic opponents

**Goal:** Determine when B benefits from anticipating, influencing, or inferring A's future learning and whether swap regularization affects vulnerability.

### Step P2.1 — The constructed Bait / Exploit / Safe game

To make manipulation unambiguous, we first used two actions for A (`X`, `Y`) and three for B (`Bait`, `Exploit`, `Safe`). This was a **constructed general-sum game**, not ordinary RPS.

| A action | A payoff: Bait | A payoff: Exploit | A payoff: Safe |
|---|---:|---:|---:|
| X | +1 | −1 | 0 |
| Y | 0 | +1 | 0 |

| A action | B payoff: Bait | B payoff: Exploit | B payoff: Safe |
|---|---:|---:|---:|
| X | −0.1 | +3 | +0.2 |
| Y | −0.1 | −3 | +0.2 |

- Myopic B chose Safe forever and earned **0.200** expected reward per round.
- A three-step planner initially chose Bait, increasing A's preference for X from `0.500` to approximately `0.924` by round 6, then switched to Exploit.
- The Hedge-based planner earned **0.693 expected** and **0.430 realized** B reward per round over 30 rounds.

**Observed mechanism:** B intentionally paid an immediate cost to alter A's future policy, then benefited by exploiting that change. This is the project's clearest positive demonstration of **learning-aware manipulation**; the game was expressly designed to make it possible.

**Code:** `learning_aware_opponent_step2.py`.

### Step P2.2 — Replace Hedge with an online neural learner and Transformer

We replaced Hedge updates with neural gradient descent. B simulated future updates on a **copy** of A's network so its planning did not accidentally train the real agent.

| Toy-game A learner | B: Myopic expected | B: Planner expected | Planner realized (example run) |
|---|---:|---:|---:|
| Small online neural policy | 0.200 | **0.646** | 0.733 |
| Online Transformer, `λ=0` | 0.200 | **0.355** | 0.405 |
| Online Transformer, `λ=1` | 0.200 | **0.438** | 0.505 |

The online Transformer experiment used a four-move history, a six-observation training buffer, a learning rate of `0.24`, 20-round games, and a two-round planning horizon; values in this table for that experiment are means over three matched seeds. The small neural-policy result was a separate run.

**Answer:** Anticipating gradient updates can provide strategic benefit in a favourable constructed game. It does **not** prove this advantage in general-sum games or RPS.

**Code:** `online_transformer_planner.py`.

### Step P2.3 — Frozen Transformer versus immediate and multi-step planning

Before enabling online weight changes in RPS, we tested a planner against frozen Transformers that adjusted only through B-action history.

| A model | Myopic B expected | Planner B expected |
|---|---:|---:|
| Reward-trained Transformer | 0.9101 | 0.9101 |
| Swap-regularized Transformer | 0.9590 | 0.9590 |

Both opponents heavily exploited nearly deterministic A policies. The planner achieved **no additional measured advantage** over myopic play. Since A's weights were frozen, this was a test of **history-conditioned in-context behaviour**, not online neural learning.

### Step P2.4 — Online Transformer in zero-sum RPS

We returned to three-action RPS. After each round A updated its neural weights using the last six observations. The white-box planner simulated those updates while considering future B actions.

A preliminary three-seed, 20-round comparison produced:

| A objective | Myopic B expected | Two-step planner B expected |
|---|---:|---:|
| Reward-only | 0.265 | 0.270 |
| Reward + swap | **0.544** | 0.443 |

**Answer:** Online neural adaptation is not enough to make a longer-horizon planner universally preferable. The short-horizon planner was only marginally better against reward-only A and worse against swap-regularized A.

**Code:** `online_transformer_rps_planner.py`.

### Step P2.5 — Compare horizons and simple baselines

We compared fixed RPS commitments, uniform play, myopic play, and two- or three-step planning across **five seeds, 20 rounds**.

| B strategy | B expected vs `λ=0` | B expected vs `λ=1` |
|---|---:|---:|
| Uniform | 0.026 | 0.080 |
| Myopic | 0.265 | **0.547** |
| Planner-2 | 0.251 | 0.429 |
| Planner-3 | **0.306** | 0.519 |

Fixed pure RPS actions were heavily punished (B reward roughly `−0.86` to `−0.92`), as A learned to counter them. Planner-3 beat myopic against `λ=0`, but not against `λ=1`. Deeper planning did not produce monotonic gains.

The swap-regularized learner often had more concentrated action probabilities (e.g. normalized entropy `0.656` vs `0.911` against myopic B), which may partly explain exploitability; that mechanism was **not causally isolated**.

**Code:** `online_transformer_rps_horizon_compare.py`.

### Step P2.6 — Model mismatch: what if B is wrong about A?

A's real update rate stayed at `0.24`; B's planner was told an exact, half, double, or zero weight-update rate, or an incorrect swap objective.

| B's assumption | B expected vs `λ=0` | B expected vs `λ=1` |
|---|---:|---:|
| Myopic | 0.265 | **0.547** |
| Exact update model | 0.306 | 0.519 |
| Half learning rate | 0.318 | 0.544 |
| Double learning rate | 0.334 | 0.504 |
| **No weight updates** | 0.219 | 0.464 |
| Wrong learning objective | **0.341** | 0.543 |

The “no weight updates” planner performed worse than the exact planner in both settings, suggesting that forecasting weight changes is useful. But *inaccurate* models sometimes beat the exact model over a longer episode because short-horizon planning need not optimize the full game.

**Code:** `online_transformer_rps_model_mismatch.py`.

### Step P2.7 — Ablation: isolate the effect of gradient updates

We held the **next input history fixed** and compared A's action probabilities with versus without the latest gradient update.

| A objective | B opponent | Average policy shift (TV) | Next best response changed |
|---|---|---:|---:|
| `λ=0` | Myopic | 0.092 | 35% |
| `λ=0` | Planner-3 | 0.085 | 15% |
| `λ=1` | Myopic | 0.197 | 30% |
| `λ=1` | Planner-3 | 0.157 | 25% |

For example, in the first `λ=1` planner round, A's same-next-history policy changed from `[0.333, 0.333, 0.333]` to `[0.311, 0.493, 0.196]` because of the gradient update (`TV≈0.160`).

**Answer:** Neural weight updates matter independently of history changes. This does not by itself show that B deliberately selected the action *because of* that effect.

**Code:** `rps_learning_update_ablation.py`.

### Step P2.8 — Stationary mixed-strategy benchmark

We searched **15** stationary mixtures on a coarse grid (`0.25` increments), chose a mixture on five tuning seeds, and evaluated it on five held-out seeds. The selected mixture was `[Rock=.25, Paper=.25, Scissors=.50]` for both objectives.

| B strategy | Expected B reward vs `λ=0` | Expected B reward vs `λ=1` |
|---|---:|---:|
| Uniform stationary | 0.054 | 0.104 |
| Selected stationary mixture | 0.063 | 0.063 |
| Myopic | 0.265 | **0.547** |
| Planner-3 | **0.306** | 0.519 |

**Answer:** The adaptive opponents outperformed the stationary candidates tested. This supports the value of **adaptivity**, not necessarily the incremental value of **learning-aware planning**. Our coarse-grid mixture is not the globally optimal stationary commitment.

**Code:** `online_transformer_rps_stationary_benchmark.py`.

### Step P2.9 — Black-box opponent: learn from observed actions

We removed B's access to A's current probabilities, weights, gradients, and learning rate. The first black-box predictor estimated A's action distribution from observed histories; evaluation included 20 exploration rounds and 40 additional rounds over 10 seeds.

| B strategy | B expected vs `λ=0` | B expected vs `λ=1` |
|---|---:|---:|
| Uniform | −0.028 | −0.018 |
| Black-box myopic | **0.238** | **0.249** |
| Black-box Planner-2 | 0.222 | 0.242 |
| Black-box Planner-3 | 0.220 | 0.244 |
| Oracle myopic (sees actual A policy) | 0.410 | 0.613 |

Black-box opponents earned positive reward but remained well below a privileged oracle. Prediction total-variation errors were substantial (roughly `0.4` for the adaptive black-box methods). A longer planning horizon did not help.

**Code:** `rps_blackbox_opponent.py`.

### Step P2.10 — Black-box GRU dynamics model

We trained an action-pair **GRU** from 64 black-box game trajectories (25 training epochs). The model saw observed A/B action pairs, **not** the real A action probability vectors.

| Training epoch | Observed-action negative log-likelihood |
|---|---:|
| 1 | 1.064 |
| 10 | 0.723 |
| 20 | 0.655 |
| 25 | **0.622** |

A lower training NLL shows better fit to training observations; it is **not** a guarantee of held-out calibration.

On five held-out seeds (25 rounds each):

| B strategy | Expected vs `λ=0` | Expected vs `λ=1` |
|---|---:|---:|
| Uniform | 0.029 | 0.034 |
| GRU myopic | **0.176** | 0.209 |
| GRU MAP Planner-2 | 0.145 | **0.290** |
| GRU MAP Planner-3 | 0.122 | 0.232 |
| Oracle myopic | 0.237 | 0.520 |

**Observation:** Two-step planning helped against `λ=1` in this small study, but deeper planning hurt and the learned model was still much weaker than the oracle.

**Code:** `rps_blackbox_dynamics.py`.

### Step P2.11 — MAP versus probability-weighted rollouts

The initial GRU planner predicted only the *most likely* future A action (MAP). We then enumerated future outcomes and weighted them by their learned probabilities. This expectation is exact under the GRU's approximation for the finite enumerated open-loop plans, **not** under A's true dynamics.

| B strategy | Expected B reward vs `λ=0` | Expected B reward vs `λ=1` |
|---|---:|---:|
| GRU myopic | **0.176** | 0.209 |
| MAP Planner-2 | 0.145 | **0.290** |
| Expected Planner-2 | 0.141 | 0.233 |
| MAP Planner-3 | 0.122 | 0.232 |
| Expected Planner-3 | 0.052 | 0.042 |

**Answer:** More accurate *integration under a misspecified model* does not ensure better performance against the true learner. Rollout model errors can compound with planning depth.

**Code:** `rps_blackbox_expected_rollouts.py`.

### Project 2 (neural) takeaway

We demonstrated a deliberate Bait/Exploit mechanism, then separated **immediate exploitation**, **history-aware prediction**, **white-box anticipation of weight updates**, and **black-box inferred behaviour**. The incremental benefit of planning was highly environment- and model-dependent. In RPS, more planning frequently failed to beat myopic play.

---

## 5. Experiments with pretrained LLMs

**Goal:** Test whether similar strategic patterns occur when A is a real pretrained, locally hosted language model making RPS decisions from text prompts.

**Critical distinction:** In all Ollama experiments, the model's pretrained weights were **frozen**. A could change its answer as the prompt history changed, but it did **not** perform online gradient updates. LLM “learning” in this part means **in-context behavioural adaptation only**.

### Step L1 — Ollama RPS baseline (reported model `llama3.2`)

The first local Ollama run used a model reported by the script as `llama3.2` (not the later controlled `llama3.1:8b`). Twelve-round comparisons were:

| B condition | B realized reward/round | A behaviour |
|---|---:|---|
| Uniform, history shown | −0.083 | A frequently chose Paper |
| Adaptive, history shown | +0.333 | B exploited repeated Paper; A eventually changed actions |
| Adaptive, no history | **+0.917** | A chose Rock in all 12 rounds; B mostly chose Paper |

The no-history condition demonstrated predictable repeated outputs from deterministic decoding. The history condition changed outputs but **did not** demonstrate LLM weight training.

**Code:** `rps_ollama_baseline.py` (also an alternative Hugging Face inference script `rps_pretrained_llm_baseline.py`).

### Step L2 — Fixed and reactive bait-and-switch

In three 18-round seeds with the reported `llama3.2` setup:

| B opponent | B mean reward/round | A normalized entropy |
|---|---:|---:|
| Uniform | −0.111 | 0.605 |
| Fixed: Rock ×4, Scissors ×2, repeat | −0.278 | 0.195 |
| Reactive baiting | +0.167 | 0.790 |
| Adaptive action-frequency response | **+0.444** | 0.782 |

In reactive baiting, B waited until it saw two consecutive Papers, then tried Scissors while the recent pattern persisted. Across 54 rounds, baiting occurred in 24 rounds (B total reward `−6`) and exploitation in 30 rounds (reward `+15`). Its net reward was `+9/54=+0.167`.

**Observation:** Reactive switching improved on the costly fixed schedule, but remained worse than the ordinary adaptive baseline. A frequently chose Paper even against uniform B; therefore its Paper preference might not have been **caused** by Rock baiting.

**Code:** `rps_ollama_bait_switch.py`, `rps_ollama_adaptive_bait.py`.

### Step L3 — Compare immediate exploitation with baiting

We shared a two-round warm-up and compared Always Scissors, Rock ×2 then Scissors, Rock ×4 then Scissors, and adaptive B. In the three-seed, 18-evaluation-round test:

| B strategy | B reward/round | Later A Paper frequency |
|---|---:|---:|
| Always Scissors | −0.167 | 23.8% |
| Bait 2 → Scissors | −0.185 | 35.7% |
| Bait 4 → Scissors | −0.148 | 52.4% |
| Adaptive | **+0.500** | 31.0% |

Longer Rock baiting coincided with a higher later Paper rate, but it remained unprofitable overall. This **did not** causally establish profitable manipulation.

**Code:** `rps_ollama_bait_causality.py`.

### Step L4 — Final controlled LLM comparison (two model families)

We compared `llama3.1:8b` and `qwen3.5:latest`, with:

- `2` initial prefixes **per model**, each containing `3` warm-up rounds;
- `12` evaluated rounds per prefix and B strategy;
- `5` B strategies, with warm-ups reused within each model/prefix;
- temperature `0`, thinking disabled where supported, and `0` invalid outputs;
- `20` evaluated episodes (`240` evaluated rounds) and `12` additional warm-up model calls.

**Agent B reward per evaluated round (higher is better for B):**

| B strategy | Llama 3.1 8B | Qwen3.5 |
|---|---:|---:|
| Uniform | −0.292 | +0.250 |
| Always Scissors | −0.250 | +0.000 |
| Bait 2 → Scissors | −0.208 | +0.125 |
| Bait 4 → Scissors | −0.083 | +0.292 |
| Adaptive | −0.375 | **+0.542** |

**Later Paper rate (evaluated rounds 5 onward):**

| B strategy | Llama 3.1 8B | Qwen3.5 |
|---|---:|---:|
| Always Scissors | 25.0% | 6.25% |
| Bait 2 → Scissors | 25.0% | 0.0% |
| Bait 4 → Scissors | 37.5% | 12.5% |
| Adaptive | 31.25% | 31.25% |

**Paired comparison with Always Scissors:**

- Llama 3.1: four-Rock baiting increased B reward by `+0.167/round` and later Paper frequency by `+12.5 percentage points`, but B's total result stayed negative.
- Qwen3.5: four-Rock baiting increased B reward by `+0.292/round` but later Paper frequency only by `+6.25 points`.

**Crucial round-level finding (verified from `rounds.csv`):** In Qwen's four-Rock condition, B earned **`+0.750` per bait round** (8 total bait rounds across both prefixes) and just **`+0.0625` per Scissors exploit round** (16 exploit rounds). The supposed “bait” was already profitable because Qwen frequently chose Scissors. Consequently, higher cumulative B reward **does not establish** that B paid a cost to induce and exploit Paper.

The two initial warm-ups were matched **within each model**, not across Llama and Qwen. Two deterministic prefixes cannot establish model-wide rankings or independent-sample statistics.

**Code and data:** `step1_controlled_ollama_rps.py`, `episodes.csv`, `rounds.csv`.

### LLM takeaway

The local LLM experiments demonstrated predictable action biases, useful in-context history adaptation, and vulnerability to reactive exploitation in some settings. They **did not** establish a general causal bait-and-switch mechanism in a real LLM, and they did not train LLM parameters with a swap objective.

---

## 6. Final controlled evaluations

### Final Step 1 — Controlled Ollama experiments

The final LLM outcomes are documented in **Step L4** above and in `episodes.csv` / `rounds.csv`. The controlled question was whether baiting improves later Paper frequency **and** B's total reward beyond a no-bait commitment. Results were mixed and provided alternative explanations, especially for Qwen.

### Final Step 2 — Paired neural reward-only versus swap-regularized learning

We reran the online RPS Transformer over **10 matched initial seeds**, **20 rounds per episode**, and three B strategies. Agent A's learning rate was `0.24`; its update used a six-observation buffer. We paired initialization seeds and A's random draws. Since B reacts to A, the observed opponent trajectories themselves were **not identical** across λ settings.

| B strategy | A objective | B expected reward | B realized reward | A external regret | A swap regret | A normalized entropy |
|---|---|---:|---:|---:|---:|---:|
| Uniform | `λ=0` | −0.006 | 0.040 | 0.210 | 0.315 | 0.584 |
| Uniform | `λ=1` | +0.021 | 0.040 | 0.210 | 0.300 | 0.383 |
| Myopic | `λ=0` | **0.250** | 0.240 | 0.320 | **0.575** | 0.909 |
| Myopic | `λ=1` | 0.499 | 0.535 | 0.635 | 1.015 | 0.707 |
| Planner-2 | `λ=0` | **0.292** | 0.250 | 0.340 | **0.600** | 0.866 |
| Planner-2 | `λ=1` | 0.491 | 0.480 | 0.610 | 0.960 | 0.677 |

The table reports per-round means. **Higher B reward means A is more exploitable; lower A realized swap regret is preferable for A.**

**Paired change for `λ=1` minus `λ=0`, with 95% seed-level percentile-bootstrap confidence intervals (3,000 bootstrap replicates):**

| B strategy | Change in B expected reward | Change in A realized swap regret | Change in A entropy |
|---|---|---|---|
| Uniform | +0.027 `[−0.009,+0.063]` | −0.015 `[−0.145,+0.115]` | −0.202 `[−0.250,−0.160]` |
| Myopic | **+0.248** `[+0.206,+0.290]` | **+0.440** `[+0.220,+0.690]` | −0.202 `[−0.239,−0.168]` |
| Planner-2 | **+0.199** `[+0.152,+0.244]` | **+0.360** `[+0.150,+0.565]` | −0.189 `[−0.229,−0.150]` |

**Incremental value of planning:** Planner-2 minus myopic B expected reward was:

- `λ=0`: `+0.042`, 95% bootstrap interval `[−0.001,+0.084]`;
- `λ=1`: `−0.008`, interval `[−0.057,+0.037]`.

Neither planning-advantage interval excludes zero. We therefore **did not demonstrate** a reliable incremental benefit of the two-step planner over myopic B in the final evaluation.

**Main neural finding:** With this particular online-learning rule and short-memory surrogate, swap regularization made A more predictable (lower policy entropy), more exploitable by adaptive B, and higher-regret under those adaptive trajectories. This is a **negative result about our setup**, not a result about all regret objectives or all LLMs.

**Code and data:** `step2_neural_swap_validation.py`, `online_transformer_rps_model_mismatch.py`, `step2_neural_swap_results/runs.csv`.

---

## 7. What the project answered

| Original research question | Evidence-based answer | Status |
|---|---|---|
| Can an adversary change what an online learner learns? | Yes, demonstrably in the constructed Bait/Exploit game. | **Demonstrated in toy game** |
| Can B profit from explicitly anticipating future neural updates? | Sometimes in toy and RPS studies, but not consistently beyond myopic baselines. | **Mixed** |
| Can a neural Transformer be trained with a swap-regret-related objective? | Yes; the differentiable expected-occupancy loss was implemented and optimized. | **Implemented** |
| Does this objective guarantee no swap regret? | No; it is a surrogate, with no theoretical guarantee from these experiments. | **Not established** |
| Did the surrogate reduce regret on a simple synthetic test? | In one sampled-action run, yes; in another held-out run, no. | **Mixed** |
| Did the surrogate protect the online neural learner under adaptive RPS? | No, not in the final measured configuration; observed exploitability and regret increased. | **Negative result** |
| Can B exploit A without inspecting weights? | Yes, using action-frequency models and a learned GRU in the tested simulations. | **Observed** |
| Does deeper planning always help? | No; it often underperformed simpler policies. | **Disproved as a universal claim** |
| Does history affect a pretrained LLM's RPS behaviour? | Yes, in observed prompt-conditioned runs; weights stayed frozen. | **Observed in-context behaviour** |
| Was profitable bait-induced manipulation established for a real LLM? | No. Baiting sometimes improved score, but simpler existing-preference explanations remain. | **Not established** |
| Was a pretrained LLM fine-tuned with swap-regret loss? | No; real-LLM experiments were inference-only. | **Not attempted** |

### One-paragraph research conclusion

This preliminary work demonstrates that learning-aware manipulation is possible in a controlled constructed game, and that anticipating an agent's neural updates can affect strategic action selection. However, in zero-sum RPS, the added value of planning is sensitive to the horizon, the quality of the opponent model, and the learner's objective. A black-box surrogate can learn useful action patterns, but model error limits long-horizon rollouts. The swap-regret surrogate is differentiable yet does not provide a no-swap-regret guarantee; in final online RPS tests, its inclusion increased the learner's exploitability under adaptive opponents. Experiments with local LLMs demonstrated history-sensitive decisions and exploitable preferences, but did not establish profitable *causal* manipulation or swap-regret-trained LLM robustness.

---

## 8. Limitations and research interpretation

1. **Small samples:** Most exploratory runs used 3–10 seeds. The final LLM comparison used only two deterministic initial histories per model; the neural validation used ten paired initialization seeds.
2. **No proven swap-regret guarantee:** Our differentiable surrogate is not the classical theoretical algorithm; minimizing it on a recent buffer does not guarantee long-term no-swap-regret.
3. **External/swap regret is not policy regret:** For adaptive B, our hindsight calculation fixes B's *observed* moves. It does not ask how B would have reacted to counterfactual changes in A's actions.
4. **White-box advantage:** The earliest planners could observe A's policy and simulate its exact gradient updates. This is stronger information than most opponents of real LLM systems receive.
5. **Black-box model mismatch:** The GRU does not know A's true probabilities or weights; open-loop rollouts can compound errors and exploit the learned model rather than the real opponent.
6. **Frozen LLMs:** Ollama models changed decisions due to context; **none** was updated using gradient descent or trained with swap-regret objectives.
7. **Baiting causal identification:** Increased action frequency after baiting is not enough to infer that baiting caused the change. Even an apparently successful bait may itself be immediately profitable.
8. **Game design matters:** Bait/Exploit was handcrafted to make manipulation valuable; symmetric zero-sum RPS need not exhibit the same effects.
9. **Training vs evaluation:** Final training-batch loss, surrogate regret, sampled regret, and opponent reward are different measurements and should not be conflated.
10. **Not a literature novelty claim:** No comprehensive recent-paper review or theoretical novelty assessment is part of this README. It documents our implemented exploration rather than claiming a new research result.

**Interpretation rule:** Treat observed improvements as **descriptive** unless replicated with controlled baselines and meaningful independent samples. A negative result in this setup neither proves the idea impossible nor establishes that other swap-regret approaches are ineffective.

---

## 9. Reproducing and navigating the code

### Requirements

- Python 3.10+ (a recent version is recommended)
- `numpy`, `torch`; and a working local Ollama installation for LLM experiments
- Local Ollama models `llama3.1:8b` and/or `qwen3.5:latest` for final LLM comparisons
- An Apple Silicon Mac can run the tiny PyTorch models on CPU. Ollama can use local hardware acceleration according to its configuration.

Example setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install numpy torch
ollama ls
```

### Core scripts by stage

| Stage | File(s) | Purpose |
|---|---|---|
| Regret fundamentals | `compare_learners_payoffs.py`, `.patch` | Hedge, SwapMW, payoff-matrix indexing |
| Constructed bait manipulation | `learning_aware_opponent_step2.py` | Hedge learns; B plans Bait/Exploit |
| Online neural in bait game | `online_transformer_planner.py` | Neural updates + white-box planner |
| Online Transformer in RPS | `online_transformer_rps_planner.py` | Reward and swap-regularized A with B planning |
| Horizon comparisons | `online_transformer_rps_horizon_compare.py` | Uniform, fixed, myopic, horizon-2/3 planning |
| Update-model mismatch | `online_transformer_rps_model_mismatch.py` | Exact/inaccurate assumptions about A's learning |
| Learning-effect ablation | `rps_learning_update_ablation.py` | Same history, weights updated versus frozen |
| Stationary benchmark | `online_transformer_rps_stationary_benchmark.py` | Coarse-grid fixed mixtures versus adaptive play |
| Black-box frequency predictor | `rps_blackbox_opponent.py` | Behavioural estimation without gradients |
| Black-box GRU model | `rps_blackbox_dynamics.py` | Learn action-pair dynamics and MAP planning |
| Probability-weighted GRU rollouts | `rps_blackbox_expected_rollouts.py` | Compare MAP versus expected future actions |
| Initial LLM baseline | `rps_ollama_baseline.py` | Ollama history/no-history conditions |
| Optional Hugging Face inference baseline | `rps_pretrained_llm_baseline.py` | Alternative model-loading interface |
| LLM fixed/reactive bait | `rps_ollama_bait_switch.py`, `rps_ollama_adaptive_bait.py` | Bait cost, reactive switching |
| LLM bait controls | `rps_ollama_bait_causality.py` | Always Scissors versus bait schedules |
| **Final Step 1** | **`step1_controlled_ollama_rps.py`** | Matched-prefix real-LLM comparison |
| **Final Step 2** | **`step2_neural_swap_validation.py`** | Paired online-neural validation with bootstrap intervals |

The original MLP, tiny prediction Transformer, reward-trained Transformer and episodic swap-surrogate examples were first developed as **inline learning exercises** in the conversation or in the user's own repository; standalone files for those particular early examples are **not included** in this artifact list.

Some scripts import another script from the same directory. For example, `step2_neural_swap_validation.py` imports `online_transformer_rps_model_mismatch.py`. The original `compare_learners_payoffs.py` imports user-created `src/` modules. Copy the required dependencies together; downloading this README alone does not bundle those Python files.

### Run the final controlled LLM experiment

Ensure Ollama is running and the model names exist in `ollama ls`:

```bash
MODELS=llama3.1:8b,qwen3.5:latest PREFIXES=2 WARMUP=3 ROUNDS=12 \
  python3 step1_controlled_ollama_rps.py
```

Outputs are written to `ollama_rps_results/episodes.csv` and `ollama_rps_results/rounds.csv`. The supplied evidence copies are named `episodes.csv` and `rounds.csv`.

### Run the final paired neural comparison

```bash
SEEDS=10 STRATEGIES=uniform,myopic,plan2 \
  python3 step2_neural_swap_validation.py
```

For a larger re-evaluation:

```bash
SEEDS=20 STRATEGIES=uniform,myopic,plan2 \
  python3 step2_neural_swap_validation.py
```

Output: `neural_swap_results/runs.csv`.

### Evidence files

| File | What it contains | Observations |
|---|---|---:|
| `episodes.csv` | Final LLM experiment per model/prefix/strategy | **20 episodes** |
| `rounds.csv` | Final LLM round-level actions and B rewards | **240 evaluation rounds** |
| `neural_swap_results/runs.csv` | Final neural per-seed results | **60 runs** |
| `llm_regret_learning_preliminary_report.md` | Shorter research-style synthesis | — |

Data and code describe distinct stages and should not be pooled without accounting for differences in models, seeds, episode lengths, opponent behaviour, and metrics.

---

## 10. Complete questions and answers

This is a consolidated **Q&A record of the conceptual and experimental questions we addressed**, arranged in the order the project developed. The answers describe *what the experiments support*, not broader claims of proof.

### A. Regret learning and game theory

**Q1. What is a repeated game?**  
**A.** Agents choose actions over successive rounds; payoff depends on the combination of actions. They may adapt using past feedback.

**Q2. Why choose RPS?**  
**A.** It has a simple three-action, zero-sum payoff matrix, allowing us to study learning and strategic responses without complex environment mechanics. It is not ideal for every game-theoretic phenomenon.

**Q3. Why does Agent B need `PAYOFF_B = -PAYOFF_A.T`?**  
**A.** A's matrix indexes `(A action, B action)`; B's own-action-first matrix indexes `(B action, A action)`. Negation converts zero-sum payoff; transposition changes index order.

**Q4. Why do A's and B's cumulative RPS scores add to zero?**  
**A.** Each played round gives B exactly the negative of A's payoff.

**Q5. What is Hedge?**  
**A.** A full-information online learner that exponentially increases weights on actions with high counterfactual rewards and samples from normalized weights.

**Q6. Does Hedge change neural weights through backpropagation?**  
**A.** No. It uses an explicit multiplicative-weight update, not gradient descent on a neural model.

**Q7. What is external regret?**  
**A.** Actual cumulative reward subtracted from the cumulative reward of the best *single fixed action* against the observed opponent sequence.

**Q8. Can realized external regret be negative?**  
**A.** Yes. A changing action sequence may outperform every fixed action in hindsight.

**Q9. What is dynamic regret?**  
**A.** A comparison against a time-varying action benchmark. It can be much larger than external regret when the environment switches regimes.

**Q10. What is swap regret?**  
**A.** The maximum hindsight gain from a fixed mapping replacing each original action with a potentially different action.

**Q11. Why is realized swap regret nonnegative and at least realized external regret?**  
**A.** The identity mapping yields zero improvement and every constant-action replacement used by external regret is a valid swap mapping.

**Q12. What do the SwapMW transition-matrix rows represent?**  
**A.** Each row is an internal replacement distribution for one original action. A stationary distribution of the whole transition matrix becomes the played mixed strategy.

**Q13. Did the first Hedge–Swap comparison identify a superior learner?**  
**A.** No. The algorithms played different self-play games, and their finite-horizon regret values are not directly controlled head-to-head measurements.

### B. Project 1 — neural models and swap-regret training

**Q14. What did the first MLP learn?**  
**A.** To predict the next RPS action from five observed opponent moves. It achieved 88.67% held-out accuracy on a deliberately predictable synthetic generator.

**Q15. Why was approximately 90% accuracy plausible?**  
**A.** The generator's preferred action had effective probability 90%, so a learned favourite-action predictor can approach that accuracy.

**Q16. How does an MLP differ from a Transformer here?**  
**A.** The MLP flattens five token embeddings; the Transformer combines token and positional embeddings through self-attention and a causal mask.

**Q17. Why do Transformers need positional information?**  
**A.** Token identity alone doesn't encode order. Positional embeddings help distinguish identical actions occurring at different history positions.

**Q18. Do positional embeddings themselves decide which past action matters most?**  
**A.** No. They supply position information; attention and other learned layers determine how to use it.

**Q19. Did the Transformer beat the MLP?**  
**A.** Not in the reported accuracy values (81.25% vs 88.67%), but they were not evaluated on one matched held-out dataset, so this isn't a conclusive architectural comparison.

**Q20. What changed when we trained directly for reward?**  
**A.** The output became A's own action distribution and the loss became negative expected RPS payoff instead of cross-entropy for predicting B.

**Q21. Why can the reward-training loss be negative?**  
**A.** If expected game reward is positive, its negative is below zero; minimizing it maximizes the expected reward.

**Q22. Why did the reward-trained Transformer become nearly deterministic?**  
**A.** The objective rewards concentrating mass on a high-payoff best response in the predictable synthetic setting. That concentration can itself create exploitability.

**Q23. How did we make swap regret differentiable?**  
**A.** We weighted each candidate action-replacement gain by the policy's probability of selecting the source action, then summed the best gains across sources.

**Q24. Is the differentiable swap-loss equal to sampled swap regret?**  
**A.** No. It uses expected action occupancy. Sampled actions add variance, and the learner's changing trajectory may create further differences.

**Q25. Did swap regularization help in Project 1?**  
**A.** Sometimes. One sampled-action run reduced realized swap regret from 0.0867 to 0.0763, but another held-out surrogate run slightly worsened it. No reliable general improvement was established.

**Q26. Does minimizing this loss prove A has no swap regret?**  
**A.** No. The objective is an empirical regularizer, not a theorem or classical online no-swap-regret algorithm with established assumptions.

### C. Project 2 — learning-aware neural planning

**Q27. What is the difference between myopic and learning-aware B?**  
**A.** Myopic B maximizes the current reward. Learning-aware B evaluates how its action changes A's future policy and the resulting future rewards.

**Q28. Did a learning-aware B deliberately manipulate a learner?**  
**A.** Yes, in the specifically constructed Bait/Exploit game. B initially accepted negative payoffs to make X more likely, then switched to a profitable Exploit action.

**Q29. What did the Bait/Exploit experiment show numerically?**  
**A.** Myopic B earned 0.200 expected reward per round; the three-step planner earned 0.693 in the Hedge version. A's P(X) grew from 0.500 to about 0.924 before exploitation.

**Q30. Can the same mechanism operate through neural gradients?**  
**A.** In our constructed neural and online-Transformer variants, B could simulate A's updates and achieved better expected rewards than myopic play.

**Q31. Did this establish a learning-aware planning advantage in RPS?**  
**A.** No. In RPS, planning sometimes helped slightly but frequently performed as badly as or worse than the myopic baseline.

**Q32. Does a longer planning horizon always help?**  
**A.** No. Two-step and three-step planners often had different outcomes, with no monotonic improvement as horizon increased.

**Q33. Does knowing A's exact learning rate guarantee best performance?**  
**A.** No. Some intentionally misspecified model assumptions produced higher finite-horizon rewards than the exact-model planner, likely reflecting short-horizon policy choices and trajectory differences.

**Q34. What did the gradient-update ablation isolate?**  
**A.** At the same next history, A's action probabilities shifted by about 0.085–0.197 on average when weights were actually updated. Thus, changes were not caused solely by moving the history window.

**Q35. Does that ablation prove B benefited from causing the update?**  
**A.** No. It identifies an effect on A's policy, not B's causal motivation or the long-run payoff benefit.

**Q36. Did adaptive B beat stationary strategies?**  
**A.** Yes, against the coarse-grid stationary mixtures we tested, but that does not show superiority to every possible stationary mixed commitment.

**Q37. Can B exploit A without reading A's gradients or weights?**  
**A.** Yes, our action-frequency and GRU black-box agents earned positive expected reward on the tested RPS episodes.

**Q38. Did the GRU learn the true neural update rule?**  
**A.** Not necessarily. It learned predictive correlations in observed action-pair sequences. Its outputs were still imperfect, and no parameter-identification guarantee was established.

**Q39. Why didn't probability-weighted GRU planning beat MAP planning?**  
**A.** Expected rollouts were more faithful to the *learned* probabilistic model, but that model was misspecified relative to the true online learner. Multi-step errors can compound.

**Q40. Why did the privileged oracle perform better?**  
**A.** It could inspect A's current true action distribution; black-box B had to infer it from finite, noisy observations.

### D. Pretrained LLM experiments

**Q41. Did we run an actual pretrained LLM?**  
**A.** Yes, locally through Ollama. Early demonstrations reported `llama3.2`; final controlled comparisons used `llama3.1:8b` and `qwen3.5:latest`.

**Q42. Was the LLM updating its neural parameters during RPS?**  
**A.** No. The LLMs were used for inference with changing prompts. This is in-context adaptation, not parameter training.

**Q43. Did game history matter for LLM actions?**  
**A.** Yes. In one 12-round early run, no-history A played Rock every round and adaptive B earned +0.917/round; the history-conditioned game produced different actions and B earned +0.333/round.

**Q44. Did a fixed Rock/Scissors bait cycle succeed?**  
**A.** No. In the early 18-round experiment B averaged −0.278/round, because repeated Rock baiting cost more than the Scissors phase recovered.

**Q45. Did reactive baiting help?**  
**A.** It improved over fixed baiting (+0.167 vs −0.278) but remained below the regular adaptive strategy (+0.444).

**Q46. Does a higher post-bait Paper rate prove causation?**  
**A.** No. The LLM may already prefer Paper, the payoff during bait may itself be favourable, and different histories can change behaviour in other ways.

**Q47. What happened in the final Llama 3.1 controlled test?**  
**A.** Four-round baiting was associated with a +0.167/round B improvement over Always Scissors and a 12.5-point higher later Paper rate, but B still earned negative reward overall. Only two prefixes were tested.

**Q48. What happened in the final Qwen3.5 controlled test?**  
**A.** Four-round baiting improved B reward by +0.292/round relative to Always Scissors, but the Rock bait phase itself earned +0.750/round while the Scissors phase earned only +0.0625/round. That is not evidence of a profitable sacrificed-bait manipulation mechanism.

**Q49. Which real-LLM opponent won most in the final Qwen test?**  
**A.** The tested adaptive frequency-based strategy, earning +0.542 B reward/round. That ranking is specific to the tested prompts and model configurations.

**Q50. Did the controlled study establish which language model is generally more intelligent or robust?**  
**A.** No. Two model-specific deterministic warm-up prefixes and short RPS episodes cannot support a general model ranking.

### E. Final evaluation and research conclusions

**Q51. Did the final neural experiment support swap-regularization as a defense?**  
**A.** No. Against myopic B, B's expected reward increased from 0.250 to 0.499 and A's realized swap regret rose from 0.575 to 1.015 when `λ` changed from 0 to 1. The planner showed the same adverse direction.

**Q52. Was the observed neural effect larger than ordinary seed variation?**  
**A.** The reported paired bootstrap intervals for B exploitation and A swap-regret changes under adaptive B excluded zero in ten-seed tests, which is preliminary repeatability evidence under this setup, not broad statistical or causal proof.

**Q53. Did the final two-step planner outperform myopic play reliably?**  
**A.** No. The paired planning-advantage intervals covered zero for both training objectives.

**Q54. Does lower entropy necessarily mean greater exploitability?**  
**A.** Not necessarily. More concentrated policies can be easier to predict, and this pattern co-occurred with higher exploitation here, but no causal mediation test was performed.

**Q55. Why is external/swap regret against recorded actions not causal policy regret?**  
**A.** With a reactive B, changing A's earlier choices could change B's later actions. Holding B's *recorded* sequence fixed ignores this counterfactual feedback loop.

**Q56. Have we tested swap-regret fine-tuning of a pretrained LLM?**  
**A.** No. We trained tiny neural/Transformer policies with a swap surrogate; pretrained Ollama LLMs remained frozen.

**Q57. Have we answered the original preliminary research question?**  
**A.** **Yes at the exploratory level.** We learned the algorithms, built the mechanisms, and established which claims our data do and do not support. We have **not** proven a universal learning-aware advantage or a robust regret-based LLM defense.

**Q58. What should a full research study do next?**  
**A.** Use many independently varied prompts and seeds, stronger calibrated opponent models, matched counterfactual interventions, longer horizons, tuned comparator strategies, multiple swap coefficients, and—if directly studying LLM training—matched small-model fine-tuning with reward-only versus reward-plus-swap objectives.

**Q59. Is the project a publishable novel method already?**  
**A.** Not established. It is a successful educational prototype and source of research hypotheses; novelty, theoretical guarantees, and publication-level evidence would require additional literature review and rigorous experimentation.

**Q60. What is the most defensible final research question?**  
**A.** *Under what conditions can a strategic opponent causally influence an adaptive neural or language-model agent, and when do regret-based training objectives reduce—rather than increase—that vulnerability?*

---

## Closing note

This project intentionally included **negative and mixed results**. That is part of its value. The completed prototype links classical regret learning, neural training, game-theoretic planning, black-box opponent modelling, and real LLM interaction while clearly distinguishing **what worked**, **what was observed**, and **what remains unproven**.
