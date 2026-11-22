# Lab architecture and API

## Project layout

```text
rock-paper-scissor/
├── src/                   # Research implementations and original command-line experiments
├── backend/
│   ├── app/
│   │   ├── catalog.py     # Experiment registry, field constraints, and event schemas
│   │   ├── main.py        # FastAPI, background execution, streaming and cancellation
│   │   ├── simulations.py # Conditions, planned workload, and adapter dispatch
│   │   ├── rounds.py      # Payoff indexing, cumulative scores, and hindsight comparators
│   │   ├── regret.py      # Hedge and SwapMW round adapters
│   │   ├── neural.py      # Small supervised and reward-trained policies
│   │   ├── strategic.py   # Planner, black-box, and validation adapters
│   │   ├── llm.py         # Local model discovery and matched-prefix inference
│   │   └── results.py     # Archive reading, summaries, and CSV formatting
│   └── tests/
├── frontend/
│   └── src/               # React routes, forms, playback, charts, and walkthroughs
├── outputs/               # Archived study CSVs and local run output
├── docs/
└── dev.sh
```

React and React Router handle the overview, experiment catalog, simulations, archive, and comparisons. Vite serves the app and proxies API requests. Recharts visualizes event values and recorded CSV metrics. Tailwind and the lab’s CSS supply responsive layouts and theme tokens.

One background worker runs blocking PyTorch and Ollama work for the local API. This keeps status, cancellation, and results routes responsive while bounding compute contention and preventing concurrent Torch initialization. Each condition has its own model, history, score, and random generator state. Matched conditions share their intended initialization and draws. Iterators yield at training iterations, warm-up rounds, or scored decisions. Cancellation is cooperative and can wait for the current planning round or HTTP call.

The API records terminal runs atomically as JSON. The original research CSVs are loaded on demand. Recorded round-level metrics are derived from the logged actions and checked against the recorded payoffs. The neural archive retains its original per-seed resolution.

## REST and event stream

| Method | Path | Response |
|---|---|---|
| GET | `/api/health` | API readiness |
| GET | `/api/experiments` | Catalog and form metadata |
| GET | `/api/experiments/{id}` | One experiment definition |
| GET | `/api/ollama/models` | Local server availability and installed model names |
| POST | `/api/simulations` | Validate inputs and start a background run |
| GET | `/api/simulations` | Saved and active run metadata |
| GET | `/api/simulations/{run_id}` | Run status, parameters, and workload |
| GET | `/api/simulations/{run_id}/events` | Ordered server-sent events |
| POST | `/api/simulations/{run_id}/cancel` | Request a stop at a work boundary |
| GET | `/api/simulations/{run_id}/results` | Available events and summaries |
| GET | `/api/simulations/{run_id}/export?format=csv` | Scored round CSV or full JSON with `format=json` |
| GET | `/api/historical-results` | Archive index |
| GET | `/api/historical-results/{dataset}` | Recorded metrics and available rounds |
| GET | `/api/historical-results/{dataset}/files/{index}` | Original recorded CSV by source index |

For example, POST `{"experiment_id":"r1","parameters":{"rounds":20,"strategy":"uniform","seed":42}}` to start a Hedge run. The response reports a run UUID and the total scored decisions across all conditions. Pydantic fills defaults and rejects out-of-range values. HTTP 409 indicates another run is active. HTTP 503 indicates that a requested Ollama service is unavailable.

Each server-sent event has a monotonic ID and a JSON body with a `type` and `data`. The event types are `started`, `training`, `warmup`, `round`, `completed`, `cancelled`, and `failed`. Reconnecting clients can send `Last-Event-ID` to continue from the last delivered event. Round indices are one-based and reset by condition.

A scored round carries actions, both payoffs, cumulative and average scores, regret, and empirical A frequencies. Pre-action policy, visible history, phase, and experiment-specific details appear where available. Raw LLM text and invalid-response flags are retained. Historical warm-up inputs are unavailable because the original files contain only evaluated rounds.

## Add an experiment

1. Define the typed parameter fields in `backend/app/catalog.py`. Use separate fields for scored rounds, input history length, warm-up, training steps, and planning horizon.
2. Reuse the underlying research algorithm. Place deterministic initialization, work units, and condition dispatch in the matching adapter.
3. Record real decisions with `Score.record` and yield typed progress events. Compute both decisions before revealing either current move. Identify privileged access to an agent’s probability distribution explicitly.
4. Add an independent worked example and interpretation in `frontend/src/content.ts`. The shared workspace renders the experiment’s form and results. Update the catalog and registration tests.
