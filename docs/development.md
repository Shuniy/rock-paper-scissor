# Run the Simulation Lab locally

## Install

Use Python 3.10 or newer and Node.js 20.19+ in the 20.x line, or 22.12 or newer. From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements.txt
npm --prefix frontend ci
```

The lab runs its small PyTorch models on CPU. Ollama is optional.

## Start

```bash
./dev.sh
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). Press Ctrl+C in the terminal to stop the servers. The helper uses ports 5173 and 8000. Keep the services running while using the lab.

To start the two services in separate terminals:

```bash
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --timeout-graceful-shutdown 5
npm --prefix frontend run dev
```

Vite sends browser API requests to port 8000. To use another Python API port, set `LAB_API_TARGET` when starting Vite. The FastAPI OpenAPI explorer is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Run, inspect, and replay

1. Open **Run Simulation** or choose an experiment category from the sidebar. Configure the per-condition evaluation length, seed, and other inputs.
2. Start the run. Each scored decision contributes to the progress bar. Neural training iterations and warm-up rounds appear separately.
3. Pause playback, step through available rounds, or adjust the playback speed. Select a round in the event log or scrub the timeline to inspect it. The **Stop** button asks the backend to finish its current unit of work and exit.
4. Open **Replay Research Results** to browse the original CSVs or reopen saved local simulations. The Ollama archive includes scored rounds. The neural archive shows per-seed aggregates and paired differences.
5. Use **Experiment Comparison** to select compatible completed local runs and compare one condition from each. Configuration differences remain visible alongside the metrics.

Run artifacts are saved as JSON in `outputs/lab_runs/`. CSV and JSON downloads are available from the workspace. The archived research CSVs are read from their original locations under `outputs/`.

## Enable Ollama experiments

Open the Ollama app or run `ollama serve` in its own terminal. Confirm that you already have a model with `ollama list`. The lab lists installed models from Ollama’s local API and never downloads models. The default chat endpoint is `http://127.0.0.1:11434/api/chat`. Set `OLLAMA_URL` before starting the backend if Ollama is listening at another local address.

The configuration previews the exact planned model call count. Controlled comparisons run all five strategies for each prefix and selected model. For the archived workload, selecting two models, two prefixes each, three warm-up rounds, and twelve scored rounds produces 252 model calls. Use the archive for immediate access to those recorded results.

## Test and build

```bash
.venv/bin/python -m pip install -r backend/requirements-dev.txt
.venv/bin/python -m pytest -q backend/tests
npm --prefix frontend test
npm --prefix frontend run build
```

The tests use mock Ollama responses and temporary run storage. A real installed model is only needed for manual inference checks. The production front-end build is written to `frontend/dist/`.
