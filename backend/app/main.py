"""Local simulation API. One worker owns computation and publishes replayable events."""
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event as StopFlag, RLock
from typing import Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse

from . import llm, results
from .catalog import CATALOG, Event, RunRequest
from .simulations import describe, execute, models_for, plan

RUN_DIRECTORY = results.OUTPUTS / "lab_runs"
LOCK = RLock()
WORKER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="simulation")
ACTIVE = {}
TERMINAL = {"completed", "cancelled", "failed"}
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    yield
    with LOCK:
        for run in ACTIVE.values():
            if run.metadata["status"] not in TERMINAL:
                run.cancel.set()
    # An empty job queued behind the active run waits for its current work
    # boundary and final write without closing the reusable executor.
    await asyncio.wrap_future(WORKER.submit(lambda: None))


app = FastAPI(title="Learning-Aware AI Simulation Lab", version="1.0.0", lifespan=lifespan)


@dataclass
class Run:
    metadata: dict
    conditions: list
    cancel: StopFlag = field(default_factory=StopFlag)
    events: list = field(default_factory=list)

    def emit(self, event):
        with LOCK:
            self.events.append(event.model_dump())
            if event.type == "round":
                self.metadata["completed_rounds"] += 1

    def serialize(self):
        summary = results.summarize(self.events)
        data = {**self.metadata, "events": self.events, "summary": summary}
        experiment_id = self.metadata["experiment_id"]
        if experiment_id == "s6":
            data["paired"] = results.validation_pairs(summary, self.metadata["parameters"]["rounds"])
        elif experiment_id == "l4":
            finished = [row for row in summary if row["rounds"] == self.metadata["parameters"]["rounds"]]
            data["paired"] = results.paired_ollama(finished, "average_b")
        return data


def work(run):
    m = run.metadata
    status, message = "completed", None
    run.emit(Event(type="started", data=dict(total_rounds=m["total_rounds"], model_calls=m["model_calls"])))
    try:
        with LOCK:
            if not run.cancel.is_set():
                m["status"] = "running"
        with _event_iterator(run) as iterator:
            while not run.cancel.is_set():
                event = next(iterator, None)
                if event is None:
                    break
                run.emit(event)
    except Exception as exc:
        LOGGER.exception("Simulation %s failed", m["id"])
        status, message = "failed", str(exc) or type(exc).__name__

    with LOCK:
        if run.cancel.is_set():
            status, message = "cancelled", None
        m.update(status=status, message=message)
        run.emit(Event(type=status, data=dict(message=message, completed_rounds=m["completed_rounds"])))
        try:
            RUN_DIRECTORY.mkdir(parents=True, exist_ok=True)
            final = RUN_DIRECTORY / f"{m['id']}.json"
            draft = final.with_suffix(".tmp")
            draft.write_text(json.dumps(run.serialize(), allow_nan=False), encoding="utf-8")
            draft.replace(final)
        except (OSError, ValueError, TypeError):
            LOGGER.exception("Could not save %s", m["id"])
            m.update(status="failed", message="Could not save this run to the local outputs folder.")
            run.events.pop()
            run.emit(Event(type="failed", data=dict(message=m["message"])))
            # Keep the partial run readable for the rest of this process.
            return
        ACTIVE.pop(m["id"], None)


def _event_iterator(run):
    m = run.metadata
    return closing(execute(m["id"], m["experiment_id"], m["parameters"], run.conditions))


def get_run(run_id):
    key = str(run_id)
    with LOCK:
        if key in ACTIVE:
            run = ACTIVE[key]
            return {**run.metadata, "events": list(run.events)}
        try:
            return json.loads((RUN_DIRECTORY / f"{key}.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise HTTPException(404, "Run not found.") from exc


def metadata(run):
    return {key: value for key, value in run.items() if key not in {"events", "summary", "paired"}}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/experiments")
def experiments():
    return list(CATALOG.values())


@app.get("/api/experiments/{experiment_id}")
def experiment(experiment_id: str):
    if experiment_id not in CATALOG:
        raise HTTPException(404, "Unknown experiment.")
    return CATALOG[experiment_id]


@app.get("/api/ollama/models")
def models():
    return llm.installed_models()


@app.post("/api/simulations", status_code=202)
def start(request: RunRequest):
    exp, params = request.experiment_id, request.parameters
    if exp.startswith("l"):
        installed = llm.installed_models()
        if not installed["available"]:
            raise HTTPException(503, installed["message"])
        if any(model not in installed["models"] for model in models_for(params)):
            raise HTTPException(422, "Select a model currently installed in Ollama.")
    conditions = plan(exp, params)
    labels, total, calls = describe(exp, params, conditions)
    with LOCK:
        if any(run.metadata["status"] not in TERMINAL for run in ACTIVE.values()):
            raise HTTPException(409, "A simulation is already running. Finish or stop it before starting another.")
        run_id = str(uuid4())
        m = dict(id=run_id, experiment_id=exp, parameters=params, game=CATALOG[exp]["game"],
                 status="queued", created_at=datetime.now(timezone.utc).isoformat(),
                 completed_rounds=0, total_rounds=total, model_calls=calls, conditions=labels)
        run = Run(m, conditions)
        ACTIVE[run_id] = run
        WORKER.submit(work, run)
        return m.copy()


@app.get("/api/simulations")
def runs():
    with LOCK:
        items = [run.metadata.copy() for run in ACTIVE.values()]
        for file in RUN_DIRECTORY.glob("*.json"):
            try:
                items.append(metadata(json.loads(file.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                LOGGER.warning("Could not read simulation file %s", file)
    return sorted(items, key=lambda item: item["created_at"], reverse=True)


@app.get("/api/simulations/{run_id}")
def status(run_id: UUID):
    return metadata(get_run(run_id))


@app.get("/api/simulations/{run_id}/events")
async def events(run_id: UUID, request: Request):
    # Validate before sending streaming headers. Last-Event-ID resumes after reconnects.
    initial = get_run(run_id)
    try:
        cursor = max(0, int(request.headers.get("Last-Event-ID", 0)))
    except ValueError:
        cursor = 0

    async def stream():
        nonlocal cursor
        snapshot = initial
        heartbeat = 0
        while not await request.is_disconnected():
            for event in snapshot["events"][cursor:]:
                cursor += 1
                yield f"id: {cursor}\ndata: {json.dumps(event)}\n\n"
            if snapshot["status"] in TERMINAL:
                break
            await asyncio.sleep(.15)
            heartbeat += 1
            if heartbeat == 60:
                yield ": heartbeat\n\n"
                heartbeat = 0
            snapshot = get_run(run_id)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/simulations/{run_id}/cancel", status_code=202)
def cancel(run_id: UUID):
    with LOCK:
        run = ACTIVE.get(str(run_id))
        if run and run.metadata["status"] not in TERMINAL:
            run.cancel.set()
            run.metadata["status"] = "cancelling"
            return run.metadata.copy()
        return metadata(get_run(run_id))


@app.get("/api/simulations/{run_id}/results")
def run_results(run_id: UUID):
    run = get_run(run_id)
    if "summary" in run:
        return run
    return {**run, "summary": results.summarize(run["events"])}


@app.get("/api/simulations/{run_id}/export")
def export(run_id: UUID, format: Literal["csv", "json"] = "csv"):
    data = run_results(run_id)
    content = (results.csv_text([e["data"] for e in data["events"] if e["type"] == "round"])
               if format == "csv" else json.dumps(data, indent=2, ensure_ascii=False))
    return Response(content, media_type="text/csv" if format == "csv" else "application/json",
                    headers={"Content-Disposition": f'attachment; filename="simulation-{run_id}.{format}"'})


@app.get("/api/historical-results")
def historical_index():
    return results.DATASETS


@app.get("/api/historical-results/{dataset}")
def historical_data(dataset: str):
    if dataset not in {d["id"] for d in results.DATASETS}:
        raise HTTPException(404, "Dataset not found.")
    try:
        return results.historical(dataset)
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"Could not load the recorded dataset: {exc}") from exc


@app.get("/api/historical-results/{dataset}/files/{index}")
def historical_file(dataset: str, index: int):
    entry = next((d for d in results.DATASETS if d["id"] == dataset), None)
    if not entry or not 0 <= index < len(entry["files"]):
        raise HTTPException(404, "Research CSV not found.")
    file = results.OUTPUTS / entry["files"][index]
    if not file.is_file():
        raise HTTPException(404, "Research CSV not found.")
    return FileResponse(file, media_type="text/csv", filename=file.name)
