"""Local simulation API. One worker owns computation and publishes replayable events."""
import asyncio
import json
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Event as StopFlag, RLock
from time import monotonic
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
TERMINAL = {"completed", "cancelled", "failed", "interrupted"}
IN_PROGRESS = {"queued", "running", "cancelling"}
CHECKPOINT_EVENTS = 50
CHECKPOINT_SECONDS = 5
SHUTDOWN_TIMEOUT = 5
SHUTTING_DOWN = False
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app):
    global SHUTTING_DOWN
    recover_interrupted()
    with LOCK:
        SHUTTING_DOWN = False
    try:
        yield
    finally:
        with LOCK:
            SHUTTING_DOWN = True
            pending = []
            for run in ACTIVE.values():
                run.request_cancel()
                if run.future and not run.future.done():
                    pending.append(asyncio.wrap_future(run.future))
        if pending:
            _, unfinished = await asyncio.wait(pending, timeout=SHUTDOWN_TIMEOUT)
            if unfinished:
                LOGGER.warning("Simulation worker is still finishing its current work during shutdown.")


app = FastAPI(title="Learning-Aware AI Simulation Lab", version="1.0.0", lifespan=lifespan)


@dataclass
class Run:
    metadata: dict
    conditions: list
    cancel: StopFlag = field(default_factory=StopFlag)
    events: list = field(default_factory=list)
    future: Future | None = None
    finishing: bool = False

    def emit(self, event):
        with LOCK:
            self.events.append(event.model_dump())
            if event.type == "round":
                self.metadata["completed_rounds"] += 1

    def snapshot(self):
        with LOCK:
            return {**self.metadata, "events": list(self.events)}

    def request_cancel(self):
        if not self.finishing and self.metadata["status"] in IN_PROGRESS:
            self.cancel.set()
            self.metadata["status"] = "cancelling"


def work(run):
    m = run.metadata
    status, message = "completed", None
    stop_token = llm.STOP_REQUEST.set(run.cancel)
    try:
        with LOCK:
            if not run.cancel.is_set():
                m["status"] = "running"
                run.emit(Event(type="started", data=dict(total_rounds=m["total_rounds"], model_calls=m["model_calls"])))
        if not run.cancel.is_set():
            checkpoint(run)
            last_save, pending = monotonic(), 0
            with _event_iterator(run) as iterator:
                while not run.cancel.is_set():
                    event = next(iterator, None)
                    if event is None:
                        break
                    run.emit(event)
                    pending += 1
                    if pending >= CHECKPOINT_EVENTS or monotonic() - last_save >= CHECKPOINT_SECONDS:
                        checkpoint(run)
                        last_save, pending = monotonic(), 0
    except llm.RequestCancelled:
        status = "cancelled"
    except Exception as exc:
        LOGGER.exception("Simulation %s failed", m["id"])
        status, message = "failed", str(exc) or type(exc).__name__
    finally:
        llm.STOP_REQUEST.reset(stop_token)

    with LOCK:
        run.finishing = True
        if run.cancel.is_set():
            status, message = "cancelled", None
        data = run.snapshot()
    mark_terminal(data, status, message)
    try:
        save_final(data)
    except (OSError, ValueError, TypeError):
        LOGGER.exception("Could not save %s", m["id"])
        with LOCK:
            mark_terminal(m, "failed", "Could not save this run to the local outputs folder.", run.events)
        # Keep the partial run readable for the rest of this process.
        return
    with LOCK:
        ACTIVE.pop(m["id"], None)


def _event_iterator(run):
    m = run.metadata
    return closing(execute(m["id"], m["experiment_id"], m["parameters"], run.conditions))


def mark_terminal(data, status, message, events=None):
    data.update(status=status, message=message)
    target = data["events"] if events is None else events
    target.append(dict(type=status, data=dict(message=message, completed_rounds=data["completed_rounds"])))


def with_summary(data):
    summary = results.summarize(data["events"])
    data = {**data, "summary": summary}
    if data["experiment_id"] == "s6":
        data["paired"] = results.validation_pairs(summary, data["parameters"]["rounds"])
    elif data["experiment_id"] == "l4":
        finished = [row for row in summary if row["rounds"] == data["parameters"]["rounds"]]
        data["paired"] = results.paired_ollama(finished, "average_b")
    return data


def write_json(file, data):
    file.parent.mkdir(parents=True, exist_ok=True)
    draft = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=file.parent, suffix=".tmp", delete=False) as writer:
            draft = Path(writer.name)
            writer.write(json.dumps(data, allow_nan=False))
        draft.replace(file)
    finally:
        if draft is not None:
            draft.unlink(missing_ok=True)


def checkpoint(run):
    data = run.snapshot()
    try:
        write_json(RUN_DIRECTORY / f"{data['id']}.partial", data)
    except (OSError, ValueError, TypeError):
        LOGGER.exception("Could not checkpoint simulation %s", data["id"])


def save_final(data):
    file = RUN_DIRECTORY / f"{data['id']}.json"
    write_json(file, with_summary(data))
    try:
        write_index(file, metadata(data), file_stamp(file))
    except OSError:
        LOGGER.warning("Could not index simulation file %s", file)
    remove_checkpoint(file.with_suffix(".partial"))


def remove_checkpoint(file):
    try:
        file.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("Could not remove simulation checkpoint %s", file)


def recover_interrupted():
    with LOCK:
        live_ids = set(ACTIVE)
    for file in RUN_DIRECTORY.glob("*.partial"):
        if file.stem in live_ids:
            continue
        final = file.with_suffix(".json")
        try:
            if final.is_file() and saved_metadata(final)["status"] in TERMINAL:
                remove_checkpoint(file)
                continue
        except (OSError, ValueError):
            LOGGER.warning("Could not read simulation file %s", final)
        try:
            data = read_run_file(file)
            if data["status"] not in IN_PROGRESS:
                raise ValueError("Expected an unfinished run.")
            mark_terminal(data, "interrupted", "The backend stopped before this simulation finished. Saved progress is available.")
            save_final(data)
        except (OSError, ValueError, TypeError, KeyError):
            LOGGER.exception("Could not recover simulation checkpoint %s", file)


def get_run(run_id):
    key = str(run_id)
    with LOCK:
        run = ACTIVE.get(key)
        if run:
            return run.snapshot()
    try:
        return read_run_file(RUN_DIRECTORY / f"{key}.json")
    except FileNotFoundError as exc:
        raise HTTPException(404, "Run not found.") from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(500, "Could not read this saved run.") from exc


def metadata(run):
    return {key: value for key, value in run.items() if key not in {"events", "summary", "paired"}}


def valid_metadata(data, run_id):
    return (isinstance(data, dict) and data.get("id") == run_id
            and isinstance(data.get("created_at"), str)
            and isinstance(data.get("experiment_id"), str)
            and isinstance(data.get("game"), str)
            and isinstance(data.get("parameters"), dict)
            and isinstance(data.get("conditions"), list)
            and isinstance(data.get("completed_rounds"), int)
            and isinstance(data.get("total_rounds"), int)
            and isinstance(data.get("status"), str) and data["status"] in TERMINAL | IN_PROGRESS)


def read_run_file(file):
    data = json.loads(file.read_text(encoding="utf-8"))
    if not valid_metadata(data, file.stem) or not isinstance(data.get("events"), list):
        raise ValueError("Invalid simulation file.")
    return data


def file_stamp(file):
    stat = file.stat()
    return [stat.st_ino, stat.st_size, stat.st_mtime_ns]


def write_index(file, item, stamp):
    write_json(file.with_suffix(".meta"), dict(stamp=stamp, metadata=item))


def saved_metadata(file):
    stamp = file_stamp(file)
    try:
        index = json.loads(file.with_suffix(".meta").read_text(encoding="utf-8"))
        if isinstance(index, dict) and index.get("stamp") == stamp and valid_metadata(index.get("metadata"), file.stem):
            return index["metadata"]
    except (OSError, ValueError):
        pass
    item = metadata(read_run_file(file))
    try:
        if file_stamp(file) == stamp:
            write_index(file, item, stamp)
    except OSError:
        LOGGER.warning("Could not index simulation file %s", file)
    return item


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
        if SHUTTING_DOWN:
            raise HTTPException(503, "The lab is shutting down.")
        if any(run.metadata["status"] not in TERMINAL for run in ACTIVE.values()):
            raise HTTPException(409, "A simulation is already running. Finish or stop it before starting another.")
        run_id = str(uuid4())
        m = dict(id=run_id, experiment_id=exp, parameters=params, game=CATALOG[exp]["game"],
                 status="queued", created_at=datetime.now(timezone.utc).isoformat(),
                 completed_rounds=0, total_rounds=total, model_calls=calls, conditions=labels)
        run = Run(m, conditions)
        ACTIVE[run_id] = run
        run.future = WORKER.submit(work, run)
        return m.copy()


@app.get("/api/simulations")
def runs():
    with LOCK:
        items = {key: run.metadata.copy() for key, run in ACTIVE.items()}
    for file in RUN_DIRECTORY.glob("*.json"):
        try:
            items[file.stem] = saved_metadata(file)
        except (OSError, ValueError):
            LOGGER.warning("Could not read simulation file %s", file)
    return sorted(items.values(), key=lambda item: item["created_at"], reverse=True)


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
    if initial["status"] == "interrupted" and initial["events"] and cursor >= len(initial["events"]):
        # A reconnect may have seen more events than the last checkpoint retained.
        cursor = len(initial["events"]) - 1

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
        if run:
            run.request_cancel()
            return run.metadata.copy()
    return metadata(get_run(run_id))


@app.get("/api/simulations/{run_id}/results")
def run_results(run_id: UUID):
    run = get_run(run_id)
    if "summary" in run:
        return run
    return with_summary(run) if run["status"] in TERMINAL else {**run, "summary": results.summarize(run["events"])}


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
