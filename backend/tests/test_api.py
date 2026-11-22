import asyncio
import json
from threading import Event
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.app import llm, main
from backend.app.catalog import Event as RunEvent


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "RUN_DIRECTORY", tmp_path)
    with TestClient(main.app) as client:
        yield client
    assert not main.ACTIVE


def submit(client, exp, **params):
    response = client.post("/api/simulations", json={"experiment_id": exp, "parameters": params})
    assert response.status_code == 202, response.text
    return response.json()


def read_events(client, run, offset=0):
    response = client.get(f"/api/simulations/{run['id']}/events", headers={"Last-Event-ID": str(offset)})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


@pytest.mark.parametrize("exp,params", [
    ("r2", {"rounds": 4}), ("n4", {"rounds": 4, "training_steps": 2}),
    ("s2", {"rounds": 3}), ("s6", {"rounds": 2, "seed_count": 2, "strategy": "plan2"}),
])
def test_live_round_stream_and_exports(client, exp, params):
    run = submit(client, exp, **params)
    events = read_events(client, run)
    assert events[0]["type"] == "started"
    assert events[-1]["type"] == "completed"
    assert sum(e["type"] == "round" for e in events) == run["total_rounds"]
    assert read_events(client, run, offset=2) == events[2:]
    data = client.get(f"/api/simulations/{run['id']}/results").json()
    assert data["events"] == events
    assert data["completed_rounds"] == run["total_rounds"]
    assert client.get("/api/simulations").json()[0]["id"] == run["id"]
    if exp == "s6":
        assert data["paired"][0]["pairs"] == 2
    response = client.get(f"/api/simulations/{run['id']}/export")
    assert response.headers["content-type"].startswith("text/csv")
    assert "cumulative_a" in response.text.splitlines()[0]


@pytest.fixture
def ollama_mock(monkeypatch):
    calls = []

    def server(request):
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "llama:test"}, {"name": "qwen:test"}]})
        payload = json.loads(request.content)
        calls.append(payload)
        raw = "I choose Rock and Paper" if len(calls) == 2 else "Paper"
        return httpx.Response(200, json={"message": {"content": raw}})

    sync_client, async_client = httpx.Client, httpx.AsyncClient
    monkeypatch.setattr(llm, "httpx", type("ClientModule", (), {
        "HTTPError": httpx.HTTPError, "Timeout": httpx.Timeout,
        "Client": staticmethod(lambda **kwargs: sync_client(transport=httpx.MockTransport(server), **kwargs)),
        "AsyncClient": staticmethod(lambda **kwargs: async_client(transport=httpx.MockTransport(server), **kwargs)),
    }))
    return calls


def test_ollama_no_current_move_leakage_and_invalid_response(client, ollama_mock):
    run = submit(client, "l2", model="llama:test", rounds=3, strategy="always_scissors")
    events = read_events(client, run)
    rows = [event["data"] for event in events if event["type"] == "round"]
    assert len(ollama_mock) == run["model_calls"] == 6
    assert rows[0]["history"] == []
    assert rows[1]["history"] == [{"a": "Paper", "b": "Scissors", "reward_a": -1}]
    assert rows[1]["a"] == "Rock" and rows[1]["detail"]["invalid"] and rows[1]["detail"]["fallback"] == "Rock"
    assert all(row["history"] == [] for row in rows[3:])
    assert "No prior rounds shown." in ollama_mock[0]["messages"][1]["content"]
    assert ollama_mock[0]["messages"] == ollama_mock[3]["messages"]
    assert ollama_mock[0]["messages"] == ollama_mock[5]["messages"]
    assert "Opponent: Scissors" in ollama_mock[1]["messages"][1]["content"]


def test_ollama_controlled_matched_prefixes_and_workload(client, ollama_mock):
    models = client.get("/api/ollama/models").json()
    assert models == {"available": True, "models": ["llama:test", "qwen:test"]}
    run = submit(client, "l4", model="llama:test", model_secondary="qwen:test", rounds=5, prefixes=1, warmup=2)
    events = read_events(client, run)
    assert len(ollama_mock) == run["model_calls"] == 54
    assert run["total_rounds"] == 50
    for start in [2, 29]:
        assert all(ollama_mock[start]["messages"] == ollama_mock[start + 5*i]["messages"] for i in range(5))
    assert len({row["model"] for row in ollama_mock}) == 2
    assert sum(e["type"] == "warmup" for e in events) == 4
    assert events[-1]["type"] == "completed"
    paired = client.get(f"/api/simulations/{run['id']}/results").json()["paired"]
    assert len(paired) == 8
    assert all(row["prefix"] == 1 and row["strategy"] != "always_scissors" for row in paired)


@pytest.mark.parametrize("exp,params", [("l1", {"rounds": 2, "history_window": 0}), ("l3", {"rounds": 6})])
def test_llm_baseline_and_baiting(client, ollama_mock, exp, params):
    run = submit(client, exp, model="llama:test", **params)
    events = read_events(client, run)
    scored = [event["data"] for event in events if event["type"] == "round"]
    assert events[-1]["type"] == "completed"
    if exp == "l1":
        assert all(row["history"] == [] for row in scored)
    else:
        assert [row["phase"] for row in scored] == ["bait"] * 4 + ["exploit"] * 2


def test_offline_ollama_error_is_friendly(client, monkeypatch):
    monkeypatch.setattr(main.llm, "installed_models", lambda: {
        "available": False, "models": [], "message": "Start the Ollama app."})
    response = client.post("/api/simulations", json={"experiment_id": "l1", "parameters": {"model": "absent"}})
    assert response.status_code == 503
    assert "Start the Ollama app" in response.json()["detail"]
    assert len(client.get("/api/historical-results").json()) == 2


def test_cancellation_completes_current_unit_and_releases_worker(client, monkeypatch):
    entered, release = Event(), Event()

    def in_flight(*args):
        entered.set()
        assert release.wait(timeout=5)
        yield RunEvent(type="training", data={"condition": "Fit", "step": 1, "total": 2})
        raise AssertionError("Worker should not continue after cancellation.")

    monkeypatch.setattr(main, "execute", in_flight)
    run = submit(client, "r1", rounds=3)
    try:
        assert entered.wait(timeout=5)
        second = client.post("/api/simulations", json={"experiment_id": "r1"})
        assert second.status_code == 409
        response = client.post(f"/api/simulations/{run['id']}/cancel")
        assert response.json()["status"] == "cancelling"
    finally:
        release.set()
    types = [event["type"] for event in read_events(client, run)]
    assert types == ["started", "training", "cancelled"]
    assert client.get(f"/api/simulations/{run['id']}").json()["status"] == "cancelled"


def test_cancel_queued_run_without_starting_computation(client, monkeypatch):
    release = Event()
    blocker = main.WORKER.submit(release.wait, 5)

    def unexpected(*args):
        pytest.fail("Queued simulation should never execute after cancellation.")

    monkeypatch.setattr(main, "execute", unexpected)
    try:
        run = submit(client, "r1")
        assert run["status"] == "queued"
        assert client.post(f"/api/simulations/{run['id']}/cancel").json()["status"] == "cancelling"
    finally:
        release.set()
    blocker.result(timeout=5)
    assert [event["type"] for event in read_events(client, run)] == ["cancelled"]
    assert client.post(f"/api/simulations/{run['id']}/cancel").json()["status"] == "cancelled"


def test_restarts_from_last_checkpoint_and_exports_partial_results(tmp_path, monkeypatch):
    live_path, restart_path = tmp_path / "live", tmp_path / "restart"
    restart_path.mkdir()
    reached, release = Event(), Event()
    execute = main.execute

    def pause_after_third_round(*args):
        iterator = execute(*args)
        try:
            for _ in range(3):
                yield next(iterator)
            reached.set()
            assert release.wait(timeout=5)
            yield from iterator
        finally:
            iterator.close()

    monkeypatch.setattr(main, "RUN_DIRECTORY", live_path)
    monkeypatch.setattr(main, "CHECKPOINT_EVENTS", 2)
    monkeypatch.setattr(main, "CHECKPOINT_SECONDS", 60)
    monkeypatch.setattr(main, "execute", pause_after_third_round)
    with TestClient(main.app) as client:
        run = submit(client, "r2", rounds=4)
        run_id = run["id"]
        try:
            assert reached.wait(timeout=5)
            checkpoint = (live_path / f"{run_id}.partial").read_bytes()
            assert json.loads(checkpoint)["completed_rounds"] == 2
            (restart_path / f"{run_id}.partial").write_bytes(checkpoint)
            (restart_path / "unfinished-write.tmp").write_text('{"events": [', encoding="utf-8")
        finally:
            release.set()
        assert read_events(client, run)[-1]["type"] == "completed"
        assert not (live_path / f"{run_id}.partial").exists()

    monkeypatch.setattr(main, "RUN_DIRECTORY", restart_path)
    with TestClient(main.app) as client:
        assert client.get("/api/simulations").json()[0]["status"] == "interrupted"
        assert client.get(f"/api/simulations/{run_id}").json()["completed_rounds"] == 2
        data = client.get(f"/api/simulations/{run_id}/results").json()
        assert data["summary"][0]["rounds"] == 2
        assert [event["type"] for event in read_events(client, run)] == ["started", "round", "round", "interrupted"]
        assert read_events(client, run, offset=4) == data["events"][-1:]
        assert client.get(f"/api/simulations/{run_id}/export?format=json").json() == data
        assert len(client.get(f"/api/simulations/{run_id}/export?format=csv").text.splitlines()) == 3
        assert not (restart_path / f"{run_id}.partial").exists()
    with TestClient(main.app) as client:
        assert read_events(client, run) == data["events"]


def test_shutdown_timeout_keeps_worker_occupied_until_cooperative_stop(tmp_path, monkeypatch):
    entered, release = Event(), Event()

    def in_flight(*args):
        entered.set()
        assert release.wait(timeout=5)
        yield RunEvent(type="training", data={"condition": "Fit", "step": 1, "total": 2})

    monkeypatch.setattr(main, "RUN_DIRECTORY", tmp_path)
    monkeypatch.setattr(main, "SHUTDOWN_TIMEOUT", .05)
    monkeypatch.setattr(main, "execute", in_flight)
    try:
        with TestClient(main.app) as client:
            run = submit(client, "r1")
            assert entered.wait(timeout=5)
        assert main.get_run(run["id"])["status"] == "cancelling"
        with TestClient(main.app) as client:
            assert client.post("/api/simulations", json={"experiment_id": "r1"}).status_code == 409
            release.set()
            assert [event["type"] for event in read_events(client, run)] == ["started", "training", "cancelled"]
    finally:
        release.set()


def test_shutdown_cancels_in_flight_ollama_request(tmp_path, monkeypatch):
    started, disconnected = Event(), Event()
    calls = 0
    async_client = httpx.AsyncClient

    async def server(request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(200, json={"message": {"content": "Paper"}})
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            disconnected.set()

    monkeypatch.setattr(main, "RUN_DIRECTORY", tmp_path)
    monkeypatch.setattr(llm, "installed_models", lambda: {"available": True, "models": ["llama:test"]})
    monkeypatch.setattr(llm.httpx, "AsyncClient",
                        lambda **kwargs: async_client(transport=httpx.MockTransport(server), **kwargs))
    with TestClient(main.app) as client:
        run = submit(client, "l1", model="llama:test", rounds=3)
        assert started.wait(timeout=5)
    assert disconnected.is_set()
    assert not main.ACTIVE
    with TestClient(main.app) as client:
        events = read_events(client, run)
        assert [event["type"] for event in events] == ["started", "round", "cancelled"]
        assert client.get(f"/api/simulations/{run['id']}/results").json()["completed_rounds"] == 1


def test_listing_indexes_legacy_runs_and_skips_bad_or_missing_files(client, tmp_path, monkeypatch):
    older = submit(client, "r1", rounds=2)
    assert read_events(client, older)[-1]["type"] == "completed"
    newer = submit(client, "r1", rounds=2)
    assert read_events(client, newer)[-1]["type"] == "completed"
    legacy = tmp_path / f"{older['id']}.json"
    legacy.with_suffix(".meta").unlink()
    invalid_json = tmp_path / f"{uuid4()}.json"
    invalid_json.write_text("{", encoding="utf-8")
    invalid_shape = tmp_path / f"{uuid4()}.json"
    invalid_shape.write_text('{"events": []}', encoding="utf-8")
    read_run_file = main.read_run_file
    reads = []

    def track_read(file):
        reads.append(file)
        return read_run_file(file)

    monkeypatch.setattr(main, "read_run_file", track_read)
    response = client.get("/api/simulations")
    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [newer["id"], older["id"]]
    assert set(reads) == {legacy, invalid_json, invalid_shape}
    assert all("events" not in row for row in response.json())

    reads.clear()
    invalid_json.unlink()
    invalid_shape.unlink()
    (tmp_path / f"{newer['id']}.json").unlink()
    assert [row["id"] for row in client.get("/api/simulations").json()] == [older["id"]]
    assert reads == []

    legacy.write_text("[", encoding="utf-8")
    assert client.get("/api/simulations").json() == []
    assert reads == [legacy]


def test_failed_run_is_persisted(client, monkeypatch):
    def fail(*args):
        raise RuntimeError("The model service stopped responding.")
        yield

    monkeypatch.setattr(main, "execute", fail)
    run = submit(client, "r1")
    events = read_events(client, run)
    assert events[-1]["type"] == "failed"
    assert "stopped responding" in events[-1]["data"]["message"]
    assert client.get(f"/api/simulations/{run['id']}/results").json()["status"] == "failed"


def test_historical_outputs_use_real_source_resolution_and_aggregate_boundaries(client):
    neural = client.get("/api/historical-results/neural-validation").json()
    assert neural["aggregate_only"] is True and len(neural["rows"]) == 60 and "runs" not in neural
    assert all(row["pairs"] == 10 for row in neural["paired"])
    ollama = client.get("/api/historical-results/ollama-controlled").json()
    assert len(ollama["rows"]) == len(ollama["runs"]) == 20
    assert sum(len(run["rounds"]) for run in ollama["runs"]) == 240
    assert all("history" not in round for run in ollama["runs"] for round in run["rounds"])
    assert client.get("/api/historical-results/ollama-controlled/files/0").status_code == 200
    assert client.get("/api/historical-results/missing").status_code == 404
