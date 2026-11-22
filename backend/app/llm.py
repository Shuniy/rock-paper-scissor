"""Local Ollama transport and matched-history conditions."""
import asyncio
import os
from contextlib import closing, contextmanager
from contextvars import ContextVar

import httpx
import numpy as np

from src.llm.controlled_ollama_rps import choose_b, make_messages, parse_action

from .catalog import Event
from .rounds import ACTIONS, RPS, Score, window

STOP_REQUEST = ContextVar("ollama_stop_request", default=None)
CHAT_TIMEOUT = 180


class RequestCancelled(Exception):
    """The local worker no longer needs the in-flight Ollama response."""


def ollama_url():
    chat = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/chat").rstrip("/")
    return chat


def installed_models():
    try:
        root = ollama_url().removesuffix("/api/chat")
        with httpx.Client(timeout=3, trust_env=False) as client:
            response = client.get(f"{root}/api/tags")
            response.raise_for_status()
            models = response.json()["models"]
        return {"available": True, "models": sorted({model["name"] for model in models})}
    except (httpx.HTTPError, ValueError, KeyError, TypeError):
        return {"available": False, "models": [],
                "message": "Ollama is unavailable. Start the Ollama app or run ollama serve."}


async def post_chat(client, payload):
    stop = STOP_REQUEST.get()
    if stop is not None and stop.is_set():
        raise RequestCancelled
    task = asyncio.create_task(client.post(ollama_url(), json=payload))
    try:
        while not task.done():
            if stop is not None and stop.is_set():
                raise RequestCancelled
            await asyncio.wait({task}, timeout=.2)
        return task.result()
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@contextmanager
def chat_connection():
    # Keep one connection pool on the worker's event loop for the whole run.
    with closing(asyncio.new_event_loop()) as loop:
        client = httpx.AsyncClient(timeout=httpx.Timeout(CHAT_TIMEOUT, connect=5), trust_env=False)
        try:
            yield lambda payload: loop.run_until_complete(asyncio.wait_for(post_chat(client, payload), CHAT_TIMEOUT))
        finally:
            loop.run_until_complete(client.aclose())


def ask(post, model, history, size, temperature, seed):
    payload = {"model": model, "messages": make_messages(history, size), "stream": False, "think": False,
               "options": {"temperature": temperature, "num_predict": 24, "seed": seed}, "keep_alive": "5m"}
    try:
        response = post(payload)
        response.raise_for_status()
        return parse_action(response.json()["message"]["content"].strip())
    except (httpx.HTTPError, asyncio.TimeoutError, TypeError, AttributeError, ValueError, KeyError) as exc:
        raise RuntimeError(f"Ollama could not return a response for {model}. "
                           "Check that the server is running and the model is installed.") from exc


def simulate(run_id, exp, p, conditions):
    with chat_connection() as post:
        for model, prefix, entries in conditions:
            initial = []
            rng = np.random.default_rng(p["seed"] + prefix * 101 + 14003)
            for t in range(p.get("warmup", 0)):
                b = int(rng.integers(3))
                a, raw, invalid = ask(post, model, initial, p["history_window"], p["temperature"], p["seed"])
                initial.append((a, b, int(RPS[a, b])))
                yield Event(type="warmup", data=dict(condition=f"{model} · Prefix {prefix + 1}", step=t + 1,
                            total=p["warmup"], a=ACTIONS[a], b=ACTIONS[b], raw=raw, invalid=invalid))
            for condition, strategy in entries:
                history = initial.copy()
                rng = np.random.default_rng(p["seed"] + prefix * 101 + 8021)
                score = Score(run_id, exp, condition)
                size = 0 if condition == "No history" else p["history_window"]
                for t in range(p["rounds"]):
                    shown = [{"a": ACTIONS[a], "b": ACTIONS[b], "reward_a": r}
                             for a, b, r in window(history, size)]
                    b, phase = choose_b(strategy, history, t, rng)
                    a, raw, invalid = ask(post, model, history, size, p["temperature"], p["seed"])
                    row = score.record(a, b, phase=phase, history=shown,
                                       detail=dict(raw=raw, invalid=invalid, strategy=strategy, model=model,
                                                   prefix=prefix + 1 if exp == "l4" else None,
                                                   fallback="Rock" if invalid else None,
                                                   explanation=("Invalid output. The reference protocol scores Rock as a fallback."
                                                                if invalid else "A selects a move from the text prompt. Model weights stay frozen.")))
                    history.append((a, b, int(row.reward_a)))
                    yield Event(type="round", data=row.model_dump(exclude_none=True))
