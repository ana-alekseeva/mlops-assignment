"""FastAPI wrapper exposing the agent over HTTP.

Run:
    uv run uvicorn agent.server:app --host 0.0.0.0 --port 8001

The /answer endpoint accepts {question, db, tags?} and returns the
agent's final SQL, the result rows, and per-iteration history.
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from pydantic import BaseModel

load_dotenv()

from agent.graph import MAX_ITERATIONS, VLLM_MODEL, AgentState, graph  # noqa: E402

# Short model id for tags (drop the HF org prefix), e.g. "Qwen3-30B-A3B-Instruct-2507".
_MODEL_SHORT = VLLM_MODEL.rsplit("/", 1)[-1]

# Langfuse tracing. With keys set we initialize the LangChain callback handler
# (langfuse 4.x imports it from langfuse.langchain) and keep a client handle so
# we can flush buffered traces on shutdown. Failures are NOT swallowed - a
# misconfigured Langfuse should not silently produce zero traces.
_lf_handler: Any = None
_lf_client: Any = None
if os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"):
    from langfuse import get_client
    from langfuse.langchain import CallbackHandler

    _lf_handler = CallbackHandler()
    _lf_client = get_client()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Best practice: flush on shutdown so the tail of a run (e.g. the last eval
    # questions) is delivered instead of dying in the buffer with the process.
    yield
    if _lf_client is not None:
        _lf_client.flush()


app = FastAPI(lifespan=lifespan)

# Prometheus instrumentation for the agent itself. vLLM's /metrics only sees the
# per-call inference layer; the end-to-end agent latency (the actual SLO) and the
# agent/HTTP-level failures the load driver counts live HERE and were previously
# invisible to Grafana. Exposed at /metrics and scraped as the "agent" job.
AGENT_REQUESTS = Counter(
    "agent_requests_total",
    "Agent /answer requests by outcome.",
    ["outcome"],  # ok | agent_error (200 but SQL failed) | exception (500)
)
AGENT_LATENCY = Histogram(
    "agent_request_latency_seconds",
    "End-to-end /answer latency - the SLO boundary (target P95 < 5s).",
    # Buckets span the SLO (5s) and the overload tail seen under load (>120s).
    buckets=(0.25, 0.5, 1, 2, 3, 5, 7.5, 10, 20, 30, 60, 120, float("inf")),
)
AGENT_INFLIGHT = Gauge(
    "agent_inflight_requests",
    "Agent /answer requests currently being processed (concurrency).",
)
# Mounted sub-app so GET /metrics returns the Prometheus exposition format.
app.mount("/metrics", make_asgi_app())


class AnswerRequest(BaseModel):
    question: str
    db: str
    tags: dict[str, str] = {}


class AnswerResponse(BaseModel):
    sql: str
    rows: list[list[Any]] | None
    iterations: int
    ok: bool
    error: str | None = None
    history: list[dict[str, Any]] = []


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _trace_metadata(req: AnswerRequest) -> dict[str, Any]:
    """Build Langfuse trace shaping for one request.

    The goal is Phase-6-grade analysis: every trace should say *which run, under
    what load, against which DB, and on what configuration* it was produced, so
    you can filter/group/compare iterations directly in the Langfuse UI.

    Caller-supplied tags (e.g. {"run": "iter1-pooled", "rps": "10",
    "phase": "load_test"}) become both filterable tag chips ("run:iter1-pooled")
    and structured metadata. We also stamp the server-side config (model,
    MAX_ITERATIONS) so a tuning change is visible per-trace, and group a whole
    run into one Langfuse session via session_id/run/phase.
    """
    tags: list[str] = ["sql-agent", "text-to-sql", f"db:{req.db}",
                       f"model:{_MODEL_SHORT}", f"max_iter:{MAX_ITERATIONS}"]
    tags += [f"{k}:{v}" for k, v in req.tags.items() if v]

    metadata: dict[str, Any] = {
        **req.tags,
        "db_id": req.db,
        "model": VLLM_MODEL,
        "max_iterations": MAX_ITERATIONS,
        # Reserved Langfuse keys consumed by the LangChain callback handler.
        "langfuse_trace_name": "sql-agent",
        "langfuse_tags": tags,
    }
    # Group all traces from one load-test / eval run under a single session so
    # you can open it and see the whole run's latency distribution at once.
    session_id = req.tags.get("session_id") or req.tags.get("run") or req.tags.get("phase")
    if session_id:
        metadata["langfuse_session_id"] = session_id
    return metadata


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest) -> AnswerResponse:
    # outcome/latency/in-flight are recorded in finally so every path - including
    # the 500 below - is counted. outcome stays "ok" unless we set it otherwise.
    t0 = time.monotonic()
    outcome = "ok"
    AGENT_INFLIGHT.inc()
    try:
        state = AgentState(question=req.question, db_id=req.db)
        config: dict[str, Any] = {
            "callbacks": [_lf_handler] if _lf_handler is not None else [],
            "metadata": _trace_metadata(req),
        }
        try:
            final = graph.invoke(state, config=config)
        except Exception as e:  # noqa: BLE001
            outcome = "exception"
            raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

        sql = final.get("sql", "")
        iteration = final.get("iteration", 0)
        history = final.get("history", [])
        execution = final.get("execution")

        if execution is None:
            outcome = "agent_error"
            return AnswerResponse(
                sql=sql,
                rows=None,
                iterations=iteration,
                ok=False,
                error="agent produced no execution result",
                history=history,
            )
        if not execution.ok:
            outcome = "agent_error"
            return AnswerResponse(
                sql=sql,
                rows=None,
                iterations=iteration,
                ok=False,
                error=execution.error,
                history=history,
            )

        return AnswerResponse(
            sql=sql,
            rows=[list(r) for r in (execution.rows or [])],
            iterations=iteration,
            ok=True,
            history=history,
        )
    finally:
        AGENT_LATENCY.observe(time.monotonic() - t0)
        AGENT_REQUESTS.labels(outcome=outcome).inc()
        AGENT_INFLIGHT.dec()
