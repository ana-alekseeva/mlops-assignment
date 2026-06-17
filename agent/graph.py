"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# Iteration 3: lowered 3 -> 2 (1 generate + at most 1 revise). The eval pass
# rate is flat across iterations (iter_0 == iter_2) - the 2nd revise spends a
# 3rd serial vLLM call without recovering accuracy, so cutting it trims tail
# work from the decode-bound path at ~no accuracy cost.
MAX_ITERATIONS = 2

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


@lru_cache(maxsize=1)
def llm() -> ChatOpenAI:
    """Shared chat client pointed at VLLM_BASE_URL (your local vLLM by default).

    Iteration 1: cached as one instance for the whole process instead of built
    per node call. A fresh ChatOpenAI opens a new httpx connection pool each
    time, so the old per-call construction churned sockets across the 2-3 calls
    per request and exhausted ephemeral ports under load (connection-reset
    client errors). One pooled client fixes that; the bounded timeout + retries
    stop a call stuck in vLLM's queue from hanging to the caller's 120s cap.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        timeout=10.0,
        # Iteration 5: max_retries 2 -> 1 to bound the worst case. A call that
        # exceeds the 10s timeout was retried up to 2x - each retry re-queues and
        # re-prefills, stacking to ~30-40s and driving the latency max. One retry
        # still rides out a transient blip but caps the tail.
        max_retries=1,
        # Iteration 3: bound the decode budget per call. Outputs are short (a
        # single SELECT, or a one-line verdict); 512 is ample yet caps a runaway
        # generation from holding decode slots in the decode-bound batch.
        max_tokens=512,
    )


# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (fenced.group(1) if fenced else text).strip()


async def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    Iteration 1: async (`ainvoke`) so the request runs on the event loop, not a
    bounded threadpool worker - the LLM call is I/O-bound on vLLM, so awaiting
    it lets hundreds of requests progress concurrently.
    """
    response = await llm().ainvoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{"node": "generate_sql", "sql": sql}],
    }


async def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result.

    Iteration 1: the blocking sqlite call is offloaded to a worker thread
    (`asyncio.to_thread`) so a slow query can't stall the event loop and freeze
    every other in-flight request.
    """
    return {"execution": await asyncio.to_thread(execute_sql, state.db_id, state.sql)}


def _parse_verdict(text: str) -> tuple[bool | None, str]:
    """Recover (ok, issue) from the verifier's reply, defensively.

    Iteration 3: the verdict contract is the compact "OK" / "BAD: <issue>" form
    (one token on the common accept path) instead of a JSON object - decode cost
    scales with output tokens x concurrency, and verify runs on every request,
    so trimming the happy verdict ~12 -> 1 token removes most of verify's decode
    contribution without dropping the check. A JSON fallback is kept so an
    old-style {"ok":..,"issue":..} reply still parses. Returns (None, snippet)
    when no verdict can be recovered, so the caller picks a fallback.
    """
    t = text.strip()
    low = t.lower()
    if low.startswith("ok"):
        return True, ""
    if low.startswith("bad"):
        # Drop the leading "BAD" and any ":"/"-"/space separator.
        return False, t[3:].lstrip(" :-\t").strip()
    # Backward-compatible fallback: a JSON object {"ok": bool, "issue": str}.
    match = re.search(r"\{.*\}", t, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            ok = obj.get("ok")
            if isinstance(ok, bool):
                return ok, str(obj.get("issue", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return None, t[:200]


async def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Build messages from the VERIFY_* prompts, call llm(), parse a small
    {"ok": bool, "issue": str} verdict defensively. A failed execution can
    never be a valid answer, so we force ok=False in that case regardless of
    what the model says; an unparseable verdict on a *successful* run is
    treated as "accept" rather than looping on a parse glitch.
    """
    result = state.execution
    # Iteration 5: shrink the verify-path prefill. verify runs on EVERY request and
    # re-prefills the (uncached) execution result; it only needs the answer's SHAPE
    # - columns + a few representative values - to judge plausibility, not the full
    # blob. Cap it hard here (3 rows x 80 chars); revise keeps the wider 200-char
    # render since it has to see the data to fix the query.
    result_text = result.render(max_rows=3, max_cell=80) if result is not None else "ERROR: no execution result"

    response = await llm().ainvoke([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            question=state.question,
            sql=state.sql,
            result=result_text,
        )),
    ])
    ok, issue = _parse_verdict(response.content)
    if ok is None:
        ok, issue = True, ""

    if result is None or not result.ok:
        ok = False
        if not issue:
            issue = result.error if result is not None else "no execution result"

    return {
        "verify_ok": ok,
        "verify_issue": issue,
        "history": state.history + [{"node": "verify", "ok": ok, "issue": issue}],
    }


async def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt carries the failing SQL,
    its execution result, and the verifier's complaint so the model can fix the
    specific problem. Bumps `iteration` so route_after_verify can terminate.
    """
    result = state.execution
    result_text = result.render(max_cell=200) if result is not None else "ERROR: no execution result"

    response = await llm().ainvoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            result=result_text,
            issue=state.verify_issue,
        )),
    ])
    sql = _extract_sql(response.content)
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [
            {"node": "revise", "sql": sql, "addressed": state.verify_issue}
        ],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    End when the verifier is satisfied or the iteration cap is reached;
    otherwise loop back through revise -> execute -> verify.
    """
    if state.verify_ok:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
