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
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 3

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

    Phase 6 / iteration 1: this is cached (one instance for the whole process)
    instead of constructed per node call. A new ChatOpenAI builds a fresh httpx
    connection pool each time, so the old code opened/tore down sockets on every
    one of the 2-6 LLM calls per request - under load that exhausts ephemeral
    ports and surfaces as connection-reset 'client errors'. Reusing one pooled
    client fixes that and cuts per-call connect latency. The bounded timeout +
    retries stop a call stuck in vLLM's queue from hanging until the caller's
    120s timeout (which showed up as load-test 'timeouts').

    Phase 6 / iteration 2: timeout tightened 60s -> 10s. The SLO budgets ~1.5s
    per call (5s end-to-end / ~3 calls), and a healthy call takes <1.3s (Phase 5),
    so any call past 10s is already an SLO miss and a slot-holder under load -
    failing it fast frees the worker instead of letting it block for a minute.
    """
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
        timeout=10.0,
        max_retries=2,
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

    Phase 6 / iteration 2: async (`ainvoke`) so the request runs on the event
    loop, not a bounded threadpool worker - the LLM call is I/O-bound on vLLM,
    so awaiting it lets hundreds of requests progress concurrently.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
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

    Phase 6 / iteration 2: the sqlite call is blocking, so it is offloaded to a
    worker thread (`asyncio.to_thread`) - otherwise a slow query (up to its 5s
    timeout) would stall the event loop and freeze every other in-flight request.
    """
    return {"execution": await asyncio.to_thread(execute_sql, state.db_id, state.sql)}


def _parse_verdict(text: str) -> tuple[bool | None, str]:
    """Pull {"ok": bool, "issue": str} out of an LLM reply, defensively.

    The model is asked for a bare JSON object, but may wrap it in prose or
    fences. Grab the first {...} span and parse it. Returns (None, snippet)
    when no boolean verdict can be recovered, so the caller picks a fallback.
    """
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            ok = obj.get("ok")
            if isinstance(ok, bool):
                return ok, str(obj.get("issue", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            pass
    return None, text.strip()[:200]


async def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Build messages from the VERIFY_* prompts, call llm(), parse a small
    {"ok": bool, "issue": str} verdict defensively. A failed execution can
    never be a valid answer, so we force ok=False in that case regardless of
    what the model says; an unparseable verdict on a *successful* run is
    treated as "accept" rather than looping on a parse glitch.
    """
    result = state.execution
    result_text = result.render() if result is not None else "ERROR: no execution result"

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
    result_text = result.render() if result is not None else "ERROR: no execution result"

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
