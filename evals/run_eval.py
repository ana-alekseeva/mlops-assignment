"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str, run_label: str = "eval_baseline") -> dict:
    """Score one question, capturing per-iteration correctness.

    Calls the agent over HTTP, then reconstructs the SQL it held after each
    generate/revise step from the returned `history`, executes each against
    the target DB, and compares the canonicalized rows to the gold query's.
    """
    db_id = question["db_id"]
    gold_sql = question.get("gold_sql", "")
    q_text = question["question"]

    # --- call the agent over HTTP ---
    payload = {"question": q_text, "db": db_id, "tags": {"phase": "eval", "run": run_label, "db": db_id}}
    t0 = time.monotonic()
    try:
        resp = httpx.post(agent_url, json=payload, timeout=180.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        return {
            "db_id": db_id, "question": q_text, "gold_sql": gold_sql,
            "final_sql": "", "final_correct": False,
            "iterations": 0, "num_steps": 0, "per_iteration_correct": [],
            "agent_ok": False, "agent_error": f"{type(e).__name__}: {e}",
            "gold_error": None, "latency_s": round(time.monotonic() - t0, 3),
        }
    latency = round(time.monotonic() - t0, 3)

    history = data.get("history") or []
    final_sql = data.get("sql") or ""

    # The SQL the agent held after each generate_sql / revise step, in order.
    # history index k corresponds to the README's "iteration k".
    step_sqls = [h["sql"] for h in history if h.get("node") in ("generate_sql", "revise") and h.get("sql")]
    if not step_sqls and final_sql:
        step_sqls = [final_sql]

    # Gold executed once; reused for every comparison.
    gold_ok, gold_rows, gold_err = run_sql(db_id, gold_sql)

    def is_correct(sql: str) -> bool:
        if not sql or not gold_ok:
            return False
        ok, rows, _ = run_sql(db_id, sql)
        return ok and matches(gold_rows, rows)

    per_iteration_correct = [is_correct(sql) for sql in step_sqls]
    final_correct = (
        is_correct(final_sql) if final_sql
        else (per_iteration_correct[-1] if per_iteration_correct else False)
    )

    return {
        "db_id": db_id,
        "question": q_text,
        "gold_sql": gold_sql,
        "final_sql": final_sql,
        "final_correct": final_correct,
        "iterations": data.get("iterations", len(step_sqls)),
        "num_steps": len(step_sqls),
        "per_iteration_correct": per_iteration_correct,
        "agent_ok": bool(data.get("ok", False)),
        "agent_error": data.get("error"),
        "gold_error": gold_err,
        "latency_s": latency,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    n = len(results)
    if n == 0:
        return {
            "n": 0, "num_correct": 0, "overall_pass_rate": 0.0,
            "pass_rate_by_iteration": {}, "iteration_distribution": {},
            "questions_with_revision": 0, "agent_failures": 0, "avg_latency_s": 0.0,
        }

    num_correct = sum(1 for r in results if r["final_correct"])

    # Horizon = the deepest iteration any question actually reached.
    horizon = max((r["num_steps"] for r in results), default=1) or 1

    pass_rate_by_iteration: dict[str, float] = {}
    for k in range(horizon):
        passed = 0
        for r in results:
            pi = r["per_iteration_correct"]
            if not pi:
                val = False
            elif k < len(pi):
                val = pi[k]
            else:  # carry-forward: the agent had already terminated
                val = pi[-1]
            passed += 1 if val else 0
        pass_rate_by_iteration[f"iter_{k}"] = round(passed / n, 4)

    dist: dict[int, int] = {}
    for r in results:
        dist[r["num_steps"]] = dist.get(r["num_steps"], 0) + 1

    return {
        "n": n,
        "num_correct": num_correct,
        "overall_pass_rate": round(num_correct / n, 4),
        "pass_rate_by_iteration": pass_rate_by_iteration,
        "iteration_distribution": {str(k): dist[k] for k in sorted(dist)},
        "questions_with_revision": sum(1 for r in results if r["num_steps"] > 1),
        "agent_failures": sum(1 for r in results if r.get("agent_error")),
        "avg_latency_s": round(sum(r.get("latency_s", 0.0) for r in results) / n, 3),
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument(
        "--run-label",
        default="eval_baseline",
        help="Langfuse run/session tag, e.g. eval_baseline, eval_after_tuning",
    )
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url, args.run_label))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
