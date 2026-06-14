# Report

## Phase 1 — vLLM Serving Configuration

### Fixed constraints
- **Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` — MoE, ~30.5B total / ~3.3B active params per token, served in bf16.
- **Hardware:** 1× H100 80GB.
- **Workload:** 1.5–3K-token prompts, short structured (SQL) outputs, ~2–3 dependent model calls per user request.
- **SLO target:** P95 end-to-end agent latency < 5s at ≥10 RPS over a 5-minute window.

### Current configuration

Launched via [`scripts/start_vllm.sh`](scripts/start_vllm.sh). Verified live: `GET /v1/models` returns the model with `max_model_len: 32768`, and manual text-to-SQL queries return sensible `SELECT` statements (see `screenshots/vllm_manual_query.png`).

| Flag | Value | One-line justification |
|------|-------|------------------------|
| `--model` | `Qwen/Qwen3-30B-A3B-Instruct-2507` | Fixed by the assignment; bf16 on the H100 so reported quality/latency are spec-compliant. |
| dtype | bf16 (auto) | H100-native; full-precision baseline to measure before trading any quality for speed. |
| `--max-model-len` | 32768 | Caps context and therefore KV-cache per sequence — **initial value is generous; the workload only needs ~4K** (≤3K prompt + short output), so this is the first lever to tighten (see below). |
| `--tensor-parallel-size` | 1 (default) | bf16 weights (~57 GiB loaded) fit in one 80 GB H100; sharding would add NCCL overhead for no memory benefit. |
| `torch.compile` | on (default) | Compiled graphs / CUDA graphs cut per-token latency, directly serving the P95 SLO. Requires `python3.12-dev` on the host (documented in README setup). |
| `--gpu-memory-utilization` | 0.90 (default) | Leaves headroom on the 80 GB card; the KV-cache pool is sized from the remainder. |
| `--max-num-seqs` | 256 (default) | Max concurrent sequences — the batching-vs-latency knob; to be tuned against the 10 RPS / P95 target once the agent and load test exist. |
| `--host` / `--port` | `0.0.0.0` / `8000` | Exposes the OpenAI-compatible API on the port the agent, the Prometheus scrape, and the SSH port-forward all expect. |

### Verification

`GET /v1/models` confirms the server is live (`max_model_len: 32768`, `owned_by: vllm`, `vllm-0.23.0`). A manual `temperature: 0` text-to-SQL request returns clean SQL — no prose, no markdown fences:

- **System:** *expert SQLite text-to-SQL assistant; reply with ONLY the SQL query.*
- **User:** schema `students(id, name, age, gpa, major)` + question *"Names of the top 3 Computer Science students by GPA?"*

```sql
SELECT name FROM students WHERE major = 'Computer Science' ORDER BY gpa DESC LIMIT 3;
```

`finish_reason: stop`, 83 prompt / 21 completion tokens. Captured in `screenshots/vllm_manual_query.png`.

### Workload → lever mapping (to tune in Phases 5–6)
- **Prompt shape (1.5–3K in, short out):** drop `--max-model-len` to ~4096. KV-cache per sequence shrinks ~8×, so more sequences fit in the same VRAM → higher achievable throughput at fixed latency.
- **MoE (3.3B active of 30.5B):** compute-light per token but memory-heavy weights — the win is batching many concurrent decodes, so prioritize concurrency (KV headroom, `--max-num-seqs`) over single-stream speed.
- **Latency SLO:** keep `torch.compile`/CUDA graphs on; evaluate FP8 weights + KV-cache to relieve the decode-time memory-bandwidth bottleneck and free KV space, validating quality against the eval set before adopting.
- **2–3 dependent calls/request:** the 5s end-to-end budget is roughly per-call latency × 3, so each call needs ≈1.5s P95 — reinforcing tight context length and effective batching.

### Status
Initial baseline is serving and returning correct SQL. Per Phase 1 step 3, these flags will be revisited once the agent is running and the load test reveals the real latency/throughput curve; changes will be logged in the Phase 6 iteration section below.

---

## Phase 3 — Agent

**Architecture.** A LangGraph text-to-SQL agent ([`agent/graph.py`](agent/graph.py)) with a self-consistency-style verify/revise loop:

```
attach_schema -> generate_sql -> execute -> verify --ok--> END
                                    ^                  |
                                    |                not-ok
                                    +------ revise <----+
```

- `generate_sql` / `verify` / `revise` are vLLM calls; `execute` runs the SQL read-only against the target sqlite DB.
- `verify` returns a `{"ok", "issue"}` verdict, parsed defensively; a failed execution can never be judged `ok`, so a broken query always routes to revise.
- The **verify→revise loop is capped at `MAX_ITERATIONS = 3`** (1 generate + up to 2 revises) so it always terminates.

**Serving.** The agent is a FastAPI app ([`agent/server.py`](agent/server.py)) run as a host process — **listening on port 8001** (`POST /answer`, `GET /health`) — not a container, so it can reach vLLM on `localhost:8000` and read the local BIRD sqlite files directly. Langfuse tracing is attached when keys are present.

**Interactive test (5 questions).** The loop fires and terminates correctly: **2 of 5 questions were revised**; the other 3 passed `verify` on the first attempt, and there were no agent failures.

| metric | value |
|--------|-------|
| questions | 5 |
| overall pass rate | 0.4 (2/5) |
| pass rate by iteration | iter_0 = 0.4, iter_1 = 0.4, iter_2 = 0.4 |
| iteration distribution | 1 step ×3, 3 steps ×2 |
| questions revised | 2 |
| agent failures | 0 |
| avg latency | 0.85 s |

The loop **engages** (2 revisions, both hitting the 3-step cap) but on this small sample does not yet raise accuracy — the pass rate is flat across iterations (iter_0 == iter_2). Calibrating the verify/revise prompts so revisions actually correct wrong answers is tracked in Phase 6.

---

## Phase 5 — Evaluation

**Method.** [`evals/run_eval.py`](evals/run_eval.py) reads the 30 curated questions in `evals/eval_set.jsonl`, calls the agent over HTTP (`POST /answer`), and scores by **execution accuracy**: the agent's final SQL and the gold SQL are both run against the target BIRD sqlite DB and their result sets compared after canonicalization (rows sorted, cells stringified, `NULL`→`''`). This is robust to the many syntactically-different-but-equivalent ways to write the same query. To get the per-iteration signal, `eval_one` reconstructs the SQL the agent held after each `generate_sql`/`revise` step from the returned `history` and executes each one; `summarize` then carries the last value forward for questions that terminated early, so "pass rate at iteration k" answers *"what would accuracy be if we always stopped after step k?"*. The gold query is executed once per question and reused. Run end-to-end with:

```bash
uv run python evals/run_eval.py --out results/eval_baseline.json
```

**Baseline result** (`results/eval_baseline.json`, 30 questions, ~40.7 s wall-clock, 0 agent failures, avg 1.28 s/question):

| metric | value |
|--------|-------|
| overall execution accuracy | **0.333 (10/30)** |
| pass rate @ iter 0 | 0.300 (9/30) |
| pass rate @ iter 1 | 0.333 (10/30) |
| pass rate @ iter 2 | 0.333 (10/30) |
| iteration distribution | 1 step ×18, 2 steps ×5, 3 steps ×7 |
| questions revised (>1 step) | 12 |
| agent failures | 0 |

**Is the loop doing real work?** Marginally, but it is net-positive. The verify→revise loop **engages on 12 of 30 questions**, yet the pass rate moves only from **0.300 (iter 0) → 0.333 (iter 1)** and then flattens. Reconstructing each question's trajectory:

- **+1 gained** (wrong→right): the loop fixed exactly one question — *"Mention the reputation of users who had obtained the badge…"* (`codebase_community`).
- **0 lost** (right→wrong): no revision ever corrupted an already-correct answer — verify is at least not actively harmful.
- The other **11 of 12 revisions failed to flip** the answer to correct (most hit the 3-step cap still wrong).

So the architecture earns its keep — iter 3 accuracy is genuinely higher than iter 0, not flat — but the effect is small (+1 question, +3.3 pp) and almost entirely capped by **revision quality, not loop wiring**. The loop fires on the right questions but the revise prompt rarely repairs them. Inspecting the misses, the dominant failure modes are semantic rather than syntactic: wrong column choice (e.g. `A14` vs `A15` for the crime-count question), case-sensitive value mismatches (`'m'` vs `'M'` for gender), and over-complex date/string arithmetic — exactly the cases a sharper verify/revise prompt could catch. **Tightening the verify/revise prompts so revisions actually correct these is the primary Phase 6 lever.**

> **Outstanding deliverable:** `screenshots/grafana_eval_run.png` (Grafana dashboard captured *while* the baseline eval runs) is not yet in `screenshots/` — re-run the eval with Grafana open and capture it. The ~60-request burst (30 questions × ~2 vLLM calls, with 12 questions making a 3rd) is the load to watch.

## Phase 6 — Performance tuning
_TODO: iteration log — "saw X → hypothesized Y → changed Z → result was W" + Grafana screenshots._

## Phase 7 — Wrap-up
_TODO: final numbers, whether quality survived, what I'd do with more time._
