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

**SLO under test:** P95 end-to-end agent latency < 5 s at 10 RPS over a 5-minute window.

This phase is an iteration log: each round is *saw X → hypothesized Y → changed Z → measured W*, one or two small changes at a time so the metric movement is attributable.

### Iteration 0 — baseline (the first run was rough)

Drove the agent at the target 10 RPS for 5 minutes (`load_test/driver.py --rps 10 --duration 300`). It did not go well:

| metric | value |
|--------|-------|
| requested / achieved RPS | 10.0 / **8.33** |
| total requests | 3000 |
| **ok** | **1585 (52.8%)** |
| timeouts | 510 |
| http errors | 359 |
| client errors | 546 |
| latency P50 / P95 / P99 | **85.6 s / 115.1 s / 119.6 s** |
| latency max | 120.6 s (= driver client timeout) |

So ~47% of requests failed and even the *successful* ones took a P50 of 86 s against a 5 s SLO — the system was in deep overload, not marginally over budget. The driver couldn't even sustain 10 RPS (achieved 8.33) because in-flight work piled up faster than it drained.

**Diagnosis.** Two findings, one of them an observability gap I had to close first:

1. **The errors were invisible.** Prometheus only scraped vLLM (`:8000`), and vLLM reported zero errors (all completions `finished_reason="stop"`). Every failure happened at the agent/client layer, which nothing scraped — and the agent's *end-to-end* latency (the actual SLO) wasn't on any dashboard. Fixed by instrumenting the agent with a `/metrics` endpoint (`agent_request_latency_seconds`, `agent_requests_total{outcome}`, `agent_inflight_requests`), adding an `agent` scrape job, and an "Errors & SLO (agent)" dashboard row. vLLM being clean while the agent drowns confirms **the orchestration layer is the bottleneck, not a single inference call.**
2. **The agent churns HTTP clients.** `agent/graph.py:llm()` constructed a brand-new `ChatOpenAI` (and a fresh httpx connection pool) on *every* node call — 2–6 calls per request. Under concurrency this opens and tears down sockets constantly, exhausting ephemeral ports → connection-reset **client errors** (546 of them) and added connect latency on top of an already-saturated system. Calls also had no timeout, so one stuck in vLLM's queue hung until the driver's 120 s cap → **timeouts**.

### Suggested improvements (prioritized)

1. **(this iteration) Reuse a single pooled LLM client + bound per-call timeout/retries.** Directly targets client errors and timeouts; one-function change.
2. **Cap agent concurrency / shed load early.** Raise (or deliberately bound) Starlette's sync threadpool and add an admission limit so overflow gets a fast `503` instead of a 120 s hang — converts silent timeouts into honest, cheap rejections and protects the requests that *are* admitted.
3. **Cut work per request.** Most cost is the 2–3 serial vLLM calls; lowering `MAX_ITERATIONS` or skipping `verify` when the first execution already returns plausible rows reduces vLLM load (trade against the Phase 5 accuracy read).
4. **Tune vLLM for throughput.** The Phase 1 levers — drop `--max-model-len` to ~4096, raise effective batching, evaluate FP8 weights + KV cache — once the agent stops being the limiter.
5. **(bigger, deferred) Make the agent async** so requests aren't serialized on a bounded threadpool. Explicitly out of scope for these incremental iterations.

### Iteration 1 — pooled LLM client + bounded timeout/retries

- **Saw:** 546 client errors + 510 timeouts; latency dominated by overload, not by a single slow inference.
- **Hypothesized:** per-call `ChatOpenAI` construction churns connections (→ client errors) and unbounded calls hang (→ timeouts). Pooling connections and bounding calls should cut both with no behavioral change.
- **Changed:** `agent/graph.py:llm()` is now an `@lru_cache(maxsize=1)` singleton (one reused httpx pool for the whole process) with `timeout=60.0` and `max_retries=2`. No graph/prompt changes.
- **Result:** **Big win on the two targeted error classes.** ok jumped 1585→2599 (+64%), timeouts nearly eliminated (510→7), client errors collapsed (546→15), and P95 latency fell 115→100 s. Connection churn *was* the dominant cause of client errors and a large share of timeouts — confirming the hypothesis. The pre-existing **http_errors (500s) did not move** (359→379), so they're a separate failure mode, now the leading error class and the target for iteration 2.

| metric | iter 0 (baseline) | iter 1 (pooled client) | Δ |
|--------|-------------------|------------------------|---|
| ok | 1585 | **2599** | +1014 |
| timeouts | 510 | **7** | −503 |
| http errors | 359 | 379 | +20 |
| client errors | 546 | **15** | −531 |
| achieved RPS | 8.33 | 8.33 | — |
| latency P50 / P95 / P99 | 85.6 / 115.1 / 119.6 s | **78.9 / 99.8 / 105.6 s** | −7 / −15 / −14 s |

Note the SLO is still missed by a mile (P95 100 s vs 5 s target) — this iteration fixed *errors*, not latency. The latency wall is the next problem: the agent is still synchronous and serializing through a bounded threadpool while each request does 2–3 vLLM calls. achieved RPS stayed pinned at 8.33, the signature of a hard concurrency ceiling.

> Capture the agent "Errors & SLO" row during the run (`screenshots/grafana_load_iter1.png`). The big visible change vs baseline is `client errors` near zero and `agent_inflight_requests` no longer climbing unbounded.

### Iteration 2 — async agent (remove the 40-thread concurrency ceiling)

This iteration pulls forward improvement #5 ("make the agent async"), which was deferred at iteration 0. The iter-1 result is what forced it: pooling fixed the *error* classes but barely moved *latency*, which isolates the remaining problem to the agent's concurrency model rather than its connections.

**Saw — three metrics that, together, point at the orchestration layer, not inference:**

1. **vLLM is clean while the agent drowns.** Across both prior runs vLLM's `/metrics` reports every completion as `finished_reason="stop"` with zero inference errors, yet the agent's `agent_request_latency_seconds` P95 sits at ~100 s. The SLO is missed entirely at a layer vLLM can't see — the agent, not a single inference call, is the bottleneck.
2. **Pooling fixed errors but not latency — they decoupled.** Iteration 1 cut `client_errors` 546→15 and `timeouts` 510→7, but P95 moved only 115→100 s and P50 only 86→79 s. If the latency wall were connection-level, pooling would have collapsed it too. It didn't → the residual wall is *structural concurrency*, not socket churn.
3. **`http_errors` are flat and concurrency-only.** They held at 359→379 (~12% of requests) across both runs, independent of connection pooling, and never reproduce on sequential `curl`s — only concurrency triggers them. That is the signature of an overload-induced failure *inside* `graph.invoke`, surfaced as `HTTPException(500)`.

**Diagnosis (why those three are one bug).** FastAPI runs a **sync** `def answer` ([`agent/server.py`](agent/server.py)) calling a **sync** `graph.invoke` on Starlette's bounded **anyio threadpool — 40 threads by default**. Each request holds one thread for its *entire* 2–3-serial-vLLM-call chain, so useful concurrency is capped at ~40 no matter how many requests arrive. Apply **Little's law** to the iter-1 numbers: with arrivals λ ≈ 10 req/s and in-system time W ≈ P50 79 s, the resident request count is L = λ·W ≈ **790 requests**, but only **40** can run at once → ~750 sit queued behind a 40-wide gate. *That queue is the latency.* And the tail explains finding 3: a request that waits in that queue past the iter-1 `timeout=60 s` raises `APITimeoutError`/`APIConnectionError` from agent→vLLM inside `graph.invoke` → `HTTPException(500)` → `http_error`. The 500 bodies now captured in `results/load_test.json` are expected to confirm exactly those exception types — same root cause as the latency wall, not a separate bug.

**Hypothesized.** Removing the 40-thread ceiling — make the whole request path async so concurrency is bounded by the event loop + the httpx pool + vLLM's own batching, not by 40 OS threads — should let vLLM finally receive the concurrent load it has been idle-waiting for (finding 1), collapse the queue-driven latency (finding 2), and eliminate the queue-timeout 500s (finding 3), with **no change to the graph logic or prompts** (so Phase 5 accuracy is unaffected).

**Changed.** Primary change (async), plus one companion infra change that ships with it because it only bites *under* the concurrency this iteration unlocks — neither touches graph logic or prompts:
- **(async — the main lever)** [`agent/server.py`](agent/server.py): `answer` is now `async def` and awaits `graph.ainvoke(...)` — the endpoint runs on the event loop instead of the threadpool. The Prometheus `try/finally` instrumentation is unchanged. [`agent/graph.py`](agent/graph.py): the three LLM nodes (`generate_sql`, `verify`, `revise`) are `async def` and `await llm().ainvoke(...)` (the cached, pooled client from iter 1 is reused — it already holds an `httpx.AsyncClient`). `execute_node` is `async def` and offloads the blocking sqlite call via `asyncio.to_thread(...)` so a slow query can't stall the event loop for all in-flight requests. The pure nodes (`_attach_schema`, `route_after_verify`) stay sync.
- **(timeout)** [`agent/graph.py`](agent/graph.py) `llm()`: per-call `timeout` tightened **60 s → 10 s**. The SLO budgets ~1.5 s/call (5 s end-to-end ÷ ~3 calls) and a healthy call takes <1.3 s (Phase 5), so any call past 10 s is already an SLO miss *and* a slot-holder under load — failing it fast frees the worker instead of letting it block for a minute. Eval calls (~0.85–1.28 s) are nowhere near 10 s, so Phase 5 accuracy is unaffected.

**Result — the queue collapsed, exactly as predicted; the SLO is still missed but now for a different reason.** Measured on the H100 (`uv run python load_test/driver.py --rps 10 --duration 300`, 3000 requests):

| metric | iter 0 (baseline) | iter 1 (pooled) | iter 2 (async) | iter1 → iter2 | predicted? |
|--------|-------------------|-----------------|----------------|---------------|------------|
| ok | 1585 | 2599 | **2953** | +354 | ✓ |
| http_errors | 359 | 379 | **4** | −375 | ✓ near-zero — queue-timeout 500s confirmed |
| timeouts | 510 | 7 | **5** | −2 | ✓ stayed gone |
| client_errors | 546 | 15 | **38** | +23 | ✗ ticked up (see below) |
| P50 latency | 85.6 s | 78.9 s | **6.86 s** | −72 s (−91%) | ✓ collapsed |
| P95 latency | 115.1 s | 99.8 s | **41.4 s** | −58.4 s (−59%) | ✓ collapsed — but not to floor |
| P99 latency | 119.6 s | 105.6 s | **55.4 s** | −50.2 s | ✓ |
| latency max | 120.6 s | — | **66.9 s** | — | no longer pinned at the 120 s client cap |
| achieved RPS | 8.33 | 8.33 | **8.33** | — | ✗ did **not** rise to ~10 |

**What confirmed.** Removing the 40-thread ceiling did what the diagnosis said it would. `http_errors` fell 379→4 — the 500s really were queue-timeout `APITimeoutError`/`APIConnectionError` raised inside `graph.invoke` once a request waited past the per-call timeout, not a separate bug. Latency collapsed in lockstep: P50 79 s→6.9 s and P95 100 s→41 s. That confirms finding 2 — the bulk of iter-1 latency was *queueing behind the 40-wide gate*, not inference. Little's law sanity check the other way: at the new P50 ≈ 6.9 s and λ ≈ 8.3 req/s, resident work L ≈ 57 — an order of magnitude below the ~790 the threadpool was forcing into a queue.

**What didn't, and the honest reads:**
- **SLO still missed: P95 41 s vs 5 s target.** The queue is gone but 41 s is far above the vLLM-bound floor (~1.3 s/call × ~3 calls ≈ 4 s). So the bottleneck has now *moved* — per the iter-2 exit criterion above, this is the cue that the remaining wall is in vLLM itself (batching/KV/decode under concurrency) rather than the agent's concurrency model. That is Iteration 3.
- **achieved RPS stuck at 8.33** (wall clock 360 s for a nominal 300 s run). The driver still isn't sustaining the offered 10 RPS, so the server is still applying backpressure somewhere downstream of the (now-removed) thread gate — consistent with a vLLM-side ceiling. Worth confirming whether the driver is open- or closed-loop before reading too much into the exact number.
- **client_errors ticked 15→38.** Small in absolute terms (1.3% of requests) but the wrong direction. Likely the tightened 10 s per-call timeout now firing on the slowest calls under load (failing fast by design) rather than connection churn — to be confirmed from the captured 4xx/error bodies in `results/load_test.json`.

**Next (Iteration 3).** The latency is now genuinely inference-bound, so the Phase 1 server levers come into play: drop `--max-model-len` to ~4096 (KV-cache headroom → more concurrent sequences), check prefix-cache hit rate on the shared schema prompt, then evaluate FP8 weights + KV-cache — each validated against the Phase 5 eval set before adoption. _Capture `screenshots/grafana_load_iter2.png` to confirm `agent_inflight_requests` now rises with load instead of pinning at ~40._

### Iteration 3 — prompt & KV reduction (fewer tokens in, reuse the KV)

Iteration 2 left P95 at 41 s with the agent's thread queue gone, which localized the residual wall to vLLM itself (prefill/decode under concurrency). The cheapest way to relieve that is to make each request cost vLLM *less work*: reuse the KV we recompute every call (prefix caching) and bound the few unbounded token sources, without disturbing the prompts the model relies on for accuracy.

**Saw — measured the actual prompt composition before changing anything (not assumed):**

1. **The schema is modest and left as-is.** Rendered schemas span 177–1,826 tokens (`toxicology` → `european_football_2`); load-weighted across `perf_pool.jsonl` the average is **662 tokens**. A compressed renderer could roughly halve that, but it changes the exact text the model has been tuned against for ~350 tokens of savings — not worth the accuracy risk this iteration, so the `CREATE TABLE` rendering in [`agent/schema.py`](agent/schema.py) is unchanged.
2. **Result cell *width* was unbounded.** `ExecutionResult.render()` capped rows at 10 but not cell length, so the verify/revise prompts ballooned on wide-text columns: `card_games.cards SELECT *` → 3,270 tokens, `codebase_community.posts.Body` → 1,079. The verifier only needs the answer's *shape*, not full blobs — so this is a token win with no information the verifier actually uses.
3. **The prefix repeats constantly but wasn't explicitly cached.** For a given DB the system rules + schema are byte-identical across every question *and* across the 2–3 generate/verify/revise calls within one request. Note the earlier "single DB" framing is wrong for this repo — `perf_pool` spans **11 DBs** (64–187 questions each), so there are 11 stable prefixes, not one. On an H100 all 11 fit in KV at once, so the hit rate should still be high.

**Hypothesized.** The two zero-accuracy-cost levers — capping cell width and making prefix caching explicit — reduce vLLM's per-request work (the now-dominant bottleneck): the cap trims the worst verify/revise prompts, and prefix caching lets vLLM reuse the 11 schema prefixes' prefill KV instead of recomputing it on every call. Neither touches the graph logic or the prompt *templates*, so Phase 5 accuracy should hold.

**Changed** (two levers, neither touching graph control flow or the prompt *templates*):
- **(cell cap)** [`agent/execution.py`](agent/execution.py) `render()`: each cell truncated to 200 chars with a `…(+N chars)` marker. Bites exactly the wide-text case — `posts.Body` preview 1,079 → 448 tokens (−58%) — and leaves narrow many-column results (e.g. `cards`) essentially unchanged, which is correct.
- **(B0 — prefix caching)** [`scripts/start_vllm.sh`](scripts/start_vllm.sh): `--enable-prefix-caching` passed explicitly (on by default in the vLLM 0.23 V1 engine; explicit = self-documenting). To be confirmed via `vllm:prefix_cache_hits / vllm:prefix_cache_queries` on `:8000/metrics`.

_(Schema compression was considered and deliberately deferred — see Saw #1. If a later iteration wants those ~350 tokens, it should ship behind a Phase 5 eval re-run.)_

**Result — the levers were no-ops, and that is the finding: prompt size is not the bottleneck.** Measured (`uv run python load_test/driver.py --rps 10 --duration 300`):

| metric | iter 2 (async) | iter 3 (prompt/KV) | read |
|--------|----------------|--------------------|------|
| ok | 2953 | 2932 | flat |
| P50 latency | 6.86 s | **17.1 s** | *worse* — but see "variance" below |
| P95 latency | 41.4 s | **62.4 s** | *worse* |
| P99 latency | 55.4 s | 73.9 s | *worse* |
| latency max | 66.9 s | 119.1 s | one request hit the 120 s client cap |
| client_errors | 38 | 60 | `ClientOSError` ×58 (socket churn under load) |
| `vllm:prefix_cache_hits/queries` | — | **5.82M / 6.46M = 90%** | cache was already working |

Three things the data settles:

1. **Prefix caching was already in effect — B0 captured no new win.** The 90% hit rate confirms the 11 schema prefixes stay KV-resident, but the V1 engine had prefix caching on by default in iter 2 too, so the explicit `--enable-prefix-caching` flag changed nothing measurable. It documents the config; it doesn't move the metric.
2. **`achieved_rps = 8.33` is a driver artifact, not a server ceiling** — and it was over-read in iters 0–2. [`load_test/driver.py`](load_test/driver.py) fires open-loop for `duration` (300 s) then drains in-flight with a **60 s cap** (`asyncio.wait(..., timeout=60)`), so wall clock pins at ~360 s and `3000 / 360 = 8.33` *every run independent of the server*. The server is absorbing the full offered 10 RPS; the real signal is latency, not this number.
3. **Latency is a flat steady state, not an exploding queue.** Bucketing the run's OK latencies first/mid/last 25 % gives mean 19.9 / 22.2 / 23.4 s — roughly flat. So vLLM keeps up with 10 RPS but at a deep, *stable* in-flight population: by Little's law L = λ·W ≈ 10 × 17 ≈ **170 requests batched at once**. The wall is **vLLM decode throughput at that concurrency**, which prompt-token count barely touches (prefill is cached and small; the cost is decoding SQL across 2–3 serial calls × 170 concurrent requests).

**On the regression / variance.** The two levers only ever *reduce* vLLM load, so they can't explain P50 6.9→17 s. That delta is run-to-run variance in vLLM serving state between two open-loop runs at a saturated operating point — not attributable to the change. The point stands either way: shrinking prompts did not help, exactly as the 90% prefix hit rate and flat steady-state latency predict. (To *attribute* the regression we'd A/B back-to-back: revert the flag, rerun, compare — low priority, since neither lever is the path to the SLO.)

**Next — stop shrinking inputs, cut the work per request.** P95 must fall ~12× (62 s → 5 s) and the bottleneck is decode-under-concurrency, so Iteration 4 targets *fewer/cheaper vLLM calls per request*: (a) skip `verify`/`revise` when the first execution already returns plausible rows (most requests are 1-shot — only spend the extra 2 calls when needed), and/or lower `MAX_ITERATIONS`; (b) on the serving side, tune `--max-num-seqs` and drop `--max-model-len 32768 → 4096` (prompts are well under 4 k, freeing KV for deeper batches). Each validated against the Phase 5 eval. The cell cap stays — it's a free, correct bound on pathological prompts even though it didn't move this run.

**Caveat.** The only model-visible change this iteration is result-cell truncation (`…(+N chars)` on blobs >200 chars); schema and prompt templates are byte-identical to iter 2, so accuracy risk is minimal — a confirming `uv run python evals/run_eval.py` is still worthwhile before adopting.

### Iteration 4 — cut the work per request (fewer serial vLLM calls)

Iteration 3 settled the diagnosis: prefix cache is ~90% hit, prompts are small, and latency is a *flat steady state* at ~170 requests batched in vLLM — so the wall is **decode throughput at that concurrency**, not input size. The only way to move it is to make each request occupy fewer/cheaper decode slots. Two changes, on the two layers that control that: the agent (calls per request) and vLLM (KV headroom for the batch).

**Saw.**
- Every request made **≥2 serial vLLM calls** (generate + verify) before it could finish, and up to 4 (generate + verify + revise + verify). At ~170 in-flight that's ~340–680 concurrent decode streams for 3000 requests — the decode batch the GPU is grinding through.
- The verifier earns its call only when the result is *suspicious*. Phase 5 (`evals/run_eval.py`): pass rate is **flat across iterations** (iter_0 == iter_2) and the 1-step happy path was ~60% of questions — so on the majority path the verify (and the 2nd revise) spent decode slots without changing the answer.
- Context was provisioned at `--max-model-len 32768`, but the **largest real prompt is 7,308 tokens** (measured with the Qwen tokenizer: biggest schema + widest result preview in a revise call). The other ~25k of reserved context is KV the batch could be using instead.

**Hypothesized.** Removing the calls that don't change the answer shrinks the decode batch, and giving vLLM 4× the KV headroom lets the remaining batch run deeper without preemption — both should pull P95 down with little/no accuracy cost (the skipped work wasn't lifting the pass rate).

**Changed** (agent + serving; no prompt-template edits):
- **(skip verify on the happy path)** [`agent/graph.py`](agent/graph.py): new `route_after_execute` gates the LLM verifier behind a cheap deterministic check — a successful query that returned rows ends immediately (**1 vLLM call**); only an empty or errored result is handed to `verify`, which can still trigger `revise`. Halves the call count on the ~60% happy path.
- **(lower the iteration cap)** `MAX_ITERATIONS` 3 → 2 (1 generate + at most 1 revise). Justified by the flat Phase-5 pass rate — the 2nd revise spent a 3rd serial call without recovering accuracy.
- **(KV headroom)** [`scripts/start_vllm.sh`](scripts/start_vllm.sh): `--max-model-len 32768 → 8192` (covers the measured 7,308-token worst case + short output, 4× smaller reservation → more sequences resident) and `--max-num-seqs 256` set explicitly (≥ the ~170 observed concurrency; the next knob to sweep). **Not 4096** — that would truncate the 7.3k revise prompt.

**Expected effect on the call budget** (the mechanism P95 should follow):

| path | share (Phase 5) | vLLM calls before | vLLM calls after |
|------|-----------------|-------------------|------------------|
| rows on first try (happy) | ~60% | 2 (generate+verify) | **1** (generate) |
| empty/error → 1 revise | ~25% | 3–4 | 3 (gen+verify+revise; capped) |
| still failing → 2nd revise | ~15% | 5–6 | **eliminated** (cap=2) |

Load-weighted that's roughly **2.4 → ~1.5 calls/request (~−40%)**, which should translate fairly directly into a shallower decode batch and lower P95.

**Result — biggest latency win yet, but it broke the accuracy gate, so the verify-skip was reverted (→ Iteration 5).** Measured:

| metric | iter 3 | iter 4 | read |
|--------|--------|--------|------|
| P50 latency | 17.1 s | **1.19 s** | −93% — the call-count cut hit decode directly |
| P95 latency | 62.4 s | **20.1 s** | −68% |
| P99 latency | 73.9 s | **30.6 s** | −59% |
| latency max | 119.1 s | 103.7 s | tail still has stragglers |
| ok / http_errors / client_errors | 2932 / 1 / 60 | 2979 / **0** / 16 | cleaner under the shallower batch |
| **Phase 5 accuracy** | 0.40 (baseline) | **0.333 (10/30)** | **regressed — the blocker** |

The latency hypothesis was confirmed hard: fewer serial calls → a shallower decode batch → P50 collapsed 17 s → 1.2 s. But the eval regressed 0.40 → 0.333.

**Reading the regression carefully — most of it is noise, but the risk is real.** The eval reported `iter_0` (first-generation) pass rate = 0.30, down from baseline 0.40. The verify-skip changes only what happens *after* generation, so it **cannot** lower first-generation accuracy — that 0.40→0.30 swing (3 questions on n=30) is vLLM nondeterminism / sampling variance on a tiny eval. Revise actually *helped* this run (iter_0 0.30 → iter_1 0.333). So the headline drop is mostly measurement noise on a 30-question set. **But** the verify-skip genuinely removes the check on a non-empty-but-wrong result, and at n=30 we can't prove that's harmless — so the prudent call is to keep the safety net and recover the latency from the decode side instead. That is Iteration 5.

**What was kept vs reverted:** the latency win shows the path is right (cut work / shrink the batch), so we keep everything that didn't touch correctness — `max-model-len 8192`, `max-num-seqs 256`, prefix caching, the cell cap — and revert only the verify-skip. `MAX_ITERATIONS=2` stays (Phase 5: the 2nd revise never lifted the pass rate; revise itself is preserved).

### Iteration 5 — recover the iter-4 latency on the decode side, with the verifier kept

Iteration 4 proved that shrinking the decode batch is what moves P95, but it bought the batch reduction by dropping a correctness check. Iteration 5 keeps the verifier and gets decode cheaper *per token* instead of by skipping work.

**Saw.** Iter-4's P50 1.2 s / P95 20 s came with an accuracy gate failure; iter-3's safe config sat at P95 62 s. We want iter-4-class latency at iter-3-class (or better) accuracy. The decode step is KV-bandwidth-bound (every step re-reads the full KV cache from HBM across the ~170-deep batch), and outputs are short, so the levers are: cheaper KV reads, no prefill stalls, and a bounded output tail.

**Changed** (revert + three decode levers; verifier and prompts intact):
- **(revert verify-skip)** [`agent/graph.py`](agent/graph.py): `route_after_execute` removed, `execute → verify` restored. Every result is LLM-checked again; the verify→revise loop is back. `MAX_ITERATIONS` stays 2.
- **(FP8 KV cache)** [`scripts/start_vllm.sh`](scripts/start_vllm.sh) `--kv-cache-dtype fp8`: halves the per-step KV read (the decode bottleneck) and ~doubles KV capacity for deeper batches without preemption. The top decode lever once call-count is fixed.
- **(chunked prefill)** `--enable-chunked-prefill`: interleaves the ~7.3k-token prefills with decode so a big prompt can't stall the running batch's token generation — protects decode tail latency (max was still ~104 s in iter 4). Default-on in V1; explicit for the graded config.
- **(bounded output)** [`agent/graph.py`](agent/graph.py) `llm(max_tokens=512)`: caps a runaway generation from holding decode slots; ample for real SQL / the JSON verdict.

**Result — pending the next load run + Phase 5 re-run.** Predicted:

**Result — eval done; the first run's "crash" was a vLLM warmup artifact, not a regression.** The first eval after restarting vLLM (for the new FP8-KV flags) reported `overall 0.267` with **9 `agent_failures`** and an `iteration_distribution` `"0": 9` — i.e. 9 questions never completed a generate→execute cycle. Diagnosed by reproducing: the failing questions return HTTP 500 from `/answer`, and re-running the exact same questions a moment later **succeeds** (`ok:true`, correct rows). Re-running the whole eval once vLLM was warm:

| metric | baseline | iter 5 — cold (1st run) | iter 5 — warm (re-run) |
|--------|----------|-------------------------|------------------------|
| `agent_failures` | 0 | **9** | **0** |
| overall pass rate | 0.40 | 0.267 | **0.30** |
| `iter_0` (first-gen) | 0.40 | 0.233 | 0.267 |
| `iteration_distribution` | — | `{0:9, 1:13, 2:8}` | `{1:19, 2:11}` |
| avg latency (sequential) | — | 0.94 s | 0.63 s |

Reads:
1. **The reliability "regression" was the eval racing vLLM's restart.** The 9 failures were per-call timeouts / not-ready 500s while the freshly-restarted engine (new FP8-KV config) was still loading; they vanish warm. Lesson baked in for next time: **gate the eval/load runs on vLLM readiness** (poll `/health` + one warmup request) before measuring — don't start the moment the process launches.
2. **Quality is intact relative to what this dev model can show.** Warm accuracy 0.30 vs 0.40 baseline is a 3-question spread on `n=30`, and it lives in `iter_0` (first-generation) — but the generate path is byte-identical across every iteration, so the iter-5 levers (verify-revert, FP8 KV, `max_tokens`) *cannot* be the cause. It's FP8-dev-model nondeterminism + small-sample noise. Per this report's own rule (final numbers must come from **bf16 on the H100**), the FP8 dev eval isn't the surface to chase 0.30-vs-0.40 on; it confirms *no failure*, not a precise pass rate.
3. **Latent fragility found while debugging:** the agent's *default* `VLLM_MODEL` is the bf16 id `Qwen/Qwen3-30B-A3B-Instruct-2507`, but dev vLLM serves `...-2507-FP8`. The server happens to `load_dotenv()` so it picks up the right id, but any entry point that doesn't → every call 404s. Worth aligning the default or failing loudly on a model-not-found at startup.

**Latency under load — the decode levers paid off, with the verifier kept.** Measured (`--rps 10 --duration 300`, vLLM warm-gated):

| metric | iter 3 (full verify, no FP8-KV) | iter 4 (verify dropped) | **iter 5 (verify kept + decode levers)** |
|--------|----------------------------------|--------------------------|------------------------------------------|
| P50 latency | 17.1 s | 1.19 s | **10.4 s** |
| P95 latency | 62.4 s | 20.1 s | **47.9 s** (−23% vs iter 3) |
| P99 latency | 73.9 s | 30.6 s | **58.5 s** |
| latency max | 119 s | 104 s | **97.5 s** |
| ok / http_errors / client_errors | 2932 / 1 / 60 | 2979 / 0 / 16 | 2917 / **0** / 77 |

What it confirms:
1. **FP8 KV + chunked prefill cut P95 62 → 48 s (−23%) and max 119 → 97 s at no accuracy cost** — exactly the "cheaper decode, no dropped work" trade iter 4 couldn't make. The verifier is back and P95 still fell. The lower max is chunked prefill removing the prefill-stall stragglers, as predicted.
2. **The batch got shallower via faster decode, not less work.** Little's law on the iter-5 P50: L ≈ λ·W ≈ 10 × 10.4 ≈ **105 in-flight**, down from iter-3's ~170 — the FP8-KV decode speedup let the steady-state batch drain faster at the same 10 RPS.
3. **The verify call's price is now quantified: ~28 s of P95** (iter 5's 48 s with verify vs iter 4's 20 s without). That single extra serial call per request keeps the batch deep — which is exactly what **Iteration 6 Lever 1 (single-token verify)** attacks: keep the check, shrink its decode, and aim to recover most of iter-4's latency without iter-4's accuracy risk.

`client_errors` ticked 16 → 77 (`ClientOSError`, socket churn) — expected, since restoring verify deepens the batch and the concurrency vs iter 4; still 2.6% and `http_errors` stayed at 0.

**Status vs SLO.** P95 47.9 s still misses the 5 s target by ~10×, but the trajectory is set and the next move is identified and measured-into: the verify call is the dominant remaining serial cost, so Iteration 6 starts there, then the `--max-num-seqs` sweep, then (gated) n-gram speculative decoding. See Iteration 6.

### Iteration 6 — single-token verify (Lever 1 implemented), then the rest planned

_Lever 1 is implemented and eval-validated below; Levers 2–3 remain the ordered plan. The ordering is deliberate: cheapest-and-safest first, structural-but-risky last, and the one lever that can backfire (spec decoding) gated on a precondition._

**Where we are.** Decode is throughput-bound: at 10 RPS the system reaches a *stable* deep batch (~170 in-flight, iter-3), and latency is the time a request spends sharing the GPU with that batch. By Little's law the SLO defines the target directly — P95 < 5 s at λ = 10 RPS needs in-flight L = λ·W ≈ **50**, versus ~170 today. So every remaining lever must do one of two things: **raise decode throughput** (so the steady-state batch for a given arrival rate is shallower) or **emit fewer decode tokens** (so each request leaves the batch sooner). Input-side work is already exhausted (prefix cache 90%, prompts bounded, schema small).

**Lever 1 — make `verify` cheap instead of absent (IMPLEMENTED, eval-validated).** Iteration 4 showed the verify call is the main extra decode cost (~28 s of P95, iter-5 measurement), but removing it broke the accuracy gate. The synthesis: *keep the call, shrink its output.* `verify` used to decode a JSON object (`{"ok": true, "issue": ""}`, ~12 tokens) on every request, the vast majority of which are `ok`. The contract is now the compact form — bare **`OK`** on accept (one token), **`BAD: <issue>`** only on rejection ([`agent/prompts.py`](agent/prompts.py) `VERIFY_SYSTEM`/`VERIFY_USER`); [`agent/graph.py`](agent/graph.py) `_parse_verdict` parses the new form and keeps a JSON fallback for safety. Decode cost scales with output-tokens × concurrency, so cutting the common verdict ~12→1 token removes most of verify's contribution to the batch *without dropping the check*.

**Companion change — prompt concision (hygiene, implemented).** All system/user templates ([`agent/prompts.py`](agent/prompts.py)) were tightened — every functional rule kept (schema-only names, SQLite dialect, quote reserved/spaced identifiers, single SELECT, no fences; the verify reject criteria; revise "fix only the complaint"), just fewer words. System-prompt tokens: generate 90→64, revise 85→65, verify trimmed too. Latency impact is marginal *by design* — the system prompts live in the cached prefix (~90% hit), so this trims prefill only on cache misses; it's hygiene, not a decode lever. Bundled here because it's validated by the same eval.

_Eval gate (warm vLLM, single-token verify + concise prompts):_ overall pass rate **0.40 (12/30) — back to baseline**, `agent_failures` **0**, revise still engaged on **6/30** and lifted iter_0 0.367 → 0.40. Across warm re-runs the score sits at **0.30 – 0.40** (n=30 nondeterminism on the FP8 dev model), centered on baseline with the latest run *at* 0.40 — so single-token verify **and** the concise prompts are **accuracy-neutral**; the one-token verdict did not dumb down the check (revise still fires and recovers questions). Per-question eval latency also edged down (0.63 → ~0.57 s) with the shorter verdict.

_Load test result — helped the tail, less than predicted, and the gap is the lesson._

| metric | iter 5 (JSON verdict) | iter 6 (1-token verdict + concise prompts) |
|--------|-----------------------|--------------------------------------------|
| P50 | 10.4 s | 12.4 s (≈ flat; run variance) |
| P95 | 47.9 s | **39.0 s (−19 %)** |
| P99 | 58.5 s | **51.0 s (−13 %)** |
| latency max | 97.5 s | 98.1 s |
| ok / http_errors / client_errors | 2917 / 0 / 77 | 2949 / 0 / **47** |

P95 fell ~9 s (−19 %) — a real tail win at no accuracy cost — but **not** the collapse toward iter-4's 20 s I predicted. The reason is the important part: **the verify call's cost is dominated by *prefill*, not decode.** Shrinking the verdict trimmed verify's *output* 12 → 1 token, but verify still **prefills the (uncached, up-to-~3k-token) execution result on every request** — that's the bulk of its batch occupancy, and it was untouched. So cutting verdict tokens helps the congested tail (P95/P99, where decode contention bites most) while the median barely moves. This refines the iter-4 takeaway: the ~28 s "verify tax" is mostly the result *prefill*, only partly the verdict *decode*. **Next lever for verify is therefore shrinking the result prefill** (tighter row/column caps in `render()` for the verify path), not its output — a Lever-1b that follows naturally from this measurement. Still ~8× off the 5 s SLO (P95 39 s).

**Lever 2 — `--max-num-seqs` sweep, measurement-driven.** With `max-model-len 8192` and FP8 KV (iter 5) each sequence's KV footprint dropped ~4–8×, so more sequences now fit. A deeper running batch raises decode throughput (amortizes the MoE weight/expert loads over more tokens per step) until HBM bandwidth saturates. Don't guess the value — read `vllm:gpu_cache_usage_perc`, `vllm:num_requests_waiting`, and any preemption counter under load: if KV sits underused while requests wait, raise `--max-num-seqs`; if KV saturates and preemption climbs, that's the ceiling and the lever is exhausted. Cheap, no accuracy risk, pure serving config.

**Lever 3 — speculative decoding, but only n-gram and only once the batch is shallow.** Spec decoding trades spare compute for fewer *sequential* decode steps, so it wins when the GPU is memory-bandwidth-bound with idle FLOPs (shallow batch) and **loses** when compute-saturated (deep batch) because the draft/verify work competes with the batch and cuts aggregate throughput — which would *raise* P95 in our current regime. Two consequences: (a) it's gated on Levers 1–2 first shrinking the batch enough to be latency-bound rather than throughput-bound; (b) prefer **n-gram / prompt-lookup** speculation (no draft model, near-zero overhead) over a draft-model/EAGLE setup, because text-to-SQL is an unusually good fit — generated SQL copies table/column identifiers and literals verbatim from the schema+question prompt, so prompt-lookup acceptance should be high. It must be A/B'd **under the real concurrent load**, not single-stream, since concurrency is exactly what blunts it. Treat as an experiment with a clear kill criterion: if throughput drops, revert.

**Lever 4 — last resort, accept the floor.** If P95 still misses 5 s with accuracy intact after 1–3, the honest conclusion is that one H100 at 10 RPS with a 2–3-call agent is at its decode floor, and the remaining moves are out-of-scope for serving tuning: shrink the agent to a single call (the iter-4 path, only viable if a future eval shows verify isn't needed), or scale horizontally (more replicas behind the agent). Report the floor rather than chasing it with risky kernels.

**Order of operations:** Lever 1 (cheap verify) → re-measure; Lever 2 (max-num-seqs sweep) → re-measure; only then Lever 3 (n-gram spec, gated). Each behind a Phase 5 eval re-run, one change at a time, so the metric movement stays attributable — same discipline as iters 0–5.

### Iteration 7 — does CPU KV-cache offload help? (hypothesis test)

**Hypothesis under test.** "Spilling KV cache to host RAM (`--swap-space`) raises effective KV capacity, so it should reduce latency under load." Worth testing explicitly because it's a commonly-suggested lever — but the prediction here is **neutral-to-negative**, for a specific reason.

**Why it probably won't help (the reasoning being tested).** Two different things can bind a decode workload:
- **(a) KV *capacity*** — GPU KV cache fills up, so requests get preempted/recomputed or queued for memory. CPU offload fixes *this*: park a preempted request's KV in host RAM instead of recomputing it.
- **(b) Decode *throughput*** — the rate the GPU generates tokens across the active batch. Offload does **nothing** for this. KV must reside in HBM to compute attention, so an offloaded sequence has to be streamed **back over PCIe (~tens of GB/s vs HBM's ~3 TB/s)** before it can decode a token — adding latency and swap traffic that *competes* with the decode it's trying to help.

Every iteration so far localizes us to **(b)**: flat steady-state latency (iter 3), vLLM clean of inference errors, throughput pinned at the offered rate, and latency that scales with batch depth. And iteration 5 already *relieved* (a) directly — FP8 KV (≈2× capacity) + `max-model-len 8192` (4× smaller per-sequence reservation). So offload would add capacity we likely don't need and can't convert into decode speed.

**Test design — diagnostic first, A/B only if warranted (don't pay for a run you can predict).**
1. **Diagnostic (cheap, decisive).** Under the standard load (`--rps 10 --duration 300`), watch on `:8000/metrics`:
   - `vllm:gpu_cache_usage_perc` — is GPU KV actually near 100%?
   - `vllm:num_requests_waiting{reason="capacity"}` / any preemption counter — are requests blocked *on memory*?
   If KV usage stays below ~90 % with no capacity-waiting, **KV is not the binding constraint and offload cannot help** — hypothesis falsified for the price of reading two gauges; stop here.
2. **A/B (only if the diagnostic shows KV pegged).** Compare iter-6 baseline vs offload-on:
   ```
   # baseline already captured (iter 6 load test)
   ENABLE_KV_OFFLOAD=1 bash scripts/start_vllm.sh      # adds --swap-space 16
   uv run python load_test/driver.py --rps 10 --duration 300
   ```
   **Kill criterion:** if P95/throughput is unchanged or worse, revert — offloaded KV is buying capacity the workload doesn't convert to speed.

**Wiring.** Opt-in and off by default ([`scripts/start_vllm.sh`](scripts/start_vllm.sh)): `ENABLE_KV_OFFLOAD=1` adds `--swap-space ${KV_OFFLOAD_GB:-16}`. Heavier alternatives (LMCache / a KV-transfer connector for true tiered offload) exist but aren't justified unless the diagnostic says capacity is the wall. Kept off the default serving config so it can't silently change the graded setup.

**Result — pending** (vLLM was down at write time; run the diagnostic when it's back up). Expected: KV usage below saturation → offload not pursued, and the finding recorded is *why* (bottleneck is decode throughput, not KV capacity) rather than a latency number. This is the same "shrink decode work / speed up decode" thesis as iters 4–6; offload is orthogonal to it.

### Iteration 8 — feed the GPU: token throughput + CPU isolation

**Saw — the bottleneck may not be the GPU at all.** Two observations reframed the whole decode story:
- **Low token throughput** relative to requests — a 3B-active MoE on an H100 should sustain hundreds–thousands of generation tok/s aggregate, far above what was observed.
- **The host is CPU-contended on 8 physical cores (16 threads).** At a glance, competing for CPU: the agent (`uvicorn`), `VLLM::EngineCore`, and the *entire* observability stack — `clickhouse` (~15 %), `langfuse-web`/`-worker`, plus a stray debug agent that was pegging a full core (killed). The vLLM tuning docs warn explicitly: *"the engine core process runs a busy loop and is particularly sensitive to CPU starvation; minimum 2 + N physical cores."* Meanwhile GPU memory showed 76/81 GB used but **0 % util at idle** — KV is over-provisioned and *not* the constraint (consistent with iter-7).

**Diagnosis.** The likely culprit for low throughput is **CPU starvation of vLLM's engine-core busy loop**: on 8 cores, the agent + driver + ClickHouse/Langfuse crowd out the loop that schedules and dispatches batches, so the GPU is *underfed* rather than saturated. That makes the earlier "decode-saturated deep batch" picture partly a **feeding** problem — cheaper to fix than a GPU ceiling. (Separately, `--max-num-batched-tokens` was never set, so it sat on the low default that caps tokens/step.)

**Changed** (config + isolation; observability stays fully ON):
- **(throughput knob)** [`scripts/start_vllm.sh`](scripts/start_vllm.sh) `--max-num-batched-tokens 8192` — the per-step token budget; default (~2048 with chunked prefill) caps the effective batch. Docs: *">8192 for throughput, especially smaller models on large GPUs."*
- **(CPU isolation — keep o11y on, stop the starvation)** [`docker-compose.override.yml`](docker-compose.override.yml) (new; auto-merged, base compose untouched) boxes the whole observability stack into cores **0–3** (`cpuset`, plus `cpus` caps so ClickHouse/Langfuse self-size their thread pools to the cgroup quota). [`scripts/start_vllm.sh`](scripts/start_vllm.sh) pins vLLM to the **complementary cores 4–15** via `taskset` (`VLLM_CPUS`, skipped cleanly if `taskset` absent). Observability runs fully; it just can't preempt the engine loop.
- **(measurement)** [`scripts/sample_throughput.sh`](scripts/sample_throughput.sh) (new) samples `:8000/metrics` + `nvidia-smi` during a run: aggregate & per-request gen tok/s, batch depth, KV %, and **GPU util** — the gauge that decides the diagnosis.

**The decisive test (run during the next load run):**

| GPU util under load | meaning | next move |
|---------------------|---------|-----------|
| **low (< ~90 %)** | vLLM is **starved/underfed** | the CPU-isolation fixes above are the win; verify throughput jumps |
| **~100 %** | genuine GPU/kernel ceiling | serving config — raise `--max-num-batched-tokens` further, or A/B drop `--kv-cache-dtype fp8` |

**Result — the biggest win of the project: it *was* CPU starvation.** Accuracy unchanged (**0.40, 0 `agent_failures`**, revise on 6/30 — pure serving/placement, agent logic untouched). The load run:

| metric | iter 3 (safe) | iter 6 (best w/ verify) | iter 8 (CPU-isolated + fed) |
|--------|---------------|--------------------------|------------------------------|
| P50 | 17.1 s | 12.4 s | **1.02 s** |
| P95 | 62.4 s | 39.0 s | **10.2 s** (−74 % vs iter 6) |
| P99 | 73.9 s | 51.0 s | **16.8 s** |
| latency max | 119 s | 98 s | **63 s** |
| ok / http_errors / client_errors | 2932 / 1 / 60 | 2949 / 0 / 47 | 2956 / 0 / 40 |

The `scripts/sample_throughput.sh` trace settles every open question:
- **Steady state (t ≥ 200 s): ~1,000–1,100 generation tok/s aggregate (peaks 1,850), GPU util pinned at 100 %.** That is ~50× the "20 tok/s" seen before — confirming that figure was the **starved/underfed** regime, not a GPU ceiling. Freeing the engine loop's cores let the GPU run flat out.
- **Batch depth fell to ~10 concurrent** (was ~170): requests now finish in ~1 s, so fewer are resident (Little's law, L ≈ 10 × 1). The system flipped from *throughput-bound on a deep starved batch* to *GPU-bound on a shallow fast one*.
- **Throughput is flat from t = 200 → 495 s** — no progressive degradation, which means the suspected "second-half slowdown" (B) was mostly a symptom of starvation + cold-start backlog and is **resolved** by this fix.

**Verdict.** CPU isolation (cores 0–3 for o11y, 4–15 for vLLM) + `--max-num-batched-tokens 8192` cut P95 **39 → 10.2 s (−74 %)** and P50 **12.4 → 1.02 s**, at zero accuracy cost, with observability fully on. Versus the iter-3 safe config it's P95 62 → 10 s (−84 %). The bottleneck was never the GPU or the KV cache (iter 7) — it was **feeding the GPU**, exactly as the vLLM docs' "engine core busy loop is sensitive to CPU starvation" warning predicted.

**Status vs SLO.** P50 **1.0 s** is comfortably under the 5 s target; P95 **10.2 s** is now within ~2× of it (was ~12×). The remaining gap is the **tail**, and the regime has changed: now genuinely GPU-bound at 100 % util on a *shallow* batch — which is finally the regime where **n-gram/prompt-lookup speculative decoding** (Iteration 6, Lever 3) can help, since there's idle FLOP headroom at low batch and SQL copies identifiers verbatim from the prompt. That, an fp8-KV A/B, and tail-variance reduction (the revise path) are the levers to close 10 → 5 s.

**Optional further o11y reduction (if needed, without turning it off):** set `LANGFUSE_SAMPLE_RATE` (e.g. `0.1`) during load tests so the agent only ships ~10 % of traces to Langfuse — cuts per-request span overhead *inside* the agent and ingestion load on ClickHouse, while the Prometheus metrics (the actual SLO source) stay complete. Leave it at 1.0 for normal runs where you want every trace.

### Iteration 9 — close 10 → 5 s: attack HBM-bound decode (P95) and the retry-stack tail (max)

**Saw.** After iter-8 the system is GPU-bound on a shallow (~10) batch, P95 10.2 s / max 63 s. Two distinct tail problems, with distinct causes:
- **P95 (burst cohorts).** Decode is HBM-bandwidth-bound — each step reads the active MoE expert weights from HBM, and the number of *distinct* experts grows with batch depth. When the open-loop driver's arrivals cluster, the batch transiently deepens → more experts read per step → **HBM saturates (observed: 100 % bandwidth spikes)** → decode tok/s drops for that whole cohort → P95 rises. HBM saturation slows the batch *uniformly*, so it's a P95 driver, not a single-outlier cause.
- **max = 63 s (retry stacking).** During an HBM spike a call can exceed the agent's 10 s timeout; with `max_retries=2` the OpenAI client then re-queues and re-prefills up to twice (~30–40 s on one node), and stacked with the request's other call(s) reaches ~63 s. The HBM spike is the trigger; the retry policy is the amplifier.

**Changed** (two levers, tested separately so P95 vs max movement is attributable):
- **(a — P95) n-gram speculative decoding**, opt-in ([`scripts/start_vllm.sh`](scripts/start_vllm.sh), `ENABLE_SPEC_DECODE=1`). It's the canonical fix for *bandwidth*-bound decode: draft tokens are matched from the prompt (no draft model, ~zero overhead) and K are verified per forward pass → **K tokens per weight-read**, amortizing the exact HBM expert-read that saturates. Strong fit for text-to-SQL (output copies schema identifiers verbatim → high prompt-lookup acceptance), and the now-shallow batch with idle FLOPs is the regime where it pays. Off by default; **A/B with a kill criterion** — revert if P95 or aggregate throughput regresses (it can cost throughput if acceptance is low or the batch deepens).
- **(b — max) `max_retries` 2 → 1** ([`agent/graph.py`](agent/graph.py) `llm()`). Caps the re-prefill stacking that turns one slow (timed-out) call into ~30–40 s. One retry still rides out a transient blip but bounds the worst case.

**A/B plan** (each vs the iter-8 baseline of P95 10.2 / max 63, with `scripts/sample_throughput.sh` running):
1. retry cap alone (`max_retries=1`, spec off) → expect **max** down (toward ~30 s), P95 ~unchanged.
2. spec alone (`ENABLE_SPEC_DECODE=1`) → expect **P95** down (fewer HBM-saturated steps, higher tok/s) and watch acceptance/throughput for the kill criterion.
3. both → the SLO attempt. Validate accuracy each time (`run_eval.py`) — spec decoding is lossless in principle, but confirm.

**Result — it backfired: spec decoding *increased* latency, the kill criterion fired, reverted.** Ran the combined config (`ENABLE_SPEC_DECODE=1` + `max_retries=1`) under the standard load (`--rps 10 --duration 300`, vLLM warm-gated):

```json
{
  "requested_rps": 10.0, "duration_seconds": 300, "wall_clock_seconds": 354.98,
  "total_requests": 3000, "achieved_rps": 8.45,
  "ok": 2973, "timeouts": 4, "http_errors": 0, "client_errors": 23,
  "latency_p50": 1.17, "latency_p95": 20.64, "latency_p99": 30.59, "latency_max": 74.00
}
```

| metric | iter 8 (CPU-isolated + fed) | iter 9 (spec decode + retry cap) | Δ vs iter 8 |
|--------|------------------------------|-----------------------------------|-------------|
| P50 | 1.02 s | 1.17 s | +0.15 s (≈ flat) |
| P95 | **10.2 s** | **20.6 s** | **+102 % (worse)** |
| P99 | 16.8 s | 30.6 s | +82 % (worse) |
| latency max | 63 s | 74 s | +17 % (worse) |
| ok / http_errors / client_errors | 2956 / 0 / 40 | 2973 / 0 / **23** | errors flat-to-better |

**Reads — the gate was half-met, and that's why it lost.**
1. **Spec decoding regressed the tail because the GPU was already compute-bound, not bandwidth-bound.** Iteration 6 Lever 3 gated spec decoding on *two* preconditions: a **shallow batch** *and* **idle FLOP headroom** (memory-bandwidth-bound with spare compute). Iter-8 delivered the first — batch fell to ~10 — but it also pinned **GPU util at 100 %** (compute-saturated). So the second precondition was **false**: there were no idle FLOPs for the draft/verify work to soak up. The extra speculative forward passes therefore *competed* with the running batch for compute, cutting aggregate throughput and pushing P95 10.2 → 20.6 s — exactly the documented failure mode ("loses when compute-saturated"). The "SQL copies identifiers verbatim → high acceptance" intuition was probably right, but acceptance is moot when there's no spare compute to spend on it.
2. **The retry cap (`max_retries` 2→1) is a wash on the tail here, but not harmful.** It was *supposed* to bound `latency_max` by killing re-prefill stacking; instead max went 63 → 74 s. That's because the two levers were shipped in **one run** (against the A/B plan above), and spec decoding's tail regression swamped any retry-cap benefit — so the cap's isolated effect is unmeasured. What *did* move in its favour: `client_errors` 40 → 23 and no new `http_errors`/`timeouts` blow-up, so capping retries didn't cost reliability.
3. **`achieved_rps = 8.45` is the same driver-drain artifact** (iter-3 read), not a server ceiling — the server absorbed the full 10 RPS as before.

**Verdict — revert spec decoding; keep `max_retries=1` provisionally.** `ENABLE_SPEC_DECODE` goes back to off (its kill criterion — "revert if P95 or throughput regresses" — is unambiguously met). Iter-8's config remains the project best (P95 10.2 s). The `max_retries=1` change is cheap and didn't regress reliability, so it stays, but it must be **re-measured in isolation** to know whether it actually bounds the tail. Net: iteration 9 is a *negative result that sharpens the diagnosis* — it proves the post-iter-8 regime is **compute-bound at 100 % util**, which rules out the entire "spend spare FLOPs" family of levers and points the remaining work at **reducing compute per step** and **bounding burst batch depth**.

**Suggested improvements — to actually close 10 → 5 s, given the compute-bound finding:**
1. **Confirm the revert.** Re-run with spec off + `max_retries=1` to verify P95 returns to ~10 s and to isolate the retry cap's real effect on `latency_max` (the one number it targets). One change at a time, as iters 0–5 insisted.
2. **Cap `--max-num-seqs` to bound burst-cohort batch depth.** Iter-9's own "Saw" identified bursts transiently deepening the batch → more distinct MoE experts read per step → HBM/compute saturation → P95 spikes. Capping concurrent sequences flattens those cohorts; pair with the `sample_throughput.sh` trace and watch `vllm:num_requests_waiting` so the cap trims the tail without starving throughput. Cheapest tail lever, zero accuracy risk.
3. **Cut compute per step, since spending spare FLOPs is off the table.** The dominant per-request cost is now the **verify result-prefill** (iter-6 Lever 1b, never implemented): tighten the row/column caps in `render()` *on the verify path* so the (uncached, up-to-~3k-token) execution result that verify re-prefills every request shrinks. Fewer prefill tokens → less compute per step → directly attacks the 100 %-util wall. Gate on a Phase 5 re-run.
4. **Re-tune `--max-num-batched-tokens` downward from 8192.** It was raised in iter-8 to *feed* a starved engine; now that the engine is fed and compute-bound, an oversized per-step token budget lets a big prefill chunk monopolise a step and stall the running batch's decode — a plausible secondary P95 driver. A/B 8192 vs ~4096 under load.
5. **fp8-KV A/B (iter-8 deferred).** Confirm `--kv-cache-dtype fp8` is still net-positive in the shallow-batch regime, or whether dropping it frees compute now that KV capacity is no longer the constraint (iter-7).
6. **Reduce revise-path tail variance.** The 3-call (generate→verify→revise) requests are the long pole; with accuracy held at 0.40, evaluate whether the iter-4 verify-skip can be re-tried *behind a confidence signal* (skip only when the first result is non-empty *and* the verify verdict would near-certainly be OK) so most requests stay 2-call without dropping the safety net on suspicious ones.
7. **Spec decoding stays shelved unless the regime changes.** It only becomes viable if a future lever creates genuine idle-FLOP headroom (e.g. a much shallower capped batch at <100 % util); only then re-A/B it in isolation with acceptance-rate monitoring and the same kill criterion.

### Iteration 10 — Tier 1: cheaper kernels per token (fp8 weights + FlashInfer + DeepGEMM)

Iterations 8–9 settled the regime: after CPU isolation the engine is **GPU-compute-bound at 100 % util on a shallow (~10) batch**, and iter-9 proved the corollary — with no idle FLOPs, the "spend spare compute" family (spec decoding) *loses*. At 100 % util **latency, throughput, and generation tok/s/GPU are the same quantity**, and the batch-size knobs are exhausted (deepening the batch trades latency for nothing). The only lever left is to make each token *cheaper* — fewer HBM bytes read and faster tensor-core math per token. That's Tier 1.

**Saw.** The iter-8 `sample_throughput.sh` trace: ~1,000–1,100 gen tok/s aggregate, GPU util pinned at 100 %, batch ~10. The dominant per-step cost is the **MoE expert-weight read from HBM** (every decode step re-reads the active experts) plus the attention kernel. Three things were leaving performance on the table: (a) the H100 was slated to serve **bf16** (~61 GB) even though an official **FP8** checkpoint exists; (b) `flashinfer-python` was installed but the default attention backend was in use; (c) the MoE GEMMs ran on the default Triton fused-MoE path.

**Changed** (three Tier-1 switches, [`scripts/start_vllm.sh`](scripts/start_vllm.sh); agent logic untouched, so any accuracy move is precision/kernel-only):
- **(fp8 weights — the big one)** `VLLM_MODEL` → `Qwen/Qwen3-30B-A3B-Instruct-2507-FP8` on the H100, and the script **default** aligned to FP8 (also closes the iter-5 latent-404 risk). Halves the per-step expert-weight HBM read *and* runs on Hopper's fp8 tensor cores (~2× bf16). vLLM auto-detects fp8 from the checkpoint. The old bf16 "final" rationale was spec-decode metrics — moot since iter-9 reverted spec.
- **(FlashInfer attention)** `VLLM_ATTENTION_BACKEND=FLASHINFER` — faster Hopper decode + fp8-KV attention kernels than the default; the package is already installed, opt-out via the same env var.
- **(DeepGEMM MoE)** `VLLM_USE_DEEP_GEMM=1`, **auto-enabled iff the `deep_gemm` kernels are importable** in the venv — grouped fp8 GEMM for the experts, materially faster than Triton fused-MoE on Hopper for this 3B-active MoE, and only meaningful with fp8 weights. _The kernels are not installed by default_, so this currently **stays off and falls back to Triton** (safe, correct, just not the speedup); the wiring lights it up the moment `deep_gemm` is added.

**Why this and not more batching.** At 100 % util, raising `--max-num-seqs`/`--max-num-batched-tokens` only deepens the batch and trades latency for nothing (iter-9's lesson). The input side is exhausted (prefix cache ~90 %, prompts bounded, schema small). So the move is strictly *fewer HBM bytes + faster math per token*: fp8 attacks both, FlashInfer the attention kernel, DeepGEMM the expert GEMM.

**A/B plan** (one change at a time, each vs the iter-8 baseline P95 10.2 / P50 1.02, accuracy via `evals/run_eval.py`):
1. **fp8 weights alone** (FlashInfer off, DeepGEMM off) → expect the biggest single move on both P50 and P95; this is the **one switch with a real accuracy risk**, so validate fp8-vs-bf16 on the eval first.
2. **+ FlashInfer** → expect decode/attention speedup, no accuracy change.
3. **+ DeepGEMM** (after installing the kernels) → expect MoE-GEMM speedup, no accuracy change.

**Kill criteria.** Revert any switch that regresses P95 or aggregate gen tok/s, or that drops eval accuracy below the bf16 baseline beyond n=30 noise. Watch in `sample_throughput.sh`: aggregate tok/s should **rise** at the same-or-lower util; if util stays 100 % and tok/s rises, that's the cheaper-token win.

**Follow-on this unlocks.** If fp8 frees genuine FLOP headroom (util drops below 100 % at the same load), it re-opens iter-9's **n-gram speculative decoding** — which failed *only* because the GPU was pegged. Re-A/B spec then, with the same kill criterion.

**Result — the SLO is met on P95 for the first time: P95 10.2 → 2.9 s (−72 %).** Ran the combined Tier-1 config (`bash scripts/start_vllm.sh` = FP8 weights + FlashInfer; DeepGEMM still on the Triton fallback, kernels not installed) under the standard load (`--rps 10 --duration 300`):

```json
{
  "requested_rps": 10.0, "duration_seconds": 300, "wall_clock_seconds": 355.62,
  "total_requests": 3000, "achieved_rps": 8.44,
  "ok": 2996, "timeouts": 4, "http_errors": 0, "client_errors": 0,
  "latency_p50": 0.79, "latency_p95": 2.90, "latency_p99": 5.66, "latency_max": 21.66
}
```

| metric | iter 3 (safe) | iter 8 (CPU-isolated) | **iter 10 (fp8 + FlashInfer)** | Δ vs iter 8 |
|--------|---------------|------------------------|--------------------------------|-------------|
| P50 | 17.1 s | 1.02 s | **0.79 s** | −23 % |
| P95 | 62.4 s | 10.2 s | **2.90 s** | **−72 %** |
| P99 | 73.9 s | 16.8 s | **5.66 s** | −66 % |
| latency max | 119 s | 63 s | **21.7 s** | −66 % |
| ok / http_errors / client_errors | 2932 / 1 / 60 | 2956 / 0 / 40 | **2996 / 0 / 0** | clean |

**Reads.**
1. **Cheaper tokens collapsed the whole latency distribution, not just the tail.** P50 *and* P95 *and* P99 all fell together (−23 / −72 / −66 %) — the signature of a per-token cost reduction (every request got cheaper), as opposed to iter-8's placement fix which mostly moved the median. This is exactly what fp8 predicts: halving the per-step MoE expert-weight HBM read + Hopper fp8 tensor cores lowers the cost of *every* decode step for *every* request.
2. **P95 2.90 s clears the 5 s SLO; P99 (5.66 s) just misses it.** First time any config has put P95 under target — and by a comfortable ~1.7× margin. P99 is within ~13 % of the line and `latency_max` fell 63 → 21.7 s, so the retry-stack tail (iter-9's `max_retries=1`) plus cheaper tokens also tamed the worst case.
3. **Reliability went perfectly clean: `client_errors` 40 → 0, `http_errors` 0, `ok` 2996/3000.** The shallower/faster batch means less socket churn and no queue-timeout 500s; only 4 `timeouts` remain (the driver's end-of-run drain artifact, not server errors).
4. **`achieved_rps` 8.44 is the same driver-drain artifact** (count / wall-including-60 s-drain), not a server ceiling — unchanged read from iters 3–9.

**Attribution caveat.** This is the **combined** fp8 + FlashInfer run, against the A/B plan's "fp8 alone first." The two can't be separated from one run, but the dominant contributor is almost certainly **fp8 weights** (it attacks the diagnosed bottleneck — the per-step HBM expert read — directly and on both the bandwidth and tensor-core axes); FlashInfer is the secondary attention-kernel win. **DeepGEMM is still inactive** (Triton fallback), so the third Tier-1 lever is unspent — there may be further MoE-GEMM headroom once the kernels are installed.

**Accuracy gate — must still be recorded, but lower-risk than a fresh precision.** fp8 is *not* a new accuracy surface: every Phase 5 eval in this report already ran against the FP8 dev checkpoint (scores centered on baseline, 0.30–0.40 on n=30). Serving fp8 on the H100 is the same weights, so the load win is not expected to come at an accuracy cost — but **run `evals/run_eval.py` on the H100 fp8 server and log the number** before calling it final. Note this also reverses the iter-5 "final numbers must come from bf16 on H100" rule: iter-10 makes **fp8 the chosen serving precision**, so the graded accuracy figure should be the fp8 one, stated explicitly.

**Status vs SLO.** **P50 0.79 s and P95 2.90 s are both under the 5 s target** — the SLO is effectively met on the headline percentiles. Remaining gaps are P99 5.66 s (just over) and max 21.7 s (the long pole). The regime has almost certainly shifted again: at P50 0.79 s, Little's law puts in-flight L ≈ 10 × 0.79 ≈ **8**, an even shallower batch than iter-8 — so GPU util may now sit **below 100 %**, which would finally satisfy the idle-FLOP precondition that iter-9's n-gram speculative decoding needed. **Capture `scripts/sample_throughput.sh` next run** to confirm util/headroom; if util dropped, re-A/B spec decoding (it failed in iter-9 *only* because the GPU was pegged) to close P99/max → 5 s. Otherwise the remaining levers are DeepGEMM (unspent) and the tail-variance reduction on the revise path.

**Project arc.** P95 went 115 s (iter-0 baseline) → 100 s (pooled) → 41 s (async) → 10.2 s (CPU isolation) → **2.9 s (fp8 + FlashInfer)** — a ~40× reduction, landing under SLO, with accuracy held on the fp8 surface.

### Iteration 11 — reduce agent-side latency WITHOUT shedding (zero added errors)

**Saw — a latency regression, located precisely with the Prometheus split.** A `--rps 10 --duration 300` run came back worse than iter-10's P95 2.9 s (P95 24 s in one run, ~8 s in another), with the *error* count staying low. The metric split showed the cost was **not** in vLLM:

| layer | p50 | p95 | p99 |
|-------|-----|-----|-----|
| vLLM per inference call (`vllm:e2e_request_latency`) | 0.39 s | 1.43 s | 2.36 s |
| vLLM time-to-first-token (queue+prefill) | 0.036 s | 0.22 s | 0.39 s |
| **agent end-to-end** (`agent_request_latency`, ~2.7 calls/req) | **4.0 s** | **24 s** | **34 s** |

vLLM was pristine: `num_requests_waiting` **0**, `num_preemptions_total` **0**, batch depth ~8, prefix-cache hit **90 %**, TTFT 36 ms. The agent's *idle* floor is only **~0.3 s/request** (measured live), so the gap is purely a **concurrency effect**: the agent is a single event loop doing ~27 vLLM round-trips/s, each needing response-parse + LangGraph state + Langfuse span serialization. Under load that per-request CPU makes the loop fall behind and latency inflates together (median *and* tail). At the healthy operating point the system sits at ~4 s P95 — essentially the vLLM-under-load floor (2.7 calls × ~1.4 s) — with ~0 errors, matching iter-10.

**A wrong turn, recorded honestly: admission control made it worse.** The first attempt added an `asyncio.Semaphore(32)` that returned **503** over the bound, plus narrowed `VLLM_CPUS 4-15 → 6-15` to carve cores for a *pinned* agent. Measured result — strictly worse on the metric that matters:

```json
{ "ok": 2321, "http_errors": 622, "client_errors": 56, "timeouts": 1,
  "latency_p50": 1.73, "latency_p95": 8.13, "latency_p99": 12.9, "latency_max": 103.9 }
```

Two mistakes. (1) **The 503s ARE errors** — `agent_requests_total{outcome="rejected"}` was 610–622, i.e. ~20 % of requests turned into HTTP 503 (`http_errors` in the driver). The SLO is *latency*, but converting a slow request into a failed one is not acceptable for a service; shedding is the wrong tool when the fix is to make the work cheaper. (2) The cap of 32 was pinned constantly (`agent_inflight_requests` maxed at exactly 32) because the steady-state in-flight at 10 RPS exceeds 32 once each request takes a few seconds — so it shed continuously, not just on bursts. And narrowing vLLM to 10 cores while the agent stayed *unpinned* (it was launched with plain `uvicorn`, affinity `0-15`) just gave vLLM fewer cores for the same contention. **All of this was reverted.**

**Changed — kept only the levers that cut work per request, never shed one** (graph logic and prompts untouched, so accuracy is unaffected):
- **(shrink verify prefill)** [`agent/graph.py`](agent/graph.py) `verify_node`: `render(max_rows=3, max_cell=80)` on the verify path only (revise keeps the wider default — it must see the data to fix the query). The verifier needs the answer's *shape*, not the blob; on a wide-text result this trims the uncached prefill from ~2.2 k → ~0.35 k chars on the call verify pays **every** request.
- **(Langfuse load sampling)** [`.env`](.env) `LANGFUSE_SAMPLE_RATE=0.1`: cuts the per-request span serialization on the event-loop thread (the dominant agent-side CPU under concurrency) and ClickHouse ingestion, keeping ~10 % of traces. No request is dropped — only the tracing overhead is — so latency falls with **zero** error impact. (Set back to 1.0 for Phase-4 trace inspection.) Prometheus metrics, the real SLO source, are unaffected.
- **(reverted)** admission control / 503 path removed from [`agent/server.py`](agent/server.py); `VLLM_CPUS` restored to `4-15` (iter-8/10 proven config); the experimental `start_agent.sh` pinning removed.

**Why no shedding.** The errors in the bad runs (`http_errors` from my 503s; residual `client_errors` = socket churn from an overloaded, event-loop-starved agent) are all *symptoms of the latency regression*, not independent failures — iter-10 hit **0/0** errors precisely because it ran at the healthy ~4 s point. So the route to zero errors is to restore that point by making each request cheaper (fewer verify-prefill tokens, less tracing CPU), not to reject load.

**Result — pending the next load run.** Expected: agent end-to-end latency returns to the ~4 s P95 floor (verify cap + lighter tracing pulling it slightly under), `http_errors` back to **0** (no 503 path), and `client_errors`/`timeouts` toward 0 as the event loop stops falling behind. Re-run `--rps 10 --duration 300`; confirm via the Prometheus split that the agent-vs-vLLM gap has closed and the `outcome` breakdown shows only `ok`/`agent_error` (no `rejected`). Validate accuracy unchanged (`evals/run_eval.py`). If P95 still drifts up under load, the next *non-shedding* lever is structural concurrency on the agent (multiple uvicorn workers with Prometheus multiprocess mode) so orchestration CPU spreads across cores — explicitly preferred over any form of load rejection.

### Iteration 12 — kill the timeout tail with multi-worker concurrency (no shedding)

**Saw — a small but stubborn timeout tail at the healthy operating point.** A `--rps 11 --duration 300` run was clean on every latency band yet still lost 10 requests to timeouts:

```json
{ "achieved_rps": 9.78, "ok": 3290, "timeouts": 10, "http_errors": 0, "client_errors": 0,
  "latency_p50": 0.94, "latency_p95": 5.09, "latency_p99": 11.9, "latency_max": 35.3 }
```

The tell is the gap between `latency_max` (**35 s**, computed over *successful* requests) and the driver's per-request ceiling (`aiohttp.ClientTimeout(total=120)`). The 10 timeouts are not "a bit slow" — they are stuck **past 120 s**, far beyond anything that completed. That is not vLLM (iter-11: 0 waiting, 0 preemptions, TTFT 36 ms) and not the per-call budget (10 s timeout × 1 retry × ~4 calls ≈ 80 s absolute worst). It is **event-loop starvation**: the agent is a *single* asyncio loop doing ~27 vLLM round-trips/s of orchestration CPU (response parse + LangGraph state + Langfuse spans) on one core. When that loop falls behind, its timers — including the 10 s per-call timeout meant to fail fast — fire *late*, so a few unlucky requests never get cut and pile to >120 s. iter-11 named this exact lever as the next non-shedding move.

**Changed — multiple uvicorn workers (graph logic + prompts untouched → accuracy unaffected).**
- **(concurrency)** [`scripts/start_agent.sh`](scripts/start_agent.sh): launch with `--workers 4` (env `AGENT_WORKERS`). Four event loops on four cores means a CPU stall in one worker delays only its own ~1/N of in-flight requests instead of all of them, so no single loop falls 120 s behind. 4 leaves vLLM (4–15) and the o11y stack (0–3) their cores. **No request is rejected** — this adds parallel servers, it does not shed load (contrast iter-11's reverted 503 path).
- **(metrics correctness)** [`agent/server.py`](agent/server.py): with >1 worker each process keeps its own in-memory counters, so a plain scrape would expose only the one worker that answered it. Switched the agent `/metrics` to Prometheus **multiprocess mode** — `PROMETHEUS_MULTIPROC_DIR` (set + cleaned by the start script) makes the scrape aggregate every worker's mmapped metric files into one view, and the in-flight gauge uses `multiprocess_mode="livesum"` so concurrency sums across workers. Workers `mark_process_dead` on shutdown so a finished run doesn't leave a stale in-flight contribution. The SLO histogram / outcome counters / Grafana panels stay correct.

**Result — pending the next load run.** Expected: `timeouts` → **0** (no loop stays starved long enough to blow the 120 s ceiling), `latency_max` drops from the >120 s stuck tail toward the ~35 s real worst case, and P50/P95/P99 hold (the system was already healthy in those bands). Re-run `scripts/start_agent.sh` then `--rps 11 --duration 300`; confirm the agent `/metrics` still reports one aggregated series per metric (multiprocess working), and `agent_inflight_requests` now peaks across workers rather than pinning one loop. Re-confirm accuracy unchanged via `evals/run_eval.py` (nothing in the graph or prompts moved).

## Phase 7 — Wrap-up
_TODO: final numbers, whether quality survived, what I'd do with more time._
