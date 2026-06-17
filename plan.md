# Plan — efficient performance-tuning rerun

A condensed, precise version of the Phase 6 tuning journey. The original run took
**11 iterations** and reached P95 2.9 s, but spent ~4 of them on levers that
turned out to be no-ops or regressions, and only discovered the single biggest
win — CPU starvation of the vLLM engine loop — at iteration 8. This plan reruns
the same problem in **5 iterations** (+ a baseline), in the order the *evidence*
justifies rather than the order it was originally found.

## Goal & fixed constraints

- **SLO:** P95 end-to-end agent latency **< 5 s at 10 RPS** over a 5-minute window, accuracy held.
- **Model:** `Qwen/Qwen3-30B-A3B-Instruct-2507` — MoE, ~30.5B total / ~3.3B active params/token.
- **Hardware:** 1× H100 80GB, 8 physical cores (16 threads) on the host.
- **Workload:** 1.5–3K-token prompts, short SQL outputs, ~2–3 dependent vLLM calls per request.

## What makes this rerun "smarter" than the 11-iteration original

1. **Merge the two agent-orchestration fixes** (pooled client + async) into one iteration — same diagnosis layer ("the agent, not inference, is the bottleneck").
2. **Diagnose the host (CPU starvation) early** — original iter 8, the biggest single win, moved to iteration 2. It reframes the whole GPU story, so doing it first stops us from misreading the regime.
3. **Drop the four dead ends** (see *Deliberately skipped* below): prompt/KV-size shrinking (no-op), verify-skip (accuracy regression), CPU KV offload (wrong bottleneck), n-gram spec decoding (compute-bound → backfired).
4. **Isolate the one accuracy-risky change** (FP8 weights) into its own iteration with a hard eval gate.
5. **No load shedding.** The route to zero errors is cheaper work per request, never 503s.

## Observability used to diagnose each iteration

Consistent with the current setup — **no agent-side `/metrics`**; diagnosis comes from three sources:

- **Load driver report** (`load_test/driver.py`): client-side P50/P95/P99 latency + error breakdown (`ok` / `timeouts` / `http_errors` / `client_errors`) and achieved RPS.
- **Langfuse traces**: per-request end-to-end latency, per-call breakdown, calls-per-request — tagged by `run`/`rps`/`phase` (grouped into one session per run).
- **vLLM Grafana dashboard** (2×2): latency by phase (queue/prefill/decode), token throughput, running/waiting + batch fullness, KV + prefix-cache. Plus `scripts/sample_throughput.sh` (aggregate gen tok/s, batch depth, **GPU util**) from iteration 2 on — GPU util is the gauge that distinguishes "starved" from "GPU-bound".

"vLLM clean (low TTFT, 0 waiting, 0 preemptions) while the driver reports a huge end-to-end P95" is the by-elimination signal that the bottleneck is *above* vLLM.

## Commit plan (branch from project origin)

| Commit | Contents |
|--------|----------|
| **1 — baseline** | Full system scaffold + observability; measure the baseline. |
| **2 — iter 1** | Agent orchestration: pooled LLM client + async path. |
| **3 — iter 2** | Host: CPU isolation + feed the GPU (`--max-num-batched-tokens`). |
| **4 — iter 3** | Right-size the working set: KV/batching config + token trims. |
| **5 — iter 4** | Cheaper tokens: FP8 weights + FlashInfer (+ DeepGEMM). |
| **6 — iter 5** | Close the tail without shedding: verify-prefill cap, trace sampling, retry cap. |

Each iteration commit changes **one diagnosis layer**, is validated by a load run **and** `evals/run_eval.py`, and records *saw → hypothesized → changed → measured*.

---

## Commit 1 — Baseline

**Implement and measure, change nothing yet.**

- **Agent** (`agent/graph.py`): LangGraph text-to-SQL with verify/revise loop — `attach_schema → generate_sql → execute → verify --ok--> END`, else `revise → execute → verify` (capped at `MAX_ITERATIONS=3`). `verify` parses a JSON `{"ok", "issue"}` verdict; a failed execution can never be `ok`.
- **Server** (`agent/server.py`): FastAPI `POST /answer`, sync, with Langfuse tracing — request `tags` mapped to `langfuse_tags` (chips) + `langfuse_session_id` (per-run session).
- **Eval** (`evals/run_eval.py` + `eval_set.jsonl`): 30 curated questions scored by **execution accuracy** (agent SQL vs gold SQL, result sets compared after canonicalization), with per-iteration pass-rate breakdown.
- **Load driver** (`load_test/driver.py`): open-loop at target RPS, reports latency percentiles + error classes, tags every request for Langfuse.
- **Serving** (`scripts/start_vllm.sh`): provided config — bf16, `--max-model-len 32768`, default `--max-num-seqs`, no special flags.
- **Observability**: Prometheus scraping vLLM `:8000`, the 2×2 Grafana dashboard, docker-compose o11y stack (Langfuse + ClickHouse + Postgres + Redis + MinIO + Grafana + Prometheus).

**Measure:** `--rps 10 --duration 300`. Expect deep overload (the prior run: ~53% ok, **P95 ~115 s**) — establishes the gap.

---

## Iteration 1 — Fix the agent orchestration layer (pooled client + async)

- **Problem (saw):** at 10 RPS ~47% of requests fail and even successful ones sit at P95 ~115 s. Errors split into socket-churn `client_errors` and unbounded-call `timeouts`; latency is dominated by overload, not any single slow inference. vLLM reports every completion `finished_reason="stop"` — it's idle-clean while the driver drowns.
- **Hypothesis (culprit):** two faults in the *agent*, not inference. (a) `graph.py:llm()` builds a fresh `ChatOpenAI` (new httpx pool) on **every** node call → ephemeral-port exhaustion → connection-reset client errors + connect latency; unbounded per-call timeout lets a queued call hang to the driver's 120 s cap → timeouts. (b) Sync `def answer` + `graph.invoke` runs on Starlette's **40-thread** pool; each request holds one thread for its whole 2–3-call chain, so concurrency caps at 40. By Little's law (λ≈10, W≈80 s) ~790 requests are resident but only 40 run → **the queue is the latency**.
- **Solution (change):**
  - Pool the client: `@lru_cache` singleton `ChatOpenAI` (one reused httpx pool), bounded `timeout` + `max_retries`.
  - Go async: `async def answer` + `await graph.ainvoke(...)`; LLM nodes `async` with `await llm().ainvoke(...)`; `execute` offloads blocking sqlite via `asyncio.to_thread`. Pure nodes stay sync. No graph/prompt changes → accuracy unaffected.
- **Expected:** `client_errors` and `timeouts` collapse; the 40-wide queue disappears. **P95 ~115 → ~40 s.** SLO still missed, but the bottleneck has moved off the agent's concurrency model and onto vLLM/host — set up by iteration 2.
- **Gate:** accuracy unchanged (no logic touched); confirm via `run_eval.py`.

---

## Iteration 2 — Feed the GPU: diagnose & fix CPU starvation of the engine loop

- **Problem (saw):** P95 ~40 s with the thread queue gone; latency is a **flat steady state** (not an exploding queue), ~170 requests in-flight by Little's law. Aggregate generation throughput is implausibly low (~tens of tok/s) for a 3B-active MoE on an H100, yet vLLM is error-clean and GPU **memory** is full while GPU **util sits near 0% at idle**.
- **Hypothesis (culprit):** the bottleneck is **not** the GPU — it's **CPU starvation of vLLM's engine-core busy loop**. On 8 physical cores the agent (`uvicorn`), the load driver, and the full observability stack (ClickHouse/Langfuse above all) crowd out the loop that schedules and dispatches batches, so the GPU is **underfed**. (vLLM docs: "the engine core runs a busy loop and is particularly sensitive to CPU starvation.") Separately, `--max-num-batched-tokens` is on its low default, capping tokens/step.
- **Solution (change):** isolation, not shutdown.
  - `docker-compose.override.yml`: box the whole o11y stack into cores **0–3** (`cpuset` + `cpus` caps so ClickHouse/Langfuse self-size to the quota).
  - `scripts/start_vllm.sh`: pin vLLM to the **complementary cores 4–15** (`taskset`); set `--max-num-batched-tokens 8192` (the per-step token budget that was capping the effective batch).
  - Add `scripts/sample_throughput.sh` (aggregate gen tok/s, batch depth, KV %, **GPU util**) — the decisive diagnostic.
- **Decisive test:** GPU util under load. **< ~90% ⇒ was starved** (the isolation is the win); **~100% ⇒ genuine GPU ceiling** (move to per-token cost).
- **Expected:** token throughput ~tens → **~1,000 tok/s**, GPU util → **100%**, batch depth ~170 → **~10**. The regime flips from *throughput-bound on a deep starved batch* to *GPU-bound on a shallow fast one*. **P95 ~40 → ~10 s.** Observability stays fully on.
- **Gate:** pure placement/serving change, agent logic untouched → accuracy unchanged; confirm.

---

## Iteration 3 — Right-size the working set (KV, batching, token trims)

- **Problem (saw):** GPU now fed and 100%-util on a shallow batch, P95 ~10 s. Context is over-provisioned at `--max-model-len 32768` while the **largest real prompt is 7,308 tokens**; result cells in verify/revise prompts are unbounded (a wide `SELECT *` blob → ~3K tokens); `verify` decodes a ~12-token JSON verdict on **every** request, the vast majority of which are `OK`; big prefills can stall the running batch's decode.
- **Hypothesis (culprit):** the GPU's working set is bigger than the workload needs. Shrinking the per-sequence KV reservation and trimming wasted prefill/decode tokens gives the batch headroom and cuts per-step cost — all at **zero/low accuracy risk** (prefill is ~90% prefix-cached, so input-size shrinking is mostly hygiene; the real trims are result-cell width and verdict length).
- **Solution (change), one accuracy-gated bundle:**
  - Serving: `--max-model-len 8192` (covers the 7,308-token worst case, ~4× smaller reservation), `--max-num-seqs 256` explicit, `--kv-cache-dtype fp8` (halves the per-step KV read, ~2× capacity), `--enable-chunked-prefill` (big prefill can't stall decode), `--enable-prefix-caching` (explicit/self-documenting; confirm ~90% hit).
  - Agent: cap result-cell width in `render()` (`…(+N chars)` marker); compact **1-token verify verdict** (`OK` / `BAD: <issue>`) instead of JSON; bounded `max_tokens`; `MAX_ITERATIONS 3 → 2` (the 2nd revise never lifted the prior pass rate).
- **Expected:** modest P95 move + **tighter tail/max** (chunked prefill removes prefill-stall stragglers; fp8-KV speeds decode). Honest note: prompt-size shrinking will be near-no-op (prefix cache already ~90%) — keep only the trims that bite (cell cap, verdict length, KV dtype, chunked prefill).
- **Gate:** **must** re-run `run_eval.py` — the cell cap and 1-token verdict are model-visible; confirm accuracy holds (revise must still fire and recover questions).

---

## Iteration 4 — Cheaper tokens per step: FP8 weights + FlashInfer (+ DeepGEMM)

- **Problem (saw):** GPU compute-bound at 100% util on a shallow (~8–10) batch, P95 ~10 s. Batch-size knobs are exhausted — at 100% util, deepening the batch trades latency for nothing. The dominant per-step cost is the **MoE expert-weight read from HBM** (every decode step re-reads the active experts) plus the attention kernel; the H100 is serving **bf16** (~61 GB) though an official FP8 checkpoint exists, and default attention/MoE kernels are in use.
- **Hypothesis (culprit):** at 100% util the only lever left is **fewer HBM bytes + faster tensor-core math per token**. FP8 weights halve the expert-weight read **and** run on Hopper fp8 tensor cores (~2× bf16); FlashInfer is a faster Hopper attention / fp8-KV kernel; DeepGEMM is a faster grouped fp8 MoE GEMM. This should lower the cost of *every* decode step → the whole latency distribution falls, not just the tail.
- **Solution (change), one switch at a time:**
  1. **FP8 weights** — `VLLM_MODEL → Qwen/Qwen3-30B-A3B-Instruct-2507-FP8`. *The one change with real accuracy risk* — eval-gate first.
  2. **FlashInfer** — `VLLM_ATTENTION_BACKEND=FLASHINFER` (no accuracy change).
  3. **DeepGEMM** — `VLLM_USE_DEEP_GEMM=1` iff the `deep_gemm` kernels are importable (else safe Triton fallback).
- **Expected:** **P50/P95/P99 fall together** (per-token cost signature). **P95 ~10 → ~2.9 s — SLO met** with a comfortable margin; `client_errors → 0`, `max` drops sharply. Watch `sample_throughput.sh`: tok/s rises at same-or-lower util = the cheaper-token win.
- **Gate:** FP8 is the chosen serving precision now, so **the graded accuracy figure is the FP8 one** — log `run_eval.py` on the FP8 H100 server; revert any switch that drops accuracy below baseline beyond n=30 noise or regresses tok/s.

---

## Iteration 5 — Close the tail without shedding (agent CPU + P99/max)

- **Problem (saw):** P95 under SLO (~2.9 s) but **P99 ~5.7 s** just over and **max ~22 s**. The Prometheus split shows vLLM is pristine (TTFT ~36 ms, 0 waiting, 0 preemptions, batch ~8) — the gap is **agent-side**: a single event loop doing ~27 vLLM round-trips/s, each with response-parse + LangGraph state + Langfuse span serialization; `verify` re-prefills the uncached result every request; a timed-out call re-prefills under `max_retries=2`, stacking the worst case.
- **Hypothesis (culprit):** the residual tail is agent orchestration CPU + verify result-prefill + retry amplification, **not** vLLM. Cutting per-request work closes it; **shedding would convert slow requests into errors** (a tried-and-reverted mistake — admission-control 503s turned ~20% of requests into `http_errors`).
- **Solution (change):**
  - Shrink verify-path prefill: `render(max_rows=3, max_cell=80)` on the **verify** path only (revise keeps the wider view — it must see the data to fix the query). The verifier needs the answer's *shape*, not the blob.
  - Langfuse `LANGFUSE_SAMPLE_RATE=0.1` under load (drops tracing CPU on the event loop + ClickHouse ingestion; **no request dropped**; Prometheus metrics unaffected; set back to 1.0 for trace inspection).
  - `max_retries 2 → 1` to bound re-prefill stacking on the worst case.
  - **Explicitly no admission control / 503 path.**
- **Expected:** P95 settles at the vLLM-under-load floor with **0 errors**; P99 and max tighten toward SLO. If P95 still drifts under load, the next *non-shedding* lever is multiple uvicorn workers (Prometheus multiprocess) to spread orchestration CPU across cores.
- **Gate:** graph logic/prompts untouched → accuracy unaffected; confirm `run_eval.py`.

---

## Deliberately skipped (negative results from the prior run — not re-tried)

Recorded so the reasoning is preserved without spending iterations on them:

1. **Prompt / schema / KV-size shrinking as a latency lever** — prefix cache is already ~90% and prompts are small; shrinking inputs was a measured **no-op**. (We keep only the result-cell cap, as a correctness bound on pathological blobs.)
2. **Verify-skip on the happy path** — gave the biggest single latency cut (P95 →20 s) but **dropped a correctness check** and failed the accuracy gate. Keep the verifier; make it *cheap* instead (iter 3 verdict + iter 5 prefill cap).
3. **CPU KV-cache offload (`--swap-space`)** — fixes KV *capacity*; our wall is decode *throughput*. KV must stream back over PCIe before it can decode → neutral-to-negative. Diagnostic (`kv_cache_usage` well under 100%, 0 preemptions) falsifies it for free.
4. **n-gram speculative decoding** — wins only when memory-bandwidth-bound **with idle FLOPs**; after iter 2 the GPU is compute-bound at 100% util, so draft/verify work *competes* with the batch → P95 regressed (10 → 20 s). Shelved unless a future lever creates genuine idle-FLOP headroom.

## Expected P95 trajectory

| Stage | P95 (target) | Bottleneck addressed |
|-------|-------------|----------------------|
| Baseline | ~115 s | deep overload |
| Iter 1 — agent async + pooled client | ~40 s | agent concurrency ceiling |
| Iter 2 — CPU isolation + feed GPU | ~10 s | host CPU starvation (the big one) |
| Iter 3 — right-size KV/batching/tokens | ~10 s | working-set hygiene, tail, safety |
| Iter 4 — FP8 weights + FlashInfer | **~2.9 s** | per-token HBM + tensor-core cost (**SLO met**) |
| Iter 5 — tail close, no shedding | ~2.9 s P95, P99/max tightened | agent-side CPU + retry tail |

~40× P95 reduction in 5 attributable iterations, accuracy held, observability fully on, zero load shedding.
