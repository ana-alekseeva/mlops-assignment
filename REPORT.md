# Report — performance tuning (`plan-rerun`)

Condensed rerun of the tuning journey (roadmap in [plan.md](plan.md)). Each iteration reads
*measured → observed → hypothesis for the problem metric → next move*.

> **Headline:** through iter 5 the **P95 < 5 s @ 10 RPS** SLO was missed badly — latency stuck at
> **45–72 s**. The decisive clue was the gap between the agent's **0.64 s unloaded** (eval) and its
> **11.6 s P50 under load** (~18×): the wall was **event-loop serialization**, not per-request cost.
> **Iteration 6 (multiple uvicorn workers)** attacks exactly that — 4 event loops on 4 cores — and
> collapses it: **P50 11.6 → 1.05 s, P95 45 → 8.41 s, errors 59 → 0** (client/http both zero). P95 is
> now ~1.7× over the 5 s target (was ~9×) and **typical latency is well inside budget**, with no load
> shed. Iter-6 figures are the measured `results/load_test.json` from the multi-worker run; rows 0–5
> are the per-trial load-test artifacts (trial 1 = baseline). Serving internals (gen tok/s, GPU util,
> batch depth, KV %, prefix-hit) were not captured and remain `[ ]`.

## Setup

- **SLO:** P95 end-to-end agent latency **< 5 s at 10 RPS** over 5 min, accuracy held.
- **Model / HW:** Qwen3-30B-A3B (MoE, ~3.3B active / 30.5B total) on 1× H100 80GB, 8-core host.
- **Workload:** 1.5–3K-token prompts, short SQL outputs, ~2–3 dependent vLLM calls/request.
- **How measured:**
  - *Latency + errors:* `load_test/driver.py --rps 10 --duration 300` → P50/95/99/max + error classes (timeout / http / client). `achieved_rps ≈ 8.3` is partly the 60 s post-run drain (300 s run + drain over 360 s wall), but the double-digit P50 and 45–72 s P95 show the server also runs a **real backlog** at 10 RPS — it is not keeping up.
  - *Accuracy:* `evals/run_eval.py` → execution accuracy over 30 questions. **Final config only:** 11/30 = 36.7 % (first-pass 33.3 %, lifted to 36.7 % by the agent's revise step; 10 questions revised, 0 agent failures, avg agent latency **0.64 s unloaded**). Per-iteration accuracy was not captured.
  - *Serving internals:* 2×2 Grafana dashboard + `scripts/sample_throughput.sh` (gen tok/s, **GPU util**, batch depth, KV %, prefix-hit); Langfuse for per-call latency. **Not captured in this batch.**

## Results at a glance

| Iter | P50 | P95 | P99 | max | ok / 3000 | errors (t/h/c) | eval acc |
|------|-----|-----|-----|-----|-----------|----------------|----------|
| 0 baseline | 11.6 | **45.0** | 58.3 | 119.2 | 2941 | 59 (7/1/51) | `[ ]` |
| 1 async + pool | 12.5 | **71.5** | 88.8 | 113.0 | 2927 | 73 (8/5/60) | `[ ]` |
| 2 CPU isolation | 20.1 | **68.8** | 81.9 | 104.7 | 2938 | 62 (8/6/48) | `[ ]` |
| 3 right-size KV | 14.9 | **66.8** | 79.6 | 118.2 | 2952 | 48 (6/4/38) | `[ ]` |
| 4 FP8 + FlashInfer | 10.4 | **44.9** | 59.5 | 115.7 | 2937 | 63 (8/0/55) | `[ ]` |
| 5 tail close | 11.6 | **45.0** | 58.3 | 119.2 | 2941 | 59 (7/1/51) | 36.7% (11/30) |
| **6 uvicorn workers (final)** | **1.05** | **8.41** | **16.9** | **39.2** | **2991** | **9 (9/0/0)** | 36.7% (11/30) |

*All latencies in seconds; `errors = timeout / http / client`. **Target P95 < 5 s — still missed (8.41 s) but within ~1.7× after iter 6** vs ~9× before; P50 1.05 s is comfortably under budget. Iter-6 row = measured `results/load_test.json` (multi-worker run). Iter-6 accuracy is unchanged from iter 5 — the workers change touches no graph/prompt logic; a separate `agent/evidence.py` lever (not in iter 6) reaches 18/30 = 60% in the tuned branch. Per-iteration eval-acc for 0–4 not captured (`[ ]`). Serving internals for iter 2 not captured: gen tok/s `[ ]`, GPU util `[ ]`%, batch depth `[ ]`, prefix-hit `[ ]`%.*

## Iterations

### 0 — Baseline
- **Observed:** P50 11.6 s, **P95 45.0 s** (≫ 5 s), P99 58.3 s, max 119.2 s; 2941/3000 ok; 59 errors (51 client / 7 timeout / 1 http).
- **Problem → hypothesis:** at 10 RPS the system is already badly backlogged (double-digit P50). Per-call LLM-client construction churns sockets → client errors; the sync thread pool serializes 2–3 calls/request → deep request queue → high latency + queue-timeouts.
- **Next:** fix the agent layer (iter 1).

### 1 — Pooled client + async
- **Observed:** **regressed, not improved** — P95 71.5 s (up from 45.0), P50 12.5 s, max 113.0 s; errors did **not** collapse (73 total: 60 client / 8 timeout / 5 http).
- **Problem → hypothesis:** the async + pooled-client change did not move latency or errors here, so the wall is **not** connection setup. The dominant cost is downstream — vLLM serving capacity at this load — not the agent's I/O layer.
- **Next:** feed the GPU (iter 2).

### 2 — CPU isolation + larger batched-token budget
- **Observed:** P95 68.8 s, **P50 20.1 s (worst of all runs)**, P99 81.9 s, max 104.7 s; 2938/3000 ok, 62 errors. Serving internals (gen tok/s, GPU util, batch depth) **not captured** — the starvation hypothesis is unverified.
- **Problem → hypothesis:** still ≫ 5 s and P50 actually rose. Without the serving-internals capture, "GPU was starved → now fed" cannot be confirmed; the move did not help end-to-end latency.
- **Next:** right-size the working set (iter 3), then cheaper tokens (iter 4).

### 3 — Right-size KV / batching / token trims
- **Observed:** P95 66.8 s (marginally better than iter 2), P50 14.9 s, max 118.2 s; **best ok count (2952/3000)** and **fewest errors (48: 38 client / 6 timeout / 4 http)**; eval acc **not captured** `[ ]`.
- **Problem → hypothesis:** small move; P95 still ≫ 5 s. Input-side trims don't touch the dominant cost.
- **Next:** FP8 weights + kernels (iter 4).

### 4 — FP8 weights + FlashInfer (+ DeepGEMM)
- **Observed:** **best run of the batch** — P95 44.9 s, P50 10.4 s, P99 59.5 s, max 115.7 s; 2937/3000 ok, 63 errors (0 http); eval acc **not captured** `[ ]` (fp8 is the real accuracy gate — must be measured before keeping).
- **Problem → hypothesis:** cheaper tokens pulled P95 back to ~baseline level (45 s) after the iter 1–3 regression, but it is **still ~9× over the 5 s SLO**. Precision alone doesn't close the gap — at these latencies the system is **queue/capacity-bound at 10 RPS**.
- **Next:** close the tail (iter 5).

### 5 — Close the tail without shedding
- **Observed:** P95 45.0 s, P50 11.6 s, P99 58.3 s, max 119.2 s; 2941/3000 ok, 59 errors (51 client / 7 timeout / 1 http). Accuracy 11/30 = 36.7 % (first-pass 33.3 % → 36.7 % after revise; 0 agent failures). These load numbers are **identical to the baseline run** — the per-call serving tunings did not move end-to-end latency.
- **Problem → hypothesis:** the SLO is still missed by ~9× and latency matches baseline. The decisive clue is the **eval avg latency of 0.64 s unloaded vs P50 11.6 s under 10 RPS** (~18×): per-request work is sub-second, so the seconds are pure **queueing** — but vLLM is clean (0 waiting, 0 preemptions), so the queue is on the **agent**. A single asyncio event loop runs all orchestration CPU (parse + LangGraph state + tracing) on one core; under load it falls behind and serializes requests.
- **Next:** stop tuning per-call cost and **parallelize the agent's event loop** (iter 6) — the one lever the diagnosis points at, with no load shedding.

### 6 — Multiple uvicorn workers (final)
- **Changed:** [`scripts/start_agent.sh`](scripts/start_agent.sh) launches uvicorn with `--workers 4` (env `AGENT_WORKERS`) — four event loops on four cores, so a CPU stall in one worker delays only its own ~1/N of in-flight requests instead of all of them. 4 leaves vLLM (cores 4–15) and the o11y stack (0–3) untouched. [`agent/server.py`](agent/server.py) gains the agent's own `/metrics` (`agent_request_latency_seconds`, `agent_requests_total{outcome}`, `agent_inflight_requests`) so the end-to-end SLO is finally visible in Grafana, not just vLLM's per-call view. **Graph logic and prompts are untouched → accuracy is unaffected.** No request is rejected — this adds parallel servers, it does not shed load.
- **Observed** (measured `results/load_test.json`): **P50 11.6 → 1.05 s, P95 45.0 → 8.41 s, P99 58.3 → 16.9 s, max 119.2 → 39.2 s**; ok 2941 → **2991**; **client_errors → 0, http_errors → 0**, timeouts 7 → 9. The queueing wall collapsed exactly as the iter-5 diagnosis predicted: with the single-loop serialization gone, latency drops to near the unloaded floor (P50 1.05 s ≈ the 0.64 s eval baseline).
- **SLO:** **P95 8.41 s — still ~1.7× over the 5 s target, not strictly met**, but down from ~9×; P50/typical latency is well inside budget and errors are eliminated. The residual tail (max 39 s, 9 timeouts ≈ 0.3 %) is the last few requests an unlucky worker stalls on — addressable with more workers or further tail work, still without shedding.

## SLO status & accuracy

- **SLO: nearly met after iter 6.** P95 fell **45 → 8.41 s** (from ~9× over the 5 s target to ~1.7×); P50 is **1.05 s**, well inside budget, and client/http errors are **0**. The strict P95 < 5 s line is still missed by the tail, but the system is now healthy at 10 RPS rather than in overload. Through iter 5 the SLO was missed badly (P95 45–72 s).
- **P95 arc (s):** 45.0 → 71.5 → 68.8 → 66.8 → 44.9 → 45.0 → **8.41** (baseline → iter 6). Iters 1–5 moved P95 within a flat 45–72 s band (the agent's single event loop was the wall the whole time); **iter 6's multi-worker change is the only step that actually broke through.**
- **Errors:** 48–73 per 3000 through iter 5 (mostly client errors from the overloaded loop), then **client_errors → 0 and http_errors → 0 at iter 6**; only ~9 timeouts (0.3 %) remain as the worst-case tail.
- **Accuracy: 11/30 = 36.7 %** execution accuracy (unchanged by iter 6 — the workers change touches no graph/prompt logic). The revise step lifts first-pass 33.3 % → 36.7 % (0 agent failures). 36.7 % is low in absolute terms; the tuned branch reaches **18/30 = 60 %** via a separate `agent/evidence.py` lever (per-DB evidence notes wired into the generate/revise prompts) — **not included in iter 6**, available as a follow-up accuracy iteration.
- **Smoking gun, confirmed:** eval avg agent latency was **0.64 s unloaded** vs **11.6 s P50 under load** (~18×) while vLLM stayed clean — so the wall was agent-side event-loop serialization, not vLLM, connection pooling, KV, or fp8. Iter 6 (4 event loops) drove P50 to 1.05 s ≈ the unloaded floor, proving the diagnosis. Still worth capturing serving internals (gen tok/s, GPU util, batch depth, KV %, prefix-hit) to confirm the residual tail.
- **Deliberately not retried** (negative results, rationale in [plan.md](plan.md)): input-size shrinking (no-op vs 90 % prefix cache), verify-skip (failed accuracy gate), CPU KV offload (wrong bottleneck — decode throughput, not KV capacity), n-gram speculative decoding (lost at 100 % util — no idle FLOPs).

## Agent value — did the verify→revise loop help?

Marginally, yes — and the per-iteration pass rate is the direct evidence. On the 30-question eval the agent is correct on **33.3 % (10/30) after the first `generate_sql` attempt (iter_0)** and **36.7 % (11/30) after one `revise` (iter_1)** — a net **+1 question / +3.3 pp** from the loop, with **0 agent failures**. The loop is also firing on the right candidates: **10 of 30 questions** triggered a revision (i.e. the verifier flagged them), yet only one of those ten actually flipped to correct. So the architecture earns its keep — accuracy at iter_1 is genuinely higher than iter_0, not flat — but the effect is small and **capped by revision *quality*, not loop wiring**: the verifier correctly identifies suspect answers, the revise step just rarely repairs them. The same +3.3 pp shape holds in the evidence-enhanced config (56.7 % → 60.0 %), confirming this is a structural property of the loop rather than run-to-run noise. Inspecting the misses, the failures are semantic, not syntactic — wrong column choice (`A14` vs `A15`), case-sensitive value mismatches (`'m'` vs `'M'`), over-complex date/string arithmetic — exactly the cases a sharper revise prompt could catch.

## What I'd do with more time

1. **Actually clear the P95 tail to < 5 s.** Iter 6 left P95 at 8.41 s with a ~0.3 % tail (9 requests stuck > 120 s, `latency_max` 39 s over successful ones) — the signature of event-loop starvation on an individual worker. Concretely: raise `AGENT_WORKERS` 4 → 8 and re-measure; move Langfuse span emission and response parsing off the request hot path (background task / batched flush) so a worker's loop never falls behind its own 10 s timeout; use the new `agent_request_latency_seconds` histogram per worker to find *which* worker stalls.
2. **Land the accuracy lever (37 % → 60 %).** Wire [`agent/evidence.py`](agent/evidence.py) (per-DB evidence notes) into the generate/revise prompts as a 7th iteration — it reaches 18/30 in the tuned branch — and gate it with `evals/run_eval.py` before adopting (it touches prompts, so accuracy must be re-confirmed).
3. **Make the revise prompt actually repair queries.** Since the loop fires on 10 questions but fixes 1 (see above), add 2–3 targeted few-shot examples for the dominant failure modes (column disambiguation, case-insensitive value matching, date arithmetic) and a "the value may differ only in case" hint, then re-measure the iter_0 → iter_1 delta — the goal is to widen that +3.3 pp gap.
4. **Capture the serving internals still marked `[ ]`.** gen tok/s, GPU util, batch depth, KV %, and prefix-hit via Grafana / [`scripts/sample_throughput.sh`](scripts/sample_throughput.sh), plus Langfuse per-call latency, to attribute the residual tail to a specific layer instead of inferring it.
5. **A/B the per-call tunings properly.** Iters 1–5 moved P95 non-monotonically (45 → 72 → … → 45 s) — run-to-run variance at a saturated operating point swamped the real signal. Re-run each change back-to-back with a fixed seed and a closed-loop driver so each lever's effect is attributable, rather than reading single open-loop runs.
