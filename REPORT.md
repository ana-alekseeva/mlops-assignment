# Report — performance tuning (`plan-rerun`)

Condensed rerun of the tuning journey (roadmap in [plan.md](plan.md)). Each iteration reads
*measured → observed → hypothesis for the problem metric → next move*.
**Every `[...]` is a placeholder — fill from the captured run artifacts.**

## Setup

- **SLO:** P95 end-to-end agent latency **< 5 s at 10 RPS** over 5 min, accuracy held.
- **Model / HW:** Qwen3-30B-A3B (MoE, ~3.3B active / 30.5B total) on 1× H100 80GB, 8-core host.
- **Workload:** 1.5–3K-token prompts, short SQL outputs, ~2–3 dependent vLLM calls/request.
- **How measured:**
  - *Latency + errors:* `load_test/driver.py --rps 10 --duration 300` → P50/95/99/max + error classes (timeout / http / client). Note: `achieved_rps ≈ 8.3` is a driver-drain artifact (300 s run + 60 s drain), not a server ceiling.
  - *Accuracy:* `evals/run_eval.py` → execution accuracy over 30 questions, with per-iteration pass rate.
  - *Serving internals:* 2×2 Grafana dashboard + `scripts/sample_throughput.sh` (gen tok/s, **GPU util**, batch depth, KV %, prefix-hit); Langfuse for per-call latency / calls-per-request.

## Results at a glance

| Iter | P50 | P95 | P99 | max | ok / 3000 | errors | eval acc |
|------|-----|-----|-----|-----|-----------|--------|----------|
| 0 baseline | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 1 async + pool | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 2 CPU isolation | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 3 right-size KV | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 4 FP8 + FlashInfer | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| 5 tail close | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

*Target P95 < 5 s. Serving internals for iter 2 below: gen tok/s `[ ]`, GPU util `[ ]`%, batch depth `[ ]`, prefix-hit `[ ]`%.*

## Iterations

### 0 — Baseline
- **Observed:** P95 `[ ]` (≫ 5 s), only `[ ]`/3000 ok; errors dominated by client `[ ]` + timeout `[ ]` + http `[ ]`.
- **Problem → hypothesis:** total overload. Per-call LLM-client construction churns sockets → **client errors**; the sync 40-thread pool serializes 2–3 calls/request → a deep request queue → latency *and* queue-timeout 500s.
- **Next:** fix the agent layer (iter 1).

### 1 — Pooled client + async
- **Observed:** client_errors `[ ]`, timeouts `[ ]` (both expected to collapse); P95 `[ ]` (down from baseline but still ≫ 5 s).
- **Problem → hypothesis:** P95 still `[ ]` though errors are fixed → the wall is **not** connections. vLLM reports clean (0 errors) with low gen tok/s `[ ]` → the GPU is **underfed**, likely CPU starvation of the engine loop.
- **Next:** feed the GPU (iter 2).

### 2 — CPU isolation + larger batched-token budget
- **Observed:** gen tok/s `[ ]` (↑ sharply), GPU util `[ ]`%, batch depth `[ ]` (↓ from ~170), P95 `[ ]`.
- **Problem → hypothesis:** if GPU util pins at `[ ]`% it **was** starvation (now fixed); P95 `[ ]` still > 5 s and the regime is now **GPU-bound on a shallow batch** → per-token cost (bf16 expert-weight HBM read) is the wall, and over-provisioned context/KV wastes batch room.
- **Next:** right-size the working set (iter 3), then cheaper tokens (iter 4).

### 3 — Right-size KV / batching / token trims
- **Observed:** P95 `[ ]` (small move expected), max `[ ]` ↓ (chunked prefill removes stragglers); prefix-hit `[ ]`% ≈ 90; **eval acc `[ ]`** (gate — cell cap + 1-token verdict are model-visible).
- **Problem → hypothesis:** P95 still `[ ]`. Input side is exhausted (prefix cached, prompts bounded), so shrinking inputs is mostly hygiene; at 100 % util only **cheaper tokens** move P95.
- **Next:** FP8 weights + kernels (iter 4).

### 4 — FP8 weights + FlashInfer (+ DeepGEMM)
- **Observed:** P50/P95/P99 fall **together** → P95 `[ ]` (**SLO met?**), client_errors `[ ]`, max `[ ]` ↓; **eval acc `[ ]`** (fp8 is the one real accuracy gate — fp8 is now the graded precision).
- **Problem → hypothesis:** if P99 `[ ]` / max `[ ]` are still over 5 s, the residual is **agent-side**, not vLLM (expect TTFT `[ ]` ms, 0 waiting, 0 preemptions): single event-loop orchestration CPU + retry stacking.
- **Next:** close the tail (iter 5).

### 5 — Close the tail without shedding
- **Observed:** P95 `[ ]`, P99 `[ ]`, max `[ ]`, errors `[ ]` (target 0); **eval acc `[ ]`** (unchanged — graph/prompts untouched).
- **Problem → hypothesis:** any residual P99/max = the single asyncio loop falling behind on per-request CPU (parse + state + tracing); verify result-prefill + double retry amplify the worst case.
- **Improvement (if still over):** multiple uvicorn workers (Prometheus multiprocess) to spread orchestration CPU across cores — **never** shed load (admission-control 503s turn slow requests into errors).

## SLO status & accuracy

- **SLO:** P95 met at iteration `[ ]` (P95 `[ ]` < 5 s); P99 `[ ]`, max `[ ]`.
- **Accuracy:** held at `[ ]` vs baseline `[ ]` across iterations (small-n noise on 30 questions; fp8 is the graded surface). Revise still engaged on `[ ]`/30.
- **Project arc (P95):** `[ ]` → `[ ]` → `[ ]` → `[ ]` → `[ ]` → `[ ]` (baseline → iter 5).
- **Deliberately not retried** (negative results, rationale in [plan.md](plan.md)): input-size shrinking (no-op vs 90 % prefix cache), verify-skip (failed accuracy gate), CPU KV offload (wrong bottleneck — decode throughput, not KV capacity), n-gram speculative decoding (lost at 100 % util — no idle FLOPs).
