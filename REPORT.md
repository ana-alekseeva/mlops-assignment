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

## Phase 5 — Evaluation
_TODO: baseline execution-accuracy on the eval set._

## Phase 6 — Performance tuning
_TODO: iteration log — "saw X → hypothesized Y → changed Z → result was W" + Grafana screenshots._

## Phase 7 — Wrap-up
_TODO: final numbers, whether quality survived, what I'd do with more time._
