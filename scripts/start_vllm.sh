#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

# Load variables from .env (HF_TOKEN, etc.) so they reach the vLLM process.
# Resolve the path relative to this script so it works from any directory.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="$REPO_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# Model id comes from .env (VLLM_MODEL) so the server and the agent always use the
# same model. Switch precision by switching VLLM_MODEL in .env:
#   - FP8  (chosen, Phase 6 / iter 10): Qwen/Qwen3-30B-A3B-Instruct-2507-FP8  (~30GB)
#   - bf16 (A/B baseline / fallback):   Qwen/Qwen3-30B-A3B-Instruct-2507      (~61GB)
# Iteration 10 makes FP8 the serving precision on the H100 too (not just the L40S
# dev box). Post-iter-8 the engine is compute-bound at 100% util, so the decode
# bottleneck is the per-step MoE expert-weight read from HBM: fp8 halves that read
# and runs on Hopper's fp8 tensor cores (~2x bf16). Gated on evals/run_eval.py. The
# old bf16 "final" rationale was spec-decode metrics, moot since iter-9 reverted spec.
# vLLM auto-detects fp8 from the checkpoint config - no quantization flag needed.
# Default aligned to FP8 so an entry point that misses .env can't silently 404 (iter-5).
MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507-FP8}"

# --max-model-len caps context (native max 262144) to keep KV-cache memory in
# check. Phase 1 grades these flags - tune them for this workload.
#
# --enable-prefix-caching (Phase 6 / iteration 3): the constant prefix of every
# prompt (system rules + the DB schema) is identical across all questions for a
# given DB AND across the 2-3 generate/verify/revise calls within one request.
# Prefix caching computes that prefill KV once and reuses it on every hit. The
# V1 engine in vLLM 0.23 has this on by default; we pass it explicitly so the
# config is self-documenting. Confirm the hit rate via the
# vllm:prefix_cache_hits / vllm:prefix_cache_queries metrics on :8000/metrics.
# --max-model-len 8192 (Phase 6 / iteration 4): the workload's largest prompt is
# a revise call with the biggest schema + a wide result preview, measured at
# 7,308 tokens with the Qwen tokenizer; outputs are short SQL (<256 tok). 8192
# covers the worst case with headroom while being 4x smaller than the previous
# 32768 - shrinking the per-sequence KV reservation so far more sequences fit in
# the cache at once. That directly serves the iter-3 finding: the system is
# decode-bound at ~170 concurrent requests, so KV headroom for deeper batches is
# the lever, not context length. (NOT 4096 - that would truncate the 7.3k case.)
# --max-num-seqs 256: the running-batch ceiling. ~170 requests are in flight at
# 10 RPS, so 256 keeps the batch from being capped below observed concurrency;
# set explicitly (it's also the default) and the knob to sweep next if decode
# throughput is still the wall.
# --kv-cache-dtype fp8 (Phase 6 / iteration 5): decode is the bottleneck, and at
# steady state every decode step re-reads the whole KV cache from HBM, so decode
# throughput is KV-bandwidth-bound. Storing KV in fp8 halves that read and ~2x's
# KV capacity (deeper batches without preemption) - the highest-value decode
# lever once per-request call count is fixed. Small numeric impact; validated
# against the Phase 5 eval before adoption.
# --enable-chunked-prefill: split large prefills (the worst prompt is ~7.3k tok)
# into chunks interleaved with decode, so a big prefill can't stall token
# generation for the running batch - protects decode tail latency under the
# mixed prefill/decode load of 10 RPS. (Default-on in the V1 engine; explicit
# here for the graded serving config.)
# Two opt-in experiments were removed here after they were shown to add no value
# (history kept in REPORT.md):
#   - CPU KV offload (--swap-space, iter 7): our wall is decode *throughput*, not
#     KV *capacity* (gpu_cache_usage well under 100%, 0 preemptions), and offloaded
#     KV must stream back over PCIe before it can decode - neutral-to-negative.
#   - n-gram speculative decoding (iter 9): measured P95 10.2 -> 20.6s. The GPU is
#     compute-bound at 100% util, so there are no spare FLOPs for the draft/verify
#     passes; they just compete with the running batch. Kill criterion met.

# --- Tier 1 (Phase 6 / iteration 10): cheaper kernels per generation token --
# Post-iter-8 the engine is GPU-compute-bound at 100% util on a shallow (~10)
# batch, so latency == throughput == generation tok/s/GPU and the batch-size
# knobs are exhausted (iter-9 corollary: deepening the batch trades latency for
# nothing). The only lever left is making each token cheaper. fp8 weights is the
# VLLM_MODEL change above; the two engine-level kernel switches are here. All
# three are accuracy-gated via evals/run_eval.py before adoption.
#
# (FlashInfer attention) faster Hopper decode + fp8-KV attention kernels than the
# default backend; flashinfer-python is installed. Opt out: VLLM_ATTENTION_BACKEND=<x>.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
#
# (DeepGEMM MoE) grouped fp8 GEMM for the experts - materially faster than the
# default Triton fused-MoE on Hopper for this 3B-active MoE, and only meaningful
# once weights are fp8. Auto-enabled IFF the kernels are importable in the venv
# (they are not installed by default - add `deep_gemm` to light this up); until
# then vLLM falls back to Triton fused-MoE, so leaving it unset is safe and
# correct. Force on/off explicitly with VLLM_USE_DEEP_GEMM=1/0.
if [[ -z "${VLLM_USE_DEEP_GEMM:-}" ]] && \
   "$REPO_DIR/.venv/bin/python" -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('deep_gemm') else 1)" 2>/dev/null; then
    export VLLM_USE_DEEP_GEMM=1
fi
echo "Tier-1 kernels: attention=$VLLM_ATTENTION_BACKEND deep_gemm=${VLLM_USE_DEEP_GEMM:-0} model=$MODEL"

# --max-num-batched-tokens 8192 (Phase 6 / iteration 8): the per-step token
# budget the scheduler fills from the running batch. Left unset it defaults low
# (~2048 with chunked prefill), which caps how many decode+prefill tokens run per
# engine step -> shallow effective batch -> low token throughput, the iter-8
# symptom. The vLLM tuning docs recommend ">8192 for throughput, especially for
# smaller models on large GPUs" - a 3B-active MoE on an H100 is exactly that. We
# match it to max-model-len (8192) so a single max-length prompt still fits one
# step. Raise further (16384) if GPU util stays high but throughput is still low.
# CPU pinning (Phase 6 / iteration 8): the vLLM engine core is a busy loop that
# starves when the agent, the load driver, and the observability stack share its
# cores. Pin vLLM (and its inherited worker threads) to dedicated cores so the
# loop always gets scheduled. Keep VLLM_CPUS the COMPLEMENT of the o11y cpuset in
# docker-compose.override.yml (default: o11y on 0-3, vLLM on 4-15). taskset is
# optional - skipped cleanly if not installed.
VLLM_CPUS="${VLLM_CPUS:-4-15}"
PIN=()
if command -v taskset >/dev/null 2>&1; then
    PIN=(taskset -c "$VLLM_CPUS")
    echo "Pinning vLLM to CPUs $VLLM_CPUS"
fi

exec "${PIN[@]}" uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 8192 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --kv-cache-dtype fp8
