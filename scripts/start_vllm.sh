#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

# Load variables from .env (HF_TOKEN, etc.) so they reach the vLLM process.
# Resolve the path relative to this script so it works from any directory.
ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

# Model id comes from .env (VLLM_MODEL) so the server and the agent always use the
# same model. Switch hardware by switching VLLM_MODEL in .env:
#   - L40S 48GB (dev):   Qwen/Qwen3-30B-A3B-Instruct-2507-FP8  (fp8,  ~30GB)
#   - H100 80GB (final): Qwen/Qwen3-30B-A3B-Instruct-2507      (bf16, ~61GB, spec metrics)
# vLLM auto-detects fp8 from the checkpoint config - no quantization flag needed.
MODEL="${VLLM_MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"

# --max-num-batched-tokens 8192 (iteration 2): the per-step token budget the
# scheduler fills from the running batch. Left unset it defaults low (~2048 with
# chunked prefill), which caps decode+prefill tokens per engine step -> a shallow
# effective batch -> low token throughput. vLLM tuning docs recommend ">8192 for
# throughput, especially smaller models on large GPUs" - a 3B-active MoE on an
# H100 is exactly that.
#
# CPU pinning (iteration 2): the vLLM engine core is a busy loop that starves
# when the agent, the load driver, and the observability stack share its cores.
# Pin vLLM to dedicated cores so the loop always gets scheduled. Keep VLLM_CPUS
# the COMPLEMENT of the o11y cpuset in docker-compose.override.yml (o11y on 0-3,
# vLLM on 4-15). taskset is optional - skipped cleanly if not installed.
VLLM_CPUS="${VLLM_CPUS:-4-15}"
PIN=()
if command -v taskset >/dev/null 2>&1; then
    PIN=(taskset -c "$VLLM_CPUS")
    echo "Pinning vLLM to CPUs $VLLM_CPUS"
fi

# Right-size the working set for the now-fed GPU (iteration 3):
# --max-model-len 8192: the largest real prompt is 7,308 tokens (biggest schema
#   + widest result preview in a revise call); 8192 covers it with headroom and
#   is 4x smaller than 32768, so the per-sequence KV reservation shrinks and far
#   more sequences fit in the cache. (Not 4096 - that truncates the 7.3k case.)
# --max-num-seqs 256: the running-batch ceiling, set explicitly (>= observed
#   concurrency); the knob to sweep if decode throughput is still the wall.
# --enable-prefix-caching: the constant prefix (system rules + DB schema) is
#   identical across questions for a DB and across the 2-3 calls per request, so
#   its prefill KV is computed once and reused (confirm ~90% hit on /metrics).
# --enable-chunked-prefill: split the ~7.3k-token prefills into chunks
#   interleaved with decode so a big prompt can't stall the running batch.
# --kv-cache-dtype fp8: every decode step re-reads the KV cache from HBM, so
#   decode is KV-bandwidth-bound; fp8 KV halves that read and ~2x's KV capacity.
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
