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
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --enable-prefix-caching
