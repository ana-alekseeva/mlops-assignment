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
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768
