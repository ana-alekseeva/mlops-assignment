#!/usr/bin/env bash
#
# Sample vLLM + GPU during a load run to tell apart the two failure modes:
#   - GPU util LOW  + low tok/s -> vLLM is STARVED/underfed (CPU/engine-loop or
#                                  client concurrency) -> fix the feeding path.
#   - GPU util HIGH + low tok/s -> genuine GPU/kernel ceiling -> serving config
#                                  (max-num-batched-tokens, kv-cache-dtype, model).
#
# Run it WHILE load_test/driver.py is firing:
#   bash scripts/sample_throughput.sh            # samples every 5s until Ctrl-C
#   INTERVAL=2 DURATION=300 bash scripts/sample_throughput.sh
#
# Each line: aggregate generation tok/s and prompt tok/s over the interval,
# running/waiting batch depth, per-request decode rate (gen tok/s / running),
# GPU-cache (KV) usage %, and GPU SM utilization %.
set -uo pipefail

METRICS_URL="${METRICS_URL:-http://localhost:8000/metrics}"
INTERVAL="${INTERVAL:-5}"
DURATION="${DURATION:-0}"   # 0 = run until Ctrl-C

# Pull a single bare vllm:<name> gauge/counter value (first non-comment line).
metric() { curl -s -m 5 "$METRICS_URL" | awk -v k="vllm:$1{" 'index($0,k)==1 {print $2; exit}'; }
gpu()    { nvidia-smi --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' '; }

printf "%-8s %10s %10s %7s %7s %12s %7s %7s\n" \
       "t(s)" "gen_tok/s" "prm_tok/s" "run" "wait" "perreq_tok/s" "kv%" "gpu%"

g_prev=$(metric generation_tokens_total); p_prev=$(metric prompt_tokens_total)
t=0
while :; do
    sleep "$INTERVAL"; t=$((t + INTERVAL))
    g_now=$(metric generation_tokens_total); p_now=$(metric prompt_tokens_total)
    run=$(metric num_requests_running); wait=$(metric num_requests_waiting)
    kv=$(metric gpu_cache_usage_perc); util=$(gpu utilization.gpu)

    # Deltas / rates (guard against empty reads when the engine is idle).
    g_rate=$(awk -v a="${g_prev:-0}" -v b="${g_now:-0}" -v i="$INTERVAL" 'BEGIN{printf "%.0f",(b-a)/i}')
    p_rate=$(awk -v a="${p_prev:-0}" -v b="${p_now:-0}" -v i="$INTERVAL" 'BEGIN{printf "%.0f",(b-a)/i}')
    perreq=$(awk -v g="$g_rate" -v r="${run:-0}" 'BEGIN{printf "%.1f", (r>0)?g/r:0}')
    kvp=$(awk -v k="${kv:-0}" 'BEGIN{printf "%.0f", k*100}')

    printf "%-8s %10s %10s %7s %7s %12s %7s %7s\n" \
           "$t" "$g_rate" "$p_rate" "${run:-?}" "${wait:-?}" "$perreq" "$kvp" "${util:-?}"

    g_prev=$g_now; p_prev=$p_now
    [ "$DURATION" -ne 0 ] && [ "$t" -ge "$DURATION" ] && break
done
