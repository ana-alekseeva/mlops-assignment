#!/usr/bin/env bash
#
# Start the agent server (FastAPI on :8001).
#
# Phase 6 / iteration 6 (final): run uvicorn with MULTIPLE WORKERS. The agent is
# a single asyncio event loop; at ~10 RPS it does ~27 vLLM round-trips/s of
# orchestration CPU (response parse + LangGraph state + Langfuse spans) on one
# core. When that loop falls behind, even the bounded 10s per-call timeout fires
# late, so a few requests pile past the load driver's 120s ceiling and surface
# as load-test "timeouts" (~0.3% of requests) - while vLLM itself stays pristine
# (0 waiting, 0 preemptions). Multiple workers give multiple event loops on
# multiple cores, so a CPU stall in one only delays its own ~1/N of requests
# instead of every in-flight request. This is the SLO win WITHOUT shedding any
# request (graph logic + prompts untouched, so accuracy is unaffected).

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

ENV_FILE="$REPO_DIR/.env"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
fi

HOST="${AGENT_HOST:-0.0.0.0}"
PORT="${AGENT_PORT:-8001}"
# 4 event loops is enough headroom against starvation at the tested load while
# still leaving cores for vLLM (4-15) and the o11y stack (0-3). Override via env.
WORKERS="${AGENT_WORKERS:-4}"

# Prometheus multiprocess mode: with >1 worker each process keeps its own
# in-memory counters, so the agent /metrics scrape must aggregate every worker's
# mmapped metric files - otherwise Grafana sees only the one worker that happened
# to serve the scrape. server.py switches to a MultiProcessCollector when this
# dir is set. Clean it on each start so dead-worker files from a prior run don't
# leak into the aggregate.
export PROMETHEUS_MULTIPROC_DIR="${PROMETHEUS_MULTIPROC_DIR:-$REPO_DIR/.prom_multiproc}"
rm -rf "$PROMETHEUS_MULTIPROC_DIR"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

echo "Starting agent: $WORKERS workers on $HOST:$PORT"
echo "PROMETHEUS_MULTIPROC_DIR=$PROMETHEUS_MULTIPROC_DIR"

exec uv run uvicorn agent.server:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS"
