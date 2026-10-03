#!/usr/bin/env bash
# magi-v3 production serving — ultrafast fork (v16b) + PLE-over-RDMA port.
# Tunables live in recipe/config/v16b/env; this wrapper pins magi's
# production choices: RDMA-exclusive fp8 PLE rows from wtako's
# ple-rdma-server (:18515), image qwen38-flash-dgx:magi-v3, container
# qwen38-flash on :8000 as "qwen". Called by ~/magi-stack/ram-client-ple.sh
# (watchdog). Forwards args (e.g. --print) to the recipe serve script.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

# vLLM aborts at boot if the table daemon is unreachable — wait here so the
# supervisor's warm timeout isn't burned on a crash loop. NOTE
# ram-client-ple.sh still waits on :18516 (NVFP4) first; flip that wait to
# :18515 when the NVFP4 server is retired.
until timeout 2 bash -c '</dev/tcp/192.168.0.1/18515' 2>/dev/null; do
  echo "serve-magi.sh: waiting for ple-rdma-server on wtako:18515..."
  sleep 5
done

export IMAGE="${IMAGE:-qwen38-flash-dgx:magi-v3}"
export NAME="${NAME:-qwen38-flash}"
export PORT="${PORT:-8000}"
export SERVED_NAME="${SERVED_NAME:-qwen}"
# RDMA-exclusive: no local table mount (TABLE_DIR="" is meaningful; an
# explicit TABLE_DIR still selects mmap mode).
export TABLE_DIR="${TABLE_DIR:-}"
export PLE_RDMA="${PLE_RDMA:-192.168.0.1:18515}"
# magi production: full draft head (DRAFT_VOCAB=0, CJK acceptance over the
# 65k cut) + T80 dense drafter on (takes effect once the T80 dir exists).
export DRAFT_VOCAB="${DRAFT_VOCAB:-0}"
# TEMPORARY bridge: the T80 dir isn't built yet — serve magi's mtpint4 hybrid
# (same family; boots this stack, minus the dense drafter) until
# ~/models/...-hybrid-mtpdense-g32 exists, when the T80 toggle default takes
# over. TODO: delete this block after the T80 build lands.
if [ -z "${MODEL_DIR:-}" ] && [ ! -d "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32" ]; then
  echo "serve-magi.sh: T80 dir absent, serving mtpint4 bridge checkpoint" >&2
  export MODEL_DIR=/mnt/storage@WTAKO/saren/AI/Qwen3.8-Flash-Next-W4A16-AutoRound-mtpint4
fi

exec recipe/config/v16b/serve.sh "$@"
