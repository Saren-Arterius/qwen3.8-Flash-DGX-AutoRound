#!/usr/bin/env bash
# Example launcher for scripts/serve-intel-ar.sh — point the paths at your
# machine (or keep your real settings as a local-only commit on top of this).
# Every knob here is passed through to serve-intel-ar.sh's docker run;
# anything unset falls back to that script's defaults.
cd "$(dirname "$0")"

# Required: the prepared checkpoint (int4 experts + int8 lm_head + fp8 side
# layers — see tools/) and the stripped fp8 ngram/PLE table directory.
export MODEL_DIR="/path/to/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid"
export TABLE_DIR="/path/to/ple-table-fp8"

export PORT=8000
export SERVED_NAME=qwen
export TOOL_PARSER=qwen3_xml
export SEQS=8
export MTP=3
export PREFIX_CACHE=1

# Deterministic memory sizing for unified-memory boxes (GB10 / DGX Spark):
# near-zero utilization fraction plus an explicit KV pool, so the driver
# never oversubscribes the unified pool (NV_ERR_NO_MEMORY / Xid 31 crashes).
export GPU_MEM=0.01
export KV_BYTES=20g

# Set to 1 when TABLE_DIR sits on remote RAM or there is no page-cache
# headroom: madvise(MADV_RANDOM) the PLE mmap so faults stay single-page.
export PLE_MADV_RANDOM=0

# Prefix-cache diagnosis logging (VLLM_HIT_DEBUG=1 in the container):
# per-group hit breakdown, mamba boundary publication, evictions, chunk stops.
export HIT_DEBUG=0

# Never-evict pin: any request whose prompt contains this exact substring has
# its prompt-prefix KV blocks pinned (held out of eviction) — meant for a
# long fixed system prompt. Empty disables it.
export PIN_PROMPT=''

exec scripts/serve-intel-ar.sh
