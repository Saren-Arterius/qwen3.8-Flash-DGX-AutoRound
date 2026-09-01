#!/usr/bin/env bash
# magi's serving config (the config commit on the "magi" branch); everything
# machine-specific lives here — scripts/serve-intel-ar.sh stays generic.
cd "$(dirname "$0")"

export MODEL_DIR="/mnt/storage@WTAKO/saren/AI/Qwen3.8-Flash-Next-W4A16-AutoRound"
export FP8_HYBRID=1
# PLE rows come from wtako's ple-rdma-server. RDMA mode is exclusive: no
# local table, no mmap — failed READs retry/stall instead of falling back.
# /mnt/ple-ram is swap-only since the 2026-08-30 cutover.
export TABLE_DIR=
export PLE_RDMA=192.168.0.1:18515

# export YARN=1
# export CTX=1000000

export SEQS=16
export TOOL_PARSER=qwen3_xml
export PORT=8000
export SERVED_NAME=qwen
# Deterministic memory: weights (~71.4G) + fixed 20G KV (~640k tokens)
# + activations, leaving real headroom for OS/q3asr/voxcpm/page cache.
# Replaces fraction sizing after dmesg showed NVRM NV_ERR_NO_MEMORY + Xid 31
# MMU faults: the unified pool was oversubscribed and the "illegal memory
# access" crashes tracked failed driver allocations.
export GPU_MEM=0.01
export KV_BYTES=20g
export MTP=3
export PREFIX_CACHE=1
# prefix-cache diagnosis: per-group hit breakdown, mamba publication,
# eviction and chunk-stop logs (one-liners per request/step)
# export HIT_DEBUG=1
# on-demand step profiler (touch /tmp/profile_trigger in the container)
export STEP_PROFILE=0
export PIN_PROMPT='You are "Magi AI", a smart home AI (via Home Assistant) and general knowledge assistant.'

exec scripts/serve-intel-ar.sh
