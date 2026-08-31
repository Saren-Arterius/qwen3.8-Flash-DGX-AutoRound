#!/usr/bin/env bash
# magi's local serving config — a LOCAL-ONLY commit on top of upstream.
# Everything machine-specific lives here; scripts/serve-intel-ar.sh stays
# generic and rebases cleanly on git pull.
cd "$(dirname "$0")"

export MODEL_DIR="/mnt/storage@WTAKO/saren/AI/Qwen3.8-Flash-Next-W4A16-AutoRound"
export FP8_HYBRID=1  # vendor build has no fp8 tensors
# no local table since the 2026-08-30 RDMA cutover: /mnt/ple-ram is swap-only,
# rows come from wtako's ple-rdma-server (PLE_RDMA below)
export TABLE_DIR=
# seq=8 + prefix cache + stock FLA gate: A/B for the 2026-08-28 CUDA
# illegal-access crashes (both on prefix-cache resume steps). If stable, the
# lowered FLA shmem gate (big GDN tiles at 99KiB) was the culprit.

# export YARN=1
# export CTX=1000000

export SEQS=16
export TOOL_PARSER=qwen3_xml
export PORT=8000
export SERVED_NAME=qwen
# Deterministic memory: weights (~71.4G) + fixed 13.2G KV (~320k tokens)
# + activations ≈ 90G, leaving ~30G real headroom for OS/q3asr/voxcpm/page
# cache. Replaces fraction sizing after dmesg showed NVRM NV_ERR_NO_MEMORY +
# Xid 31 MMU faults: the unified pool was oversubscribed and the "illegal
# memory access" crashes tracked failed driver allocations.
export GPU_MEM=0.01
export KV_BYTES=20g
export MTP=3
export PREFIX_CACHE=1
# remote-RAM PLE + no cache headroom: single-page faults, no readahead waste
export PLE_MADV_RANDOM=1
# decode gathers (<=512 rows) were inline+serial (~50us/fault to remote RAM);
# pooling them (FAST_ROWS=16, CHUNK=16, WORKERS=64) halves median cold-gather
# latency in isolation. Measured NEUTRAL on tg: gather is ~3.5ms of a ~65ms
# MTP decode step (~5%) - keep for cold-cache tails, don't expect tok/s.
export PLE_FAST_ROWS=16
export PLE_CHUNK=16
export WORKERS=64
# prefix-cache diagnosis: per-group hit breakdown, mamba publication,
# eviction and chunk-stop logs (one-liners per request/step)
# export HIT_DEBUG=1
# perf campaign: on-demand step profiler (touch /tmp/profile_trigger in ctr)
export STEP_PROFILE=0
# PLE table over RDMA from the wtako daemon (phase-2 VERIFY soak passed with
# zero mismatches; mmap dir dropped at the phase-3 cutover)
export PLE_RDMA=192.168.0.1:18515
export PLE_RDMA_VERIFY=0
# fp8 kernel experiment E2: FP8 Marlin (small-M specialist, handles block
# scales). E1 DeepGEMM rejected: SM120 kernels CUDA_ERROR_LAUNCH_FAILED on
# GB10/sm121 (likely sm_120a-locked cubins). Marlin sits below cutlass, so
# cutlass must be disabled by name.
# export FORCE_FP8_MARLIN=1
# export DISABLED_KERNELS=CutlassFp8BlockScaledMMKernel
export PIN_PROMPT='You are "Magi AI", a smart home AI (via Home Assistant) and general knowledge assistant.'

exec scripts/serve-intel-ar.sh
