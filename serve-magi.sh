#!/usr/bin/env bash
# magi-v3-public on vLLM 0.32.0 — SSD mmap default, RDMA optional.
# Same stack as magi-v3 (Qwen4Exp + DFlash2, FULL_AND_PIECEWISE), but the
# PLE n-gram table comes from local SSD (/ple-table-fp8) unless PLE_RDMA
# is set. Based on serve-magi.sh.
# Usage: serve-magi.sh [--print]
set -euo pipefail

IMAGE="${IMAGE:-qwen38-flash-dgx:magi-v3-public}"
NAME="${NAME:-qwen38-flash}"
PORT="${PORT:-8000}"
AI=/mnt/storage@WTAKO/saren/AI
MODEL_DIR="${MODEL_DIR:-$AI/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid}"
DRAFT_DIR="${DRAFT_DIR:-$AI/Qwen3.8-Flash-Next-DFlash2}"
# Public default: local SSD. Set PLE_RDMA=192.168.0.1:18515 for the private RDMA path.
PLE_RDMA="${PLE_RDMA:-}"
TABLE_DIR="${TABLE_DIR:-$HOME/ple-table-fp8}"
KV_BYTES="${KV_BYTES:-20g}"
SEQS="${SEQS:-16}"
SPEC="${SPEC:-dflash}"
NSPEC="${NSPEC:-3}"
PIN_PROMPT="${PIN_PROMPT:-You are \"Magi AI\", a smart home AI (via Home Assistant) and general knowledge assistant.}"

SPEC_ARGS=() DRAFT_MOUNT=()
case "$SPEC" in
  mtp)    SPEC_ARGS=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$NSPEC,\"rejection_sample_method\":\"block\",\"draft_sample_method\":\"probabilistic\"}") ;;
  dflash) DRAFT_MOUNT=(-v "$DRAFT_DIR:/drafter:ro")
          SPEC_ARGS=(--speculative-config "{\"method\":\"dflash\",\"model\":\"/drafter\",\"num_speculative_tokens\":$NSPEC,\"disable_eagle_block_drop\":true}") ;;
  0)      ;;
  *) echo "SPEC must be mtp|dflash|0" >&2; exit 64 ;;
esac

# PLE source: RDMA (private) or local SSD mmap (public default).
PLE_ARGS=() PLE_MOUNTS=() PLE_DEVICES=()
if [ -n "$PLE_RDMA" ]; then
  PLE_DEVICES=(--device /dev/infiniband)
  PLE_ARGS=(-e VLLM_PLE_RDMA="$PLE_RDMA" -e VLLM_PLE_RDMA_DEV=roceP2p1s0f0 -e VLLM_PLE_RDMA_GID=auto
            -e VLLM_PLE_RDMA_PREFETCH=0 -e VLLM_PLE_RDMA_GRAPH="${PLE_GRAPH:-1}")
else
  PLE_MOUNTS=(-v "$TABLE_DIR:/ple-table-fp8:ro")
  PLE_ARGS=(-e VLLM_PLE_MMAP_DIR=/ple-table-fp8
            -e VLLM_PLE_MMAP_WORKERS="${PLE_WORKERS:-32}" -e VLLM_PLE_MMAP_CHUNK="${PLE_CHUNK:-2048}"
            -e VLLM_PLE_MMAP_PREWARM="${PLE_PREWARM:-0}")
fi

RUN=(docker run -d --name "$NAME" --restart "${RESTART:-unless-stopped}"
  --gpus all --ipc=host --shm-size 16g -p "$PORT:8000"
  "${PLE_DEVICES[@]}" --ulimit memlock=-1:-1
  -v "$MODEL_DIR:/model:ro" "${DRAFT_MOUNT[@]}" "${PLE_MOUNTS[@]}"
  "${PLE_ARGS[@]}"
  -e VLLM_FP8_HYBRID=1 -e VLLM_MARLIN_USE_ATOMIC_ADD=1
  -e VLLM_USE_DEEP_GEMM=0 -e VLLM_USE_FLASHINFER_SAMPLER=1
  "$IMAGE"
  /model --served-model-name qwen --host 0.0.0.0 --port 8000
  --load-format fastsafetensors --max-model-len 262144 --max-num-seqs "$SEQS"
  --gpu-memory-utilization 0.01 --kv-cache-memory-bytes "$KV_BYTES"
  --enable-prefix-caching --enable-chunked-prefill --max-num-batched-tokens 8192
  -cc.cudagraph_mode="${CG_MODE:-FULL_AND_PIECEWISE}" --no-enable-flashinfer-autotune --kv-cache-dtype auto --enable-prompt-tokens-details
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
  --never-evict-kv-cache-prompt-includes "$PIN_PROMPT" \
  "${SPEC_ARGS[@]}")

[ "${1:-}" = --print ] && { printf '%q ' "${RUN[@]}"; echo; exit 0; }
docker rm -f "$NAME" >/dev/null 2>&1 || true
"${RUN[@]}"
echo ">> $NAME on :$PORT (vLLM 0.32 public, spec=$SPEC/$NSPEC, kv=$KV_BYTES, ple=${PLE_RDMA:-local:$TABLE_DIR})"
