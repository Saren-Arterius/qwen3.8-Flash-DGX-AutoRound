#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32}"
# NOTE "-" not ":-": TABLE_DIR="" is meaningful (no local table; PLE rows
# come via VLLM_PLE_RDMA instead), only an UNSET var gets the default.
TABLE_DIR="${TABLE_DIR-/models/ple-table-fp8}"
PLE_RDMA="${PLE_RDMA:-}"
PLE_RDMA_DEV="${PLE_RDMA_DEV:-roceP2p1s0f0}"
PLE_RDMA_GID="${PLE_RDMA_GID:-auto}"
PLE_RDMA_PREFETCH="${PLE_RDMA_PREFETCH:-1}"
PORT="${PORT:-18300}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-8}"
# Local release safety defaults mirror the promoted env even when this script
# is inspected or called directly. launch.sh still supplies the exact env.
GPU_MEM="${GPU_MEM:-0.01}"
MTP="${MTP:-3}"
PREWARM="${PREWARM:-1}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
EXTRA="${EXTRA:-}"
KV_BYTES="${KV_BYTES:-16g}"
EXTRA="--kv-cache-memory-bytes $KV_BYTES $EXTRA"

SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

AT_ARG=--no-enable-flashinfer-autotune
[ "${FLASHINFER_AUTOTUNE:-0}" = 1 ] && AT_ARG=

SPEC=()
if [ "$MTP" != 0 ]; then
  SPEC_JSON="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}"
  if [ -n "${SPEC_EXTRA:-}" ]; then
    if ! command -v jq >/dev/null 2>&1; then
      echo "serve.sh: SPEC_EXTRA is set but jq is not installed" >&2
      exit 2
    fi
    if ! SPEC_JSON="$(jq -c -n --argjson a "$SPEC_JSON" --argjson b "$SPEC_EXTRA" '$a + $b')"; then
      echo "serve.sh: SPEC_EXTRA is not a valid JSON object: $SPEC_EXTRA" >&2
      exit 2
    fi
  fi
  SPEC=(--speculative-config "$SPEC_JSON")
fi

PC_ARG=--no-enable-prefix-caching
[ "${PREFIX_CACHE:-1}" = 1 ] && PC_ARG=--enable-prefix-caching

PIN_PROMPT="${PIN_PROMPT:-}"
PIN_ARG=()
if [ -n "$PIN_PROMPT" ] && [ "${PREFIX_CACHE:-0}" = 1 ]; then
  PIN_ARG=(--never-evict-kv-cache-prompt-includes "$PIN_PROMPT"
           --never-evict-kv-cache-max-fraction "${PIN_MAX_FRACTION:-0.25}")
fi

PLE_FAST_PATH="${PLE_FAST_PATH:-1}"
PLE_FAST_MAX_ROWS="${PLE_FAST_MAX_ROWS:-262144}"
MTP_DRAFT_VOCAB="${MTP_DRAFT_VOCAB:-${HOME}/.cache/qwen38-v16b/draft-vocab-ids-K65536.txt}"
DRAFTER_EXPERTS_FP8="${DRAFTER_EXPERTS_FP8:-1}"

PLE_ARGS=()
if [ -n "${PLE_FAST_ROWS:-}" ]; then PLE_ARGS+=(-e "VLLM_PLE_MMAP_FAST_ROWS=$PLE_FAST_ROWS"); fi
if [ -n "${PLE_CHUNK:-}" ];     then PLE_ARGS+=(-e "VLLM_PLE_MMAP_CHUNK=$PLE_CHUNK");         fi
if [ -n "${PLE_STATS_SEC:-}" ]; then PLE_ARGS+=(-e "VLLM_PLE_MMAP_STATS_SEC=$PLE_STATS_SEC"); fi
if [ -n "${PLE_FAST_PATH:-}" ]; then PLE_ARGS+=(-e "VLLM_PLE_MMAP_FAST_PATH=$PLE_FAST_PATH"); fi
if [ -n "${PLE_FAST_MAX_ROWS:-}" ]; then PLE_ARGS+=(-e "VLLM_PLE_MMAP_FAST_MAX_ROWS=$PLE_FAST_MAX_ROWS"); fi

DV_ARGS=()
if [ -n "${MTP_DRAFT_VOCAB:-}" ]; then
  [ -r "$MTP_DRAFT_VOCAB" ] || [ "${1:-}" = --print ] || { echo "serve.sh: MTP_DRAFT_VOCAB not readable: $MTP_DRAFT_VOCAB" >&2; exit 66; }
  DV_ARGS+=(-v "$MTP_DRAFT_VOCAB:/draft-vocab/ids.txt:ro" -e "VLLM_MTP_DRAFT_VOCAB=/draft-vocab/ids.txt")
fi
if [ -n "${DRAFTER_EXPERTS_FP8:-}" ]; then
  DV_ARGS+=(-e "VLLM_DRAFTER_EXPERTS_FP8=$DRAFTER_EXPERTS_FP8")
fi
if [ -n "${DRAFTER_EXPERTS_FP8_PREFIX:-}" ]; then
  DV_ARGS+=(-e "VLLM_DRAFTER_EXPERTS_FP8_PREFIX=$DRAFTER_EXPERTS_FP8_PREFIX")
fi

LOW_LATENCY_GEMM="${LOW_LATENCY_GEMM:-1}"
LLG_PDL="${LLG_PDL:-0}"
VERIFY_TOPK_TRITON="${VERIFY_TOPK_TRITON:-1}"
KEEP_DRAFT_BLOCKS="${KEEP_DRAFT_BLOCKS:-1}"

IT6_ARGS=()
if [ -n "${LOW_LATENCY_GEMM:-}" ];   then IT6_ARGS+=(-e "QWEN38NEXT_LOW_LATENCY_GEMM=$LOW_LATENCY_GEMM"); fi
if [ -n "${LLG_PDL:-}" ];            then IT6_ARGS+=(-e "QWEN38NEXT_LLG_PDL=$LLG_PDL"); fi
if [ -n "${VERIFY_TOPK_TRITON:-}" ]; then IT6_ARGS+=(-e "VLLM_VERIFY_TOPK_TRITON=$VERIFY_TOPK_TRITON"); fi
if [ -n "${KEEP_DRAFT_BLOCKS:-}" ];  then IT6_ARGS+=(-e "VLLM_KEEP_DRAFT_BLOCKS=$KEEP_DRAFT_BLOCKS"); fi

# The four flag strings below intentionally expand into separate argv words;
# the promoted env supplies fixed, whitespace-separated flags.
# TABLE_DIR="": no local table mount — PLE rows come from the RDMA daemon
# (VLLM_PLE_RDMA) instead of a mmapped directory. RDMA mode is exclusive.
TABLE_ARGS=()
if [ -n "${TABLE_DIR:-}" ]; then
  TABLE_ARGS=(-v "$TABLE_DIR:/ple-table:ro" -e VLLM_PLE_MMAP_DIR=/ple-table)
fi
RDMA_ARGS=()
IB_ARGS=()
if [ -n "${PLE_RDMA:-}" ]; then
  RDMA_ARGS=(-e "VLLM_PLE_RDMA=$PLE_RDMA" -e "VLLM_PLE_RDMA_DEV=$PLE_RDMA_DEV"
             -e "VLLM_PLE_RDMA_GID=$PLE_RDMA_GID" -e "VLLM_PLE_RDMA_PREFETCH=$PLE_RDMA_PREFETCH")
  IB_ARGS=(--device /dev/infiniband --ulimit memlock=-1:-1)
fi
# shellcheck disable=SC2206
DOCKER_RUN=(docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  "${IB_ARGS[@]}" \
  -v "$MODEL_DIR:/model:ro" "${TABLE_ARGS[@]}" \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS:-32}" -e VLLM_PLE_MMAP_PREWARM="$PREWARM" -e VLLM_PLE_MMAP_PREFETCH="${PLE_PREFETCH:-0}" \
  -e VLLM_PLE_MMAP_MADV_RANDOM="${PLE_MADV_RANDOM:-0}" \
  "${PLE_ARGS[@]}" \
  "${RDMA_ARGS[@]}" \
  "${DV_ARGS[@]}" \
  "${IT6_ARGS[@]}" \
  -e VLLM_HIT_DEBUG="${HIT_DEBUG:-0}" \
  -e VLLM_STEP_PROFILE="${STEP_PROFILE:-0}" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_FP8_HYBRID="${FP8_HYBRID:-1}" \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
  "$IMAGE" \
  /model --served-model-name "${SERVED_NAME:-qwen3.8-flash-next}" \
    --host 0.0.0.0 --port 8000 --load-format "${LOAD_FORMAT:-fastsafetensors}" \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM" \
    $PC_ARG --enable-chunked-prefill --max-num-batched-tokens 8192 \
    $CC \
    $AT_ARG \
    --kv-cache-dtype auto \
    $EXTRA \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser qwen3 \
    "${PIN_ARG[@]}" "${SPEC[@]}")

[ $# -le 1 ] || { echo "serve.sh: too many arguments" >&2; exit 64; }
case "${1:-}" in
  --print) printf '%q\n' "${DOCKER_RUN[@]}"; exit 0 ;;
  "") ;;
  *) echo "serve.sh: unknown argument '$1'" >&2; exit 64 ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
"${DOCKER_RUN[@]}"

echo ">> $NAME starting on :$PORT (ctx $CTX, mtp=$MTP, seqs=$SEQS, gpu_mem=$GPU_MEM)"
echo ">> follow with: docker logs -f $NAME"
