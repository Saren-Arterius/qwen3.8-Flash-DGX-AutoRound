#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
# magi3: T80 toggle. An explicit MODEL_DIR always wins; otherwise T80=1
# (default) serves the dense-MTP g32 dir, T80=0 the base hybrid dir.
T80="${T80:-1}"
HYBRID_DIR="${HYBRID_DIR:-/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid}"
T80_DIR="${T80_DIR:-/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32}"
if [ -z "${MODEL_DIR:-}" ]; then
  if [ "$T80" = 1 ]; then MODEL_DIR="$T80_DIR"; else MODEL_DIR="$HYBRID_DIR"; fi
fi
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
# magi-v3 dynamic draft depth (image patch; off = fixed K above). Baked into
# the image at build; these only tune the runtime controller.
export MTP_DEPTH="${MTP_DEPTH:-off}"
export MTP_DEPTH_MIN="${MTP_DEPTH_MIN:-3}"
export MTP_DEPTH_WINDOW="${MTP_DEPTH_WINDOW:-48}"
export MTP_DEPTH_PROMOTE="${MTP_DEPTH_PROMOTE:-60,45}"
export MTP_DEPTH_DEMOTE="${MTP_DEPTH_DEMOTE:-25,15}"
export MTP_DEPTH_LOG="${MTP_DEPTH_LOG:-0}"
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
# magi3: 65k draft-vocab cut toggle (option 1 of 2; T80 is the other).
# DRAFT_VOCAB=1 (default, shipped) scores only the 65,536-id slice: cheaper
# draft passes, but CJK acceptance/tg drops. DRAFT_VOCAB=0 scores the full
# head. An explicit MTP_DRAFT_VOCAB path is honoured only when the cut is on
# (use it to supply a custom id set).
DRAFT_VOCAB="${DRAFT_VOCAB:-1}"
if [ "$DRAFT_VOCAB" = 0 ]; then
  MTP_DRAFT_VOCAB=""
elif [ -z "${MTP_DRAFT_VOCAB:-}" ]; then
  MTP_DRAFT_VOCAB="${HOME}/.cache/qwen38-v16b/draft-vocab-ids-K65536.txt"
fi
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

DD_ARGS=()
if [ -n "${MTP_DEPTH:-}" ];          then DD_ARGS+=(-e "MTP_DEPTH=$MTP_DEPTH"); fi
if [ -n "${MTP_DEPTH_MIN:-}" ];      then DD_ARGS+=(-e "MTP_DEPTH_MIN=$MTP_DEPTH_MIN"); fi
if [ -n "${MTP_DEPTH_WINDOW:-}" ];    then DD_ARGS+=(-e "MTP_DEPTH_WINDOW=$MTP_DEPTH_WINDOW"); fi
if [ -n "${MTP_DEPTH_PROMOTE:-}" ];   then DD_ARGS+=(-e "MTP_DEPTH_PROMOTE=$MTP_DEPTH_PROMOTE"); fi
if [ -n "${MTP_DEPTH_DEMOTE:-}" ];    then DD_ARGS+=(-e "MTP_DEPTH_DEMOTE=$MTP_DEPTH_DEMOTE"); fi
if [ -n "${MTP_DEPTH_LOG:-}" ];       then DD_ARGS+=(-e "MTP_DEPTH_LOG=$MTP_DEPTH_LOG"); fi

# magi-v3 dynamic depth: EngineCore/workers spawn with a scrubbed env, so the
# controller reads /tmp/mtp_depth.json, not environ (same reason upstream uses
# a JSON file). Written on every launch in the upstream schema
# (mode/min/window/promote/demote/log); policy math is libmbx_mtp.so.
DD_FILE="${DD_FILE:-/tmp/mtp_depth_${NAME}.json}"
DD_MOUNT=(-v "$DD_FILE:/tmp/mtp_depth.json:ro")

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
  "${DD_ARGS[@]}" \
  "${DD_MOUNT[@]}" \
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
# Upstream-schema depth config (myllmbox recipe.yaml mtp_depth section).
python3 - "$DD_FILE" <<'EOF'
import json, os, sys
path = sys.argv[1]
def pair(s, default):
    try:
        a, b = (float(x) for x in str(s).replace(",", " ").split()[:2])
        return [a, b]
    except Exception:
        return default
mode = "dynamic" if os.environ.get("MTP_DEPTH", "off").strip().lower() == "dynamic" else "off"
log = os.environ.get("MTP_DEPTH_LOG", "0").lower() in ("1", "true", "yes")
cfg = {
    "mode": mode,
    "min": int(os.environ.get("MTP_DEPTH_MIN", "3")),
    "window": int(os.environ.get("MTP_DEPTH_WINDOW", "48")),
    "promote": pair(os.environ.get("MTP_DEPTH_PROMOTE", "60,45"), [60.0, 45.0]),
    "demote": pair(os.environ.get("MTP_DEPTH_DEMOTE", "25,15"), [25.0, 15.0]),
    "log": log,
}
with open(path, "w") as fh:
    json.dump(cfg, fh)
print("wrote %s: %s" % (path, cfg))
EOF
"${DOCKER_RUN[@]}"

echo ">> $NAME starting on :$PORT (ctx $CTX, mtp=$MTP, seqs=$SEQS, gpu_mem=$GPU_MEM)"
echo ">> follow with: docker logs -f $NAME"
