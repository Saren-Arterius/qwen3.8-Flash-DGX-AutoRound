#!/usr/bin/env bash
# Serve the int4+int8+fp8 hybrid build of Qwen3.8-Flash-Next on the b12x stack (Dockerfile.b12x).
# Same env contract as scripts/serve-intel-ar.sh, so serve-magi.sh drives either stack:
#
#   STACK=b12x ./serve-magi.sh
#
# Differences from serve-intel-ar.sh, all inherent to the b12x stack:
#   * PLE table: RDMA only (PLE_RDMA required; b12x's own disk/RAM table modes are not used)
#   * b12x kernels: --linear-backend b12x, --gdn-decode-kernel b12x, b12x QSA (deterministic
#     top-k is native, so DET_TOPK/EXACT_TOPK do not apply)
#   * CUDA graphs: b12x's default FULL_AND_PIECEWISE (no -cc override); PLE split ops are
#     registered by the model itself
#   * MTP draft loads from a slim draft folder of the same checkpoint (built per container start)
#   * DRAFT_VOCAB / DRAFT_HEAD (patch 14) are not ported
set -euo pipefail

NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-b12x}"
MODEL_DIR="${MODEL_DIR:?MODEL_DIR is required}"
PLE_RDMA="${PLE_RDMA:?PLE_RDMA is required on the b12x stack (RDMA is its only PLE table source)}"
PORT="${PORT:-18300}"
CTX="${CTX:-262144}"
YARN="${YARN:-0}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-2}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
[ "${EXACT_TOPK:-0}" = 1 ] && { echo "EXACT_TOPK: not applicable on b12x (its QSA top-k is exact and deterministic)" >&2; exit 1; }
[ "${DRAFT_VOCAB:-0}" = 0 ] || { echo "DRAFT_VOCAB=$DRAFT_VOCAB: patch 14 is not ported to b12x" >&2; exit 1; }
[ "${DRAFT_HEAD:-int8}" = int8 ] || { echo "DRAFT_HEAD=$DRAFT_HEAD: patch 14 is not ported to b12x" >&2; exit 1; }

EXTRA="${EXTRA:-}"
[ "${ITER_DETAILS:-0}" = 1 ] && EXTRA="--enable-logging-iteration-details $EXTRA"
[ -n "${LONG_PREFILL_THRESHOLD:-}" ] && EXTRA="--long-prefill-token-threshold $LONG_PREFILL_THRESHOLD $EXTRA"
[ -n "${KV_BYTES:-}" ] && EXTRA="--kv-cache-memory-bytes $KV_BYTES $EXTRA"
AT_ARG=--no-enable-flashinfer-autotune
[ "${FLASHINFER_AUTOTUNE:-0}" = 1 ] && AT_ARG=

OVR_ARGS=()
YARN_OVR='{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}'
ALLOW_LONG=0
if [ "$YARN" != 0 ]; then OVR_ARGS=(--hf-overrides "$YARN_OVR"); ALLOW_LONG=1; fi

SPEC=()
if [ "$MTP" != 0 ]; then
  S="\"method\":\"mtp\",\"num_speculative_tokens\":${MTP},\"model\":\"/workspace/flashnext/draft\""
  [ "$YARN" != 0 ] && S="$S,\"max_model_len\":${CTX}"
  SPEC=(--speculative-config "{$S}")
fi

# mamba_cache_mode=align: what the production stack resolves on its own with prefix caching,
# stated explicitly as in the b12x recipe.
PC_ARGS=(--no-enable-prefix-caching)
[ "${PREFIX_CACHE:-0}" = 1 ] && PC_ARGS=(--enable-prefix-caching --mamba-cache-mode align)

PIN_PROMPT="${PIN_PROMPT:-}"
PIN_ARG=()
if [ -n "$PIN_PROMPT" ] && [ "${PREFIX_CACHE:-0}" = 1 ]; then
  PIN_ARG=(--never-evict-kv-cache-prompt-includes "$PIN_PROMPT"
           --never-evict-kv-cache-max-fraction "${PIN_MAX_FRACTION:-0.25}")
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2086
docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  --device /dev/infiniband --ulimit memlock=-1:-1 \
  -v "$MODEL_DIR:/model-src:ro" \
  -e CUTE_DSL_ARCH=sm_121a -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e VLLM_SSM_CONV_STATE_LAYOUT=DS \
  -e VLLM_USE_AOT_COMPILE=1 -e VLLM_USE_MEGA_AOT_ARTIFACT=1 -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e B12X_POLICY_MODE=auto -e VLLM_PLE_TABLE_MEMORY=disk \
  -e VLLM_PLE_RDMA="$PLE_RDMA" \
  -e VLLM_PLE_RDMA_DEV="${PLE_RDMA_DEV:-roceP2p1s0f0}" \
  -e VLLM_PLE_RDMA_GID="${PLE_RDMA_GID:-auto}" \
  -e VLLM_HIT_DEBUG="${HIT_DEBUG:-0}" \
  -e VLLM_STEP_PROFILE="${STEP_PROFILE:-0}" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_FP8_HYBRID="${FP8_HYBRID:-1}" \
  -e VLLM_USE_DEEP_GEMM=0 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN="$ALLOW_LONG" \
  -e CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}" \
  "$IMAGE" \
  /workspace/flashnext/model --served-model-name "${SERVED_NAME:-qwen3.8-flash-next}" \
    --host 0.0.0.0 --port 8000 --load-format "${LOAD_FORMAT:-fastsafetensors}" \
    --max-model-len "$CTX" --max-num-seqs "$SEQS" --gpu-memory-utilization "$GPU_MEM" \
    "${PC_ARGS[@]}" --enable-chunked-prefill --max-num-batched-tokens "$MAX_BATCHED" \
    --linear-backend b12x --gdn-decode-kernel b12x \
    $AT_ARG \
    --kv-cache-dtype auto \
    "${OVR_ARGS[@]}" $EXTRA \
    --enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER" --reasoning-parser qwen3 \
    "${PIN_ARG[@]}" "${SPEC[@]}"

echo ">> $NAME (b12x) starting on :$PORT (ctx $CTX, mtp=$MTP, seqs=$SEQS, gpu_mem=$GPU_MEM, batched=$MAX_BATCHED, long_prefill=${LONG_PREFILL_THRESHOLD:-off})"
echo ">> follow with: docker logs -f $NAME"
