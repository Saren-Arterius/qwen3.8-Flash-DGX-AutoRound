#!/usr/bin/env bash
# Serve the int4+int8+fp8 hybrid build of Qwen3.8-Flash-Next on a DGX Spark:
# Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound experts via GPTQ-Marlin int4,
# int8 GPTQ lm_head, blockwise-fp8 side layers, PLE n-gram table mmapped from
# a separate directory (fp8 table recommended — see tools/fetch-ple-table-fp8.sh).
#
# The checkpoint must be prepared first: see docs/OPTIMIZATIONS.md ("Checkpoint
# preparation") — tools/quantize_lm_head_int8.py, tools/fp8_convert.py,
# tools/strip_ngram_index.py, plus the quantization_config for config.json.
#
#   MODEL_DIR=/models/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound \
#   TABLE_DIR=/models/ple-table-fp8 scripts/serve-intel-ar.sh
#
#   MTP=0 ... scripts/serve-intel-ar.sh     # no speculation (first-boot sanity)
set -euo pipefail

NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound}"
# NOTE "-" not ":-": TABLE_DIR="" is meaningful (no local table; PLE rows
# come via VLLM_PLE_RDMA instead), only an UNSET var gets the default.
TABLE_DIR="${TABLE_DIR-/models/ple-table-fp8}"
PORT="${PORT:-18300}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-8}"
GPU_MEM="${GPU_MEM:-0.85}"
MTP="${MTP:-2}"
PREWARM="${PREWARM:-1}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
EXTRA="${EXTRA:-}"
# KV_BYTES: size the KV cache explicitly (e.g. 13.2g) instead of by
# gpu-memory-utilization fraction — deterministic footprint on unified-memory
# boxes where "free memory" profiling is unreliable. Pair with a tiny GPU_MEM.
[ -n "${KV_BYTES:-}" ] && EXTRA="--kv-cache-memory-bytes $KV_BYTES $EXTRA"

# PLE gather = CPU op + H2D copy: must run OUTSIDE CUDA graphs.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

# FLASHINFER_AUTOTUNE=1 drops --no-enable-flashinfer-autotune (longer warmup,
# possibly faster kernels; default off as inherited from the NVFP4 recipe).
AT_ARG=--no-enable-flashinfer-autotune
[ "${FLASHINFER_AUTOTUNE:-0}" = 1 ] && AT_ARG=

# TABLE_DIR="": no local table mount — PLE rows come from the RDMA daemon
# (VLLM_PLE_RDMA) instead of a mmapped directory.
TABLE_ARGS=()
[ -n "${TABLE_DIR:-}" ] && TABLE_ARGS=(-v "$TABLE_DIR:/ple-table:ro" -e VLLM_PLE_MMAP_DIR=/ple-table)

SPEC=()
if [ "$MTP" != 0 ]; then
  SPEC=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}")
fi

# PREFIX_CACHE=1: upstream disabled prefix caching over a CUBLAS error in the
# GDN in_proj GEMM on the cached-block path; the fp8-hybrid in_proj bypasses
# that kernel, and with FP8_HYBRID=1 prefix caching has been stable here.
PC_ARG=--no-enable-prefix-caching
[ "${PREFIX_CACHE:-0}" = 1 ] && PC_ARG=--enable-prefix-caching

# Never-evict pin: PIN_PROMPT="some exact substring of your system prompt"
# keeps that prompt's KV blocks resident across other traffic (needs
# PREFIX_CACHE=1). See docs/OPTIMIZATIONS.md.
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
  -v "$MODEL_DIR:/model:ro" "${TABLE_ARGS[@]}" \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS:-32}" -e VLLM_PLE_MMAP_PREWARM="$PREWARM" \
  -e VLLM_PLE_MMAP_MADV_RANDOM="${PLE_MADV_RANDOM:-0}" \
  -e VLLM_PLE_MMAP_FAST_ROWS="${PLE_FAST_ROWS:-512}" -e VLLM_PLE_MMAP_CHUNK="${PLE_CHUNK:-2048}" \
  -e VLLM_HIT_DEBUG="${HIT_DEBUG:-0}" \
  -e VLLM_PLE_RDMA="${PLE_RDMA:-}" \
  -e VLLM_PLE_RDMA_VERIFY="${PLE_RDMA_VERIFY:-0}" \
  -e VLLM_PLE_RDMA_DEV="${PLE_RDMA_DEV:-roceP2p1s0f0}" \
  -e VLLM_PLE_RDMA_GID="${PLE_RDMA_GID:-5}" \
  -e VLLM_STEP_PROFILE="${STEP_PROFILE:-0}" \
  -e VLLM_MARLIN_USE_ATOMIC_ADD=1 \
  -e VLLM_FP8_HYBRID="${FP8_HYBRID:-1}" \
  -e VLLM_USE_DEEP_GEMM="${USE_DEEP_GEMM:-0}" \
  -e VLLM_MOE_USE_DEEP_GEMM="${MOE_DEEP_GEMM:-0}" \
  -e VLLM_USE_DEEP_GEMM_E8M0="${DEEP_GEMM_E8M0:-1}" \
  -e VLLM_TEST_FORCE_FP8_MARLIN="${FORCE_FP8_MARLIN:-0}" \
  -e VLLM_DISABLED_KERNELS="${DISABLED_KERNELS:-}" \
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
    "${PIN_ARG[@]}" "${SPEC[@]}"

echo ">> $NAME starting on :$PORT (ctx $CTX, mtp=$MTP, seqs=$SEQS, gpu_mem=$GPU_MEM)"
echo ">> follow with: docker logs -f $NAME"
