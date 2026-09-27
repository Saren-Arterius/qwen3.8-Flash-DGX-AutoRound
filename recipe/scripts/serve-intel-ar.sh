#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Derived from Saren-Arterius/qwen3.8-Flash-DGX-AutoRound, commit 01c5914.
# Adds portable configuration forwarding:
#   PLE_FAST_ROWS/PLE_CHUNK/PLE_STATS_SEC forwarded as VLLM_PLE_MMAP_* when set; SPEC_EXTRA merged into
#   --speculative-config; a --print mode that prints the docker argv and exits before any docker command.
set -euo pipefail

NAME="${NAME:-qwen38-flash}"
IMAGE="${IMAGE:-qwen38-flash-dgx}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid}"
TABLE_DIR="${TABLE_DIR:-/models/ple-table-fp8}"
PORT="${PORT:-18300}"
CTX="${CTX:-262144}"
SEQS="${SEQS:-8}"
# Local release safety defaults: the promoted GB10 memory rails, also pinned
# explicitly in the README quick start.
GPU_MEM="${GPU_MEM:-0.01}"
MTP="${MTP:-3}"
PREWARM="${PREWARM:-1}"
TOOL_PARSER="${TOOL_PARSER:-qwen3_coder}"
EXTRA="${EXTRA:-}"
# KV_BYTES: size the KV cache explicitly (e.g. 13.2g) instead of by
# gpu-memory-utilization fraction — deterministic footprint on unified-memory
# boxes where "free memory" profiling is unreliable. Pair with a tiny GPU_MEM.
KV_BYTES="${KV_BYTES:-16g}"
EXTRA="--kv-cache-memory-bytes $KV_BYTES $EXTRA"

# PLE gather = CPU op + H2D copy: must run OUTSIDE CUDA graphs.
SPLIT='["vllm::unified_attention_with_output","vllm::unified_mla_attention_with_output","vllm::mamba_mixer2","vllm::mamba_mixer","vllm::short_conv","vllm::qwen3_8_flash_next_ple_short_conv","vllm::qwen3_8_flash_next_qsa_with_output","vllm::linear_attention","vllm::qwen_gdn_attention_core","vllm::qwen_gdn_attention_core_fused_norm_packed","vllm::sparse_attn_indexer","vllm::ple_mmap_lookup"]'
CC="${CC:--cc.cudagraph_mode=PIECEWISE -cc.splitting_ops=$SPLIT}"

# FLASHINFER_AUTOTUNE=1 drops --no-enable-flashinfer-autotune (longer warmup,
# possibly faster kernels; default off as inherited from the NVFP4 recipe).
AT_ARG=--no-enable-flashinfer-autotune
[ "${FLASHINFER_AUTOTUNE:-0}" = 1 ] && AT_ARG=

SPEC=()
if [ "$MTP" != 0 ]; then
  # local: (b) base object is the pre-adoption one; SPEC_EXTRA (a JSON
  # local: *object* string, e.g. '{"rejection_sample_method":"block"}') is
  # local: merged on top of it so the launcher can add speculative-config
  # local: keys without editing this script. Right-hand keys win. jq is
  # local: required only on this path (SPEC_EXTRA set and non-empty).
  SPEC_JSON="{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP}}"
  if [ -n "${SPEC_EXTRA:-}" ]; then
    if ! command -v jq >/dev/null 2>&1; then
      echo "serve-intel-ar.sh: SPEC_EXTRA is set but jq is not installed" >&2
      exit 2
    fi
    if ! SPEC_JSON="$(jq -c -n --argjson a "$SPEC_JSON" --argjson b "$SPEC_EXTRA" '$a + $b')"; then
      echo "serve-intel-ar.sh: SPEC_EXTRA is not a valid JSON object: $SPEC_EXTRA" >&2
      exit 2
    fi
  fi
  SPEC=(--speculative-config "$SPEC_JSON")
fi

# PREFIX_CACHE=1: upstream disabled prefix caching over a CUBLAS error in the
# GDN in_proj GEMM on the cached-block path; the fp8-hybrid in_proj bypasses
# that kernel, and with FP8_HYBRID=1 prefix caching has been stable here.
PC_ARG=--no-enable-prefix-caching
[ "${PREFIX_CACHE:-1}" = 1 ] && PC_ARG=--enable-prefix-caching

# Never-evict pin: PIN_PROMPT="some exact substring of your system prompt"
# keeps that prompt's KV blocks resident across other traffic (needs
# PREFIX_CACHE=1). See docs/CONFIGURATION.md.
PIN_PROMPT="${PIN_PROMPT:-}"
PIN_ARG=()
if [ -n "$PIN_PROMPT" ] && [ "${PREFIX_CACHE:-0}" = 1 ]; then
  PIN_ARG=(--never-evict-kv-cache-prompt-includes "$PIN_PROMPT"
           --never-evict-kv-cache-max-fraction "${PIN_MAX_FRACTION:-0.25}")
fi

# local: (a) PLE mmap gather knobs. Each variable is forwarded ONLY when it
# local: is set and non-empty, so leaving one unset reproduces the
# local: pre-adoption container environment exactly (vLLM's own built-in
# local: default applies inside the container) rather than pinning a value
# local: here. PLE_STATS_SEC in particular is a diagnostics cadence and is
# local: intentionally left unset by the promoted launcher.
PLE_ARGS=()
if [ -n "${PLE_FAST_ROWS:-}" ]; then PLE_ARGS+=(-e "VLLM_PLE_MMAP_FAST_ROWS=$PLE_FAST_ROWS"); fi
if [ -n "${PLE_CHUNK:-}" ];     then PLE_ARGS+=(-e "VLLM_PLE_MMAP_CHUNK=$PLE_CHUNK");         fi
if [ -n "${PLE_STATS_SEC:-}" ]; then PLE_ARGS+=(-e "VLLM_PLE_MMAP_STATS_SEC=$PLE_STATS_SEC"); fi

# local: (c) the docker run command is captured verbatim into an array so
# local: that --print can dump it instead of executing it. Unquoted
# local: $PC_ARG/$CC/$AT_ARG/$EXTRA still word-split exactly as they did in
# local: the original command line, so the argv is unchanged.
# shellcheck disable=SC2086,SC2206
DOCKER_RUN=(docker run -d --name "$NAME" --restart unless-stopped \
  --gpus all --ipc=host --shm-size 16g -p "${PORT}:8000" \
  -v "$MODEL_DIR:/model:ro" -v "$TABLE_DIR:/ple-table:ro" \
  -e VLLM_PLE_MMAP=1 -e VLLM_PLE_MMAP_WORKERS="${WORKERS:-32}" -e VLLM_PLE_MMAP_PREWARM="$PREWARM" -e VLLM_PLE_MMAP_PREFETCH="${PLE_PREFETCH:-0}" \
  -e VLLM_PLE_MMAP_MADV_RANDOM="${PLE_MADV_RANDOM:-0}" \
  "${PLE_ARGS[@]}" \
  -e VLLM_HIT_DEBUG="${HIT_DEBUG:-0}" \
  -e VLLM_STEP_PROFILE="${STEP_PROFILE:-0}" \
  -e VLLM_PLE_MMAP_DIR=/ple-table \
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

# local: (c) dry-run: print the argv, one shell-quoted argument per line, and
# local: exit before `docker rm -f` / `docker run` touch anything.
[ $# -le 1 ] || { echo "serve-intel-ar.sh: too many arguments" >&2; exit 64; }
case "${1:-}" in
  --print) printf '%q\n' "${DOCKER_RUN[@]}"; exit 0 ;;
  "") ;;
  *) echo "serve-intel-ar.sh: unknown argument '$1'" >&2; exit 64 ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1 || true
"${DOCKER_RUN[@]}"

echo ">> $NAME starting on :$PORT (ctx $CTX, mtp=$MTP, seqs=$SEQS, gpu_mem=$GPU_MEM)"
echo ">> follow with: docker logs -f $NAME"
