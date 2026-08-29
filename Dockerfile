# Qwen3.8-Flash-Next on a single DGX Spark / GB10, via vLLM.
#
# Starts from the official Qwen3.8-Flash-Next vLLM image and appends one patch:
# it serves the 51B-parameter n-gram ("PLE") table from disk via mmap instead of
# keeping it resident in the 128 GB unified pool. That is the single change that
# lets the ~176B checkpoint (int4/int8/fp8 hybrid, ~71 GiB resident here;
# 122 GiB NVFP4 upstream) fit next to a real KV cache on one box.
#
#   docker build -t qwen38-flash-dgx .
#
# The base image is multi-arch (arm64 for the Spark's Grace CPU). Pinned by digest
# for reproducibility; bump the tag below if the upstream recipe moves.
FROM vllm/vllm-openai:qwen38-flash-next@sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8

# Package layout inside the official image (vLLM 0.1.dev20073, torch 2.13 cu130,
# numpy 2.2.6 — the patch needs numpy, already present).
ARG SP=/usr/local/lib/python3.12/dist-packages
ARG PLE=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py

COPY src/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py

# Append the hook to the model file. No-op unless VLLM_PLE_MMAP=1 at runtime, so
# the image still behaves exactly like upstream when the flag is off.
RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ple_layer.py patched OK')"

# spark-fla-shmem (from Saren's 122B recipe): sm121 reports 99 KiB shared mem
# (= ADA, where big tiles fit) but the FLA gate demands 100 KiB -> the 36 GDN
# layers run small Triton tiles. Lower the gate so GB10 gets big tiles.
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem: GB10 99KiB = ADA, big GDN tiles fit|' ${FLA_UTILS} \
 && grep -q "spark-fla-shmem" ${FLA_UTILS} && echo "fla shmem gate patched"

# spark-fla-warps (fla-org/flash-linear-attention#953): the chunked delta-rule
# state kernel races on Blackwell when autotune picks num_warps=4 — a tl.dot
# recurrence race yielding nondeterministic h/v_new, i.e. corrupt GDN state.
# The USE_INITIAL_STATE (prefix-cache resume) variant autotunes separately, so
# corruption tracks the cached-block path. Upstream pins num_warps=2 on
# Blackwell; this image only ever runs on GB10, so pin unconditionally.
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps: fla#953 Blackwell tl.dot race|' ${FLA_CDH} \
 && grep -q "spark-fla-warps" ${FLA_CDH} && echo "fla num_warps pinned"

# int4+fp8 hybrid: dispatch blockwise-fp8 side layers from AutoGPTQConfig
# (no-op unless VLLM_FP8_HYBRID=1 at runtime).
ARG GPTQ_PY=${SP}/vllm/model_executor/layers/quantization/auto_gptq.py
COPY src/vllm_fp8_hybrid.py ${SP}/vllm_fp8_hybrid.py
RUN printf '\n\n# --- qwen38-flash-dgx: int4+fp8 hybrid dispatch (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> ${GPTQ_PY} \
 && python3 -c "import ast; ast.parse(open('${GPTQ_PY}').read()); print('auto_gptq.py patched OK')"

# never-evict prompt pinning (pin-only port of the 122B recipe's arc_pin2):
# --never-evict-kv-cache-prompt-includes pins the HA system prompt's KV blocks
# against eviction. No-op unless the flag is passed at runtime.
COPY src/patch_never_evict.py /tmp/patch_never_evict.py
RUN python3 /tmp/patch_never_evict.py && rm /tmp/patch_never_evict.py

# Let the LM head pick up the checkpoint's quantization (int8 GPTQ head):
# upstream constructs ParallelLMHead without quant_config, forcing bf16.
ARG MODEL_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/model.py
ARG MTP_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/mtp.py
RUN cp ${MODEL_PY} ${MODEL_PY}.orig && cp ${MTP_PY} ${MTP_PY}.orig \
 && sed -i 's|prefix=maybe_prefix(prefix, "lm_head"),|quant_config=vllm_config.quant_config,\n            prefix=maybe_prefix(prefix, "lm_head"),|' ${MODEL_PY} \
 && sed -i 's|prefix=maybe_prefix(prefix, "lm_head"),|quant_config=vllm_config.quant_config,\n                    prefix=maybe_prefix(prefix, "lm_head"),|' ${MTP_PY} \
 && grep -c 'quant_config=vllm_config.quant_config' ${MODEL_PY} ${MTP_PY} \
 && python3 -c "import ast; [ast.parse(open(p).read()) for p in ('${MODEL_PY}','${MTP_PY}')]; print('lm_head patched OK in model.py + mtp.py')"

# mamba align-mode state-copy hardening (the "Xid 31 / illegal memory access
# under load" crash with PREFIX_CACHE=1 + MTP — also blazux/qwen3.8-Flash-DGX#2):
# CUDA_LAUNCH_BLOCKING=1 caught the fault synchronously inside vLLM's
# precopy_mamba_align_fused_kernel reading a wild address derived from a bad
# block id. src/mamba_utils_guarded.py is the image's stock
# vllm/v1/worker/mamba_utils.py plus:
#   1. upstream a02cfccbc6 "[Bugfix][Mamba] Fix overlapping state copy race"
#      (vllm#50729, landed after this image's vLLM snapshot)
#   2. a bounds guard in _copy_mamba_state_block: block ids are validated
#      against each state pool before dereferencing; an out-of-range id skips
#      the copy and bumps a counter (logged as "mamba state-copy guard")
#      instead of taking down the CUDA context.
ARG MAMBA_UTILS=${SP}/vllm/v1/worker/mamba_utils.py
RUN cp ${MAMBA_UTILS} ${MAMBA_UTILS}.orig
COPY src/mamba_utils_guarded.py ${MAMBA_UTILS}
RUN python3 -c "import ast; ast.parse(open('${MAMBA_UTILS}').read()); print('mamba_utils.py guarded OK')"

# prefix-cache diagnosis logging (VLLM_HIT_DEBUG=1): per-group hit breakdown,
# mamba boundary-state publication, cached-block eviction, prefill chunk stops.
COPY src/patch_hit_debug.py /tmp/patch_hit_debug.py
RUN python3 /tmp/patch_hit_debug.py && rm /tmp/patch_hit_debug.py

# prefill chunks must end at MAMBA block boundaries (1600), not the scheduler
# minimum block size (8) — otherwise cold requests publish no mamba state and
# repeated prompts only hit the prefix cache from the 3rd request on.
COPY src/patch_mamba_align_split.py /tmp/patch_mamba_align_split.py
RUN python3 /tmp/patch_mamba_align_split.py && rm /tmp/patch_mamba_align_split.py

# On-demand torch.profiler around engine steps (VLLM_STEP_PROFILE=1 +
# touch /tmp/profile_trigger). This vLLM predates VLLM_TORCH_PROFILER_DIR.
COPY src/patch_step_profile.py /tmp/patch_step_profile.py
RUN python3 /tmp/patch_step_profile.py && rm /tmp/patch_step_profile.py

# PLE table over RDMA (VLLM_PLE_RDMA=<host:port>): one-sided READs against
# the wtako table daemon; ~20us per decode gather vs ~3ms mmap page faults.
# Container must run with --device /dev/infiniband --ulimit memlock=-1:-1.
RUN apt-get update && apt-get install -y --no-install-recommends libibverbs-dev gcc \
    && rm -rf /var/lib/apt/lists/*
COPY src/ple_rdma/ple_rdma.c /tmp/ple_rdma.c
RUN gcc -O2 -Wall -shared -fPIC /tmp/ple_rdma.c -libverbs \
      -o /usr/local/lib/python3.12/dist-packages/libple_rdma.so && rm /tmp/ple_rdma.c
COPY src/ple_rdma/ple_rdma.py /usr/local/lib/python3.12/dist-packages/ple_rdma.py
