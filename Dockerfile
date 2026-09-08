# Qwen3.8-Flash-Next on a single DGX Spark / GB10, via vLLM — the int4/int8/fp8
# AutoRound fork of blazux/qwen3.8-Flash-DGX. (The magi branch adds a 12th
# patch: PLE table over RDMA.)
#
# Starts from the official Qwen3.8-Flash-Next vLLM image and layers patches on
# it. Numbering follows the README's "The patches" section; each is a no-op
# unless its runtime flag is set, or is a pure bug fix:
#
#   1. PLE table served from disk via mmap            (VLLM_PLE_MMAP=1)         — the one that makes it fit
#   2. GB10 FLA fixes (shmem gate, fla#953 warps)      (always on)
#   3. int8 GPTQ lm_head enablement                   (always on; reads the checkpoint's quant)
#   4. int4 + blockwise-fp8 hybrid dispatch            (VLLM_FP8_HYBRID=1)
#   5. Prefix caching: mamba-aligned chunk splits      (always on; needed for --enable-prefix-caching)
#   6. Never-evict prompt pinning                      (--never-evict-kv-cache-prompt-includes)
#   7. Mamba state-copy race fix + bounds guard        (always on)
#   8. Prefix-cache tracing                            (VLLM_HIT_DEBUG=1)
#   9. Exact, deterministic QSA top-k                  (VLLM_QSA_EXACT_TOPK=1)  — from upstream blazux
#  10. Deterministic persistent_topk kernel           (VLLM_QSA_DET_TOPK=1)    — @jschmied; replaces 9 at no prefill cost
#  11. On-demand step profiler                        (VLLM_STEP_PROFILE=1)
#  12. Per-step prefill metrics                      (--enable-logging-iteration-details)
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
ARG QSA_OPS=${SP}/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py

# --- 1. PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---------------------------------
# Upstream blazux's patch, extended here (any table dtype, relocatable dir, fast gather).
COPY src/vllm_ple_mmap.py ${SP}/vllm_ple_mmap.py
RUN cp ${PLE} ${PLE}.orig \
 && printf '\n\n# --- qwen38-flash-dgx: serve the PLE n-gram table from disk (VLLM_PLE_MMAP=1) ---\nfrom vllm_ple_mmap import apply as _ple_mmap_apply\n_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)\n' >> ${PLE} \
 && python3 -c "import ast; ast.parse(open('${PLE}').read()); print('ple_layer.py patched OK')"

# --- 2. GB10 FLA fixes ------------------------------------------------------------------
# 1) spark-fla-shmem (from the Qwen3.5-122B Spark recipe, Entrpi/qwen3.5-122B-A10B-on-spark):
#    sm_121 reports 99 KiB of shared memory per block (= ADA, where big tiles fit) but
#    the flash-linear-attention gate demands 100 KiB, so all 36 GDN layers silently
#    fell back to small Triton tiles. Lower the gate so GB10 gets big tiles.
# 2) spark-fla-warps (fla#953): the chunked delta-rule state kernel races on Blackwell
#    when autotune picks num_warps=4 — a tl.dot recurrence race yielding corrupt GDN
#    state; the USE_INITIAL_STATE (prefix-cache resume) variant autotunes separately,
#    so corruption tracked the cached-block path. Pin num_warps=2 unconditionally.
ARG FLA_UTILS=${SP}/vllm/third_party/flash_linear_attention/ops/utils.py
ARG FLA_CDH=${SP}/vllm/third_party/flash_linear_attention/ops/chunk_delta_h.py
RUN sed -i 's|DEFAULT = 102400|DEFAULT = 101376  # spark-fla-shmem: GB10 99KiB = ADA, big GDN tiles fit|' ${FLA_UTILS} \
 && grep -q "spark-fla-shmem" ${FLA_UTILS} && echo "fla shmem gate patched" \
 && sed -i 's|for num_warps in \[2, 4\]|for num_warps in [2]  # spark-fla-warps: fla#953 Blackwell tl.dot race|' ${FLA_CDH} \
 && grep -q "spark-fla-warps" ${FLA_CDH} && echo "fla num_warps pinned"

# --- 3. int8 GPTQ lm_head --------------------------------------------------------------
# Upstream constructs ParallelLMHead without quant_config, forcing a bf16 head; one
# added kwarg in the main model and the MTP draft lets it pick up the checkpoint's int8
# GPTQ packing (without the mtp.py half, MTP >= 3 crashes at load).
ARG MODEL_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/model.py
ARG MTP_PY=${SP}/vllm/models/qwen3_8_flash_next/nvidia/mtp.py
RUN cp ${MODEL_PY} ${MODEL_PY}.orig && cp ${MTP_PY} ${MTP_PY}.orig \
 && sed -i 's|prefix=maybe_prefix(prefix, "lm_head"),|quant_config=vllm_config.quant_config,\n            prefix=maybe_prefix(prefix, "lm_head"),|' ${MODEL_PY} \
 && sed -i 's|prefix=maybe_prefix(prefix, "lm_head"),|quant_config=vllm_config.quant_config,\n                    prefix=maybe_prefix(prefix, "lm_head"),|' ${MTP_PY} \
 && grep -c 'quant_config=vllm_config.quant_config' ${MODEL_PY} ${MTP_PY} \
 && python3 -c "import ast; [ast.parse(open(p).read()) for p in ('${MODEL_PY}','${MTP_PY}')]; print('lm_head patched OK in model.py + mtp.py')"

# --- 4. int4 + blockwise-fp8 hybrid dispatch (VLLM_FP8_HYBRID=1) ------------------------
# Routes the checkpoint's F8_E4M3 side layers (GDN in/out proj, QSA q/k/v/o, shared
# expert) from AutoGPTQConfig to vLLM's blockwise Fp8Config. No-op unless the flag is set.
ARG GPTQ_PY=${SP}/vllm/model_executor/layers/quantization/auto_gptq.py
COPY src/vllm_fp8_hybrid.py ${SP}/vllm_fp8_hybrid.py
RUN printf '\n\n# --- qwen38-flash-dgx: int4+fp8 hybrid dispatch (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid import apply as _fp8_hybrid_apply\n_fp8_hybrid_apply()\n' >> ${GPTQ_PY} \
 && python3 -c "import ast; ast.parse(open('${GPTQ_PY}').read()); print('auto_gptq.py patched OK')"

# --- 5. Prefix caching: mamba-aligned chunk splits + block_size seed fix -----------------
# Prefill chunks must end at MAMBA block boundaries (1600), not the scheduler minimum
# block size (8) — otherwise cold requests publish no mamba state and repeated prompts
# only hit the prefix cache from the 3rd request on. Also folds the mamba_hybrid.py
# state-slot seed fix (root-caused upstream by blazux, 8347e7c).
COPY src/patch_mamba_align_split.py /tmp/patch_mamba_align_split.py
RUN python3 /tmp/patch_mamba_align_split.py && rm /tmp/patch_mamba_align_split.py

# --- 6. Never-evict prompt pinning (--never-evict-kv-cache-prompt-includes) -------------
# Pin-only port of the Qwen3.5-122B recipe's arc_pin2 (on top of vllm#40270's ARC work):
# keeps the system prompt's KV blocks against eviction. No-op unless the flag is passed.
COPY src/patch_never_evict.py /tmp/patch_never_evict.py
RUN python3 /tmp/patch_never_evict.py && rm /tmp/patch_never_evict.py

# --- 7. Mamba state copy: vllm#50729 + bounds guard --------------------------------------
# The "Xid 31 / illegal memory access under load" crash with PREFIX_CACHE=1 + MTP (also
# blazux/qwen3.8-Flash-DGX#2): CUDA_LAUNCH_BLOCKING=1 caught the fault synchronously
# inside precopy_mamba_align_fused_kernel reading a wild address from a bad block id.
# src/mamba_utils_guarded.py = the image's vllm/v1/worker/mamba_utils.py plus
#   1. vllm#50729 "[Bugfix][Mamba] Fix overlapping state copy race" (@AndreasKaratzas)
#   2. a bounds guard in _copy_mamba_state_block: an out-of-range block id skips the
#      copy and bumps a counter ("mamba state-copy guard") instead of taking down
#      the CUDA context.
ARG MAMBA_UTILS=${SP}/vllm/v1/worker/mamba_utils.py
RUN cp ${MAMBA_UTILS} ${MAMBA_UTILS}.orig
COPY src/mamba_utils_guarded.py ${MAMBA_UTILS}
RUN python3 -c "import ast; ast.parse(open('${MAMBA_UTILS}').read()); print('mamba_utils.py guarded OK')"

# --- 8. Prefix-cache tracing (VLLM_HIT_DEBUG=1) -------------------------------------------
# Per-group hit breakdown, mamba boundary-state publication, cached-block eviction,
# prefill chunk stops. Costs nothing when off.
COPY src/patch_hit_debug.py /tmp/patch_hit_debug.py
RUN python3 /tmp/patch_hit_debug.py && rm /tmp/patch_hit_debug.py

# --- 9. Exact QSA top-k (VLLM_QSA_EXACT_TOPK=1) --------------------------------------------
# From upstream blazux/qwen3.8-Flash-DGX (8347e7c). The stock persistent_topk kernel is
# non-deterministic on GB10 and can drop real top-k candidates (vllm#51782; reported by
# @k3dani, blazux#3). The exact path uses torch.topk over the visible columns. Opt-in;
# superseded by 10 at kernel speed, kept as the fallback (wins over 10 when set).
COPY src/patch_qsa_exact_topk.py /tmp/patch_qsa_exact_topk.py
RUN python3 /tmp/patch_qsa_exact_topk.py ${QSA_OPS} && rm /tmp/patch_qsa_exact_topk.py

# --- 10. Deterministic persistent_topk kernel (VLLM_QSA_DET_TOPK=1) --------------------------
# Kernel-side fix for the non-deterministic / candidate-dropping QSA top-k (issue blazux#3,
# vllm#51782): @jschmied's deterministic persistent_topk, upstream as vllm#55122, built here
# as a standalone extension (_C_det.so) with the image's nvcc — no vLLM rebuild. Same
# determinism as patch 9's exact torch.topk, but at kernel speed: on the GX10 it recovers
# the whole prefill penalty (8k: 1,476 -> 2,488 tok/s; 32k: 1,794 -> 2,996; needle 92k:
# 69 s -> 46 s) with decode unchanged. Wiring and pins taken from blazux/qwen3.8-Flash-DGX
# (4b723de; pin bumped in 0022e36 by @jschmied: signed-zero canonicalisation, a
# deterministic low-shared-memory fallback and a launcher chunk-sizing fix for 24576/49152-
# wide rows, i.e. long contexts on the QSA path). Sources are fetched from
# https://github.com/jschmied/qwen38-flash-next-gb10 at a pinned commit (Apache-2.0;
# attribution: @jschmied). Set DET_ARCH=120a for an x86 Blackwell (RTX 5090).
# Inert unless VLLM_QSA_DET_TOPK=1; VLLM_QSA_EXACT_TOPK=1 still wins.
ARG KDET_SHA=e0ef69d4f5575dad00d34e05479eaf4c6547bace
ARG KDET=https://raw.githubusercontent.com/jschmied/qwen38-flash-next-gb10/${KDET_SHA}
ARG DET_ARCH=121a
# Pinned by commit AND sha256 (ADD --checksum needs BuildKit, the default since Docker 23).
ADD --checksum=sha256:138cacfc5eb117f0922d53c88727e4d0dc26dcfb246c3d401fc280cfc726cc71 ${KDET}/patches/kernel-det/build_det.py /opt/llm/kernel-det/src/build_det.py
ADD --checksum=sha256:b103fbeaf7589b9468471142ad0b30012a076f93d20ba11fc5ff6dcb1ecd32a6 ${KDET}/patches/kernel-det/bindings_det.cpp /opt/llm/kernel-det/src/bindings_det.cpp
ADD --checksum=sha256:19e1d53425ea9a839445722fd1dac1c41727128eebdce84508c1bfb8592afecf ${KDET}/patches/kernel-det/topk_det.cu /opt/llm/kernel-det/src/topk_det.cu
ADD --checksum=sha256:16939700ae389750782ff5c0d5b9caef59aa0ff8b869b64ec94fa72c814910ee ${KDET}/patches/kernel-det/torch_utils.h /opt/llm/kernel-det/src/torch_utils.h
ADD --checksum=sha256:b4ef9ce298d43d6c0e6db9fcca451df20815b2cfe33791919c1ad9c0e84f0ba7 ${KDET}/patches/kernel-det/persistent_topk.cuh /opt/llm/kernel-det/src/persistent_topk.cuh
ADD --checksum=sha256:70905073fe3fa361030bf1cb469b74610766bdfe361419cd7df50af2561322e3 ${KDET}/tools/determinism/qsadet_patch.py /tmp/qsadet_patch.py
RUN cd /opt/llm/kernel-det/src && DET_BUILD_DIR=/opt/llm/kernel-det/build DET_ARCH=${DET_ARCH} python3 build_det.py 2>&1 | tail -2 \
 && cp /opt/llm/kernel-det/build/_C_det.so /opt/llm/kernel-det/_C_det.so \
 && VLLM_QSA_PY=${QSA_OPS} python3 /tmp/qsadet_patch.py && rm /tmp/qsadet_patch.py \
 && python3 -c "import ast; ast.parse(open('${QSA_OPS}').read()); print('qsadet wired OK')"

# --- 11. On-demand step profiler (VLLM_STEP_PROFILE=1 + touch /tmp/profile_trigger) ---------
# torch.profiler around engine steps; this vLLM predates VLLM_TORCH_PROFILER_DIR.
COPY src/patch_step_profile.py /tmp/patch_step_profile.py
RUN python3 /tmp/patch_step_profile.py && rm /tmp/patch_step_profile.py

# --- 12. Per-step prefill metrics (--enable-logging-iteration-details) --------------------
# vllm:prompt_tokens_total is credited only when a prefill FINISHES; this adds
# vllm:scheduled_ctx_tokens_total (+ scheduled_iterations_total), fed every engine
# step from the iteration details, and mutes the stock one-line-per-step logger.
# Inert without the flag (ITER_DETAILS=1 in scripts/serve-intel-ar.sh); bench/ppwatch.sh.
COPY src/patch_prefill_metrics.py /tmp/patch_prefill_metrics.py
RUN python3 /tmp/patch_prefill_metrics.py && rm /tmp/patch_prefill_metrics.py
