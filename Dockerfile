# Qwen3.8-Flash-Next on a single DGX Spark / GB10, via vLLM.
#
# Starts from the official Qwen3.8-Flash-Next vLLM image and appends one patch:
# it serves the 51B-parameter n-gram ("PLE") table from disk via mmap instead of
# keeping it resident in the 128 GB unified pool. That is the single change that
# lets the ~176B (122 GiB NVFP4) checkpoint fit next to a real KV cache on one box.
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
