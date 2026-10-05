# magi-v4: myllmbox/qwen38-flash-next-vllm:v5.2 (vLLM 0.30.0 + mbx patches) + two magi patches:
#   - never-evict prompt pin (v2/patch_never_evict_v030.py, anchors fit v5.2: vLLM 0.30.0)
#   - FP8 PLE rows over RDMA from wtako (MBX_PLE_RDMA_FP8=host:port, :18515) +
#     the sync RDMA client (vllm_ple_rdma_fp8.py) + shared verbs lib
# Build from the llm/ root:  docker build -f v4/Dockerfile -t qwen38-flash-dgx:magi-v4 .
FROM myllmbox/qwen38-flash-next-vllm:v5.2
ARG NG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ngram_embedding.py
ARG PKG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia
RUN apt-get update && apt-get install -y --no-install-recommends libibverbs-dev gcc && rm -rf /var/lib/apt/lists/*
COPY v4/ple_rdma.c /tmp/ple_rdma.c
RUN mkdir -p /opt/rb && gcc -O2 -Wall -shared -fPIC /tmp/ple_rdma.c -libverbs -o /opt/rb/libple_rdma.so && rm /tmp/ple_rdma.c
COPY v4/patch_never_evict_v030.py /tmp/patch_never_evict.py
RUN python3 /tmp/patch_never_evict.py && rm /tmp/patch_never_evict.py
COPY v4/vllm_ple_rdma_fp8.py ${PKG}/vllm_ple_rdma_fp8.py
COPY v4/patch_ple_fp8_rdma.py /tmp/patch_ple_fp8_rdma.py
RUN cp ${NG} ${NG}.orig && cat /tmp/patch_ple_fp8_rdma.py >> ${NG} && rm /tmp/patch_ple_fp8_rdma.py \
 && python3 -c "import ast; ast.parse(open(\"${NG}\").read()); ast.parse(open(\"${PKG}/vllm_ple_rdma_fp8.py\").read()); print(\"ple fp8-rdma patched OK\")"
