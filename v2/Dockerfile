# magi-v2: bilikaz/qwen38-flash-next-recipe v4 image (vLLM 0.30.0 + mbx patches) + two magi patches:
#   - never-evict prompt pin (patch 6 of ../Dockerfile, one scheduler anchor ported to vLLM 0.30)
#   - PLE rows over RDMA from wtako (MBX_PLE_RDMA=host:port) + PLE read profiler (MBX_PLE_PROF=1)
# Build from the llm/ root:  docker build -f v2/Dockerfile -t qwen38-flash-dgx:magi-v2 .
FROM myllmbox/qwen38-flash-next-vllm:v4
ARG NG=/usr/local/lib/python3.12/dist-packages/vllm/models/qwen4_exp/nvidia/ngram_embedding.py
RUN apt-get update && apt-get install -y --no-install-recommends libibverbs-dev gcc && rm -rf /var/lib/apt/lists/*
COPY src/ple_rdma/ple_rdma.c /tmp/ple_rdma.c
RUN mkdir -p /opt/rb && gcc -O2 -Wall -shared -fPIC /tmp/ple_rdma.c -libverbs -o /opt/rb/libple_rdma.so && rm /tmp/ple_rdma.c
COPY v2/patch_never_evict_v030.py /tmp/patch_never_evict.py
RUN python3 /tmp/patch_never_evict.py && rm /tmp/patch_never_evict.py
COPY v2/patch_ple_prof_rdma.py /tmp/patch_ple_prof_rdma.py
RUN cp ${NG} ${NG}.orig && cat /tmp/patch_ple_prof_rdma.py >> ${NG} && rm /tmp/patch_ple_prof_rdma.py \
 && python3 -c "import ast; ast.parse(open(\"${NG}\").read()); print(\"ple prof/rdma patched OK\")"
