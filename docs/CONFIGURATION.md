<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Configuration reference

[`config/v16b/env`](../config/v16b/env) is sourced under a cleared environment by `launch.sh`. `serve.sh` then constructs Docker argv; `--print` emits one shell-quoted argument per line before any Docker command. The entries below are the **promoted values**. The published paths use `$HOME` and point at the prepared artifacts on the GB10.

| Setting | v16b value | Meaning / change boundary |
|---|---|---|
| `NAME` | `qwen38-flash` | Docker container name; change for side-by-side instances, with extra memory accounted for |
| `MODEL_DIR` | `$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-mtpdense-g32` | Host checkpoint; must be the T80 dense drafter variant for a v16b claim |
| `TABLE_DIR` | `$HOME/models/ple-table-fp8` | Host fp8 PLE table; local NVMe recommended |
| `PORT` | `8000` | Host HTTP port; any free host port, with firewall policy considered |
| `SERVED_NAME` | `qwen` | API model alias; change clients together with it |
| `TOOL_PARSER` | `qwen3_xml` | Tool-call parser matched to the source workload |
| `PREFIX_CACHE` | `1` | Prefix caching on; changing it affects warm TTFT and memory/cache behavior |
| `GPU_MEM` | `0.01` | Fraction used with explicit KV bytes |
| `KV_BYTES` | `16g` | Explicit KV pool |
| `SEQS` | `8` | Maximum simultaneous sequences |
| `MTP` | `3` | Speculative draft depth |
| `CTX` | `262144` | Configured model-token limit |
| `EXTRA` | `--enable-prompt-tokens-details` | Additional vLLM argument string |
| `LOAD_FORMAT` | `fastsafetensors` | Checkpoint loader; tested value only |
| `CC` | empty (script default) | Piecewise CUDA graphs and twelve splitting ops; PLE CPU gathering remains outside capture |
| `PREWARM` | `1` | Prewarm the PLE mmap path |
| `WORKERS` | `32` | PLE mmap worker count; changes affect CPU and memory pressure |
| `PLE_PREFETCH` | `0` | PLE prefetch mode; disabled in promoted config |
| `PLE_MADV_RANDOM` | `1` | Random-access mmap advice for the PLE table |
| `PLE_FAST_ROWS` | `16` | Fast gather row threshold |
| `PLE_CHUNK` | `8` | PLE gather chunk value |
| `PLE_STATS_SEC` | empty | Optional PLE diagnostic cadence; empty means vLLM default |
| `SPEC_EXTRA` | block rejection + probabilistic draft sampling | JSON object merged with MTP depth; requires `jq` |
| `HIT_DEBUG` | `0` | Prefix-cache diagnostics off; `1` enables logging |
| `STEP_PROFILE` | `0` | Step profiler off; `1` enables profiling overhead |
| `FP8_HYBRID` | `1` | Dispatch fp8 side layers from hybrid checkpoint |
| `CUDA_LAUNCH_BLOCKING` | `0` | Synchronous CUDA debugging when `1`; not a speed setting |
| `FLASHINFER_AUTOTUNE` | `0` | Keep autotuning off as measured |
| `PIN_PROMPT` | empty | Optional exact substring for KV prompt pinning; empty disables |
| `PIN_MAX_FRACTION` | `0.25` | Maximum pinned KV fraction when prompt pinning is used |

`serve.sh` also pins `PLE_FAST_PATH=1`, `PLE_FAST_MAX_ROWS=262144`, `DRAFTER_EXPERTS_FP8=1`, `LOW_LATENCY_GEMM=1`, `LLG_PDL=0`, `VERIFY_TOPK_TRITON=1`, and `KEEP_DRAFT_BLOCKS=1`. The low-latency GEMM and `LLG_PDL=0` are a measured pair. The launcher supplies the bundled 65,536-id `MTP_DRAFT_VOCAB` path. `DRAFTER_EXPERTS_FP8_PREFIX` is an optional filter and was unset. Keep these pins for the v16b identity; changing any makes a new variant.

The Docker argv fixes `--shm-size 16g`, `--ipc=host`, the container API port `8000`, `VLLM_PLE_MMAP=1`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`, `VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_FLASHINFER_SAMPLER=1`, `--enable-chunked-prefill`, `--max-num-batched-tokens 8192`, `--kv-cache-dtype auto`, tool auto-choice, Qwen reasoning parser and prefix caching. The `16g` shared-memory allotment is container shared memory, distinct from the `16g` KV pool. Keep KV bytes, `SEQS`, batched tokens and `GPU_MEM` at the promoted values for this recipe.
