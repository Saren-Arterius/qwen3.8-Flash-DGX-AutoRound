# AI Slop Warning
Although the working vLLM is in my GB10, this fork is prepared by Mr. Claude so it's very likely something will break or cannot reproduce, ~~especially the claimed prefix cache fix part.~~ should be really fixed.

# Qwen3.8-Flash-Next on a single DGX Spark (GB10)

Run **Qwen3.8-Flash-Next** — a ~176B-parameter model (125B main + 51B n-gram, 6B
active) — on **one NVIDIA DGX Spark / ASUS GX10** with **vLLM**, at full prefill
speed, with MTP speculative decoding, and up to **500k tokens of context**.

The catch this repo solves: the NVFP4 checkpoint is **122 GiB**, which does not fit
next to a usable KV cache in the Spark's **128 GB unified pool**. 44 GiB of that is
the n-gram embedding ("PLE") table — a pure lookup that a token only touches 16 rows
of. This repo adds one patch to the official vLLM image that **serves that table from
NVMe via `mmap`** instead of keeping it resident. Weights drop to **~76 GiB**, the
rest of the pool goes to KV, and everything runs on stock GB10 kernels.

> **Independently reproduced** on a DGX Spark (not a GX10) by
> [@jschmied](https://github.com/jschmied) — see
> [issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1) and their
> [write-up](https://github.com/jschmied/qwen38-flash-next-gb10), which also
> contributed the concurrency findings below.

The result, versus the llama.cpp GGUF that was the only working option on a Spark
before: **~4–5× faster prefill, MTP (which the GGUF cannot do), and 2× the context.**

> ## This fork: int4 + int8 + fp8 hybrid, ~49 tok/s
>
> This fork replaces the NVFP4 checkpoint with
> [Intel's W4A16 AutoRound int4](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound)
> plus an int8 GPTQ lm_head and blockwise-fp8 side layers (all prepared by the
> CPU-only tools in `tools/`), an **fp8** PLE table, a GB10 Triton-tile fix for
> the 36 GDN layers, a much faster PLE gather hot path, working **prefix
> caching**, and an optional **never-evict pin** that keeps your system
> prompt's KV resident. Measured on a DGX Spark: **~49 tok/s single-stream
> decode with MTP=2** (~2,000 tok/s prefill) — vs 25–28 tok/s for the NVFP4
> recipe below. The full recipe, per-patch docs, and checkpoint preparation:
> **[docs/OPTIMIZATIONS.md](docs/OPTIMIZATIONS.md)**; serve it with
> `scripts/serve-intel-ar.sh`. Upstream's NVFP4 path (`scripts/serve.sh`)
> still works unchanged.

| | llama.cpp IQ4_XS | **this repo (vLLM NVFP4)** |
|---|---|---|
| Prefill | ~540 tok/s | **~2,000–2,600 tok/s** (warm page cache) |
| Decode, single stream | ~22 tok/s (no MTP) | **25–28 tok/s** typical with MTP=2, up to ~36 on predictable text; ~17 without MTP |
| Context | 262k | **262k native, 500k with YaRN** (needle found at 414k) |
| Resident GPU memory | ~94 GiB (GGUF) | ~76 GiB weights + KV |
| Aggregate throughput | single stream only | scales with concurrency — see below |

*Measured on an ASUS GX10 (GB10, 128 GB). Prefill is the headline: Flash-Next's whole
point is its sparse attention (QSA), and llama.cpp has no QSA kernel — it runs dense,
so its prefill is its weakest axis. vLLM uses the real kernels.*

---

## Requirements

- An **NVIDIA DGX Spark or compatible GB10 (sm_121)** box, 128 GB unified memory,
  aarch64, recent NVIDIA driver, Docker with the NVIDIA container runtime.
- **~130 GB free disk** for the checkpoint, on reasonably fast storage (the table is
  read from it at runtime — NVMe strongly recommended; the Spark's onboard NVMe is ideal).
- The base image is multi-arch, so `docker build` also works on x86 Blackwell
  (sm_120, e.g. RTX PRO 6000) for testing, though this is tuned for the Spark.

## Quickstart

```bash
git clone https://github.com/blazux/qwen3.8-Flash-DGX.git
cd qwen3.8-Flash-DGX

docker build -t qwen38-flash-dgx .        # ~1 min: official image + one patch
scripts/download-weights.sh               # ~122 GiB, resumable (one-time)
scripts/serve.sh                          # boots on :18300 (~8 min to load)
docker logs -f qwen38-flash               # wait for "Application startup complete"
scripts/smoke-test.sh                     # health + coherence + prefill/decode numbers
```

Then hit the OpenAI-compatible API:

```bash
curl http://localhost:18300/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-flash-next",
  "messages": [{"role":"user","content":"Write a haiku about a desktop supercomputer."}],
  "max_tokens": 512
}'
```

500k context (YaRN, validated with a needle-in-a-haystack at 414k tokens):

```bash
YARN=1 CTX=500000 scripts/serve.sh
```

## Tuning (env vars for `scripts/serve.sh`)

| Var | Default | Notes |
|---|---|---|
| `PORT` | `18300` | API port |
| `CTX` | `262144` | Max context. Native is 262144; with `YARN=1` up to `500000` is validated. |
| `YARN` | `0` | `1` = YaRN rope scaling (factor 4, Qwen's recipe) for `CTX` > 262144. |
| `SEQS` | `8` | Max concurrent sequences. **Do not benchmark with 1–2**: excess requests queue silently and aggregate tok/s flatlines (see below). |
| `GPU_MEM` | `0.85` | Fraction of the 128 GB pool for weights+KV. `0.875` got OOM-killed on a 300k-token prefill with MTP — keep the margin. On a Spark the OS and the GPU share this pool, and an OOM there can freeze the box. |
| `MTP` | `2` | Speculative tokens from the model's MTP head (`0` = off). |
| `KV_DTYPE` | `auto` | Keep `auto` (bf16): `fp8` is refused — the QSA layers require a bf16 KV cache. |
| `PREWARM` | `0` | `1` streams the 48 GiB table once at boot to warm the page cache — steadier first-request latency, ~10 s extra startup. |
| `WORKERS` | `32` | Threads used for the mmap gather. |
| `EXTRA` | | Extra vLLM flags, passed verbatim. |

## Throughput and concurrency

Single-stream numbers understate this model on a GB10. @jschmied traced one box
under load (RadixArk NVFP4, 8k ctx, **no** speculative decoding, using vLLM's native
PLE CPU offload rather than this repo's mmap — the table-serving cost behaves the
same way) and found aggregate throughput scales far past single-stream:

| concurrent streams | aggregate tok/s | per stream | major faults / token | TTFT |
|---:|---:|---:|---:|---:|
| 1 | 17.1 | 17.1 | 16.0 | 0.22 s |
| 8 | 87.5 | 10.9 | 7.0 | 0.53 s |
| 16 | 131.6 | 8.2 | 9.6 | 0.83 s |
| 32 | 212.0 | 6.6 | 4.3 | 1.19 s |
| 48 | **266.8** | 5.6 | 3.6 | 1.60 s |

Two things worth knowing (their words, lightly condensed):

- **The paged table is an argument *for* concurrency, not against it.** Page-fault cost
  per token *falls* 4.4× from c=1 to c=48: batched tokens share n-gram rows and the
  page cache keeps the hot set, so the marginal token is far cheaper than the first.
  The table gather itself never exceeded ~25% of one CPU core.
- **A low `--max-num-seqs` is indistinguishable from saturation if you only look at
  tok/s.** With `--max-num-seqs 2` their sweep flatlined at ~33 tok/s while
  `vllm:request_queue_time_seconds_sum` climbed to 142 s. Check `max-num-seqs` before
  quoting an aggregate number — this repo's default is now `8` for that reason.

Method and harness: [load-and-waits.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/load-and-waits.md).

## How it fits — the one idea

A token's n-gram lookup reads **16 rows × 160 bytes ≈ 2.5 KB**. Over a 20k-token
prefill that's ~1.3 GB of small reads — under a second on NVMe, and the hot n-grams
stay in the page cache. So the 44 GiB table never needs to be in the unified pool:
we `mmap` the checkpoint's `model-plefp8-*.safetensors` shards and gather rows on
demand. Nothing else about the model changes — the hashing, dequant, and the sparse
attention all run stock.

Full details, including the GB10-specific bugs this works around and the long-context
findings, are in [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md).

### Alternative: vLLM's native PLE CPU offload

vLLM ships its own path (`VLLM_PLE_CPU_OFFLOAD=1`) that keeps the table in pinned host
RAM in a separate worker process. On a Spark that RAM is the same pool as the GPU, so
it saves less than the mmap — but @jschmied got it running and documented two things
you will need if you go that way (neither applies to the mmap patch, which is a single
process):

1. `_get_ple_embedding_quant_method()` in `ple_layer.py` only accepts `Fp8Config`;
   with the NVFP4 checkpoint the quant config is `modelopt_fp4`, so the FP8 PLE shards
   are rejected and loading dies on `ngram_embedding.weight_scale`. Accepting
   `modelopt`/`modelopt_fp4` there fixes it.
2. The worker hands CUDA tensors to the GPU process over IPC via `pidfd_getfd`, which
   `kernel.yama.ptrace_scope=1` (the Ubuntu/DGX OS default) forbids between sibling
   processes. In Docker: `--cap-add=SYS_PTRACE`. Under systemd:
   `AmbientCapabilities=CAP_SYS_PTRACE`. It fails ~10 minutes in, after all shards
   have loaded, with an unhelpful `Engine core initialization failed`.

Details: [results-radixark-vllm.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/results-radixark-vllm.md).

## What's in here

```
Dockerfile                 official vLLM Flash-Next image + the patches below
src/vllm_ple_mmap.py       mmap PLE table (any dtype, relocatable dir, fast gather)
src/vllm_fp8_hybrid.py     int4+fp8 hybrid dispatch on the GPTQ config (this fork)
src/patch_never_evict.py   never-evict system-prompt KV pinning (this fork)
src/test_ple_mmap_cpu.py   CPU unit test for the gather (no GPU needed)
src/test_never_evict_pin.py  CPU unit test for the pin (no GPU needed)
scripts/download-weights.sh
scripts/serve.sh           upstream NVFP4 serving
scripts/serve-intel-ar.sh  int4/int8/fp8 hybrid serving (this fork)
scripts/smoke-test.sh
tools/                     CPU-only checkpoint preparation (this fork)
docs/HOW-IT-WORKS.md       the original mmap-PLE story
docs/OPTIMIZATIONS.md      this fork's recipe, patch by patch
```

Run the unit test (no GPU):

```bash
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 \
  qwen38-flash-dgx test_ple_mmap_cpu.py
```

## Limitations & notes

- **One big model at a time.** At `GPU_MEM=0.85` this uses most of the 128 GB pool;
  don't co-locate another large model (an 8B embedding model next to it already
  starves the KV cache — we moved ours to another machine).
- **`--no-enable-prefix-caching` is required on the NVFP4 path** (a GB10 GDN kernel
  bug corrupts on the cached-block path) and **full `torch.compile` is off** (an
  Inductor int64-indexing assert on sm_121); the serve script sets both. On this
  fork's fp8-hybrid path the affected GEMM runs a different kernel and prefix
  caching works — `PREFIX_CACHE=1` in `scripts/serve-intel-ar.sh`.
- **1M context is out of reach on one box**: the QSA layers refuse an fp8 KV cache, and
  in bf16 a single 1M request needs ~30 GiB of KV. 500k with YaRN is the validated
  ceiling; 800k booted but got OOM-killed on a long prefill.
- Decode without MTP is a touch slower than the GGUF, because the gather does one
  host↔device sync per step; MTP more than makes up for it. Removing that sync
  (pinned staging buffer) is a natural next optimization — PRs welcome.
- **Weights are not included** and the checkpoint carries Qwen's license (with a
  MAU/revenue clause) — review it before production use.

## Credits

- Model: **Qwen team, Alibaba** — Qwen3.8-Flash-Next.
- NVFP4 checkpoint: **[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)**.
- int4 checkpoint this fork builds on: **[Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound)**
  (AutoRound); fp8 PLE table from **[Qwen/Qwen3.8-Flash-Next-FP8](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)**.
- Several pieces originate in the Qwen3.5-122B-A10B Spark recipes:
  the int4 AutoRound + int8 lm_head serving approach from
  **[albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4)**;
  the FLA shared-memory gate fix and the int4+fp8 hybrid idea from
  **[Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)**.
  The never-evict pin was built for that 122B stack on top of the ARC
  GPU-eviction work in [vllm#40270](https://github.com/vllm-project/vllm/pull/40270)
  and re-ported here.
- Serving engine and base image: **vLLM** (`vllm/vllm-openai:qwen38-flash-next`,
  the `release/qwen38next` recipe / PR #53896).
- Independent reproduction on a DGX Spark, the native-offload fixes and the
  concurrency measurements: **[@jschmied](https://github.com/jschmied)**
  ([issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1),
  [qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)).
- The mmap-PLE patch and the GB10 serving recipe in this repo: see [LICENSE](LICENSE) (Apache-2.0).
