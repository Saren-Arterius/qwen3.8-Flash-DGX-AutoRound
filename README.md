# AI Slop Warning
Although the working vLLM is in my GB10, this fork is prepared by Mr. Claude so it's very likely something will break or cannot reproduce, ~~especially the claimed prefix cache fix part.~~ should be really fixed.

# Qwen3.8-Flash-Next on a single DGX Spark (GB10) — int4 + int8 + fp8 hybrid

Run **Qwen3.8-Flash-Next** — a ~176B-parameter model (125B main + 51B n-gram, 6B
active) — on **one NVIDIA DGX Spark / ASUS GX10** with **vLLM**: **~49 tok/s
single-stream decode with MTP=3** (~2,000 tok/s prefill), working **prefix
caching**, and a **never-evict pin** that keeps your system prompt's KV resident
through arbitrary traffic.

Forked from **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)**,
which established the foundation this fork stands on: the ~49 GiB fp8 n-gram ("PLE")
table is a pure lookup that a token only touches 16 rows of, so it is served
**from NVMe via `mmap`** instead of living in the 128 GB unified pool
(full story: [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md)). Upstream serves the
official **NVFP4** checkpoint at 25–28 tok/s; for that path and its tuning
guide, use upstream. This fork replaces the checkpoint with
**[Intel's W4A16 AutoRound int4](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound)**
plus an int8 GPTQ lm_head and blockwise-fp8 side layers (all prepared by
CPU-only tools in `tools/`), an **fp8** PLE table, and a set of GB10/vLLM
patches — roughly **1.8× faster decode** than the NVFP4 recipe on the same box.

> **Upstream's NVFP4 recipe independently reproduced** on a DGX Spark by
> [@jschmied](https://github.com/jschmied) — see
> [blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1) and their
> [write-up](https://github.com/jschmied/qwen38-flash-next-gb10), which also
> contributed the concurrency findings below.

| | llama.cpp IQ4_XS | upstream (vLLM NVFP4) | **this fork (int4/int8/fp8)** |
|---|---|---|---|
| Prefill | ~540 tok/s | ~2,000–2,600 tok/s | **~2,000 tok/s** |
| Decode, single stream | ~22 tok/s (no MTP) | 25–28 tok/s (MTP=2) | **~49 tok/s (MTP=3)** |
| Prefix caching | — | off (GDN kernel bug) | **on** (+ never-evict pin) |
| Context | 262k | 262k native / 500k YaRN | 262k native / 500k YaRN |
| Weights resident | ~94 GiB (GGUF) | ~76 GiB | **~71 GiB** |

## Throughput and concurrency

Measured on this stack (DGX Spark, MTP=3 speculative decoding, prefix caching
on, `SEQS=8`). Single-stream decode by workload — reproduce with
[bench_qwen35.sh](https://github.com/albond/DGX_Spark_Qwen3.5-122B-A10B-AR-INT4/blob/master/bench_qwen35.sh)
(from albond's 122B recipe) pointed at your endpoint; two runs, best of:

| workload | tok/s |
|---|---:|
| Q&A | 46.1 |
| Code | 49.1 |
| JSON | **58.1** |
| Math | 48.8 |
| LongCode (2048 tok) | 47.2 |

Structured output decodes fastest — MTP draft acceptance is highest on
predictable text. Under concurrency and context depth — reproduce with
[tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)
(`pp2048 tg128`; `d` = tokens already in context, `c` = concurrent streams):

| depth | c | pp tok/s | tg tok/s | TTFT (ms) | total (ms) |
|---|---|---:|---:|---:|---:|
| d0 | 1 | 2,104 | 45.4 | 1,151 | 3,820 |
| d0 | 2 | 1,519 | 50.3 | 2,668 | 7,038 |
| d0 | 4 | 1,723 | 81.0 | 4,870 | 10,087 |
| d4096 | 1 | 1,805 | 39.9 | 3,629 | 6,682 |
| d4096 | 2 | 1,373 | 52.2 | 8,958 | 13,359 |
| d4096 | 4 | 1,248 | 44.6 | 18,177 | 24,474 |
| d8192 | 1 | 1,606 | 36.7 | 7,180 | 10,513 |
| d8192 | 2 | 1,416 | 39.5 | 13,731 | 18,797 |
| d8192 | 4 | 1,174 | 23.4 | 29,745 | 37,899 |

The TTFT growth with concurrency is MTP's prefill cost (see the next
section), not the paged table. On upstream's NVFP4 path
[@jschmied](https://github.com/jschmied) measured aggregate throughput
scaling to ~267 tok/s at 48 streams
([load-and-waits.md](https://github.com/jschmied/qwen38-flash-next-gb10/blob/main/notes/load-and-waits.md));
two portable takeaways: per-token page-fault cost *falls* with concurrency
(batched tokens share n-gram rows), and a low `--max-num-seqs` silently
queues requests — check `vllm:request_queue_time_seconds_sum` before quoting
an aggregate number.

## Requirements

- An **NVIDIA DGX Spark or compatible GB10 (sm_121)** box, 128 GB unified memory,
  aarch64, recent NVIDIA driver, Docker with the NVIDIA container runtime.
- **~130 GB free disk** for the checkpoint + fp8 PLE table, on reasonably fast
  storage (the table is read at runtime — NVMe strongly recommended).
- The base image is multi-arch, so `docker build` also works on x86 Blackwell
  (sm_120) for testing, though this is tuned for the Spark.

## Quickstart

```bash
git clone https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound.git
cd qwen3.8-Flash-DGX-AutoRound

docker build -t qwen38-flash-dgx .   # official image + this fork's patches

# The prepared checkpoint + PLE table (one-time, ~122 GiB):
hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid --local-dir /models/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound
hf download Saren/Qwen3.8-Flash-Next-ple-table-fp8 --local-dir /models/ple-table-fp8
# (or build them yourself from Intel's release: ./prepare.sh — see below)

# Point serve.sh at your checkpoint + table dirs, then:
./serve.sh                           # boots on :8000 (~5 min with fastsafetensors)
docker logs -f qwen38-flash          # wait for "Application startup complete"
```

Then hit the OpenAI-compatible API:

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen",
  "messages": [{"role":"user","content":"Write a haiku about a desktop supercomputer."}],
  "max_tokens": 512
}'
```

`serve.sh` is a thin example config over `scripts/serve-intel-ar.sh` — every
knob is an env var. The defaults below are `serve.sh`'s (the recommended
entry point); where the bare `scripts/serve-intel-ar.sh` falls back to
something else, the note says so. Keep your machine's real settings as an
edited copy or a local-only commit on top.

## Modify the weights yourself

The quickstart's two `hf download` repos are the finished artifacts — hashes
verified against the local originals. If you'd rather build (or audit) them
yourself from
[Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound),
one script runs the whole pipeline (CPU-only — a NAS box is fine):

```bash
./prepare.sh /models/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound /models/ple-table-fp8
```

Each step is explained in `prepare.sh`'s header comments: int8 lm_head repack,
fp8 side-layer conversion, n-gram index strip, fp8 table fetch, and the
`quantization_config` rewrite. On that last one: this vLLM build has no
auto-round loader, but its GPTQ config (`AutoGPTQConfig` → Marlin kernels)
reads the same packed tensors — the GPTQModel-style `dynamic` rules exclude
the families that are not int4-packed and flip the head to 8-bit. The original
AutoRound config is kept as `config.json.autoround`.

## Serving

```bash
docker build -t qwen38-flash-dgx .
MODEL_DIR=/models/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound \
TABLE_DIR=/models/ple-table-fp8 \
PREFIX_CACHE=1 PIN_PROMPT="You are HomeBot, the household assistant." \
scripts/serve-intel-ar.sh
```

or edit the paths in `serve.sh` (the example config used above) and run it.

| Var | `serve.sh` default | Notes |
|---|---|---|
| `MODEL_DIR` / `TABLE_DIR` | `/path/to/...` — edit these | Prepared checkpoint / fp8 PLE table dirs. On this branch `TABLE_DIR=""` + `PLE_RDMA=<ip>:<port>` skips the mmap dir entirely and serves rows over RDMA (see the RDMA section) |
| `PORT` | `8000` | API port (bare script: `18300`) |
| `CTX` | `262144` | Max context |
| `SEQS` | `8` | Max concurrent sequences (don't benchmark with 1–2, see below) |
| `GPU_MEM` | `0.01` | Near-zero pool fraction, paired with `KV_BYTES`: deterministic sizing, so the driver never oversubscribes the unified pool (`NV_ERR_NO_MEMORY` / Xid 31 freezes). Bare script: a `0.85` fraction — avoid on unified-memory boxes. |
| `KV_BYTES` | `20g` | Explicit KV pool size, passed as `--kv-cache-memory-bytes` (bare script: unset) |
| `MTP` | `3` | Speculative tokens from the MTP head (`0` = off; bare script: `2`) |
| `PREFIX_CACHE` | `1` | Prefix caching — fixed and recommended on this fork (bare script: `0`) |
| `PIN_PROMPT` / `PIN_MAX_FRACTION` | unset / `0.25` | Never-evict pin (patch 6); needs `PREFIX_CACHE=1` |
| `FP8_HYBRID` | `1` | int4+fp8 hybrid dispatch (patch 4) |
| `PLE_MADV_RANDOM` | `0` | `MADV_RANDOM` on the table mmap (patch 1) |
| `HIT_DEBUG` | `0` | Prefix-cache tracing (patch 8) |
| `PREWARM` | `1` | Stream the table once at boot to warm the page cache |
| `WORKERS` | `32` | Threads for the mmap gather |
| `LOAD_FORMAT` | `fastsafetensors` | Noticeably faster cold boots |
| `TOOL_PARSER` | `qwen3_xml` | Tool-call parser (bare script: `qwen3_coder`) |
| `SERVED_NAME` | `qwen` | Model id on the API (bare script: `qwen3.8-flash-next`) |
| `EXTRA` | | Extra vLLM flags, passed verbatim |

## Limitations & notes

- **One big model at a time** — and on a Spark the OS and GPU share the pool;
  prefer the deterministic `GPU_MEM=0.01` + `KV_BYTES` sizing over a large
  fraction (an OOM inside the unified pool can freeze the box).
- **1M context is out of reach on one box**: the QSA layers refuse an fp8 KV
  cache, and in bf16 a single 1M request needs ~30 GiB of KV. 500k with YaRN
  was upstream's validated ceiling.
- **Weights are not included** and the checkpoint carries Qwen's license (with
  a MAU/revenue clause) — review it before production use.

## What runs in what precision

| Component | Precision | How |
|---|---|---|
| 512-expert MoE (48 layers + MTP layer's own 512) | **int4** GPTQ-Marlin g128 | Intel checkpoint as-is |
| lm_head (shared with MTP draft head) | **int8** GPTQ-Marlin (uint8b128) | `tools/quantize_lm_head_int8.py` + `"lm_head": true` |
| GDN in/out projections, QSA q/k/v/o, shared expert | **fp8** blockwise e4m3 (128×128) | `tools/fp8_convert.py` + `src/vllm_fp8_hybrid.py` |
| Embeddings, hyper-connections, norms, MoE gates, fc_hidden | bf16 | excluded via `dynamic` rules |
| PLE n-gram table (51B params, layer 1) | **fp8** rows, mmapped from disk | `tools/fetch-ple-table-fp8.sh` + the mmap patch |
| KV cache | bf16 | QSA refuses fp8 KV |

## The patches

Everything is applied at image build time (see the `Dockerfile`); each patch is
independent and gated by an env var where it changes behavior.

### 1. PLE mmap upgrades (`src/vllm_ple_mmap.py`, extends upstream's patch)

> On this branch the mmap gather is the *generic* path. Production here
> serves the rows over one-sided RDMA instead (`src/vllm_ple_rdma.py`,
> `TABLE_DIR=""` + `PLE_RDMA` — see "PLE table over RDMA" below); the
> mmap dir remains the fallback and backs `PLE_RDMA_VERIFY=1`.

- **Any table dtype**: bf16/f16 tables and fp8 (with `weight_scale`) are all
  accepted; row size is derived from the safetensors headers. The fp8 table
  halves the bytes read per token vs bf16.
- **`VLLM_PLE_MMAP_DIR`**: the table no longer has to live inside the
  checkpoint dir — point it at any directory of safetensors shards (NFS, local
  NVMe, a RAM-backed device...). The backend matters: the ~49 GiB table
  outgrows what the page cache can keep warm next to the model, so gathers
  cost ~1.3 ms/op from a RAM-backed source vs 5–9 ms from local NVMe vs
  30–50 ms over NFS — decode impact in the main branch's README appendix.
- **Hot path**: per-step dedup of row ids on CPU (`np.unique`), gather of
  unique rows only, staging through a persistent pinned buffer with an async
  H2D copy, and GPU-side expansion via the inverse index. A decode fast path
  (`VLLM_PLE_MMAP_FAST_ROWS`, default 512) skips the thread pool entirely for
  small gathers. Net effect: ~7.2 → ~2.5–3.8 ms per lookup op on a RAM-backed table.
- **`VLLM_PLE_MMAP_MADV_RANDOM=1`**: `madvise(MADV_RANDOM)` the mmap so faults
  stay single-page — for tables on remote RAM or boxes with no page-cache
  headroom.
- **Stats**: `VLLM_PLE_MMAP_STATS_SEC` (default 30) logs
  `PLE mmap stats (last Ns): calls, op ms, gather ms, rows, MB` and resets the
  counters each period.

### 2. FLA shared-memory gate (`Dockerfile` sed)

sm_121 reports 99 KiB of shared memory per block — the same as ADA, where the
flash-linear-attention Triton kernels use their big GDN tiles — but the gate in
`vllm/third_party/flash_linear_attention/ops/utils.py` demands 100 KiB, so all
36 GDN layers silently fell back to small tiles. Lowering the gate to 99 KiB
(101376) lets GB10 take the big-tile path. (Found the hard way in the
Qwen3.5-122B Spark recipe — ported from
[Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)'s
`patch_fla_shmem.py`.)

### 3. int8 lm_head enablement (`Dockerfile` sed on `model.py` / `mtp.py`)

Upstream constructs `ParallelLMHead` without `quant_config`, forcing a bf16
head (1.27 GiB, and a bf16 GEMV per token over a 248320 vocab). One added
kwarg in both the main model and the MTP draft lets the head pick up the
checkpoint's int8 GPTQ packing. Without the `mtp.py` half, MTP ≥ 3 crashes at
load ("no module or parameter named 'lm_head.qweight'").

### 4. int4+fp8 hybrid dispatch (`src/vllm_fp8_hybrid.py`, `VLLM_FP8_HYBRID=1`)

vLLM's GPTQ config quantizes listed layers and leaves the rest to
`UnquantizedLinearMethod` — it has no notion of "this excluded layer is
actually fp8 in the checkpoint". This shim wraps `AutoGPTQConfig`: it scans the
checkpoint metadata for `F8_E4M3` weights with a `weight_scale_inv` sibling and
routes exactly those layers to vLLM's blockwise-`Fp8Config`
(`weight_block_size=[128,128]`, dynamic activation scheme) while everything
else keeps the GPTQ path. `VLLM_USE_DEEP_GEMM=0` is required on sm_121
(DeepGEMM hits `CUDA_ERROR_LAUNCH_FAILED`); the triton fallback is fine.

### 5. Prefix caching: on, and fixed (`src/patch_mamba_align_split.py`)

Upstream runs `--no-enable-prefix-caching` because of a CUBLAS error in the
GDN `in_proj` GEMM on the cached-block path. With the fp8 side layers that
GEMM runs a different kernel, and prefix caching is stable in our serving
(`PREFIX_CACHE=1`).

It also *works properly* now. On this hybrid model the reconciled cache hit is
the **minimum across all KV cache groups** (full attention + four mamba/GDN
state groups), and mamba "align" mode can only cache a state at a prefill
chunk end on a 1600-token boundary. The image's scheduler aligned those chunk
ends to `cache_config.block_size` — which the engine rewrites to the *minimum*
group block size (8, the MTP draft granularity) — so chunks ended where no
mamba state was cacheable, a cold request published **zero** reusable mamba
states, and a repeated prompt only got fast on the **3rd** try (the miss
triggers junction machinery that rebuilds the boundary one request late).
Long prompts effectively never hit. The patch makes the split use
`cache_config.mamba_block_size` (1600). Verified: an 8k-token repeat goes
10.1 s → **0.90 s on the 2nd request**.

The same rewritten `block_size` also poisoned the **worker** side: the
align-mode state-slot seed (`mamba_hybrid.py`) divided by it too, so a prefix
hit at 6400 tokens seeded state column 799 instead of 3, read past the
block-table row and restored a wrong (often all-zero / stale) mamba state —
greedy outputs visibly changed on cache hits. Root-caused upstream by
[blazux](https://github.com/blazux/qwen3.8-Flash-DGX/issues/2#issuecomment-546252046)
(his fix: `8347e7c`); the same one-line seed fix is folded into this patch.
Verified: cold-vs-hit first-token logprobs now agree within the stack's
normal run-to-run jitter (Marlin atomic-add nondeterminism), where before the
fix greedy outputs diverged within the first few tokens.

Notes: the prefix-cache granularity is large (1600 tokens; shorter prefixes
get no reuse), and a repeat hit tops out at `round_down(P,1600) − 1600` — MTP
(eagle-style) always recomputes the last matched block.

### 6. Never-evict prompt pinning (`src/patch_never_evict.py`)

`--never-evict-kv-cache-prompt-includes "<substring of your system prompt>"`
pins the KV blocks of any prompt containing that marker: they are held in a
side queue on `BlockPool`, excluded from the free count, and thus never handed
out for eviction — your assistant's system prompt stays cached no matter what
other traffic does. `--never-evict-kv-cache-max-fraction` (default 0.25) caps
the pin. The pin set is *replaced* on each matching request, so an updated
system prompt releases the old blocks automatically.

Verified end-to-end: a pinned 8k prompt still answers in **0.94 s after 2M
tokens of unique traffic** (3.1× full KV-pool turnover).

Implementation notes: the marker is tokenized once and matched as a token-id
subsequence (first/last token dropped — BPE merges at the boundaries); the pin
is keyed on block *hashes*, not block ids, because this hybrid model frees
mamba/GDN state blocks mid-request — each freed block is re-claimed by the pin
the moment `free_blocks()` sees it. Only prefix-cacheable KV groups
participate (the MTP draft layer's group is not, and must be skipped). This is
a pin-only port of our `arc_pin2` patch from the Qwen3.5-122B Spark stack,
which built the pin on top of the ARC GPU-eviction work in
[vllm#40270](https://github.com/vllm-project/vllm/pull/40270); the ARC/2Q
policies themselves were deliberately dropped — the stock free queue is
C-speed on a path this model hits every step.

Self-check (no GPU): `docker run --rm -v "$PWD/src:/t" --entrypoint python3
qwen38-flash-dgx /t/test_never_evict_pin.py`.

### 7. Mamba state-copy guard (`src/mamba_utils_guarded.py`)

Hardens the align-mode state-copy kernels against the "CUDA illegal memory
access / Xid 31 under load" crash class (also
[blazux#2](https://github.com/blazux/qwen3.8-Flash-DGX/issues/2)): backports
[vllm#50729](https://github.com/vllm-project/vllm/pull/50729) (overlapping
state-copy race) and bounds-checks every block id against its state pool
before dereferencing — an out-of-range id skips the copy and bumps a counter
(logged as `mamba state-copy guard`) instead of taking down the CUDA context.

With the block-size seed fix (patch 5) the out-of-range ids the guard was
absorbing are gone at the root: the counter is expected to stay at **0**, and
a nonzero count is logged as an *error* — it now indicates a new bug worth
reporting, not a known quirk.

### 8. Prefix-cache tracing (`src/patch_hit_debug.py`, `HIT_DEBUG=1`)

Set `HIT_DEBUG=1` (→ `VLLM_HIT_DEBUG=1` in the container) to log, per request:
the per-KV-group hit reconciliation (which group truncated the hit), mamba
boundary-state publication (which slots were real/null/hashed), cached-block
evictions, and prefill chunk-stop decisions. This is what found the bug in
patch 5; costs nothing when off.

## Speculative decoding and TTFT

MTP raises decode substantially but puts a floor (~0.8 s) under
time-to-first-token: vLLM's v1 engine only emits the first token after the
drafter has run, and MTP's draft layer is a stateful autoregressive
transformer — on every prefill chunk it must run a full-chunk-width forward
(always eager: above the cudagraph capture sizes) to sync its own KV/GDN
state, plus k−1 sequential single-token passes. Cross-attention drafters like
DFlash don't pay this, but Flash-Next has no such drafter — MTP is what ships
in the checkpoint. If your workload is TTFT-sensitive, weigh `MTP` depth
against `MTP=0`; only 0 removes the floor.

## What's in here

```
Dockerfile                    official vLLM Flash-Next image + the patches above
serve.sh                      example launcher config (edit paths, run)
prepare.sh                    build the checkpoint + table from Intel's release
src/vllm_ple_mmap.py          mmap PLE table (any dtype, relocatable dir, fast gather)
src/vllm_ple_rdma.py          PLE rows via one-sided RDMA READ (this branch's prod path)
src/ple_rdma/                 the RDMA daemon, C helper (libple_rdma.so) + test client
src/vllm_fp8_hybrid.py        int4+fp8 hybrid dispatch on the GPTQ config
src/patch_never_evict.py      never-evict system-prompt KV pinning
src/patch_mamba_align_split.py  prefix-cache chunk-alignment fix
src/patch_hit_debug.py        prefix-cache tracing (VLLM_HIT_DEBUG)
src/mamba_utils_guarded.py    hardened align-mode state copy (vllm#50729 + guard)
src/test_ple_mmap_cpu.py      CPU unit test for the gather (no GPU needed)
src/test_never_evict_pin.py   CPU unit test for the pin (no GPU needed)
scripts/serve-intel-ar.sh     the docker run behind serve.sh
scripts/smoke-test.sh         health + coherence + prefill/decode numbers
bench/decode_bench.py         batch-1 decode / TTFT / spec-acceptance bench
bench/RESULTS.md              step-time profile + knob experiments
tools/                        CPU-only checkpoint preparation
docs/HOW-IT-WORKS.md          upstream's mmap-PLE story
docs/OPTIMIZATIONS.md         this fork's patches in depth
```

## Credits

- Model: **Qwen team, Alibaba** — Qwen3.8-Flash-Next.
- **This is a fork of [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)** —
  the original mmap-PLE idea, the GB10 serving recipe, the NVFP4 path, and
  docs/HOW-IT-WORKS.md are theirs.
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
- Independent reproduction of the upstream recipe, the native-offload fixes and
  the concurrency measurements: **[@jschmied](https://github.com/jschmied)**
  ([blazux#1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1),
  [qwen38-flash-next-gb10](https://github.com/jschmied/qwen38-flash-next-gb10)).
- License: [Apache-2.0](LICENSE).


## PLE table over RDMA (only if you're crazy enough)

This branch's production setup does not mmap the table from storage at all:
a ~200-line daemon on the NAS pins all 33 fp8 shards (~49 GiB) into RAM as
**one contiguous ibverbs memory region**, hands each client
`(gid, qpn, rkey, addr, geometry)` over a tiny TCP handshake, then idles —
every gather is a batch of **one-sided RDMA READs** of 160-byte rows,
initiated by the Spark, zero server CPU per request. Code: `src/ple_rdma/`
(`ple_rdma.c` + ctypes wrapper, shared by both ends); vLLM side:
`src/vllm_ple_rdma.py`.

Why bother: gather latency drops to **~1.3 ms/op** vs 5–9 ms from local NVMe
and 30–50 ms from NFS — ~42 vs ~36 tok/s single-stream decode (see main's
README appendix). So: **~+15% decode**, 49 GiB of local NVMe and its page-cache
pressure back, and the warm glow of having built it. That's the whole payoff;
decide accordingly.

You need: **RoCE v2** between the boxes (this setup: a 100 Gb link,
Spark ↔ NAS), a NAS with **≥64 GB RAM** (49 GiB gets pinned, `memlock`
unlimited), `libibverbs` on both ends, `gcc` + python3-numpy on the NAS.

**Server (NAS)** — copy `src/ple_rdma/` over, then:

```bash
gcc -O2 -Wall -shared -fPIC ple_rdma.c -libverbs -o libple_rdma.so
./ple_rdma_server.py /path/to/ple-table-fp8 --dev <ibdev> --gid <rocev2-gid> --port 18515
```

(`ibv_devinfo` for the device name; `show_gids`-style tooling or
`/sys/class/infiniband/*/ports/1/gid_attrs/types/*` for the RoCE v2 GID
index. `--shards 0-3` serves a subset for smoke tests.)

As a systemd unit:

```ini
[Unit]
Description=PLE table RDMA server (one-sided READ target)
After=network-online.target remote-fs.target
Wants=network-online.target

[Service]
Type=simple
User=you
LimitMEMLOCK=infinity
ExecStart=/usr/bin/python3 /opt/ple_rdma/ple_rdma_server.py /path/to/ple-table-fp8
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Client (Spark)** — serve with an empty `TABLE_DIR` and:

| Var | Default | Notes |
|---|---|---|
| `PLE_RDMA` | unset | `<nas-ip>:18515` — enables the RDMA path |
| `PLE_RDMA_DEV` | `roceP2p1s0f0` | Spark-side ibverbs device |
| `PLE_RDMA_GID` | `auto` | auto-detects the RoCE v2 GID index |
| `PLE_RDMA_PREFETCH` | `1` | batch-assembly prefetch; measured neutral here |
| `PLE_RDMA_VERIFY` | `0` | `1` = cross-check every gather against a local mmap table (`TABLE_DIR` set too); soak passed with zero mismatches before the cutover |

**Sanity first**: `ple_test_client.py <server> <table-dir-over-nfs>` reads
random rows via RDMA and byte-compares them against the safetensors source.

Fine print: both ends must serve the *same table build* (wire order is sorted
shard filenames); the server has **no auth and no encryption** — trusted LAN
only; there is no reconnect logic — if the server bounces, restart the vLLM
container; the table is gone from RAM on every NAS reboot (the daemon reloads
it in under a minute).
