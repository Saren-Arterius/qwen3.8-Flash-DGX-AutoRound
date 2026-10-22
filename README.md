<!-- SPDX-License-Identifier: Apache-2.0 -->

# magi-v3-public: Qwen3.8-Flash-Next serving on vLLM 0.32 + DFlash2 (local SSD)

Production serving stack for **Qwen3.8-Flash-Next** on a single NVIDIA DGX
Spark (GB10). vLLM **0.32.0** with native Qwen4Exp support, **DFlash2**
speculative decoding (3 draft tokens), and the PLE n-gram table served
from **local SSD via a graph-capturable mmap host func** — no wtako
dependency (RDMA remains available via PLE_RDMA). OpenAI-compatible
API on port 8000 under the model name `qwen`, with tool calling
(`qwen3_xml`) and reasoning (`qwen3`) parsers enabled.

This branch forked from the upstream v16b recipe (MTP + mmap table). The
upstream docs under `docs/` and `recipe/` describe that lineage and its
numbers; they do not apply to this stack.

## Current stack

| Piece | Value |
|---|---|
| Base image | `vllm/vllm-openai:v0.32.0` (pinned sha256, see `Dockerfile.v032`) |
| Built image | `qwen38-flash-dgx:magi-v3-public` |
| Target checkpoint | `Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid`, table-stripped GPTQ: INT4 routed experts, blockwise-FP8 side layers, INT8 `lm_head` |
| Drafter | `Qwen3.8-Flash-Next-DFlash2` (vLLM native `dflash`, EAGLE3-style) |
| Context / KV / streams | 262144 tokens, `KV_BYTES=20g`, `SEQS=16`, prefix cache + chunked prefill on |
| PLE table | 320001536 rows x 160 B FP8 from local SSD (VLLM_PLE_MMAP_DIR, default /ple-table-fp8), graph-capturable host-func memcpy (RDMA optional via PLE_RDMA) |
| CUDA graphs | `FULL_AND_PIECEWISE` (breakable, auto-on for Qwen4Exp) |
| GPU residency | ~89 GiB observed at boot |

## Measured speed

2026-10-19, single stream, 5 prompt types × 2 runs against the chat API:

| Task (completion tok/s) | Spec off | DFlash2 ×3 |
|---|---:|---:|
| Q&A (~195 tok) | ~30.8 | 57–59 |
| Code (188 tok) | ~30.7 | 67–68 |
| JSON (~848 tok) | ~31.0 | 69–71 |
| Math (9 tok) | ~25–30 | ~29–30 |
| LongCode (2048 tok) | ~30.9 | 70–71 |

2026-10-23, same 5 prompts via bench_qwen35.sh x3 against
magi-v3-public (local SSD, DFlash2 x3). First pass ignored (cold page
cache: Q&A 44.5, Code 58.5, JSON 63.9, LongCode 64.9); steady-state
passes 2–3, 4 runs each:

| Task (completion tok/s, public SSD) | DFlash2 x3 | RDMA 10-19 |
|---|---:|---:|
| Q&A (~195 tok) | 57.5-60.5 (avg ~58.6) | 57–59 |
| Code (188 tok) | 67.6-68.8 (avg ~68.3) | 67–68 |
| JSON (~848 tok) | 68.2-70.9 (avg ~69.6) | 69–71 |
| Math (9 tok) | ~28-30 (timing noise) | ~29–30 |
| LongCode (2048 tok) | 69.1-70.0 (avg ~69.5) | 70–71 |

Local SSD is consistent with RDMA within noise (~1 tok/s on LongCode).
Correctness: Reply with OK. gives OK; greedy outputs coherent (wrong PLE
rows would derail immediately).

FP8 KV investigation (2026-10-22, fixed 20g pool, dflash on): `fp8_e4m3`
gives 531k tokens vs 688k bf16 (0.77x) - it shrinks the pool despite
halving attention bytes. Verified causes: (1) the GDN recurrent state
lives in the pool and does not shrink (f32->bf16 SSM alone: 320k->515k);
(2) halving attn bytes doubles the block the mamba page forces
(832->1664, and 1600->3200 with f32 SSM); (3) the DFlash2 drafter
(sliding attention, no fp8 path on sm121) splinters into a 64-token
group drawing 4096 blocks/request from the shared pool. Turned off
(`SPEC=0`), fp8 reaches 1.49M tokens (5.70x) - the drafter, not fp8,
was the hog. Landed mitigations: default bf16 SSM;
`--kv-cache-dtype-skip-layers sliding_window` under fp8 (drafter stays
bf16 at 416-token blocks: 531k->1.10M); compressed-indexer fp8 patch
(`VLLM_INDEXER_FP8`, +3%, raw indexer stays bf16 for mRoPE positions).
Verdict: bf16 KV stays the default (quality-verified, benched);
fp8+skip is the documented 1M-context path (needs YaRN past 262k).

Available KV, all measured live (`/health` pool line, dflash x3 unless
noted, prefix cache on):

| Config @ pool bytes | Tokens | Max concurrency @262k |
|---|---|---:|
| bf16 + bf16 SSM @20g (current default) | 688,400 | 2.63x |
| bf16 + f32 SSM @28g (RDMA era) | 913,532 | 3.48x |
| fp8 + bf16 SSM + skip-sliding + indexer-fp8 @20g | 1,104,500 | 4.21x |
| fp8 + bf16 SSM, no skip @20g | 531,009 | 2.03x |
| fp8 + f32 SSM @20g | 319,655 | 1.22x |
| fp8, SPEC=0 (no drafter) @20g | 1,494,069 | 5.70x |

A single 1M-token request needs ~1M pool tokens: only the fp8 rows
qualify at 20g (bf16@20g holds 688k). What to run:

```bash
# 1M KV cache (1.10M tokens w/ dflash, 1.49M with SPEC=0):
KV_DTYPE=fp8_e4m3 KV_BYTES=20g bash serve-magi.sh
# 1M context (YaRN factor-4 rope override + allow-long flag):
YARN=1 CTX=1000000 bash serve-magi.sh
# both:
KV_DTYPE=fp8_e4m3 KV_BYTES=20g YARN=1 CTX=1000000 bash serve-magi.sh
```

dflash+YARN is wired but not yet booted; SPEC=0 is the safer 1M arm.

Two findings from the same session:

- **Full-graph vs piecewise graphs are identical within noise**, spec-off and
  spec-on. The graph-capturable RDMA lookup is kept because it is correct and
  costs nothing, not because it is faster — at batch 1 the bottleneck is MoE
  compute, not kernel launches.
- **Correctness over exact token-match.** Greedy (temp 0) outputs share
  100+ token identical prefixes with the Python-lookup baseline; the only
  divergences sit at exact 50/50 (`dans`/`au`, −0.693 nats each) sampling
  ties where any BF16 noise flips the pick. Wrong PLE rows would derail
  generation immediately, not produce coherent paraphrases.

## Requirements

- One GB10, ARM64 Linux, Docker + NVIDIA container runtime with unlimited memlock.
- Container pinned to the X925 perf cores by default (`CPUSET=` to unpin).
- Local FP8 table dir (default ~/ple-table-fp8, shards 5-37 of the official
  FP8 checkpoint: 128 shards, 320001536 rows). No /dev/infiniband needed.
- RDMA remains available: set PLE_RDMA to the wtako server (needs RoCE
  plus ple_rdma_server); the launcher waits for it.
- Local paths for the hybrid checkpoint and the DFlash2 drafter (defaults in
  `serve-magi.sh` point at shared storage).

## Quick start

```bash
git checkout magi-v3-public
# weights (~78 GB hybrid) + FP8 PLE table (~49 GB) into ~/models:
hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid \
  --revision 8b82f0b7abe3d1150a7827d298c75e86267636ae \
  --local-dir ~/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid
hf download Saren/Qwen3.8-Flash-Next-ple-table-fp8 \
  --revision 50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --local-dir ~/models/ple-table-fp8
# (or: bash recipe/tools/fetch-ple-table-fp8.sh ~/models/ple-table-fp8
#  for the table straight from the official Qwen FP8 checkpoint)
# No public standalone DFlash2 drafter exists: serve with SPEC=0 (or
# SPEC=mtp with an MTP-grafted checkpoint), or point DRAFT_DIR at
# your own drafter.
cd recipe/build/image
docker build -f Dockerfile.v032 -t qwen38-flash-dgx:magi-v5 .
cd ../../..
bash serve-magi.sh --print   # inspect the docker command
bash serve-magi.sh           # start qwen38-flash on :8000 (detached)

until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 10; done
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":512}'
```

Follow startup with `docker logs -f qwen38-flash`. Expect: PLE mmap graph log (/ple-table-fp8 layer 1, 128 shards,
320001536 rows, scale ~1.99e-04), DFlash capture, Graph capturing
finished (PIECEWISE 19/19 + FULL 9/9 + dflash2 9/9), then /health 200.

## Configuration

`serve-magi.sh` defaults; override via environment. `SPEC=mtp` keeps the old
MTP path, `SPEC=0` disables speculation.

| Variable | Default | Meaning |
|---|---|---|
| `SPEC` / `NSPEC` | `dflash` / `3` | Speculative method (`mtp`, `dflash`, `0`) and draft tokens |
| `MODEL_DIR` / `DRAFT_DIR` | shared-storage paths | Target checkpoint / DFlash2 drafter (drafter mounted at `/drafter` only for `dflash`) |
| `PLE_RDMA` | empty | Table server; empty (default) = local SSD mmap |
| `VLLM_PLE_RDMA_DEV` / `_GID` | `roceP2p1s0f0` / `auto` | RDMA device and GID index |
| `VLLM_PLE_RDMA_GRAPH` | `1` | `1`: graph-capturable host-func lookup; `0`: Python eager-break path |
| `CG_MODE` | `FULL_AND_PIECEWISE` | vLLM CUDA-graph mode |
| `KV_BYTES` / `SEQS` | `20g` / `16` | KV pool bytes / max sequences |
| `VLLM_FP8_HYBRID` | `1` | INT4 + blockwise-FP8 hybrid dispatch + QSA indexer fix |
| `PIN_PROMPT` | Magi AI prompt | Marker substring: a request containing it gets its prompt-prefix KV blocks pinned (never evicted, 25% pool cap) |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `1` | MoE GEMM backend workaround |
| `VLLM_USE_DEEP_GEMM` / `VLLM_USE_FLASHINFER_SAMPLER` | `0` / `1` | GEMM and sampler backend selection |
| `MODELS_ROOT` / `MODEL_DIR` / `DRAFT_DIR` / `TABLE_DIR` | `~/models` (+ names above) | Override any single one; the launcher fails fast if `dflash` has no drafter |
| `CPUSET` | `5-9,15-19` | Docker `--cpuset-cpus`: X925 perf cores; empty = all 20 cores |
| `MAMBA_SSM_DTYPE` | `bfloat16` | `--mamba-ssm-cache-dtype`: halves the in-pool GDN state; empty = vLLM default (float32) |
| `KV_DTYPE` | `auto` | `--kv-cache-dtype`: bf16 recommended; `fp8_e4m3` measured below (loses pool on this stack) |
| `VLLM_PLE_MMAP_WORKERS` / `_CHUNK` / `_PREWARM` | `32` / `2048` / `0` | mmap host-func threads / min rows per thread / page-cache prewarm at load |
| `EXTRA` | empty | Extra verbatim vLLM flags (e.g. `--block-size 1664`) |

Gotcha: `VLLM_PLE_RDMA_GRAPH=0` with `FULL_*` graphs **crashes graph
capture** (`Cannot copy between CPU and CUDA tensors … unless pinned`) — the
Python path forces piecewise, so pair it with `CG_MODE=PIECEWISE`.

## How it works

```text
INT4 experts + block-FP8 sides + INT8 head (table-stripped hybrid)
             │
             ├── PLE rows: local SSD mmap via graph host-func (RDMA optional)
             │     graph path: ids → pinned host buf → cuLaunchHostFunc memcpy
             │     → rows to GPU, all inside the graph; pthreads, MADV_RANDOM
             ├── DFlash2 drafter ×3, verified by the target (block rejection)
             └── FULL+PIECEWISE graphs + prefix cache + chunked prefill
                         ↓
                vLLM 0.32 verifier and sampler → response
```

- **RDMA lookup** (`src/vllm_ple_rdma_v032.py` + `src/ple_rdma/ple_rdma_graph.c`
  → `libple_rdma_graph.so`). Replaces both 0.32 PLE backends with a
  table-less embedding. Failed READs close the dead QP, redial every 2 s and
  retry; the engine stalls meanwhile, never silently wrong rows. Out-of-range
  ids (padding, warm-up garbage) become zero rows.
- **mmap lookup** (`src/vllm_ple_mmap_v032.py` + `src/ple_mmap_graph.c`
  -> `libple_mmap_graph.so`). Same host-func pattern as RDMA, `memcpy`
  from mmap'd safetensors shards instead of READs. Pthread fan-out
  (`VLLM_PLE_MMAP_WORKERS`, min rows `VLLM_PLE_MMAP_CHUNK`) saturates SSD
  queue depth on cold faults. Dormant when `VLLM_PLE_RDMA` is set.
  (A Python prefetch variant was tried and reverted: `.cpu()` cannot
  survive graph capture, even under `eager_break`.)
- **Indexer FP8 + drafter compat** (`src/vllm_indexer_fp8.py`, serve-side
  `VLLM_INDEXER_FP8` + `--kv-cache-dtype-skip-layers sliding_window`
  under fp8 KV). Compressed indexer cache in fp8; raw indexer stays
  bf16 (mRoPE positions); drafter kept bf16 (no fp8 sliding path on
  sm121) at 416-token blocks instead of stranded 64-token pages.
- **DFlash2 capture** (`src/vllm_qwen4exp_dflash_capture.py`). Ports the model
  card's SGLang patch: completed mid-model layer outputs averaged over the 4
  hyper-connection streams, with vLLM's delayed HC combine materialised for
  capture only. Plus the `dflash` allow-list for Qwen4Exp.
- **QSA indexer fix** (`src/vllm_qsa_indexer_fp8.py`). 0.32 fuses
  `indexer.index_qk_proj` into `qkv_proj` and rejects mixed-precision fused
  layers; the small BF16 indexer weight (640×2560 per QSA layer) is quantised
  to 128×128-block FP8 e4m3 at load. No checkpoint change.
- **Never-evict pin** (`src/patch_never_evict_v032.py`, build-time). Requests containing `PIN_PROMPT` get their prefix KV blocks held out of the free pool (hash-keyed, survives mamba/GDN mid-request frees). Verified live: 8 blocks / 25 MiB held across 6 of 7 KV groups.
- **Carried over**: FLA GB10 shmem/warps fixes, INT8 `lm_head` wiring,
  INT4+FP8 hybrid dispatch (`src/vllm_fp8_hybrid.py`).

## Repository layout

| Path | Contents |
|---|---|
| [`serve-magi.sh`](serve-magi.sh) | 0.32 launcher (this stack). `SPEC=mtp\|dflash\|0`, `--print` to preview |
| [`recipe/build/image/Dockerfile.v032`](recipe/build/image/Dockerfile.v032) | Image build: base pin, FLA fixes, hybrid dispatch, RDMA libs, capture + QSA patches, allow-list |
| [`recipe/build/image/src/ple_mmap_graph.c`](recipe/build/image/src/ple_mmap_graph.c) | mmap host-func (pthread fan-out, MADV_RANDOM) -> `libple_mmap_graph.so` |
| [`recipe/build/image/src/vllm_ple_mmap_v032.py`](recipe/build/image/src/vllm_ple_mmap_v032.py) | 0.32 mmap embedding (graph path) |
| [`recipe/build/image/src/vllm_indexer_fp8.py`](recipe/build/image/src/vllm_indexer_fp8.py) | Optional compressed-indexer fp8 (env-gated, off unless KV is fp8) |
| [`recipe/build/image/src/vllm_ple_rdma_v032.py`](recipe/build/image/src/vllm_ple_rdma_v032.py) | 0.32 RDMA embedding (graph + Python paths) |
| [`recipe/build/image/src/ple_rdma/`](recipe/build/image/src/ple_rdma/) | `ple_rdma.c` (shared READ helper) + `ple_rdma_graph.c` (host-func) |
| [`recipe/build/image/src/vllm_qwen4exp_dflash_capture.py`](recipe/build/image/src/vllm_qwen4exp_dflash_capture.py) | DFlash2 target-feature capture |
| [`recipe/build/image/src/vllm_qsa_indexer_fp8.py`](recipe/build/image/src/vllm_qsa_indexer_fp8.py) | QSA indexer load-time FP8 |
| [`recipe/`](recipe/) `docs/` `AGENTS.md` | Upstream v16b recipe material, retained for reference |

## Ops

Production is supervised from the sibling `magi-stack` repo:
`ram-client-ple.sh` waits for wtako `:18515`, then execs `serve-magi.sh`.
The 0.32 stack above was validated 2026-10-19 as a test container
(`qwen38-flash` on `:8000`, `RESTART=no`), then cut over as the supervised production launcher.
The bench/greedy test scripts used for the
numbers above live next to the supervisor logs, not in this repo.

## Upstream

Base recipe: [dime-online/qwen3.8-Flash-DGX-UltraFast](https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast)
(v16b, MTP + dense drafter + mmap table, GB10). Its benchmark tables,
`docs/BENCHMARKS.md`, `docs/EVALS.md` and `AGENTS.md` describe that stack,
not this branch. This branch's own deltas are exactly: the 0.32 port files
listed above, the RDMA-exclusive table, DFlash2 instead of MTP+T80, and
`KV_BYTES` 16g → 18g.

## Credits and licenses

Base recipe derives from
[Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound),
Apache-2.0, copyright blazux; via dime-online's UltraFast recipe. Thanks to
the Qwen model authors and to the vLLM, Intel AutoRound and FlashInfer
contributors. Model weights and downloaded datasets keep their own terms.

Everything in this repository — code, docs, data and images — is licensed
under **Apache-2.0**. If you redistribute it, keep the [NOTICE](NOTICE) file,
which credits **dime-online** and the upstream authors. See [LICENSE](LICENSE).

## Citation

GitHub's "Cite this repository" menu reads [CITATION.cff](CITATION.cff)
(which describes the upstream v16b recipe).

```bibtex
@software{dime_online_qwen38_gb10_v16b,
  title = {Qwen3.8 Flash · DGX UltraFast: v16b serving recipe},
  author = {{dime-online}},
  year = {2026},
  url = {https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast}
}
```
