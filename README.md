<!-- SPDX-License-Identifier: Apache-2.0 -->

# magi-v3: Qwen3.8-Flash-Next serving on vLLM 0.32 + DFlash2

Production serving stack for **Qwen3.8-Flash-Next** on a single NVIDIA DGX
Spark (GB10). vLLM **0.32.0** with native Qwen4Exp support, **DFlash2**
speculative decoding (3 draft tokens), and the PLE n-gram table served
**exclusively over RDMA** from wtako — no local table copy. OpenAI-compatible
API on port 8000 under the model name `qwen`, with tool calling
(`qwen3_xml`) and reasoning (`qwen3`) parsers enabled.

This branch forked from the upstream v16b recipe (MTP + mmap table). The
upstream docs under `docs/` and `recipe/` describe that lineage and its
numbers; they do not apply to this stack.

## Current stack

| Piece | Value |
|---|---|
| Base image | `vllm/vllm-openai:v0.32.0` (pinned sha256, see `Dockerfile.v032`) |
| Built image | `qwen38-flash-dgx:magi-v3-v032` |
| Target checkpoint | `Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid`, table-stripped GPTQ: INT4 routed experts, blockwise-FP8 side layers, INT8 `lm_head` |
| Drafter | `Qwen3.8-Flash-Next-DFlash2` (vLLM native `dflash`, EAGLE3-style) |
| Context / KV / streams | 262144 tokens, `KV_BYTES=18g`, `SEQS=16`, prefix cache + chunked prefill on |
| PLE table | 320001536 rows × 160 B FP8 over RDMA (`192.168.0.1:18515`), graph-capturable host-func lookup (default) |
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

- One GB10, ARM64 Linux, Docker + NVIDIA container runtime, `/dev/infiniband`
  (RoCE device, default `roceP2p1s0f0`) with unlimited memlock.
- wtako's `ple_rdma_server` reachable at `192.168.0.1:18515` with the FP8
  table. vLLM aborts at boot without it; the launchers wait first.
- Local paths for the hybrid checkpoint and the DFlash2 drafter (defaults in
  `serve-magi.sh` point at shared storage).

## Quick start

```bash
git checkout magi-v3
cd recipe/build/image
docker build -f Dockerfile.v032 -t qwen38-flash-dgx:magi-v3-v032 .
cd ../../..
bash serve-magi.sh --print   # inspect the docker command
bash serve-magi.sh           # start qwen38-flash on :8000 (detached)

until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 10; done
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":512}'
```

Follow startup with `docker logs -f qwen38-flash`. Expect: PLE RDMA hello
(320001536 rows, scale ~1.99e-04), DFlash capture on mid-model aux layers,
`Graph capturing finished`, then `/health` 200.

## Configuration

`serve-magi.sh` defaults; override via environment. `SPEC=mtp` keeps the old
MTP path, `SPEC=0` disables speculation.

| Variable | Default | Meaning |
|---|---|---|
| `SPEC` / `NSPEC` | `dflash` / `3` | Speculative method (`mtp`, `dflash`, `0`) and draft tokens |
| `MODEL_DIR` / `DRAFT_DIR` | shared-storage paths | Target checkpoint / DFlash2 drafter (drafter mounted at `/drafter` only for `dflash`) |
| `PLE_RDMA` | `192.168.0.1:18515` | Table server; unset reverts to local-table behavior |
| `VLLM_PLE_RDMA_DEV` / `_GID` | `roceP2p1s0f0` / `auto` | RDMA device and GID index |
| `VLLM_PLE_RDMA_GRAPH` | `1` | `1`: graph-capturable host-func lookup; `0`: Python eager-break path |
| `CG_MODE` | `FULL_AND_PIECEWISE` | vLLM CUDA-graph mode |
| `KV_BYTES` / `SEQS` | `18g` / `16` | KV pool bytes / max sequences (18g is the post-OOM value) |
| `VLLM_FP8_HYBRID` | `1` | INT4 + blockwise-FP8 hybrid dispatch + QSA indexer fix |
| `PIN_PROMPT` | Magi AI prompt | Marker substring: a request containing it gets its prompt-prefix KV blocks pinned (never evicted, 25% pool cap) |
| `VLLM_MARLIN_USE_ATOMIC_ADD` | `1` | MoE GEMM backend workaround |
| `VLLM_USE_DEEP_GEMM` / `VLLM_USE_FLASHINFER_SAMPLER` | `0` / `1` | GEMM and sampler backend selection |

Gotcha: `VLLM_PLE_RDMA_GRAPH=0` with `FULL_*` graphs **crashes graph
capture** (`Cannot copy between CPU and CUDA tensors … unless pinned`) — the
Python path forces piecewise, so pair it with `CG_MODE=PIECEWISE`.

## How it works

```text
INT4 experts + block-FP8 sides + INT8 head (table-stripped hybrid)
             │
             ├── PLE rows: one-sided RDMA READs from wtako (no local table)
             │     graph path: ids → pinned host buf → cuLaunchHostFunc READ
             │     → rows back to GPU, all inside the CUDA graph
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
