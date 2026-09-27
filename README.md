# magi-v2 — Qwen3.8-Flash-Next on one DGX Spark, n-gram table over RDMA

> This branch is one specific deployment: a DGX Spark (GB10, "magi") serving
> **Qwen3.8-Flash-Next** with its n-gram ("PLE") table held in the RAM of a second
> machine ("wtako") and read over a 100G RoCE link. Without a box with ~32 GB of spare
> RAM and an RDMA NIC next to your Spark, this branch isn't for you — use
> [bilikaz/qwen38-flash-next-recipe](https://github.com/bilikaz/qwen38-flash-next-recipe)
> directly (it reads the table from local NVMe).

magi-v2 is [bilikaz's recipe](https://github.com/bilikaz/qwen38-flash-next-recipe) v4
(vLLM 0.30 + the `myllmbox/Qwen3.8-Flash-Next-hibrid48` checkpoint) with three changes:

1. **PLE rows over RDMA.** Each token's 16 table rows are fetched with one-sided RDMA
   READs from wtako (~1 ms per call) instead of NVMe reads (~11 ms). With RDMA on,
   magi never opens the table files at all.
2. **Never-evict prompt pin.** The Home Assistant system prompt's KV stays resident
   through any traffic.
3. **Memory trims for a shared box.** The KV size is set explicitly, and torch's
   pinned host cache is flushed after loading, so ASR and TTS fit alongside.

| | |
|---|---|
| Model | myllmbox/Qwen3.8-Flash-Next-hibrid48 (NVFP4 body, 4-bit head, NVFP4 n-gram table) |
| Engine | vLLM 0.30.0, image `qwen38-flash-dgx:magi-v2` (on top of `myllmbox/qwen38-flash-next-vllm:v4`) |
| Served as | `qwen` on `:8000` (OpenAI-compatible), container `qwen38-flash` |
| Speculative decoding | MTP, 3 draft tokens (block rejection, probabilistic draft sampling) |
| KV cache | 20 GiB = 691,493 tokens (`kv-cache-memory: "21474836480"`) |
| Context | 262,144 tokens, up to 16 concurrent requests |
| Weights | read from wtako over NFS (`/mnt/storage@WTAKO/saren/AI/Qwen3.8-Flash-Next-hibrid48`), ~72.4 GiB resident |
| n-gram table | 320,001,536 rows × 90 B (80 packed + 10 fp8 scales) = 26.8 GiB, in wtako's RAM, served on `192.168.0.1:18516` |

## Layout

```
serve-magi.sh            entry point: exec v2/run.sh (called by ~/magi-stack/ram-client-ple.sh)
v2/
  recipe.yaml            THE config: image, env, every vLLM flag. run.sh reads only this.
  run.sh stop.sh view.sh bilikaz's recipe scripts (container renamed to qwen38-flash, --restart unless-stopped)
  Dockerfile             base image + libple_rdma.so + the two patches below
  patch_never_evict_v030.py   never-evict pin, one scheduler anchor ported to vLLM 0.30
  patch_ple_prof_rdma.py      appended to ngram_embedding.py: RDMA row source, profiler, no-table-files loader, pinned-host flush
src/ple_rdma/
  ple_rdma.c             tiny verbs library (device open, MR register, QP connect, batched READs)
  ple_nvfp4_rdma_server.py + ple-nvfp4-rdma-server.service   the table server that runs on wtako
```

Everything else at the top level (`Dockerfile`, `prepare.sh`, `serve.sh`, `src/`,
`tools/`, `docs/`, `bench/`) is the **v1 stack** (Intel AutoRound int4 + fp8 table, the
`magi` branch). It's kept so a rollback is one checkout. v2 doesn't use it.

## Running

magi's supervisor, `~/magi-stack/magi-stack.sh` (started at boot), owns the whole chain:
wait for wtako's table server → `serve-magi.sh` → LLM warm → vllm-q3asr (ASR) →
voxcpm (TTS). If the LLM goes unhealthy it stops ASR and TTS first, to free memory, and
brings them back once the LLM is warm again. `touch /tmp/magi-stack-no-deps` keeps ASR
and TTS down until the next reboot.

By hand:

```bash
docker build -f v2/Dockerfile -t qwen38-flash-dgx:magi-v2 .   # from this directory
./serve-magi.sh                        # (re)creates qwen38-flash; waits for ≥100G free first
v2/view.sh                             # live throughput / KV / acceptance
docker logs -f qwen38-flash
```

Changing a flag means editing `v2/recipe.yaml` and recreating the container.
`docker restart` reuses the old command line.

A healthy boot logs:

```
PLE: loader skips 8 table files (MBX_PLE_RDMA)
PLE: rows over RDMA from 192.168.0.1:18516 (320001536 rows x 90 B, dev roceP2p1s0f0 gid 5)
PLE: table files not opened (MBX_PLE_RDMA)
PLE-HOST: after load_model: ... shared-anon RSS 4680 -> 73 MiB after flush
... reserved 20.0 GiB memory for KV Cache ...
```

### wtako: the table server

`ple_nvfp4_rdma_server.py` loads the NVFP4 table (`/mnt/storage/saren/AI/ple-table-nvfp4`)
into one registered memory region of interleaved 90-byte rows. It then hands the
region's address and key to each client over TCP `:18516`. Clients read rows directly;
the server's CPU is idle while serving.

```bash
sudo cp src/ple_rdma/ple-nvfp4-rdma-server.service /etc/systemd/system/
sudo systemctl enable --now ple-nvfp4-rdma-server     # ~27.5 GiB MR, pinned
```

The older fp8 server (`ple-rdma-server.service`, `:18515`) belongs to v1. It stays
disabled while v2 runs, with its unit file unchanged.

## The patches (`v2/patch_ple_prof_rdma.py`)

The file is appended to the image's `ngram_embedding.py` and replaces module functions
that the table code looks up at call time. That leaves the breakable-cudagraph split
points unchanged: the graph pieces are the same as in stock NVMe mode. Everything
below applies only when `MBX_PLE_RDMA=host:port` is set, except the pinned-host flush,
which always runs.

- **RDMA row source.** It replaces `_mbx_t_gather`, which turns ids into their unique
  rows, reads them over one QP into a pinned MR buffer, and scatters them back. There's
  no fallback to local reads: an RDMA error raises.
- **No table files.** `_mbx_t_setup` checks the server's geometry and global scale
  against the checkpoint config instead of opening files. `_mbx_mmap_attach` maps
  nothing and starts no prewarm thread. The loader leaves the 8 `ple-nvfp4-*`
  safetensors out of its file list. In their place it yields zero-storage stand-in
  tensors that carry the row counts, and `nvfp4_global` comes from the server.
  Graph capture gets zero rows.
- **Profiler (`MBX_PLE_PROF=1`).** Logs `PLE-PROF src=rdma calls=… read_avg=…ms` every
  30 s (`MBX_PLE_PROF_EVERY`).
- **Pinned-host flush.** After `load_model` and after `compile_or_warm_up_model`, it
  calls `torch.cuda.empty_cache()` + `torch._C._host_emptyCache()`. The weight load
  leaves ~4.6 GiB in torch's pinned-host cache (power-of-two blocks, never returned to
  the OS), and on GB10 that's system RAM taken from everything else.

`v2/patch_never_evict_v030.py` ports v1's never-evict pin (`src/patch_never_evict.py`).
Blocks of requests whose prompt contains `never-evict-kv-cache-prompt-includes` are never
evicted from the prefix cache, capped at `never-evict-kv-cache-max-fraction` (0.25)
of the pool.

## Memory budget (121.6 GiB unified)

GB10 GPU allocations are system RAM, and an overrun doesn't just kill one process:
the driver fails allocations (`NV_ERR_NO_MEMORY`) and the whole box can go down.

| Consumer | ≈ GiB |
|---|---|
| LLM weights | 72.4 |
| KV cache | 20.0 |
| cudagraphs, CUDA context, workspaces | ~4 |
| LLM activations (grow at runtime, see below) | 1–3 |
| LLM host side (API server, engine core, worker heap) | ~5 |
| vllm-q3asr (Qwen3-ASR-1.7B, BF16) | ~5.5 |
| voxcpm | ~7.5 |

That's more or less all of it. With a 20 GiB KV the full stack runs with a few GB
free, or slightly overcommitted with some swap going to wtako's NVMe-oF RAM device.
Levers, from cheapest to most costly:

- **`max-num-batched-tokens` 8192 → 4096.** With `kv-cache-memory` set, vLLM skips memory
  profiling, so the prefill activation peak is never reserved. It's allocated the first
  time a big prefill arrives and then kept. Halving the chunk roughly halves that peak.
- **`kv-cache-memory` 20 → 16 GiB.** Frees 4 GiB (553K tokens is still ~2 full contexts).
- ASR in FP8 is **not** an option on its image: vLLM 0.19.1-dev crashes the GPU (Xid 43)
  during inductor autotuning with `--quantization fp8`.

## Benchmarks

Measured 2026-09-27/28 with `tool-eval-bench --perf-only` (llama-benchy), MTP=3,
weights from local disk, 25 GiB KV. v1 = Intel AutoRound int4 + fp8 table over RDMA.
`tg t/s` at c>1 is per-stream, not aggregate.

| pp2048 tg128 | v1 pp t/s | v1 tg t/s | v1 TTFT ms | **v2 pp t/s** | **v2 tg t/s** | **v2 TTFT ms** |
|---|---|---|---|---|---|---|
| d0 c1 | 1,673 | 41.4 | 1,962 | **2,424** | **48.5** | **1,001** |
| d0 c2 | 1,629 | 54.0 | 2,623 | 2,142 | 64.9 | 1,765 |
| d0 c4 | 1,639 | 86.9 | 5,081 | 2,291 | 81.2 | 3,099 |
| d4096 c1 | 2,099 | 41.6 | 3,104 | 2,257 | 52.4 | 2,881 |
| d4096 c2 | 1,737 | 58.7 | 6,923 | 2,112 | 40.3 | 4,380 |
| d4096 c4 | 1,613 | 48.4 | 13,876 | 2,241 | 44.6 | 8,252 |
| d8192 c1 | 2,193 | 41.2 | 4,835 | 2,322 | 48.0 | 4,568 |
| d8192 c2 | 1,952 | 46.1 | 9,782 | 2,229 | 31.9 | 6,894 |
| d8192 c4 | 1,734 | 29.0 | 19,480 | 2,230 | 27.7 | 12,860 |

16 streams, pp0 tg1024: ~290 t/s aggregate.

v2 wins single-stream decode, prefill and time to first token everywhere. v1 still
decodes faster at c2/c4 with a deep context. Table reads over RDMA average ~1.25 ms
per call, against ~11.4 ms from NVMe, which is worth +5–8% prefill.

## Rollback to v1

```bash
git checkout magi                                  # this repo
# ~/magi-stack: revert the :18516 wait in ram-client-ple.sh (commit cce7309)
# on wtako:
sudo systemctl disable --now ple-nvfp4-rdma-server && sudo systemctl enable --now ple-rdma-server
sudo reboot                                        # magi
```

## Credits

- Model: **Qwen team, Alibaba** — Qwen3.8-Flash-Next.
- Serving recipe, scripts and image: **[bilikaz/qwen38-flash-next-recipe](https://github.com/bilikaz/qwen38-flash-next-recipe)**
  (`myllmbox/qwen38-flash-next-vllm:v4`, vLLM 0.30 + the mbx patches for the NVFP4
  table, 4-bit head, fused MTP draft and NVMe/mmap table reader).
- Checkpoint: **[myllmbox/Qwen3.8-Flash-Next-hibrid48](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48)**.
- The mmap-PLE idea and the original GB10 recipe:
  **[blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)**, which this
  repository was forked from (the v1 stack and `docs/` are built on it).
- The never-evict pin was built for a Qwen3.5-122B Spark stack on top of the ARC
  GPU-eviction work in [vllm#40270](https://github.com/vllm-project/vllm/pull/40270).
- Serving engine: **vLLM**.
- License: [Apache-2.0](LICENSE).
