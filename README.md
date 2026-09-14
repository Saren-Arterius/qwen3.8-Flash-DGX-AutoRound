# Qwen3.8-Flash-Next on one DGX Spark

### Two checkpoints: `hibrid48`, the default — and `hibrid48-uncensored`, made from it (abliterated, gated). Switch with one line in `recipe.yaml`

[`hibrid48`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48) (the base model, default, every speed number below)
and [`hibrid48-uncensored`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored) (OrcaRouter's abliterated
body with the same output head, no refusals, no guardrails — gated, research / private use). Same shapes tensor for tensor,
so the same speed. To switch: in `recipe.yaml` comment the active `model:` line and uncomment the other, then `./run.sh`.
Details in [Which checkpoint](#which-checkpoint).

**One box. 262k context. 60 tok/s single-stream on code at 17.7 engine steps/s — v2 did 50 at 14.4 — and 194 tok/s across
eight streams. Three commands.**

**v3 (2026-09-14)** serves Qwen's flagship 180B MoE (6B active), vision included, on a **single NVIDIA DGX Spark (GB10, 119G
unified memory)**: the same checkpoint our [2-Spark kit](https://github.com/bilikaz/qwen38-flash-next-cluster-recipe) serves,
on the same engine (vLLM 0.29). What changed against v2: the engine (vendor pin → vLLM 0.29), and one tensor of the checkpoint
(the 1.18 GB bf16 output head → 0.33 GB NVFP4 — it was 27 % of every decode step). The fit trick is v2's, ported: the 26.9 GiB
n-gram table is never allocated, the GPU reads it out of the checkpoint files through unified memory.

## Quality

Both checkpoints, lm-evaluation-harness against the running serve, **thinking on**, temperature 0.6 / top-p 0.95 / top-k 20,
a 32k-token budget per answer, HumanEval complete and a fixed 200-question subset (seed 123123123) of the others — the same
763 questions for both columns. Measured on the two-Spark serve of these checkpoints (bf16 KV; this kit's fp8 KV is the one
difference — the runner, `bench/quality/` in the myllmbox repo, works against this serve too). Qwen publishes no numbers for
these four tests (its card reports LiveCodeBench v6 91.9, GPQA Diamond 91.7, IFBench 81.3, SWE-bench Pro 62.5).

| task | questions | hibrid48 | hibrid48-uncensored |
|---|---|---|---|
| HumanEval pass@1 | 164 | **95.7** | **94.5** |
| GSM8K exact match | 200 | **98.0** | **97.5** |
| IFEval prompt-level strict / instruction-level strict | 200 | **91.5** / 93.4 | **94.5** / 96.2 |
| MMLU-Pro (14 subjects, sampled by size) | 200 | **84.9** | **82.9** |
| answers that ran into the 32k budget while thinking (count as wrong) | 763 | 11 | 6 |

The IFEval gain is the one difference larger than the subsets' sampling error (about ±3 points); the other deltas are one to
four questions each. The abliterated body thinks shorter (median reasoning −8 %, 90th percentile −27 %) and runs away half as
often. Published leaderboard numbers use other prompts, few-shot counts and full sets — a sanity band, not a column.

## Measured performance (this exact kit, single Spark, K=3, fp8 KV 8 GB, `vm.compaction_proactiveness=0`)

Boot 2026-09-14, myllmbox "pasture" prompt, 10-second engine windows (all streams decoding, zero prefill in the window);
**sustained** = the run average, **peak** = the best window. v2 numbers are the ones this README carried at tag `v2`.

| concurrent requests | v2 sustained | v3 sustained | **v3 peak** | engine steps/s (v2 → v3) | acceptance |
|---|---|---|---|---|---|
| 1 · code (thinking off) | 50–51 | 60 | **64** | 14.4 → 17.7 | 3.4 |
| 1 · thinking on, full 45k-token request | 39–42 | 46–50 | **68** | 14.4 → 16.8 | 2.7–3.0 (2.3 reasoning → 3.6 answer) |
| 2 · code | — | 94 | **100** | — → 13.9 | 3.4 |
| 4 · code | 129 | 134 | **145** | 9.3 → 10.0 | 3.4 |
| 6 · code | — | 169 | **180** | — → 8.3 | 3.4 |
| 8 · code (every seat taken) | 182 | 194 | **205** | 6.6 → 7.1 | 3.4 |

Reading it: the 4-bit head is a fixed per-step saving, so it shows most where the step is cheapest — +23 % engine steps at
c=1, +8 % at c=4 and c=8, where the expert GEMMs dominate. Acceptance is flat across the ladder, so tok/s follows the step
rate: 60 at one stream, 134 at four, 194 at eight. The step is 57 ms (v2: 70) from the first token to the last; tok/s = steps ×
accepted tokens, so the *text* decides the number — reasoning prose yields ~2.3 tokens a step, code ~3.5, and you can read
the phase change of a thinking-on request straight off the throughput line.

**K=3 vs K=4, measured on two Sparks the same day (same checkpoint, pin and prompt):** K=4 gives +3–8 % tok/s at 1–6
streams on code and +6–16 % on thinking-on prose (52.7 avg, 76 peak against 46–50 / 62–68), for 10 % fewer engine steps —
and the eighth seat no longer fits the 8 GB pool (the per-request state is one block larger). K=3 keeps all eight seats at
the same single-stream ceiling within noise, so it ships. If you run 1–4 long thinking-on requests and never eight, set
`num_speculative_tokens` to 4 and the capture sizes to multiples of 5 (see Tuning).

## Quick start

```bash
git clone https://github.com/bilikaz/qwen38-flash-next-recipe.git
cd qwen38-flash-next-recipe
./run.sh        # downloads ~98G from HF on first run, serves OpenAI API on :8000
```

`./stop.sh` stops it. `./view.sh` shows live stats (throughput, KV usage, speculative-decoding acceptance).
`./ple.sh status` shows the table (rows in memory, free, swap); `./ple.sh populate` re-pulls it. Requirements: a DGX
Spark with docker + NVIDIA container runtime. First boot reaches healthy in ~12 minutes (weights load ~11 min + one-time
compile warmup); later boots are faster. Swap on the box is a plus, not a must (see Memory).

**All configuration lives in [`recipe.yaml`](recipe.yaml)** — one file: image, weights repo, port, context length,
KV budget, every vLLM flag. Nothing else to edit.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen/Qwen3.8-Flash-Next",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

**Hugging Face token.** Anonymous downloads are rate-limited, and gated models (license-agreement repos, e.g. uncensored
variants) refuse anonymous access. `run.sh` looks for `HF_TOKEN`, then `~/.cache/huggingface/token` (`hf auth login`), and asks
for one when the repo is gated — after you accepted its agreement on the model page. Nothing is stored by the kit.

## Which checkpoint

Two checkpoints run on this stack; `recipe.yaml` ships with the first active and the second commented out under it.
Switching is comment one line, uncomment the other, `./run.sh`:

| `model:` | what it is | on this kit |
|---|---|---|
| `myllmbox/Qwen3.8-Flash-Next-hibrid48` (default) | the base model, calibrated body, NVFP4 output head — every speed number above | 17.7 steps/s, 60 tok/s at c=1, 64 peak |
| `myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored` | OrcaRouter's abliterated (refusal-removed) body with the same head — **no guardrails**; research, red-teaming, private use behind your own moderation | same shapes and precisions tensor for tensor → same speed; quality table above (IFEval 94.5, HumanEval 94.5, GSM8K 97.5, MMLU-Pro 82.9) |

The uncensored repo is **gated**: open its Hugging Face page, accept the agreement, then `hf auth login` (or `export
HF_TOKEN=…`) before `./run.sh` — the kit checks both and tells you what is missing. Running both checkpoints at different
times? Set `served-model-name` to something distinct (e.g. `Qwen/Qwen3.8-Flash-Next-Uncensored`) so clients and logs can tell
them apart. Both weigh 98 GB; the first download of the second one is a full download (different body), the 8 table shards are
shared bytes. hibrid47 (the v2 checkpoint, bf16 head) runs on this image too — slower, no reason to.

## Memory on a Spark: what the kit does about it

Unified memory means the GPU driver, the page cache and the kernel share one pool. This serve fills it on purpose:
~85 GB GPU process (72 weights + 8 KV + graphs) + 27 GB table + 7 GB of host processes ≈ the box. The kit never asks
for your password; it stabilises memory with what a user may do:

- **waits** after removing the old container until the box reports ≥ 100 GB available (unified memory takes 30–60 s to
  come back after a container dies; launching earlier gives a phantom "CUDA out of memory"),
- **evicts its own checkpoint files from the page cache** before launch (`dd iflag=nocache`, no privileges),
- the image's loader **drops each shard from the cache as soon as it has been consumed**,
- the table is read with **direct NVMe reads during boot** (no page-cache growth while autotune needs the room) and
  memory-mapped from then on; **one populate pass** right after warm-up pulls all 26.9 GiB in (~28 s).

What to expect after that pass: `./ple.sh status` shows ~18–21 GB of the table mapped and ~7 GB free — the kernel keeps a
floor free and reclaims the rest. Under sustained load it then parks ~7 GB of cold process heaps in **swap** (once; swap-ins
stay ≈ 0) and the whole table ends up resident. Until it does, a request that touches cold rows — the first minutes of a
thinking-on request, whose reasoning prose spans far more n-grams than code — reads them from NVMe (a GPU page fault ≈ 200 µs;
measured 15–17 steps/s instead of 17.7 for those minutes). Without swap the kernel evicts table rows instead and they come back
on demand. `./ple.sh populate` repeats the pass any time.

One thing you *can* do, with root, and it is worth ~10 % on a serve that runs this close to the memory edge:

```
./tune-host.sh      # sets vm.compaction_proactiveness=0; shows the command, asks, then sudo prompts; persisted
```

`run.sh` reads the value before every launch (no privilege needed) and warns while it is not 0; it never applies the
change itself. The kernel's background page compactor migrates pages to build large contiguous blocks; on a Spark the
GPU's memory *is* those pages, so every migration first unmaps them from the GPU — measured as a 4–5 s slowdown every
~37 s. A serving box allocates once at boot and gains nothing from the upkeep.

## Tuning (recipe.yaml)

- **`kv-cache-memory`** (bytes): 8 GB fp8 = 447,935 tokens. The table and the pool are the same memory: a bigger pin
  means fewer table rows resident (more NVMe re-reads), not a faster serve; 7 GB hands the table a gigabyte back.
  `kv-cache-dtype: fp8` costs no acceptance at K=3 (3.41 vs 3.32 bf16 measured) — drop the line for bf16 (~250k tokens).
- **`max-num-seqs`** 8: ~12 % of the pool per running request regardless of length (the model's GDN state). 8 seats =
  ~3k tokens of context each; 4 seats = ~50k each; 1–2 = the full 262k. Set it for your workload, not for the ladder.
- **`speculative-config`** K=3 (the A/B above). K=4: `num_speculative_tokens: 4` **and** `cudagraph_capture_sizes` to
  multiples of 5 up to 40 — vLLM 0.29 rounds every FULL decode-graph size up to a multiple of K+1 and drops the rest, so a
  list that stops short silently leaves the upper seats without a graph.
- **`MBX_PLE_MMAP_PREWARM`** (env): `auto` = populate the whole table after boot; `0` = fill on demand only (the hot set
  of a workload is ~19 GB); `<seconds>` = populate that long after boot.
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits one max-resolution image (~4.1k tokens)
  with room; don't lower it if you send images (4096 also made 0.29 add a compile range that crashed the MoE profile run).
- **`async-scheduling` on**: +9–12 % throughput at 4–5 concurrent requests, neutral at 8.
- Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.

## What's in the image

`myllmbox/qwen38-flash-next-vllm:v3` — `vllm/vllm-openai:v0.29.0` (Qwen3.8-Flash-Next is upstream there) plus **eight
readable patches**:

1. the NVFP4 table loader (stock 0.29 knows bf16/FP8 tables only; here its parameters are zero-sized),
2. QSA pre-indexer RoPE clamp and 3. the fused MTP draft (from the two-Spark kit),
4. **quant_config on both output heads** — the 4-bit head loads; stock 0.29 fails on it with a shape mismatch,
5. **fp8 KV on the QSA path** — upstream vLLM PR #54846 ported onto 0.29 (whole-file overlays, sha256-pinned),
6. the loader drops each shard's pages from the page cache as soon as its tensors are consumed,
7. **demand-paged table** (`MBX_PLE_MMAP=1`): the 8 shard files are mmapped, a DLPack capsule over the mapping gives the
   GPU a tensor to gather from (GB10: `pageableMemoryAccess`, `usesHostPageTables`) — v2's patch, re-anchored,
8. **dual gather path** (`vllm::mbx_ple_gather`): direct NVMe reads during boot, the mapping afterwards
   (`MBX_PLE_MMAP_MODE=auto`); populate-after-boot (`MBX_PLE_MMAP_PREWARM`); flag files under `cache/` for `ple.sh`.

Each patch refuses to apply twice and fails the build if its anchor moved; the build runs a CPU-side test of the whole
table path on a synthetic checkpoint. The Dockerfile, the patch scripts and the build ledger live in the myllmbox repo under
[`builds/qwen38-flash-next/solo/`](https://github.com/bilikaz/myllmbox-runner/tree/main/builds/qwen38-flash-next/solo)
— rebuild and diff it yourself. Digest: `sha256:61d2bc6ba5977024895734d0ef94918ac806d936c4e17a263906e246599f9862`.

v2 (`…-vllm:v2`, hibrid47 on the vendor SM121 pin, six patches) and v1 (`…-vllm:v1`, hibrid46, int3 table in a CPU worker)
stay available: `git checkout v2` / `git checkout v1`.

## The full box

This kit serves one model, plain. The same model runs under
[myllmbox](https://github.com/bilikaz/myllmbox-runner) with a public HTTPS tunnel, keepalive and multi-model
management — same image, same weights, one `./run.sh qwen38-flash-next`.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products
must display the model name; Model-as-a-Service businesses need a separate Qwen license). Kit
scripts and image patches: MIT.
