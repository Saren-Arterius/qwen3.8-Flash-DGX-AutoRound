# Qwen3.8-Flash-Next on one DGX Spark

### Two checkpoints: `hibrid48`, the default — and `hibrid48-uncensored`, made from it (abliterated, gated). Switch with one line in `recipe.yaml`

[`hibrid48`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48) (the base model, default, every speed number below)
and [`hibrid48-uncensored`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored) (OrcaRouter's abliterated
body with the same output head, no refusals, no guardrails — gated, research / private use). Same shapes tensor for tensor,
so the same speed. To switch: in `recipe.yaml` comment the active `model:` line and uncomment the other, then `./run.sh`.
Details in [Which checkpoint](#which-checkpoint).

One box, one model. **73 tok/s single-stream (82 peak), 288 tok/s at 16 streams (318 peak), an 830k-token KV pool** — and
it boots in about four minutes. Three commands. **New in v5: an opt-in dynamic draft depth — +12 to +19 % on long prose, level
on everything else** ([Dynamic draft depth](#dynamic-draft-depth-v5-opt-in)).

## Measured performance (this exact kit, one DGX Spark, K=5, `vm.compaction_proactiveness=0`)

**v4 ladder (2026-09-25, vLLM 0.30, image v4, `hibrid48`, 27G bf16 pin, 16 seats, Marlin MoE)** — this kit exactly as shipped,
myllmbox "pasture" prompt, 10-second engine windows (all streams decoding, zero prefill in the window), averages of 3 runs per
rung. Thinking off; the single-stream peak is from a full thinking-on request (its code phase).

| concurrent requests | **PEAK tok/s** (v3 → v4) | average tok/s | per-stream (v3 → v4) | acceptance |
|---|---|---|---|---|
| 1 | 68 → **82** | 73 | 60 → **73** | 5.06 |
| 2 | 100 → **123** | 113 | 47 → **56** | 5.08 |
| 4 | 145 → **162** | 154 | 33 → **38** | 5.11 |
| 6 | 180 → **198** | 188 | 28 → **31** | 5.14 |
| 8 | 205 → **227** | 215 | 24 → **27** | 5.12 |
| 12 | — → **270** | 253 | — → **21** | 5.14 |
| 16 | — → **318** | 288 | — → **18** | 5.13 |

Reading it: **every rung is 11–22 % faster than v3, and the box now seats twice as many requests.** Each engine step drafts
five tokens instead of three and keeps 5.1 of a possible 6 — the drafter's tokens are sampled from its own distribution and
verified against the model's (see [What changed](#what-changed)). One full thinking-on request (reasoning, then the answer):
**57 tok/s on average, 77 while writing the code, 82 at peak** (v3: 50 / 62 / 67). The pool is 2.5× v3's in tokens with bf16
KV, so every seat carries more context; 16 streams is the practical top (each running request holds ~36k tokens of pool for the
model's recurrent state).

**v3 ladder (2026-09-14, vLLM 0.29, K=3, 8G fp8 pin)** — kept as the reference v4 is measured against:

| concurrent requests | v2 sustained | v3 sustained | **v3 peak** | engine steps/s (v2 → v3) | acceptance |
|---|---|---|---|---|---|
| 1 · code | 50–51 | 60 | **68** | 14.4 → 17.7 | 3.4 |
| 2 · code | — | 94 | **100** | — → 13.9 | 3.4 |
| 4 · code | 129 | 134 | **145** | 9.3 → 10.0 | 3.4 |
| 6 · code | — | 169 | **180** | — → 8.3 | 3.4 |
| 8 · code  | 182 | 194 | **205** | 6.6 → 7.1 | 3.4 |

## Dynamic draft depth (v5, opt-in)

With `mtp_depth.mode: dynamic` in `recipe.yaml`, each request picks its own draft depth (3 to 6) from what it has recently
accepted, and a batch drafts its requests' average depth. Prose wants short drafts, code and JSON long ones; the fixed K=5 of
v4 is right for code and too deep for prose. The default stays `mode: off` — exactly v4.

**Dynamic vs fixed K=5 (v4), aggregate tok/s** (one DGX Spark, thinking off, averages of 3 runs; c=1 = one full answer,
c≥2 = 300 s with every stream kept busy):

| content | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| long prose (7k-word story) | 39.7 → **44.3** (+12 %) | 61.0 → **69.7** (+14 %) | 90.3 → **102.3** (+13 %) | 108.7 → **124.0** (+14 %) | 124.7 → **143.0** (+15 %) | 144.7 → **172.3** (+19 %) |
| mixed (code + explanation) | 52.7 → 55.0 | 81.3 → 82.7 | 120.0 → 120.0 | 154.3 → 151.3 | 182.0 → 181.0 | 218.0 → 214.3 |
| JSON (schema-constrained) | 69.7 → 69.3 | 103.7 → 105.7 | 149.3 → 150.0 | 178.3 → 178.0 | 203.7 → 201.7 | 232.7 → 233.3 |
| code (TypeScript) | 69.7 → 69.7 | 102.3 → 103.0 | 140.0 → 142.3 | 164.0 → 169.0 | 185.3 → 185.3 | 217.0 → 213.3 |

One full thinking-on request (the pasture prompt), split by phase so answers that think longer or shorter stay comparable:
thinking 49.7 → **51.4** tok/s, writing the code 70.9 → **72.1** tok/s (fixed K=3 / 4 / 6: 50.7 / 50.0 / 48.0 thinking, 63.2 / 66.3 /
68.8 writing the code).

**To turn it on**, in `recipe.yaml`:

1. `mtp_depth.mode: dynamic`,
2. swap in the two commented lines next to `speculative-config` and `compilation-config` (`num_speculative_tokens: 6` and
   its capture sizes),
3. `./run.sh`.

**Knobs** (`mtp_depth` in `recipe.yaml`; `run.sh` writes them to `cache/mbx-depth.json`, and the serve re-reads that file
while it runs — edit it to change them live):

| knob | default | what it does |
|---|---|---|
| `min` | 3 | lowest draft depth |
| `window` | 48 | steps of history each decision looks at (8–128) |
| `promote` | [60, 45] | +2 / +1 depth when the deepest position was accepted on more than this % of the window |
| `demote` | [25, 15] | −1 / −2 depth when below this % |
| `log` | false | `true` = one log line per decision: depth before → after and the measured rate |

The deepest depth is `num_speculative_tokens`. For 7, add the multiples of 8 (8 × 1 … 8 × seats) to the capture sizes.

## Quality (measured on this model, thinking on)

Both checkpoints, lm-evaluation-harness against a running serve, **thinking on**, temperature 0.6 / top-p 0.95 / top-k 20,
a 32k-token budget per answer, HumanEval complete and a fixed 200-question subset (seed 123123123) of the others — the same
763 questions for both columns. Qwen publishes no numbers for these four tests (its card reports LiveCodeBench v6 91.9, GPQA
Diamond 91.7, IFBench 81.3, SWE-bench Pro 62.5). The v4 changes are output-exact (speculative decoding keeps the model's own
distribution), so the scores carry over.

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

## What changed

**v5 (2026-10-01): dynamic draft depth, opt-in.** Per-request draft depth from measured acceptance, live-tunable knobs,
+12 to +19 % on long prose at every concurrency, level on mixed text, JSON and code
([Dynamic draft depth](#dynamic-draft-depth-v5-opt-in)). With `mode: off` (the default) the serve is v4. v4 stays available:
`git checkout v4`.

**v4 (2026-09-25): vLLM 0.30, five sampled draft tokens, an 830k-token pool and 16 seats.**

1. **Engine: upstream vLLM 0.30** (was 0.29), with the fused multi-step MTP draft for this model's attention (proposed upstream
   as vllm-project/vllm#58449).
2. **Five draft tokens, sampled.** The drafter proposes 5 tokens instead of 3 and samples them from its own distribution
   (`draft_sample_method: probabilistic`); the model checks them with the exact probability-ratio test and block verification.
   Output quality cannot change — the text follows the model's distribution either way — but more drafts survive: 5.1 accepted
   per step on code (v3: 3.4 of 4).
3. **The n-gram table no longer takes memory.** On the first boot the image prepares a table map in `./cache` (~31G, a few
   minutes, once; it works for both checkpoints) and serves from it. The memory that frees goes to the KV pool: 8G fp8 →
   **27G bf16 = 830,582 tokens** (3.17 full 262K requests), and 16 seats instead of 8.
4. **Boots in about four minutes** instead of about twelve (`load-format: fastsafetensors`, and no per-boot compilation on 0.30).

v3 stays available: `git checkout v3` (image `…-vllm:v3`, vLLM 0.29, K=3, fp8 KV 8G, 8 seats).

**v3 (2026-09-14): vLLM 0.29 and a 4-bit output head.** The engine moved from the vendor SM121 pin to vLLM 0.29, and the
checkpoint's 1.18 GB bf16 output head became 0.33 GB of NVFP4 (`hibrid48`) — it was 27 % of every decode step: 60 tok/s
single-stream at 17.7 engine steps/s (v2: 50 at 14.4). v2 stays available: `git checkout v2`.

## Quick start

```bash
git clone https://github.com/bilikaz/qwen38-flash-next-recipe.git
cd qwen38-flash-next-recipe
./run.sh        # downloads ~98G from HF on first run, serves the OpenAI API on :8000
```

`./stop.sh` stops it. `./view.sh` shows live stats (throughput, KV usage, speculative-decoding acceptance). Requirements: a DGX
Spark with docker + the NVIDIA container runtime, and ~31G free on the disk under `./cache` for the table map. A boot reaches
healthy in about four minutes (weights ~80 s); the very first boot on a box adds a few minutes to prepare the map.

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
| `myllmbox/Qwen3.8-Flash-Next-hibrid48` (default) | the base model, calibrated body, NVFP4 output head — every speed number above | v4: 73 tok/s at c=1 (82 peak), 288 at 16 streams |
| `myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored` | OrcaRouter's abliterated (refusal-removed) body with the same head — **no guardrails**; research, red-teaming, private use behind your own moderation | same shapes, same n-gram table → same speed and the same table map; quality table above (IFEval 94.5, HumanEval 94.5, GSM8K 97.5, MMLU-Pro 82.9) |

The uncensored repo is **gated**: open its Hugging Face page, accept the agreement, then `hf auth login` (or `export
HF_TOKEN=…`) before `./run.sh` — the kit checks both and tells you what is missing. Running both checkpoints at different
times? Set `served-model-name` to something distinct (e.g. `Qwen/Qwen3.8-Flash-Next-Uncensored`) so clients and logs can tell
them apart. Both weigh 98 GB; the first download of the second one is a full download (different body), the 8 table shards are
shared bytes.

## Memory on a Spark: what the kit does about it

Unified memory means the GPU driver, the page cache and the kernel share one pool, and this serve uses most of it on purpose:
~72G of weights + the 27G KV pool + graphs and host processes. The kit never asks for your password; it stabilises memory with
what a user may do:

- **waits** after removing the old container until the box reports ≥ 100G available (unified memory takes 30–60 s to come back
  after a container dies; launching earlier gives a phantom "CUDA out of memory"),
- **evicts its own checkpoint files from the page cache** before launch (`dd iflag=nocache`, no privileges),
- the weights load with `fastsafetensors`, and the image drops each shard from the page cache once it is consumed.

Measured on this kit: ~5.7G available at one stream, 2.8–2.9G at 16 streams. If you run other things on the box, set
`kv-cache-memory` to 26G (~800k tokens).

One thing you *can* do, with root, and it is worth ~10 % on a serve that runs this close to the memory edge:

```
./tune-host.sh      # sets vm.compaction_proactiveness=0; shows the command, asks, then sudo prompts; persisted
```

`run.sh` reads the value before every launch (no privilege needed) and warns while it is not 0; it never applies the
change itself. The kernel's background page compactor migrates pages to build large contiguous blocks; on a Spark the
GPU's memory *is* those pages, so every migration first unmaps them from the GPU — measured as a 4–5 s slowdown every
~37 s. A serving box allocates once at boot and gains nothing from the upkeep.

## Tuning (recipe.yaml)

- **`kv-cache-memory`** (bytes): 27G bf16 = 830,582 tokens; 26G ≈ 800k with more headroom.
- **`max-num-seqs`** 16: each running request holds ~36k tokens of pool for the model's recurrent state regardless of length,
  so 16 seats leave ~250k tokens of shared context; fewer seats = more context each.
- **`speculative-config`** K=5 with sampled drafts and block verification. K=4 (`num_speculative_tokens: 4`, capture sizes in
  multiples of 5 up to seats × 5, and remove `block-size`) measured 2–8 % slower up to 6 streams and equal at 8.
- **`compilation-config`**: the capture sizes are multiples of K+1 (6) up to seats × 6 = 96; shorten them only together with
  `max-num-seqs`, or the upper rungs decode without CUDA graphs.
- **`block-size: 1632`**: required at K=5 — the boot stops with "QSA ring capacity 12 must divide the attention block size 1616"
  without it.
- **`docker_args: --security-opt seccomp=unconfined`**: required by the image's table reader; the container still runs
  unprivileged otherwise.
- **`load-format: fastsafetensors`**: weights in ~80 s. Remove it for vLLM's default loader (~10 min).
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits one max-resolution image (~4.1k tokens).
- **`async-scheduling` on**. Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.
- **`mtp_depth`**: dynamic draft depth, off by default — see [Dynamic draft depth](#dynamic-draft-depth-v5-opt-in).
- **`patches`** (server): optional vLLM patches from [`patches/`](patches/), off by default — e.g. `patches: hermes-chat`
  for the Hermes agent (contributed by [@yume-arasaki](https://github.com/yume-arasaki)). Applied at launch over the image's
  files; the image itself is unchanged.

## What's in the image

`myllmbox/qwen38-flash-next-vllm:v5` — upstream `vllm/vllm-openai:v0.30.0` plus patches, each an anchored or sha256-checked
edit that refuses to apply twice and fails the build if its target moved:

1. **the NVFP4 n-gram table** on 0.30's embedding plugin — stock 0.30 refuses this checkpoint's table,
2. **the 4-bit output head** — stock 0.30 builds both output heads without the checkpoint's quantization config,
3. **fused multi-step draft metadata** — the code proposed upstream as
   [vllm-project/vllm#58449](https://github.com/vllm-project/vllm/pull/58449),
4. the loader's page-cache drop, the QSA pre-indexer rope clamp, and two inert knobs,
5. **the table library** (`/opt/mbx/lib/libmbx_ple_nvme.so`, binary): prepares and serves the table map; it only accepts this
   release's n-gram table and stops with "invalid quant" otherwise,
6. **draft-depth hooks** and the depth library (`/opt/mbx/lib/libmbx_mtp.so`, binary): idle with `mtp_depth.mode: off`.

Digest: `sha256:695882cca3c64ff49d538fd37d72ae7db4537d039ff633da67909648b5ba4676`.

v4 (`…-vllm:v4`, digest `sha256:51629f438f5ba3f7a96db110826c783d69b91a6851c43fb07d447644f157dcc4`), v3 (`…-vllm:v3`, vLLM 0.29,
digest `sha256:61d2bc6ba5977024895734d0ef94918ac806d936c4e17a263906e246599f9862`), v2 and v1 stay available: `git checkout v4` /
`v3` / `v2` / `v1`.

## The full box

This kit serves one model, plain. The same model also runs under [myllmbox](https://myllmbox.com) with a public HTTPS
tunnel, keepalive and multi-model management — same image, same weights.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products
must display the model name; Model-as-a-Service businesses need a separate Qwen license). Kit
scripts and image patches: MIT. The table library is distributed in binary form inside the image.
