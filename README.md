# Qwen3.8-Flash-Next on one DGX Spark

### Two checkpoints: `hibrid48`, the default — and `hibrid48-uncensored`, made from it (abliterated, gated). Switch with one line in `recipe.yaml`

[`hibrid48`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48) (the base model, default, every speed number below)
and [`hibrid48-uncensored`](https://huggingface.co/myllmbox/Qwen3.8-Flash-Next-hibrid48-uncensored) (OrcaRouter's abliterated
body with the same output head, no refusals, no guardrails — gated, research / private use). Same shapes tensor for tensor,
so the same speed. To switch: in `recipe.yaml` comment the active `model:` line and uncomment the other, then `./run.sh`.
Details in [Which checkpoint](#which-checkpoint).

One box, one model, three commands. **v5.1: RecoverSSM + dynamic draft depth up to 7 by default, an
877k-token KV pool, 322 tok/s peak at 16 streams, and an optional front proxy with a loop guard.**

## v5.1 (2026-10-02)

v5 stays available: `git checkout v5`.

**New**

1. **RecoverSSM** for the model's recurrent (GDN) layers and the PLE short conv — [vllm-project/vllm#58863](https://github.com/vllm-project/vllm/pull/58863)
   by [@jschmied](https://github.com/jschmied), ported to 0.30. The drafts are verified from one saved state and only the
   accepted tokens are replayed: faster steps, and no per-draft state copies, so a deeper draft costs no memory. Pool
   **876,726 tokens** at depth 7 (v5: 813,457 at depth 6).
2. **Dynamic draft depth is the default**, up to **7** (v5: opt-in, up to 6). Fixed K=5: `mtp_depth.mode: off` and the two
   commented K=5 lines in `recipe.yaml`.
3. **Optional front proxy** (`proxy:` section in `recipe.yaml`, absent = no proxy): keepalive pings during long prefill and
   thinking, a generic loop guard (any phrase repeated, or `2222…` / `?!?!…` runs), your own pattern list, and optional
   logging of every request and response to `cache/requests/`. A cut answer ends with `<stopped reason="repetition"/>` (or
   `reason="pattern"`) and frees the seat. See [Front proxy](#front-proxy-v51-optional).
4. GB10 plan table for the model's small decode GEMMs, **off by default** (`MBX_SKINNY_GEMM_SM12X: "1"` in `env:`): faster
   kernels, neutral end to end on one Spark. Found and first tuned for two Sparks by
   [@sethforprivacy](https://github.com/sethforprivacy) ([vllm-project/vllm#59605](https://github.com/vllm-project/vllm/issues/59605));
   the table here is tuned for one.

**Measured** (one DGX Spark, image v5.1, the shipped `recipe.yaml`, one `bench/full.py` run)

| | v5.1 |
|---|---|
| thinking-on request (pasture), c=1 | **61.2** tok/s average · **86.2** peak |
| structured output (JSON schema), c=1 | **73.1** |
| long prose (7,000-word story), c=1 | **44.4** |
| peak at c=1 / 2 / 4 / 8 / 16 (thinking off) | **75 / 112 / 167 / 236 / 322** |
| average at c=1 / 2 / 4 / 8 / 16 (thinking off) | 59 / 84 / 128 / 197 / 282 |
| prefill, 128k-token prompt | **2,053** tok/s |
| first token, 1k prompt | **0.68** s |
| KV pool | **876,726** tokens |

**v5 → v5.1 per prompt** (same box, thinking off, v5 = the dynamic-depth runs below, average tok/s of all streams together, averages of 3 runs; c=1 = one
full answer, c≥2 = 300 s with every stream kept busy)

| prompt | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| mixed (code + explanation) | 55.0 → **56.4** | 82.7 → **83.3** | 120.0 → **126.3** | 151.3 → **161.4** | 181.0 → **192.1** | 214.3 → **239.2** |
| structured output (JSON schema) | 69.3 → **74.2** | 105.7 → **111.9** | 150.0 → **162.3** | 178.0 → **198.4** | 201.7 → **228.6** | 233.3 → **267.5** |
| long prose (7,000-word story) | 44.3 → **44.7** | **69.7** → 69.4 | 102.3 → **107.8** | 124.0 → **132.1** | 143.0 → **152.1** | 172.3 → **185.7** |
| code (TypeScript) | 69.7 → **74.6** | 103.0 → **107.3** | 142.3 → **151.3** | 169.0 → **185.7** | 185.3 → **209.9** | 213.3 → **243.9** |

## Dynamic draft depth

**How the dynamic depth works.** Speculative decoding drafts several tokens ahead and the model checks them in one step; how
many of them survive depends on what is being written. Long prose keeps about the first three, code and JSON often all six. A
fixed depth is therefore too deep for prose (every rejected draft is wasted work) and too shallow for code. With
`mode: dynamic` every request keeps its own depth between `min` and `num_speculative_tokens`: it watches, over its last
`window` steps, how often its deepest drafted position was accepted, and moves one or two steps deeper when that rate is above
`promote` and shallower when it is below `demote`. Requests that are batched together draft their average depth. Prose
settles at 3, code and JSON at 5–6, and a thinking-on request changes depth between its reasoning and its answer.

**v5 vs every fixed depth on one full thinking-on request** (the pasture prompt: reasoning, then an HTML/JS scene), each phase on its own, because
answers think for different lengths (averages of 3 runs, tok/s):

| phase | fixed K=3 | fixed K=4 | fixed K=5 | fixed K=6 | **v5 (dynamic)** |
|---|---|---|---|---|---|
| thinking | 50.7 | 50.0 | 49.7 | 48.0 | **51.4** |
| writing the code | 63.2 | 66.3 | 70.9 | 68.8 | **72.1** |

**v5 and v5.1 vs fixed K=3 and K=5 at every concurrency, aggregate tok/s** (same box and image, thinking off, averages of 3 runs; c=1 = one full answer,
c≥2 = 300 s with every stream kept busy; fixed K=3 / K=5 = `mode: dynamic` with `min` pinned and promotion off). The best
value of each row is bold; the mixed prompt is in the table above.

**Long prose (7,000-word story)**

| depth | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| fixed K=3 | 44.0 | **70.0** | 103.3 | 124.3 | 143.0 | 172.0 |
| fixed K=5 | 39.7 | 61.0 | 90.3 | 108.7 | 124.7 | 144.7 |
| **v5 (dynamic)** | 44.3 | 69.7 | 102.3 | 124.0 | 143.0 | 172.3 |
| **v5.1 (dynamic, ≤7)** | **44.7** | 69.4 | **107.8** | **132.1** | **152.1** | **185.7** |

**Code (TypeScript)**

| depth | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| fixed K=3 | 62.3 | 93.0 | 132.3 | 160.3 | 180.3 | 214.3 |
| fixed K=5 | 69.7 | 102.3 | 140.0 | 164.0 | 185.3 | 217.0 |
| **v5 (dynamic)** | 69.7 | 103.0 | 142.3 | 169.0 | 185.3 | 213.3 |
| **v5.1 (dynamic, ≤7)** | **74.6** | **107.3** | **151.3** | **185.7** | **209.9** | **243.9** |

**Structured output (JSON schema)**

| depth | c=1 | c=2 | c=4 | c=6 | c=8 | c=12 |
|---|---|---|---|---|---|---|
| fixed K=3 | 61.7 | 95.3 | 139.3 | 168.3 | 193.0 | 229.7 |
| fixed K=5 | 69.7 | 103.7 | 149.3 | 178.3 | 203.7 | 232.7 |
| **v5 (dynamic)** | 69.3 | 105.7 | 150.0 | 178.0 | 201.7 | 233.3 |
| **v5.1 (dynamic, ≤7)** | **74.2** | **111.9** | **162.3** | **198.4** | **228.6** | **267.5** |

**On by default since v5.1** (deepest depth 7; the tables above are v5, deepest 6). **Fixed K=5 instead**, in
`recipe.yaml`: `mtp_depth.mode: off` and swap in the two commented K=5 lines next to `speculative-config` and
`compilation-config`.

**Knobs** (`mtp_depth` in `recipe.yaml`; `run.sh` writes them to `cache/mbx-depth.json`, and the serve re-reads that file
while it runs — edit it to change them live):

| knob | default | what it does |
|---|---|---|
| `min` | 3 | lowest draft depth |
| `window` | 48 | steps of history each decision looks at (8–128) |
| `promote` | [60, 45] | +2 / +1 depth when the deepest position was accepted on more than this % of the window |
| `demote` | [25, 15] | −1 / −2 depth when below this % |
| `log` | false | `true` = one log line per decision: depth before → after and the measured rate |

The deepest depth is `num_speculative_tokens`; the capture sizes must cover every depth × seats (the shipped list covers
3 … 7 × 16).

## Front proxy (v5.1, optional)

Uncomment the `proxy:` section in `recipe.yaml`. `run.sh` then starts a second container (`qwen38-flash-next-proxy`) on the
public port and moves vLLM to `127.0.0.1:<port+1>`; `./stop.sh` removes both.

| key | default | what it does |
|---|---|---|
| `keepalive` | 30 | seconds between SSE pings while the model is silent (long prefill, long thinking) |
| `loop_guard` | on | any phrase of 3–`loop_period` chars repeated `loop_repeats` times in a row, or a `size`-char 2-char run |
| `size` / `loop_period` / `loop_repeats` | 12 / 64 / 8 | loop guard thresholds |
| `loop_fields` | `reasoning_content,reasoning` | where both guards look; add `content` to watch the answer too |
| `pattern_guard` | off | your strings, one per line in `cache/<patterns_file>`; a hit at `pattern_count` appearances |
| `stop` | false | false = log hits to `cache/logs/`; true = also end the answer with `<stopped reason="…"/>` and free the seat |
| `log_all` | off | every request + full response → `cache/requests/` (holds your users' prompts and answers) |
| `token` | — | optional Bearer token for the API |

## Measured performance (v5, one DGX Spark, `vm.compaction_proactiveness=0`)

**2026-10-01, vLLM 0.30, image v5, `hibrid48`, bf16 KV, 16 seats, Marlin MoE.** Mixed prompt (a code task plus its prose
explanation), thinking off, averages of 3 runs per row. c=1 is one full answer; c≥2 is 300 s with every stream kept busy, so a
finished request is replaced at once and its new prompt's prefill is part of the average.

| concurrent requests | `mode: off` (K=5, = v4) tok/s | **`mode: dynamic`** tok/s | per stream (dynamic) | tokens per step (off → dynamic) |
|---|---|---|---|---|
| 1 | 52.7 | **55.0** | 55.0 | 3.60 → 3.72 |
| 2 | 81.3 | **82.7** | 41.4 | 3.71 → 3.96 |
| 4 | 120.0 | **120.0** | 30.0 | 3.84 → 4.07 |
| 6 | 154.3 | **151.3** | 25.2 | 4.12 → 4.31 |
| 8 | 182.0 | **181.0** | 22.6 | 4.27 → 4.59 |
| 12 | 218.0 | **214.3** | 17.9 | 4.37 → 4.68 |

**Single-stream peak: 86.4 tok/s** — the best 10-second window, while a thinking-on request writes its code. **Prefill:
1,988 tok/s** on a 128k-token prompt, **first token in 0.71 s** on a 1k prompt. The KV pool is 830,582 tokens with `mode: off`
(K=5) and 813,457 with `mode: dynamic` (K=6 graphs); the v4 and v3 ladders are in their tags (`git checkout v4`).

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

**v5.1 (2026-10-02): RecoverSSM, dynamic depth up to 7 by default, optional front proxy.** See
[v5.1](#v51-2026-10-02). v5 stays available: `git checkout v5`.

**v5 (2026-10-01): dynamic draft depth, opt-in.** Per-request draft depth from measured acceptance, live-tunable knobs,
+12 to +19 % on long prose at every concurrency, level on mixed text, JSON and code
([Dynamic draft depth](#dynamic-draft-depth)). With `mode: off` (the default) the serve is v4. v4 stays available:
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
| `myllmbox/Qwen3.8-Flash-Next-hibrid48` (default) | the base model, calibrated body, NVFP4 output head — every speed number above | v5.1: 86 tok/s single-stream peak, 322 tok/s peak at 16 streams |
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

Measured on v5.1: ~1G available at its lowest during a 16-stream run. If you run other things on the box, set
`kv-cache-memory` to 26G (~845k tokens).

One thing you *can* do, with root, and it is worth ~10 % on a serve that runs this close to the memory edge:

```
./tune-host.sh      # sets vm.compaction_proactiveness=0; shows the command, asks, then sudo prompts; persisted
```

`run.sh` reads the value before every launch (no privilege needed) and warns while it is not 0; it never applies the
change itself. The kernel's background page compactor migrates pages to build large contiguous blocks; on a Spark the
GPU's memory *is* those pages, so every migration first unmaps them from the GPU — measured as a 4–5 s slowdown every
~37 s. A serving box allocates once at boot and gains nothing from the upkeep.

## Tuning (recipe.yaml)

- **`kv-cache-memory`** (bytes): 27G bf16 = 876,726 tokens; 26G for more headroom.
- **`max-num-seqs`** 16: with RecoverSSM each request holds one recurrent state (no per-draft copies); 16 streams use ~43 % of
  the pool.
- **`speculative-config`** up to 7 sampled draft tokens with block verification, depth chosen per request
  (`mtp_depth`). Fixed K=5: the commented line.
- **`use-replayssm: true`**: RecoverSSM. Remove it for the stock GDN verify path (more memory per seat, slower steps).
- **`compilation-config`**: the capture sizes cover every draft depth × seats; shorten them only together with
  `max-num-seqs`, or the upper rungs decode without CUDA graphs.
- **`block-size: 1632`**: required at K=5 — the boot stops with "QSA ring capacity 12 must divide the attention block size 1616"
  without it.
- **`docker_args: --security-opt seccomp=unconfined`**: required by the image's table reader; the container still runs
  unprivileged otherwise.
- **`load-format: fastsafetensors`**: weights in ~80 s. Remove it for vLLM's default loader (~10 min).
- **`max-num-batched-tokens`**: also the image-input encoder budget — 8192 fits one max-resolution image (~4.1k tokens).
- **`async-scheduling` on**. Thinking is ON by default (model native); disable per request with
  `"chat_template_kwargs": {"enable_thinking": false}` for max speed on structured output.
- **`mtp_depth`**: dynamic draft depth, on by default — see [Dynamic draft depth](#dynamic-draft-depth).
- **`proxy`**: optional front proxy, absent by default — see [Front proxy](#front-proxy-v51-optional).
- **`patches`** (server): optional vLLM patches from [`patches/`](patches/), off by default — e.g. `patches: hermes-chat`
  for the Hermes agent (contributed by [@yume-arasaki](https://github.com/yume-arasaki)). Applied at launch over the image's
  files; the image itself is unchanged.

## What's in the image

`myllmbox/qwen38-flash-next-vllm:v5.1` — upstream `vllm/vllm-openai:v0.30.0` plus patches, each an anchored or sha256-checked
edit that refuses to apply twice and fails the build if its target moved:

1. **the NVFP4 n-gram table** on 0.30's embedding plugin — stock 0.30 refuses this checkpoint's table,
2. **the 4-bit output head** — stock 0.30 builds both output heads without the checkpoint's quantization config,
3. **fused multi-step draft metadata** — the code proposed upstream as
   [vllm-project/vllm#58449](https://github.com/vllm-project/vllm/pull/58449),
4. the loader's page-cache drop, the QSA pre-indexer rope clamp, and two inert knobs,
5. **the table library** (`/opt/mbx/lib/libmbx_ple_nvme.so`, binary): prepares and serves the table map; it only accepts this
   release's n-gram table and stops with "invalid quant" otherwise,
6. **draft-depth hooks** and the depth library (`/opt/mbx/lib/libmbx_mtp.so`, binary): idle with `mtp_depth.mode: off`,
7. **RecoverSSM** ([vllm-project/vllm#58863](https://github.com/vllm-project/vllm/pull/58863), ported to 0.30, base files
   sha256-checked), used with `use-replayssm`,
8. **the front proxy** (`mbx_proxy`, compiled), started only with a `proxy:` section,
9. a GB10 plan table for the small decode GEMMs (after [@sethforprivacy](https://github.com/sethforprivacy)'s TP=2 table), off unless
   `MBX_SKINNY_GEMM_SM12X=1`.

Digest: `sha256:733f1a576e7e5a4192b0475ac4c3c53b5e5f27a182e92de0972ce88853ce1edd`.

v5 (`…-vllm:v5`, digest `sha256:695882cca3c64ff49d538fd37d72ae7db4537d039ff633da67909648b5ba4676`), v4 (`…-vllm:v4`, digest `sha256:51629f438f5ba3f7a96db110826c783d69b91a6851c43fb07d447644f157dcc4`), v3 (`…-vllm:v3`, vLLM 0.29,
digest `sha256:61d2bc6ba5977024895734d0ef94918ac806d936c4e17a263906e246599f9862`), v2 and v1 stay available: `git checkout v5` /
`v4` / `v3` / `v2` / `v1`.

## The full box

This kit serves one model, plain. The same model also runs under [myllmbox](https://myllmbox.com) with a public HTTPS
tunnel, keepalive and multi-model management — same image, same weights.

## License

Weights: Qwen Community License 1.0 (permissive incl. commercial; >100M MAU/$20M-revenue products
must display the model name; Model-as-a-Service businesses need a separate Qwen license). Kit
scripts and image patches: MIT. The table library is distributed in binary form inside the image.
