<!-- SPDX-License-Identifier: Apache-2.0 -->
<p align="center"><a href="https://github.com/dime-online"><img src="docs/assets/hero-banner-v7.png" alt="Qwen3.8 Flash · DGX UltraFast banner with 74 tok/s single stream and 212 tok/s aggregate on one DGX Spark, over cyan and magenta speed rays" width="100%"></a></p>

# Qwen3.8 Flash · DGX UltraFast

A vLLM serving recipe for Qwen3.8-Flash-Next on a single NVIDIA DGX Spark or other GB10 system. The promoted configuration, v16b, decodes a copy-heavy workload at up to **74.1 tok/s** on one stream and **212.2 tok/s** aggregate across eight concurrent streams, among the fastest publicly documented GB10 recipes in that workload class. The speed comes from MTP speculative decoding with a dense drafter and a leaner per-step path, not from lower-bit target weights: the W4A16/FP8 target verifies every drafted token with block rejection, the served stack scores 93% on a 492-item evaluation suite on two seeds, and its teacher-forced agreement with the original checkpoint sits inside the measurement noise band. Every published number ships with its raw per-round data and the script that measured it, and the whole stack builds from public downloads.

Based on [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) (Apache-2.0, (c) blazux).

![License](https://img.shields.io/badge/license-Apache%202.0-244d64) ![Hardware](https://img.shields.io/badge/hardware-GB10-244d64) ![Model](https://img.shields.io/badge/model-Qwen3.8--Flash--Next-244d64) ![vLLM](https://img.shields.io/badge/vLLM-0.1.dev20073-244d64) ![CUDA](https://img.shields.io/badge/CUDA-13.0-244d64)

## Results

v16b on one GB10 with the promoted launch settings. Each throughput figure is the fastest of three rounds on a copy-heavy, low-effort workload with a shared 9,600-token cached prefix, counted as completion tokens per second over the window in which all streams are decoding. [Method and per-round data](docs/BENCHMARKS.md#measured-peak-rates).

| Streams | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Peak aggregate decode, tok/s | **74.1** | 110.0 | 132.9 | 155.5 | 175.8 | 191.5 | 205.8 | **212.2** |

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/throughput-dark.svg"><img src="docs/assets/throughput-light.svg" alt="Measured peak copy-heavy decode throughput at one through eight streams" width="900"></picture></p>

Cold prefill, with nothing cached, is 2× to 3.4× faster than the original recipe on the same GB10. Each rate is prompt tokens divided by cold time to first token, three repeats per cell, so it slightly understates pure prefill speed. [Method and data](docs/BENCHMARKS.md#cold-prefill).

| Cold prompt | 16k | 64k | 128k |
|---|---:|---:|---:|
| v16b prefill, tok/s | **4,016** | **2,426** | **2,213** |
| Original recipe prefill, tok/s | 1,171 | 1,071 | 1,065 |
| Increase | **+243%** | **+127%** | **+108%** |

| Measure | v16b | Conditions |
|---|---:|---|
| Decode step time | **52.33 ms** | Agent-shaped coding workload with thinking and tools; upstream base: 68.276 ms |
| Emitted tokens per target step | **3.69** | Median, copy-heavy workload, one stream |
| Time to first token, warm | **0.57 s** | Cached coding prompt |
| Time to first token, cold | **27.0 s** | 64k-token prompt, three repeats |
| Fixed-suite evaluation | **458/492** and **459/492** | 93.09% and 93.29%; seeds 20260908 and 20260910 |
| Long generation | **21/24** and **22/24** | Two readings, 12,000-token cap |
| Teacher-forced top-1 agreement delta | **−0.06 pp** | Pre-registered control band: 0.15 pp |
| KV pool / configured context | **16 GB / 262,144 tokens** | Promoted launch settings |
| Input | **Text, image and video** | The vision encoder is kept; checked with image and video requests on v16b |
| Model residency / available memory | **~71 GiB / 16.5 GiB** | Base recipe residency / v16b capacity-run low-water |

On the agent-shaped workload, v16b spends 23% less time per decode step than the upstream base, or 1.30× as many steps per second. [Benchmark method](docs/BENCHMARKS.md) · [Evaluation](docs/EVALS.md).

## Requirements

- One NVIDIA GB10 with about 121 GB of unified memory, ARM64 Linux, Docker and the NVIDIA container runtime.
- A driver compatible with the pinned CUDA 13.0 image. Check `nvidia-smi` and a GPU container before building.
- Local storage, preferably NVMe, for the checkpoint, FP8 PLE table, downloads and Docker layers (about 130 GB free for the upstream checkpoint and table), plus about 4.8 GiB for the T80 drafter shard data.
- Python 3, the Hugging Face `hf` CLI, `jq`, GNU `md5sum` and `sha256sum`.

## Quick start

### With a coding agent

Run a coding agent that can use a terminal on your everyday computer, and paste this prompt. The agent connects to your GB10 over SSH and does the setup there. Set up key-based SSH login to the GB10 first.

```text
Set up the Qwen3.8 Flash DGX UltraFast recipe on my GB10 over SSH.
Read https://raw.githubusercontent.com/dime-online/qwen3.8-Flash-DGX-UltraFast/main/AGENTS.md and follow it exactly.
Ask me how to reach the GB10, run the read-only checks, show me what you found, and get my approval before you install, download, build or launch anything.
```

The agent first checks the GB10's hardware, memory, disk, Docker and port 8000 without changing anything. It then asks for your approval before each of five steps: getting the repository and tools, downloading about 130 GB of public model files, building the image, building the drafter directory, and launching the server. Long steps run in the background on the GB10, so a dropped connection doesn't interrupt them. The agent never stops or removes anything that was already running. [AGENTS.md](AGENTS.md) has the full instructions.

### Manual

Download the pinned public checkpoint and PLE table, build the image and the T80 drafter directory, then launch v16b and send a test request. [BUILD.md](docs/BUILD.md) has the reference hashes, build report and what each step checks.

```bash
# Clone and install the Hugging Face CLI
git clone https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast.git
cd qwen3.8-Flash-DGX-UltraFast
python3 -m venv recipe/.venv
. recipe/.venv/bin/activate
pip install --upgrade huggingface_hub

# Download the pinned checkpoint and FP8 PLE table
mkdir -p "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid" "$HOME/models/ple-table-fp8"
hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid \
  --revision 8b82f0b7abe3d1150a7827d298c75e86267636ae \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid"
hf download Saren/Qwen3.8-Flash-Next-ple-table-fp8 \
  --revision 50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --local-dir "$HOME/models/ple-table-fp8"

# Build the image, then the T80 dense-MTP drafter directory
bash recipe/build/image/build.sh --run
bash recipe/build/model/build.sh --run

# Show the full docker command, then start the server (runs detached)
bash recipe/config/v16b/launch.sh --print
bash recipe/config/v16b/launch.sh --run

# Wait for the server to load, then send a test request
until curl -fsS http://localhost:8000/v1/models >/dev/null 2>&1; do sleep 10; done
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":512}'
```

Follow startup with `docker logs -f qwen38-flash`. The server exposes an OpenAI-compatible API on port 8000 under the model name `qwen`. It accepts text, images (`image_url`) and video (`video_url`), since the recipe keeps the model's vision encoder. Audio input is not supported. The T80 builder converts nine drafter modules at group size 32; its reference conversion took 4.626 s after download and wrote 5,118,048,680 new bytes.

> **Running on your own GB10?** If the recipe helped, a ⭐ helps other Spark owners find it, and your numbers are welcome as a [benchmark submission](https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast/issues/new?template=benchmark_submission.md).

## Configuration

The promoted launch pins `GPU_MEM=0.01`, `KV_BYTES=16g`, `SEQS=8`, `CTX=262144`, MTP depth 3, block rejection with probabilistic draft sampling, and prefix caching. `launch.sh` sources [`recipe/config/v16b/env`](recipe/config/v16b/env) under a cleared environment, so edit that file rather than exporting variables. Changing any pinned value produces a new variant whose speed and quality must be measured separately. [Configuration reference](docs/CONFIGURATION.md).

## How it works

```text
AutoRound INT4 experts + FP8 side layers + INT8 output head
             │
             ├── FP8 PLE table, read by mmap from storage
             ├── MTP depth 3, 65,536-id draft vocabulary, dense drafter head
             └── fast PLE gather + piecewise CUDA graphs + prefix cache
                         ↓
                vLLM verifier and sampler → response
```

**Target precision.** AutoRound W4A16 packs the routed-expert weights to four bits with 16-bit activations. FP8 side layers and an INT8 output head keep the nonexpert paths at eight bits, where 3-bit GGUF and NVFP4 recipes compress target weights further. The FP8 PLE table is memory-mapped from storage rather than held entirely in GPU memory, which leaves room for the 16 GB KV pool.

**Drafting.** The dense T80 MTP drafter proposes three tokens per step. A 65,536-id draft vocabulary and FP8 drafter experts make each proposal cheaper. The target verifies every drafted token with block rejection, so the drafter changes speed, not the output distribution. Unlike n-gram copy speculation, which needs a matching earlier span, the MTP head drafts from the model itself and also covers original code, reasoning and tool output.

**Per-step overhead.** A low-latency SM12x GEMM, a sort-free verify-path top-k and piecewise CUDA graphs trim compute, selection and launch overhead around each target pass. The image runs vLLM's V2 model runner, where async scheduling is on by default with MTP, and adds an asynchronous short-convolution host-to-device transfer, guarded recurrent-state alignment and retained draft prefix-cache blocks.

[Architecture and build provenance](docs/ARCHITECTURE.md).

## Benchmarks and evaluation

- **Stream curve.** Three rounds at each of 1 to 8 streams on the copy-heavy workload, reported as peak and median, with tokens per step and time to first token. The five-stream point was measured in two sessions. [Runner](recipe/benchmarks/bench_copy_streams.py) · [results](docs/BENCHMARKS.md#measured-peak-rates) · [raw records](docs/results/).
- **Agent-shaped decode.** Cached long system prefix, tool schemas, six coding tasks, three repeats each, 1,500 generated tokens, thinking on. [Step-time comparison](docs/BENCHMARKS.md#agent-shaped-decode).
- **Quality.** A fixed 492-item suite (code with execution checks, math, knowledge, instruction following, tool calls, long-context needles), a 24-item long-generation arm and a teacher-forced prefill comparison. [Evaluation](docs/EVALS.md).

## Repository layout

| Path | Contents |
|---|---|
| [`AGENTS.md`](AGENTS.md) | Step-by-step instructions for a coding agent to set up the recipe on a GB10 over SSH, with your approval at each step |
| [`recipe/config/v16b/`](recipe/config/v16b/) | Promoted launcher, `env`, image tag and draft vocabulary ids |
| [`recipe/build/image/`](recipe/build/image/) | Staged iter6c and iter6d Dockerfiles, patches, hash preflight and CPU tests |
| [`recipe/build/model/`](recipe/build/model/) | T80 dense-MTP g32 checkpoint builder, tests and reference report |
| [`recipe/benchmarks/`](recipe/benchmarks/) | Stream-curve, agent-shaped and standalone decode benchmark scripts |
| [`docs/`](docs/) | [Build](docs/BUILD.md), [configuration](docs/CONFIGURATION.md), [architecture](docs/ARCHITECTURE.md), [benchmarks](docs/BENCHMARKS.md), [evaluation](docs/EVALS.md), [changelog](docs/CHANGELOG.md) and [result tables](docs/results/) |

## magi-v3 production fork

This branch (`magi-v3`) is the recipe above **plus** the PLE-over-RDMA port
from the `magi` line, serving production on a GB10 with wtako holding the
table. Everything upstream still applies; the deltas are:

- **PLE table over RDMA (exclusive).** `recipe/build/image/src/vllm_ple_rdma.py`
  (+ `libple_rdma.so`, built in `Dockerfile.iter6d`) fetches rows with
  one-sided READs from `ple-rdma-server` on wtako (`VLLM_PLE_RDMA`, default
  `192.168.0.1:18515`). With it set, no table is mmapped locally, failed
  READs retry/stall, and the mmap knobs (`PREWARM`, `PREFETCH`, `FAST_PATH`)
  are ignored. Same hook architecture as the mmap patch (same class, same
  `vllm::ple_mmap_lookup` op, same `prepare_inputs` prefetch pattern), so the
  rendezvous anchors and CPU gates are unaffected.
- **Two serving toggles.** `T80` (default 1: `...-hybrid-mtpdense-g32`, else
  the base hybrid; explicit `MODEL_DIR` wins) and `DRAFT_VOCAB` (default 1:
  the 65,536-id slice; `0` scores the full 248,320-id head — production runs
  `0` for CJK draft acceptance at ~458 MiB extra head read per draft pass).
- **Watchdog entrypoint.** `serve-magi.sh` (called by the supervisor's
  `ram-client-ple.sh`) waits for `:18515`, pins the production image,
  RDMA-exclusive flags, `DRAFT_VOCAB=0`, KV bytes and the never-evict pin
  prompt, then execs the recipe serve script.
- **Combined T80+int4 checkpoint.** Production serves
  `...-hybrid-mtpdense-g32-mtpint4`: the T80 g32 sides plus magi's int4 g128
  routed experts. Requires one routing fix in its `config.json`: an explicit
  `+:.*\.mlp\.experts$` int4 rule ahead of T80's `-:.*\bmtp\..*` guard,
  otherwise the experts fall into the unquantized path and the fp8 arm builds
  the wrong parameters (`AttributeError: no parameter 'w2_qweight'`).

The published benchmark numbers above are upstream v16b (mmap table, 65k
cut). magi-v3 production differs (RDMA table, full head, combined
checkpoint), so re-measure with `recipe/benchmarks/` before comparing.

## Credits and licenses

The base recipe derives from [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound), Apache-2.0, copyright blazux. Thanks to the Qwen model authors and to the vLLM, Intel AutoRound and FlashInfer contributors. Model weights and downloaded datasets keep their own terms.

Everything in this repository, code, docs, data and images, is licensed under **Apache-2.0**, the same license as the upstream recipe. You can use, modify and redistribute it, including commercially. If you redistribute it, keep the [NOTICE](NOTICE) file, which credits **dime-online** and the upstream authors. See [LICENSE](LICENSE).

## Citation

GitHub's "Cite this repository" menu reads [CITATION.cff](CITATION.cff).

```bibtex
@software{dime_online_qwen38_gb10_v16b,
  title = {Qwen3.8 Flash · DGX UltraFast: v16b serving recipe},
  author = {{dime-online}},
  year = {2026},
  url = {https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast}
}
```

## Draft vocabulary note

The MTP drafter proposes tokens from a bundled 65,536-id draft vocabulary built from English and code text. Output in other languages, notably CJK, gets lower draft acceptance and therefore lower decode speed; upstream reports the same effect for its own English/code-weighted set. Output quality is unaffected, because the target model verifies every drafted token with block rejection. See [Configuration](docs/CONFIGURATION.md#draft-vocabulary) for details.
