<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<p align="center"><a href="https://github.com/dime-online"><img src="docs/assets/hero-banner-v7.png" alt="Qwen3.8 Flash · DGX UltraFast banner with 74 tok/s single stream and 212 tok/s aggregate on one DGX Spark, over cyan and magenta speed rays" width="100%"></a></p>

# Qwen3.8 Flash · DGX UltraFast

A vLLM serving recipe for Qwen3.8-Flash-Next on a single NVIDIA DGX Spark or other GB10 system. The promoted configuration, v16b, decodes a copy-heavy workload at up to **74.1 tok/s** on one stream and **212.2 tok/s** aggregate across eight concurrent streams, among the fastest publicly documented GB10 recipes in that workload class. The speed comes from MTP speculative decoding with a dense drafter and a leaner per-step path, not from lower-bit target weights: the W4A16/FP8 target verifies every drafted token with block rejection, and the served stack scores 93% on a 492-item evaluation suite on two seeds. Every published number ships with its raw per-round data and the script that measured it, and the whole stack builds from public downloads.

Based on [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) (Apache-2.0, (c) blazux).

![License](https://img.shields.io/badge/license-mixed%20Apache%202.0%20%7C%20noncommercial-244d64) ![Hardware](https://img.shields.io/badge/hardware-GB10-244d64) ![Model](https://img.shields.io/badge/model-Qwen3.8--Flash--Next-244d64) ![vLLM](https://img.shields.io/badge/vLLM-0.1.dev20073-244d64) ![CUDA](https://img.shields.io/badge/CUDA-13.0-244d64)

## Results

v16b on one GB10 with the promoted launch settings. Each throughput figure is the fastest of three rounds on a copy-heavy, low-effort workload with a shared 9,600-token cached prefix, counted as completion tokens per second over the window in which all streams are decoding. [Method and per-round data](docs/BENCHMARKS.md#measured-peak-rates).

| Streams | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Peak aggregate decode, tok/s | **74.1** | 110.0 | 132.9 | 155.5 | 175.8 | 191.5 | 205.8 | **212.2** |

<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="docs/assets/throughput-dark.svg"><img src="docs/assets/throughput-light.svg" alt="Measured peak copy-heavy decode throughput at one through eight streams" width="900"></picture></p>

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
| Model residency / available memory | **~71 GiB / 16.5 GiB** | Base recipe residency / v16b capacity-run low-water |

On the agent-shaped workload, v16b spends 23% less time per decode step than the upstream base, or 1.30× as many steps per second. [Benchmark method](docs/BENCHMARKS.md) · [Evaluation](docs/EVALS.md).

## Requirements

- One NVIDIA GB10 with about 121 GB of unified memory, ARM64 Linux, Docker and the NVIDIA container runtime.
- A driver compatible with the pinned CUDA 13.0 image. Check `nvidia-smi` and a GPU container before building.
- Local storage, preferably NVMe, for the checkpoint, FP8 PLE table, downloads and Docker layers (about 130 GB free for the upstream checkpoint and table), plus about 4.8 GiB for the T80 drafter shard data.
- Python 3, the Hugging Face `hf` CLI, `jq`, GNU `md5sum` and `sha256sum`.

## Quick start

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

Follow startup with `docker logs -f qwen38-flash`. The server exposes an OpenAI-compatible API on port 8000 under the model name `qwen`. The T80 builder converts nine drafter modules at group size 32; its reference conversion took 4.626 s after download and wrote 5,118,048,680 new bytes.

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
| [`recipe/config/v16b/`](recipe/config/v16b/) | Promoted launcher, `env`, image tag and draft vocabulary ids |
| [`recipe/build/image/`](recipe/build/image/) | Staged iter6c and iter6d Dockerfiles, patches, hash preflight and CPU tests |
| [`recipe/build/model/`](recipe/build/model/) | T80 dense-MTP g32 checkpoint builder, tests and reference report |
| [`recipe/benchmarks/`](recipe/benchmarks/) | Stream-curve, agent-shaped and standalone decode benchmark scripts |
| [`docs/`](docs/) | [Build](docs/BUILD.md), [configuration](docs/CONFIGURATION.md), [architecture](docs/ARCHITECTURE.md), [benchmarks](docs/BENCHMARKS.md), [evaluation](docs/EVALS.md), [changelog](docs/CHANGELOG.md) and [result tables](docs/results/) |

## Credits and licenses

The base recipe derives from [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound), Apache-2.0, copyright blazux. Thanks to the Qwen model authors and to the vLLM, Intel AutoRound and FlashInfer contributors. Model weights and downloaded datasets keep their own terms.

The upstream-derived `recipe/{Dockerfile,.dockerignore,prepare.sh,src/,tools/,scripts/,build/}`, `recipe/benchmarks/decode_bench.py` and `recipe/config/v16b/{serve.sh,launch.sh}` remain **Apache-2.0**. Original code, where present, uses **PolyForm Noncommercial 1.0.0**; original prose, tables and assets use **CC BY-NC 4.0**. Attribution for original material is **dime-online**. See [LICENSE](LICENSE), [NOTICE](docs/NOTICE) and [license texts](docs/LICENSES/).

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
