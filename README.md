<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
<p align="center"><img src="assets/hero.png" alt="Compact unbranded desktop compute system" width="100%"></p>

# Qwen3.8 Flash · DGX UltraFast

A measured vLLM serving configuration for Qwen3.8-Flash-Next on one GB10, with an explicit quality record.

Based on [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound) (Apache-2.0, (c) blazux).

**Single stream: up to 73.2 tok/s peak; 69.0 tok/s median** (copy-heavy, low-effort requests). **Aggregate: up to 146.3 tok/s peak** (five equal-length streams); **97.1 tok/s median** (three separate four-stream agent-shaped probes). These workloads differ; see [measurement scope](docs/BENCHMARKS.md#measured-peak-rates).

![License](https://img.shields.io/badge/license-mixed%20Apache%202.0%20%7C%20noncommercial-244d64) ![Hardware](https://img.shields.io/badge/hardware-GB10-244d64) ![Model](https://img.shields.io/badge/model-Qwen3.8--Flash--Next-244d64) ![vLLM](https://img.shields.io/badge/vLLM-0.1.dev20073-244d64) ![CUDA](https://img.shields.io/badge/CUDA-13.0-244d64) ![Single-stream peak / median](https://img.shields.io/badge/single%20peak%20%2F%20median-73.2%20%2F%2069.0%20tok%2Fs-244d64) ![Aggregate peak / median](https://img.shields.io/badge/aggregate%20peak%20%2F%20median-146.3%20%2F%2097.1%20tok%2Fs-244d64)

This repository includes the promoted v16b configuration, measurements, the T80 dense-MTP g32 builder and the complete staged iter6c-to-iter6d image build inputs. [BUILD.md](docs/BUILD.md) gives the clean GB10 path and the checks performed here. The historical binary image is not distributed.

## Results at a glance

| Measure | v16b reading | Scope |
|---|---:|---|
| Single-stream decode | **73.2 tok/s peak; 69.0 tok/s median** | One copy-heavy, low-effort request; median of nine such requests in the v16b control |
| Decode step | **52.33 ms** | Candidate interleave; bridge to the four-recipe window is about 52.3 ms |
| Aggregate decode | **146.3 tok/s peak; 97.1 tok/s median** | Peak: five equal-length 600-token streams; median: three four-stream agent-shaped probes |
| Measured aggregate, 1 / 2 / 3 / 4 / 5 equal-length streams | **51.1 / 73.8 / 115.1 / 129.1 / 146.3 tok/s** | One probe per cell; the one-stream cell failed the occupancy gate |
| Warm time to first token | **0.57 s** | Cached, agent-shaped prompt |
| Cold time to first token at 64k | **27.0 s** | Long-context bench; not comparable to warm TTFT |
| KV pool / configured context | **16 GB / 262,144 tokens** | Maximum context is a setting, not a guarantee of equal speed at that length |
| Memory | **~71 GiB** model residency in the base recipe; **16.5 GiB** available low-water in a v16b-class capacity run | Different measurements; no single peak-used figure was recorded |
| Evaluation | **459/492** at seed 20260910; **458/492** at seed 20260908 | Same fixed suite, different sampling seeds; pooled 917/984 |
| Long generation | **21/24**, then **22/24** | Two readings; neither proves equivalence |

The historical 50 / 74 / 94 / 101 / 107 tok/s stream series is a computed planning estimate and is excluded from the peak selection. The copy-heavy single-request peak is not a typical coding-agent rate: the v16b agent-shaped median was 48.8 tok/s across 18 requests. The five-stream peak has no repeat median at that exact workload. An earlier code-path review found that an upstream strided-state-index defect did not apply to this image; later research re-raised the concern from a missing patch symbol, and an agreement check with four requests in flight remains outstanding. Treat concurrent figures as descriptive. [Full method and raw-data mapping](docs/BENCHMARKS.md).

<p align="center"><img src="assets/throughput.svg" alt="Computed planning estimates and measured probe throughput by stream count" width="720"></p>

## Requirements

- A single NVIDIA GB10 system with roughly 121 GB unified memory, ARM64 Linux, Docker and the NVIDIA container runtime.
- A driver compatible with the pinned CUDA 13.0 base image. Check `nvidia-smi` and a simple GPU container before building. The source records no tested minimum driver, so none is asserted here.
- Fast local storage, preferably NVMe. The upstream recipe calls for about 130 GB free for its checkpoint and fp8 PLE table; the T80 variant adds about 4.8 GiB of new shard data. Budget further space for downloads and Docker layers; their total peak was not recorded.
- Python 3, the Hugging Face `hf` CLI, and `jq` for the promoted `SPEC_EXTRA` merge. Building the checkpoint yourself with `prepare.sh` also needs PyTorch and safetensors. Confirm sufficient **available** memory before a build and before a multi-stream launch.
- The pinned base image in the [iter6c Dockerfile](build/image/Dockerfile.iter6c), whose source records vLLM 0.1.dev20073 and CUDA 13.0. Keep its digest fixed when comparing measurements.

## Quick start: download, build, launch, test

The sequence below downloads the public checkpoints, builds the pinned image lineage and T80 drafter directory, then launches v16b. The image script checks every source input and the release source hashes before pulling. This end-to-end path has not yet been run on a clean GB10; [BUILD.md](docs/BUILD.md) records the checks and expected results.

```bash
git clone https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast.git
cd qwen3.8-Flash-DGX-UltraFast
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade huggingface_hub
mkdir -p "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid" "$HOME/models/ple-table-fp8"
hf download Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid \
  --revision 8b82f0b7abe3d1150a7827d298c75e86267636ae \
  --local-dir "$HOME/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid"
hf download Saren/Qwen3.8-Flash-Next-ple-table-fp8 \
  --revision 50511b0a41aa1d34b8beb7e5d4bb06a0b650dc14 \
  --local-dir "$HOME/models/ple-table-fp8"
bash build/image/build.sh --run
bash build/model/build.sh --run
bash config/v16b/launch.sh --print
bash config/v16b/launch.sh --run
curl -fsS http://localhost:8000/v1/models
curl -fsS http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":512}'
```

The T80 builder uses the public base checkpoint and converts nine drafter modules at group size 32 without target-side conversion. Its reference conversion took 4.626 seconds after download and produced 5,118,048,680 new bytes; the wrapper validates the input hashes and output report. The promoted launch pins `GPU_MEM=0.01`, `KV_BYTES=16g`, `SEQS=8`, `CTX=262144`, MTP 3 and prefix caching. Check the model licenses and available memory before downloading and launching. [Full build procedure](docs/BUILD.md).

## How the speed is obtained

```text
AutoRound int4 experts + fp8 side layers + int8 output head
             │
             ├── fp8 PLE table, read by mmap from storage
             ├── MTP depth 3, 65,536-id draft vocabulary, dense drafter head
             └── fast PLE gather + split CUDA graphs + prefix cache
                         ↓
                vLLM verifier and sampler → response
```

The v16b image also used a low-latency GEMM, verify-path top-k kernel and draft-block retention. Their Docker build lineage is in [BUILD.md](docs/BUILD.md). The reduced vocabulary and fp8 drafter experts improve draft cost; the verifier still determines emitted tokens. The verify-path kernel and cache change are measured-equivalent or distribution-preserving claims, not bit identity. [Architecture details](docs/ARCHITECTURE.md).

## Configuration

The promoted [env file](config/v16b/env) and [serve script](config/v16b/serve.sh) pin the measurement: KV `16g`, `SEQS=8`, chunked prefill `8192`, context `262144`, GPU fraction `0.01`, MTP `3`, prefix cache on, fp8 hybrid on, and the four iteration-6 switches. The launcher clears the caller environment before sourcing `env`. The full knob and flag reference, including safe limits and memory costs, is in [CONFIGURATION.md](docs/CONFIGURATION.md). Keep KV, sequence count, batched-token rail and GPU fraction at or below the recorded settings unless you measure the resulting memory demand; a 20 GB KV experiment consumed about 3.7 GiB more available memory, and a 16k prefill chunk triggered a driver out-of-memory warning.

## Benchmarks and quality

The single-stream benchmark used a long cached system prefix, tool schemas, six coding tasks, three repeats and 1,500 generated tokens, with thinking enabled. That is what **agent-shaped coding traffic** means here. Step time is the main comparison metric because draft acceptance changes tok/s. A gain counted only beyond `max(2 tok/s, 1.5 × anchor spread)` and with a consistent sign across interleaves. Measurements come from one box, with container age and thermal state as possible sources of drift. [Tables and caveats](docs/BENCHMARKS.md).

The fixed 492-item suite spans code, math, knowledge, instruction following, tool calls and long-context needles. Both v16b seeds landed in the family band, and the long-generation arm did not show a new loop or cap issue. That supports “no difference detected at this suite's resolution,” not a guarantee of unchanged answers. Teacher-forced prefill agreement matched the measured control floor on its discriminating clauses; one raw log-likelihood clause was undecidable. [Scorer, seeds and limitations](docs/EVALS.md).

## Comparison with other public GB10 recipes

Public recipes use different models or checkpoints, prompt shapes, temperatures, output lengths and engines. For example, [MiaAI-Lab's single-Spark NVFP4 recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) reports 48.7 tok/s single stream on a short prose prompt at temperature zero, while [the EXL3 recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe) reports 58.8 tok/s using a different quantization and native engine. Neither is a like-for-like race against this cached coding workload. The widely repeated “80 tok/s on one Spark” number was not verified as single-stream output for this model in the supplied research. [Comparison notes](docs/BENCHMARKS.md#public-comparisons).

## Optional and experimental

`--long-prefill-token-threshold 1024` cut an interactive request's TTFT during a 64k prefill from 31.40 s to 1.43 s, while cold 64k prefill TTFT rose 12.8%. It is an optional latency trade and is off in v16b. An experimental FlashInfer GDN prefill overlay reduced cold 32k/64k TTFT by about 7% in one incomplete pair, but a teacher-forced agreement test found a small numerical difference. It is **not adopted**. [Experimental records](docs/EXPERIMENTS.md).

| Lever tried | Result | Why closed |
|---|---|---|
| Increase KV pool to 20 GB | Decode step stayed inside the null band; four-stream aggregate fell 1.5% | Capacity was not the bottleneck; available memory fell |
| Raise prefill chunk to 16k | Driver out-of-memory during four cold 64k requests | Unsafe on the tested box |
| MTP depth 4 | Extra verification cost exceeded the measured acceptance benefit | No agent-window break-even |
| Target-side calibrated splice (v16) | Faster; full v16 recipe moved teacher-forced top-1 agreement −2.58 points, T85 alone about −2.53 | Reported, not adopted |

[More closed levers and source references](docs/EXPERIMENTS.md).

## Troubleshooting and FAQ

If the image cannot be found, run the image build in BUILD.md and inspect its first failed gate; do not retag the base build as v16b. If the model fails to load, check model/table paths, Docker GPU access, free unified memory, the exact container tag and `docker logs`. If TTFT rises after a cache miss, inspect PLE storage and prefix-cache metrics. If `SPEC_EXTRA` fails, install `jq` and validate its JSON object. [Detailed checks](docs/TROUBLESHOOTING.md).

## Roadmap

See the [changelog](CHANGELOG.md) for the recipe version history.

1. Verify the complete image and T80 builds, printed argv, serving behavior and quality on a clean GB10.
2. Reconcile the two state-index readings, check agreement with four requests in flight, and rerun the equal-length stream probe.
3. Repeat quality and performance on another GB10, including thermal and storage variance.

## Credits and licenses

The base recipe derives from [Saren-Arterius/qwen3.8-Flash-DGX-AutoRound](https://github.com/Saren-Arterius/qwen3.8-Flash-DGX-AutoRound), Apache-2.0, copyright blazux. Thanks to the Qwen model authors, vLLM, Intel AutoRound and FlashInfer contributors. The model weights and downloaded datasets are not included and retain their own terms.

The upstream-derived `Dockerfile`, `.dockerignore`, `prepare.sh`, `src/`, `tools/`, `scripts/`, `benchmarks/decode_bench.py`, `build/` and `config/v16b/{serve.sh,launch.sh}` remain **Apache-2.0**. Original code, where present, uses **PolyForm Noncommercial 1.0.0**; original prose, tables and assets use **CC BY-NC 4.0**. Attribution for original material is **dime-online**. See [LICENSE](LICENSE), [NOTICE](NOTICE) and [license texts](LICENSES/). These noncommercial licenses do not restrict the Apache-2.0 portions.

## Citation

```bibtex
@software{dime_online_qwen38_gb10_v16b,
  title = {Qwen3.8 Flash · DGX UltraFast: v16b serving recipe},
  author = {{dime-online}},
  year = {2026},
  url = {https://github.com/dime-online/qwen3.8-Flash-DGX-UltraFast}
}
```
