<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Architecture and build provenance

The source snapshot's base [Dockerfile](../Dockerfile) starts from the digest-pinned Qwen preview vLLM image and adds PLE mmap, fp8 hybrid dispatch, prefix-cache support, model-loading changes and guard patches. The fp8 PLE table is read through mmap; the 16 GB KV pool is sized explicitly in the promoted launcher. The base build is useful on GB10 and remains Apache-2.0.

The measured v16b composition follows successive image layers: a reduced MTP draft vocabulary and fp8 drafter experts, a short-convolution H2D backport, a low-latency SM12x GEMM, a sort-free verify-path top-k implementation, retained draft prefix-cache blocks and an R7 logger fix. The measured checkpoint uses the T80 dense-MTP g32 drafter head. The staged Dockerfiles and T80 builder are in [BUILD.md](BUILD.md). `config/v16b/image` names the image tag.

| Component | Purpose | Location |
|---|---|---|
| Base AutoRound / fp8 / PLE-mmap recipe | Fit model and KV together | Root Dockerfile and source tree |
| Promoted launcher and 65,536 draft ids | Set runtime arguments | `config/v16b/`; compressed ids expand on launch |
| T80 dense-MTP g32 checkpoint transform | Lower MTP draft cost | Builder, CPU tests, pinned input hashes and reference report |
| Reduced-vocabulary and fp8-drafter image lineage | Lower draft cost | Clone modules, Dockerfile and hash preflight |
| Iteration-6d image layers | Low-latency GEMM, top-k path, cache behavior | Iter6c and iter6d Dockerfiles with every `COPY` input |

## Build and runtime path

`config/v16b/launch.sh --print` displays the promoted command with portable host paths. `--run` checks the fixed image tag. It uses the T80 checkpoint and PLE table at the paths in `env`. The T80 directory and iter6d image build paths start from public downloads. The staged image build uses the iter6c and iter6d Dockerfiles.

## Quality record

The v16b verifier determines emitted tokens after MTP proposes draft tokens. The two-seed evaluation scored 458/492 and 459/492, and the teacher-forced prefill comparison measured a top-1 agreement delta of −0.06 percentage points. See [EVALS.md](EVALS.md).
