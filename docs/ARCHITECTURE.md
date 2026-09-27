<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Architecture and build provenance

The source snapshot's base [Dockerfile](../Dockerfile) starts from the digest-pinned Qwen preview vLLM image and adds PLE mmap, fp8 hybrid dispatch, prefix-cache support, model-loading changes and guard patches. The fp8 PLE table is read through mmap; the 16 GB KV pool is sized explicitly in the promoted launcher. The base build is useful on GB10 and remains Apache-2.0.

The measured v16b composition was **not** that base image. It was built through successive local images: a reduced MTP draft vocabulary and fp8 drafter experts, a short-convolution H2D backport, a low-latency SM12x GEMM, a sort-free verify-path top-k implementation, retained draft prefix-cache blocks and an R7 logger fix. The measured checkpoint also used the T80 dense-MTP g32 drafter head. The staged Dockerfiles and T80 builder are now in [BUILD.md](BUILD.md). `config/v16b/image` names the measured image tag.

| Component | Purpose | Available here? |
|---|---|---|
| Base AutoRound / fp8 / PLE-mmap recipe | Fit model and KV together | Yes: root Dockerfile and source tree |
| Promoted launcher and 65,536 draft ids | Reproduce runtime arguments | Yes: `config/v16b/`; compressed ids expand on launch |
| T80 dense-MTP g32 checkpoint transform | Lower MTP draft cost | Yes: builder, CPU tests, pinned input hashes and reference report |
| Reduced-vocabulary and fp8-drafter image lineage | Lower draft cost | Yes: exact clone modules, Dockerfile and hash preflight |
| Iteration-6d image layers | Low-latency GEMM, top-k path, cache behavior | Yes: iter6c and iter6d Dockerfiles with every `COPY` input |

## Verification scope

`config/v16b/launch.sh --print` exposes the promoted command with portable host paths. `--run` checks for the fixed image tag before touching an existing container. It requires the T80 checkpoint and PLE table at the paths in `env`. The T80 directory and iter6d image now have complete source build paths from public downloads. Rebuilding only the root Dockerfile and assigning it the promoted tag would produce a different engine.

A clean GB10 image build, launch, smoke and quality check still need to be performed; the Windows source checkout cannot provide GPU acceptance. The public 65,536-id list is included, but its private generating corpus is not supplied. Exact benchmark prompts are also private and excluded.

## Correctness scope

The original report distinguished target-side changes from drafter-side changes. MTP proposal cost can change without necessarily changing the verifier's target distribution. The v16 target-side calibrated splice moved teacher-forced agreement; v16 was reported but not adopted by this program. The v16b verify-path top-k implementation was tested for measured equivalence, not bit identity; retaining draft blocks changes floating-point association across cache reuse. The two-seed evaluation saw no detectable regression at its resolution. See [EVALS.md](EVALS.md).
