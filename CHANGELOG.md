<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Qwen3.8 Flash · DGX UltraFast — changelog

## v16b — promoted configuration

Dense MTP drafter head (T80), iteration-6d image, low-latency GEMM, verify-path top-k and retained draft blocks. Direct candidate-window step 52.33 ms, about 50.8 tok/s at 2.66 accepted tokens/step (an equal-acceptance estimate). A v16b control measured 73.2 tok/s peak and 69.0 tok/s median on copy-heavy, low-effort single requests. A separate five-stream equal-length probe measured 146.3 tok/s aggregate peak; three four-stream agent-shaped probes had a 97.1 tok/s median. Two eval seeds: 459/492 and 458/492. Added the pinned public-checkpoint download path, T80 builder with CPU tests and reference report, complete iter6c-to-iter6d image inputs, source hash preflight and CPU image gates. A clean GB10 build and serving run have not yet been recorded for this release checkout.

## v16 — measured, not adopted

Added target-side calibrated splice to v16b's stack. About 0.88 ms faster in the candidate comparison, but teacher-forced top-1 agreement moved −2.58 percentage points from the original.

## v15 — fallback candidate

Fast PLE gather, 65,536-id draft vocabulary, fp8 drafter experts, 16 GB KV pool and short-convolution H2D backport. Candidate-window step 55.65 ms; one eval seed 459/492.

## v13 — early refined baseline

Block rejection, probabilistic draft sampling, PLE gather tuning. Interleaved production step 67.167 ms versus original 68.276 ms in the same window.
