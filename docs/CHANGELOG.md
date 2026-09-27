<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Qwen3.8 Flash · DGX UltraFast — changelog

## v16b — promoted configuration

The v16b recipe combines the T80 dense-MTP drafter head, iteration-6d image, low-latency GEMM, verify-path top-k and retained draft blocks. Its candidate-window decode step measured **52.33 ms**. Three copy-heavy rounds at each stream count on one GB10 measured peak decode throughput of **74.1 / 110.0 / 132.9 / 155.5 / 175.8 / 191.5 / 205.8 / 212.2 tok/s at 1 / 2 / 3 / 4 / 5 / 6 / 7 / 8 streams**. The five-stream point was measured in two sessions: **174.87** and **175.78 tok/s** peak. The two evaluation seeds scored **459/492** and **458/492**.

This release includes the pinned public checkpoint downloads, T80 builder and reference report, complete iter6c-to-iter6d image inputs, source hash preflight and CPU tests.

## v15 — recipe progression

Fast PLE gather, 65,536-id draft vocabulary, fp8 drafter experts, 16 GB KV pool and short-convolution H2D backport. Candidate-window step: **55.65 ms**. One evaluation seed: **459/492**.

## v13 — recipe progression

Block rejection, probabilistic draft sampling and PLE gather tuning. Interleaved step: **67.167 ms**, alongside the original recipe at **68.276 ms**.
