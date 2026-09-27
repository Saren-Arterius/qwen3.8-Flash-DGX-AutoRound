<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Qwen3.8 Flash · DGX UltraFast — changelog

## v16b — promoted configuration

The v16b recipe combines the T80 dense-MTP drafter head, iteration-6d image, low-latency GEMM, verify-path top-k and retained draft blocks. Its candidate-window decode step measured **52.33 ms**. Copy-heavy single requests measured **73.2 tok/s peak** and **69.0 tok/s median**; agent-shaped single requests measured **48.8 tok/s median**. A five-stream equal-length decode probe measured **146.3 tok/s peak aggregate**; three four-stream agent probes measured **97.1 tok/s median**. The two evaluation seeds scored **459/492** and **458/492**.

This release includes the pinned public checkpoint downloads, T80 builder and reference report, complete iter6c-to-iter6d image inputs, source hash preflight and CPU tests.

## v15 — recipe progression

Fast PLE gather, 65,536-id draft vocabulary, fp8 drafter experts, 16 GB KV pool and short-convolution H2D backport. Candidate-window step: **55.65 ms**. One evaluation seed: **459/492**.

## v13 — recipe progression

Block rejection, probabilistic draft sampling and PLE gather tuning. Interleaved step: **67.167 ms**, alongside the original recipe at **68.276 ms**.
