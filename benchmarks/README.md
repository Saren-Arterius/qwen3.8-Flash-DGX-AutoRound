<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Benchmark tools

`decode_bench.py` is the upstream-derived standalone decoder benchmark. `bench_agent.py` runs the coding benchmark with `--prompts benchmarks/prompts.example.json`. `probe_equal_length.py` runs equal-length stream probes with an explicit `--server-guard-file`. Aggregate measurements are in `results/`. The benchmark method and the difference between step time and tok/s are in [BENCHMARKS.md](../docs/BENCHMARKS.md). When publishing a result, include prompt shape, cache state, acceptance, image digest and repetition count.
