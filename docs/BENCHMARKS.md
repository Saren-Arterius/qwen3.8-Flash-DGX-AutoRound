<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Benchmarks

These v16b measurements use one GB10, the promoted image and the T80 drafter checkpoint. The [CSV tables](../results/) drive the charts.

## Measured peak rates

| Workload | Measurement | Reading |
|---|---:|---|
| Copy-heavy, low-effort single request | **73.2 tok/s peak** | Fastest request, 871 completion tokens |
| Copy-heavy, low-effort single requests | **69.0 tok/s median** | Nine requests |
| Agent-shaped coding, single requests | **48.8 tok/s median** | 18 requests |
| Equal-length, five-stream decode window | **146.3 tok/s peak aggregate** | One probe, 600 completion tokens per stream |
| Four-stream agent-shaped aggregate | **97.1 tok/s median** | Three probes: 97.70, 97.13 and 96.7780 tok/s |

The source records the exact single-request peak as 73.1987316 tok/s and its copy-heavy median as 69.0156140 tok/s. The agent-shaped median is 48.8178301 tok/s. The five-stream and four-stream rows use their stated workloads and stream counts.

## Agent-shaped decode

The coding benchmark used a cached long system prefix, tool schemas, six coding tasks, three repeats per task and 1,500 generated tokens, with thinking enabled. The v16b candidate window measured **52.33 ms per decode step**. The interleaved original, v13 and v15 windows measured 68.276, 67.167 and 55.65 ms respectively. [Speed ladder](../assets/speed-ladder.svg).

| Recipe | Measured step time, ms |
|---|---:|
| Original | 68.276 |
| v13 | 67.167 |
| v15 | 55.65 |
| v16b | **52.33** |

## Stream count and time to first token

The equal-length v16b decode probe measured aggregate throughput of **73.8 / 115.1 / 129.1 / 146.3 tok/s** at **2 / 3 / 4 / 5 streams** respectively. [Throughput data](../results/throughput.csv) · [Chart](../assets/throughput.svg).

Warm time to first token on the cached coding prompt measured **0.57 s**. Long-context cold time to first token measured **4.1 / 27.0 / 59.2 s** at 16k / 64k / 128k prompt tokens. Warm cached readings at those lengths measured **0.963 / 0.682 / 0.465 s**. Each long-context cell had three repeats.
