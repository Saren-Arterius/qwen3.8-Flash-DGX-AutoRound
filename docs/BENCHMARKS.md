<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Benchmarks

These v16b measurements use one GB10, the promoted image and the T80 drafter checkpoint. The [CSV tables](results/) drive the charts.

## Measured peak rates

| Workload | Measurement | Reading |
|---|---:|---|
| 1 stream, copy-heavy single request | **73.2 tok/s peak** | Fastest request, 871 completion tokens |
| 2 equal-length streams | **73.8 tok/s peak aggregate** | 600 completion tokens per stream |
| 3 equal-length streams | **115.1 tok/s peak aggregate** | 600 completion tokens per stream |
| 4 equal-length streams | **129.1 tok/s peak aggregate** | 600 completion tokens per stream |
| 5 equal-length streams | **146.3 tok/s peak aggregate** | 600 completion tokens per stream |

Typical medians provide workload context: **69.0 tok/s** for nine copy-heavy single requests, **48.8 tok/s** for 18 agent-shaped single requests and **97.1 tok/s aggregate** for three four-stream agent-shaped runs. The exact single-request peak was 73.1987316 tok/s.

## Agent-shaped decode

The coding benchmark used a cached long system prefix, tool schemas, six coding tasks, three repeats per task and 1,500 generated tokens, with thinking enabled. The v16b candidate window measured **52.33 ms per decode step**. The interleaved original, v13 and v15 windows measured 68.276, 67.167 and 55.65 ms respectively. [Speed ladder](assets/speed-ladder.svg).

| Recipe | Measured step time, ms |
|---|---:|
| Original | 68.276 |
| v13 | 67.167 |
| v15 | 55.65 |
| v16b | **52.33** |

## Stream count and time to first token

Measured peak decode throughput is **73.2 / 73.8 / 115.1 / 129.1 / 146.3 tok/s** at **1 / 2 / 3 / 4 / 5 streams** respectively. The one-stream peak is the fastest copy-heavy single request; the multi-stream peaks come from equal-length decode windows. [Throughput data](results/throughput.csv) · [Chart](assets/throughput.svg).

Warm time to first token on the cached coding prompt measured **0.57 s**. Long-context cold time to first token measured **4.1 / 27.0 / 59.2 s** at 16k / 64k / 128k prompt tokens. Warm cached readings at those lengths measured **0.963 / 0.682 / 0.465 s**. Each long-context cell had three repeats.
