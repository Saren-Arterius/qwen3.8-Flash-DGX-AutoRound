<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Benchmarks

All results are from one GB10. They are records of a specific image, checkpoint, launch and workload. They have not been repeated on a clean public build. The [CSV tables](../results/) drive the charts.

## Measured peak rates

The v16b control's fastest **single request** decoded at **73.1987 tok/s** (73.2 at one decimal): a copy-heavy, low-effort request with 871 completion tokens. The median across its nine copy-heavy requests was **69.0156 tok/s** (69.0). These are raw request rates, not the equal-acceptance estimate in the table below. The same control's 18 agent-shaped requests had a **48.8178 tok/s** median (48.8), so the copy-heavy peak should not be used as a coding-agent expectation.

The highest recorded v16b **multi-stream aggregate** was **146.3 tok/s** in one decode-window probe with five equal-length 600-token streams. There is no repeat median for that exact five-stream workload. Three separate four-stream agent-shaped aggregate probes read **97.70, 97.13 and 96.7780 tok/s**; their median is **97.13 tok/s** (97.1). The peak and median therefore differ in stream count and workload, and the five-stream decode-window rate is not whole-request wall throughput. The one-stream equal-length cell failed its occupancy gate. The 50/74/94/101/107 series is computed planning data and was excluded from peak selection.

## Single-stream decision metric

The candidate comparison used a cached long system prefix, tool schemas, six coding tasks, three repeats per task and 1,500 generated tokens, with thinking on. The work resembled long-context coding sessions with repeated tool descriptions, called **agent-shaped** in the source program. Decode tok/s changes when MTP accepts a different number of tokens; step time, computed per sample as `1000 × accepted_tokens_per_step / decode_tok_per_second`, isolates the cost of an engine step. The original, production v13, v15 and v16 were interleaved in three rounds; v16b was measured in a later paired window and bridged through the common v15/v16 controls. That bridge is about 52.3 ms; v16b's directly measured step was 52.33 ms. A gain had to exceed `max(2 tok/s, 1.5 × anchor spread)` and keep the same sign across interleaves. A single box, warm-cache state, launch age and temperature limit generalization.

| Recipe | Step time, ms | Rate at 2.66 accepted tokens/step, tok/s | Reading |
|---|---:|---:|---|
| Original | 68.276 | — | Interleaved comparison window |
| v13 | 67.167 | — | Interleaved comparison window |
| v15 | 55.65 | 47.8 | Candidate comparison window |
| v16b | 52.33 | about 50.8 | Candidate comparison window |
| v16 | 51.45 | 51.7 | Candidate comparison window; quality criterion failed |

The rate column is an equal-acceptance arithmetic view, not five independent raw runs. [Speed ladder](../assets/speed-ladder.svg) uses the same data. The 52.3-ms bridge gives about 28% more accepted-token throughput than v13 at equal acceptance, and about 31% more than the original. The source program measured end-to-end composition: the v16 lever sum differed from its measured gain by 0.014 ms, and the v16b sum by 0.12 ms. Those closures apply to the tested composition and workload; they do not establish independent gains in other combinations.

## Stream count and TTFT

The later program summary **computed planning figures** of 50 / 74 / 94 / 101 / 107 tok/s at one through five streams. The directly measured v16b equal-length probe sent 600 tokens per stream and reported **51.1 / 73.8 / 115.1 / 129.1 / 146.3 tok/s** at one through five streams; its one-stream cell failed the occupancy gate. [throughput.csv](../results/throughput.csv) labels the two series separately. A code-path review concluded that an upstream strided-state-index defect does not reach this image; a later research pass raised the concern again from the absence of the upstream fix symbol. The four-in-flight agreement instrument remained open. These figures are descriptive, not a reliability guarantee or a clean cross-recipe comparison.

Warm TTFT on the cached coding prompt was 0.57 s. Long-context cold TTFT was 4.1 / 27.0 / 59.2 s for 16k / 64k / 128k prompts, while warm cached TTFT was 0.963 / 0.682 / 0.465 s. Those are different prompt shapes and cache states. The long-context bench had three repeats per cell; changing accepted length can swing tok/s, so step time is preferred there too.

## Public comparisons

The [MiaAI-Lab NVFP4 recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark) reports 48.7 tok/s at one stream on a short prose prompt at temperature zero, plus 74.6 at two and 113.7 at four. The [EXL3 recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-DGX-Spark-recipe) reports 58.8 tok/s single-stream with native ExLlamaV3 and a different quantization; its vLLM plugin path has separate concurrency numbers. Engine, checkpoint, prompt, temperature, cache state and output length vary, so the reports should be reproduced under one harness before ranking them. The source investigation did not validate an 80 tok/s single-stream number for this exact model on one GB10; that headline may refer to a different model or aggregate throughput.
