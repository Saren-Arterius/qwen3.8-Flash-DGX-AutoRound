<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Evaluation and quality

The fixed suite had 492 items: code 100 (execution checked), math 140, knowledge 160, instruction following 60, tool calls 20 and long-context needles 12. Sampling used temperature 1.0, top-p 0.95, top-k 20, medium thinking, concurrency four and a 6,000-token cap. Both seeds used the same item manifest.

| Run | Correct | Rate | Detail |
|---|---:|---:|---|
| v16b, seed 20260908 | 458/492 | 93.09% | Code 99/100; instruction 52/60; knowledge 147/160; long context 12/12; math 133/140; tools 15/20 |
| v16b, seed 20260910 | 459/492 | 93.29% | Same suite, second sampling seed |
| Both readings | 917/984 | 93.19% | Two readings of the 492-item suite |

The separate long-generation arm used 24 items, a 12,000-token cap and sampling seed 20260908. The two readings scored **21/24** and **22/24**. [Evaluation table](../results/evals.csv) · [Long-generation table](../results/longgen.csv).

The teacher-forced prefill comparison measured a **−0.06 percentage-point** top-1 agreement delta against a **0.15-point** control band. The per-token perturbation-scale ratio was **1.012 ± 0.023** against a **1.00 ± 0.02** control band. These measurements accompany the v16b performance results.
