<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Evaluation and quality

The fixed suite had 492 items: code 100 (execution checked), math 140, knowledge 160, instruction following 60, tool calls 20 and long-context needles 12. The scorer checked category-specific answers and monitored empty, truncated, looping and repeated output. Sampling used temperature 1.0, top-p 0.95, top-k 20, medium thinking, concurrency four and a 6,000-token cap. The two seeds were 20260908 and 20260910; the same item manifest was used in both.

| Run | Correct | Rate | Notes |
|---|---:|---:|---|
| v16b, seed 20260908 | 458/492 | 93.09% | Code 99/100; instruction 52/60; knowledge 147/160; long context 12/12; math 133/140; tools 15/20 |
| v16b, seed 20260910 | 459/492 | 93.29% | Same suite, different sampling seed |
| pooled | 917/984 | 93.19% | Repeated items; do not treat as independent 984-item coverage |

The long-generation arm was separate: 24 items, 12,000-token cap, two runs at sampling seed 20260908, 21/24 then 22/24 correct, with no cap hits, degeneration flags or loops on either reading. The difference between those two runs illustrates the noise floor. V16b did not show a detectable loss against the original or v15 on this suite, but the sample size cannot prove equal quality for every task.

The teacher-forced prefill comparison with the original checkpoint found a top-1 agreement delta of −0.06 percentage points against a measured 0.15-point null band, and a per-token perturbation-scale ratio 1.012 ± 0.023 against a 1.00 ± 0.02 null band. The raw 0.010-nats/token log-likelihood clause was **undecidable** under its precision gate. The full v16 recipe moved top-1 agreement by −2.58 points; the target-side T85 splice alone moved it by about −2.53 points. V16 was reported and remains available, but was not adopted or recommended by this program; the choice belongs to the user. V16b is therefore described as agreement at the test's resolution, never as bit-identical or universally lossless.

The local evaluator source contains private tool descriptions and private test material; it is intentionally not shipped. [results/evals.csv](../results/evals.csv) and [results/longgen.csv](../results/longgen.csv) are aggregate, privacy-reviewed exports. The categories and scorer behavior are summarized above, but the dataset IDs and full item manifest are not published here. A third party can build a replacement suite, but cannot exactly replay this one from the repository.
