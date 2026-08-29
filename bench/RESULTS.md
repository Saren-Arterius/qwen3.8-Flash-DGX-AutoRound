# Batch-1 decode perf campaign — 2026-08-29

Bench: `bench/decode_bench.py` (W1 = 1000 forced greedy tokens, fresh prompt;
W2 = pinned ~8k prefix hit + 256 tokens). Medians of 3.

## Step-time share table (torch.profiler, 24 decode steps, MTP=3, batch 1)

| slice                              | share | notes |
|------------------------------------|-------|-------|
| fp8 blockwise GEMMs (side layers)  | 24%   | 192 calls/step, ~94 us each (cutlass_3x blockwise, M~4) |
| int4 Marlin MoE experts            | 21%   | core model math |
| small bf16 wmma GEMMs              | 17%   | ~313 tiny calls/step (hyper_connection/gate/drafter glue) |
| int8 Marlin lm_head                | 15%   | ~2.95 ms/call x ~3.8 calls/step (verify + each draft pass) |
| GPU idle / launch gaps             | 12%   | piecewise cudagraph boundaries |
| PLE mmap gather (CPU op)           | 7%    | 5.5 ms/step |
| GDN + QSA + short-conv             | ~4%   | attention is NOT the bottleneck |

Step ~68 ms real (74 under profiler). Profile trace: /tmp/step_profile_1.json
(container) — retrigger anytime: `docker exec qwen38-flash touch /tmp/profile_trigger`
(needs STEP_PROFILE=1).

## Experiments (all on 2026-08-29, each = fresh container)

| config            | W1 med | W2 med | tok/step W1 | verdict |
|-------------------|--------|--------|-------------|---------|
| MTP=3 (baseline)  | 36.9   | 39.3   | 2.52        | keep (reference) |
| MTP=2             | 33.7   | 39.7   | 2.31        | reject: saved draft pass < lost p2 acceptance (p2=0.31) |
| MTP=4             | 30.5   | 37.3   | 2.67        | reject: p3=0.21 doesn't pay for +1 lm_head+draft pass |
| MTP=3 + --async-scheduling | 33.8 | 40.4 | 2.50 | reject: no win |
| MTP=3 + flashinfer autotune | 33.3 | 37.4 | 2.46 | reject: no win |
| MTP=3 re-check (same as baseline) | 31.8 | 46.3 | 2.47 | — |

## Caveat that dominates everything

Baseline vs baseline-recheck on the IDENTICAL config: W1 36.9 -> 31.8,
W2 39.3 -> 46.3. Between-restart / content-dependent variance is >10%,
larger than any knob effect measured. W1's forced-1000-token essay varies in
acceptance per run; box thermal/cache state drifts. The MTP=2/4 rejections
are still solid (tok/step moves exactly as the per-position acceptance
predicts); async/autotune are "no evidence of win", not "proven loss".

## Where the remaining headroom actually is (Phase 2 candidates, by share)

1. fp8 blockwise GEMM at M~4 (24%): kernel choice for tiny-M decode shapes.
2. Drafter lm_head (15%): 3 of the ~3.8 full-vocab lm_head calls per step
   are draft passes. Ideas: smaller draft vocab, reuse verify logits, k=3
   already optimal.
3. Small-GEMM storm (17%): 313 tiny bf16 GEMMs/step — fusion territory.
4. Launch gaps (12%): capture-size tuning; async-scheduling already failed.
5. PLE gather (7%): GPU hot-row cache; IBGDA science project (separate plan).
