<!-- SPDX-License-Identifier: CC-BY-NC-4.0 -->
# Closed and optional experiments

| Lever | Measurement | Decision |
|---|---|---|
| KV 20 GB vs 16 GB | Decode delta about +0.03 ms median, inside ±0.29 ms null band; four-stream aggregate −1.5%; about 3.7 GiB less available memory | Keep 16 GB |
| Prefill chunk 16k vs 8192 | Driver out-of-memory warning during four concurrent cold 64k prefills, with worse TTFT | Keep 8192 |
| MTP depth 4 | Extra verified position cost 3.3–4.1 ms on the coding window; measured acceptance below break-even | Keep depth 3 |
| v16 target-side calibrated splice | Roughly 0.88 ms faster than v16b in the paired window; full v16 recipe −2.58 percentage points top-1 agreement against original, T85 alone about −2.53 points | Not adopted in v16b; v16 remains a reported option |
| NVFP4 experts | About 5.25 GB more resident bytes under the compared layout; same-hardware eager arm 40.9 tok/s | Not adopted |
| FP8 KV | QSA attention raised `NotImplementedError` in the tested image | Not used |
| Full GDN graph capture | Historic crash path and unresolved implementation risk | Not used |
| LPT threshold 1024 | Interactive TTFT during a 64k prefill 31.40 → 1.43 s; cold 64k prefill 32.29 → 36.41 s (+12.8%) | Optional latency trade; off by default |
| FlashInfer GDN prefill overlay | Cold 32k/64k TTFT roughly −6.5% to −7.0% in one incomplete pair; small teacher-forced numerics deviation | Experimental, not adopted |

The first two rows are capacity tests, not recommendations to raise memory settings. The LPT row compares fresh control and candidate lifetimes. The GDN overlay was not run through the complete promotion battery. See [benchmark method and provenance](BENCHMARKS.md) for the public performance context.
