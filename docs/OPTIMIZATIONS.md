# The int4 + int8 + fp8 hybrid recipe

This fork extends the original mmap-PLE recipe (see [HOW-IT-WORKS.md](HOW-IT-WORKS.md))
with a quantized serving stack built on
**[Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound)**
instead of the NVFP4 checkpoint, plus a set of GB10 patches. Measured on a
DGX Spark: **~49 tok/s single-stream decode with MTP=2** (~2,000 tok/s prefill),
versus ~25–28 tok/s for the NVFP4 recipe.

Everything is applied at image build time (see the `Dockerfile`); each patch is
independent and gated by an env var where it changes behavior.

## What runs in what precision

| Component | Precision | How |
|---|---|---|
| 512-expert MoE (48 layers + MTP layer's own 512) | **int4** GPTQ-Marlin g128 | Intel checkpoint as-is |
| lm_head (shared with MTP draft head) | **int8** GPTQ-Marlin (uint8b128) | `tools/quantize_lm_head_int8.py` + `"lm_head": true` |
| GDN in/out projections, QSA q/k/v/o, shared expert | **fp8** blockwise e4m3 (128×128) | `tools/fp8_convert.py` + `src/vllm_fp8_hybrid.py` |
| Embeddings, hyper-connections, norms, MoE gates, fc_hidden | bf16 | excluded via `dynamic` rules |
| PLE n-gram table (51B params, layer 1) | **fp8** rows, mmapped from disk | `tools/fetch-ple-table-fp8.sh` + the mmap patch |
| KV cache | bf16 | QSA refuses fp8 KV |

## Checkpoint preparation

One-time, CPU-only (a NAS box is fine — ours has no GPU). Order matters only
in that the index strip should come last.

```bash
CKPT=/models/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound
hf download Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound --local-dir "$CKPT"

# 1. bf16 lm_head -> int8 GPTQ (bit-exact reproduction of what we serve;
#    full-range symmetric, largest-|w| element maps to the -128 slot)
tools/quantize_lm_head_int8.py "$CKPT"

# 2. bf16 side layers -> blockwise fp8 (300 tensors across 51 shards;
#    originals kept as .bf16.bak, worst per-tensor rel err ~0.035)
tools/fp8_convert.py "$CKPT"

# 3. drop the 51B n-gram table from the index (it is served separately)
tools/strip_ngram_index.py "$CKPT"

# 4. the fp8 n-gram table, served from a separate dir
tools/fetch-ple-table-fp8.sh /models/ple-table-fp8
```

Then replace `quantization_config` in `$CKPT/config.json` with:

```json
{
  "quant_method": "gptq",
  "bits": 4,
  "group_size": 128,
  "desc_act": false,
  "sym": true,
  "lm_head": true,
  "dynamic": {
    "+:.*lm_head$": {"bits": 8},
    "-:.*linear_attn.*": {},
    "-:.*self_attn.*": {},
    "-:.*hyper_connection.*": {},
    "-:.*visual.*": {},
    "-:.*shared_expert.*": {},
    "-:.*\\.ple\\..*": {},
    "-:.*embed.*": {},
    "-:.*fc_hidden.*": {},
    "-:.*layers\\.48\\..*": {},
    "-:.*\\.gate$": {}
  }
}
```

Why: this vLLM build has no auto-round loader, but its GPTQ config
(`AutoGPTQConfig` → Marlin kernels) reads the same packed tensors. The
`dynamic` rules are GPTQModel-style (`re.match`, start-anchored — hence the
`.*` prefixes): `-:` excludes families that are not int4-packed, `+:.*lm_head$`
overrides the head to 8-bit. `-:.*layers\.48\..*` keeps the MTP draft layer's
non-expert weights out of the int4 path (its experts are packed int4 in the
checkpoint and load fine).

## The patches

### 1. PLE mmap upgrades (`src/vllm_ple_mmap.py`, extends upstream's patch)

- **Any table dtype**: bf16/f16 tables and fp8 (with `weight_scale`) are all
  accepted; row size is derived from the safetensors headers. The fp8 table
  halves the bytes read per token vs bf16.
- **`VLLM_PLE_MMAP_DIR`**: the table no longer has to live inside the
  checkpoint dir — point it at any directory of safetensors shards (NFS, local
  NVMe, a RAM-backed device...). In our measurements the backend barely
  matters once the page cache is warm; the gather is Python-dispatch-bound.
- **Hot path**: per-step dedup of row ids on CPU (`np.unique`), gather of
  unique rows only, staging through a persistent pinned buffer with an async
  H2D copy, and GPU-side expansion via the inverse index. A decode fast path
  (`VLLM_PLE_MMAP_FAST_ROWS`, default 512) skips the thread pool entirely for
  small gathers. Net effect: ~7.2 → ~2.5–3.8 ms per lookup op.
- **Stats**: `VLLM_PLE_MMAP_STATS_SEC` (default 30) logs
  `PLE mmap stats (last Ns): calls, op ms, gather ms, rows, MB` and resets the
  counters each period.

### 2. FLA shared-memory gate (`Dockerfile` sed)

sm_121 reports 99 KiB of shared memory per block — the same as ADA, where the
flash-linear-attention Triton kernels use their big GDN tiles — but the gate in
`vllm/third_party/flash_linear_attention/ops/utils.py` demands 100 KiB, so all
36 GDN layers silently fell back to small tiles. Lowering the gate to 99 KiB
(101376) lets GB10 take the big-tile path. (Found the hard way in the
Qwen3.5-122B Spark recipe — ported from
[Entrpi/qwen3.5-122B-A10B-on-spark](https://github.com/Entrpi/qwen3.5-122B-A10B-on-spark)'s
`patch_fla_shmem.py`.)

### 3. int8 lm_head enablement (`Dockerfile` sed on `model.py` / `mtp.py`)

Upstream constructs `ParallelLMHead` without `quant_config`, forcing a bf16
head (1.27 GiB, and a bf16 GEMV per token over a 248320 vocab). One added
kwarg in both the main model and the MTP draft lets the head pick up the
checkpoint's int8 GPTQ packing. Without the `mtp.py` half, MTP ≥ 3 crashes at
load ("no module or parameter named 'lm_head.qweight'").

### 4. int4+fp8 hybrid dispatch (`src/vllm_fp8_hybrid.py`, `VLLM_FP8_HYBRID=1`)

vLLM's GPTQ config quantizes listed layers and leaves the rest to
`UnquantizedLinearMethod` — it has no notion of "this excluded layer is
actually fp8 in the checkpoint". This shim wraps `AutoGPTQConfig`: it scans the
checkpoint metadata for `F8_E4M3` weights with a `weight_scale_inv` sibling and
routes exactly those layers to vLLM's blockwise-`Fp8Config`
(`weight_block_size=[128,128]`, dynamic activation scheme) while everything
else keeps the GPTQ path. `VLLM_USE_DEEP_GEMM=0` is required on sm_121
(DeepGEMM hits `CUDA_ERROR_LAUNCH_FAILED`); the triton fallback is fine.

### 5. Never-evict prompt pinning (`src/patch_never_evict.py`)

`--never-evict-kv-cache-prompt-includes "<substring of your system prompt>"`
pins the KV blocks of any prompt containing that marker: they are held in a
side queue on `BlockPool`, excluded from the free count, and thus never handed
out for eviction — your assistant's system prompt stays cached no matter what
other traffic does. `--never-evict-kv-cache-max-fraction` (default 0.25) caps
the pin. The pin set is *replaced* on each matching request, so an updated
system prompt releases the old blocks automatically.

Implementation notes: the marker is tokenized once and matched as a token-id
subsequence (first/last token dropped — BPE merges at the boundaries); the pin
is keyed on block *hashes*, not block ids, because this hybrid model frees
mamba/GDN state blocks mid-request — each freed block is re-claimed by the pin
the moment `free_blocks()` sees it. Only prefix-cacheable KV groups
participate (the MTP draft layer's group is not, and must be skipped). This is
a pin-only port of our `arc_pin2` patch from the Qwen3.5-122B Spark stack,
which built the pin on top of the ARC GPU-eviction work in
[vllm#40270](https://github.com/vllm-project/vllm/pull/40270); the ARC/2Q
policies themselves were deliberately dropped — the stock free queue is
C-speed on a path this model hits every step.

Self-check (no GPU): `docker run --rm -v "$PWD/src:/t" --entrypoint python3
qwen38-flash-dgx /t/test_never_evict_pin.py`.

### Prefix caching: on

Upstream runs `--no-enable-prefix-caching` because of a CUBLAS error in the
GDN `in_proj` GEMM on the cached-block path. With the fp8 side layers that
GEMM runs a different kernel, and prefix caching has been stable in our
serving (`PREFIX_CACHE=1`). Two behaviors worth knowing on this hybrid model:

- The prefix-cache block size is large (1600 tokens): prefixes shorter than
  that get no reuse.
- A repeated prefix reaches full TTFT benefit from the **3rd** identical
  request, not the 2nd: request 1 caches attention blocks, request 2 produces
  the durable copy-on-write mamba-state snapshots ("align" mode), request 3
  hits both.

## Serving

```bash
docker build -t qwen38-flash-dgx .
MODEL_DIR=/models/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound \
TABLE_DIR=/models/ple-table-fp8 \
PREFIX_CACHE=1 PIN_PROMPT="You are HomeBot, the household assistant." \
scripts/serve-intel-ar.sh
```

Knobs specific to this script (on top of upstream's): `FP8_HYBRID` (default 1),
`PREFIX_CACHE` (default 0), `PIN_PROMPT`/`PIN_MAX_FRACTION`, `LOAD_FORMAT`
(default `fastsafetensors` — noticeably faster cold boots), `SERVED_NAME`.

## Speculative decoding and TTFT

MTP raises decode substantially but puts a floor (~0.8 s) under
time-to-first-token: vLLM's v1 engine only emits the first token after the
drafter has run, and MTP's draft layer is a stateful autoregressive
transformer — on every prefill chunk it must run a full-chunk-width forward
(always eager: above the cudagraph capture sizes) to sync its own KV/GDN
state, plus k−1 sequential single-token passes. Cross-attention drafters like
DFlash don't pay this (their context comes from the target's hidden states
and they propose all k tokens in one tiny cudagraphed forward), but
Flash-Next has no such drafter — MTP is what ships in the checkpoint. If your
workload is TTFT-sensitive, weigh `MTP` depth against `MTP=0`; only 0 removes
the floor.
