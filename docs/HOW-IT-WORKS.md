# How it works

> **Upstream's doc, NVFP4-era.** The mmap-PLE story below is the foundation
> this fork builds on, but every size and speed figure refers to the NVFP4
> checkpoint that [blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX)
> serves. This fork's checkpoint, numbers and serving path live in the
> [README](../README.md).

## The memory problem

Qwen3.8-Flash-Next is a sparse MoE with an unusual extra component: a **51B-parameter
n-gram embedding table** (the paper calls it PLE / "Engram"). The `RadixArk` NVFP4
checkpoint breaks down roughly as:

| Component | Format | Size |
|---|---|---|
| Routed experts (48 layers × 512 experts, 10 active) | NVFP4 | ~63 GiB |
| Attention / GDN / QSA / shared experts / gate / lm_head / MTP | bf16 | ~15 GiB |
| **N-gram (PLE) table** — 16 heads × 20M rows × 160 dims | FP8 e4m3 + 1 scale | **~44 GiB** |
| **Total** | | **~122 GiB** |

A DGX Spark has **128 GB unified memory**, of which ~10 GiB is OS/driver/Docker. So
122 GiB of weights leaves essentially nothing for the KV cache — you cannot serve.

vLLM ships an offload path (`VLLM_PLE_CPU_OFFLOAD=1`) that moves the table to pinned
**host** RAM. On a discrete-GPU server that frees VRAM. On a Spark, host and device
are the **same physical pool**, so it frees nothing. That is why, until now, the only
thing that ran Flash-Next on a Spark was a llama.cpp GGUF — which mmaps its weights by
default, but has no sparse-attention kernel and so has poor prefill and no MTP.

## The lever

The table is a **lookup**, not compute. Per token the model reads exactly
**16 rows × 160 bytes = 2.5 KB**, at hashed (random) addresses. Even a 20k-token
prefill is ~320k row reads ≈ 1.3 GB — under a second on NVMe — and natural language
and code hit a very concentrated set of n-grams, so the hot rows stay in the page
cache after the first pass.

So the table does not need to be resident. This repo `mmap`s the checkpoint's
`model-plefp8-*.safetensors` shards and gathers rows on demand. That is exactly what
llama.cpp does with its GGUF — we just bring it to the vLLM path, which keeps the real
QSA/GDN kernels and MTP.

Result: **~76 GiB resident** (78 GiB of non-table weights minus a little), leaving
~20–22 GiB for KV at `GPU_MEM=0.85` — a 720–790k-token pool, i.e. ~3× concurrency at
the native 262k or a single 500k request with YaRN.

## The patch (`src/vllm_ple_mmap.py`)

Enabled by `VLLM_PLE_MMAP=1`; a complete no-op otherwise. It patches exactly one
class, `Qwen3_8FlashNextNGramEmbedding`, in three small ways:

1. **`__init__`** — swap the 44/95 GiB `VocabParallelEmbedding` for a tiny
   placeholder. No large parameter is ever allocated. The placeholder's `forward(ids)`
   gathers rows from `np.memmap` views of the shards (dedup + sort for locality, a
   thread pool so page faults overlap), returns an fp8 tensor on the GPU.

2. **`load_weights`** — drop the 128 shard tensors on the floor (they're served from
   disk) and keep only the global FP8 `weight_scale`, stored as
   `_offload_weight_scale` — which the **unmodified** `Qwen3_8FlashNextPLELayer.
   _dequantize_embeddings` already knows how to consume. Then open the memmaps.

3. **`forward_impl`** — wrap the hashing+lookup in a custom op
   `vllm::ple_mmap_lookup`. This is the crucial bit for GB10 (below).

Everything else — the n-gram hashing, the short-conv, the dequant, the sparse
attention — is stock vLLM.

## Three GB10 bugs this works around

Bringing the official image up on a real Spark with real weights surfaced three
issues. All are handled by the patch + the flags in `scripts/serve-intel-ar.sh`:

1. **`Cannot copy between CPU and CUDA tensors during CUDA graph capture`.**
   The gather is CPU work plus a pageable host→device copy; that cannot live inside a
   captured CUDA graph. Fix: the lookup is a **custom op declared as a splitting op**,
   so vLLM runs it *between* graph segments. Use `-cc.cudagraph_mode=PIECEWISE` (never
   `FULL*`). `--enforce-eager` also avoids it but is slower — and note it does **not**
   fully suppress capture here (the mamba/short-conv path still captures), so PIECEWISE
   + the splitting op is the right answer.

2. **`KeyError` on the layer registry during capture.** The custom op looks the layer
   up by name; registering it inside `forward_impl` fails because torch.compile does
   not re-run that Python line on graph replay. Fix: register in `__init__`.

3. **Two stock-model issues on sm_121, unrelated to this patch but required to run:**
   - `--no-enable-prefix-caching` — a GDN `in_proj` GEMM hits
     `CUBLAS_STATUS_INTERNAL_ERROR` on the cached-block path (2nd identical prompt).
   - full `torch.compile` off — an Inductor int64-indexing assert
     (`index out of bounds`) fires in the embedding gather codegen on sm_121. PIECEWISE
     capture with compile disabled on the splitting op sidesteps it.

## Long context: what works and what does not

Measured on the GX10 with the mmap patch, `GPU_MEM=0.85`, MTP=2 unless noted.

| Config | Result |
|---|---|
| 262144 native, MTP | KV pool ~720–790k tokens, ~3× concurrency at full length. Baseline prod. |
| **YaRN, CTX 500000, MTP** | **Works.** Pool ~724k tokens. Needle-in-a-haystack found at 276k and 414k tokens; decode 25–28 tok/s typical (36 on predictable text, ~94% draft acceptance there); no OOM. **This is the validated ceiling.** |
| YaRN, CTX 800000, MTP, `GPU_MEM=0.875` | Boots (pool 928k) and answers, but a 300k-token prefill got **SIGTERM from earlyoom** at 1.96% free memory: the prefill's activation peak plus the draft do not fit in ~5 GiB of headroom. |
| `--kv-cache-dtype fp8` | **Refused by the model**: `NotImplementedError: Qwen3.8-Flash-Next QSA requires a BF16 main KV cache`. So the "halve the KV" lever does not exist; at ~29 KB/token a single 1M request needs ~30 GiB of KV. |

Two YaRN-specific traps, both handled by upstream's `serve.sh`
([blazux/qwen3.8-Flash-DGX](https://github.com/blazux/qwen3.8-Flash-DGX));
this fork's `scripts/serve-intel-ar.sh` does not wire YaRN — pass the same
flags via `EXTRA` if you want >262k:

- YaRN is applied with Qwen's published `--hf-overrides` (rope `yarn`, factor 4,
  `original_max_position_embeddings` 262144) and needs `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`.
- **YaRN + MTP fails to boot** with `--mamba-block-size can only be set with
  --enable-prefix-caching`. Cause: dict `hf_overrides` are not propagated to the draft
  model (`SpeculativeConfig.compose_draft_hf_overrides` only forwards callables), so the
  draft keeps `max_model_len=262144` while sharing the `cache_config`, whose
  `mamba_block_size` was auto-set to the target's `max_model_len`. Fix: put
  `"max_model_len": <CTX>` inside `--speculative-config`, which overrides the draft's
  length (`_maybe_override_draft_max_model_len`).

## Correctness

`src/test_ple_mmap_cpu.py` builds synthetic FP8 shards (with the real safetensors
layout and non-trivial data offsets) and checks the mmap gather bit-for-bit against a
reference `table[ids]`, including dedup, multi-shard spans, the fp8 view path used by
the placeholder, and out-of-range → `IndexError`. It needs only numpy+torch (no GPU):

```bash
docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_ple_mmap_cpu.py
```

End-to-end, the served model is coherent ("The capital of France is Paris."), which is
the real test that the FP8 rows are being gathered and dequantized correctly — a wrong
gather turns the n-gram contribution to noise and the model degrades immediately.

## Performance notes

- **Prefill** ~2,400–2,660 tok/s (ctx 32k, single request). This is the axis that
  matters most versus llama.cpp (~540 tok/s), because Flash-Next's QSA prefill kernels
  only exist in vLLM/SGLang.
- **Decode** ~17 tok/s without speculation; with `MTP=2` **25–28 tok/s** on free-form
  prose (~63% draft acceptance) and up to ~36 tok/s on predictable text (~94%). The gather does one host↔device sync per decode step, which is pure
  latency at batch 1; MTP amortizes it. Removing that sync (staging ids through a
  pinned buffer, or a small resident hot-row cache) is the obvious next optimization.
- **First request into a cold region** of the table pays some NVMe I/O; it smooths out
  as the page cache warms. `PREWARM=1` streams the whole table once at boot (~10 s) for
  steadier first-request latency.

## Independent reproduction and the native offload path

[@jschmied](https://github.com/jschmied) reproduced this recipe on a DGX Spark
([issue #1](https://github.com/blazux/qwen3.8-Flash-DGX/issues/1)). They also
ran vLLM's native `VLLM_PLE_CPU_OFFLOAD=1` path and documented what it needs on the
NVFP4 checkpoint (the `Fp8Config` gate in `_get_ple_embedding_quant_method`, and
`CAP_SYS_PTRACE` because `yama.ptrace_scope=1` blocks the sibling-process
`pidfd_getfd` used for the CUDA-IPC handoff), plus concurrency traces showing
aggregate throughput of ~267 tok/s at 48 streams with page-fault cost per token
*falling* with batch size. Full notes:
<https://github.com/jschmied/qwen38-flash-next-gb10>.

## Upstream references

- vLLM recipe: <https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next>
- vLLM PR (Flash-Next support): <https://github.com/vllm-project/vllm/pull/53896>
- NVFP4 checkpoint: <https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4>
- SGLang day-0 write-up (PLE offload mechanics): <https://www.lmsys.org/blog/2026-08-26-qwen-flash-next>
