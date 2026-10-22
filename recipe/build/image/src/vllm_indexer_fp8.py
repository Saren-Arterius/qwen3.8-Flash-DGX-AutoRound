# SPDX-License-Identifier: Apache-2.0
"""Force the Qwen4Exp indexer COMPRESSED key cache to fp8 (magi-v3-public).

Upstream `AttentionConfig.indexer_kv_dtype` defaults to "auto", which the QSA
indexer resolves to "bf16" — so even with `--kv-cache-dtype fp8_e4m3` the
compressed indexer copy stays bf16. The kernels already support e4m3
(RMSNormed before quantization, fp8xfp8 dot, no scales), but there is no CLI
flag; this flips the resolver instead.

`raw_key_cache` is deliberately untouched: it packs int64 mRoPE positions
into bf16 cells, and quantizing it destroys position bits.

Gated on VLLM_INDEXER_FP8=1 (serve-magi.sh sets it when KV_DTYPE=fp8*).
Dormant otherwise: explicit non-bf16 resolutions (e.g. mxfp4) pass through.
"""
from __future__ import annotations

import os


def apply() -> None:
    try:
        from vllm.config.attention import AttentionConfig as _AC
    except Exception:
        return
    if getattr(_AC, "_indexer_fp8_patched", False):
        return
    orig = _AC.resolve_indexer_kv_dtype

    def resolved(self, default="bf16"):
        try:
            r = orig(self, default)
        except TypeError:
            r = orig(self)
        if os.environ.get("VLLM_INDEXER_FP8") == "1" and r == "bf16":
            return "fp8"
        return r

    _AC.resolve_indexer_kv_dtype = resolved
    _AC._indexer_fp8_patched = True
