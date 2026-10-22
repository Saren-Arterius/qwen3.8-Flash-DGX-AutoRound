# SPDX-License-Identifier: Apache-2.0
"""Qwen4Exp indexer compressed-KV dtype policy (magi-v5-public).

The compressed indexer cache SHOULD match the main KV dtype: under fp8 KV
a bf16 compressed cache splinters the KV groups (extra 64-token-style
group, stranded pool). But the raw indexer cache must stay bf16 always
(it packs int64 mRoPE positions into bf16 cells).

Policy (evaluated at indexer construction, i.e. boot time):
- VLLM_INDEXER_FP8=1  -> fp8, always (explicit opt-in).
- VLLM_INDEXER_FP8=0  -> bf16, always (explicit opt-out / skip).
- unset               -> follow KV_DTYPE: fp8* implies fp8, else bf16.
  (serve-magi.sh exports KV_DTYPE into the container, so fp8 KV
  automatically requires the fp8 indexer; normal KV stays skippable.)

Non-fp8 resolutions (e.g. an explicit mxfp4 config) always pass through.
"""
from __future__ import annotations

import os


def _want_fp8() -> bool:
    override = os.environ.get("VLLM_INDEXER_FP8")
    if override == "1":
        return True
    if override == "0":
        return False
    return str(os.environ.get("KV_DTYPE", "auto")).startswith("fp8")


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
        if r == "bf16" and _want_fp8():
            return "fp8"
        return r

    _AC.resolve_indexer_kv_dtype = resolved
    _AC._indexer_fp8_patched = True
