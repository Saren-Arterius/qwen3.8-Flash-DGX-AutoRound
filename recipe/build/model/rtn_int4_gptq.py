#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
import time

import numpy as np

GROUP_SIZE = 128  # default; every entry point takes an explicit group_size
SUPPORTED_GROUP_SIZES = (32, 64, 128)  # marlin_utils.MARLIN_SUPPORTED_GROUP_SIZES
PACK_FACTOR = 8  # 32 // 4
UINT4B8_MAX = 7
UINT4B8_MIN = -8
UINT4B8_BIAS = 8
QZEROS_WORD_INT4 = 0x77777777

# --------------------------------------------------------------------------
# Which modules Iteration 4A converts.
#
# Every one of these is (a) F8_E4M3 + weight_scale_inv in the checkpoint and
# (b) a distinct vLLM Linear, or one half of a *fully* converted fused Linear.
#
# packed_modules_mapping (model.py:593-602) fuses:
#     in_proj_qkvz = [in_proj_qkv, in_proj_z]      <- both fp8, both converted
#     qkv_proj     = [q_proj, k_proj, v_proj]      <- all-or-nothing
#     gate_up_proj = [gate_proj, up_proj]          <- both fp8, both converted
# and ``gptq_utils.is_layer_gptq_quantized`` raises if only *some* shards of a
# fused layer are quantised, so the QSA q/k/v triple can only move as a unit
# and so can the two GDN in-projection shards.  This is the "#40252 trap in
# reverse": the tensors are emitted under the checkpoint's split names,
# but the *decision* is taken per fused module.
#
# Target-side conversion tiers:
#   all        -- every one of the 300 tensors goes to int4      (-1,274 MB/step)
#   fallbackA  -- QSA fused qkv_proj stays blockwise fp8         (-1,095 MB/step)
#   fallbackB  -- fallbackA + GDN out_proj stays blockwise fp8   (  -847 MB/step)
# A protected layer stays at *blockwise fp8*, never int8 GPTQ.
#
# NOT converted in any tier, and why:
#   * self_attn.indexer.index_qk_proj  -- BF16 already, not fp8; it is the QSA
#     sparse indexer's own projection and stays untouched.
#   * linear_attn.in_proj_a / in_proj_b / conv1d / norms, mlp.gate,
#     shared_expert_gate, ple.*, embed_tokens, visual.*, every mtp.* tensor
#     -- BF16 already.
# --------------------------------------------------------------------------
P_GDN_IN = [
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.linear_attn\.in_proj_qkv$",
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.linear_attn\.in_proj_z$",
]
P_GDN_OUT = [
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.linear_attn\.out_proj$",
]
P_QSA_QKV = [
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.self_attn\.q_proj$",
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.self_attn\.k_proj$",
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.self_attn\.v_proj$",
]
P_QSA_O = [
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.self_attn\.o_proj$",
]
P_SHARED = [
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.shared_expert\.gate_proj$",
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.shared_expert\.up_proj$",
    r"^(?:model\.)?(?:language_model\.)?layers\.\d+\.mlp\.shared_expert\.down_proj$",
]

TIERS = {
    # primary arm: all 300 tensors
    "all": P_GDN_IN + P_GDN_OUT + P_QSA_QKV + P_QSA_O + P_SHARED,
    # fallback A: protect the QSA fused qkv_proj (qsa.py:242 without_modelopt_fp4)
    "fallbackA": P_GDN_IN + P_GDN_OUT + P_QSA_O + P_SHARED,
    # fallback B: also protect the GDN out_proj (arXiv 2609.04098 Table 3)
    "fallbackB": P_GDN_IN + P_QSA_O + P_SHARED,
    # diagnostic arm: the GDN block only (71.4 % of the slice)
    "gdn": P_GDN_IN + P_GDN_OUT,
}
# Legacy spellings kept so older commands and notes still resolve.
TIERS["full"] = TIERS["all"]
TIERS["core"] = TIERS["fallbackA"]

TIER_ALIASES = {"full": "all", "core": "fallbackA"}

# Human-readable description, used by the builder's report and the README.
TIER_DOC = {
    "all": "all 300 side-layer tensors -> int4 (primary arm)",
    "fallbackA": "QSA fused qkv_proj stays blockwise fp8",
    "fallbackB": "QSA fused qkv_proj and GDN out_proj stay blockwise fp8",
    "gdn": "the GDN block only -- in_proj_qkv, in_proj_z, out_proj (diagnostic; "
    "71.4 % of the slice and all of the unmeasured risk)",
}


def canonical_tier(tier: str) -> str:
    return TIER_ALIASES.get(tier, tier)


# ==========================================================================
# The MTP drafter's dense side layers: bf16 to int4 g32.
#
# Scope, and why each line of it is what it is (all read out of the image
# ``qwen38-flash-dgx:known-good-20260906``):
#
#   * the drafter has NO Gated DeltaNet block: ``mtp.py:200-207`` builds its one
#     decoder layer with ``layer_type="full_attention"``, so there is no
#     ``linear_attn.*`` to convert.  Its dense weights are the QSA attention
#     projections, the two ``fc`` projections and the shared expert.
#   * ``mtp.hyper_connection_mixer.*`` and both per-layer hyper-connections are
#     NOT targets and cannot be targets: every HC Linear is constructed with
#     ``quant_config=None`` (``hyperconnection.py:102,113,122``), so
#     ``LinearBase.__init__`` takes the ``UnquantizedLinearMethod`` branch and
#     ``AutoGPTQConfig.get_quant_method`` is never called for them.  No
#     checkpoint change can move them; they stay bf16.
#   * ``mtp.layers.N.mlp.gate`` (the router) and ``mlp.shared_expert_gate`` are
#     built with ``quant_config=None`` as well (``qwen3_next.py``
#     ``Qwen3NextSparseMoeBlock``: ``gate``/``shared_expert_gate``), so they are
#     out of scope for the same reason.
#   * ``mtp.layers.N.self_attn.indexer.index_qk_proj`` IS a real quantisable
#     ``ReplicatedLinear`` (``indexer_qsa.py:134-139`` passes the quant config),
#     but it is deliberately left bf16, exactly as 4A leaves the target's
#     indexer bf16: it is the sparse indexer's own projection and it is 3.125
#     MiB/pass (1.8 % of the drafter's dense slice).
#   * the drafter's 512 routed experts stay bf16 on disk: they are taken to
#     online per-tensor fp8 at runtime by DRAFTER_EXPERTS_FP8=1. A
#     checkpoint-side conversion would collide with that path.
#   * ``lm_head`` is SHARED with the target (``mtp.py:91-96`` maps
#     ``mtp.shared_head.head.*``/``lm_head.*`` onto the one ``lm_head``) and is
#     already int8 GPTQ.  Untouched.
#
# The layer index: the CHECKPOINT spells the drafter ``mtp.layers.0.*`` but the
# runtime builds it under the vLLM prefix ``mtp.layers.48.*`` --
# ``mtp.py:166`` sets ``mtp_start_layer_idx = config.num_hidden_layers`` (48)
# and ``mtp.py:204`` passes ``f"{prefix}.layers.{mtp_start_layer_idx + idx}"``
# with ``prefix = maybe_prefix("", "mtp") = "mtp"``.  ``dynamic`` patterns and
# ``modules_in_block_to_quantize`` are both matched against the vLLM prefix, so
# they must use 48; the tensors on disk must use 0.  ``mtp_vllm_prefix`` is the
# one place that conversion lives.
# ==========================================================================
MTP_START_LAYER_IDX = 48  # mtp.py:166, config.text_config.num_hidden_layers
MTP_DRAFT_PASSES = 3  # MTP depth 3: the drafter's weights are read 3x per step

_MTP_HEAD = r"^(?:model\.)?(?:language_model\.)?mtp\."

P_MTP_ATTN = [
    _MTP_HEAD + r"layers\.\d+\.self_attn\.q_proj$",
    _MTP_HEAD + r"layers\.\d+\.self_attn\.k_proj$",
    _MTP_HEAD + r"layers\.\d+\.self_attn\.v_proj$",
    _MTP_HEAD + r"layers\.\d+\.self_attn\.o_proj$",
]
P_MTP_FC = [
    _MTP_HEAD + r"fc_embedding$",
    _MTP_HEAD + r"fc_hidden$",
]
P_MTP_SHARED = [
    _MTP_HEAD + r"layers\.\d+\.mlp\.shared_expert\.gate_proj$",
    _MTP_HEAD + r"layers\.\d+\.mlp\.shared_expert\.up_proj$",
    _MTP_HEAD + r"layers\.\d+\.mlp\.shared_expert\.down_proj$",
]

MTP_TIERS = {
    "drafter-dense": P_MTP_ATTN + P_MTP_FC + P_MTP_SHARED,
}
MTP_TIER_DOC = {
    "drafter-dense": "the MTP drafter's dense side layers (QSA q/k/v/o, "
    "fc_embedding, fc_hidden, shared expert gate/up/down) bf16 -> int4; the "
    "hyper-connections, the router, the QSA indexer, the routed experts and "
    "the shared lm_head are untouched",
}

# The fused modules the runtime builds out of the checkpoint's split names.
MTP_FUSE = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
}


def mtp_vllm_prefix(ckpt_module: str, start_idx: int = MTP_START_LAYER_IDX) -> str:
    """Checkpoint module name -> the vLLM prefix the quant config matches.

    ``mtp.layers.0.self_attn.q_proj`` -> ``mtp.layers.48.self_attn.q_proj``.
    Leading ``model.`` / ``language_model.`` are stripped: the drafter is loaded
    through ``mtp.py:_remap_mtp_weight_name``, not through the target's
    ``checkpoint_prefix_mapper``.
    """
    n = re.sub(r"^(?:model\.)?(?:language_model\.)?", "", ckpt_module)
    m = re.match(r"^mtp\.layers\.(\d+)\.(.*)$", n)
    if m:
        return f"mtp.layers.{start_idx + int(m.group(1))}.{m.group(2)}"
    return n


def mtp_vllm_fused_prefix(ckpt_module: str, start_idx: int = MTP_START_LAYER_IDX) -> str:
    """As ``mtp_vllm_prefix`` but with packed_modules_mapping fusion applied."""
    p = mtp_vllm_prefix(ckpt_module, start_idx)
    head, _, proj = p.rpartition(".")
    return f"{head}.{MTP_FUSE[proj]}" if proj in MTP_FUSE else p


def select_mtp_modules(index_map: dict, tier: str = "drafter-dense") -> list[str]:
    """Drafter dense module names this tier converts, from the weight map.

    Membership is decided on ``<m>.weight`` and the module is REFUSED if it also
    carries a ``<m>.weight_scale_inv`` (i.e. it is blockwise fp8, not bf16) --
    this tier is bf16-source only.
    """
    pats = [re.compile(p) for p in MTP_TIERS[tier]]
    mods = set()
    for name in index_map:
        if not name.endswith(".weight"):
            continue
        m = name[: -len(".weight")]
        if any(p.match(m) for p in pats):
            if m + ".weight_scale_inv" in index_map:
                raise ValueError(
                    f"{m} carries weight_scale_inv: it is blockwise fp8, not "
                    "bf16; the drafter-dense tier refuses it"
                )
            mods.add(m)
    return sorted(mods)


def check_mtp_fused_uniformity(mods: list[str]) -> list[str]:
    """All shards of mtp.* qkv_proj / gate_up_proj must move together (#40252).

    ``is_layer_gptq_quantized`` raises "Detected some but not all shards of ..."
    when a fused module is half converted, so this is a build-time refusal.
    """
    return check_fused_uniformity(mods)


def mtp_step_bytes(
    shapes: dict, group_size: int, passes: int = MTP_DRAFT_PASSES
) -> dict:
    """Byte arithmetic for the drafter-dense tier.

    ``shapes`` maps checkpoint module name -> (out_features, in_features).
    Every converted weight is read once per draft pass, and there are `passes`
    draft passes per engine step.
    """
    params = sum(o * i for (o, i) in shapes.values())
    bf16_pass = params * 2
    b_run = bits_per_weight(group_size)
    b_disk = bits_per_weight(group_size, on_disk=True)
    int4_pass = params * b_run / 8.0
    disk_bytes = params * b_disk / 8.0
    saved_pass = bf16_pass - int4_pass
    saved_step = saved_pass * passes
    return {
        "modules": len(shapes),
        "params": params,
        "bits_per_weight_runtime": b_run,
        "bits_per_weight_disk": b_disk,
        "bf16_bytes_per_pass": bf16_pass,
        "int4_bytes_per_pass": int4_pass,
        "int4_bytes_on_disk": disk_bytes,
        "saved_bytes_per_pass": saved_pass,
        "draft_passes_per_step": passes,
        "saved_bytes_per_step": saved_step,
        "saved_MiB_per_step": saved_step / 2**20,
        "saved_GB_per_step": saved_step / 1e9,
        "ms_at_235GBs": saved_step / 235e9 * 1e3,
        "ms_at_219GBs": saved_step / 219e9 * 1e3,
        "resident_saved_MiB": saved_pass / 2**20,
    }


# --------------------------------------------------------------------------
# Iteration 4A / thread L3: the hyper-connection tier ("hc").
#
# The HC projections are real vLLM Linears but they are constructed with
# ``quant_config=None`` (image ``vllm/models/qwen3_8_flash_next/nvidia/
# hyperconnection.py:101,112,121``), so ``LinearBase.__init__`` takes the
# ``quant_config is None -> UnquantizedLinearMethod()`` branch
# (``linear.py:258-259``) and ``AutoGPTQConfig.get_quant_method`` -- the hook
# ``vllm_fp8_hybrid.py`` overrides -- is never called for them.  A converted
# checkpoint alone therefore cannot move them; the ``hc`` tier only makes sense
# together with ``patch_hc_fp8.py`` (VLLM_HC_FP8=1).
#
# What moves, and what cannot:
#   * ``input_mix_weight_down`` + ``block_inject_weight`` -> ONE pre-merged
#     [336, 10240] blockwise-fp8 tensor named after the runtime's merged
#     module.  K = 10240 = 80 x 128, so the activation quantiser and the
#     production CutlassFp8BlockScaledMMKernel both accept it.
#   * the two final mixers' ``input_mix_weight_down`` [320, 10240] -- same K.
#   * ``input_mix_weight_up`` [10240, 320] CANNOT move: K = 320 and
#     ``per_token_group_quant_fp8`` asserts ``x.shape[-1] % group_size == 0``
#     (``fp8_utils.py:563``), while ``CutlassFp8BlockScaledMMKernel``
#     ``can_implement`` demands ``GroupShape(1, 128)`` exactly
#     (``kernels/linear/scaled_mm/cutlass.py:304-310``).  It stays bf16.
#   * the drafter's three HC modules (``mtp.*``) stay bf16: their weight names
#     go through ``mtp.py:_remap_mtp_weight_name`` and the layer index is
#     rewritten by ``mtp_start_layer_idx`` (``mtp.py:166,204``), a separate
#     load path that this tier deliberately does not touch -- and drafter
#     damage shows up as a lower acceptance rate, not as a quality signal any
#     eval would catch.
# --------------------------------------------------------------------------
HC_BLOCK = (128, 128)
HC_MERGED_SUFFIX = "input_mix_weight_down_block_inject"
HC_DOWN_SUFFIX = "input_mix_weight_down"
HC_INJECT_SUFFIX = "block_inject_weight"
HC_UP_SUFFIX = "input_mix_weight_up"

HC_TIERS = {
    "hc": "target-model hyper-connection down GEMMs -> blockwise fp8 "
    "(96 merged down+inject + 1 final mixer); up projections and every "
    "mtp.* module stay bf16",
}


def hc_pad_size(lora_rank: int, hc_count: int) -> int:
    """hyperconnection.py:94 -- ``(-(lora_rank + hc_count)) % 16``."""
    return (-(lora_rank + hc_count)) % 16


def select_hc_groups(index_map: dict, tier: str = "hc") -> list[dict]:
    """The hyper-connection modules this tier converts, as merge groups.

    Each group is ``{"kind", "prefix", "out_name", "srcs", "shape"}``.  A
    ``combine`` group merges ``input_mix_weight_down`` + ``block_inject_weight``
    (+ zero padding) into the runtime's merged module name; a ``mix`` group
    rewrites a final mixer's ``input_mix_weight_down`` in place.
    """
    if tier not in HC_TIERS:
        raise ValueError(f"unknown hc tier {tier!r}; known: {sorted(HC_TIERS)}")
    groups = []
    for name in sorted(index_map):
        if not name.endswith("." + HC_DOWN_SUFFIX + ".weight"):
            continue
        prefix = name[: -len("." + HC_DOWN_SUFFIX + ".weight")]
        if "hyper_connection" not in prefix:
            continue
        if prefix.startswith("mtp.") or ".mtp." in prefix:
            continue  # drafter: see the comment above
        inject = f"{prefix}.{HC_INJECT_SUFFIX}.weight"
        if inject in index_map:
            groups.append(
                {
                    "kind": "combine",
                    "prefix": prefix,
                    "out_name": f"{prefix}.{HC_MERGED_SUFFIX}",
                    "srcs": [name, inject],
                }
            )
        else:
            groups.append(
                {
                    "kind": "mix",
                    "prefix": prefix,
                    "out_name": f"{prefix}.{HC_DOWN_SUFFIX}",
                    "srcs": [name],
                }
            )
    return groups


def convert_hc_group(
    group: dict,
    w_down: np.ndarray,
    w_inject: np.ndarray | None,
    lora_rank: int,
    hc_count: int,
    block: tuple[int, int] = HC_BLOCK,
) -> dict:
    """One HC group -> {weight uint8 e4m3, weight_scale_inv f32, stats}.

    ``w_down`` is [lora_rank, hyper_hidden] and ``w_inject`` (combine groups
    only) is [hc_count, hyper_hidden], both already float32.  The merged tensor
    is [lora_rank + hc_count + pad, hyper_hidden] with the pad rows ZERO --
    the runtime discards that slice (``hyperconnection.py:141``) and today
    leaves it uninitialised, so zeros are both safe and reproducible.
    """
    name = group["out_name"]
    w_down = np.ascontiguousarray(np.asarray(w_down, dtype=np.float32))
    assert_sane_magnitude(w_down, name + " (down)")
    if group["kind"] == "combine":
        if w_inject is None:
            raise ValueError(f"{name}: combine group needs block_inject_weight")
        w_inject = np.ascontiguousarray(np.asarray(w_inject, dtype=np.float32))
        assert_sane_magnitude(w_inject, name + " (inject)")
        if w_down.shape[0] != lora_rank or w_inject.shape[0] != hc_count:
            raise ValueError(
                f"{name}: expected down [{lora_rank}, *] and inject "
                f"[{hc_count}, *], got {w_down.shape} and {w_inject.shape}"
            )
        if w_down.shape[1] != w_inject.shape[1]:
            raise ValueError(f"{name}: down/inject hidden size mismatch")
        pad = hc_pad_size(lora_rank, hc_count)
        hidden = w_down.shape[1]
        w = np.zeros((lora_rank + hc_count + pad, hidden), dtype=np.float32)
        w[:lora_rank] = w_down
        w[lora_rank : lora_rank + hc_count] = w_inject
        inject_rows = (lora_rank, lora_rank + hc_count)
    else:
        w = w_down
        inject_rows = None

    out_f, in_f = w.shape
    if in_f % block[1]:
        raise ValueError(
            f"{name}: in_features {in_f} is not a multiple of block_k "
            f"{block[1]} -- per_token_group_quant_fp8 would assert "
            "(fp8_utils.py:563)"
        )
    q, scale_inv = quantize_blockwise_fp8(w, block)
    w_hat = dequant_blockwise_fp8(q, scale_inv, block[0]) if block[0] == block[1] else None
    if w_hat is None:  # pragma: no cover - only reachable for a non-square block
        raise ValueError("non-square fp8 blocks are not supported by the verifier")

    def _sqnr(a, b):
        den = float(np.sqrt((a.astype(np.float64) ** 2).sum())) or 1.0
        rel = float(np.sqrt(((b - a).astype(np.float64) ** 2).sum())) / den
        return (float(20.0 * np.log10(1.0 / rel)) if rel > 0 else float("inf")), rel

    sqnr_db, rel_fro = _sqnr(w, w_hat)
    stats = {
        "name": name,
        "kind": group["kind"],
        "out_features": out_f,
        "in_features": in_f,
        "block": list(block),
        # what leaves the checkpoint (the pad rows are not stored today) vs
        # what the GEMM actually reads per launch (the padded merged param).
        "ckpt_bf16_bytes": int(w_down.size * 2)
        + (int(w_inject.size) * 2 if group["kind"] == "combine" else 0),
        "resident_bf16_bytes": out_f * in_f * 2,
        "fp8_bytes": int(q.nbytes + scale_inv.nbytes),
        "rel_fro_err": rel_fro,
        "sqnr_db": sqnr_db,
        "w_absmax": float(np.abs(w).max()),
        "scale_blocks": list(scale_inv.shape),
        "degenerate_blocks": int((scale_inv == 1.0).sum())
        if float(np.abs(w).max()) > 0
        else 0,
    }
    if inject_rows is not None:
        r0, r1 = inject_rows
        stats["inject_sqnr_db"], stats["inject_rel_fro_err"] = _sqnr(
            w[r0:r1], w_hat[r0:r1]
        )
        stats["down_sqnr_db"], _ = _sqnr(w[:r0], w_hat[:r0])
        # The injection rows share a 128-row scale block with down rows
        # 256..319 and the zero pad.  If their magnitudes differ a lot the
        # shared scale costs the small matrix precision -- report the ratio.
        blk = w[(r0 // block[0]) * block[0] : r1]
        stats["inject_block_absmax_ratio"] = float(
            np.abs(blk).max() / max(np.abs(w[r0:r1]).max(), 1e-30)
        )
    return {"weight": q, "weight_scale_inv": scale_inv, "stats": stats}


# --------------------------------------------------------------------------
# Byte arithmetic for runtime and on-disk storage.
# --------------------------------------------------------------------------
def bits_per_weight(group_size: int, on_disk: bool = False) -> float:
    """Runtime vs on-disk bits/weight for int4 uint4b8 at this group size.

    ``uint4b8`` is a *biased* type, so GPTQ-Marlin never reads ``qzeros`` for a
    symmetric checkpoint -- but the file still carries it.  Runtime is
    4 + 16/G (qweight + fp16 scales); disk adds the 4/G bits of qzeros.
    """
    runtime = 4.0 + 16.0 / group_size
    return runtime + (4.0 / group_size if on_disk else 0.0)


# --------------------------------------------------------------------------
# fp8 e4m3fn
# --------------------------------------------------------------------------
def e4m3fn_lut() -> np.ndarray:
    """256-entry decode table for float8_e4m3fn (1-4-3, bias 7, no inf)."""
    lut = np.zeros(256, dtype=np.float32)
    for b in range(256):
        sign = -1.0 if (b >> 7) & 1 else 1.0
        exp = (b >> 3) & 0xF
        man = b & 0x7
        if exp == 0:
            val = (man / 8.0) * (2.0**-6)
        elif exp == 0xF and man == 0x7:
            val = float("nan")
        else:
            val = (1.0 + man / 8.0) * (2.0 ** (exp - 7))
        lut[b] = np.float32(sign * val)
    return lut


_E4M3_LUT = e4m3fn_lut()
E4M3_MAX = 448.0


def bf16_to_f32(raw_u16: np.ndarray) -> np.ndarray:
    """BF16 stored as uint16 -> float32 (exact: bf16 is the top half of f32)."""
    return (raw_u16.astype(np.uint32) << np.uint32(16)).view(np.float32)


def dequant_blockwise_fp8(
    w_fp8_u8: np.ndarray, scale_inv: np.ndarray, block: int = 128
) -> np.ndarray:
    """[out, in] uint8 e4m3 + [ceil(out/b), ceil(in/b)] f32 -> [out, in] float32.

    vLLM's blockwise-fp8 convention (``Fp8LinearMethod`` with
    ``weight_block_size=[128, 128]``): the stored value is the *multiplier*,
    i.e. ``w = fp8(w) * weight_scale_inv``.  ``assert_sane_magnitude`` below
    checks that the result looks like a weight and not like 1/weight.
    """
    out, inn = w_fp8_u8.shape
    assert scale_inv.shape == ((out + block - 1) // block, (inn + block - 1) // block), (
        f"scale_inv {scale_inv.shape} does not match weight {w_fp8_u8.shape}"
    )
    w = _E4M3_LUT[w_fp8_u8]
    s = np.repeat(np.repeat(scale_inv.astype(np.float32), block, axis=0), block, axis=1)
    return w * s[:out, :inn]


def e4m3fn_grid() -> tuple[np.ndarray, np.ndarray]:
    """(values, codes) for every finite NON-NEGATIVE float8_e4m3fn, ascending.

    127 entries: exp 0..14 x mantissa 0..7 (exp 0 = subnormals, includes +0)
    plus exp 15 mantissa 0..6 (0x7F is NaN).  Max = 1.75 * 2**8 = 448.
    """
    vals, codes = [], []
    for b in range(0x80):  # sign 0 only
        exp, man = (b >> 3) & 0xF, b & 0x7
        if exp == 0xF and man == 0x7:
            continue  # NaN
        vals.append(float(_E4M3_LUT[b]))
        codes.append(b)
    order = np.argsort(np.asarray(vals, dtype=np.float64), kind="stable")
    return (
        np.asarray(vals, dtype=np.float32)[order],
        np.asarray(codes, dtype=np.uint8)[order],
    )


_E4M3_VALS, _E4M3_CODES = e4m3fn_grid()


def f32_to_e4m3fn(x: np.ndarray) -> np.ndarray:
    """float -> uint8 float8_e4m3fn bytes, round-to-nearest-even, saturating.

    Exact: the 127 non-negative finite e4m3fn values are enumerated from the
    same decode LUT ``dequant_blockwise_fp8`` uses, the two bracketing grid
    points are compared, and a tie goes to the even *code* -- which for a
    binary float grid is exactly IEEE ties-to-even.  |x| > 448 saturates to
    448 (vLLM clamps to +-fp8_max before its own cast, fp8_utils.py:568).
    """
    xf = np.asarray(x, dtype=np.float32)
    sign = np.signbit(xf)
    a = np.abs(xf)
    a = np.where(np.isnan(a), np.float32(0.0), a)
    a = np.minimum(a, np.float32(E4M3_MAX))

    hi = np.searchsorted(_E4M3_VALS, a, side="left")
    np.clip(hi, 1, len(_E4M3_VALS) - 1, out=hi)
    lo = hi - 1
    v_lo = _E4M3_VALS[lo]
    v_hi = _E4M3_VALS[hi]
    d_lo = a - v_lo
    d_hi = v_hi - a
    take_hi = d_hi < d_lo
    # tie -> even code (the low mantissa bit of the byte)
    tie = d_hi == d_lo
    take_hi |= tie & ((_E4M3_CODES[lo] & 1) == 1)
    code = np.where(take_hi, _E4M3_CODES[hi], _E4M3_CODES[lo]).astype(np.uint8)
    return np.where(sign, code | np.uint8(0x80), code).astype(np.uint8)


def quantize_blockwise_fp8(
    w_oi: np.ndarray, block: tuple[int, int] = (128, 128)
) -> tuple[np.ndarray, np.ndarray]:
    """[out, in] float -> ([out, in] uint8 e4m3, [ceil(out/bn), ceil(in/bk)] f32).

    The inverse of ``dequant_blockwise_fp8``: the stored f32 is the
    *multiplier* (``w ~= fp8(w) * weight_scale_inv``), so the scale is
    ``amax(block) / 448``.  The last block on each axis may be partial --
    ``create_fp8_scale_parameter`` sizes the scale tensor with ceil division
    (fp8_utils.py:1283-1287) and vLLM's kernels use ``cdiv`` throughout, so a
    dimension that is not a multiple of the block (N = 336 or 320 here) is a
    supported layout, not a special case.
    """
    bn, bk = block
    w = np.asarray(w_oi, dtype=np.float32)
    out, inn = w.shape
    nb_o = (out + bn - 1) // bn
    nb_i = (inn + bk - 1) // bk
    scale_inv = np.zeros((nb_o, nb_i), dtype=np.float32)
    q = np.zeros((out, inn), dtype=np.uint8)
    for i in range(nb_o):
        r0, r1 = i * bn, min((i + 1) * bn, out)
        for j in range(nb_i):
            c0, c1 = j * bk, min((j + 1) * bk, inn)
            blk = w[r0:r1, c0:c1]
            amax = float(np.abs(blk).max())
            if amax == 0.0:
                # an all-zero block: store 1.0 so dequant is exact and nothing
                # divides by zero (same convention as the int4 path).
                scale_inv[i, j] = np.float32(1.0)
                continue
            s = np.float32(amax / E4M3_MAX)
            scale_inv[i, j] = s
            q[r0:r1, c0:c1] = f32_to_e4m3fn(blk / s)
    return q, scale_inv


def assert_sane_magnitude(w: np.ndarray, name: str) -> None:
    """Guard against a wrong scale convention (multiply vs divide).

    A transformer projection weight has |w| well under 10. If the scales were
    meant to be divided the magnitudes would land around 1e3..1e6.
    """
    amax = float(np.abs(w).max())
    if not (1e-6 < amax < 10.0):
        raise ValueError(
            f"{name}: dequantised |w|max = {amax:g}, outside the plausible "
            "range (1e-6, 10). The blockwise-fp8 scale convention is wrong."
        )


# --------------------------------------------------------------------------
# int4 symmetric, vLLM uint4b8 convention
# --------------------------------------------------------------------------
def quantize_int4_sym(
    w_kn: np.ndarray, group_size: int = GROUP_SIZE, scale_dtype=np.float16
):
    """w_kn: float32 [K, N] (K = in_features). Returns (q_kn uint8 in [0,15],
    scales [K/G, N] scale_dtype, n_degenerate_groups).

    Transcribes vllm quant_utils.quantize_weights for scalar_types.uint4b8.
    """
    if group_size not in SUPPORTED_GROUP_SIZES:
        raise ValueError(
            f"group_size {group_size} is not one of {SUPPORTED_GROUP_SIZES} "
            "(marlin_utils.MARLIN_SUPPORTED_GROUP_SIZES)"
        )
    k, n = w_kn.shape
    assert k % group_size == 0, f"K={k} not divisible by group_size={group_size}"
    ng = k // group_size

    # [K, N] -> [G, ng*N] exactly as the reference does
    w = w_kn.reshape(ng, group_size, n).transpose(1, 0, 2).reshape(group_size, ng * n)

    max_val = w.max(axis=0, keepdims=True)
    min_val = w.min(axis=0, keepdims=True)
    s = np.maximum(
        np.abs(max_val / np.float32(UINT4B8_MAX)),
        np.abs(min_val / np.float32(UINT4B8_MIN)),
    ).astype(np.float32)

    degenerate = int((s == 0).sum())
    s_store = s.astype(scale_dtype)
    s_eff = s_store.astype(np.float32)
    # An all-zero group has scale 0; store a 1.0 so dequant is exact (q == bias)
    # and nothing divides by zero.
    zero_mask = s_eff == 0
    if degenerate:
        s_store = np.where(zero_mask, np.array(1.0, dtype=scale_dtype), s_store)
        s_eff = s_store.astype(np.float32)

    q = np.rint(w / s_eff)
    np.clip(q, UINT4B8_MIN, UINT4B8_MAX, out=q)
    q = (q + UINT4B8_BIAS).astype(np.uint8)

    # back to [K, N] / [ng, N]
    q = q.reshape(group_size, ng, n).transpose(1, 0, 2).reshape(k, n)
    s_store = s_store.reshape(ng, n)
    return np.ascontiguousarray(q), np.ascontiguousarray(s_store), degenerate


# Back-compat name (the g128-only spelling the first draft shipped).
def quantize_int4_g128_sym(w_kn, group_size=GROUP_SIZE, scale_dtype=np.float16):
    return quantize_int4_sym(w_kn, group_size, scale_dtype)


def pack_rows_int4(q_kn: np.ndarray) -> np.ndarray:
    """[K, N] uint8 values in [0,15] -> int32 [K//8, N], low nibble = lowest k.

    Transcribes vllm quant_utils.pack_rows(num_bits=4).  Independent of the
    group size: packing is along K in words of 8, grouping only affects scales.
    """
    k, n = q_kn.shape
    assert k % PACK_FACTOR == 0
    q = q_kn.astype(np.uint32)
    out = np.zeros((k // PACK_FACTOR, n), dtype=np.uint32)
    for i in range(PACK_FACTOR):
        out |= q[i::PACK_FACTOR, :] << np.uint32(4 * i)
    return out.astype(np.int32)


def unpack_rows_int4(qw: np.ndarray, size_k: int) -> np.ndarray:
    """Inverse of pack_rows_int4, for tests."""
    q = qw.astype(np.uint32)
    n = q.shape[1]
    out = np.zeros((size_k, n), dtype=np.uint8)
    for i in range(PACK_FACTOR):
        out[i::PACK_FACTOR, :] = ((q >> np.uint32(4 * i)) & np.uint32(0xF)).astype(
            np.uint8
        )
    return out


def dequant_int4_sym(
    qw: np.ndarray, scales: np.ndarray, size_k: int, group_size: int = GROUP_SIZE
) -> np.ndarray:
    """int32 qweight + scales -> float32 [K, N] (reference dequant, for tests)."""
    q = unpack_rows_int4(qw, size_k).astype(np.float32) - UINT4B8_BIAS
    s = np.repeat(scales.astype(np.float32), group_size, axis=0)
    return q * s


def dequant_int4_g128_sym(qw, scales, size_k, group_size=GROUP_SIZE):
    return dequant_int4_sym(qw, scales, size_k, group_size)


def qzeros_int4(size_k: int, size_n: int, group_size: int = GROUP_SIZE) -> np.ndarray:
    """The constant qzeros block the checkpoint's own int4 tensors carry.

    Shape is [K/G, N/8] -- it scales with the group size, the *word* does not:
    ``uint4b8`` is biased, so every nibble is ``zero_point - 1 = 7`` whatever G.
    """
    assert size_n % PACK_FACTOR == 0
    assert size_k % group_size == 0
    return np.full(
        (size_k // group_size, size_n // PACK_FACTOR),
        np.int32(QZEROS_WORD_INT4),
        dtype=np.int32,
    )


def convert_module(
    w_fp8_u8: np.ndarray | None,
    scale_inv: np.ndarray | None,
    name: str = "<module>",
    group_size: int = GROUP_SIZE,
    w_bf16: np.ndarray | None = None,
    src_bytes: int | None = None,
) -> dict:
    """One side-layer module -> the three GPTQ tensors + stats.

    Either give the blockwise-fp8 pair (``w_fp8_u8``, ``scale_inv``) or a
    dequantised/original float weight ``w_bf16`` of shape [out, in].
    """
    if w_bf16 is not None:
        w = np.asarray(w_bf16, dtype=np.float32)
        out_f, in_f = w.shape
        src = "bf16"
    else:
        assert w_fp8_u8 is not None and scale_inv is not None
        out_f, in_f = w_fp8_u8.shape
        w = dequant_blockwise_fp8(w_fp8_u8, scale_inv)
        src = "fp8"
    if in_f % group_size:
        raise ValueError(
            f"{name}: in_features {in_f} not a multiple of group_size {group_size}"
        )
    if out_f % PACK_FACTOR:
        raise ValueError(f"{name}: out_features {out_f} not a multiple of 8")

    assert_sane_magnitude(w, name)
    w_kn = np.ascontiguousarray(w.T)  # [in, out] = [K, N]

    q, scales, degen = quantize_int4_sym(w_kn, group_size)
    qweight = pack_rows_int4(q)
    qzeros = qzeros_int4(in_f, out_f, group_size)

    w_hat = dequant_int4_sym(qweight, scales, in_f, group_size)
    err = w_hat - w_kn
    denom = float(np.sqrt((w_kn.astype(np.float64) ** 2).sum())) or 1.0
    rel_fro = float(np.sqrt((err.astype(np.float64) ** 2).sum())) / denom
    sqnr_db = float(20.0 * np.log10(1.0 / rel_fro)) if rel_fro > 0 else float("inf")

    # ITER6: `src_bytes` lets a caller state the true source size.  The default
    # keeps 4A's convention exactly (fp8-equivalent bytes, so a --bf16-store
    # build still reports the saving against the fp8 the checkpoint ships);
    # the drafter-dense tier passes out_f * in_f * 2, the real bf16 size.
    if src_bytes is None:
        src_bytes = (
            int(w_fp8_u8.nbytes + scale_inv.nbytes) if src == "fp8" else out_f * in_f
        )
    return {
        "qweight": qweight,
        "scales": scales,
        "qzeros": qzeros,
        "stats": {
            "name": name,
            "source": src,
            "group_size": group_size,
            "out_features": out_f,
            "in_features": in_f,
            "fp8_bytes": src_bytes,  # "source bytes"; name kept for 4A reports
            "src_bytes": src_bytes,
            "int4_bytes": int(qweight.nbytes + scales.nbytes + qzeros.nbytes),
            "rel_fro_err": rel_fro,
            "sqnr_db": sqnr_db,
            "degenerate_groups": degen,
            "w_absmax": float(np.abs(w_kn).max()),
        },
    }


# --------------------------------------------------------------------------
# safetensors I/O (header-only reads, streaming writes -- no external deps)
# --------------------------------------------------------------------------
_ST_DTYPE_NP = {
    "F32": np.float32,
    "F16": np.float16,
    "BF16": np.uint16,
    "I32": np.int32,
    "I64": np.int64,
    "F8_E4M3": np.uint8,
    "U8": np.uint8,
}
_NP_ST_DTYPE = {
    np.dtype(np.float32): "F32",
    np.dtype(np.float16): "F16",
    np.dtype(np.int32): "I32",
    np.dtype(np.uint8): "U8",
    # numpy has no bfloat16; the reader already carries BF16 as uint16
    # (``_ST_DTYPE_NP``), so the writer uses the same convention.
    np.dtype(np.uint16): "BF16",
}


def read_st_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def read_tensor(path: str, name: str, hdr=None, base=None) -> np.ndarray:
    if hdr is None:
        hdr, base = read_st_header(path)
    info = hdr[name]
    s, e = info["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(base + s)
        raw = fh.read(e - s)
    dt = _ST_DTYPE_NP[info["dtype"]]
    return np.frombuffer(raw, dtype=dt).reshape(info["shape"]).copy()


def write_safetensors(path: str, tensors: dict[str, np.ndarray]) -> None:
    """Write a .safetensors shard. Tensors are written in the given order."""
    hdr = {}
    off = 0
    for name, arr in tensors.items():
        arr = np.ascontiguousarray(arr)
        nbytes = arr.nbytes
        hdr[name] = {
            "dtype": _NP_ST_DTYPE[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [off, off + nbytes],
        }
        off += nbytes
    blob = json.dumps(hdr, separators=(",", ":")).encode("utf-8")
    pad = (-len(blob)) % 8
    blob += b" " * pad
    tmp = path + ".partial"
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for name in hdr:
            fh.write(np.ascontiguousarray(tensors[name]).tobytes())
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# the fetched-BF16 store (see fetch_bf16_side_tensors.py)
# --------------------------------------------------------------------------
class Bf16Store:
    """Read-only view of the per-tensor .bin store fetched by byte range."""

    def __init__(self, root: str):
        self.root = os.path.expanduser(root)
        with open(os.path.join(self.root, "index.json")) as fh:
            self.index = json.load(fh)["tensors"]

    def __contains__(self, tensor_name: str) -> bool:
        return tensor_name in self.index

    def load(self, tensor_name: str) -> np.ndarray:
        e = self.index[tensor_name]
        path = os.path.join(self.root, e["file"])
        raw = np.fromfile(path, dtype=np.uint8)
        if raw.nbytes != e["nbytes"]:
            raise ValueError(
                f"{tensor_name}: store holds {raw.nbytes} bytes, index says "
                f"{e['nbytes']} (re-run the fetch with --resume)"
            )
        if e["dtype"] == "BF16":
            return bf16_to_f32(raw.view(np.uint16)).reshape(e["shape"])
        if e["dtype"] == "F32":
            return raw.view(np.float32).reshape(e["shape"])
        if e["dtype"] == "F16":
            return raw.view(np.float16).astype(np.float32).reshape(e["shape"])
        raise ValueError(f"{tensor_name}: unsupported store dtype {e['dtype']}")


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def select_modules(index_map: dict, tier: str) -> list[str]:
    """Module names whose blockwise-fp8 pair this tier converts.

    ``index_map`` is the checkpoint's ``weight_map``; membership is decided by
    the presence of ``<m>.weight_scale_inv``, i.e. only genuinely blockwise-fp8
    modules can ever be selected.
    """
    pats = [re.compile(p) for p in TIERS[tier]]
    mods = set()
    for name in index_map:
        if not name.endswith(".weight_scale_inv"):
            continue
        m = name[: -len(".weight_scale_inv")]
        # The patterns are fully anchored and only allow the optional
        # "model." / "language_model." prefixes, so "mtp.layers.N...." and
        # "model.visual...." can never match.
        if any(p.match(m) for p in pats):
            mods.add(m)
    return sorted(mods)


def check_fused_uniformity(mods: list[str]) -> list[str]:
    """The #40252 trap, in reverse: both shards of every fused module must move.

    Returns a list of problems; empty means every fused group is all-or-nothing.
    """
    fused = {
        "in_proj_qkvz": ("in_proj_qkv", "in_proj_z"),
        "qkv_proj": ("q_proj", "k_proj", "v_proj"),
        "gate_up_proj": ("gate_proj", "up_proj"),
    }
    sel = set(mods)
    problems = []
    for m in sorted(mods):
        head, _, proj = m.rpartition(".")
        for fname, shards in fused.items():
            if proj in shards:
                have = [s for s in shards if f"{head}.{s}" in sel]
                if len(have) != len(shards):
                    problems.append(
                        f"{head}.{fname}: only {have} selected of {list(shards)} "
                        "-- is_layer_gptq_quantized would raise (vLLM #40252)"
                    )
                break
    return sorted(set(problems))


def run(args) -> int:
    model_dir = os.path.expanduser(args.model_dir)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tier = canonical_tier(args.tier)
    g = args.group_size
    index = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))
    wmap = index["weight_map"]

    is_mtp = tier in MTP_TIERS  # bf16 drafter-dense source
    mods = select_mtp_modules(wmap, tier) if is_mtp else select_modules(wmap, tier)
    if args.limit:
        mods = mods[: args.limit]
    problems = check_fused_uniformity(mods)
    if problems and not args.limit:
        for p in problems:
            print("FUSED-UNIFORMITY PROBLEM: " + p)
        return 1
    print(f"tier={tier} g={g}: {len(mods)} modules to convert", flush=True)

    store = Bf16Store(args.bf16_store) if args.source == "bf16" else None

    # group modules by the shard they live in so each shard is opened once
    by_shard: dict[str, list[str]] = {}
    for m in mods:
        by_shard.setdefault(wmap[m + ".weight"], []).append(m)

    report = {
        "model_dir": model_dir,
        "out_dir": out_dir,
        "tier": tier,
        "group_size": g,
        "source": args.source,
        "bits_per_weight_runtime": bits_per_weight(g),
        "bits_per_weight_disk": bits_per_weight(g, on_disk=True),
        "modules": [],
        "totals": {},
    }
    tot_fp8 = tot_int4 = 0
    written: dict[str, str] = {}  # tensor name -> new shard basename
    t0 = time.time()

    for si, (shard, names) in enumerate(sorted(by_shard.items())):
        path = os.path.join(model_dir, shard)
        hdr, base = read_st_header(path)
        outname = "iter4-int4side-%s" % shard.replace("model-", "").replace(
            "model_extra_tensors", "extra"
        )
        tensors: dict[str, np.ndarray] = {}
        for m in sorted(names):
            if is_mtp:
                # The drafter's dense weights are BF16 in the
                # checkpoint (uint16 raw), never blockwise fp8.
                info = hdr[m + ".weight"]
                if info["dtype"] != "BF16":
                    raise SystemExit(
                        f"{m}.weight is {info['dtype']}, expected BF16 -- refusing"
                    )
                raw = read_tensor(path, m + ".weight", hdr, base)
                w = bf16_to_f32(raw)
                res = convert_module(
                    None, None, m, g, w_bf16=w, src_bytes=int(raw.nbytes)
                )
            elif store is not None:
                res = convert_module(
                    None, None, m, g, w_bf16=store.load(m + ".weight")
                )
            else:
                w = read_tensor(path, m + ".weight", hdr, base)
                s = read_tensor(path, m + ".weight_scale_inv", hdr, base)
                res = convert_module(w, s, m, g)
            tensors[m + ".qweight"] = res["qweight"]
            tensors[m + ".qzeros"] = res["qzeros"]
            tensors[m + ".scales"] = res["scales"]
            st = res["stats"]
            report["modules"].append(st)
            tot_fp8 += st["fp8_bytes"]
            tot_int4 += st["int4_bytes"]
            for suffix in (".qweight", ".qzeros", ".scales"):
                written[m + suffix] = outname
            if args.verbose:
                print(
                    f"  {m}  {st['out_features']}x{st['in_features']}  "
                    f"SQNR {st['sqnr_db']:.2f} dB  degen {st['degenerate_groups']}",
                    flush=True,
                )
        write_safetensors(os.path.join(out_dir, outname), tensors)
        print(
            f"[{si + 1}/{len(by_shard)}] {outname}: {len(names)} modules, "
            f"{sum(a.nbytes for a in tensors.values()) / 2**20:.1f} MiB "
            f"({time.time() - t0:.0f}s elapsed)",
            flush=True,
        )

    sq = [m["sqnr_db"] for m in report["modules"]]
    report["totals"] = {
        "modules": len(report["modules"]),
        "fp8_bytes": tot_fp8,
        "int4_bytes": tot_int4,
        "saved_bytes": tot_fp8 - tot_int4,
        "saved_GiB": (tot_fp8 - tot_int4) / 2**30,
        "sqnr_db_min": min(sq) if sq else None,
        "sqnr_db_median": float(np.median(sq)) if sq else None,
        "degenerate_groups": sum(m["degenerate_groups"] for m in report["modules"]),
        "wall_s": time.time() - t0,
    }
    report["written_tensors"] = written
    with open(os.path.expanduser(args.report), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report["totals"], indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir")
    ap.add_argument("--out-dir")
    ap.add_argument(
        "--tier",
        choices=["all", "fallbackA", "fallbackB", "gdn", "core", "full"]
        + list(MTP_TIERS),
        default="all",
        help="; ".join(
            f"{k}: {v}" for k, v in list(TIER_DOC.items()) + list(MTP_TIER_DOC.items())
        ),
    )
    ap.add_argument(
        "--group-size", type=int, choices=list(SUPPORTED_GROUP_SIZES), default=32
    )
    ap.add_argument("--source", choices=("fp8", "bf16"), default="fp8")
    ap.add_argument("--bf16-store", default="", help="store from fetch_bf16_side_tensors.py")
    ap.add_argument("--report", default="rtn-report.json")
    ap.add_argument("--limit", type=int, default=0, help="convert only the first N")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        from test_rtn_int4_gptq import main as t

        return t()
    if not (args.model_dir and args.out_dir):
        ap.error("--model-dir and --out-dir are required unless --self-test")
    if args.source == "bf16" and not args.bf16_store:
        ap.error("--source bf16 needs --bf16-store")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
