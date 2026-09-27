#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import time

import numpy as np

import rtn_int4_gptq as R

CHUNK = 8 << 20

# --------------------------------------------------------------------------
# quantization_config.dynamic
#
# ``gptq_utils.get_dynamic_override`` walks ``dynamic`` in insertion order and
# returns on the FIRST match (a "-:" match returns False = leave unquantised).
# The new positive rules must therefore precede the inherited negative ones,
# and the inherited negatives must be KEPT after them so that everything the
# positives do not name -- the BF16 QSA indexer ``index_qk_proj``, the GDN
# ``in_proj_a`` / ``in_proj_b``, the routers, the hyper-connections -- still
# resolves to "skip".
#
# Patterns are matched with ``regex.match`` against the *vLLM* module prefix,
# i.e. after ``packed_modules_mapping`` fusion:
#     in_proj_qkvz = in_proj_qkv + in_proj_z
#     qkv_proj     = q_proj + k_proj + v_proj
#     gate_up_proj = gate_proj + up_proj
# so both the fused and the unfused spellings are listed.
#
# The routed experts are NOT named by any rule, so they fall through to the
# base ``quantization_config.group_size`` (128) -- ``get_linear_quant_method``
# and ``get_moe_quant_method`` both ``deepcopy(config)`` before
# ``override_config``, so the side layers' g32 cannot leak into them.
# ``assert_experts_keep_g128`` checks this explicitly.
# --------------------------------------------------------------------------
POSITIVE_GDN_IN = [
    r"+:.*\.linear_attn\.in_proj_qkvz$",
    r"+:.*\.linear_attn\.in_proj_qkv$",
    r"+:.*\.linear_attn\.in_proj_z$",
]
POSITIVE_GDN_OUT = [r"+:.*\.linear_attn\.out_proj$"]
POSITIVE_QSA_O = [r"+:.*\.self_attn\.o_proj$"]
POSITIVE_SHARED = [
    r"+:.*\.mlp\.shared_expert\.gate_up_proj$",
    r"+:.*\.mlp\.shared_expert\.gate_proj$",
    r"+:.*\.mlp\.shared_expert\.up_proj$",
    r"+:.*\.mlp\.shared_expert\.down_proj$",
]
POSITIVE_QSA_QKV = [
    r"+:.*\.self_attn\.qkv_proj$",
    r"+:.*\.self_attn\.q_proj$",
    r"+:.*\.self_attn\.k_proj$",
    r"+:.*\.self_attn\.v_proj$",
]

POSITIVE_BY_TIER = {
    "all": POSITIVE_GDN_IN
    + POSITIVE_GDN_OUT
    + POSITIVE_QSA_O
    + POSITIVE_SHARED
    + POSITIVE_QSA_QKV,
    "fallbackA": POSITIVE_GDN_IN + POSITIVE_GDN_OUT + POSITIVE_QSA_O + POSITIVE_SHARED,
    "fallbackB": POSITIVE_GDN_IN + POSITIVE_QSA_O + POSITIVE_SHARED,
    "gdn": POSITIVE_GDN_IN + POSITIVE_GDN_OUT,
}
POSITIVE_BY_TIER["full"] = POSITIVE_BY_TIER["all"]
POSITIVE_BY_TIER["core"] = POSITIVE_BY_TIER["fallbackA"]

# ==========================================================================
# Dense-MTP positive rules for the drafter's side layers.
#
# These INVERT the 4A MTP guard for exactly nine modules and nothing else, so
# the two rule sets are written together and the order is load-bearing:
#
#   1. the mtp positives below            (exactly the converted drafter modules)
#   2. MTP_GUARD  -:.*\bmtp\..*           (everything ELSE in mtp.* -> skip)
#   3. INDEXER_GUARD                      (the QSA indexer, both models)
#   4. the target's int4 positives        (only when an int4 tier is composed)
#   5. the checkpoint's inherited rules, in their original order, unchanged
#
# ``get_dynamic_override`` returns on the FIRST match, so (2) before (4) is what
# stops the target's ``+:.*\.self_attn\.qkv_proj$`` from reaching
# ``mtp.layers.48.self_attn.qkv_proj``, and (1) before (2) is what lets the nine
# named drafter modules through the guard.
#
# The patterns are matched against the vLLM prefix, which spells the drafter's
# decoder layer ``mtp.layers.48`` (mtp.py:166,204) -- ``\d+`` covers both that
# and the checkpoint's ``0`` so the same rule set is correct whatever
# ``mtp_num_hidden_layers`` becomes.  Both the fused and the unfused spellings
# are listed because ``packed_modules_mapping`` fuses q/k/v -> qkv_proj and
# gate/up -> gate_up_proj before the prefix reaches the rule walk.
#
# ``index_qk_proj`` is NOT named by any positive and therefore falls to
# MTP_GUARD: the ``$``-anchored suffixes cannot match it.
# ==========================================================================
POSITIVE_MTP_DENSE = [
    r"+:.*\bmtp\.fc_embedding$",
    r"+:.*\bmtp\.fc_hidden$",
    r"+:.*\bmtp\.layers\.\d+\.self_attn\.qkv_proj$",
    r"+:.*\bmtp\.layers\.\d+\.self_attn\.q_proj$",
    r"+:.*\bmtp\.layers\.\d+\.self_attn\.k_proj$",
    r"+:.*\bmtp\.layers\.\d+\.self_attn\.v_proj$",
    r"+:.*\bmtp\.layers\.\d+\.self_attn\.o_proj$",
    r"+:.*\bmtp\.layers\.\d+\.mlp\.shared_expert\.gate_up_proj$",
    r"+:.*\bmtp\.layers\.\d+\.mlp\.shared_expert\.gate_proj$",
    r"+:.*\bmtp\.layers\.\d+\.mlp\.shared_expert\.up_proj$",
    r"+:.*\bmtp\.layers\.\d+\.mlp\.shared_expert\.down_proj$",
]

# Guard rail: the MTP drafter's tensors are bf16 in this checkpoint and must
# stay unquantised whatever the positive rules say.  In 4A it is the FIRST
# rule; with the dense-MTP tier it moves behind POSITIVE_MTP_DENSE and keeps its job
# for every other mtp.* module.
MTP_GUARD = r"-:.*\bmtp\..*"
# Second guard: the QSA sparse indexer owns a BF16 index_qk_proj that the
# "self_attn" positives must never reach (indexer_qsa.py:134-141).
INDEXER_GUARD = r"-:.*\.self_attn\.indexer\..*"


def int4_rule(group_size: int) -> dict:
    return {"bits": 4, "group_size": group_size, "sym": True, "desc_act": False}


def build_dynamic(
    orig_dynamic: dict, tier: str, group_size: int, mtp_tier: str = ""
) -> dict:
    """Assemble ``quantization_config.dynamic`` in first-match order.

    `tier` is the TARGET int4 tier ("" or "hc" convert nothing through GPTQ);
    `mtp_tier` is the ITER6 drafter tier ("" or "drafter-dense").
    """
    tier = R.canonical_tier(tier)
    int4_tier = "" if tier in ("", "hc") else tier
    if not int4_tier and not mtp_tier:
        # The hc tier quantises nothing through the GPTQ path: the HC Linears
        # take quant_config=None (hyperconnection.py:101,112,121) so no rule can
        # reach them, and the inherited "-:.*hyper_connection.*" already routes
        # them to skip. Leave `dynamic` byte-identical.
        return dict(orig_dynamic)
    rule = int4_rule(group_size)
    out: dict[str, dict] = {}
    if mtp_tier:  # (1) exactly the converted drafter modules
        for pat in POSITIVE_MTP_DENSE:
            out[pat] = dict(rule)
    out[MTP_GUARD] = {}  # (2) every other mtp.* module
    out[INDEXER_GUARD] = {}  # (3) the QSA indexer, both models
    if int4_tier:  # (4) the target's side layers
        for pat in POSITIVE_BY_TIER[int4_tier]:
            out[pat] = dict(rule)
    for pat, val in orig_dynamic.items():  # (5) inherited, original order
        if pat in out:
            continue
        out[pat] = val
    return out


def _override(dynamic: dict, name: str):
    """Replay gptq_utils.get_dynamic_override's first-match walk."""
    import re as _re

    for pat, val in dynamic.items():
        if pat.startswith("-:"):
            if _re.match(pat[2:], name):
                return False
        elif _re.match(pat.removeprefix("+:"), name):
            return val
    return None


EXPERT_PROBES = [
    "model.layers.5.mlp.experts.w13_weight",
    "model.layers.5.mlp.experts.w2_weight",
    "model.layers.5.mlp.experts",
]


def assert_experts_keep_g128(dynamic: dict, base_group_size: int) -> list[str]:
    """No rule may hand the routed experts anything but the base g128."""
    problems = []
    if base_group_size != 128:
        problems.append(
            f"quantization_config.group_size is {base_group_size}, not 128 -- "
            "the routed experts would move off the 4.15625 b/w they ship at"
        )
    for probe in EXPERT_PROBES:
        v = _override(dynamic, probe)
        if v is False:
            problems.append(f"{probe} would be skipped entirely")
        elif isinstance(v, dict) and v.get("group_size", 128) != 128:
            problems.append(
                f"{probe} picked up group_size={v['group_size']} from a side-layer rule"
            )
    return problems


# ==========================================================================
# ITER6 -- the SECOND gate.
#
# ``get_linear_quant_method`` (gptq_utils.py:133-162) needs BOTH
#   (a) ``get_dynamic_override(prefix) is not False``   -- the rules above, and
#   (b) ``is_layer_gptq_quantized(prefix, modules_in_block_to_quantize,
#        packed_modules_mapping)``                      -- a SUBSTRING test.
#
# (b) is why the drafter needs an explicit ``modules_in_block_to_quantize``.
# When the config carries none, ``AutoGPTQConfig.maybe_update_config``
# (auto_gptq.py:278-303) derives it from the checkpoint's safetensors metadata:
# every param whose dtype is not F16/BF16/F32, stripped of its last component.
# That yields the drafter's modules under their CHECKPOINT names
# (``mtp.layers.0.self_attn.q_proj``) while the runtime asks about the vLLM
# prefix (``mtp.layers.48.self_attn.q_proj``) -- and
# ``"mtp.layers.0.self_attn.q_proj" in "mtp.layers.48.self_attn.q_proj"`` is
# FALSE, so every converted drafter decoder-layer module would fall through to
# ``UnquantizedLinearMethod`` and the loader would then reject its
# qweight/scales/qzeros as unexpected weights.
#
# The target escapes this because ``configure_quant_config`` calls
# ``apply_vllm_mapper`` with the target class's
# ``checkpoint_prefix_mapper = WeightsMapper(orig_to_new_prefix={"model.language_model.":
# "model."})`` (model.py:604).  ``Qwen3_8FlashNextMTP`` has NO
# ``checkpoint_prefix_mapper`` (verified by introspection in the image), so the draft
# quant config's list is never remapped, and ``_make_draft_vllm_config``
# remaps only ``ignored_layers`` / ``exclude_modules`` (mtp.py:120-133), which
# ``AutoGPTQConfig`` does not have.
#
# So the builder writes the list explicitly.  It is EXACTLY what
# ``maybe_update_config`` would derive from the built checkpoint, except that
#   * the 73,728 routed-expert entries are collapsed to the single substring
#     ``"mlp.experts."`` -- provably equivalent, because the only consumer is
#     ``is_layer_gptq_quantized`` and that is only ever called for a
#     ``LinearBase`` / ``ParallelLMHead`` (``RoutedExperts`` is dispatched to
#     ``get_moe_quant_method``, which never reads the list), and no LinearBase
#     prefix in either model contains ``mlp.experts.``; and
#   * the drafter's converted modules are ADDITIONALLY listed under the vLLM
#     layer index 48.
# ``check_target_routing_unchanged`` proves the first bullet on the real
# prefixes rather than asserting it.
# ==========================================================================
EXPERTS_COLLAPSED = "mlp.experts."
_UNQUANT_ST_DTYPES = {"F16", "BF16", "F32"}


def derive_modules_list(headers: dict) -> list[str]:
    """Replay ``AutoGPTQConfig.maybe_update_config``'s metadata scan.

    `headers` maps shard basename -> the shard's safetensors header dict.
    """
    mods = set()
    for hdr in headers.values():
        for name, info in hdr.items():
            if info["dtype"] not in _UNQUANT_ST_DTYPES:
                mods.add(name.rsplit(".", 1)[0])
    return sorted(mods)


def build_modules_in_block(derived: list[str], mtp_mods: list[str]) -> list[str]:
    """The explicit list the built config.json carries."""
    out = [m for m in derived if ".mlp.experts." not in m]
    out.append(EXPERTS_COLLAPSED)
    # the drafter's converted modules under the vLLM layer index (mtp.py:166)
    out += [R.mtp_vllm_prefix(m) for m in mtp_mods]
    return sorted(set(out))


def _is_layer_gptq_quantized(prefix: str, quantized_layers, fused_mapping) -> bool:
    """Transcription of gptq_utils.is_layer_gptq_quantized (raises the same)."""
    proj_name = prefix.split(".")[-1]
    if proj_name in fused_mapping:
        is_quantized = None
        for shard in fused_mapping[proj_name]:
            sp = prefix.replace(proj_name, shard)
            is_shard = any(layer in sp for layer in quantized_layers)
            if is_quantized is None:
                is_quantized = is_shard
            elif is_shard != is_quantized:
                raise ValueError(
                    f"Detected some but not all shards of {prefix} are quantized"
                )
    else:
        is_quantized = any(layer in prefix for layer in quantized_layers)
    return bool(is_quantized)


# packed_modules_mapping, identical on Qwen3_8FlashNextMTP and the target
# (model.py:593-602, mtp.py:356-366).
PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
    "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
    "in_proj_ba": ["in_proj_b", "in_proj_a"],
    "input_mix_weight_down_block_inject": [
        "input_mix_weight_down",
        "block_inject_weight",
        "_input_mix_padding",
    ],
}


# Qwen3_8FlashNextForConditionalGeneration.checkpoint_prefix_mapper, read out of the
# image by introspection on 2026-09-09.  configure_quant_config applies its
# unstacked form to modules_in_block_to_quantize for the TARGET config; the
# draft config gets no mapper at all (Qwen3_8FlashNextMTP has none), which is
# the whole reason the drafter needs its own layer-48 spellings.
TARGET_PREFIX_MAP = {
    "model.visual.": "visual.",
    "lm_head.": "language_model.lm_head.",
    "model.language_model.": "language_model.model.",
}


def _apply_target_mapper(names) -> list[str]:
    out = []
    for n in names:
        for old, new in TARGET_PREFIX_MAP.items():
            if n.startswith(old):
                n = new + n[len(old) :]
                break
        out.append(n)
    return out


def route(dynamic: dict, modules_list, prefix: str, target: bool):
    """Replay get_linear_quant_method's decision for one vLLM prefix.

    Returns False (skip / UnquantizedLinearMethod) or the effective rule dict.
    `target` selects whether the target's checkpoint_prefix_mapper has been applied to
    `modules_list` (it is, for the target; it is NOT, for the drafter).
    """
    lst = _apply_target_mapper(modules_list) if target else list(modules_list)
    quantized = _is_layer_gptq_quantized(prefix, lst, PACKED_MODULES_MAPPING)
    dyn = _override(dynamic, prefix)
    if dyn is False or not quantized:
        return False
    return dyn if isinstance(dyn, dict) else {}


def check_expert_collapse(derived_full, new_list, prefixes) -> list[str]:
    """The 73,728 expert entries and the one substring must agree on gate 2.

    Run on a sample: the full list is O(74k) per prefix, so this is the one
    place that pays that cost, and it is the only claim that needs it.
    """
    full = _apply_target_mapper(derived_full)
    new = _apply_target_mapper(new_list)
    problems = []
    for p in prefixes:
        if ".mtp." in p or p.startswith("mtp."):
            continue  # the drafter's entries are the deliberate addition
        a = any(e in p for e in full)
        b = any(e in p for e in new)
        if a != b:
            problems.append(f"expert collapse changes gate 2 for {p}: {a} -> {b}")
    return problems


def without_experts(names):
    return [n for n in names if ".mlp.experts." not in n and n != EXPERTS_COLLAPSED]


def check_target_routing_unchanged(
    old_dynamic: dict, old_list, new_dynamic: dict, new_list, prefixes
) -> list[str]:
    """Every non-mtp prefix must route exactly as it does in the source dir.

    Pass the two lists with the routed-expert entries removed (see
    ``without_experts``) and use ``check_expert_collapse`` for that half:
    the expert entries are O(74k) and cannot match a LinearBase prefix.
    """
    problems = []
    for p in prefixes:
        if ".mtp." in p or p.startswith("mtp."):
            continue
        try:
            a = route(old_dynamic, old_list, p, target=True)
        except ValueError as e:
            a = f"RAISE:{e}"
        try:
            b = route(new_dynamic, new_list, p, target=True)
        except ValueError as e:
            b = f"RAISE:{e}"
        if a != b:
            problems.append(f"target routing changed for {p}: {a!r} -> {b!r}")
    return problems


def check_second_gate(
    dynamic: dict, modules_list, mtp_mods: list[str], group_size: int
) -> list[str]:
    """Both gates, on the drafter's real vLLM prefixes."""
    problems = []
    want = sorted({R.mtp_vllm_fused_prefix(m) for m in mtp_mods})
    for p in want:
        try:
            v = route(dynamic, modules_list, p, target=False)
        except ValueError as e:
            problems.append(f"{p}: is_layer_gptq_quantized raises ({e})")
            continue
        if v is False:
            problems.append(
                f"{p} would NOT reach AutoGPTQLinearMethod "
                f"(dynamic={_override(dynamic, p)!r}, "
                f"second gate="
                f"{_is_layer_gptq_quantized(p, list(modules_list), PACKED_MODULES_MAPPING)})"
            )
        elif v.get("group_size") != group_size or v.get("sym") is not True:
            problems.append(f"{p} routed with {v!r}, want g{group_size} sym")
    # everything else in the drafter must still be skipped
    for p in MTP_MUST_SKIP:
        try:
            v = route(dynamic, modules_list, p, target=False)
        except ValueError as e:
            problems.append(f"{p}: is_layer_gptq_quantized raises ({e})")
            continue
        if v is not False:
            problems.append(f"{p} should be skipped by the mtp guard, got {v!r}")
    return problems


# Drafter prefixes that must resolve to "skip" whatever the tier.  The layer
# index is 48 because that is what the runtime builds (mtp.py:166,204); the
# checkpoint's own "0" spellings are probed as well so a future index change
# cannot silently open a hole.
MTP_MUST_SKIP = [
    "mtp.layers.48.self_attn.indexer.index_qk_proj",
    "mtp.layers.48.mlp.gate",
    "mtp.layers.48.mlp.shared_expert_gate",
    "mtp.layers.48.attn_hyper_connection.input_mix_weight_down_block_inject",
    "mtp.layers.48.attn_hyper_connection.input_mix_weight_up",
    "mtp.layers.48.mlp_hyper_connection.input_mix_weight_down_block_inject",
    "mtp.layers.48.mlp_hyper_connection.input_mix_weight_up",
    "mtp.hyper_connection_mixer.input_mix_weight_down_block_inject",
    "mtp.hyper_connection_mixer.input_mix_weight_up",
    "mtp.layers.48.mlp.experts",
    "mtp.embed_tokens",
    "mtp.layers.0.self_attn.indexer.index_qk_proj",
    "mtp.layers.0.mlp.gate",
]


# The served checkpoint's own quantization_config.dynamic, verbatim
# (~/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid/config.json, 2026-09-09).
SOURCE_DYNAMIC_JSON = (
    '{"+:.*lm_head$":{"bits":8},"-:.*linear_attn.*":{},'
    '"-:.*self_attn.*":{},"-:.*hyper_connection.*":{},'
    '"-:.*visual.*":{},"-:.*shared_expert.*":{},'
    '"-:.*\\\\.ple\\\\..*":{},"-:.*embed.*":{},"-:.*fc_hidden.*":{},'
    '"-:.*layers\\\\.48\\\\..*":{},"-:.*\\\\.gate$":{}}'
)

# A representative slice of what maybe_update_config derives from the served
# checkpoint (verified against the real headers on 2026-09-09: 74,030 modules,
# of which 73,728 are routed experts, 300 blockwise-fp8 side layers, one
# ple_embedding and lm_head).
SAMPLE_DERIVED_LIST = [
    "lm_head",
    "model.language_model.layers.0.linear_attn.in_proj_qkv",
    "model.language_model.layers.0.linear_attn.in_proj_z",
    "model.language_model.layers.0.linear_attn.out_proj",
    "model.language_model.layers.3.self_attn.q_proj",
    "model.language_model.layers.3.self_attn.k_proj",
    "model.language_model.layers.3.self_attn.v_proj",
    "model.language_model.layers.3.self_attn.o_proj",
    "model.language_model.layers.11.mlp.shared_expert.gate_proj",
    "model.language_model.layers.11.mlp.shared_expert.up_proj",
    "model.language_model.layers.11.mlp.shared_expert.down_proj",
    "model.language_model.layers.2.ple.ple_embedding",
    "model.language_model.layers.5.mlp.experts.0.gate_proj",
    "model.language_model.layers.5.mlp.experts.0.up_proj",
    "model.language_model.layers.5.mlp.experts.0.down_proj",
    "model.language_model.layers.5.mlp.experts.511.down_proj",
]
SAMPLE_MTP_MODULES = [
    "mtp.fc_embedding",
    "mtp.fc_hidden",
    "mtp.layers.0.mlp.shared_expert.down_proj",
    "mtp.layers.0.mlp.shared_expert.gate_proj",
    "mtp.layers.0.mlp.shared_expert.up_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.v_proj",
]
SAMPLE_TARGET_PROBES = SAMPLE_DERIVED_LIST + [
    # the real target spellings (outer prefix language_model.model.*), verified
    # against the candidate image's own log line
    #   "fp8 hybrid: RoutedExperts prefix 'language_model.model.layers.0.mlp.experts'"
    "language_model.model.layers.3.self_attn.qkv_proj",
    "language_model.model.layers.0.linear_attn.in_proj_qkvz",
    "language_model.model.layers.11.mlp.shared_expert.gate_up_proj",
    "language_model.model.layers.3.self_attn.indexer.index_qk_proj",
    "language_model.model.layers.3.mlp.gate",
    "language_model.model.layers.3.attn_hyper_connection.input_mix_weight_up",
    "language_model.model.embed_tokens",
    "language_model.lm_head",
    # and the shorter spellings the 4A tests use (the rules are prefix-agnostic)
    "model.layers.3.self_attn.qkv_proj",
    "model.layers.0.linear_attn.in_proj_qkvz",
    "model.layers.0.linear_attn.in_proj_ba",
    "model.layers.11.mlp.shared_expert.gate_up_proj",
    "model.layers.3.self_attn.indexer.index_qk_proj",
    "model.layers.3.mlp.gate",
    "model.layers.3.mlp.shared_expert_gate",
    "model.layers.3.attn_hyper_connection.input_mix_weight_down_block_inject",
    "model.layers.3.attn_hyper_connection.input_mix_weight_up",
    "model.layers.2.ple.key_proj",
    "model.embed_tokens",
    "visual.blocks.0.attn.qkv",
    "lm_head",
]


def check_dynamic_routing(
    dynamic: dict, tier: str, group_size: int, mtp_tier: str = ""
) -> list[str]:
    tier = R.canonical_tier(tier)
    problems = []

    gdn_in = [
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.7.linear_attn.in_proj_qkv",
    ]
    gdn_out = ["model.layers.7.linear_attn.out_proj"]
    qsa_o_shared = [
        "model.layers.3.self_attn.o_proj",
        "model.layers.11.mlp.shared_expert.gate_up_proj",
        "model.layers.11.mlp.shared_expert.down_proj",
    ]
    qsa_qkv = ["model.layers.3.self_attn.qkv_proj"]

    want_quant = list(gdn_in)
    want_skip = [
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.0.linear_attn.in_proj_a",
        "model.layers.3.self_attn.indexer.index_qk_proj",
        "model.layers.3.mlp.gate",
        "model.layers.3.mlp.shared_expert_gate",
        "model.layers.3.attn_hyper_connection.input_mix_weight_down_block_inject",
        "model.layers.3.attn_hyper_connection.input_mix_weight_up",
        "model.layers.2.ple.key_proj",
        "model.embed_tokens",
        "visual.blocks.0.attn.qkv",
    ]
    # ITER6: with the drafter tier on, exactly the nine converted drafter
    # modules invert the guard; without it, every mtp.* prefix still skips.
    mtp_probe = [
        "mtp.layers.48.self_attn.o_proj",
        "mtp.layers.48.self_attn.qkv_proj",
        "mtp.layers.48.mlp.shared_expert.gate_up_proj",
        "mtp.layers.48.mlp.shared_expert.down_proj",
        "mtp.fc_embedding",
        "mtp.fc_hidden",
    ]
    if mtp_tier:
        want_quant += mtp_probe
    else:
        want_skip += mtp_probe
    want_skip += MTP_MUST_SKIP + ["mtp.layers.48.linear_attn.in_proj_qkvz"]
    if tier in ("", "hc"):
        # nothing on the TARGET is routed to int4; every side layer keeps its
        # blockwise fp8 (tier "" = the ITER6 drafter-only build)
        want_skip += gdn_in + gdn_out + qsa_o_shared + qsa_qkv
        want_quant = [x for x in want_quant if x not in gdn_in]
    elif tier == "gdn":
        want_quant += gdn_out
        want_skip += qsa_o_shared + qsa_qkv
    elif tier == "fallbackB":
        want_quant += qsa_o_shared
        want_skip += gdn_out + qsa_qkv
    elif tier == "fallbackA":
        want_quant += qsa_o_shared + gdn_out
        want_skip += qsa_qkv
    else:  # all
        want_quant += qsa_o_shared + gdn_out + qsa_qkv

    for n in want_quant:
        v = _override(dynamic, n)
        if v is False or v is None:
            problems.append(f"{n} should be routed to int4, got {v!r}")
        elif v.get("group_size") != group_size:
            problems.append(
                f"{n} routed with group_size={v.get('group_size')}, want {group_size}"
            )
        elif v.get("sym") is not True or v.get("desc_act") is not False:
            problems.append(f"{n} routed with {v!r}; want sym=True desc_act=False")
    for n in want_skip:
        v = _override(dynamic, n)
        if v is not False:
            problems.append(f"{n} should be skipped, got {v!r}")
    if _override(dynamic, "lm_head") != {"bits": 8}:
        problems.append("lm_head lost its 8-bit override")
    # ITER6: rule ORDER is the whole design.  Without the drafter tier the mtp
    # guard is first (4A).  With it, the guard must sit after the mtp positives
    # and before every other rule -- in particular before the target's
    # self_attn / shared_expert positives, which would otherwise reach
    # mtp.layers.48.
    pats = list(dynamic)
    n_pos = len(POSITIVE_MTP_DENSE)
    if mtp_tier:
        if pats[:n_pos] != POSITIVE_MTP_DENSE:
            problems.append("the mtp positives must be the first rules, in order")
        elif pats[n_pos] != MTP_GUARD:
            problems.append("the mtp guard must directly follow the mtp positives")
    elif MTP_GUARD in pats:
        if pats[0] != MTP_GUARD:
            problems.append("the mtp guard must be the first rule")
    # else: `dynamic` was left byte-identical (the hc tier converts nothing
    # through the GPTQ path), so there is no rule order of ours to assert.
    return problems


# --------------------------------------------------------------------------
# a pre-packed tensor store (gptq_solve.py output)
# --------------------------------------------------------------------------
class PackedStore:
    def __init__(self, root: str):
        self.root = os.path.expanduser(root)
        with open(os.path.join(self.root, "index.json")) as fh:
            meta = json.load(fh)
        self.group_size = meta["group_size"]
        self.modules = meta["modules"]

    def get(self, module: str) -> dict:
        e = self.modules[module]
        path = os.path.join(self.root, e["file"])
        hdr, base = R.read_st_header(path)
        return {
            "qweight": R.read_tensor(path, module + ".qweight", hdr, base),
            "scales": R.read_tensor(path, module + ".scales", hdr, base),
            "qzeros": R.read_tensor(path, module + ".qzeros", hdr, base),
            "stats": e.get("stats", {"name": module}),
        }


# --------------------------------------------------------------------------
# shard rewriting
# --------------------------------------------------------------------------
def _copy_range(src, dst, start, length):
    src.seek(start)
    left = length
    while left:
        b = src.read(min(CHUNK, left))
        if not b:
            raise IOError("short read")
        dst.write(b)
        left -= len(b)


def rewrite_shard(
    src_path,
    dst_path,
    modules,
    group_size,
    packed=None,
    bf16=None,
    hc_groups=(),
    hc_drop=(),
    hc_read=None,
    hc_cfg=None,
    mtp_modules=(),
):
    """Copy every tensor of `src_path` except the fp8 pair of each module in
    `modules`, then append that module's qweight/qzeros/scales.

    `hc_groups` are the hyper-connection merge groups whose *home* shard this
    is (the shard holding their ``input_mix_weight_down.weight``); their merged
    fp8 weight + ``weight_scale_inv`` are appended here.  `hc_drop` names every
    HC source tensor that lives in THIS shard and must disappear -- which is
    not the same set, because one of the 97 combine groups (layer 13 attn) has
    its ``block_inject_weight`` in a different shard from its
    ``input_mix_weight_down``.

    Returns (new_header, per_module_stats).
    """
    hdr, base = R.read_st_header(src_path)
    drop = set(hc_drop)
    for m in modules:
        drop.add(m + ".weight")
        drop.add(m + ".weight_scale_inv")
    # ITER6: the drafter's dense weights are plain BF16 -- there is no
    # weight_scale_inv sibling to drop, and the source is read as bf16.
    for m in mtp_modules:
        info = hdr.get(m + ".weight")
        if info is None:
            raise KeyError(f"{src_path}: {m}.weight absent")
        if info["dtype"] != "BF16":
            raise ValueError(
                f"{m}.weight is {info['dtype']}, expected BF16 "
                "(the drafter-dense tier is bf16-source only)"
            )
        if m + ".weight_scale_inv" in hdr:
            raise ValueError(f"{m} carries weight_scale_inv; refusing")
        drop.add(m + ".weight")
    missing = drop - set(hdr)
    if missing:
        raise KeyError(f"{src_path}: expected tensors absent: {sorted(missing)[:3]}")

    new_tensors = {}
    new_dtypes = {}
    stats = []
    for g in hc_groups:
        w_down, w_inject = hc_read(g)
        res = R.convert_hc_group(
            g, w_down, w_inject, hc_cfg["hc_lowrank"], hc_cfg["hc_count"]
        )
        new_tensors[g["out_name"] + ".weight"] = res["weight"]
        new_dtypes[g["out_name"] + ".weight"] = "F8_E4M3"
        new_tensors[g["out_name"] + ".weight_scale_inv"] = res["weight_scale_inv"]
        stats.append(res["stats"])
    for m in sorted(modules):
        if packed is not None:
            res = packed.get(m)
        elif bf16 is not None:
            res = R.convert_module(
                None, None, m, group_size, w_bf16=bf16.load(m + ".weight")
            )
        else:
            w = R.read_tensor(src_path, m + ".weight", hdr, base)
            s = R.read_tensor(src_path, m + ".weight_scale_inv", hdr, base)
            res = R.convert_module(w, s, m, group_size)
        new_tensors[m + ".qweight"] = res["qweight"]
        new_tensors[m + ".qzeros"] = res["qzeros"]
        new_tensors[m + ".scales"] = res["scales"]
        stats.append(res["stats"])

    # The drafter's dense modules have a bf16 source.
    for m in sorted(mtp_modules):
        if packed is not None and m in getattr(packed, "modules", ()):
            res = packed.get(m)
        else:
            raw = R.read_tensor(src_path, m + ".weight", hdr, base)  # uint16
            res = R.convert_module(
                None,
                None,
                m,
                group_size,
                w_bf16=R.bf16_to_f32(raw),
                src_bytes=int(raw.nbytes),
            )
        res["stats"]["mtp"] = True
        res["stats"]["vllm_prefix"] = R.mtp_vllm_prefix(m)
        res["stats"]["vllm_fused_prefix"] = R.mtp_vllm_fused_prefix(m)
        new_tensors[m + ".qweight"] = res["qweight"]
        new_tensors[m + ".qzeros"] = res["qzeros"]
        new_tensors[m + ".scales"] = res["scales"]
        stats.append(res["stats"])

    keep = [(n, hdr[n]) for n in hdr if n not in drop]
    keep.sort(key=lambda kv: kv[1]["data_offsets"][0])

    out_hdr = {}
    off = 0
    for n, info in keep:
        ln = info["data_offsets"][1] - info["data_offsets"][0]
        out_hdr[n] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [off, off + ln],
        }
        off += ln
    for n, arr in new_tensors.items():
        out_hdr[n] = {
            "dtype": new_dtypes.get(n) or R._NP_ST_DTYPE[arr.dtype],
            "shape": list(arr.shape),
            "data_offsets": [off, off + arr.nbytes],
        }
        off += arr.nbytes

    blob = json.dumps(out_hdr, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    tmp = dst_path + ".partial"
    with open(src_path, "rb") as src, open(tmp, "wb") as dst:
        dst.write(struct.pack("<Q", len(blob)))
        dst.write(blob)
        for n, info in keep:
            s0, s1 = info["data_offsets"]
            _copy_range(src, dst, base + s0, s1 - s0)
        for n, arr in new_tensors.items():
            dst.write(np.ascontiguousarray(arr).tobytes())
    os.replace(tmp, dst_path)
    return out_hdr, stats


# --------------------------------------------------------------------------
def hc_settings(cfg: dict) -> dict:
    """hc_lowrank / hc_count / hidden_size from the checkpoint's config.json."""
    t = cfg.get("text_config", cfg)
    out = {
        "hc_lowrank": int(t["hc_lowrank"]),
        "hc_count": int(t["hc_count"]),
        "hidden_size": int(t["hidden_size"]),
        "num_hidden_layers": int(t["num_hidden_layers"]),
    }
    out["hyper_hidden_size"] = out["hc_count"] * out["hidden_size"]
    out["pad_size"] = R.hc_pad_size(out["hc_lowrank"], out["hc_count"])
    out["merged_out_features"] = out["hc_lowrank"] + out["hc_count"] + out["pad_size"]
    return out


def _ckpt_prefix_to_vllm(name: str) -> str:
    """Checkpoint tensor prefix -> the vLLM module prefix that will own it.

    ``Qwen3_8FlashNextForCausalLM.checkpoint_prefix_mapper`` is exactly
    ``WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})``
    (image ``model.py:604-606``), and the target model is constructed with
    prefix ``"model"``, so this one rewrite is the whole mapping for the
    modules this tier touches.  ``mtp.*`` is never converted (see
    ``rtn_int4_gptq.select_hc_groups``).
    """
    if name.startswith("model.language_model."):
        return "model." + name[len("model.language_model.") :]
    return name


# Per-step launch counts for the HC GEMMs, from the module structure:
#   every layer has an attn_ and an mlp_ GatedResidual(use_combine=True);
#   the target model has one final mixer (use_combine=False);
#   the drafter contributes 2 combine modules + 1 mixer on EACH of its 3 draft
#   passes -- those stay bf16 in this tier and are reported separately.
HC_DRAFT_PASSES = 3


def hc_step_bytes(hc_groups, hc_cfg, draft_passes: int = HC_DRAFT_PASSES) -> dict:
    """Bytes read per decode step by the HC GEMMs, before and after the tier."""
    layers = hc_cfg["num_hidden_layers"]
    hidden = hc_cfg["hyper_hidden_size"]
    merged_n = hc_cfg["merged_out_features"]
    lowrank = hc_cfg["hc_lowrank"]
    bn, bk = R.HC_BLOCK

    def fp8_bytes(n, k):
        return n * k + ((n + bn - 1) // bn) * ((k + bk - 1) // bk) * 4

    launches = {
        "target.merged_down_inject": (layers * 2, merged_n, hidden),
        "target.mixer_down": (1, lowrank, hidden),
        "draft.merged_down_inject": (draft_passes * 2, merged_n, hidden),
        "draft.mixer_down": (draft_passes, lowrank, hidden),
        "target.up": (layers * 2 + 1, hidden, lowrank),
        "draft.up": (draft_passes * 3, hidden, lowrank),
    }
    converted = {g["out_name"] for g in hc_groups}
    rows, before, after = {}, 0, 0
    for label, (count, n, k) in launches.items():
        b16 = count * n * k * 2
        moves = label.startswith("target.") and not label.endswith(".up")
        f8 = count * fp8_bytes(n, k) if moves else b16
        rows[label] = {
            "launches": count,
            "N": n,
            "K": k,
            "bf16_bytes": b16,
            "after_bytes": f8,
            "converted": moves,
        }
        before += b16
        after += f8
    saved = before - after
    return {
        "per_launch": rows,
        "hc_bf16_bytes_per_step": before,
        "hc_bytes_per_step_after": after,
        "saved_bytes_per_step": saved,
        "saved_ms_at_235GBs": saved / 235e9 * 1e3,
        "saved_ms_at_242GBs_measured_ceiling": saved / 241.702e9 * 1e3,
        "groups_converted": len(converted),
    }


def make_hc_reader(src_dir: str, wmap: dict):
    """Read a group's bf16 sources as float32, wherever their shards are."""
    cache: dict[str, tuple] = {}

    def hdr_of(shard):
        if shard not in cache:
            cache[shard] = R.read_st_header(os.path.join(src_dir, shard))
        return cache[shard]

    def load(name):
        shard = wmap[name]
        path = os.path.join(src_dir, shard)
        hdr, base = hdr_of(shard)
        info = hdr[name]
        raw = R.read_tensor(path, name, hdr, base)
        if info["dtype"] == "BF16":
            return R.bf16_to_f32(raw).reshape(info["shape"])
        return np.asarray(raw, dtype=np.float32)

    def read(group):
        srcs = group["srcs"]
        w_down = load(srcs[0])
        w_inject = load(srcs[1]) if len(srcs) > 1 else None
        return w_down, w_inject

    return read


def build(args) -> int:
    src_dir = os.path.realpath(os.path.expanduser(args.model_dir))
    out_dir = os.path.expanduser(args.out_dir)
    tier = R.canonical_tier(args.tier)
    hc_tier = args.hc_tier or ("hc" if tier == "hc" else "")
    # --tier drafter-dense is shorthand for "no target int4 tier,
    # drafter tier on"; --mtp-tier composes it with any existing int4 tier.
    mtp_tier = args.mtp_tier or (tier if tier in R.MTP_TIERS else "")
    int4_tier = "" if tier in ("hc",) or tier in R.MTP_TIERS else tier
    g = args.group_size
    if os.path.realpath(out_dir) == src_dir:
        raise SystemExit("refusing to write into the source model directory")
    os.makedirs(out_dir, exist_ok=True)

    index = json.load(open(os.path.join(src_dir, "model.safetensors.index.json")))
    wmap = dict(index["weight_map"])
    mods = R.select_modules(wmap, int4_tier) if int4_tier else []
    mtp_mods = R.select_mtp_modules(wmap, mtp_tier) if mtp_tier else []
    problems = R.check_fused_uniformity(mods) + R.check_fused_uniformity(mtp_mods)
    if problems:
        for p in problems:
            print("FUSED-UNIFORMITY PROBLEM: " + p)
        raise SystemExit("a fused module would be only partially converted (#40252)")
    print(
        f"tier={int4_tier or '(none)'} ({R.TIER_DOC.get(int4_tier, '')}) "
        f"group_size={g}: {len(mods)} target modules"
    )
    if mtp_tier:
        print(
            f"mtp_tier={mtp_tier} ({R.MTP_TIER_DOC[mtp_tier]}): "
            f"{len(mtp_mods)} drafter modules -> "
            f"{sorted({R.mtp_vllm_fused_prefix(m) for m in mtp_mods})}"
        )

    packed = PackedStore(args.packed_dir) if args.packed_dir else None
    if packed is not None:
        if packed.group_size != g:
            raise SystemExit(
                f"--packed-dir was built at g{packed.group_size}, --group-size is g{g}"
            )
        absent = [m for m in mods if m not in packed.modules]
        if absent:
            raise SystemExit(
                f"--packed-dir is missing {len(absent)} of {len(mods)} modules, "
                f"first: {absent[:3]}"
            )
    bf16 = R.Bf16Store(args.bf16_store) if args.bf16_store else None

    src_cfg = json.load(open(os.path.join(src_dir, "config.json")))
    hc_cfg = hc_settings(src_cfg) if hc_tier else None
    hc_groups = R.select_hc_groups(wmap, hc_tier) if hc_tier else []
    hc_home: dict[str, list[dict]] = {}
    hc_drop: dict[str, set] = {}
    for grp in hc_groups:
        hc_home.setdefault(wmap[grp["srcs"][0]], []).append(grp)
        for s in grp["srcs"]:
            hc_drop.setdefault(wmap[s], set()).add(s)
    if hc_tier:
        n_comb = sum(1 for x in hc_groups if x["kind"] == "combine")
        print(
            f"hc_tier={hc_tier} ({R.HC_TIERS[hc_tier]}): {len(hc_groups)} groups "
            f"({n_comb} merged down+inject, {len(hc_groups) - n_comb} final mixer), "
            f"merged out_features={hc_cfg['merged_out_features']} "
            f"(pad {hc_cfg['pad_size']}), block {list(R.HC_BLOCK)}"
        )
        cross = [
            g["out_name"]
            for g in hc_groups
            if len(g["srcs"]) > 1 and wmap[g["srcs"][0]] != wmap[g["srcs"][1]]
        ]
        if cross:
            print(f"  {len(cross)} group(s) merge across shards, e.g. {cross[0]}")

    by_shard: dict[str, list[str]] = {}
    for m in mods:
        by_shard.setdefault(wmap[m + ".weight"], []).append(m)
    mtp_by_shard: dict[str, list[str]] = {}
    for m in mtp_mods:
        mtp_by_shard.setdefault(wmap[m + ".weight"], []).append(m)
    all_shards = sorted({v for v in wmap.values()})
    touched = sorted(set(by_shard) | set(mtp_by_shard) | set(hc_drop) | set(hc_home))
    print(
        f"{len(touched)}/{len(all_shards)} shards rewritten, "
        f"{len(all_shards) - len(touched)} linked"
    )

    hdr_cache: dict[str, dict] = {}

    def _shape(m):
        shard = wmap[m + ".weight"]
        if shard not in hdr_cache:
            hdr_cache[shard] = R.read_st_header(os.path.join(src_dir, shard))[0]
        return tuple(hdr_cache[shard][m + ".weight"]["shape"])

    mtp_shapes = {m: _shape(m) for m in mtp_mods}
    mtp_bytes = R.mtp_step_bytes(mtp_shapes, g) if mtp_mods else None
    if args.dry_run:
        gib = sum(os.path.getsize(os.path.join(src_dir, s)) for s in touched) / 2**30
        params = sum(o * i for (o, i) in (_shape(m) for m in mods))
        saved = params * (8.002 - R.bits_per_weight(g)) / 8
        print(f"[dry-run] would rewrite {gib:.1f} GiB")
        if mods:
            print(
                f"[dry-run] target: {params / 1e6:.1f} M params move to "
                f"{R.bits_per_weight(g):.3f} b/w runtime "
                f"({R.bits_per_weight(g, True):.5f} on disk), "
                f"-{saved / 1e6:.0f} MB per decode step"
            )
        if mtp_bytes:
            print("[dry-run] drafter-dense " + json.dumps(mtp_bytes, indent=1))
            for m in mtp_mods:
                o, i = mtp_shapes[m]
                print(
                    f"[dry-run]   {m}  [{o}, {i}]  BF16 {o * i * 2 / 2**20:8.4f} MiB"
                    f"  -> int4 g{g} {o * i * R.bits_per_weight(g, True) / 8 / 2**20:8.4f}"
                    f" MiB   vllm={R.mtp_vllm_fused_prefix(m)}"
                )
        if hc_tier:
            print("[dry-run] " + json.dumps(hc_step_bytes(hc_groups, hc_cfg), indent=1))
        return 0

    report = {
        "src": src_dir,
        "out": out_dir,
        "tier": tier,
        "int4_tier": int4_tier,
        "mtp_tier": mtp_tier,
        "hc_tier": hc_tier,
        "group_size": g,
        "weight_source": "packed" if packed else ("bf16" if bf16 else "fp8-rtn"),
        "modules": [],
        "mtp_modules": [],
        "hc_modules": [],
    }
    if mtp_tier:
        report["mtp_step_bytes"] = mtp_bytes
        report["mtp_vllm_prefixes"] = sorted(
            {R.mtp_vllm_fused_prefix(m) for m in mtp_mods}
        )
    if hc_tier:
        report["hc_step_bytes"] = hc_step_bytes(hc_groups, hc_cfg)
    t0 = time.time()
    hc_read = make_hc_reader(src_dir, wmap) if hc_tier else None

    # 1. rewrite the affected shards
    for i, shard in enumerate(touched):
        homes = hc_home.get(shard, [])
        _, stats = rewrite_shard(
            os.path.join(src_dir, shard),
            os.path.join(out_dir, shard),
            by_shard.get(shard, []),
            g,
            packed=packed,
            bf16=bf16,
            hc_groups=homes,
            hc_drop=hc_drop.get(shard, ()),
            hc_read=hc_read,
            hc_cfg=hc_cfg,
            mtp_modules=mtp_by_shard.get(shard, []),
        )
        for st in stats:
            if "kind" in st:
                report["hc_modules"].append(st)
            elif st.get("mtp"):
                report["mtp_modules"].append(st)
            else:
                report["modules"].append(st)
        print(
            f"[{i + 1}/{len(touched)}] {shard}: "
            f"-{len(by_shard.get(shard, []))} fp8 modules, "
            f"-{len(mtp_by_shard.get(shard, []))} mtp bf16 modules, "
            f"+{len(homes)} hc groups ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # 2. link everything else (shards and side files); config.json is rewritten.
    #
    # HARDLINKS, not symlinks: serve-intel-ar.sh bind-mounts only $MODEL_DIR at
    # /model:ro, so a symlink pointing at the sibling source directory would
    # dangle inside the serving container. A hardlink costs no disk, keeps the
    # source file unmodified (nothing ever writes to a shard) and resolves
    # inside the mount. Falls back to a symlink across devices.
    rewritten = {"config.json", "model.safetensors.index.json"}
    touched_set = set(touched)
    n_link = 0
    for name in sorted(os.listdir(src_dir)):
        if name.startswith(".") or name in rewritten or name in touched_set:
            continue
        s = os.path.join(src_dir, name)
        d = os.path.join(out_dir, name)
        if os.path.lexists(d) or os.path.isdir(s):
            continue
        try:
            os.link(s, d)
        except OSError:
            os.symlink(os.path.realpath(s), d)
        n_link += 1
    print(f"linked {n_link} unchanged files (hardlink where possible)")

    # 3. new index
    new_map = {}
    modset = set(mods)
    mtpset = set(mtp_mods)
    hc_dropped = {s for g in hc_groups for s in g["srcs"]}
    for name, shard in wmap.items():
        mod = name.rsplit(".", 1)[0]
        if mod in modset and name.endswith((".weight", ".weight_scale_inv")):
            continue
        if mod in mtpset and name.endswith(".weight"):
            continue
        if name in hc_dropped:
            continue
        new_map[name] = shard
    for m in list(mods) + list(mtp_mods):
        for suf in (".qweight", ".qzeros", ".scales"):
            new_map[m + suf] = wmap[m + ".weight"]
    for grp in hc_groups:
        home = wmap[grp["srcs"][0]]
        for suf in (".weight", ".weight_scale_inv"):
            new_map[grp["out_name"] + suf] = home
    total = 0
    for shard in all_shards:
        p = os.path.join(out_dir, shard)
        total += os.path.getsize(os.path.realpath(p))
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": total}, "weight_map": new_map}, fh)

    # 4. new config.json
    cfg = json.load(open(os.path.join(src_dir, "config.json")))
    qc = cfg["quantization_config"]
    old_dynamic = dict(qc["dynamic"])
    # ITER6: the explicit modules_in_block_to_quantize (see the SECOND GATE
    # comment above).  Derived from the SOURCE headers, which is identical to
    # deriving it from the built directory for every target module (fp8 ->
    # int4 keeps the dtype non-float) -- with the experts collapsed and the
    # drafter's vLLM (layer 48) spellings added.
    src_headers = {
        s: R.read_st_header(os.path.join(src_dir, s))[0] for s in all_shards
    }
    derived = derive_modules_list(src_headers)
    old_list = derived  # what vLLM derives from the SOURCE directory today
    new_list = build_modules_in_block(derived, mtp_mods) if mtp_tier else None
    if "modules_in_block_to_quantize" in qc:
        # maybe_update_config only derives the list from the safetensors dtypes
        # when the config does NOT carry one; an explicit list would have to be
        # extended by hand or the new int4 side layers would fall through
        # is_layer_gptq_quantized to UnquantizedLinearMethod and the loader
        # would then reject qweight/scales/qzeros as unexpected weights.
        raise SystemExit(
            "config.json carries an explicit modules_in_block_to_quantize; "
            "extend it with the converted module names before building "
            "(auto_gptq.py:284-303, gptq_utils.get_linear_quant_method)"
        )
    qc["dynamic"] = build_dynamic(old_dynamic, int4_tier, g, mtp_tier=mtp_tier)
    if new_list is not None:
        qc["modules_in_block_to_quantize"] = new_list
    problems = check_dynamic_routing(qc["dynamic"], int4_tier, g, mtp_tier=mtp_tier)
    problems += assert_experts_keep_g128(qc["dynamic"], qc.get("group_size"))
    if mtp_tier:
        eff_list = new_list
        problems += check_second_gate(qc["dynamic"], eff_list, mtp_mods, g)
        # every prefix the two models actually build, from the checkpoint's own
        # names, so the "target routing is unchanged" claim is measured
        all_mods = sorted(
            {
                n.rsplit(".", 1)[0]
                for hdr in src_headers.values()
                for n in hdr
                if n.endswith((".weight", ".qweight"))
            }
        )
        probes = _apply_target_mapper(without_experts(all_mods))
        expert_mods = _apply_target_mapper(
            [n for n in all_mods if ".mlp.experts." in n]
        )
        expert_sample = expert_mods[:150] + expert_mods[-150:]
        expert_sample += [m + "x" for m in expert_sample[:5]]  # near misses
        expert_sample += probes[:100]
        problems += check_expert_collapse(old_list, eff_list, expert_sample)
        print(
            f"routing self-test: expert collapse agrees on {len(expert_sample)} "
            f"prefixes ({len(old_list)} full entries vs "
            f"{len(eff_list)} collapsed)"
        )
        old_list = without_experts(old_list)
        eff_list = without_experts(eff_list)
        probes += [
            "model.layers.%d.self_attn.qkv_proj" % i for i in (3, 7, 11)
        ] + [
            "model.layers.%d.mlp.shared_expert.gate_up_proj" % i for i in (0, 11)
        ] + [
            "model.layers.%d.linear_attn.in_proj_qkvz" % i for i in (0, 7)
        ] + ["lm_head"]
        if int4_tier:
            # a target int4 tier is composed: target routing is MEANT to move,
            # so only the "nothing else moved" half of the claim is checkable
            probes = [
                p
                for p in probes
                if _override(old_dynamic, p) is False
                and _override(qc["dynamic"], p) is False
            ]
        problems += check_target_routing_unchanged(
            old_dynamic, old_list, qc["dynamic"], eff_list, probes
        )
        print(
            f"routing self-test: {len(probes)} target prefixes unchanged, "
            f"{len(set(R.mtp_vllm_fused_prefix(m) for m in mtp_mods))} drafter "
            f"prefixes routed to int4 g{g}, "
            f"{len(MTP_MUST_SKIP)} drafter prefixes still skipped"
        )
    if problems:
        for p in problems:
            print("DYNAMIC-RULE PROBLEM: " + p)
        raise SystemExit("dynamic rules do not route as intended; nothing published")
    if mtp_tier:
        cfg["_iter6a3i"] = {
            "mtp_tier": mtp_tier,
            "mtp_tier_doc": R.MTP_TIER_DOC[mtp_tier],
            "group_size": g,
            "mtp_start_layer_idx": R.MTP_START_LAYER_IDX,
            "modules_converted": len(mtp_mods),
            "checkpoint_names": sorted(mtp_mods),
            "vllm_prefixes": report["mtp_vllm_prefixes"],
            "step_bytes": mtp_bytes,
            "modules_in_block_to_quantize": (
                "explicit: maybe_update_config's derived list with the 73,728 "
                "routed-expert entries collapsed to the substring "
                f"{EXPERTS_COLLAPSED!r} and the drafter's modules added under "
                "the vLLM layer index (mtp.py:166); required because "
                "is_layer_gptq_quantized substring-matches the vLLM prefix and "
                "the drafter's checkpoint names carry layer 0, not 48"
            ),
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    cfg["_iter4a"] = {
        "tier": int4_tier,
        "tier_doc": R.TIER_DOC.get(int4_tier, ""),
        "group_size": g,
        "bits_per_weight_runtime": R.bits_per_weight(g),
        "bits_per_weight_disk": R.bits_per_weight(g, on_disk=True),
        "weight_source": report["weight_source"],
        "source": src_dir,
        "modules_converted": len(mods),
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if hc_tier:
        # Read back by patch_hc_fp8.py only as documentation: the patch decides
        # per module from the safetensors metadata, never from this stamp.
        cfg["_iter4hc"] = {
            "hc_tier": hc_tier,
            "hc_tier_doc": R.HC_TIERS[hc_tier],
            "block": list(R.HC_BLOCK),
            "requires_env": "VLLM_HC_FP8=1",
            "groups": len(hc_groups),
            "merged_out_features": hc_cfg["merged_out_features"],
            "pad_size": hc_cfg["pad_size"],
            "up_projection": "left bf16: K=320 is not a multiple of block_k=128 "
            "(fp8_utils.py:563; cutlass.py:304-310)",
            "vllm_prefixes": sorted(
                _ckpt_prefix_to_vllm(g2["out_name"]) for g2 in hc_groups
            ),
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)

    if report["hc_modules"]:
        hsq = [m["sqnr_db"] for m in report["hc_modules"]]
        isq = [
            m["inject_sqnr_db"] for m in report["hc_modules"] if "inject_sqnr_db" in m
        ]
        report["hc_totals"] = {
            "groups": len(report["hc_modules"]),
            "resident_bf16_GiB": sum(
                m["resident_bf16_bytes"] for m in report["hc_modules"]
            )
            / 2**30,
            "fp8_GiB": sum(m["fp8_bytes"] for m in report["hc_modules"]) / 2**30,
            "sqnr_db_min": min(hsq),
            "sqnr_db_median": float(np.median(hsq)),
            "inject_sqnr_db_min": min(isq) if isq else None,
            "inject_sqnr_db_median": float(np.median(isq)) if isq else None,
            "inject_block_absmax_ratio_max": max(
                (
                    m["inject_block_absmax_ratio"]
                    for m in report["hc_modules"]
                    if "inject_block_absmax_ratio" in m
                ),
                default=None,
            ),
            "step_bytes": report.get("hc_step_bytes"),
        }
        print("hc: " + json.dumps(report["hc_totals"], indent=1))

    sq = [m["sqnr_db"] for m in report["modules"] if "sqnr_db" in m]
    fp8 = sum(m.get("fp8_bytes", 0) for m in report["modules"])
    i4 = sum(m.get("int4_bytes", 0) for m in report["modules"])
    report["totals"] = {
        "modules": len(report["modules"]),
        "fp8_GiB": fp8 / 2**30,
        "int4_GiB": i4 / 2**30,
        "saved_GiB": (fp8 - i4) / 2**30,
        "sqnr_db_min": min(sq) if sq else None,
        "sqnr_db_median": float(np.median(sq)) if sq else None,
        "degenerate_groups": sum(
            m.get("degenerate_groups", 0) for m in report["modules"]
        ),
        "wall_s": time.time() - t0,
        "dir_bytes_new": sum(
            os.path.getsize(os.path.join(out_dir, s))
            for s in touched
            if not os.path.islink(os.path.join(out_dir, s))
        ),
    }
    if report["mtp_modules"]:
        msq = [m["sqnr_db"] for m in report["mtp_modules"]]
        bf = sum(m["src_bytes"] for m in report["mtp_modules"])
        mi4 = sum(m["int4_bytes"] for m in report["mtp_modules"])
        report["mtp_totals"] = {
            "modules": len(report["mtp_modules"]),
            "bf16_bytes_per_pass": bf,
            "int4_bytes_on_disk": mi4,
            "saved_bytes_per_pass": bf - mi4,
            "draft_passes_per_step": R.MTP_DRAFT_PASSES,
            "saved_MiB_per_step": (bf - mi4) * R.MTP_DRAFT_PASSES / 2**20,
            "saved_GB_per_step": (bf - mi4) * R.MTP_DRAFT_PASSES / 1e9,
            "resident_saved_MiB": (bf - mi4) / 2**20,
            "sqnr_db_min": min(msq),
            "sqnr_db_median": float(np.median(msq)),
            "degenerate_groups": sum(
                m["degenerate_groups"] for m in report["mtp_modules"]
            ),
            "step_bytes_model": report.get("mtp_step_bytes"),
        }
        print("drafter-dense: " + json.dumps(report["mtp_totals"], indent=1))
    name = "dense-mtp-build-report.json" if mtp_tier else "iter4a-build-report.json"
    with open(os.path.join(out_dir, name), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report["totals"], indent=1))
    return verify(
        out_dir,
        src_dir,
        int4_tier,
        g,
        sample=args.verify_sample,
        hc_tier=hc_tier,
        mtp_tier=mtp_tier,
    )


# --------------------------------------------------------------------------
def _sha_tensor(path, name, hdr, base):
    info = hdr[name]
    s0, s1 = info["data_offsets"]
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        fh.seek(base + s0)
        left = s1 - s0
        while left:
            b = fh.read(min(CHUNK, left))
            h.update(b)
            left -= len(b)
    return h.hexdigest()


def verify(
    out_dir, src_dir, tier, group_size, sample=6, hc_tier="", mtp_tier=""
) -> int:
    """Structural + byte-level checks on the built directory."""
    out_dir = os.path.expanduser(out_dir)
    src_dir = os.path.realpath(os.path.expanduser(src_dir))
    tier = R.canonical_tier(tier)
    tier = "" if tier in R.MTP_TIERS else tier
    bad = []
    idx = json.load(open(os.path.join(out_dir, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    qc = cfg["quantization_config"]
    dyn = qc["dynamic"]
    if not mtp_tier and cfg.get("_iter6a3i"):
        mtp_tier = cfg["_iter6a3i"]["mtp_tier"]
    bad += check_dynamic_routing(dyn, tier, group_size, mtp_tier=mtp_tier)
    bad += assert_experts_keep_g128(dyn, qc.get("group_size"))
    if mtp_tier:
        # ITER6: the explicit list is REQUIRED, not forbidden (see the SECOND
        # GATE comment); check it is present, complete and correct.
        lst = qc.get("modules_in_block_to_quantize")
        if not lst:
            bad.append(
                "config.json is missing modules_in_block_to_quantize; the "
                "drafter's modules would fall through is_layer_gptq_quantized"
            )
        else:
            mtp_ck = sorted(cfg.get("_iter6a3i", {}).get("checkpoint_names", []))
            bad += check_second_gate(dyn, lst, mtp_ck, group_size)
            src_headers = {
                s: R.read_st_header(os.path.join(src_dir, s))[0]
                for s in sorted(set(json.load(
                    open(os.path.join(src_dir, "model.safetensors.index.json"))
                )["weight_map"].values()))
            }
            derived = derive_modules_list(src_headers)
            src_cfg = json.load(open(os.path.join(src_dir, "config.json")))
            all_mods = sorted(
                {
                    n.rsplit(".", 1)[0]
                    for hdr in src_headers.values()
                    for n in hdr
                    if n.endswith((".weight", ".qweight"))
                }
            )
            probes = _apply_target_mapper(without_experts(all_mods))
            expert_mods = _apply_target_mapper(
                [n for n in all_mods if ".mlp.experts." in n]
            )
            exp_sample = expert_mods[:150] + expert_mods[-150:] + probes[:100]
            bad += check_expert_collapse(derived, lst, exp_sample)
            src_dyn = src_cfg["quantization_config"]["dynamic"]
            if tier:  # a target int4 tier is composed: see build()
                probes = [
                    p
                    for p in probes
                    if _override(src_dyn, p) is False and _override(dyn, p) is False
                ]
            bad += check_target_routing_unchanged(
                src_dyn, without_experts(derived), dyn, without_experts(lst), probes
            )
            print(
                f"verify: modules_in_block_to_quantize has {len(lst)} entries; "
                f"{len(probes)} target prefixes route unchanged; "
                f"expert collapse checked on {len(exp_sample)}"
            )
    elif "modules_in_block_to_quantize" in qc:
        bad.append("config.json carries modules_in_block_to_quantize")

    shards = sorted(set(wmap.values()))
    present = {}
    for shard in shards:
        p = os.path.join(out_dir, shard)
        if not os.path.exists(p):
            bad.append(f"missing shard {shard}")
            continue
        hdr, base = R.read_st_header(p)
        present[shard] = (p, hdr, base)

    # every tensor named in the index exists, and every tensor in every shard
    # is named in the index (this is what stops a stale fp8 weight reaching the
    # loader)
    in_files = set()
    for shard, (p, hdr, base) in present.items():
        in_files |= set(hdr)
    named = set(wmap)
    for n in sorted(named - in_files)[:5]:
        bad.append(f"index names {n} but no shard holds it")
    for n in sorted(in_files - named)[:5]:
        bad.append(f"shard holds {n} but the index does not name it (stale tensor)")

    # no converted module keeps its fp8 pair; each has all three int4 tensors
    mods = {n.rsplit(".", 1)[0] for n in named if n.endswith(".qweight")}
    side = {m for m in mods if ".mlp.experts." not in m and m != "lm_head"}
    for m in sorted(side):
        for suf in (".qweight", ".qzeros", ".scales"):
            if m + suf not in named:
                bad.append(f"{m} missing {suf}")
        for suf in (".weight", ".weight_scale_inv"):
            if m + suf in in_files:
                bad.append(f"{m} still carries {suf}")
    # both shards of every fused module moved together (#40252, in reverse)
    bad += R.check_fused_uniformity(sorted(side))
    print(f"verify: {len(side)} converted side modules, {len(shards)} shards")

    # byte-identity of a sample of untouched tensors
    src_idx = json.load(
        open(os.path.join(src_dir, "model.safetensors.index.json"))
    )["weight_map"]
    checked = 0
    for shard, (p, hdr, base) in sorted(present.items()):
        if os.path.islink(p):
            continue
        names = [
            n
            for n in hdr
            if not n.endswith((".qweight", ".qzeros", ".scales")) and n in src_idx
        ]
        step = max(1, len(names) // max(1, sample))
        for n in names[::step][:sample]:
            sp = os.path.join(src_dir, src_idx[n])
            shdr, sbase = R.read_st_header(sp)
            if _sha_tensor(p, n, hdr, base) != _sha_tensor(sp, n, shdr, sbase):
                bad.append(f"byte mismatch on copied tensor {n}")
            checked += 1
    print(f"verify: {checked} copied tensors byte-identical to the source")

    # shape/dtype of the new tensors: qweight [K/8, N], scales [K/G, N],
    # qzeros [K/G, N/8].  With qweight's first dim = K/8, that is
    # K/G = (K/8) * 8/G, i.e. rows_scales = rows_qweight * 8 // G.
    ratio = R.PACK_FACTOR  # 8
    n_shape_checked = 0
    for shard, (p, hdr, base) in present.items():
        for n, info in hdr.items():
            if n.endswith(".qweight") and n.rsplit(".", 1)[0] in side:
                m = n.rsplit(".", 1)[0]
                kq, nn = info["shape"]
                want_rows = kq * ratio // group_size
                if info["dtype"] != "I32":
                    bad.append(f"{n} dtype {info['dtype']} != I32")
                sc = hdr.get(m + ".scales")
                qz = hdr.get(m + ".qzeros")
                if sc and (sc["dtype"] != "F16" or sc["shape"] != [want_rows, nn]):
                    bad.append(
                        f"{m}.scales {sc['dtype']} {sc['shape']} != F16 "
                        f"[{want_rows}, {nn}] for g{group_size}"
                    )
                if qz and (
                    qz["dtype"] != "I32" or qz["shape"] != [want_rows, nn // ratio]
                ):
                    bad.append(
                        f"{m}.qzeros {qz['dtype']} {qz['shape']} != I32 "
                        f"[{want_rows}, {nn // ratio}] for g{group_size}"
                    )
                n_shape_checked += 1
    print(f"verify: {n_shape_checked} qweight/scales/qzeros shape triples at g{group_size}")

    # ---- hyper-connection tier -------------------------------------------
    if hc_tier:
        hcfg = hc_settings(cfg)
        bn, bk = R.HC_BLOCK
        want = R.select_hc_groups(src_idx, hc_tier)
        stamp = cfg.get("_iter4hc", {})
        if stamp.get("hc_tier") != hc_tier:
            bad.append(f"config.json _iter4hc says {stamp.get('hc_tier')!r}")
        n_hc = 0
        for grp in want:
            wname = grp["out_name"] + ".weight"
            sname = grp["out_name"] + ".weight_scale_inv"
            for nm in (wname, sname):
                if nm not in named:
                    bad.append(f"{nm} missing from the index")
            for src in grp["srcs"]:
                # a "mix" group rewrites its tensor under its own name, so only
                # a source that is NOT the output must have disappeared
                if src != wname and src in in_files:
                    bad.append(f"{src} survived the hc merge (stale bf16 tensor)")
            shard = wmap.get(wname)
            if shard is None or shard not in present:
                continue
            _, hdr, _ = present[shard]
            wi, si = hdr.get(wname), hdr.get(sname)
            out_f = (
                hcfg["merged_out_features"]
                if grp["kind"] == "combine"
                else hcfg["hc_lowrank"]
            )
            in_f = hcfg["hyper_hidden_size"]
            if wi and (wi["dtype"] != "F8_E4M3" or wi["shape"] != [out_f, in_f]):
                bad.append(
                    f"{wname} {wi['dtype']} {wi['shape']} != F8_E4M3 [{out_f}, {in_f}]"
                )
            want_s = [(out_f + bn - 1) // bn, (in_f + bk - 1) // bk]
            if si and (si["dtype"] != "F32" or si["shape"] != want_s):
                bad.append(f"{sname} {si['dtype']} {si['shape']} != F32 {want_s}")
            n_hc += 1
        # the up projections and every mtp.* HC tensor must be untouched bf16
        for nm in sorted(named):
            if nm.endswith("." + R.HC_UP_SUFFIX + ".weight") or (
                nm.startswith("mtp.") and "hyper_connection" in nm
            ):
                shard = wmap[nm]
                if shard in present and present[shard][1].get(nm, {}).get(
                    "dtype"
                ) not in ("BF16", None):
                    bad.append(f"{nm} is no longer BF16 -- the hc tier must not move it")
        print(f"verify: {n_hc} hc groups, fp8 [O,I] + f32 scale blocks {[bn, bk]}")

    if bad:
        print("VERIFY FAILED:")
        for b in bad[:40]:
            print("  " + b)
        return 1
    print("VERIFY OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model-dir", default="~/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid"
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--tier",
        choices=["all", "fallbackA", "fallbackB", "gdn", "core", "full", "hc"]
        + list(R.MTP_TIERS),
        default="all",
        help="'hc' converts ONLY the hyper-connection down GEMMs to blockwise "
        "fp8 and leaves every int4 decision alone; combine it with an int4 "
        "tier by passing --hc-tier hc instead. 'drafter-dense' "
        "converts ONLY the MTP drafter's dense side layers bf16 -> int4; "
        "compose it with an int4 tier by passing --mtp-tier drafter-dense",
    )
    ap.add_argument(
        "--mtp-tier",
        choices=[""] + list(R.MTP_TIERS),
        default="",
        help="also convert the MTP drafter's dense side layers to int4 "
        "and compose it with any target int4 tier",
    )
    ap.add_argument(
        "--hc-tier",
        choices=["", "hc"],
        default="",
        help="also convert the hyper-connection down GEMMs to blockwise fp8 "
        "(needs the image patch, VLLM_HC_FP8=1)",
    )
    ap.add_argument(
        "--group-size", type=int, choices=list(R.SUPPORTED_GROUP_SIZES), default=32
    )
    ap.add_argument("--packed-dir", default="", help="splice gptq_solve.py output")
    ap.add_argument("--bf16-store", default="", help="RTN from fetched BF16 instead of fp8")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--verify-sample", type=int, default=6)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        problems = []
        n = 0
        for tier in ("gdn", "fallbackB", "fallbackA", "all", "hc", ""):
            for mtp in ("", "drafter-dense"):
                for g in R.SUPPORTED_GROUP_SIZES:
                    if not tier and not mtp:
                        continue  # nothing would be converted at all
                    orig = json.loads(SOURCE_DYNAMIC_JSON)
                    d = build_dynamic(orig, tier, g, mtp_tier=mtp)
                    tag = f"[{tier or '(none)'}+{mtp or '(none)'} g{g}]"
                    problems += [
                        f"{tag} {x}"
                        for x in check_dynamic_routing(d, tier, g, mtp_tier=mtp)
                    ]
                    problems += [
                        f"{tag} {x}" for x in assert_experts_keep_g128(d, 128)
                    ]
                    if mtp:
                        lst = build_modules_in_block(
                            SAMPLE_DERIVED_LIST, SAMPLE_MTP_MODULES
                        )
                        problems += [
                            f"{tag} {x}"
                            for x in check_second_gate(
                                d, lst, SAMPLE_MTP_MODULES, g
                            )
                        ]
                        problems += [
                            f"{tag} {x}"
                            for x in check_target_routing_unchanged(
                                orig,
                                SAMPLE_DERIVED_LIST,
                                d,
                                lst,
                                _apply_target_mapper(SAMPLE_TARGET_PROBES),
                            )
                            if not tier  # only claim "unchanged" when the
                            # target tier converts nothing
                        ]
                    n += 1
        if problems:
            for p in problems:
                print("FAIL: " + p)
            return 1
        print(f"dynamic-rule self-test: {n} tier x mtp-tier x group-size cases OK")
        return 0
    if a.verify_only:
        # Prefer the stamp the build left behind, so --verify-only cannot be
        # run against the wrong tier/group and pass for the wrong reason.
        tier, g = R.canonical_tier(a.tier), a.group_size
        hc_tier = a.hc_tier or ("hc" if tier == "hc" else "")
        try:
            published = json.load(
                open(os.path.join(os.path.expanduser(a.out_dir), "config.json"))
            )
        except Exception:
            published = {}
        mtp_tier = a.mtp_tier or (tier if tier in R.MTP_TIERS else "")
        stamp = published.get("_iter4a", {})
        if stamp:
            tier, g = R.canonical_tier(stamp.get("tier", tier)), stamp.get(
                "group_size", g
            )
            hc_tier = published.get("_iter4hc", {}).get("hc_tier", hc_tier)
            mtp_tier = published.get("_iter6a3i", {}).get("mtp_tier", mtp_tier)
            print(
                f"verify: using the build stamp -- tier={tier or 'none'} "
                f"group_size={g} hc_tier={hc_tier or 'none'} "
                f"mtp_tier={mtp_tier or 'none'}"
            )
        return verify(
            a.out_dir, a.model_dir, tier, g, a.verify_sample, hc_tier, mtp_tier
        )
    return build(a)


if __name__ == "__main__":
    sys.exit(main())
