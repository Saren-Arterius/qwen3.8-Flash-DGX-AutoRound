#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np

import build_int4side_model_dir as B
import rtn_int4_gptq as R

CHECKS = 0
FAILS: list[str] = []
SQNR_BY_G: dict[int, float] = {}


def ok(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)


def eq(a, b, msg):
    ok(a == b, f"{msg}: {a!r} != {b!r}")


# --------------------------------------------------------------------------
# the miniature checkpoint
# --------------------------------------------------------------------------
TARGET_LAYERS = (0, 3)
TARGET_SHAPES = {  # blockwise fp8, as 4A's own test uses
    "linear_attn.in_proj_qkv": (256, 2560),
    "linear_attn.in_proj_z": (256, 2560),
    "linear_attn.out_proj": (256, 6144),
    "self_attn.q_proj": (256, 2560),
    "self_attn.k_proj": (512, 2560),
    "self_attn.v_proj": (512, 2560),
    "self_attn.o_proj": (256, 6144),
    "mlp.shared_expert.gate_proj": (640, 2560),
    "mlp.shared_expert.up_proj": (640, 2560),
    "mlp.shared_expert.down_proj": (256, 640),
}

# The drafter, at the real names and reduced but admissible shapes:
# in_features % 128 == 0 and % group_size == 0; out_features % 64 == 0 and
# every fused shard boundary % 8 == 0.
MTP_TARGET_SHAPES = {
    "mtp.fc_embedding": (256, 256),
    "mtp.fc_hidden": (256, 256),
    "mtp.layers.0.self_attn.q_proj": (1536, 256),
    "mtp.layers.0.self_attn.k_proj": (64, 256),
    "mtp.layers.0.self_attn.v_proj": (64, 256),
    "mtp.layers.0.self_attn.o_proj": (256, 768),
    "mtp.layers.0.mlp.shared_expert.gate_proj": (640, 256),
    "mtp.layers.0.mlp.shared_expert.up_proj": (640, 256),
    "mtp.layers.0.mlp.shared_expert.down_proj": (256, 640),
}
# every other mtp.* tensor -- these must come through untouched
MTP_KEEP_SHAPES = {
    "mtp.layers.0.mlp.gate": (512, 256),
    "mtp.layers.0.mlp.shared_expert_gate": (1, 256),
    "mtp.layers.0.self_attn.indexer.index_qk_proj": (640, 256),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down": (20, 256),
    "mtp.layers.0.attn_hyper_connection.block_inject_weight": (4, 256),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up": (256, 20),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down": (20, 256),
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight": (4, 256),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up": (256, 20),
    "mtp.hyper_connection_mixer.input_mix_weight_down": (20, 256),
    "mtp.hyper_connection_mixer.input_mix_weight_up": (256, 20),
    "mtp.layers.0.mlp.experts.0.gate_proj": (640, 256),
    "mtp.layers.0.mlp.experts.0.up_proj": (640, 256),
    "mtp.layers.0.mlp.experts.0.down_proj": (256, 640),
    "mtp.layers.0.mlp.experts.1.gate_proj": (640, 256),
    "mtp.layers.0.mlp.experts.1.up_proj": (640, 256),
    "mtp.layers.0.mlp.experts.1.down_proj": (256, 640),
}
MTP_KEEP_VECTORS = (
    "mtp.pre_fc_norm_embedding",
    "mtp.pre_fc_norm_hidden",
    "mtp.layers.0.self_attn.q_norm",
    "mtp.layers.0.self_attn.k_norm",
    "mtp.layers.0.self_attn.indexer.q_layernorm",
    "mtp.layers.0.self_attn.indexer.k_layernorm",
    "mtp.layers.0.attn_hyper_connection.hc_norm",
    "mtp.layers.0.mlp_hyper_connection.hc_norm",
    "mtp.hyper_connection_mixer.hc_norm",
)

SHARD_A = "model-00001-of-00003.safetensors"
SHARD_B = "model-00002-of-00003.safetensors"
SHARD_X = "model_extra_tensors.safetensors"


def _bf16(arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return (a.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def _fp8_pair(out_f, in_f, rng):
    codes = rng.integers(1, 250, size=(out_f, in_f), dtype=np.uint8)
    codes[codes == 0x7F] = 0x30
    codes[codes == 0xFF] = 0xB0
    sc = (
        rng.random(((out_f + 127) // 128, (in_f + 127) // 128)).astype(np.float32) * 1e-4
        + 1e-5
    )
    return codes, sc


def make_fake_model(root, rng):
    os.makedirs(root, exist_ok=True)
    ta, tb, tx = {}, {}, {}
    for L in TARGET_LAYERS:
        p = f"model.language_model.layers.{L}."
        for mod, (o, i) in TARGET_SHAPES.items():
            w, s = _fp8_pair(o, i, rng)
            ta[p + mod + ".weight"] = w
            ta[p + mod + ".weight_scale_inv"] = s
        ta[p + "self_attn.indexer.index_qk_proj.weight"] = _bf16(
            rng.standard_normal((640, 256)) * 0.02
        )
        ta[p + "mlp.gate.weight"] = _bf16(rng.standard_normal((512, 256)) * 0.02)
        # two int4 routed experts, so the miniature checkpoint exercises the
        # "collapse the expert entries" half of modules_in_block_to_quantize
        for e in (0, 1):
            for proj, (o, i) in (
                ("gate_proj", (640, 256)),
                ("up_proj", (640, 256)),
                ("down_proj", (256, 640)),
            ):
                ep = p + f"mlp.experts.{e}.{proj}"
                ta[ep + ".qweight"] = rng.integers(
                    -(2**31), 2**31 - 1, (i // 8, o), dtype=np.int32
                )
                ta[ep + ".scales"] = rng.random((i // 128, o)).astype(np.float16)
                ta[ep + ".qzeros"] = np.full(
                    (i // 128, o // 8), np.int32(0x77777777), dtype=np.int32
                )
    tb["lm_head.qweight"] = rng.integers(
        -(2**31), 2**31 - 1, (32, 256), dtype=np.int32
    )
    tb["lm_head.scales"] = rng.random((4, 256)).astype(np.float16)
    tb["lm_head.qzeros"] = np.full((4, 32), np.int32(0x7F7F7F7F), dtype=np.int32)
    tb["model.language_model.embed_tokens.weight"] = _bf16(
        rng.standard_normal((1024, 256)) * 0.02
    )

    for name, (o, i) in MTP_TARGET_SHAPES.items():
        tx[name + ".weight"] = _bf16(rng.standard_normal((o, i)) * 0.02)
    for name, (o, i) in MTP_KEEP_SHAPES.items():
        tx[name + ".weight"] = _bf16(rng.standard_normal((o, i)) * 0.02)
    for name in MTP_KEEP_VECTORS:
        tx[name + ".weight"] = _bf16(rng.standard_normal(256) * 0.01)

    R.write_safetensors(os.path.join(root, SHARD_A), ta)
    R.write_safetensors(os.path.join(root, SHARD_B), tb)
    R.write_safetensors(os.path.join(root, SHARD_X), tx)
    wmap = {k: SHARD_A for k in ta} | {k: SHARD_B for k in tb} | {k: SHARD_X for k in tx}
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": 0}, "weight_map": wmap}, fh)
    cfg = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "hc_count": 4,
            "hc_lowrank": 20,
            "hidden_size": 64,
            "num_hidden_layers": 48,
            "mtp_num_hidden_layers": 1,
        },
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
            "lm_head": True,
            "dynamic": json.loads(B.SOURCE_DYNAMIC_JSON),
        },
    }
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    for extra in ("tokenizer.json", "chat_template.jinja"):
        with open(os.path.join(root, extra), "w") as fh:
            fh.write("{}")
    return wmap


def dir_fingerprint(root):
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        out[name] = (h.hexdigest(), os.stat(p).st_size, os.stat(p).st_mtime_ns)
    return out


class Args:
    def __init__(self, **kw):
        self.model_dir = ""
        self.out_dir = ""
        self.tier = "drafter-dense"
        self.mtp_tier = ""
        self.hc_tier = ""
        self.group_size = 32
        self.packed_dir = ""
        self.bf16_store = ""
        self.dry_run = False
        self.verify_only = False
        self.verify_sample = 4
        self.self_test = False
        self.__dict__.update(kw)


# --------------------------------------------------------------------------
def check_prefix_mapping():
    eq(
        R.mtp_vllm_prefix("mtp.layers.0.self_attn.q_proj"),
        "mtp.layers.48.self_attn.q_proj",
        "layer index 0 -> 48",
    )
    eq(
        R.mtp_vllm_prefix("model.language_model.mtp.layers.0.mlp.shared_expert.up_proj"),
        "mtp.layers.48.mlp.shared_expert.up_proj",
        "leading model./language_model. stripped",
    )
    eq(R.mtp_vllm_prefix("mtp.fc_hidden"), "mtp.fc_hidden", "fc has no layer index")
    eq(
        R.mtp_vllm_fused_prefix("mtp.layers.0.self_attn.k_proj"),
        "mtp.layers.48.self_attn.qkv_proj",
        "k_proj fuses into qkv_proj",
    )
    eq(
        R.mtp_vllm_fused_prefix("mtp.layers.0.mlp.shared_expert.gate_proj"),
        "mtp.layers.48.mlp.shared_expert.gate_up_proj",
        "gate_proj fuses into gate_up_proj",
    )
    eq(
        R.mtp_vllm_fused_prefix("mtp.layers.0.mlp.shared_expert.down_proj"),
        "mtp.layers.48.mlp.shared_expert.down_proj",
        "down_proj is not fused",
    )
    eq(R.MTP_START_LAYER_IDX, 48, "mtp_start_layer_idx (mtp.py:166)")


def check_selection(wmap):
    mods = R.select_mtp_modules(wmap, "drafter-dense")
    eq(sorted(mods), sorted(MTP_TARGET_SHAPES), "selected exactly the nine targets")
    ok(
        not any("hyper_connection" in m for m in mods),
        "no hyper-connection selected (quant_config=None: hyperconnection.py:102)",
    )
    ok(
        not any(m.endswith(".mlp.gate") for m in mods),
        "the router is not selected (quant_config=None)",
    )
    ok(
        not any("indexer" in m for m in mods),
        "the QSA indexer is not selected",
    )
    ok(
        not any(".mlp.experts." in m for m in mods),
        "the routed experts are not selected (fp8 online at runtime)",
    )
    ok(not any("lm_head" in m for m in mods), "the shared lm_head is not selected")
    eq(R.check_fused_uniformity(mods), [], "fused groups are all-or-nothing")
    # half a fused group must be refused
    part = [m for m in mods if not m.endswith("k_proj")]
    ok(R.check_fused_uniformity(part), "a half-converted qkv_proj is refused (#40252)")
    # an fp8 module under mtp.* must be refused outright
    bad = dict(wmap)
    bad["mtp.fc_hidden.weight_scale_inv"] = SHARD_X
    try:
        R.select_mtp_modules(bad, "drafter-dense")
        ok(False, "an fp8 mtp module should be refused")
    except ValueError:
        ok(True, "an fp8 mtp module is refused")


def check_byte_model():
    """The byte arithmetic on the REAL drafter shapes (checkpoint headers)."""
    real = {
        "mtp.fc_embedding": (2560, 2560),
        "mtp.fc_hidden": (2560, 2560),
        "mtp.layers.0.self_attn.q_proj": (12288, 2560),
        "mtp.layers.0.self_attn.k_proj": (512, 2560),
        "mtp.layers.0.self_attn.v_proj": (512, 2560),
        "mtp.layers.0.self_attn.o_proj": (2560, 6144),
        "mtp.layers.0.mlp.shared_expert.gate_proj": (640, 2560),
        "mtp.layers.0.mlp.shared_expert.up_proj": (640, 2560),
        "mtp.layers.0.mlp.shared_expert.down_proj": (2560, 640),
    }
    d = R.mtp_step_bytes(real, 32)
    eq(d["params"], 67_829_760, "converted parameter count")
    eq(d["bf16_bytes_per_pass"], 135_659_520, "bf16 bytes per draft pass")
    eq(d["bits_per_weight_runtime"], 4.5, "g32 runtime bits/weight (4 + 16/32)")
    eq(d["bits_per_weight_disk"], 4.625, "g32 on-disk bits/weight (+ 4/32 qzeros)")
    eq(int(d["int4_bytes_on_disk"]), 39_214_080, "int4 bytes on disk")
    eq(int(d["saved_bytes_per_pass"] * 8), 8 * (135_659_520 - 38_154_240), "runtime saving/pass")
    eq(d["draft_passes_per_step"], 3, "three draft passes per step")
    ok(
        abs(d["saved_MiB_per_step"] - 279.0) < 1.5,
        f"saved MiB/step {d['saved_MiB_per_step']:.2f} (runtime b/w)",
    )
    ok(
        abs(d["ms_at_235GBs"] - 1.245) < 0.02,
        f"ms saved at 235 GB/s: {d['ms_at_235GBs']:.4f}",
    )
    # the on-disk figure (what the built directory actually shrinks by)
    disk_saved_step = (135_659_520 - 39_214_080) * 3
    ok(
        abs(disk_saved_step / 2**20 - 275.93) < 0.05,
        f"on-disk saving per step {disk_saved_step / 2**20:.2f} MiB",
    )


def check_no_target_leak(src, out, g, composed):
    """No dynamic rule and no list entry may change a TARGET decision."""
    scfg = json.load(open(os.path.join(src, "config.json")))["quantization_config"]
    ocfg = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    headers = {}
    for s in (SHARD_A, SHARD_B, SHARD_X):
        headers[s] = R.read_st_header(os.path.join(src, s))[0]
    derived = B.derive_modules_list(headers)
    # every module the miniature checkpoint actually holds, plus the fused
    # spellings the runtime builds out of them
    extra = [
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.3.self_attn.qkv_proj",
        "model.layers.0.linear_attn.in_proj_qkvz",
        "model.layers.3.linear_attn.in_proj_qkvz",
        "model.layers.0.linear_attn.in_proj_ba",
        "model.layers.0.mlp.shared_expert.gate_up_proj",
        "model.layers.3.mlp.shared_expert.gate_up_proj",
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
    probes = (
        B._apply_target_mapper(
            sorted({n.rsplit(".", 1)[0] for hdr in headers.values() for n in hdr})
        )
        + extra
    )
    if composed:
        # the target tier moves target modules on purpose; only the prefixes
        # both configs skip are claimed unchanged
        probes = [
            p
            for p in probes
            if B._override(scfg["dynamic"], p) is False
            and B._override(ocfg["dynamic"], p) is False
        ]
    lst = ocfg["modules_in_block_to_quantize"]
    problems = B.check_expert_collapse(
        derived, lst, [p for p in probes if ".mlp.experts." in p] + probes[:40]
    )
    eq(problems, [], f"g{g}: the expert-entry collapse changes no gate-2 answer")
    problems = B.check_target_routing_unchanged(
        scfg["dynamic"],
        B.without_experts(derived),
        ocfg["dynamic"],
        B.without_experts(lst),
        B.without_experts(probes),
    )
    eq(problems, [], f"g{g}: target routing unchanged over {len(probes)} prefixes")


def check_regression_witness(out):
    """Without the explicit list the decoder-layer modules fail the 2nd gate."""
    ocfg = json.load(open(os.path.join(out, "config.json")))["quantization_config"]
    dyn = ocfg["dynamic"]
    derived_only = sorted(
        set(B.SAMPLE_DERIVED_LIST) | set(MTP_TARGET_SHAPES)
    )  # what maybe_update_config yields from the BUILT dir
    fell_through = []
    for m in sorted({R.mtp_vllm_fused_prefix(x) for x in MTP_TARGET_SHAPES}):
        if B.route(dyn, derived_only, m, target=False) is False:
            fell_through.append(m)
    eq(
        sorted(fell_through),
        [
            "mtp.layers.48.mlp.shared_expert.down_proj",
            "mtp.layers.48.mlp.shared_expert.gate_up_proj",
            "mtp.layers.48.self_attn.o_proj",
            "mtp.layers.48.self_attn.qkv_proj",
        ],
        "witness: the four decoder-layer modules need the explicit list",
    )
    # ...and with the list they all pass
    eq(
        B.check_second_gate(
            dyn, ocfg["modules_in_block_to_quantize"], sorted(MTP_TARGET_SHAPES), 32
        )
        if ocfg["dynamic"]
        and json.load(open(os.path.join(out, "config.json")))["_iter6a3i"][
            "group_size"
        ]
        == 32
        else [],
        [],
        "with the explicit list every drafter prefix reaches Marlin",
    )


def check_built(src, out, g, composed):
    idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    cfg = json.load(open(os.path.join(out, "config.json")))
    qc = cfg["quantization_config"]

    # only the drafter shard was rewritten when the target tier is off
    a_same = os.stat(os.path.join(out, SHARD_A)).st_ino == os.stat(
        os.path.join(src, SHARD_A)
    ).st_ino
    b_same = os.stat(os.path.join(out, SHARD_B)).st_ino == os.stat(
        os.path.join(src, SHARD_B)
    ).st_ino
    x_same = os.stat(os.path.join(out, SHARD_X)).st_ino == os.stat(
        os.path.join(src, SHARD_X)
    ).st_ino
    ok(not x_same, f"g{g}: {SHARD_X} rewritten")
    ok(b_same, f"g{g}: {SHARD_B} hardlinked")
    if composed:
        ok(not a_same, f"g{g}: {SHARD_A} rewritten (target tier composed)")
    else:
        ok(a_same, f"g{g}: {SHARD_A} hardlinked (drafter-only build)")

    hdr, base = R.read_st_header(os.path.join(out, SHARD_X))
    for m in MTP_TARGET_SHAPES:
        ok(m + ".weight" not in hdr, f"g{g}: {m}.weight dropped")
        for suf in (".qweight", ".scales", ".qzeros"):
            ok(m + suf in hdr, f"g{g}: {m}{suf} present")
            ok(m + suf in wmap, f"g{g}: {m}{suf} in the index")
        o, i = MTP_TARGET_SHAPES[m]
        eq(hdr[m + ".qweight"]["shape"], [i // 8, o], f"g{g}: {m}.qweight shape")
        eq(hdr[m + ".scales"]["shape"], [i // g, o], f"g{g}: {m}.scales shape")
        eq(hdr[m + ".qzeros"]["shape"], [i // g, o // 8], f"g{g}: {m}.qzeros shape")
        eq(hdr[m + ".qweight"]["dtype"], "I32", f"g{g}: {m}.qweight dtype")
        eq(hdr[m + ".scales"]["dtype"], "F16", f"g{g}: {m}.scales dtype")
        qz = R.read_tensor(os.path.join(out, SHARD_X), m + ".qzeros", hdr, base)
        ok(
            bool((qz.view(np.uint32) == 0x77777777).all()),
            f"g{g}: {m}.qzeros is the uint4b8 constant word",
        )

    # every other mtp tensor is byte-identical bf16
    shdr, sbase = R.read_st_header(os.path.join(src, SHARD_X))
    n_keep = 0
    for name in list(MTP_KEEP_SHAPES) + list(MTP_KEEP_VECTORS):
        t = name + ".weight"
        ok(t in hdr, f"g{g}: {t} survived")
        eq(hdr[t]["dtype"], "BF16", f"g{g}: {t} still BF16")
        a = R.read_tensor(os.path.join(out, SHARD_X), t, hdr, base)
        b = R.read_tensor(os.path.join(src, SHARD_X), t, shdr, sbase)
        ok(bool((a == b).all()), f"g{g}: {t} byte-identical")
        n_keep += 1
    ok(n_keep == len(MTP_KEEP_SHAPES) + len(MTP_KEEP_VECTORS), "kept-tensor count")

    # index <-> shards agree in both directions
    in_files = set()
    for s in (SHARD_A, SHARD_B, SHARD_X):
        in_files |= set(R.read_st_header(os.path.join(out, s))[0])
    eq(sorted(in_files - set(wmap)), [], f"g{g}: no tensor missing from the index")
    eq(sorted(set(wmap) - in_files), [], f"g{g}: no index entry without a tensor")

    # numerics round-trip
    worst = 1e9
    for m in MTP_TARGET_SHAPES:
        o, i = MTP_TARGET_SHAPES[m]
        qw = R.read_tensor(os.path.join(out, SHARD_X), m + ".qweight", hdr, base)
        sc = R.read_tensor(os.path.join(out, SHARD_X), m + ".scales", hdr, base)
        w_hat = R.dequant_int4_sym(qw, sc, i, g)
        w = R.bf16_to_f32(
            R.read_tensor(os.path.join(src, SHARD_X), m + ".weight", shdr, sbase)
        ).T
        rel = float(np.linalg.norm(w_hat - w) / np.linalg.norm(w))
        worst = min(worst, 20 * np.log10(1.0 / rel))
    # i.i.d. Gaussian is the hardest case for RTN int4: with a group max of
    # ~2.8 sigma the step is 0.4 sigma and the RMS error 0.115 sigma, i.e. an
    # 18.8 dB ceiling by construction.  Real weights are more structured; the
    # floor here only has to catch a wrong scale/pack convention.
    ok(worst > 17.0, f"g{g}: worst round-trip SQNR {worst:.1f} dB > 17")
    SQNR_BY_G[g] = worst

    # the config
    ok("modules_in_block_to_quantize" in qc, f"g{g}: explicit list written")
    lst = qc["modules_in_block_to_quantize"]
    ok(B.EXPERTS_COLLAPSED in lst, "the routed experts are collapsed to a substring")
    ok(
        not any(".mlp.experts." in e for e in lst),
        "no per-expert entry survives in the list",
    )
    for m in MTP_TARGET_SHAPES:
        ok(R.mtp_vllm_prefix(m) in lst, f"g{g}: {m} listed under the vLLM index")
    eq(
        B.check_second_gate(qc["dynamic"], lst, sorted(MTP_TARGET_SHAPES), g),
        [],
        f"g{g}: both gates pass on every drafter prefix",
    )
    eq(
        B.check_dynamic_routing(
            qc["dynamic"], "core" if composed else "", g, mtp_tier="drafter-dense"
        ),
        [],
        f"g{g}: dynamic routing self-test",
    )
    pats = list(qc["dynamic"])
    eq(
        pats[: len(B.POSITIVE_MTP_DENSE)],
        B.POSITIVE_MTP_DENSE,
        f"g{g}: the mtp positives come first, in order",
    )
    eq(
        pats[len(B.POSITIVE_MTP_DENSE)],
        B.MTP_GUARD,
        f"g{g}: the mtp guard directly follows them",
    )
    inherited = list(json.loads(B.SOURCE_DYNAMIC_JSON))
    eq(
        [p for p in pats if p in inherited],
        inherited,
        f"g{g}: the inherited rules keep their original order",
    )
    eq(cfg["_iter6a3i"]["mtp_tier"], "drafter-dense", f"g{g}: stamp written")
    eq(cfg["_iter6a3i"]["group_size"], g, f"g{g}: stamp group size")


def main() -> int:
    rng = np.random.default_rng(20260909)
    root = tempfile.mkdtemp(prefix="iter6-mtpdense-")
    try:
        src = os.path.join(root, "src")
        wmap = make_fake_model(src, rng)
        before = dir_fingerprint(src)

        check_prefix_mapping()
        check_selection(wmap)
        check_byte_model()

        # dry run reads no tensor data and publishes nothing
        out0 = os.path.join(root, "dry")
        rc = B.build(Args(model_dir=src, out_dir=out0, dry_run=True))
        eq(rc, 0, "--dry-run returns 0")
        ok(
            not os.path.exists(os.path.join(out0, "config.json")),
            "--dry-run publishes nothing",
        )

        for g in (32, 64, 128):
            for composed in (False, True):
                tag = f"g{g}{'-composed' if composed else ''}"
                out = os.path.join(root, "out-" + tag)
                a = Args(
                    model_dir=src,
                    out_dir=out,
                    group_size=g,
                    tier="core" if composed else "drafter-dense",
                    mtp_tier="drafter-dense" if composed else "",
                )
                rc = B.build(a)
                eq(rc, 0, f"{tag}: build + verify returned 0")
                check_built(src, out, g, composed)
                check_no_target_leak(src, out, g, composed)
                rc = B.verify(
                    out, src, "core" if composed else "", g, sample=3,
                    mtp_tier="drafter-dense",
                )
                eq(rc, 0, f"{tag}: --verify-only returned 0")
                shutil.rmtree(out)

        # rebuild once at g32 to exercise the witness and the source-immutability
        out = os.path.join(root, "out-final")
        B.build(Args(model_dir=src, out_dir=out, group_size=32))
        check_regression_witness(out)
        # the CLI --verify-only path, which reads the tier back off the stamp
        rc = B.main(
            [
                "--model-dir",
                src,
                "--out-dir",
                out,
                "--tier",
                "drafter-dense",
                "--group-size",
                "32",
                "--verify-only",
            ]
        )
        eq(rc, 0, "CLI --verify-only returned 0")

        ok(
            SQNR_BY_G[32] >= SQNR_BY_G[64] >= SQNR_BY_G[128],
            f"finer groups are not worse: {SQNR_BY_G}",
        )
        after = dir_fingerprint(src)
        eq(before, after, "the source model directory is byte-for-byte unmodified")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    if FAILS:
        print(f"FAILED {len(FAILS)} of {CHECKS} checks:")
        for f in FAILS[:40]:
            print("  " + f)
        return 1
    print(f"test_build_mtpdense: {CHECKS}/{CHECKS} checks pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())
