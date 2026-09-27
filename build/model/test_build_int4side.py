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

LAYERS = (0, 3)  # 0 = GDN (linear_attention), 3 = QSA (full_attention)

# tier -> number of converted modules in the miniature checkpoint (x2 layers)
TIER_COUNTS = {"gdn": 6, "fallbackB": 12, "fallbackA": 14, "all": 20}


def _fp8_pair(out_f, in_f, rng):
    codes = rng.integers(1, 250, size=(out_f, in_f), dtype=np.uint8)
    codes[codes == 0x7F] = 0x30
    codes[codes == 0xFF] = 0xB0
    sc = (
        rng.random(((out_f + 127) // 128, (in_f + 127) // 128)).astype(np.float32) * 1e-4
        + 1e-5
    )
    return codes, sc


MINI_SHAPES = {
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


# Miniature hyper-connection geometry (thread L3 / the hc tier).
# hc_count 4 x hidden 64 -> hyper_hidden 256 = 2 x block_k; lowrank 20 so
# pad = (-(20 + 4)) % 16 = 8 and the merged module is 20 + 4 + 8 = 32 rows.
HC_COUNT, HC_HIDDEN, HC_LOWRANK = 4, 64, 20
HC_HYPER = HC_COUNT * HC_HIDDEN
HC_PAD = (-(HC_LOWRANK + HC_COUNT)) % 16
HC_MERGED = HC_LOWRANK + HC_COUNT + HC_PAD


def _bf16(arr):
    """float32 -> the uint16 top-half safetensors stores as BF16."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    return (a.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def make_fake_model(root, rng):
    os.makedirs(root, exist_ok=True)
    shard_a = "model-00001-of-00003.safetensors"
    shard_b = "model-00002-of-00003.safetensors"
    shard_c = "model-00003-of-00003.safetensors"
    ta, tb, tc = {}, {}, {}
    for L in LAYERS:
        p = f"model.language_model.layers.{L}."
        for mod, (o, i) in MINI_SHAPES.items():
            w, s = _fp8_pair(o, i, rng)
            ta[p + mod + ".weight"] = w
            ta[p + mod + ".weight_scale_inv"] = s
        # bf16-ish neighbours that must survive untouched
        ta[p + "self_attn.indexer.index_qk_proj.weight"] = rng.integers(
            0, 255, (640, 256), dtype=np.uint8
        )
        ta[p + "linear_attn.in_proj_a.weight"] = rng.integers(
            0, 255, (32, 256), dtype=np.uint8
        )
        ta[p + "mlp.gate.weight"] = rng.integers(0, 255, (512, 256), dtype=np.uint8)
        ta[p + "mlp.shared_expert_gate.weight"] = rng.integers(
            0, 255, (1, 256), dtype=np.uint8
        )
        # hyper-connections: bf16, one attn_ and one mlp_ GatedResidual per
        # layer.  Layer 3's attn injection lives in shard C, so the builder's
        # cross-shard merge path (real: layer 13 attn) is exercised.
        for role in ("attn", "mlp"):
            hp = p + f"{role}_hyper_connection."
            ta[hp + "input_mix_weight_down.weight"] = _bf16(
                rng.standard_normal((HC_LOWRANK, HC_HYPER)) * 0.02
            )
            inj = _bf16(rng.standard_normal((HC_COUNT, HC_HYPER)) * 0.05)
            if L == 3 and role == "attn":
                tc[hp + "block_inject_weight.weight"] = inj
            else:
                ta[hp + "block_inject_weight.weight"] = inj
            ta[hp + "input_mix_weight_up.weight"] = _bf16(
                rng.standard_normal((HC_HYPER, HC_LOWRANK)) * 0.02
            )
            ta[hp + "hc_norm.weight"] = _bf16(rng.standard_normal(HC_HYPER) * 0.01)
    # the target's final mixer (use_combine=False: no block_inject_weight)
    mp = "model.language_model.hyper_connection_mixer."
    ta[mp + "input_mix_weight_down.weight"] = _bf16(
        rng.standard_normal((HC_LOWRANK, HC_HYPER)) * 0.02
    )
    ta[mp + "input_mix_weight_up.weight"] = _bf16(
        rng.standard_normal((HC_HYPER, HC_LOWRANK)) * 0.02
    )
    # shard B: no fp8 at all -> must end up linked
    tb["mtp.layers.0.self_attn.o_proj.weight"] = rng.integers(
        0, 255, (256, 128), dtype=np.uint8
    )
    # the drafter's own hyper-connections: the hc tier must never touch them
    for role in ("attn", "mlp"):
        hp = f"mtp.layers.0.{role}_hyper_connection."
        tb[hp + "input_mix_weight_down.weight"] = _bf16(
            rng.standard_normal((HC_LOWRANK, HC_HYPER)) * 0.02
        )
        tb[hp + "block_inject_weight.weight"] = _bf16(
            rng.standard_normal((HC_COUNT, HC_HYPER)) * 0.05
        )
    tb["mtp.hyper_connection_mixer.input_mix_weight_down.weight"] = _bf16(
        rng.standard_normal((HC_LOWRANK, HC_HYPER)) * 0.02
    )
    tb["lm_head.qweight"] = rng.integers(-(2**31), 2**31 - 1, (32, 256), dtype=np.int32)
    tb["lm_head.scales"] = rng.random((4, 256)).astype(np.float16)
    tb["lm_head.qzeros"] = np.full((4, 32), np.int32(0x7F7F7F7F), dtype=np.int32)

    R.write_safetensors(os.path.join(root, shard_a), ta)
    R.write_safetensors(os.path.join(root, shard_b), tb)
    R.write_safetensors(os.path.join(root, shard_c), tc)
    wmap = (
        {k: shard_a for k in ta}
        | {k: shard_b for k in tb}
        | {k: shard_c for k in tc}
    )
    with open(os.path.join(root, "model.safetensors.index.json"), "w") as fh:
        json.dump({"metadata": {"total_size": 0}, "weight_map": wmap}, fh)
    cfg = {
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "hc_count": HC_COUNT,
            "hc_lowrank": HC_LOWRANK,
            "hidden_size": HC_HIDDEN,
            "num_hidden_layers": len(LAYERS),
        },
        "quantization_config": {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
            "lm_head": True,
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
                "-:.*\\.gate$": {},
            },
        },
    }
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    for extra in ("tokenizer.json", "chat_template.jinja"):
        with open(os.path.join(root, extra), "w") as fh:
            fh.write("{}")
    return shard_a, shard_b, shard_c


def same_file(a, b):
    """True if `a` is the same content as `b` (hardlink, symlink or copy)."""
    if not os.path.exists(a):
        return False
    sa, sb = os.stat(a), os.stat(b)
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino) or (
        open(a, "rb").read() == open(b, "rb").read()
    )


def dir_fingerprint(root):
    fp = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        fp[name] = (h.hexdigest(), os.path.getsize(p))
    return fp


SPLIT_NAMES = (
    "in_proj_qkv",
    "in_proj_z",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
FUSED_NAMES = ("in_proj_qkvz", "qkv_proj", "gate_up_proj")


def main() -> int:
    fails = []

    def ck(cond, msg):
        if not cond:
            fails.append(msg)
            print("FAIL: " + msg)

    tmp = tempfile.mkdtemp(prefix="iter4-build-test-")
    try:
        src = os.path.join(tmp, "src")
        rng = np.random.default_rng(101)
        shard_a, shard_b, shard_c = make_fake_model(src, rng)
        before = dir_fingerprint(src)

        cases = [(t, g) for t in ("gdn", "fallbackB", "fallbackA", "all")
                 for g in (32, 128)]
        cases.append(("all", 64))
        for tier, g in cases:
            tag = f"{tier}-g{g}"
            out = os.path.join(tmp, "out-" + tag)
            rc = B.main(
                ["--model-dir", src, "--out-dir", out, "--tier", tier,
                 "--group-size", str(g), "--verify-sample", "4"]
            )
            ck(rc == 0, f"[{tag}] builder returned {rc}")

            idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))
            wmap = idx["weight_map"]
            conv = sorted(
                {n.rsplit(".", 1)[0] for n in wmap if n.endswith(".qweight")}
                - {"lm_head"}
            )
            ck(
                len(conv) == TIER_COUNTS[tier],
                f"[{tag}] converted {len(conv)} modules, expected {TIER_COUNTS[tier]}",
            )
            # split names only; never a fused name on disk
            ck(
                all(m.rsplit(".", 1)[-1] in SPLIT_NAMES for m in conv),
                f"[{tag}] every emitted tensor uses a checkpoint SPLIT name",
            )
            ck(
                not any(f in n for f in FUSED_NAMES for n in wmap),
                f"[{tag}] no fused name reached the index",
            )
            ck(
                not R.check_fused_uniformity(conv),
                f"[{tag}] fused uniformity: {R.check_fused_uniformity(conv)[:1]}",
            )
            ck(
                same_file(os.path.join(out, shard_b), os.path.join(src, shard_b)),
                f"[{tag}] the fp8-free shard is linked to the source",
            )
            ck(
                not same_file(os.path.join(out, shard_a), os.path.join(src, shard_a)),
                f"[{tag}] the affected shard is a new file",
            )
            ck(
                same_file(
                    os.path.join(out, "tokenizer.json"),
                    os.path.join(src, "tokenizer.json"),
                ),
                f"[{tag}] side files are linked",
            )
            hdr, _ = R.read_st_header(os.path.join(out, shard_a))
            stale = [
                n
                for n in hdr
                if n.rsplit(".", 1)[0] in set(conv)
                and n.endswith((".weight", ".weight_scale_inv"))
            ]
            ck(not stale, f"[{tag}] stale fp8 tensors remain: {stale[:2]}")
            ck(set(hdr) <= set(wmap), f"[{tag}] every shard tensor is in the index")

            # group-size-dependent shapes
            for m in conv:
                kq, nn = hdr[m + ".qweight"]["shape"]
                rows = kq * 8 // g
                ck(
                    hdr[m + ".scales"]["shape"] == [rows, nn],
                    f"[{tag}] {m}.scales {hdr[m + '.scales']['shape']} != [{rows},{nn}]",
                )
                ck(
                    hdr[m + ".qzeros"]["shape"] == [rows, nn // 8],
                    f"[{tag}] {m}.qzeros shape wrong for g{g}",
                )
            # the qzeros word is the checkpoint's, at every group size
            z = R.read_tensor(os.path.join(out, shard_a), conv[0] + ".qzeros")
            ck(
                bool((z.view(np.uint32) == 0x77777777).all()),
                f"[{tag}] qzeros is 0x77777777",
            )

            # config
            cfg = json.load(open(os.path.join(out, "config.json")))
            qc = cfg["quantization_config"]
            ck(qc["group_size"] == 128, f"[{tag}] the base (expert) group size stays 128")
            ck(
                not B.assert_experts_keep_g128(qc["dynamic"], qc["group_size"]),
                f"[{tag}] experts keep g128: "
                f"{B.assert_experts_keep_g128(qc['dynamic'], qc['group_size'])}",
            )
            ck(
                list(qc["dynamic"])[0] == B.MTP_GUARD,
                f"[{tag}] the mtp guard is the first dynamic rule",
            )
            ck(
                B._override(qc["dynamic"], "mtp.layers.0.linear_attn.in_proj_qkvz")
                is False,
                f"[{tag}] the drafter stays out",
            )
            ck(
                B._override(
                    qc["dynamic"], "model.layers.3.self_attn.indexer.index_qk_proj"
                )
                is False,
                f"[{tag}] the bf16 QSA indexer stays out",
            )
            v = B._override(qc["dynamic"], "model.layers.0.linear_attn.in_proj_qkvz")
            ck(
                isinstance(v, dict) and v.get("group_size") == g,
                f"[{tag}] the fused GDN in-projection is routed at g{g}, got {v!r}",
            )
            ck(cfg["_iter4a"]["group_size"] == g, f"[{tag}] the build stamp records g{g}")

            # neighbours untouched
            for keep in (
                "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
                "model.language_model.layers.3.mlp.gate.weight",
                "model.language_model.layers.0.linear_attn.in_proj_a.weight",
                "model.language_model.layers.0.mlp.shared_expert_gate.weight",
                "model.language_model.layers.0.attn_hyper_connection."
                "input_mix_weight_up.weight",
            ):
                ck(keep in hdr, f"[{tag}] neighbour survived: {keep}")
            if tier in ("gdn", "fallbackA", "fallbackB"):
                ck(
                    "model.language_model.layers.3.self_attn.q_proj.weight" in hdr,
                    f"[{tag}] QSA q_proj stays fp8 outside the all tier",
                )
            if tier == "fallbackB":
                ck(
                    "model.language_model.layers.0.linear_attn.out_proj.weight" in hdr,
                    f"[{tag}] fallbackB leaves GDN out_proj at fp8",
                )

            # dequant sanity on one converted module
            m = conv[0]
            qw = R.read_tensor(os.path.join(out, shard_a), m + ".qweight")
            sc = R.read_tensor(os.path.join(out, shard_a), m + ".scales")
            w_hat = R.dequant_int4_sym(qw, sc, qw.shape[0] * 8, g)
            w0 = R.read_tensor(os.path.join(src, shard_a), m + ".weight")
            s0 = R.read_tensor(os.path.join(src, shard_a), m + ".weight_scale_inv")
            w_ref = R.dequant_blockwise_fp8(w0, s0).T
            rel = float(
                np.linalg.norm(w_hat - w_ref) / max(np.linalg.norm(w_ref), 1e-30)
            )
            ck(rel < 0.15, f"[{tag}] {m} reconstruction rel-Fro {rel:.4f} < 0.15")

        # ---- --packed-dir: the gate-3 splice ---------------------------
        packed = os.path.join(tmp, "packed")
        os.makedirs(packed)
        g = 32
        mods = R.select_modules(
            json.load(open(os.path.join(src, "model.safetensors.index.json")))[
                "weight_map"
            ],
            "gdn",
        )
        meta = {"group_size": g, "modules": {}}
        want = {}
        for m in mods:
            w0 = R.read_tensor(os.path.join(src, shard_a), m + ".weight")
            s0 = R.read_tensor(os.path.join(src, shard_a), m + ".weight_scale_inv")
            res = R.convert_module(w0, s0, m, g)
            # perturb one code so a byte-for-byte splice is provable
            res["qweight"][0, 0] ^= np.int32(0xF)
            fn = m.replace("/", "_") + ".gptq.safetensors"
            R.write_safetensors(
                os.path.join(packed, fn),
                {
                    m + ".qweight": res["qweight"],
                    m + ".qzeros": res["qzeros"],
                    m + ".scales": res["scales"],
                },
            )
            meta["modules"][m] = {"file": fn, "stats": res["stats"]}
            want[m] = res["qweight"]
        with open(os.path.join(packed, "index.json"), "w") as fh:
            json.dump(meta, fh)
        out = os.path.join(tmp, "out-packed")
        rc = B.main(
            ["--model-dir", src, "--out-dir", out, "--tier", "gdn",
             "--group-size", str(g), "--packed-dir", packed, "--verify-sample", "2"]
        )
        ck(rc == 0, f"[packed] builder returned {rc}")
        for m, qw in want.items():
            got = R.read_tensor(os.path.join(out, shard_a), m + ".qweight")
            ck(np.array_equal(got, qw), f"[packed] {m}.qweight spliced byte-for-byte")
        # a group-size mismatch is refused rather than silently mis-shaped
        try:
            B.main(
                ["--model-dir", src, "--out-dir", os.path.join(tmp, "out-bad"),
                 "--tier", "gdn", "--group-size", "128", "--packed-dir", packed]
            )
            ck(False, "[packed] a group-size mismatch must be refused")
        except SystemExit:
            ck(True, "[packed] a group-size mismatch is refused")

        # ---- the hc tier: hyper-connection down GEMMs -> blockwise fp8 ----
        src_wmap = json.load(
            open(os.path.join(src, "model.safetensors.index.json"))
        )["weight_map"]
        for tag, argv in (
            ("hc", ["--tier", "hc"]),
            ("all+hc", ["--tier", "all", "--group-size", "32", "--hc-tier", "hc"]),
        ):
            out = os.path.join(tmp, "out-" + tag)
            rc = B.main(
                ["--model-dir", src, "--out-dir", out, "--verify-sample", "3"] + argv
            )
            ck(rc == 0, f"[{tag}] builder returned {rc}")
            idx = json.load(open(os.path.join(out, "model.safetensors.index.json")))
            wmap = idx["weight_map"]
            cfg = json.load(open(os.path.join(out, "config.json")))
            groups = R.select_hc_groups(src_wmap, "hc")
            ck(
                len(groups) == len(LAYERS) * 2 + 1,
                f"[{tag}] {len(groups)} hc groups (4 combine + 1 mixer expected)",
            )

            # 1. the merged tensor exists with the runtime's name/shape/dtype
            for grp in groups:
                wn = grp["out_name"] + ".weight"
                sn = grp["out_name"] + ".weight_scale_inv"
                ck(wn in wmap and sn in wmap, f"[{tag}] {wn} + scale are in the index")
                shard = wmap[wn]
                hdr, _ = R.read_st_header(os.path.join(out, shard))
                o = HC_MERGED if grp["kind"] == "combine" else HC_LOWRANK
                ck(
                    hdr[wn]["dtype"] == "F8_E4M3" and hdr[wn]["shape"] == [o, HC_HYPER],
                    f"[{tag}] {wn} is F8_E4M3 [{o},{HC_HYPER}], got "
                    f"{hdr[wn]['dtype']} {hdr[wn]['shape']}",
                )
                ck(
                    hdr[sn]["dtype"] == "F32"
                    and hdr[sn]["shape"] == [-(-o // 128), -(-HC_HYPER // 128)],
                    f"[{tag}] {sn} is F32 {[-(-o // 128), -(-HC_HYPER // 128)]}",
                )
                # 2. the bf16 sources are gone from the index AND from the
                #    shards (a "mix" group keeps its own name, dtype changes)
                for s in grp["srcs"]:
                    if s == wn:
                        continue
                    ck(s not in wmap, f"[{tag}] {s} left the index")
                    sh, _ = R.read_st_header(os.path.join(out, src_wmap[s]))
                    ck(s not in sh, f"[{tag}] {s} left its shard")

            # 3. reconstruction: down rows, injection rows, zero pad
            grp = [x for x in groups if x["kind"] == "combine"][0]
            wn = grp["out_name"] + ".weight"
            shard = wmap[wn]
            q = R.read_tensor(os.path.join(out, shard), wn)
            s = R.read_tensor(os.path.join(out, shard), grp["out_name"] + ".weight_scale_inv")
            w_hat = R.dequant_blockwise_fp8(q, s, 128)
            d0 = R.bf16_to_f32(
                R.read_tensor(os.path.join(src, src_wmap[grp["srcs"][0]]), grp["srcs"][0])
            )
            i0 = R.bf16_to_f32(
                R.read_tensor(os.path.join(src, src_wmap[grp["srcs"][1]]), grp["srcs"][1])
            )
            rel_d = float(np.linalg.norm(w_hat[:HC_LOWRANK] - d0) / np.linalg.norm(d0))
            rel_i = float(
                np.linalg.norm(w_hat[HC_LOWRANK : HC_LOWRANK + HC_COUNT] - i0)
                / np.linalg.norm(i0)
            )
            ck(rel_d < 0.05, f"[{tag}] down rows rel-Fro {rel_d:.4f} < 0.05")
            ck(rel_i < 0.05, f"[{tag}] injection rows rel-Fro {rel_i:.4f} < 0.05")
            ck(
                bool(np.all(w_hat[HC_LOWRANK + HC_COUNT :] == 0.0)),
                f"[{tag}] the {HC_PAD} pad rows are exactly zero",
            )

            # 4. what must NOT move
            for keep in (
                "model.language_model.layers.0.attn_hyper_connection."
                "input_mix_weight_up.weight",
                "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
                "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
                "mtp.hyper_connection_mixer.input_mix_weight_down.weight",
                "model.language_model.layers.0.attn_hyper_connection.hc_norm.weight",
            ):
                ck(keep in wmap, f"[{tag}] untouched by the hc tier: {keep}")
                sh, _ = R.read_st_header(os.path.join(out, wmap[keep]))
                ck(sh[keep]["dtype"] == "BF16", f"[{tag}] {keep} is still BF16")
            ck(
                not any("mtp." in n and "block_inject" not in n and
                        n.endswith("input_mix_weight_down_block_inject.weight")
                        for n in wmap),
                f"[{tag}] no drafter HC module was merged",
            )

            # 5. the config stamp, and the dynamic rules
            st = cfg["_iter4hc"]
            ck(st["hc_tier"] == "hc" and st["block"] == [128, 128],
               f"[{tag}] _iter4hc stamp: {st.get('hc_tier')} {st.get('block')}")
            ck(st["requires_env"] == "VLLM_HC_FP8=1",
               f"[{tag}] the stamp records the env gate")
            ck(
                all(p.startswith("model.") and "language_model" not in p
                    for p in st["vllm_prefixes"]),
                f"[{tag}] vllm_prefixes are mapped through model.language_model. -> "
                f"model.: {st['vllm_prefixes'][:2]}",
            )
            ck(
                B._override(
                    cfg["quantization_config"]["dynamic"],
                    "model.layers.0.attn_hyper_connection."
                    "input_mix_weight_down_block_inject",
                )
                is False,
                f"[{tag}] the merged HC module still routes to 'skip' in dynamic",
            )
            if tag == "hc":
                ck(
                    cfg["quantization_config"]["dynamic"]
                    == json.load(open(os.path.join(src, "config.json")))[
                        "quantization_config"
                    ]["dynamic"],
                    "[hc] the hc-only tier leaves `dynamic` byte-identical",
                )
                ck(
                    not any(n.endswith(".qweight") and not n.startswith("lm_head")
                            for n in wmap),
                    "[hc] the hc-only tier emits no GPTQ tensors",
                )
            else:
                ck(
                    len({n.rsplit(".", 1)[0] for n in wmap if n.endswith(".qweight")}
                        - {"lm_head"}) == TIER_COUNTS["all"],
                    "[all+hc] the int4 tier is unaffected by --hc-tier",
                )

            # 6. the byte model
            rep = json.load(open(os.path.join(out, "iter4a-build-report.json")))
            sb = rep["hc_step_bytes"]
            ck(
                sb["saved_bytes_per_step"] > 0
                and sb["hc_bytes_per_step_after"] < sb["hc_bf16_bytes_per_step"],
                "[hc] the step-byte model saves bytes",
            )
            ck(
                not sb["per_launch"]["target.up"]["converted"]
                and not sb["per_launch"]["draft.merged_down_inject"]["converted"],
                "[hc] the up projections and the drafter are outside the model",
            )
            ck(
                rep["hc_totals"]["sqnr_db_min"] > 25.0,
                f"[{tag}] worst hc SQNR {rep['hc_totals']['sqnr_db_min']:.1f} dB",
            )

            # 7. --verify-only picks the hc tier up from the stamp
            ck(
                B.main(["--model-dir", src, "--out-dir", out, "--verify-only"]) == 0,
                f"[{tag}] --verify-only re-verifies from the build stamp",
            )

        after = dir_fingerprint(src)
        ck(before == after, "the source model directory is byte-for-byte unmodified")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if fails:
        print(f"\n{len(fails)} failures")
        return 1
    print("\nbuild_int4side_model_dir end-to-end test: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
