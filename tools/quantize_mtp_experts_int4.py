#!/usr/bin/env python3
"""Quantize the MTP draft layer's 512 routed experts to int4 GPTQ (g128, symmetric,
full-range) in exactly the layout Intel ships for the main layers' experts, so vLLM's
GPTQ-Marlin MoE path serves the drafter too (drop the `-:.*layers\\.48\\..*` rule).

Intel leaves the MTP layer unquantized: 1,536 bf16 tensors (~4.7 GiB) that run on the
unquantized MoE backend. RTN int4 here = the same recipe as Intel's *-RTN-AutoRound
release for the main experts. Output layout per expert projection (w: [out, in] bf16):
  <p>.qweight  int32 [in/8, out]   8 nibbles per int32 along in, little-endian
  <p>.qzeros   int32 [in/128, out/8]  all nibbles 7 (zp=8, GPTQ v1 storage = zp-1)
  <p>.scales   f16   [in/128, out]  scale = w[argmax|w|] / -8 per group (full-range:
                                    the max-magnitude value lands on the -8 slot)

usage: quantize_mtp_experts_int4.py <src_ckpt_dir> <dst_ckpt_dir>
dst is created if missing: every untouched file is hardlinked from src (copied if
that fails), then model_extra_tensors.safetensors, model.safetensors.index.json and
config.json are written fresh (never through a hardlink — that would edit src).
"""
import json, os, re, sys
import numpy as np
import torch
from safetensors.torch import load_file, save_file

GROUP, PACK, BITS = 128, 8, 4
EXPERT_RE = re.compile(r"^mtp\.layers\.\d+\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)\.weight$")


def quantize(w: torch.Tensor):
    out_f, in_f = w.shape
    assert in_f % GROUP == 0 and in_f % PACK == 0 and out_f % PACK == 0, w.shape
    groups = in_f // GROUP
    wg = w.to(torch.float32).T.reshape(groups, GROUP, out_f)          # [g, 128, out]
    amax_idx = wg.abs().argmax(dim=1, keepdim=True)                     # [g, 1, out]
    picked = wg.gather(1, amax_idx).squeeze(1)                          # signed max-magnitude
    scale = (picked / -(2 ** (BITS - 1))).to(torch.float16)             # [g, out]
    s32 = scale.to(torch.float32).unsqueeze(1)
    s32 = torch.where(s32 == 0, torch.ones_like(s32), s32)              # all-zero group guard
    q = torch.clamp(torch.round(wg / s32) + 2 ** (BITS - 1), 0, 2 ** BITS - 1).to(torch.int32)
    q = q.reshape(in_f, out_f).numpy().astype(np.uint32)
    q = q.reshape(in_f // PACK, PACK, out_f)
    shifts = (np.arange(PACK, dtype=np.uint32) * BITS).reshape(1, PACK, 1)
    qweight = (q << shifts).sum(axis=1, dtype=np.uint32).astype(np.int32)   # [in/8, out]
    qzeros = np.full((groups, out_f // PACK), 0x77777777, dtype=np.int32)
    return (torch.from_numpy(np.ascontiguousarray(qweight)),
            torch.from_numpy(np.ascontiguousarray(qzeros)), scale.contiguous())


def dequant(qweight, qzeros, scales):
    q = qweight.numpy().astype(np.uint32)
    in8, out_f = q.shape
    nib = np.stack([(q >> (i * BITS)) & 0xF for i in range(PACK)], axis=1).reshape(in8 * PACK, out_f)
    zp = ((qzeros.numpy().astype(np.uint32)[0, 0] & 0xF) + 1)
    w = (nib.astype(np.float32) - zp).reshape(-1, GROUP, out_f) * scales.to(torch.float32).numpy()[:, None, :]
    return torch.from_numpy(w.reshape(in8 * PACK, out_f).T)


REGEN = ("model_extra_tensors.safetensors", "model.safetensors.index.json", "config.json")


def link_tree(src, dst):
    import shutil
    os.makedirs(dst, exist_ok=True)
    for name in sorted(os.listdir(src)):
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if name in REGEN or not os.path.isfile(s) or os.path.lexists(d):
            continue
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    for name in REGEN:  # never write through a hardlink of the source
        d = os.path.join(dst, name)
        if os.path.lexists(d) and os.stat(d).st_nlink > 1:
            os.unlink(d)


def main(src, dst):
    link_tree(src, dst)
    extra = os.path.join(src, "model_extra_tensors.safetensors")
    tensors = load_file(extra)
    out, n_q, worst = {}, 0, 0.0
    for k, v in tensors.items():
        if not EXPERT_RE.match(k):
            out[k] = v
            continue
        qw, qz, sc = quantize(v)
        base = k[: -len(".weight")]
        out[base + ".qweight"], out[base + ".qzeros"], out[base + ".scales"] = qw, qz, sc
        n_q += 1
        if n_q % 128 == 1:  # spot-check reconstruction on every 128th tensor
            err = ((dequant(qw, qz, sc) - v.float()).abs().max() / v.float().abs().max()).item()
            worst = max(worst, err)
            print(f"  [{n_q}] {k}: max-rel reconstruction error {err:.4f}", flush=True)
    assert n_q == 1536, n_q
    dst_extra = os.path.join(dst, "model_extra_tensors.safetensors")
    if os.path.lexists(dst_extra):
        os.unlink(dst_extra)
    save_file(out, dst_extra, metadata={"format": "pt"})
    print(f"wrote {dst_extra}: {n_q} expert tensors -> int4 ({3*n_q} tensors), {len(out)-3*n_q} kept bf16; worst spot-check {worst:.4f}")

    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wm = {k: f for k, f in idx["weight_map"].items() if not EXPERT_RE.match(k)}
    for k in out:
        if k not in wm:
            wm[k] = "model_extra_tensors.safetensors"
    idx["weight_map"] = wm
    json.dump(idx, open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)

    cfg = json.load(open(os.path.join(src, "config.json")))
    dyn = cfg["quantization_config"]["dynamic"]
    dropped = [p for p in dyn if "layers\\.48" in p or "layers.48" in p]
    assert dropped, f"layers.48 exclusion not found in {list(dyn)}"
    for p in dropped:
        del dyn[p]
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    print(f"index: {len(wm)} entries; config: dropped rule(s) {dropped}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
