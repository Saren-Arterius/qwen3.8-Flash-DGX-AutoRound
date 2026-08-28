#!/usr/bin/env python3
"""Repack a bf16 lm_head into 8-bit GPTQ (group 128, full-range symmetric) in
place, inside the shard that holds it. CPU-only, needs numpy + safetensors +
torch. This is what turns the Intel W4A16 checkpoint's bf16 head into the
uint8b128 head that vLLM's GPTQ-Marlin serves (with `"lm_head": true` and a
`"+:.*lm_head$": {"bits": 8}` dynamic rule in quantization_config).

Layout produced (matches auto_round:auto_gptq 8-bit):
  lm_head.qweight  int32 [in/4, out]   4 uint8 per int32, little-endian in-dim
  lm_head.qzeros   int32 [groups, out/4]  all bytes 127 (zp=128, v1 storage)
  lm_head.scales   f16   [groups, out]

Quantization: per group of 128 in-features, scale = w[argmax|w|] / 128 with
the SIGN of the largest-magnitude element kept (full-range symmetric: the
-128 slot is usable), q = clamp(round(w/scale) + 128, 0, 255).

Usage:
  quantize_lm_head_int8.py <checkpoint_dir> [--shard model-00001-of-*.safetensors]
The original shard is kept as <shard>.bf16head.bak; the model index json is
updated. Run with --dry-run to write to <shard>.int8head instead.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import safe_open, save_file

GROUP = 128
BITS = 8
PACK = 32 // BITS  # 4 uint8 per int32


def quantize(w: torch.Tensor):
    """w: [out, in] bf16 -> (qweight int32 [in/4, out], qzeros, scales f16)."""
    out_f, in_f = w.shape
    assert in_f % GROUP == 0
    groups = in_f // GROUP
    wg = w.to(torch.float32).T.reshape(groups, GROUP, out_f)  # [g, 128, out]
    # Full-range symmetric: the largest-magnitude element of each (group, out)
    # maps to the -128 slot, so the scale carries the OPPOSITE sign of it.
    # Ties (|min| == max) resolve to the positive element.
    wmin = wg.amin(dim=1)
    wmax = wg.amax(dim=1)
    picked = torch.where(-wmin > wmax, wmin, wmax)
    scale = (picked / -128).to(torch.float16)  # [g, out]
    q = torch.clamp(
        torch.round(wg / scale.to(torch.float32).unsqueeze(1)) + 128, 0, 255
    ).to(torch.uint8)  # [g, 128, out]
    q = q.reshape(in_f, out_f)
    qweight = (
        q.reshape(in_f // PACK, PACK, out_f)
        .permute(0, 2, 1)
        .contiguous()
        .numpy()
        .view(np.int32)
        .reshape(in_f // PACK, out_f)
    )
    qzeros = np.full((groups, out_f // PACK), 0x7F7F7F7F, dtype=np.int32)
    return (
        torch.from_numpy(qweight),
        torch.from_numpy(qzeros),
        scale.contiguous(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir")
    ap.add_argument("--shard", default=None, help="shard holding lm_head.weight")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    index_path = os.path.join(args.ckpt_dir, "model.safetensors.index.json")
    index = json.load(open(index_path))
    shard_name = args.shard or index["weight_map"].get("lm_head.weight")
    if shard_name is None:
        sys.exit("lm_head.weight not in index — already quantized?")
    shard_path = os.path.join(args.ckpt_dir, shard_name)

    tensors, w = {}, None
    with safe_open(shard_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            if k == "lm_head.weight":
                w = f.get_tensor(k)
            else:
                tensors[k] = f.get_tensor(k)
    assert w is not None, f"lm_head.weight not in {shard_name}"
    print(f"quantizing lm_head {tuple(w.shape)} bf16 -> int8 g{GROUP} sym")
    qweight, qzeros, scales = quantize(w)
    tensors["lm_head.qweight"] = qweight
    tensors["lm_head.qzeros"] = qzeros
    tensors["lm_head.scales"] = scales

    if args.dry_run:
        save_file(tensors, shard_path + ".int8head")
        print(f"dry run: wrote {shard_path}.int8head")
        return
    os.rename(shard_path, shard_path + ".bf16head.bak")
    save_file(tensors, shard_path)
    wm = index["weight_map"]
    del wm["lm_head.weight"]
    for k in ("lm_head.qweight", "lm_head.qzeros", "lm_head.scales"):
        wm[k] = shard_name
    index["metadata"]["total_size"] = sum(
        os.path.getsize(p) - (8 + int.from_bytes(open(p, "rb").read(8), "little"))
        for p in glob.glob(os.path.join(args.ckpt_dir, "model-*-of-*.safetensors"))
    )
    json.dump(index, open(index_path, "w"), indent=2)
    print(f"done: {shard_name} repacked, original at {shard_name}.bf16head.bak")


if __name__ == "__main__":
    main()
