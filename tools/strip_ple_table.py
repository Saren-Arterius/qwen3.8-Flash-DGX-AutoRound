#!/usr/bin/env python3
"""Copy only the PLE n-gram tensors (shards + weight_scale) out of a set of
checkpoint shards into a lean standalone table directory for VLLM_PLE_MMAP_DIR.

Usage: strip_ple_table.py <src_dir> <dst_dir>

Useful when the destination is small/fast (e.g. a RAM-backed or NVMe-oF
device): the FP8 checkpoint's ngram shards carry a few non-table tensors that
this drops (52.3 GB -> ~48 GB for the fp8 table).
"""

import glob
import os
import sys

from safetensors.torch import safe_open, save_file

if len(sys.argv) != 3:
    sys.exit("usage: strip_ple_table.py <src_dir> <dst_dir>")
SRC, DST = sys.argv[1], sys.argv[2]
os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(SRC + "/*.safetensors"))
total = 0
for i, p in enumerate(files):
    out = {}
    with safe_open(p, framework="pt", device="cpu") as f:
        for k in f.keys():
            if "ngram_embedding.shard_" in k or "ngram_embedding.weight_scale" in k:
                out[k] = f.get_tensor(k)
    if out:
        save_file(out, os.path.join(DST, os.path.basename(p)))
        total += len(out)
    print(f"[{i + 1}/{len(files)}] {os.path.basename(p)}: {len(out)} tensors", flush=True)
print("done,", total, "tensors")
