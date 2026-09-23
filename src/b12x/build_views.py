#!/usr/bin/env python3
"""Build, inside the container, the two model folders the b12x vLLM needs (symlinks + small JSON,
nothing copied). Adapted from azampatti/Qwen3.8-Flash-Next-Int4-FAST flashnext-int4-b12x/build_views.py
for magi: the PLE table is served over RDMA, so it is neither linked nor indexed here.

  <out>/model      every checkpoint file linked; config.json + ple_embedding_dtype=float8_e4m3fn (b12x's
                   PLE storage mode) + the Qwen3_8FlashNextForConditionalGeneration arch name this vLLM knows
  <out>/draft      slim MTP draft: only the files holding mtp.* / lm_head.* / embed_tokens tensors + an index
                   of those keys (a draft folder linking every shard makes the draft load walk the whole model
                   again -- out-of-memory upstream)

  usage: build_views.py <model dir> <output root>
"""
import json
import os
import shutil
import sys

src, root = (os.path.abspath(a) for a in sys.argv[1:3])
view, draft = os.path.join(root, "model"), os.path.join(root, "draft")
B12X_ARCH = "Qwen3_8FlashNextForConditionalGeneration"


def fresh(d):
    if os.path.lexists(d):
        shutil.rmtree(d)
    os.makedirs(d)


wm = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
if any(".ngram_embedding.shard_" in k for k in wm):
    sys.exit("build_views: the index carries PLE table shards -- expected an RDMA (stripped) checkpoint")
cfg = json.load(open(os.path.join(src, "config.json")))
cfg.get("text_config", cfg)["ple_embedding_dtype"] = "float8_e4m3fn"
if B12X_ARCH not in cfg.setdefault("architectures", []):
    cfg["architectures"].append(B12X_ARCH)

fresh(view)
for name in os.listdir(src):
    if name != "config.json":
        os.symlink(os.path.join(src, name), os.path.join(view, name))
json.dump(cfg, open(os.path.join(view, "config.json"), "w"), indent=2)

fresh(draft)
keep = {k: v for k, v in wm.items()
        if k.startswith("mtp.") or k.startswith("lm_head.") or k.endswith("embed_tokens.weight")}
if not any(k.startswith("mtp.") for k in keep):
    sys.exit("build_views: no mtp.* tensors in the index")
for name in os.listdir(src):
    if name in ("config.json", "model.safetensors.index.json"):
        continue
    if name.endswith(".safetensors") and name not in set(keep.values()):
        continue
    os.symlink(os.path.join(src, name), os.path.join(draft, name))
json.dump({"metadata": {}, "weight_map": keep}, open(os.path.join(draft, "model.safetensors.index.json"), "w"))
json.dump(cfg, open(os.path.join(draft, "config.json"), "w"), indent=2)
print(f"model view: {len(os.listdir(view))} entries | draft: {len(keep)} keys in {len(set(keep.values()))} files")
