#!/usr/bin/env python3
"""Remove the PLE n-gram table entries from a checkpoint's safetensors index,
so vLLM does not try to load the 51B-parameter table from the checkpoint —
the mmap patch serves it from VLLM_PLE_MMAP_DIR instead.

Usage: strip_ngram_index.py <checkpoint_dir>
Backup written to model.safetensors.index.json.with-ngram.bak.
The shard files themselves are untouched (delete them separately if you want
the disk back; nothing references them after this).
"""

import json
import shutil
import sys

d = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: strip_ngram_index.py <checkpoint_dir>")
path = f"{d}/model.safetensors.index.json"
idx = json.load(open(path))
removed = [k for k in idx["weight_map"] if ".ngram_embedding.shard_" in k]
if not removed:
    sys.exit("no ngram entries in index — already stripped?")
for k in removed:
    del idx["weight_map"][k]
shutil.copy(path, path + ".with-ngram.bak")
json.dump(idx, open(path, "w"))
print(f"stripped {len(removed)} ngram entries (backup: {path}.with-ngram.bak)")
