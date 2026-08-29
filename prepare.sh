#!/usr/bin/env bash
# prepare.sh — build the ready-to-serve checkpoint from Intel's AutoRound
# release. CPU-only (a NAS box is fine); needs python3 with torch +
# safetensors, and the `hf` CLI. One-time, ~30 min + downloads.
#
# You can skip all of this: the finished outputs of exactly this script are
# published at
#   https://huggingface.co/Saren/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid
#   https://huggingface.co/Saren/Qwen3.8-Flash-Next-ple-table-fp8
# Run this only if you'd rather build (or audit) the artifacts yourself.
#
# What it does, in order:
#   1. Download Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound (int4 experts,
#      everything else bf16).
#   2. quantize_lm_head_int8.py — repack the bf16 lm_head as int8 GPTQ-Marlin
#      (full-range symmetric). Kills a 1.27 GiB bf16 head and its per-token
#      bf16 GEMV; also used by the MTP draft head.
#   3. fp8_convert.py — convert the 300 bf16 side-layer tensors (GDN in/out
#      projections, QSA q/k/v/o, shared expert) to blockwise fp8 e4m3
#      (128x128). Originals kept as .bf16.bak until the cleanup step.
#   4. strip_ngram_index.py — drop the 51B n-gram ("PLE") table from the
#      safetensors index: it is NOT loaded as a weight, it is mmapped from a
#      separate directory at runtime (see the README's PLE mmap patch).
#   5. fetch-ple-table-fp8.sh — download that table's fp8 shards (from
#      Qwen/Qwen3.8-Flash-Next-FP8) into the separate table dir.
#   6. Rewrite config.json's quantization_config for vLLM's GPTQ loader
#      (this vLLM has no auto-round loader; the GPTQ config reads the same
#      packed tensors). The dynamic rules exclude the non-int4 families and
#      flip the head to 8-bit; the original config is kept as
#      config.json.autoround.
#   7. Delete the .bf16.bak originals (rollback = re-download from Intel and
#      re-run; steps 2+3 are deterministic, verified bit-exact).
#
# Usage: prepare.sh <checkpoint-dir> <ple-table-dir>
# Then point serve.sh's MODEL_DIR/TABLE_DIR at the two dirs.
set -euo pipefail
CKPT="${1:?usage: prepare.sh <checkpoint-dir> <ple-table-dir>}"
TABLE="${2:?usage: prepare.sh <checkpoint-dir> <ple-table-dir>}"
cd "$(dirname "$0")"

hf download Intel/Qwen3.8-Flash-Next-W4A16-RTN-AutoRound --local-dir "$CKPT"

tools/quantize_lm_head_int8.py "$CKPT"
tools/fp8_convert.py "$CKPT"
tools/strip_ngram_index.py "$CKPT"
tools/fetch-ple-table-fp8.sh "$TABLE"

python3 - "$CKPT" <<'EOF'
import json, shutil, sys
ckpt = sys.argv[1]
shutil.copy2(f"{ckpt}/config.json", f"{ckpt}/config.json.autoround")
cfg = json.load(open(f"{ckpt}/config.json"))
cfg["quantization_config"] = {
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
}
json.dump(cfg, open(f"{ckpt}/config.json", "w"), indent=2)
print(">> quantization_config rewritten (original: config.json.autoround)")
EOF

rm -f "$CKPT"/*.bf16.bak
echo ">> done. MODEL_DIR=$CKPT TABLE_DIR=$TABLE"
