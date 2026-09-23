#!/usr/bin/env python3
"""Patch the vllm-node-b12x image IN THE RUNNING CONTAINER so it can serve the INT4-AutoRound checkpoint. Every edit is
idempotent (marker comment) and checked (anchor must match exactly once, file must still parse). The image on disk is
never modified -- a new container starts clean and the mod re-applies this.

  REQUIRED (exit 1 if they cannot be applied -- the launch is aborted rather than serving a broken model):
    fp8-hybrid   300 side layers of the checkpoint are blockwise fp8 inside a GPTQ checkpoint; stock AutoGPTQConfig would
                 build them as bf16 and fail to load. Saren's vllm_fp8_hybrid.py (verbatim from the qwen38-flash-dgx
                 image) + its hook, gated by VLLM_FP8_HYBRID=1.
    int4 head    the model's ParallelLMHead is built without quant_config, so the int4 AutoRound lm_head cannot load
                 (the MTP head in this vLLM already passes it).
    index filter safety net for --load-format safetensors: reads tensors only where the folder's index says they live.
                 The b12x loader already does this; the stock safetensors loader reads whole files and would load the
                 original layer-1 weights sitting in two PLE-table files over V8's healed ones.
  OPTIONAL (a warning, the launch goes on):
    draft x2     VLLM_MTP_DRAFT_SCALE scales the draft's logits (exact: the verifier decides) -- +2.5pp acceptance.
    GDN fixes    spark-fla-shmem (GB10 gets the big GDN tiles) and spark-fla-warps (num_warps=2: fla#953 Blackwell race
                 that corrupts GDN state on the prefix-cache path), same one-liners as the qwen38-flash-dgx image.

  usage: patch_b12x.py <mod dir>
"""
import ast, os, re, shutil, sys

MOD = sys.argv[1]
SP = "/usr/local/lib/python3.12/dist-packages"
V = f"{SP}/vllm"
MARK = "flashnext-int4-b12x"
done, warn, fatal = [], [], []


def edit(path, fn, label, required):
    try:
        src = open(path).read()
        if MARK + ":" + label in src:
            done.append(f"{label} (already)")
            return
        new = fn(src)
        if new is None or new == src:
            raise ValueError("anchor not found")
        ast.parse(new)
        open(path, "w").write(new)
        done.append(label)
    except Exception as e:  # noqa: BLE001
        (fatal if required else warn).append(f"{label}: {e} [{path}]")


def once(src, old, new):
    if src.count(old) != 1:
        return None
    return src.replace(old, new)


# 1. fp8 hybrid ----------------------------------------------------------------------------------------------------
shutil.copy(os.path.join(MOD, "vllm_fp8_hybrid.py"), f"{SP}/vllm_fp8_hybrid.py")
edit(f"{V}/model_executor/layers/quantization/auto_gptq.py",
     lambda s: s + f"\n\n# --- {MARK}:fp8-hybrid (VLLM_FP8_HYBRID=1) ---\nfrom vllm_fp8_hybrid import apply as _fp8_hybrid_apply\n"
                   "_fp8_hybrid_apply()\n",
     "fp8-hybrid", True)

# The model code moved: images up to 2026-09-13 ship it as vllm/models/qwen3_8_flash_next/{model,mtp}.py; from the
# 2026-09-21 image (vLLM 0.1.dev21473) it is vllm/models/qwen4_exp/nvidia/{model,mtp}.py (qwen3_8_flash_next is only a
# compatibility alias there). Patch whichever this image has.
MODEL_DIR = next((d for d in (f"{V}/models/qwen4_exp/nvidia", f"{V}/models/qwen3_8_flash_next")
                  if os.path.isfile(f"{d}/model.py")), f"{V}/models/qwen3_8_flash_next")

# 2. int4 lm_head ----------------------------------------------------------------------------------------------------
edit(f"{MODEL_DIR}/model.py",
     lambda s: once(s, '            prefix=maybe_prefix(prefix, "lm_head"),\n',
                    f'            quant_config=vllm_config.quant_config,  # {MARK}:int4-head\n'
                    '            prefix=maybe_prefix(prefix, "lm_head"),\n'),
     "int4-head", True)

# 3. index filter for the stock safetensors loader ----------------------------------------------------------------------
FILTER = f'''

# --- {MARK}:index-filter ---
# Only for folders under /workspace/flashnext: yield a tensor only from the file the folder's index names for it.
def _flashnext_index_filter():
    import json as _json
    from pathlib import Path as _Path
    _orig = DefaultModelLoader._safetensors_weights_iterator
    _maps = {{}}

    def _iterate(self, hf_weights_files, source):
        files = list(hf_weights_files)
        folders = {{_Path(f).parent for f in files}}
        if len(folders) != 1 or not str(next(iter(folders))).startswith("/workspace/flashnext/"):
            yield from _orig(self, files, source)
            return
        folder = next(iter(folders))
        if folder not in _maps:
            _maps[folder] = _json.load(open(folder / "model.safetensors.index.json"))["weight_map"]
        weight_map = _maps[folder]
        for f in files:
            here = _Path(f).name
            for name, tensor in _orig(self, [f], source):
                if weight_map.get(name) == here:
                    yield name, tensor

    DefaultModelLoader._safetensors_weights_iterator = _iterate


_flashnext_index_filter()
'''
edit(f"{V}/model_executor/model_loader/default_loader.py", lambda s: s + FILTER, "index-filter", True)

# 3b. the same filter for "mixed" files ------------------------------------------------------------------------------------
# A file that holds any PLE-table tensor does NOT go through _safetensors_weights_iterator: b12x's vLLM reads it directly in
# file_backed_safetensors_weights_iterator (every tensor in it, table rows as on-disk references). Two of the table files also
# carry layer 1's ORIGINAL experts / shared expert / gate -> boot 2 died on "experts has no parameter 'w2_weight'". Wrap that
# iterator too (default_loader calls it by its module-global name, so rebinding the name there is enough).
MIXED = f'''

# --- {MARK}:index-filter-mixed ---
def _flashnext_index_filter_mixed():
    import json as _json
    from pathlib import Path as _Path
    global file_backed_safetensors_weights_iterator
    _orig = file_backed_safetensors_weights_iterator
    _maps = {{}}

    def _filtered(hf_weights_files, ordinary_iterator, file_weight_filter, **kwargs):
        files = list(hf_weights_files)
        folders = {{_Path(f).parent for f in files}}
        if len(folders) != 1 or not str(next(iter(folders))).startswith("/workspace/flashnext/"):
            yield from _orig(files, ordinary_iterator, file_weight_filter, **kwargs)
            return
        folder = next(iter(folders))
        if folder not in _maps:
            _maps[folder] = _json.load(open(folder / "model.safetensors.index.json"))["weight_map"]
        weight_map = _maps[folder]
        for f in files:
            here = _Path(f).name
            for name, tensor in _orig([f], ordinary_iterator, file_weight_filter, **kwargs):
                if weight_map.get(name) == here:
                    yield name, tensor

    file_backed_safetensors_weights_iterator = _filtered


_flashnext_index_filter_mixed()
'''
edit(f"{V}/model_executor/model_loader/default_loader.py",
     lambda s: s + MIXED if "file_backed_safetensors_weights_iterator" in s else None, "index-filter-mixed", True)

# 4. draft x2 --------------------------------------------------------------------------------------------------------
def _scale(s):
    pat = re.compile(r"(self\.logits_processor = LogitsProcessor\(\n(\s+)config\.vocab_size,\n)")
    if len(pat.findall(s)) != 1:
        return None
    return pat.sub(lambda m: m.group(1) + m.group(2) +
                   f'scale=float(__import__("os").environ.get("VLLM_MTP_DRAFT_SCALE", "1.0")),  # {MARK}:draft-scale\n', s)
edit(f"{MODEL_DIR}/mtp.py", _scale, "draft-scale", False)

# 5. GDN fixes ---------------------------------------------------------------------------------------------------------
fla = f"{V}/third_party/flash_linear_attention/ops"
edit(f"{fla}/utils.py", lambda s: once(s, "DEFAULT = 102400", f"DEFAULT = 101376  # {MARK}:fla-shmem GB10 99KiB"),
     "fla-shmem", False)
edit(f"{fla}/chunk_delta_h.py",
     lambda s: once(s, "for num_warps in [2, 4]", f"for num_warps in [2]  # {MARK}:fla-warps fla#953"),
     "fla-warps", False)

print("patched: " + ", ".join(done) + ("" if not warn else " | WARNING (optional, skipped): " + "; ".join(warn)))
if fatal:
    sys.exit("FATAL required patch failed: " + "; ".join(fatal))
