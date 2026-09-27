#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

import ast
import sys

GDN = (
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/"
    "mamba/gdn/qwen_gdn_linear_attn.py"
)

HOOK = (
    "\n\n"
    "# --- qwen38-flash-dgx: in_proj_ba small-N GEMV (VLLM_INPROJ_BA_GEMV=1) ---\n"
    "from vllm_inproj_ba_gemv import apply as _in_proj_ba_gemv_apply\n"
    "_in_proj_ba_gemv_apply(QwenGatedDeltaNetAttention)\n"
)

src = open(GDN).read()

# Fail loudly if any anchor this patch depends on has moved.
for anchor in (
    "class QwenGatedDeltaNetAttention(",
    "self.in_proj_ba = self.create_ba_proj(",
    "ba, _ = self.in_proj_ba(hidden_states)",
    "def create_ba_proj(",
):
    assert anchor in src, "qwen_gdn_linear_attn.py: anchor moved -- %r" % anchor

if "_in_proj_ba_gemv_apply" in src:
    print("patch_in_proj_ba_gemv.py: already applied, skipping", file=sys.stderr)
else:
    open(GDN, "a").write(HOOK)
    ast.parse(open(GDN).read())
    print("patch_in_proj_ba_gemv.py applied OK", file=sys.stderr)
