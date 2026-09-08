#!/usr/bin/env python3
"""CPU unit test for the exact QSA top-k path added by patch_qsa_exact_topk.py (no GPU needed).

    docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_qsa_exact_topk_cpu.py
"""
import os
import torch

QSA = "/usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
src = open(QSA).read()
tail = src[src.index("# --- GX10: QSA top-k variants"):]
ns = {"torch": torch, "os": os}
exec(tail, ns)  # noqa: S102 - loads only the appended helper functions

torch.manual_seed(0)
rows, cols, k = 6, 1000, 8
logits = torch.randn(rows, cols)
visible = torch.tensor([1000, 600, 128, 8, 3, 0], dtype=torch.int32)
ref = [set(torch.topk(logits[r, : int(visible[r])], min(k, int(visible[r]))).indices.tolist())
       if int(visible[r]) > 0 else set() for r in range(rows)]

blocks = torch.empty(rows, k, dtype=torch.int32)
ns["_qsa_exact_topk"](logits.clone(), visible, blocks, k, cols)
for r in range(rows):
    v = int(visible[r])
    if v == 0:
        continue
    got = set(blocks[r, : min(k, v)].tolist())
    assert got == ref[r], (r, got, ref[r])
    assert all(i < v for i in got), "selected an invisible column"

again = torch.empty(rows, k, dtype=torch.int32)
ns["_qsa_exact_topk"](logits.clone(), visible, again, k, cols)
assert torch.equal(blocks, again), "exact top-k is not deterministic"
print("exact top-k: OK (sets match torch.topk over visible columns, deterministic)")
