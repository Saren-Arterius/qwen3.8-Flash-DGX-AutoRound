#!/usr/bin/env python3
"""Opt-in top-k variants for the Qwen3.8-Flash-Next QSA indexer (VLLM_QSA_EXACT_TOPK):
  0 / unset : stock torch.ops._C.persistent_topk (non-deterministic on GB10, may drop candidates — vllm#51782,
              blazux/qwen3.8-Flash-DGX#3 by @k3dani)
  1         : exact torch.topk over the visible columns (deterministic; costs prefill/decode time)
  fill      : -inf-fill the never-written columns (>= visible) then stock kernel (cheap; tests whether the
              non-determinism comes from uninitialized logits rather than the histogram bug)
Default behaviour unchanged."""
import sys
F = sys.argv[1]
src = open(F).read()
CALL = "        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n"
assert src.count(CALL) == 1, "topk call site not found exactly once"
src = src.replace(CALL,
    "        if _QSA_TOPK_MODE == \"1\":\n"
    "            _qsa_exact_topk(logits, visible_blocks, blocks, block_topk, columns)\n"
    "        elif _QSA_TOPK_MODE == \"fill\":\n"
    "            _qsa_mask_invisible_(logits, visible_blocks, columns)\n"
    "            topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n"
    "        else:\n"
    "            topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n")
src = src.replace("import math\n", "import math\nimport os\n", 1)
src += '''

# --- GX10: QSA top-k variants (VLLM_QSA_EXACT_TOPK = 0 | 1 | fill), see vllm#51782 / qwen3.8-Flash-DGX#3 ---
_QSA_TOPK_MODE = os.environ.get("VLLM_QSA_EXACT_TOPK", "0").lower()
if _QSA_TOPK_MODE in ("true", "yes"):
    _QSA_TOPK_MODE = "1"
_QSA_COLS_CACHE: dict = {}


def _qsa_cols(columns: int, device) -> torch.Tensor:
    key = (columns, device)
    t = _QSA_COLS_CACHE.get(key)
    if t is None:
        t = torch.arange(columns, device=device, dtype=torch.int32)
        _QSA_COLS_CACHE[key] = t
    return t


def _qsa_mask_invisible_(logits, visible_blocks, columns):
    """In-place: columns >= visible_blocks[row] are never written by the scoring kernel (torch.empty)."""
    cols = _qsa_cols(columns, logits.device)
    logits[:, :columns].masked_fill_(cols[None, :] >= visible_blocks[:, None], float("-inf"))


def _qsa_exact_topk(logits, visible_blocks, blocks, block_topk, columns):
    """Exact, deterministic block top-k via torch.topk (in place mask, unsorted).

    The expansion kernel only reads min(visible, block_topk) entries per row, so slots past
    that are don't-care (-1 for hygiene).
    """
    _qsa_mask_invisible_(logits, visible_blocks, columns)
    k = min(block_topk, columns)
    idx = torch.topk(logits[:, :columns], k, dim=1, sorted=False).indices
    if k < block_topk:
        blocks[:, :k].copy_(idx)
        blocks[:, k:].fill_(-1)
    else:
        blocks.copy_(idx)
'''
open(F, "w").write(src)
import ast; ast.parse(src)
print("qsa.py: top-k variants (1|fill) added OK")
