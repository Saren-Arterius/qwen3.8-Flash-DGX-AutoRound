#!/usr/bin/env python3
"""Fix: _mamba_block_aligned_split aligns prefill chunks to the wrong size.

The scheduler's ``cache_config.block_size`` is rewritten by EngineCore to the
*minimum* block size across KV cache groups (the fine/draft granularity — 8 on
this model), while the mamba/GDN state groups use 1600-token blocks
(``cache_config.mamba_block_size``, set by _align_hybrid_block_size and left
intact). ``_mamba_block_aligned_split`` exists purely to make prefill chunks
end where a mamba state can be cached, i.e. at MAMBA block boundaries — but it
reads ``cache_config.block_size``.

Observed effect (VLLM_HIT_DEBUG traces, 8036-token prompt): the only chunk
stop lands at 8024 (8-aligned, not 1600-aligned), the state slots all stay
null, the cold request publishes ZERO mamba hashes, and since the hybrid
cache-hit is the min across all groups the 2nd identical request misses
completely (hit=0). Only the shared-prefix-junction machinery, triggered by
that miss, forces a 1600-aligned stop — so the 3rd request is the first to
hit. Long production prompts published no mamba states at all (~1% hit rate),
which also starved the never-evict pin of mamba blocks to hold.

Build-time script: edits vLLM sources in place inside the image. The anchor
is asserted unique so upstream drift fails the build instead of silently
mis-patching.
"""

import ast
import sys

SCHED = "/usr/local/lib/python3.12/dist-packages/vllm/v1/core/sched/scheduler.py"


def edit(path: str, old: str, new: str) -> None:
    src = open(path).read()
    n = src.count(old)
    assert n == 1, f"{path}: anchor found {n} times (want 1):\n{old[:200]}"
    open(path, "w").write(src.replace(old, new))


edit(
    SCHED,
    """        block_size = self.cache_config.block_size
        # The last block-aligned position whose state can be cached. With
""",
    """        # cache_config.block_size is the MINIMUM across KV cache groups (the
        # fine/draft granularity), but this function's whole job is to end
        # chunks at MAMBA state boundaries. With the fine size, chunk ends
        # land where no mamba state is cacheable, a cold request publishes no
        # reusable state, and the min-across-groups hit rule makes the next
        # identical request a full miss.
        block_size = (
            self.cache_config.mamba_block_size or self.cache_config.block_size
        )
        # The last block-aligned position whose state can be cached. With
""",
)

ast.parse(open(SCHED).read())
print("patch_mamba_align_split.py applied OK", file=sys.stderr)
