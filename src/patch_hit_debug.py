#!/usr/bin/env python3
"""Prefix-cache diagnosis logging, gated by VLLM_HIT_DEBUG=1 (off = zero cost).

Four log points to pinpoint why a repeated prompt misses the prefix cache on
a hybrid (attention + mamba align-mode) model, where the reconciled hit is the
min across all KV cache groups:

  1. [hit-debug] reconcile  — per-group hit lengths after the fixed point in
     HybridKVCacheCoordinator.find_longest_cache_hit (which group truncated).
  2. [hit-debug] mamba-pub  — MambaManager.cache_blocks scan range and per-slot
     real/null/hashed state (was the boundary state ever published?).
  3. [hit-debug] evict      — a cached block losing its hash(es) to eviction
     (did churn kill the entry between two requests?).
  4. [hit-debug] chunk-stop — _mamba_block_aligned_split clip decisions (where
     did prefill chunks actually end?).

Build-time script: edits vLLM sources in place inside the image. Every anchor
is asserted unique so upstream drift fails the build instead of silently
mis-patching.
"""

import ast
import sys

SP = "/usr/local/lib/python3.12/dist-packages/vllm"
FILES = {
    "coord": f"{SP}/v1/core/kv_cache_coordinator.py",
    "single": f"{SP}/v1/core/single_type_kv_cache_manager.py",
    "pool": f"{SP}/v1/core/block_pool.py",
    "sched": f"{SP}/v1/core/sched/scheduler.py",
}

GATE = '''
import os as _os_hitdbg

_HIT_DEBUG = _os_hitdbg.environ.get("VLLM_HIT_DEBUG") == "1"
'''


def edit(path: str, old: str, new: str) -> None:
    src = open(path).read()
    n = src.count(old)
    assert n == 1, f"{path}: anchor found {n} times (want 1):\n{old[:200]}"
    open(path, "w").write(src.replace(old, new))


def add_gate(path: str) -> None:
    edit(path, "\nlogger = init_logger(__name__)\n",
         "\nlogger = init_logger(__name__)\n" + GATE)


for key in ("coord", "pool", "sched"):
    add_gate(FILES[key])

# single_type_kv_cache_manager.py has no logger of its own; add both.
edit(
    FILES["single"],
    """from vllm.v1.request import Request
""",
    """from vllm.v1.request import Request

from vllm.logger import init_logger

logger = init_logger(__name__)
"""
    + GATE,
)

# ------------------------------------------------- kv_cache_coordinator.py
# 1. Per-group breakdown after the cross-group fixed point.
edit(
    FILES["coord"],
    """        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
""",
    """        num_uncached_common_prefix_tokens = longest_hit_length - hit_length
        if _HIT_DEBUG:
            logger.info(
                "[hit-debug] reconcile: max=%d hit=%d longest=%d per-group=%s",
                max_cache_hit_length,
                hit_length,
                longest_hit_length,
                "|".join(
                    f"g{gid}:{length}"
                    for gid, length in enumerate(hit_length_by_group)
                ),
            )
""",
)

# ------------------------------------------- single_type_kv_cache_manager.py
# 2. Mamba publication: which slots were real/null/hashed at scan time.
edit(
    FILES["single"],
    """        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)
""",
    """        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)
        if _HIT_DEBUG:
            _blocks = self.req_to_blocks[request.request_id]
            _hi = min(max(num_cached_blocks_after, num_cached_blocks_before + 1),
                      len(_blocks))
            logger.info(
                "[hit-debug] mamba-pub g%d req=%s num_tokens=%d scan=[%d,%d) "
                "slots=%s",
                self.kv_cache_group_id,
                request.request_id,
                num_tokens,
                num_cached_blocks_before,
                num_cached_blocks_after,
                "|".join(
                    f"{i}:" + (
                        "null" if _blocks[i].is_null
                        else f"b{_blocks[i].block_id}"
                        + ("+h" if _blocks[i].block_hash is not None else "-h")
                    )
                    for i in range(num_cached_blocks_before, _hi)
                ),
            )
""",
)

# ---------------------------------------------------------- block_pool.py
# 3. Eviction of a cached block (hash destroyed).
edit(
    FILES["pool"],
    """        evicted_hashes = self._remove_cached_block_hashes(block)
        if not evicted_hashes:
""",
    """        evicted_hashes = self._remove_cached_block_hashes(block)
        if _HIT_DEBUG and evicted_hashes:
            from vllm.v1.core.kv_cache_utils import get_group_id as _ggid
            logger.info(
                "[hit-debug] evict b%d groups=%s",
                block.block_id,
                ",".join(str(_ggid(h)) for h in evicted_hashes),
            )
        if not evicted_hashes:
""",
)

# ---------------------------------------------------- sched/scheduler.py
# 4. Chunk clip decisions in the mamba align split.
edit(
    FILES["sched"],
    """        # Stop at the earliest mandatory position strictly inside the chunk.
        end = min((s for s in stops if start < s < end), default=end)
        return max(end - start, 0)
""",
    """        # Stop at the earliest mandatory position strictly inside the chunk.
        end = min((s for s in stops if start < s < end), default=end)
        if _HIT_DEBUG:
            logger.info(
                "[hit-debug] chunk-stop req=%s start=%d end=%d "
                "prompt=%d last_cache=%d stops=%s",
                request.request_id,
                start,
                end,
                request.num_prompt_tokens,
                last_cache_position,
                stops,
            )
        return max(end - start, 0)
""",
)

for path in FILES.values():
    ast.parse(open(path).read())
print("patch_hit_debug.py applied OK", file=sys.stderr)
