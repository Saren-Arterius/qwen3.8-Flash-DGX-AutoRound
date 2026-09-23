#!/usr/bin/env python3
"""Mamba state-copy bounds guard for the b12x image's vllm/v1/worker/mamba_utils.py.

The production image replaces the whole file (src/mamba_utils_guarded.py = vllm#50729 + this
guard). b12x's vLLM already carries vllm#50729 (memmove-safe conv shifts, left-overlap barrier)
but not the guard, and the file is a rewrite, so the guard goes in here by anchors instead:
an out-of-range block id skips that copy and bumps a counter (one request keeps a stale mamba
state -- recoverable) instead of a wild address killing the CUDA context (Xid 31 MMU faults on
GB10). Every copy path runs through _copy_mamba_state_block, so the check lives there; the
counter is reported every 512 precopy steps, sync-free, one round late.

    python3 patch_mamba_guard.py <path/to/mamba_utils.py>
"""
import ast
import re
import sys

F = sys.argv[1]
src = open(F).read()


def edit(old: str, new: str) -> None:
    global src
    n = src.count(old)
    assert n == 1, f"anchor found {n} times (want 1):\n{old[:200]}"
    src = src.replace(old, new)


def sub(pattern: str, repl: str, want: int) -> None:
    global src
    src, n = re.subn(pattern, repl, src, flags=re.M)
    assert n == want, f"{pattern!r}: {n} matches (want {want})"


CHECK = """{i}if {v} < 0 or {v} >= num_blocks:
{i}    tl.atomic_add(guard_hits_ptr, 1)
{i}    return
"""

# plumbing: every kernel signature / device-call that carries the DS conv row stride also
# carries the per-state block count + hit counter; every launch passes them
sub(r"^(\s*)state_dim_row_stride_ptr,[^\n]*\n", r"\g<0>\1state_num_blocks_ptr,\n\1guard_hits_ptr,\n", 7)
sub(r"^(\s*)self\.state_dim_row_stride,\n", r"\g<0>\1self.state_num_blocks,\n\1self.guard_hits,\n", 4)

# the checks, at each block id load in _copy_mamba_state_block
edit("""        dest_block_id = destination_block_id.to(tl.int64)
    dst_addr = state_base_addr + dest_block_id * state_block_stride
""", """        dest_block_id = destination_block_id.to(tl.int64)
    # qwen38-flash-dgx guard: a corrupt/stale block id would send the copy through a
    # wild address (Xid 31 MMU faults on GB10); skip it and count it instead.
    num_blocks = tl.load(state_num_blocks_ptr + state_idx)
""" + CHECK.format(i="    ", v="dest_block_id") + """    dst_addr = state_base_addr + dest_block_id * state_block_stride
""")
edit("""        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        dim_rows = """, """        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
""" + CHECK.format(i="        ", v="src_block_id") + """        dim_rows = """)
edit("""        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
        src_block_addr = state_base_addr + src_block_id * state_block_stride
        token_bytes""", """        src_block_id = tl.load(block_table_base + src_col).to(tl.int64)
""" + CHECK.format(i="        ", v="src_block_id") + """        src_block_addr = state_base_addr + src_block_id * state_block_stride
        token_bytes""")
edit("""    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
""", """    actual_src_block_id = tl.load(block_table_base + src_col + token_bias).to(tl.int64)
""" + CHECK.format(i="    ", v="actual_src_block_id"))

# context fields, allocation, population
edit("""    state_dim_row_stride: torch.Tensor  # int64: bytes between rows
""", """    state_dim_row_stride: torch.Tensor  # int64: bytes between rows
    # Guard: valid block count per state pool + skipped-copy counter
    state_num_blocks: torch.Tensor  # int64: shape[0] of each state tensor
    guard_hits: torch.Tensor  # int32[1], GPU: out-of-range copies skipped
""")
edit("""    # Flag to track if metadata has been populated
    is_initialized: bool = False
""", """    # Flag to track if metadata has been populated
    is_initialized: bool = False

    # Guard telemetry (sync-free: pinned mirror refreshed asynchronously, read one round late)
    guard_hits_cpu: torch.Tensor | None = None
    guard_step: int = 0
    guard_reported: int = 0
""")
edit("""            state_dim_row_stride=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
""", """            state_dim_row_stride=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            state_num_blocks=torch.zeros(
                total_states, dtype=torch.int64, device=device
            ),
            guard_hits=torch.zeros(1, dtype=torch.int32, device=device),
            guard_hits_cpu=torch.zeros(1, dtype=torch.int32, pin_memory=True),
""")
edit("""                    self.state_base_addrs[idx] = _reinterpret_u64_as_i64(
                        state.data_ptr()
                    )
""", """                    self.state_base_addrs[idx] = _reinterpret_u64_as_i64(
                        state.data_ptr()
                    )
                    self.state_num_blocks[idx] = (
                        state.size(0) if state.dim() > 1 else 1
                    )
""")

# telemetry, after the per-step precopy launch
edit("""            TEMPORAL_TILES=_TEMPORAL_TILES,
        )

    def checkpoint_request_boundaries(""", """            TEMPORAL_TILES=_TEMPORAL_TILES,
        )
        self.guard_step += 1
        if self.guard_step % 512 == 0:
            hits = int(self.guard_hits_cpu[0])
            if hits > self.guard_reported:
                logger.error(
                    "mamba state-copy guard: %d out-of-range block id(s) skipped "
                    "(would have been an illegal memory access). Expected 0 -- "
                    "this indicates a new bug, please report it.",
                    hits,
                )
                self.guard_reported = hits
            self.guard_hits_cpu.copy_(self.guard_hits, non_blocking=True)

    def checkpoint_request_boundaries(""")

ast.parse(src)
open(F, "w").write(src)
print("mamba_utils.py: state-copy bounds guard applied OK")
