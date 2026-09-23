"""patch_mamba_guard.py check (GPU, inside the b12x image): an out-of-range block id is skipped
and counted instead of faulting, while a valid copy in the same launch still lands.

    docker run --rm --gpus all --entrypoint python3 qwen38-flash-b12x /opt/llm/test_mamba_guard_gpu.py
"""
import torch

from vllm.v1.worker import mamba_utils as mu

dev = "cuda"
state = torch.arange(4 * 256, dtype=torch.float32, device=dev).reshape(4, 256)  # 4 blocks, temporal
orig = state.clone()
# req 0: col 0 (block 1) -> col 1 (block 2), valid; req 1: col 0 (block 3) -> col 1 (block 99), out of range
bt = torch.tensor([[1, 2] + [0] * 6, [3, 99] + [0] * 6], dtype=torch.int32, device=dev)
i64 = lambda *v: torch.tensor(v, dtype=torch.int64, device=dev)  # noqa: E731
i32 = lambda *v: torch.tensor(v, dtype=torch.int32, device=dev)  # noqa: E731
guard_hits = torch.zeros(1, dtype=torch.int32, device=dev)

mu.precopy_mamba_align_fused_kernel[(2, 1, 1)](
    i32(1, 1), i32(0, 0), i32(0, 0),                  # dst col, src col, token bias
    i64(bt.data_ptr()), bt.stride(0),
    i64(state.data_ptr()), i64(state.stride(0) * 4),  # base addr, block stride (bytes)
    i32(4), i64(256), i32(0), i32(0),                 # elem size, inner size, conv width (temporal), group
    i32(0), i64(0),                                   # DS conv rows (unused)
    i64(state.shape[0]), guard_hits,                  # the guard
    i32(0), 2,                                        # idx_mapping (unused), num_reqs
    COPY_BLOCK_SIZE=1024, CONV_STATE_DIM_FIRST=False, HAS_IDX_MAPPING=False, TEMPORAL_TILES=1,
)
torch.cuda.synchronize()
assert torch.equal(state[2], orig[1]), "valid copy did not land"
assert torch.equal(state[[0, 1, 3]], orig[[0, 1, 3]]), "untouched blocks changed"
assert guard_hits.item() == 1, f"guard_hits={guard_hits.item()}, want 1"
print("mamba guard OK: bad block id skipped + counted, valid copy landed, context alive")
