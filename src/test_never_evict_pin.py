#!/usr/bin/env python3
"""Self-check for the never-evict pin (patch_never_evict.py). Runs inside the
image: docker run --rm --entrypoint python3 qwen38-flash-dgx /t/test_never_evict_pin.py
No GPU needed — BlockPool is pure Python bookkeeping."""

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

pool = BlockPool(num_gpu_blocks=33, enable_caching=True, hash_block_size=16)
base_free = pool.get_num_free_blocks()  # 32 (null block taken)

blocks = pool.get_new_blocks(8)
keys = []
for i, b in enumerate(blocks):
    k = make_block_hash_with_group_id(BlockHash(b"hash%02d" % i), 0)
    b.set_block_hash(k)
    keys.append(k)

# Arming is lazy: nothing is freed yet, so nothing is reserved.
assert pool.set_pinned_hashes(set(keys[:4]), 0.25) == 0

# Freeing claims the pinned hashes; the rest go to the free queue.
pool.free_blocks(blocks)
assert pool.get_num_free_blocks() == base_free - 4, "pinned blocks counted free"
reserved, held, groups = pool.get_pin_stats()
assert (reserved, held, groups) == (4, 4, {0: 4}), (reserved, held, groups)

# Draining the whole free pool must never hand out a pinned block.
pinned_ids = {b.block_id for b in blocks[:4]}
drained = pool.get_new_blocks(pool.get_num_free_blocks())
assert not (pinned_ids & {b.block_id for b in drained}), "pin leaked into alloc"
pool.free_blocks(drained)

# A prefix hit loans a pinned block out (touch) and returns it on free.
pb = blocks[0]
pool.touch([pb])
assert pool.get_pin_stats()[:2] == (4, 3)
pool.free_blocks([pb])
assert pool.get_pin_stats()[:2] == (4, 4)

# Re-arming with a new set releases the old blocks back to the free queue.
free_before = pool.get_num_free_blocks()
assert pool.set_pinned_hashes(set(keys[4:6]), 0.25) == 0
assert pool.get_num_free_blocks() == free_before + 4
assert pool.get_pin_stats()[0] == 0

# The cap bounds the pin: fraction 1/33 -> cap 1 block.
b2 = pool.get_new_blocks(2)
k2 = []
for i, b in enumerate(b2):
    k = make_block_hash_with_group_id(BlockHash(b"cap%02d" % i), 1)
    b.set_block_hash(k)
    k2.append(k)
pool.set_pinned_hashes(set(k2), 1 / 33)
pool.free_blocks(b2)
assert pool.get_pin_stats()[0] == 1, "cap not enforced"

# reset_prefix_cache drops the pin and succeeds.
assert pool.reset_prefix_cache(), "reset failed with live pin"
assert pool.get_pin_stats()[0] == 0
assert pool.get_num_free_blocks() == base_free

print("never-evict pin self-check PASSED")
