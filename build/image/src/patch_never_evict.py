#!/usr/bin/env python3
"""Pin-only port of arc_pin2.diff (from the 122B recipe) onto the qwen38next tree.

Adds --never-evict-kv-cache-prompt-includes / --never-evict-kv-cache-max-fraction:
a request whose prompt contains the marker gets its prompt-prefix KV blocks
pinned — held in a side queue on BlockPool, excluded from the free count, so
they are never handed out for eviction. Hash-keyed (not block-id-keyed) because
the hybrid model frees mamba/GDN state blocks mid-request; each freed block is
re-claimed by the pin the moment free_blocks() sees it.

Deliberately dropped from arc_pin2: the ARC/two_queue eviction policies and all
the eviction_policy constructor threading (13 of the diff's 15 coordinator
hunks). The pin sits directly on the stock FreeKVCacheBlockQueue instead.

Build-time script: edits vLLM sources in place inside the image. Every anchor
is asserted unique so upstream drift fails the build instead of silently
mis-patching.
"""

import ast
import sys

SP = "/usr/local/lib/python3.12/dist-packages/vllm"
FILES = {
    "cache": f"{SP}/config/cache.py",
    "args": f"{SP}/engine/arg_utils.py",
    "pool": f"{SP}/v1/core/block_pool.py",
    "sched": f"{SP}/v1/core/sched/scheduler.py",
}


def edit(path: str, old: str, new: str) -> None:
    src = open(path).read()
    n = src.count(old)
    assert n == 1, f"{path}: anchor found {n} times (want 1):\n{old[:200]}"
    open(path, "w").write(src.replace(old, new))


# ---------------------------------------------------------------- cache.py
edit(
    FILES["cache"],
    """    KV offloading is only activated when kv_offloading_size is set.\"\"\"
""",
    """    KV offloading is only activated when kv_offloading_size is set.\"\"\"

    never_evict_kv_cache_prompt_includes: str | None = None
    \"\"\"If set, any request whose prompt contains this exact substring has its
    prompt KV cache blocks pinned: they are held out of the free pool and are
    never handed out for eviction. The pin set is *replaced* the next time a
    matching request arrives, so a changed system prompt releases the old
    blocks automatically. Requires prefix caching.\"\"\"

    never_evict_kv_cache_max_fraction: float = 0.25
    \"\"\"Upper bound on the never-evict pin, as a fraction of the GPU block
    pool. Pinning stops (with a warning) once the cap is reached.\"\"\"
""",
)
edit(
    FILES["cache"],
    """            "prefix_caching_hash_algo",
""",
    """            "prefix_caching_hash_algo",
            "never_evict_kv_cache_prompt_includes",
            "never_evict_kv_cache_max_fraction",
""",
)

# ------------------------------------------------------------ arg_utils.py
edit(
    FILES["args"],
    """    kv_offloading_backend: KVOffloadingBackend = CacheConfig.kv_offloading_backend
""",
    """    kv_offloading_backend: KVOffloadingBackend = CacheConfig.kv_offloading_backend
    never_evict_kv_cache_prompt_includes: str | None = (
        CacheConfig.never_evict_kv_cache_prompt_includes
    )
    never_evict_kv_cache_max_fraction: float = (
        CacheConfig.never_evict_kv_cache_max_fraction
    )
""",
)
edit(
    FILES["args"],
    """        cache_group.add_argument(
            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]
        )
""",
    """        cache_group.add_argument(
            "--kv-offloading-backend", **cache_kwargs["kv_offloading_backend"]
        )
        cache_group.add_argument(
            "--never-evict-kv-cache-prompt-includes",
            **cache_kwargs["never_evict_kv_cache_prompt_includes"],
        )
        cache_group.add_argument(
            "--never-evict-kv-cache-max-fraction",
            **cache_kwargs["never_evict_kv_cache_max_fraction"],
        )
""",
)
edit(
    FILES["args"],
    """            kv_offloading_backend=self.kv_offloading_backend,
""",
    """            kv_offloading_backend=self.kv_offloading_backend,
            never_evict_kv_cache_prompt_includes=(
                self.never_evict_kv_cache_prompt_includes
            ),
            never_evict_kv_cache_max_fraction=(
                self.never_evict_kv_cache_max_fraction
            ),
""",
)

# ----------------------------------------------------------- block_pool.py
edit(
    FILES["pool"],
    """        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True
""",
    """        self.null_block = self.free_block_queue.popleft()
        self.null_block.is_null = True

        # never-evict pin (pin-only port of arc_pin2). Pinned blocks live in
        # their own queue: not in free_block_queue means get_new_blocks() can
        # never hand them out and get_num_free_blocks() never counts them, so
        # the scheduler never even asks for them. Keyed on block *hash*: the
        # hybrid model frees mamba/GDN state blocks mid-request (hash intact),
        # and _acquire_pin() claims each one the moment it is freed.
        self._pinned_hashes: set[BlockHashWithGroupId] = set()
        self._pin_cap: int = 0
        self._pin: FreeKVCacheBlockQueue = FreeKVCacheBlockQueue([])
        # hash -> block_id currently holding the reservation (survives loans).
        self._pinned_by_hash: dict[BlockHashWithGroupId, int] = {}
        # block ids sitting in _pin right now; lets touch() pick the queue.
        self._pinned_ids: set[int] = set()
        self._pin_cap_warned: bool = False
""",
)
edit(
    FILES["pool"],
    """        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(blocks_to_evict_first)
""",
    """        if self._pinned_hashes:
            # Claim pinned hashes as they come home. Only hashed blocks can
            # match, so only evict_last can hit; filter both defensively.
            blocks_to_evict_first = [
                b for b in blocks_to_evict_first if not self._acquire_pin(b)
            ]
            blocks_to_evict_last = [
                b for b in blocks_to_evict_last if not self._acquire_pin(b)
            ]
        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(blocks_to_evict_first)
""",
)
edit(
    FILES["pool"],
    """            if block.ref_cnt == 0 and not block.is_null:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1
""",
    """            if block.ref_cnt == 0 and not block.is_null:
                if block.block_id in self._pinned_ids:
                    # Loaned out to a request. Keep its _pinned_by_hash
                    # reservation so free_blocks() re-pins it when it returns.
                    self._pinned_ids.discard(block.block_id)
                    self._pin.remove(block)
                else:
                    self.free_block_queue.remove(block)
            block.ref_cnt += 1
""",
)
edit(
    FILES["pool"],
    """        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
""",
    """        # Drop the pin first: pinned blocks count as used, so a live pin
        # would make the reset fail forever.
        self.set_pinned_hashes(set(), 0.0)
        num_used_blocks = self.num_gpu_blocks - self.get_num_free_blocks()
""",
)
edit(
    FILES["pool"],
    """        return self.free_block_queue.num_free_blocks

    def get_usage(self) -> float:
""",
    '''        return self.free_block_queue.num_free_blocks

    def _acquire_pin(self, block: KVCacheBlock) -> bool:
        """Put *block* in the pin queue if its hash is pinned. True if pinned.

        Called from free_blocks(), so a block gets held the moment it is
        freed — including the mid-request frees from remove_skipped_blocks()
        that a finish-time snapshot cannot see.
        """
        key = block.block_hash
        if key is None or key not in self._pinned_hashes:
            return False
        holder = self._pinned_by_hash.get(key)
        if holder is None:
            if len(self._pinned_by_hash) >= self._pin_cap:
                if not self._pin_cap_warned:
                    self._pin_cap_warned = True
                    logger.warning(
                        "[never-evict] pin cap reached at %d blocks: the "
                        "marked prefix does not fit and will not stay "
                        "resident. Shorten it or raise "
                        "--never-evict-kv-cache-max-fraction.",
                        self._pin_cap,
                    )
                return False
            self._pinned_by_hash[key] = block.block_id
        elif holder != block.block_id:
            # A different block already holds this hash; pinning both would
            # just burn pool.
            return False
        self._pinned_ids.add(block.block_id)
        self._pin.append(block)
        return True

    def set_pinned_hashes(
        self, hashes: set[BlockHashWithGroupId], max_fraction: float
    ) -> int:
        """Replace the never-evict hash set. Returns the blocks now reserved.

        Releasing the old pin is eager (get_num_free_blocks() drives
        admission control); acquiring the new one is lazy via _acquire_pin()
        as the marked request re-caches and frees its prefix while running.
        """
        cap = max(1, int(self.num_gpu_blocks * max_fraction))
        if hashes == self._pinned_hashes and cap == self._pin_cap:
            # Same marked prompt as last time (every HA turn re-arms it).
            return len(self._pinned_by_hash)

        self._pinned_hashes = hashes
        self._pin_cap = cap
        self._pin_cap_warned = False

        kept: dict[BlockHashWithGroupId, int] = {}
        rehomed: list[KVCacheBlock] = []
        for block in self._pin.popleft_n(self._pin.num_free_blocks):
            key = block.block_hash
            if (
                key is not None
                and key in self._pinned_hashes
                and key not in kept
                and len(kept) < self._pin_cap
            ):
                kept[key] = block.block_id
                self._pin.append(block)
            else:
                rehomed.append(block)
        self.free_block_queue.append_n(rehomed)
        # Blocks on loan are in no queue; they lose their reservation here
        # and re-acquire it (or not) via free_blocks() when they come back.
        self._pinned_by_hash = kept
        self._pinned_ids = set(kept.values())
        return len(kept)

    def get_pin_stats(self) -> tuple[int, int, dict[int, int]]:
        """(blocks reserved, blocks held in the pin queue, per-group counts)."""
        counts: dict[int, int] = {}
        for key in self._pinned_by_hash:
            gid = get_group_id(key)
            counts[gid] = counts.get(gid, 0) + 1
        return len(self._pinned_by_hash), self._pin.num_free_blocks, counts

    def get_usage(self) -> float:
''',
)

# ------------------------------------------------------------ scheduler.py
edit(
    FILES["sched"],
    """from vllm.v1.core.kv_cache_utils import KVCacheBlock
""",
    """from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashListWithBlockSize,
    BlockHashWithGroupId,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
""",
)
edit(
    FILES["sched"],
    """class Scheduler(SchedulerInterface):
""",
    '''def _contains_subseq(haystack: list[int], needle: list[int]) -> bool:
    """True if *needle* appears as a contiguous run inside *haystack*.

    list.index does the scanning in C, so this stays around a millisecond
    even on a 262k-token prompt, and it runs once per request.
    """
    n = len(needle)
    if not n or len(haystack) < n:
        return False
    first = needle[0]
    i = 0
    try:
        while True:
            i = haystack.index(first, i)
            if haystack[i : i + n] == needle:
                return True
            i += 1
    except ValueError:
        return False


class Scheduler(SchedulerInterface):
''',
)
edit(
    FILES["sched"],
    """            watermark=self.scheduler_config.watermark,
        )
""",
    """            watermark=self.scheduler_config.watermark,
        )

        # --never-evict-kv-cache-prompt-includes: token-id needle matched
        # against every new request's prompt.
        self._pin_needle: list[int] = []
        self._pin_hash_block_size = hash_block_size
        self._pin_page_sizes = [
            g.kv_cache_spec.page_size_bytes for g in kv_cache_config.kv_cache_groups
        ]
        marker = self.cache_config.never_evict_kv_cache_prompt_includes
        if marker:
            if not self.cache_config.enable_prefix_caching:
                logger.warning(
                    "[never-evict] disabled: prefix caching is off, so there "
                    "is nothing to pin"
                )
            else:
                tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
                ids = (
                    tokenizer.encode(marker, add_special_tokens=False)
                    if tokenizer is not None
                    else []
                )
                # Drop the first and last token — BPE can merge them with
                # whatever text surrounds the marker in the real prompt.
                self._pin_needle = ids[1:-1] if len(ids) > 2 else ids
                num_gpu_blocks = self.kv_cache_manager.block_pool.num_gpu_blocks
                logger.info(
                    "[never-evict] armed: %d-token marker, pin capped at %d "
                    "blocks (%.0f%% of %d)",
                    len(self._pin_needle),
                    max(
                        1,
                        int(
                            num_gpu_blocks
                            * self.cache_config.never_evict_kv_cache_max_fraction
                        ),
                    ),
                    100 * self.cache_config.never_evict_kv_cache_max_fraction,
                    num_gpu_blocks,
                )
""",
)
edit(
    FILES["sched"],
    """            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)
""",
    """            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)
            if self._pin_needle and _contains_subseq(
                request.prompt_token_ids or [], self._pin_needle
            ):
                self._arm_never_evict_pin(request)
""",
)
edit(
    FILES["sched"],
    """    def finish_requests(
""",
    '''    def _arm_never_evict_pin(self, request: Request) -> None:
        """Pin the marked prompt's prefix, replacing whatever was pinned before.

        Armed when the request *arrives*, not when it finishes: a hybrid model
        releases cached blocks while the request is still running (mamba/GDN
        groups free their state block every step via remove_skipped_blocks),
        so by the time it finishes those blocks are already ordinary eviction
        candidates. Arming up front means each one is claimed by the pin at
        the moment it is freed.
        """
        block_pool = self.kv_cache_manager.block_pool
        # Report what the outgoing pin managed to hold before replacing it.
        self.log_never_evict_pin()
        hashes: set[BlockHashWithGroupId] = set()
        for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
            block_size = group.kv_cache_spec.block_size
            if (
                not group.kv_cache_spec.prefix_cacheable
                or block_size % self._pin_hash_block_size != 0
            ):
                # Non-cacheable groups (e.g. the MTP draft layer) never hash
                # their blocks — nothing to pin, and the hash conversion below
                # would assert on their block size.
                continue
            # Groups coarser than hash_block_size key their cache entries on
            # *combined* hashes — the same conversion find_longest_cache_hit
            # does, so reuse it rather than cross-producing the fine hashes.
            group_hashes: BlockHashList = (
                request.block_hashes
                if block_size == self._pin_hash_block_size
                else BlockHashListWithBlockSize(
                    request.block_hashes, self._pin_hash_block_size, block_size
                )
            )
            for i in range(
                min(len(group_hashes), request.num_prompt_tokens // block_size)
            ):
                hashes.add(make_block_hash_with_group_id(group_hashes[i], gid))

        reserved = block_pool.set_pinned_hashes(
            hashes, self.cache_config.never_evict_kv_cache_max_fraction
        )
        logger.info(
            "[never-evict] armed from request %s (%d prompt tokens): %d "
            "hashes over %d groups, %d blocks already reserved, %d free",
            request.request_id,
            request.num_prompt_tokens,
            len(hashes),
            len(self.kv_cache_config.kv_cache_groups),
            reserved,
            block_pool.get_num_free_blocks(),
        )

    def log_never_evict_pin(self) -> None:
        """Log what the pin actually holds."""
        block_pool = self.kv_cache_manager.block_pool
        reserved, held, per_group = block_pool.get_pin_stats()
        if not reserved:
            return
        num_bytes = sum(
            count * self._pin_page_sizes[gid] for gid, count in per_group.items()
        )
        logger.info(
            "[never-evict] holding %d blocks (%d in the pin queue, rest on "
            "loan), %.1f MiB, %.2f%% of the KV pool; per-group %s",
            reserved,
            held,
            num_bytes / (1 << 20),
            100.0 * reserved / block_pool.num_gpu_blocks,
            "|".join(
                f"g{gid}:{per_group.get(gid, 0)}"
                for gid in range(len(self.kv_cache_config.kv_cache_groups))
            ),
        )

    def finish_requests(
''',
)

for name, path in FILES.items():
    ast.parse(open(path).read())
print("never-evict pin patched OK:", ", ".join(FILES))
sys.exit(0)
