#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

import ast
import sys

SP = "/usr/local/lib/python3.12/dist-packages/vllm"

SPEC = SP + "/config/speculative.py"
SCHED = SP + "/v1/core/sched/scheduler.py"
KVUTILS = SP + "/v1/core/kv_cache_utils.py"
STKV = SP + "/v1/core/single_type_kv_cache_manager.py"
OFFLOAD = SP + "/v1/simple_kv_offload/manager.py"
CONNSCHED = SP + "/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py"
MOONCAKE = SP + "/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py"

# (path, sentinel, [(old, new), ...])
EDITS = [
    # ------------------------------------------------------------- 1. config
    (
        SPEC,
        "_KEEP_DRAFT_BLOCKS",
        [
            # 1a. the import-time env latch.
            (
                """logger = init_logger(__name__)
""",
                """import os as _kdb_os

logger = init_logger(__name__)

# --- qwen38-flash-dgx R8 (backport of vllm#53388). Latched once at import, so
# --- with VLLM_KEEP_DRAFT_BLOCKS unset `use_eagle_block_drop()` below reduces
# --- to upstream's expression and every call site takes the branch it took
# --- before this patch existed.
_KEEP_DRAFT_BLOCKS = _kdb_os.environ.get("VLLM_KEEP_DRAFT_BLOCKS", "0") == "1"
""",
            ),
            # 1b. upstream's field, verbatim (name, type, default, docstring).
            (
                """    use_local_argmax_reduction: bool = False
    \"\"\"Use vocab-parallel local argmax instead of all-gathering full logits
""",
                """    disable_eagle_block_drop: bool = False
    \"\"\"Disable dropping the trailing prefix-cache block for EAGLE-like
    speculative methods. This is an experimental option for measuring the
    acceptance-rate impact of reusing that block. It does not disable the
    speculative drafter itself.\"\"\"
    use_local_argmax_reduction: bool = False
    \"\"\"Use vocab-parallel local argmax instead of all-gathering full logits
""",
            ),
            # 1c. upstream's predicate, plus the env disjunct.
            (
                """        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")

    def use_dflash(self) -> bool:
""",
                """        return self.method in ("eagle", "eagle3", "mtp", "dflash", "dspark")

    def use_eagle_block_drop(self) -> bool:
        \"\"\"Whether volatile trailing cache blocks should be discarded.\"\"\"
        # Upstream vllm#53388 is exactly:
        #     return self.use_eagle() and not self.disable_eagle_block_drop
        # The `or _KEEP_DRAFT_BLOCKS` disjunct is ours: it lets the harness gate
        # the arm on an env name instead of a sixth --speculative-config key.
        # With VLLM_KEEP_DRAFT_BLOCKS unset it is False and this is upstream.
        return self.use_eagle() and not (
            self.disable_eagle_block_drop or _KEEP_DRAFT_BLOCKS
        )

    def use_dflash(self) -> bool:
""",
            ),
        ],
    ),
    # ---------------------------------------------------------- 2. scheduler
    (
        SCHED,
        "use_eagle_block_drop",
        [
            # 2a. declare it beside self.use_eagle.
            (
                """        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
""",
                """        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        # R8 (vllm#53388): the prefix-cache half of `use_eagle`. Identical to
        # `use_eagle` unless the trailing-block drop is disabled.
        self.use_eagle_block_drop = False
        self.num_spec_tokens = vllm_config.num_speculative_tokens
""",
            ),
            # 2b. set it, and warn -- upstream's own wording.
            (
                """            self.use_eagle = speculative_config.use_eagle()
            if self.use_eagle:
                self.num_prefill_lookahead = (
                    self.num_spec_tokens
                    if speculative_config.use_multi_module_mtp()
                    else 1
                )
""",
                """            self.use_eagle = speculative_config.use_eagle()
            if self.use_eagle:
                self.num_prefill_lookahead = (
                    self.num_spec_tokens
                    if speculative_config.use_multi_module_mtp()
                    else 1
                )
            # R8 (vllm#53388). NOTE the asymmetry, which is upstream's and is
            # deliberate: `num_prefill_lookahead` above stays on `use_eagle`
            # (the drafter still reads one token past the chunk boundary); only
            # the prefix-cache consumers move to `use_eagle_block_drop`.
            self.use_eagle_block_drop = speculative_config.use_eagle_block_drop()
            if self.use_eagle and not self.use_eagle_block_drop:
                logger.warning(
                    "EAGLE trailing prefix-cache block dropping is disabled. "
                    "This is experimental and may affect speculative-token "
                    "acceptance rates."
                )
""",
            ),
            # Install signal for this patch:
            # exactly one INFO line, always, so an unattended arm is gated on
            # the log rather than assumed to have installed.
            (
                """        # Create the KV cache manager.
        if hash_block_size is None:
""",
                """        # R8 install signal: exactly
        # one INFO line on every start, whichever way the knob is set.
        if self.use_eagle and not self.use_eagle_block_drop:
            logger.info(
                "iter6 R8 installed: keep-draft-blocks on "
                "(vllm#53388 disable_eagle_block_drop; method=%s, "
                "prefix_caching=%s, mamba_cache_mode=%s, block_size=%s, "
                "mamba_block_size=%s, prefill_lookahead=%d)",
                speculative_config.method,
                self.cache_config.enable_prefix_caching,
                self.cache_config.mamba_cache_mode,
                self.cache_config.block_size,
                self.cache_config.mamba_block_size,
                self.num_prefill_lookahead,
            )
        else:
            logger.info("iter6 R8 disabled")

        # Create the KV cache manager.
        if hash_block_size is None:
""",
            ),
            # 2d. consumer 1 of 2: the KV cache manager.
            (
                """            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            num_prefill_lookahead=self.num_prefill_lookahead,
""",
                """            enable_caching=self.cache_config.enable_prefix_caching,
            # R8 (vllm#53388): the ONE flip that removes the -1600 term from
            # our measured cache formula. Propagates through
            # KVCacheManager -> KVCacheCoordinator.eagle_group_ids ->
            # SpecGroup.use_eagle -> FullAttentionManager.find_longest_cache_hit
            # (`hit_length -= min(alignment_tokens, block_size)`).
            use_eagle=self.use_eagle_block_drop,
            num_prefill_lookahead=self.num_prefill_lookahead,
""",
            ),
            # 2e. consumer 2 of 2: the mamba align-chunk split.
            (
                """        last_cache_position = request.num_tokens - request.num_tokens % block_size
        if self.use_eagle:
            last_cache_position = max(last_cache_position - block_size, 0)
""",
                """        last_cache_position = request.num_tokens - request.num_tokens % block_size
        # R8 (vllm#53388): without the drop there is nothing to back off from,
        # so the mamba checkpoint lands at the same boundary the attention hit
        # now reaches. Backing off here while the attention group no longer
        # drops would cap the reconciled hybrid hit one block short and throw
        # the whole lever away.
        if self.use_eagle_block_drop:
            last_cache_position = max(last_cache_position - block_size, 0)
""",
            ),
        ],
    ),
    # ------------------------------------------------------- 3. kv_cache_utils
    (
        KVUTILS,
        "use_eagle_block_drop",
        [
            (
                """    spec_config = vllm_config.speculative_config
    if spec_config is None or not spec_config.use_eagle():
        return
    # Detection uses the merged MLA spec's model_version.
""",
                """    spec_config = vllm_config.speculative_config
    # R8 (vllm#53388). Upstream's hunk is in the later, generalised
    # `_annotate_eagle_groups`; ours is still the DeepseekV4-only predecessor,
    # so this is inert on Qwen3.8-Flash-Next (the model_version test below
    # returns first). Ported anyway so the two trees read the same.
    if spec_config is None or not spec_config.use_eagle_block_drop():
        return
    # Detection uses the merged MLA spec's model_version.
""",
            ),
        ],
    ),
    # -------------------------------------------- 4. single-type KV managers
    (
        STKV,
        "_KEEP_DRAFT_BLOCKS",
        [
            (
                """_HIT_DEBUG = _os_hitdbg.environ.get("VLLM_HIT_DEBUG") == "1"
""",
                """_HIT_DEBUG = _os_hitdbg.environ.get("VLLM_HIT_DEBUG") == "1"

# --- qwen38-flash-dgx R8 (backport of vllm#53388), see below.
_KEEP_DRAFT_BLOCKS = _os_hitdbg.environ.get("VLLM_KEEP_DRAFT_BLOCKS", "0") == "1"
""",
            ),
            (
                """        # Fine-grained partial hits are not supported for sliding window now
        assert alignment_tokens % kv_cache_spec.block_size == 0, (
            "SlidingWindowManager does not support fine-grained (partial) cache hits"
        )
""",
                """        # R8 (vllm#53388). Upstream DELETES this precondition outright, with the
        # comment "Sliding-window cache hits must stay at the group's physical
        # block granularity. resolve_block_hashes() converts finer-grained
        # hashes to that view when the hybrid-cache alignment is smaller than
        # block_size." We keep it whenever the knob is off, so that with
        # VLLM_KEEP_DRAFT_BLOCKS unset this file is behaviourally identical
        # rather than merely identical in practice. Inert on this model: no
        # SlidingWindowSpec group is built for Qwen3.8-Flash-Next.
        assert _KEEP_DRAFT_BLOCKS or alignment_tokens % kv_cache_spec.block_size == 0, (
            "SlidingWindowManager does not support fine-grained (partial) cache hits"
        )
""",
            ),
        ],
    ),
    # ------------------------------------------------ 5. simple KV offloading
    (
        OFFLOAD,
        "use_eagle_block_drop",
        [
            (
                """        spec_config = vllm_config.speculative_config
        use_eagle = spec_config is not None and spec_config.use_eagle()
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
""",
                """        spec_config = vllm_config.speculative_config
        # R8 (vllm#53388), verbatim. Inert here: kv_offloading_size is unset.
        use_eagle_block_drop = (
            spec_config is not None and spec_config.use_eagle_block_drop()
        )
        self.cpu_coordinator: KVCacheCoordinator = get_kv_cache_coordinator(
""",
            ),
            (
                """            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=use_eagle,
            enable_caching=True,
""",
                """            max_in_flight_tokens=vllm_config.max_in_flight_tokens,
            use_eagle=use_eagle_block_drop,
            enable_caching=True,
""",
            ),
        ],
    ),
    # -------------------------------------------- 6. offloading connector sched
    (
        CONNSCHED,
        "use_eagle_block_drop",
        [
            (
                """        use_eagle = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle()
        )
        if use_eagle and not eagle_groups:
""",
                """        # R8 (vllm#53388), verbatim. Inert here: no offloading connector.
        use_eagle_block_drop = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle_block_drop()
        )
        if use_eagle_block_drop and not eagle_groups:
""",
            ),
        ],
    ),
    # ---------------------------------------------------- 7. mooncake worker
    (
        MOONCAKE,
        "use_eagle_block_drop",
        [
            (
                """        use_eagle = bool(
            spec_cfg.use_eagle()
            if spec_cfg is not None and callable(getattr(spec_cfg, "use_eagle", None))
            else False
        )
""",
                """        # R8 (vllm#53388), verbatim. Inert here: no Mooncake connector.
        use_eagle_block_drop = bool(
            spec_cfg.use_eagle_block_drop()
            if spec_cfg is not None
            and callable(getattr(spec_cfg, "use_eagle_block_drop", None))
            else False
        )
""",
            ),
            (
                """            hash_block_size=self.hash_block_size,
            use_eagle=use_eagle,
            retention_interval=envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL,
""",
                """            hash_block_size=self.hash_block_size,
            use_eagle=use_eagle_block_drop,
            retention_interval=envs.VLLM_PREFIX_CACHE_RETENTION_INTERVAL,
""",
            ),
        ],
    ),
]


def main() -> None:
    applied, skipped = [], []
    for path, sentinel, pairs in EDITS:
        src = open(path).read()
        name = path.rsplit("/", 1)[-1]
        if sentinel in src:
            skipped.append(name)
            continue
        for old, new in pairs:
            n = src.count(old)
            assert n == 1, "%s: anchor found %d times (want 1) --\n%s" % (name, n, old)
            src = src.replace(old, new, 1)
        open(path, "w").write(src)
        ast.parse(open(path).read())
        applied.append(name)
    if skipped:
        print(
            "patch_block_drop.py: already applied to %s" % ", ".join(skipped),
            file=sys.stderr,
        )
    if applied:
        print(
            "patch_block_drop.py applied OK to %s" % ", ".join(applied),
            file=sys.stderr,
        )


main()
