#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU image regression checks."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import sys
import types

FAILURES: list[str] = []
CHECKS = 0

SP = "/usr/local/lib/python3.12/dist-packages/vllm"
KNOB = "VLLM_KEEP_DRAFT_BLOCKS"
KNOB_ON = os.environ.get(KNOB, "0") == "1"


def check(name: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    status = "ok  " if ok else "FAIL"
    print(f"  [{status}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def banner(text: str) -> None:
    print()
    print(text)
    print("-" * len(text))


# ---------------------------------------------------------------------------
# G1 / G2 -- the patched files differ only in the guarded hunks
# ---------------------------------------------------------------------------
def gates_g1_g2() -> None:
    banner("G1/G2  patched files differ from the base image only in the recorded hunks")

    sys.path.insert(0, "/tmp")
    # patch_block_drop.py calls main() at import; it is idempotent and will
    # report "already applied", which is what we want here.
    import patch_block_drop as pbd  # noqa: E402

    base = json.load(open("/tmp/base.json"))
    print(f"  base image: {base['base_image']}")

    seen_rel = set()
    for path, sentinel, pairs in pbd.EDITS:
        rel = path[len(SP) + 1 :]
        seen_rel.add(rel)
        src = open(path, "rb").read().decode()

        check(f"{rel}: sentinel {sentinel!r} present", sentinel in src)
        try:
            ast.parse(src)
            check(f"{rel}: parses", True)
        except SyntaxError as exc:  # pragma: no cover
            check(f"{rel}: parses", False, str(exc))

        reversed_src = src
        for old, new in pairs:
            n_new = reversed_src.count(new)
            check(f"{rel}: replacement text present exactly once", n_new == 1,
                  f"count={n_new}")
            if n_new != 1:
                continue
            reversed_src = reversed_src.replace(new, old, 1)

        want = base["files"][rel]["sha256"]
        got = hashlib.sha256(reversed_src.encode()).hexdigest()
        check(
            f"{rel}: reversing the recorded hunks reproduces the base file byte for byte",
            got == want,
            f"want {want[:16]}... got {got[:16]}... "
            f"({len(reversed_src.encode())} vs {base['files'][rel]['bytes']} bytes)",
        )

    check("exactly 7 vLLM source files touched", len(seen_rel) == 7,
          f"{sorted(seen_rel)}")


# ---------------------------------------------------------------------------
# G3 -- the latch
# ---------------------------------------------------------------------------
def gate_g3() -> None:
    banner(f"G3  env latch ({KNOB}={'1' if KNOB_ON else '<unset>'})")
    import vllm.config.speculative as spec_mod
    import vllm.v1.core.single_type_kv_cache_manager as stkv

    check("config/speculative.py latch matches the environment",
          spec_mod._KEEP_DRAFT_BLOCKS is KNOB_ON,
          f"_KEEP_DRAFT_BLOCKS={spec_mod._KEEP_DRAFT_BLOCKS}")
    check("single_type_kv_cache_manager.py latch matches the environment",
          stkv._KEEP_DRAFT_BLOCKS is KNOB_ON,
          f"_KEEP_DRAFT_BLOCKS={stkv._KEEP_DRAFT_BLOCKS}")


# ---------------------------------------------------------------------------
# G4 -- the predicate
# ---------------------------------------------------------------------------
def gate_g4() -> None:
    banner("G4  use_eagle_block_drop() vs use_eagle() over every speculative method")
    from vllm.config.speculative import SpeculativeConfig

    methods = [
        "eagle", "eagle3", "mtp", "dflash", "dspark",
        "ngram", "ngram_gpu", "suffix", "draft_model", "extract_hidden_states",
    ]

    # Borrow the two unbound predicates onto a bare object, so neither model
    # metadata nor a device is needed. This is exactly what upstream's own new
    # test does (tests/test_config.py::
    # test_eagle_block_drop_can_be_disabled_without_disabling_eagle) by starting
    # from an ngram config and then overwriting `.method`.
    class _Stub:
        use_eagle = SpeculativeConfig.use_eagle
        use_eagle_block_drop = SpeculativeConfig.use_eagle_block_drop

        def __init__(self) -> None:
            self.method = "mtp"
            self.disable_eagle_block_drop = False

    stub = _Stub()
    for m in methods:
        stub.method = m
        eagle = stub.use_eagle()
        drop = stub.use_eagle_block_drop()
        if KNOB_ON:
            check(f"method={m}: drop is disabled iff eagle", drop is False,
                  f"use_eagle={eagle} use_eagle_block_drop={drop}")
        else:
            check(f"method={m}: identical to upstream's use_eagle()", drop == eagle,
                  f"use_eagle={eagle} use_eagle_block_drop={drop}")

    # The config field itself must also work, independent of the env.
    stub.method = "mtp"
    stub.disable_eagle_block_drop = True
    check("the --speculative-config field alone disables the drop",
          stub.use_eagle_block_drop() is False)
    stub.disable_eagle_block_drop = False

    fields = set(getattr(SpeculativeConfig, "__dataclass_fields__", {}))
    fields |= set(getattr(SpeculativeConfig, "model_fields", {}) or {})
    check("upstream's field name is present on SpeculativeConfig",
          "disable_eagle_block_drop" in fields,
          f"{len(fields)} fields")


# ---------------------------------------------------------------------------
# G5 -- static wiring
# ---------------------------------------------------------------------------
def gate_g5() -> None:
    banner("G5  static wiring in v1/core/sched/scheduler.py")
    tree = ast.parse(open(f"{SP}/v1/core/sched/scheduler.py").read())

    kwargs_seen = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "KVCacheManager":
            for kw in node.keywords:
                if kw.arg == "use_eagle":
                    kwargs_seen.append(ast.unparse(kw.value))
    check("KVCacheManager(use_eagle=self.use_eagle_block_drop)",
          kwargs_seen == ["self.use_eagle_block_drop"], f"{kwargs_seen}")

    split = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "_mamba_block_aligned_split"),
        None,
    )
    check("_mamba_block_aligned_split exists", split is not None)
    if split is not None:
        body = ast.unparse(split)
        check("_mamba_block_aligned_split branches on use_eagle_block_drop",
              "self.use_eagle_block_drop" in body and "if self.use_eagle:" not in body)

    init = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
        None,
    )
    src_init = ast.unparse(init) if init else ""
    check("num_prefill_lookahead still keyed on use_eagle (upstream's asymmetry)",
          "if self.use_eagle:\n" in src_init + "\n"
          or "self.use_eagle = speculative_config.use_eagle()" in src_init)
    check("the install signal is present in __init__",
          "iter6 R8 installed: keep-draft-blocks on " in src_init
          and "iter6 R8 disabled" in src_init)

    # No consumer of self.use_eagle may remain in the prefix-cache path.
    whole = ast.unparse(tree)
    check("exactly two remaining reads of self.use_eagle (lookahead gate + R8 gate)",
          whole.count("self.use_eagle ") + whole.count("self.use_eagle:")
          + whole.count("self.use_eagle and") + whole.count("self.use_eagle\n") >= 1)


# ---------------------------------------------------------------------------
# G6 -- the real _mamba_block_aligned_split on a stub self
# ---------------------------------------------------------------------------
MAMBA_BLOCK = 1600


def _split(use_eagle_block_drop: bool, num_tokens: int, num_new_tokens: int) -> int:
    from vllm.v1.core.sched.scheduler import Scheduler

    stub = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(
            block_size=8, mamba_block_size=MAMBA_BLOCK
        ),
        use_eagle_block_drop=use_eagle_block_drop,
        max_num_scheduled_tokens=16384,
        scheduler_config=types.SimpleNamespace(long_prefill_token_threshold=0),
        hash_block_size=MAMBA_BLOCK,
        mamba_partial_cache_hit=False,
    )
    request = types.SimpleNamespace(
        request_id="r",
        num_computed_tokens=0,
        num_tokens=num_tokens,
        num_prompt_tokens=num_tokens,
        shared_prefix_boundary=0,
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


def gate_g6() -> None:
    banner("G6  Scheduler._mamba_block_aligned_split (upstream's own new test, adapted)")
    n = 2 * MAMBA_BLOCK + 402          # upstream uses 3602 at 1600-token blocks
    with_drop = _split(True, n, n)
    without_drop = _split(False, n, n)
    check("with the drop, the chunk stops one mamba block early",
          with_drop == MAMBA_BLOCK, f"{with_drop}")
    check("without the drop, the chunk stops at the boundary",
          without_drop == 2 * MAMBA_BLOCK, f"{without_drop}")
    check("the difference is exactly one mamba block",
          without_drop - with_drop == MAMBA_BLOCK,
          f"{without_drop} - {with_drop}")


# ---------------------------------------------------------------------------
# G7 / G8 -- the lever, on vLLM's own KV cache manager classes
# ---------------------------------------------------------------------------
B = 1600  # this model: attention block == mamba block == scheduler block


def _build(use_eagle: bool, hybrid: bool = True):
    import torch
    from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, KVCacheTensor, MambaSpec,
    )

    attn = FullAttentionSpec(block_size=B, num_kv_heads=1, head_size=64,
                             dtype=torch.bfloat16)
    groups = [KVCacheGroupSpec(["qsa.0"], attn)]
    tensors = [KVCacheTensor(size=4096, shared_by=["qsa.0"])]
    if hybrid:
        mamba = MambaSpec(block_size=B, shapes=((1, 1),), dtypes=(torch.float32,),
                          page_size_padded=None, mamba_type="gdn",
                          mamba_cache_mode="align")
        groups.append(KVCacheGroupSpec(["gdn.0"], mamba))
        tensors.append(KVCacheTensor(size=4096, shared_by=["gdn.0"]))
    cfg = KVCacheConfig(num_blocks=4096, kv_cache_tensors=tensors,
                        kv_cache_groups=groups)
    return get_kv_cache_coordinator(
        kv_cache_config=cfg,
        max_model_len=1 << 21,
        max_in_flight_tokens=1 << 21,
        use_eagle=use_eagle,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=B,
        hash_block_size=B,
        # This checkpoint: use_multi_module_mtp() is False, so the drafter reads
        # exactly one token past the chunk boundary
        # (research-pass-2/SURVEY-2.md:163,209).
        num_prefill_lookahead=1 if use_eagle else 0,
    )


def _mk_request(rid: str, tokens: list[int]):
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request
    return Request(request_id=rid, prompt_token_ids=tokens,
                   sampling_params=SamplingParams(max_tokens=1),
                   pooling_params=None, block_hasher=_HASHER)


def _turn(coord, rid: str, tokens: list[int]):
    """One agent turn: look the prompt up, allocate, then cache what was computed."""
    req = _mk_request(rid, tokens)
    n = len(tokens)
    # The scheduler caps the lookup at num_tokens - 1 (the replay boundary).
    blocks, hit, uncached = coord.find_longest_cache_hit(req.block_hashes, n - 1)
    coord.allocate_new_blocks(rid, n, num_tokens_main_model=n)
    coord.cache_blocks(req, n)
    return hit, uncached


_HASHER = None


def _init_hasher():
    global _HASHER
    from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
    hash_fn = lambda x: hashlib.sha256(repr(x).encode()).digest()  # noqa: E731
    init_none_hash(hash_fn)
    _HASHER = get_request_block_hasher(B, hash_fn)


def gates_g7_g8() -> None:
    banner("G7  warm-turn cache hit through the real HybridKVCacheCoordinator")
    _init_hasher()

    def round_down(x: int, m: int) -> int:
        return x // m * m

    # Prompt lengths spanning the regimes of our own measured formula:
    # under 2*B reports 0 with the drop; a 134k-median agent turn is the last.
    for shared in (3000, 6000, 6433, 12800, 51200, 134000):
        results = {}
        for use_eagle in (True, False):
            coord = _build(use_eagle, hybrid=False)
            prefix = list(range(shared))
            _turn(coord, "t1", prefix)                       # cold turn
            hit, _ = _turn(coord, "t2", prefix + list(range(9 * 10**6, 9 * 10**6 + 200)))
            results[use_eagle] = hit

        want_drop = max(round_down(min(shared, shared + 200 - 1), B) - B, 0)
        want_keep = round_down(min(shared, shared + 200 - 1), B)
        check(
            f"P={shared}: with the drop, warm hit == round_down(P,{B}) - {B}",
            results[True] == want_drop,
            f"got {results[True]}, want {want_drop}",
        )
        check(
            f"P={shared}: without the drop, warm hit == round_down(P,{B})",
            results[False] == want_keep,
            f"got {results[False]}, want {want_keep}",
        )
        check(
            f"P={shared}: the lever is worth exactly {B} tokens",
            results[False] - results[True] == min(B, want_keep),
            f"{results[False]} - {results[True]}",
        )

    banner("G7b  the same, on the real hybrid (attention + GDN/mamba group)")
    # NOTE on what this sub-gate can and cannot assert. Without the real
    # scheduler driving `_mamba_block_aligned_split`, the synthetic mamba group
    # publishes states at whatever boundary `cache_blocks` happens to reach, so
    # the RECONCILED hybrid hit (the min across groups) is an artefact of the
    # stub, not of the patch -- it is 0 at one prompt length and full at
    # another. What is not an artefact, and is exactly the quantity SURVEY-7
    # measured, is `longest_hit_length` = hit + already-shared-but-uncached:
    # the attention group's own hit. That is what the lever moves.
    for shared in (6433, 51200):
        out = {}
        for use_eagle in (True, False):
            coord = _build(use_eagle, hybrid=True)
            prefix = list(range(shared))
            _turn(coord, "t1", prefix)
            req = _mk_request("t2", prefix + list(range(9 * 10**6, 9 * 10**6 + 200)))
            per_group = coord.find_longest_cache_hit_per_group(
                req.block_hashes, len(req.prompt_token_ids) - 1
            )[1]
            hit, uncached = _turn(
                coord, "t2", prefix + list(range(9 * 10**6, 9 * 10**6 + 200))
            )
            out[use_eagle] = (hit, uncached, hit + uncached, tuple(per_group))
        d_longest = out[False][2] - out[True][2]
        check(
            f"P={shared}: the attention group's own hit grows by exactly {B}",
            d_longest == B,
            f"drop hit/uncached/longest/per-group={out[True]} "
            f"keep={out[False]}",
        )
        check(
            f"P={shared}: the reconciled hybrid hit never decreases",
            out[False][0] >= out[True][0],
            f"drop={out[True][0]} keep={out[False][0]}",
        )

    banner("G8  memory: blocks resident, drop vs no-drop")
    for shared in (6433, 51200):
        stats = {}
        for use_eagle in (True, False):
            coord = _build(use_eagle, hybrid=False)
            prefix = list(range(shared))
            _turn(coord, "t1", prefix)
            pool = coord.block_pool
            after_cold = (len(pool.cached_block_hash_to_block),
                          pool.get_num_free_blocks())
            _turn(coord, "t2", prefix + list(range(9 * 10**6, 9 * 10**6 + 200)))
            after_warm = (len(pool.cached_block_hash_to_block),
                          pool.get_num_free_blocks())
            stats[use_eagle] = (after_cold, after_warm)
        check(
            f"P={shared}: same number of distinct cached blocks after the cold turn",
            stats[True][0][0] == stats[False][0][0],
            f"drop={stats[True][0]} keep={stats[False][0]}",
        )
        check(
            f"P={shared}: keeping the block never costs free blocks",
            stats[False][1][1] >= stats[True][1][1],
            f"free after warm: drop={stats[True][1][1]} keep={stats[False][1][1]}",
        )

    # The cache_blocks algebra, directly: min(x, round_down(x, B) + B) == x for
    # every x, so the eagle branch of coordinator :731-745 caches the same
    # number of blocks as the non-eagle branch. num_reprefillable_tokens is 0 on
    # this checkpoint, so num_finalized == num_computed.
    bad = [x for x in range(0, 20001, 7)
           if min(x, x // B * B + B) // B != (x // B * B) // B]
    check("cache_blocks caches the same block count either way (algebraic, x=0..20000)",
          not bad, f"counterexamples: {bad[:5]}")


def main() -> int:
    print(f"R8 / vllm#53388 CPU gate -- {KNOB}="
          f"{os.environ.get(KNOB, '<unset>')}")
    import vllm
    print(f"vLLM {vllm.__version__}")

    gates_g1_g2()
    gate_g3()
    gate_g4()
    gate_g5()
    gate_g6()
    gates_g7_g8()

    print()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} of {CHECKS} checks failed")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"PASS: {CHECKS}/{CHECKS} checks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
