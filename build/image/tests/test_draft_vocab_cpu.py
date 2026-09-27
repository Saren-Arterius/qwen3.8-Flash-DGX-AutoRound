#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Host-side unit tests for the reduced draft vocabulary.

Standard library only - no torch, no numpy, no GPU. Runs on the workstation:

    python3 src/test_draft_vocab_cpu.py

What it proves, on the reference implementations in ``draft_vocab_common``
(which the GPU patch mirrors one-to-one):

  (a) with the full id list the scatter is the identity, so the emitted row
      equals the original row element for element;
  (b) with a subset, in-slice entries equal the original and every out-of-slice
      entry is exactly -inf;
  (c) argmax and the probabilistic draft sampler (softmax after temperature
      division, then argmax of probs / Exponential(1) noise - the shape of
      vllm/v1/spec_decode/llm_base_proposer.py:1859-1897) never select an
      out-of-slice id, at any temperature, over randomised trials;

plus the id-list parser, the slice-size guard and the corpus-builder selection
and coverage helpers.

What it does NOT prove: that slicing the GPTQ columns before Marlin's repack
reproduces the full head's logits. That needs the real kernel and lives in
src/test_draft_vocab_gpu.py, which runs inside the image.
"""

from __future__ import annotations

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from draft_vocab_common import (  # noqa: E402
    NEG_INF,
    DraftVocabError,
    argmax_reference,
    coverage,
    format_id_list,
    gumbel_argmax_reference,
    parse_id_list,
    plan_column_chunks,
    rank_ids,
    scatter_reference,
    select_top_k,
    softmax_reference,
    validate_slice_size,
)

FAILURES: list[str] = []
CHECKS = 0


def check(cond, what):
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(what)


def expect_error(fn, what):
    global CHECKS
    CHECKS += 1
    try:
        fn()
    except DraftVocabError:
        return
    except Exception as exc:  # noqa: BLE001
        FAILURES.append(f"{what}: raised {type(exc).__name__} not DraftVocabError")
        return
    FAILURES.append(f"{what}: no error raised")


# ------------------------------------------------------------ id list I/O


def test_parse_id_list():
    text = "# header\n\n5\n3\n3\n  7  # trailing comment\n\n0\n"
    check(parse_id_list(text, 10) == [0, 3, 5, 7], "parse: sort + dedupe")
    check(parse_id_list("1\n", 2) == [1], "parse: single id")
    expect_error(lambda: parse_id_list("", 10), "parse: empty list")
    expect_error(lambda: parse_id_list("# only a comment\n", 10), "parse: comment only")
    expect_error(lambda: parse_id_list("10\n", 10), "parse: id == vocab_size")
    expect_error(lambda: parse_id_list("-1\n", 10), "parse: negative id")
    expect_error(lambda: parse_id_list("3.5\n", 10), "parse: non-integer")
    expect_error(lambda: parse_id_list("0x10\n", 32), "parse: hex literal")
    expect_error(lambda: parse_id_list("1\n", 0), "parse: zero vocab")

    ids = [9, 1, 5, 1]
    rendered = format_id_list(ids, header=["built by a test", ""])
    check(rendered.startswith("# built by a test\n#\n"), "format: header comments")
    check(parse_id_list(rendered, 10) == [1, 5, 9], "format: round-trips")


def test_validate_slice_size():
    validate_slice_size(32768, 248320)
    validate_slice_size(65536, 248320)
    validate_slice_size(248320, 248320)
    expect_error(lambda: validate_slice_size(32769, 248320), "slice: not 64-aligned")
    expect_error(lambda: validate_slice_size(1024, 248320), "slice: below floor")
    expect_error(lambda: validate_slice_size(248384, 248320), "slice: bigger than vocab")


# --------------------------------------------------------- scatter (a)(b)


def test_scatter_identity_is_bit_identical():
    """(a) full id list -> the emitted row equals the original row exactly."""
    rng = random.Random(20260909)
    vocab = 512
    ids = list(range(vocab))
    for _ in range(200):
        row = [rng.uniform(-30.0, 30.0) for _ in range(vocab)]
        out = scatter_reference(row, ids, vocab)
        check(out == row, "scatter: full id list is the identity")
        check(all(v != NEG_INF for v in out), "scatter: full list leaves no -inf")


def test_scatter_subset():
    """(b) subset -> in-slice equals the original, out-of-slice is exactly -inf."""
    rng = random.Random(4242)
    vocab = 1024
    for _ in range(200):
        k = rng.choice([64, 128, 256, 512])
        ids = sorted(rng.sample(range(vocab), k))
        full_row = [rng.uniform(-30.0, 30.0) for _ in range(vocab)]
        slice_row = [full_row[i] for i in ids]
        out = scatter_reference(slice_row, ids, vocab)
        chosen = set(ids)
        ok_in = all(out[i] == full_row[i] for i in ids)
        ok_out = all(out[i] == NEG_INF for i in range(vocab) if i not in chosen)
        check(ok_in, "scatter: in-slice logits equal the original")
        check(ok_out, "scatter: out-of-slice logits are exactly -inf")
        check(
            sum(1 for v in out if v != NEG_INF) == k,
            "scatter: exactly k finite entries",
        )
    expect_error(
        lambda: scatter_reference([1.0, 2.0], [0], 8), "scatter: length mismatch"
    )
    expect_error(
        lambda: scatter_reference([1.0], [99], 8), "scatter: id out of range"
    )


# ------------------------------------------------------------ sampling (c)


def test_argmax_never_leaves_the_slice():
    rng = random.Random(7)
    vocab = 2048
    for _ in range(300):
        k = rng.choice([64, 256, 1024])
        ids = sorted(rng.sample(range(vocab), k))
        chosen = set(ids)
        slice_row = [rng.uniform(-40.0, 40.0) for _ in range(k)]
        out = scatter_reference(slice_row, ids, vocab)
        check(argmax_reference(out) in chosen, "argmax: stays inside the slice")


def test_probabilistic_sampling_never_leaves_the_slice():
    """The exact shape of compute_probs_and_sample_next_token.

    logits.div_(temperature) then softmax then argmax(probs / Exponential(1)).
    Out-of-slice logits are -inf, so their probability is exactly 0.0 and their
    ratio is exactly 0.0, which can never be the strict maximum while at least
    one in-slice probability is positive.
    """
    rng = random.Random(1234567)
    vocab = 1024
    n_bad_prob = 0
    n_bad_pick = 0
    for _ in range(120):
        k = rng.choice([64, 128, 512])
        ids = sorted(rng.sample(range(vocab), k))
        chosen = set(ids)
        # A deliberately harsh row: one huge in-slice logit and a long tail of
        # very negative ones, so the softmax underflows almost everywhere.
        slice_row = [rng.uniform(-60.0, -40.0) for _ in range(k)]
        slice_row[rng.randrange(k)] = rng.uniform(20.0, 40.0)
        row = scatter_reference(slice_row, ids, vocab)
        for temperature in (0.1, 0.6, 1.0, 2.0):
            probs = softmax_reference(row, temperature)
            if any(probs[i] != 0.0 for i in range(vocab) if i not in chosen):
                n_bad_prob += 1
            if abs(sum(probs) - 1.0) > 1e-9:
                n_bad_prob += 1
            for _trial in range(8):
                noise = [rng.expovariate(1.0) for _ in range(vocab)]
                picked = gumbel_argmax_reference(probs, noise)
                if picked not in chosen:
                    n_bad_pick += 1
    check(n_bad_prob == 0, f"sampling: out-of-slice probability is 0 ({n_bad_prob} bad)")
    check(n_bad_pick == 0, f"sampling: sampled id stays in the slice ({n_bad_pick} bad)")
    expect_error(lambda: softmax_reference([NEG_INF] * 4), "softmax: all -inf")
    expect_error(lambda: softmax_reference([1.0], 0.0), "softmax: zero temperature")


# ------------------------------------------------------- builder helpers


def test_rank_and_select():
    counts = {5: 10.0, 2: 10.0, 9: 3.0, 1: 1.0}
    check(rank_ids(counts) == [2, 5, 9, 1], "rank: frequency then ascending id")

    ids, pad = select_top_k(counts, k=4, vocab_size=32)
    check(ids == [1, 2, 5, 9] and pad == 0, "select: exactly the observed ids")

    ids, pad = select_top_k(counts, k=6, vocab_size=32)
    check(len(ids) == 6 and pad == 2, "select: pads to k with lowest unused ids")
    check(set([2, 5, 9, 1]).issubset(ids), "select: keeps every observed id")

    ids, pad = select_top_k(counts, k=3, force_include=[30, 31], vocab_size=32)
    check(ids == [2, 30, 31] and pad == 0, "select: forced ids win over frequency")
    expect_error(
        lambda: select_top_k(counts, k=1, force_include=[30, 31], vocab_size=32),
        "select: forced ids do not fit",
    )
    expect_error(
        lambda: select_top_k(counts, k=2, force_include=[99], vocab_size=32),
        "select: forced id out of range",
    )
    expect_error(
        lambda: select_top_k({0: 1.0}, k=8, vocab_size=4), "select: vocab exhausted"
    )


def test_plan_column_chunks():
    """The checkpoint fallback's gather plan must reconstruct the slice exactly."""
    rng = random.Random(99)
    for _ in range(150):
        vocab = rng.choice([256, 1000, 4096])
        k = rng.randrange(1, vocab + 1)
        ids = sorted(rng.sample(range(vocab), k))
        chunk = rng.choice([1, 7, 64, 512, vocab, vocab * 2])
        plan = plan_column_chunks(ids, vocab, chunk)
        # Replay the plan against a source row and compare to a direct gather.
        source = [i * 3 + 1 for i in range(vocab)]
        dest = [None] * k
        for lo, hi, offset, local in plan:
            check(0 <= lo < hi <= vocab, "plan: block inside the vocabulary")
            check(hi - lo <= chunk, "plan: block no wider than chunk")
            block = source[lo:hi]
            for j, col in enumerate(local):
                check(0 <= col < hi - lo, "plan: local column inside the block")
                dest[offset + j] = block[col]
        check(dest == [source[i] for i in ids], "plan: replay equals a direct gather")
        check(
            sum(len(local) for _, _, _, local in plan) == k,
            "plan: every id is placed exactly once",
        )
    check(plan_column_chunks([], 64, 8) == [], "plan: empty id list")
    expect_error(lambda: plan_column_chunks([1, 0], 8, 4), "plan: unsorted ids")
    expect_error(lambda: plan_column_chunks([1, 1], 8, 4), "plan: duplicate ids")
    expect_error(lambda: plan_column_chunks([9], 8, 4), "plan: id out of range")
    expect_error(lambda: plan_column_chunks([1], 8, 0), "plan: zero chunk")


def test_coverage():
    heldout = {1: 90.0, 2: 5.0, 7: 5.0}
    check(abs(coverage([1], heldout) - 0.90) < 1e-12, "coverage: single id")
    check(abs(coverage([1, 2, 7], heldout) - 1.0) < 1e-12, "coverage: everything")
    check(abs(coverage([3], heldout) - 0.0) < 1e-12, "coverage: nothing")
    check(
        abs(coverage([1, 2], heldout) - 0.95) < 1e-12,
        "coverage: occurrences not distinct ids",
    )
    expect_error(lambda: coverage([1], {}), "coverage: empty held-out split")


# --------------------------------------------------- patch wiring (no torch)
#
# vllm_mtp_draft_vocab imports only logging and os at module level: torch and
# vLLM are imported lazily inside the build/forward helpers. So the enable gate,
# the hook installation and the fail-closed guards are all testable here.


class _StubLogitsProcessor:
    def __init__(self, **kw):
        self.scale = kw.get("scale", 1.0)
        self.soft_cap = kw.get("soft_cap")
        self.logits_as_input = kw.get("logits_as_input", False)
        self.head_dtype = kw.get("head_dtype")


class _StubHead:
    def __init__(self, tp_size=1, params_dtype="bf16"):
        self.tp_size = tp_size
        self.params_dtype = params_dtype


class _StubMTP:
    def __init__(self):
        self.calls = []

    def load_weights(self, weights):
        self.calls.append(("load_weights", weights))
        return {"lm_head.qweight", "lm_head.scales"}

    def compute_logits(self, hidden_states, spec_step_idx=0):
        self.calls.append(("compute_logits", hidden_states, spec_step_idx))
        return "original-logits"


def _fresh_stub_class():
    return type("StubMTP", (_StubMTP,), {})


def test_apply_is_a_no_op_when_unset():
    import vllm_mtp_draft_vocab as dv

    os.environ.pop(dv.ENV_PATH, None)
    cls = _fresh_stub_class()
    before = (cls.load_weights, cls.compute_logits)
    dv.apply(cls)
    check(
        (cls.load_weights, cls.compute_logits) == before,
        "apply: unset env installs nothing",
    )
    check(
        not getattr(cls, "_mtp_draft_vocab_patched", False),
        "apply: unset env leaves no sentinel",
    )

    os.environ[dv.ENV_PATH] = "   "
    dv.apply(cls)
    check(
        (cls.load_weights, cls.compute_logits) == before,
        "apply: whitespace-only env installs nothing",
    )
    os.environ.pop(dv.ENV_PATH, None)


def test_apply_installs_hooks_and_delegates():
    import vllm_mtp_draft_vocab as dv

    os.environ[dv.ENV_PATH] = "/nonexistent/ids.txt"
    try:
        cls = _fresh_stub_class()
        original_compute = cls.compute_logits
        dv.apply(cls)
        check(cls.compute_logits is not original_compute, "apply: compute_logits wrapped")
        check(getattr(cls, "_mtp_draft_vocab_patched", False), "apply: sentinel set")

        wrapped = cls.compute_logits
        dv.apply(cls)
        check(cls.compute_logits is wrapped, "apply: second call is a no-op")

        obj = cls()
        # No slice built yet -> must delegate to the unpatched implementation.
        check(
            obj.compute_logits("h", 2) == "original-logits",
            "compute_logits: delegates when no slice is built",
        )
        check(
            obj.calls[-1] == ("compute_logits", "h", 2),
            "compute_logits: forwards spec_step_idx",
        )

        # load_weights must call through AND then fail closed, because the stub
        # has no GPTQ lm_head. A silent pass here would mean a server that looks
        # patched and is not.
        global CHECKS
        CHECKS += 1
        try:
            obj.load_weights(["w"])
        except RuntimeError as exc:
            if "lm_head" not in str(exc):
                FAILURES.append(f"load_weights: wrong error {exc}")
        except Exception as exc:  # noqa: BLE001
            FAILURES.append(f"load_weights: {type(exc).__name__} not RuntimeError")
        else:
            FAILURES.append("load_weights: did not fail closed without a GPTQ head")
        check(
            any(c[0] == "load_weights" for c in obj.calls),
            "load_weights: original still ran before the guard fired",
        )
    finally:
        os.environ.pop(dv.ENV_PATH, None)


def test_logits_path_guard():
    import vllm_mtp_draft_vocab as dv

    class _Model:
        def __init__(self, lp, head):
            self.logits_processor = lp
            self._head = head

    ok_head = _StubHead()
    dv._check_logits_path(_Model(_StubLogitsProcessor(), ok_head), ok_head)
    CHECKS_BEFORE = len(FAILURES)
    check(CHECKS_BEFORE == len(FAILURES), "guard: default configuration passes")

    bad_cases = [
        ("scale", _StubLogitsProcessor(scale=0.5), _StubHead()),
        ("soft_cap", _StubLogitsProcessor(soft_cap=30.0), _StubHead()),
        ("logits_as_input", _StubLogitsProcessor(logits_as_input=True), _StubHead()),
        ("head_dtype", _StubLogitsProcessor(head_dtype="fp32"), _StubHead()),
        ("tp_size", _StubLogitsProcessor(), _StubHead(tp_size=2)),
    ]
    for name, lp, head in bad_cases:
        global CHECKS
        CHECKS += 1
        try:
            dv._check_logits_path(_Model(lp, head), head)
        except RuntimeError:
            continue
        FAILURES.append(f"guard: {name} was not rejected")


def main() -> int:
    for fn in (
        test_parse_id_list,
        test_validate_slice_size,
        test_scatter_identity_is_bit_identical,
        test_scatter_subset,
        test_argmax_never_leaves_the_slice,
        test_probabilistic_sampling_never_leaves_the_slice,
        test_rank_and_select,
        test_plan_column_chunks,
        test_coverage,
        test_apply_is_a_no_op_when_unset,
        test_apply_installs_hooks_and_delegates,
        test_logits_path_guard,
    ):
        fn()
    if FAILURES:
        print(f"FAIL: {len(FAILURES)} of {CHECKS} checks failed")
        for line in FAILURES:
            print("  -", line)
        return 1
    print(f"OK: {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
