# SPDX-License-Identifier: Apache-2.0
"""draft_vocab_common - pure-python helpers for the reduced draft vocabulary.

Standard library only: no torch, no numpy, no safetensors. That is deliberate,
so the same code runs

  * inside the vLLM image, imported by ``vllm_mtp_draft_vocab`` at server start,
  * inside a CPU-only helper container, imported by ``tools/build_draft_vocab.py``,
  * on the workstation, imported by ``src/test_draft_vocab_cpu.py``.

Nothing here touches a GPU or a checkpoint. The numeric semantics that the
runtime patch must preserve (the -inf scatter, and the fact that no sampler can
select an out-of-slice id) are expressed here as reference implementations so
they can be unit-tested without torch.
"""

from __future__ import annotations

import math

# Marlin's GEMM/repack requires (n % 64, k % 128) or (n % 128, k % 64); the
# smaller family is chosen by marlin_padded_nk(). vLLM additionally pads a
# ParallelLMHead's vocabulary to DEFAULT_VOCAB_PADDING_SIZE (64). Keeping K a
# multiple of 64 means neither pads, so the sliced head has exactly K columns
# and the GEMM shape family matches the full head's.
SLICE_ALIGN = 64

# A slice smaller than this is almost certainly a mistake (a truncated id file,
# a build that ran on an empty corpus). Refuse it rather than serve a drafter
# that can only ever propose a few thousand tokens.
MIN_SLICE = 4096

NEG_INF = float("-inf")


class DraftVocabError(ValueError):
    """Raised for any malformed id list or unusable slice size."""


# ---------------------------------------------------------------- id lists


def parse_id_list(text: str, vocab_size: int) -> list[int]:
    """Parse an id-list file into a sorted, de-duplicated list of token ids.

    Format: one non-negative integer per line. Blank lines are skipped, and
    everything from a ``#`` to the end of a line is a comment. A trailing
    comment on a data line is allowed, so the builder can annotate ids.

    Raises DraftVocabError on any non-integer token, any id outside
    [0, vocab_size), or an empty list.
    """
    if vocab_size <= 0:
        raise DraftVocabError(f"vocab_size must be positive, got {vocab_size}")
    ids: set[int] = set()
    n_lines = 0
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        n_lines += 1
        try:
            value = int(line, 10)
        except ValueError:
            raise DraftVocabError(
                f"line {lineno}: expected one integer token id, got {raw.strip()!r}"
            ) from None
        if value < 0 or value >= vocab_size:
            raise DraftVocabError(
                f"line {lineno}: token id {value} out of range [0, {vocab_size})"
            )
        ids.add(value)
    if not ids:
        raise DraftVocabError("id list is empty")
    return sorted(ids)


def validate_slice_size(k: int, vocab_size: int, align: int = SLICE_ALIGN) -> None:
    """Refuse slice sizes that would change the Marlin GEMM shape family."""
    if k > vocab_size:
        raise DraftVocabError(f"slice size {k} exceeds vocab size {vocab_size}")
    if k < MIN_SLICE:
        raise DraftVocabError(
            f"slice size {k} is below the {MIN_SLICE}-id sanity floor; "
            "the id list is probably truncated"
        )
    if k % align:
        raise DraftVocabError(
            f"slice size {k} is not a multiple of {align}; a sliced head whose "
            "column count is not tile-aligned would be zero-padded by Marlin "
            "and the saved bytes would come back as padding"
        )


def format_id_list(ids: list[int], header: list[str] | None = None) -> str:
    """Render an id list back to the on-disk format, with optional comments."""
    out = []
    for line in header or []:
        out.append("# " + line if line else "#")
    out.extend(str(i) for i in sorted(set(ids)))
    return "\n".join(out) + "\n"


# ---------------------------------------------- reference numeric semantics
#
# These mirror what src/vllm_mtp_draft_vocab.py does on the GPU. They exist so
# the properties the design depends on can be asserted without torch.


def scatter_reference(
    slice_logits: list[float], ids: list[int], vocab_size: int
) -> list[float]:
    """Scatter K slice logits into a full-width row, -inf everywhere else.

    This is the tonyd2wild / vLLM-d2t shape: the drafter emits a full-width
    logits row so that every downstream consumer (argmax, gumbel sampling, the
    block rejection sampler's draft_probs) keeps working unchanged, while every
    id outside the slice is unreachable.
    """
    if len(slice_logits) != len(ids):
        raise DraftVocabError(
            f"slice_logits has {len(slice_logits)} entries but the id list has "
            f"{len(ids)}"
        )
    row = [NEG_INF] * vocab_size
    for value, token_id in zip(slice_logits, ids):
        if token_id < 0 or token_id >= vocab_size:
            raise DraftVocabError(f"token id {token_id} out of range")
        row[token_id] = value
    return row


def softmax_reference(row: list[float], temperature: float = 1.0) -> list[float]:
    """Softmax with -inf handling, matching logits.div_(T).softmax(-1).

    Entries that are -inf map to exactly 0.0, which is what makes a restricted
    proposal distribution q valid for rejection sampling: q is zero outside the
    slice, so p/q is only ever evaluated where q > 0.
    """
    if temperature <= 0.0:
        raise DraftVocabError("temperature must be positive")
    scaled = [v / temperature for v in row]
    finite = [v for v in scaled if v != NEG_INF]
    if not finite:
        raise DraftVocabError("every logit is -inf")
    top = max(finite)
    exps = [0.0 if v == NEG_INF else math.exp(v - top) for v in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def argmax_reference(row: list[float]) -> int:
    best_i, best_v = 0, row[0]
    for i in range(1, len(row)):
        if row[i] > best_v:
            best_i, best_v = i, row[i]
    return best_i


def gumbel_argmax_reference(probs: list[float], noise: list[float]) -> int:
    """argmax(probs / exponential_noise), the shape vLLM's draft sampler uses.

    ``compute_probs_and_sample_next_token`` draws q ~ Exponential(1) and takes
    argmax(probs / q). A zero probability yields a zero ratio, so an id whose
    probability is exactly 0 can never win as long as at least one id has a
    strictly positive probability.
    """
    if len(probs) != len(noise):
        raise DraftVocabError("probs and noise must have the same length")
    best_i, best_v = -1, -1.0
    for i, (p, q) in enumerate(zip(probs, noise)):
        if q <= 0.0:
            raise DraftVocabError("exponential noise must be positive")
        ratio = p / q
        if ratio > best_v:
            best_i, best_v = i, ratio
    return best_i


# ------------------------------------------------------- chunked reading


def plan_column_chunks(
    ids: list[int], vocab_size: int, chunk: int
) -> list[tuple[int, int, int, list[int]]]:
    """Plan a column gather as a sequence of contiguous source-column reads.

    Used by the checkpoint fallback in the runtime patch: reading
    ``lm_head.qweight`` whole is 606 MiB, but safetensors can slice contiguous
    column ranges, so the same gather can be done in ``chunk``-wide bites.

    Returns [(lo, hi, dest_offset, local_cols)], where local_cols are the wanted
    columns of source block [lo, hi) expressed relative to lo, and dest_offset is
    where they land in the K-wide destination. ``ids`` must be sorted and unique.
    """
    if chunk <= 0:
        raise DraftVocabError("chunk must be positive")
    if any(b <= a for a, b in zip(ids, ids[1:])):
        raise DraftVocabError("ids must be sorted and unique")
    if ids and (ids[0] < 0 or ids[-1] >= vocab_size):
        raise DraftVocabError("ids out of range")
    plan: list[tuple[int, int, int, list[int]]] = []
    pos = 0
    n = len(ids)
    for lo in range(0, vocab_size, chunk):
        hi = min(lo + chunk, vocab_size)
        start = pos
        local: list[int] = []
        while pos < n and ids[pos] < hi:
            local.append(ids[pos] - lo)
            pos += 1
        if local:
            plan.append((lo, hi, start, local))
    if pos != n:
        raise DraftVocabError("internal error: not every id was placed")
    return plan


# --------------------------------------------------------------- coverage


def rank_ids(counts: dict[int, float]) -> list[int]:
    """Rank token ids by weighted count, breaking ties by ascending id.

    A deterministic tie-break matters: many ids in the tail occur exactly once,
    and an unstable order would make two builds of the same corpus produce
    different K=32768 slices.
    """
    return sorted(counts, key=lambda i: (-counts[i], i))


def select_top_k(
    counts: dict[int, float],
    k: int,
    force_include: list[int] | None = None,
    vocab_size: int | None = None,
) -> tuple[list[int], int]:
    """Choose exactly k ids: forced ids first, then by weighted frequency.

    Returns (ids, n_padding). If the corpus produced fewer than k distinct ids,
    the remainder is padded with the lowest unused ids - Qwen's BPE-merge order,
    which the prior research measured at 80.9 % coverage at K=32768 on its own,
    so it is a reasonable filler but a bad primary signal. n_padding says how
    many slots came from that filler; a large value means the corpus was too
    small for this K.
    """
    if vocab_size is None:
        vocab_size = max(max(counts, default=0), max(force_include or [0])) + 1
    chosen: list[int] = []
    seen: set[int] = set()
    for token_id in sorted(set(force_include or [])):
        if token_id < 0 or token_id >= vocab_size:
            raise DraftVocabError(f"forced token id {token_id} out of range")
        if token_id not in seen:
            seen.add(token_id)
            chosen.append(token_id)
    if len(chosen) > k:
        raise DraftVocabError(
            f"{len(chosen)} forced ids do not fit in a slice of {k}"
        )
    for token_id in rank_ids(counts):
        if len(chosen) >= k:
            break
        if token_id not in seen:
            seen.add(token_id)
            chosen.append(token_id)
    n_padding = 0
    filler = 0
    while len(chosen) < k:
        if filler >= vocab_size:
            raise DraftVocabError("vocabulary exhausted before reaching k")
        if filler not in seen:
            seen.add(filler)
            chosen.append(filler)
            n_padding += 1
        filler += 1
    return sorted(chosen), n_padding


def coverage(selected: list[int], heldout_counts: dict[int, float]) -> float:
    """Fraction of held-out token OCCURRENCES covered by the slice.

    Occurrences, not distinct ids: the metric that predicts acceptance is how
    often the next token the drafter wants to propose is reachable at all.
    """
    total = sum(heldout_counts.values())
    if total <= 0:
        raise DraftVocabError("held-out split is empty")
    chosen = set(selected)
    hit = sum(c for i, c in heldout_counts.items() if i in chosen)
    return hit / total
