#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

import ast
import sys

SP = "/usr/local/lib/python3.12/dist-packages/vllm"
SAMPLER_OPS = SP + "/v1/sample/ops/topk_topp_sampler.py"
STATES = SP + "/v1/worker/gpu/sample/states.py"
GPU_SAMPLER = SP + "/v1/worker/gpu/sample/sampler.py"
REJ = SP + "/v1/worker/gpu/spec_decode/rejection_sampler.py"

EDITS = [
    # ---------------------------------------------------------------- 1 ----
    (
        SAMPLER_OPS,
        "_verify_topk_triton",
        [
            (
                """def apply_top_k_top_p(
    logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None
) -> torch.Tensor:
    if p is None and k is None:
        return logits
""",
                '''# --- qwen38-flash-dgx L5a: sort-free truncation on the speculative verify
# --- path (VLLM_VERIFY_TOPK_TRITON=1). Latched at import; unset => the
# --- prefer_triton argument below can never change a branch.
import os as _vtt_os

_verify_topk_triton = _vtt_os.environ.get("VLLM_VERIFY_TOPK_TRITON", "0") == "1"

# Emit exactly one INFO line at install time, so an
# unattended arm is gated on the log rather than assumed to have installed. The
# module-level `logger` is init_logger(__name__) at the head of this file.
if _verify_topk_triton:
    logger.info(
        "iter6 L5a installed: 1 sampler hook on apply_top_k_top_p "
        "(prefer_triton, passed only by RejectionSampler._verify; the fp32 "
        "logits precondition is asserted at the fork before the Triton call)"
    )
else:
    logger.info("iter6 L5a disabled")


def apply_top_k_top_p(
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    prefer_triton: bool = False,
) -> torch.Tensor:
    if p is None and k is None:
        return logits
''',
            ),
            (
                """    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)""",
                """    # The >= 8 gate is a batch-size heuristic for the ordinary decode path. The
    # speculative verify path always presents num_reqs * (num_spec + 1) rows --
    # 4 at one stream -- and pays a full-vocabulary sort for them. prefer_triton
    # is passed only from RejectionSampler._verify, and is honoured only when
    # VLLM_VERIFY_TOPK_TRITON=1, so the ordinary path's heuristic is unchanged.
    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)

    if HAS_TRITON and prefer_triton and _verify_topk_triton:
        # apply_top_k_top_p_triton asserts fp32 (topk_topp_triton.py:882);
        # assert it here so a future caller gets a clear failure, not an
        # assertion from three frames down.
        assert logits.dtype == torch.float32, (
            "VLLM_VERIFY_TOPK_TRITON=1 needs fp32 logits, got %s" % logits.dtype
        )
        return apply_top_k_top_p_triton(logits, k, p)

    # Use pytorch sort implementation for small batch sizes.
    return apply_top_k_top_p_pytorch(logits, k, p)""",
            ),
        ],
    ),
    # ---------------------------------------------------------------- 2 ----
    (
        STATES,
        "prefer_triton",
        [
            (
                """    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> torch.Tensor:
        top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        if top_k is None and top_p is None:
            return logits
        return apply_top_k_top_p(logits, top_k, top_p)""",
                """    def apply_top_k_top_p(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        prefer_triton: bool = False,
    ) -> torch.Tensor:
        top_k, top_p = self.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
        if top_k is None and top_p is None:
            return logits
        return apply_top_k_top_p(logits, top_k, top_p, prefer_triton=prefer_triton)""",
            ),
        ],
    ),
    # ---------------------------------------------------------------- 3 ----
    (
        GPU_SAMPLER,
        "prefer_triton",
        [
            (
                """        expanded_local_pos: torch.Tensor,
        skip_top_k_top_p: bool = False,
    ) -> torch.Tensor:
        if not np.any(self.needs_logits_processing[idx_mapping_np]):
            return logits""",
                """        expanded_local_pos: torch.Tensor,
        skip_top_k_top_p: bool = False,
        prefer_triton: bool = False,
    ) -> torch.Tensor:
        if not np.any(self.needs_logits_processing[idx_mapping_np]):
            return logits""",
            ),
            (
                """        # Apply top_k and/or top_p. This might or might not return a new tensor.
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np
        )""",
                """        # Apply top_k and/or top_p. This might or might not return a new tensor.
        return self.sampling_states.apply_top_k_top_p(
            logits, expanded_idx_mapping, idx_mapping_np, prefer_triton=prefer_triton
        )""",
            ),
        ],
    ),
    # ---------------------------------------------------------------- 4 ----
    (
        REJ,
        "prefer_triton=True",
        [
            (
                """        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
        )""",
                """        processed_logits = self.sampler.apply_sampling_params(
            logits,
            expanded_idx_mapping,
            idx_mapping,
            idx_mapping_np,
            pos,
            draft_sampled,
            expanded_local_pos,
            # L5a: ask for the sort-free pivot truncation. Honoured only when
            # VLLM_VERIFY_TOPK_TRITON=1 (topk_topp_sampler.py).
            prefer_triton=True,
        )""",
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
            assert old in src, "%s: anchor moved --\n%s" % (name, old)
            assert src.count(old) == 1, "%s: anchor is not unique --\n%s" % (name, old)
            src = src.replace(old, new, 1)
        open(path, "w").write(src)
        ast.parse(open(path).read())
        applied.append(name)
    if skipped:
        print(
            "patch_verify_topk_pivot.py: already applied to %s" % ", ".join(skipped),
            file=sys.stderr,
        )
    if applied:
        print(
            "patch_verify_topk_pivot.py applied OK to %s" % ", ".join(applied),
            file=sys.stderr,
        )


main()
