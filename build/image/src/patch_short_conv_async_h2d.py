#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Backport of vLLM PR #55054 to this image's short_conv_attn.py.

PR #55054 ("async H2D for the PLE/MTP short-conv metadata", one file, +12/-9)
replaces five blocking ``Tensor.to(device)`` calls in
``Qwen3_8FlashNextPLEShortConvMetadataBuilder.build`` with ``async_tensor_h2d``,
and hoists the two request-index transfers so each is issued once instead of
once per use. Every touched line was verified present verbatim in this image
before the patch was written; the replacements below assert that, so a base-image
change turns into a build failure rather than a silent no-op.

WHY IT IS SAFE HERE
    ``vllm/utils/torch_utils.py:690-703``:

        if isinstance(data, torch.Tensor):
            t = data.pin_memory() if PIN_MEMORY else data
        ...
        return t.to(device=device, dtype=dtype, non_blocking=True)

    so the source buffer IS pinned (PIN_MEMORY is is_pin_memory_available(),
    true on CUDA). The pinned staging tensor is a temporary, but it comes from
    PyTorch's CachingHostAllocator, which records an event on the copy stream and
    withholds the block from reuse until that event completes - the standard
    reason ``async_tensor_h2d`` is safe to call with a throwaway tensor.

    The tensors involved are per-request index vectors of at most --max-num-seqs
    (8) int64 elements, so the extra pinned staging copy is nanoseconds.

    ``non_spec_req_idx`` is only read inside the mixed-batch ``else`` branch and
    inside ``if num_decodes > 0 or num_prefills > 0``. Those two conditions are
    exactly complementary to the pure-spec-decode branch that leaves it None,
    which is why upstream's ``assert non_spec_req_idx is not None`` is sound; the
    dead ``if non_spec_req_idx_cpu is not None`` it replaces could never be false
    because line 332 always assigns it.

EXPECTED GAIN
    Small and probably below the harness's tok/s noise band. Upstream measured
    1.9 ms of an 11.2 ms GB300 step; our step is ~66 ms and we do not run async
    scheduling, so there is less run-ahead to unlock. Attribute it with
    STEP_PROFILE, not with tok/s.
"""

import ast
import sys

TARGET = (
    "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/"
    "short_conv_attn.py"
)

# (description, exact text that must be present exactly once, replacement)
EDITS = [
    (
        "hoist spec_req_idx to async H2D; defer non_spec_req_idx",
        """        spec_req_idx = spec_req_idx_cpu.to(query_start_loc.device)
        non_spec_req_idx = non_spec_req_idx_cpu.to(query_start_loc.device)
""",
        """        spec_req_idx = async_tensor_h2d(
            spec_req_idx_cpu, device=query_start_loc.device
        )
        non_spec_req_idx: torch.Tensor | None = None
""",
    ),
    (
        "issue the mixed-batch index transfers once, asynchronously",
        """            req_group = torch.full(
""",
        """            non_spec_req_idx = async_tensor_h2d(
                non_spec_req_idx_cpu, device=query_start_loc.device
            )
            decode_req_idx = async_tensor_h2d(
                decode_req_idx_cpu, device=query_start_loc.device
            )
            req_group = torch.full(
""",
    ),
    (
        "reuse the hoisted decode_req_idx",
        """            req_group[decode_req_idx_cpu.to(query_start_loc.device)] = 1
""",
        """            req_group[decode_req_idx] = 1
""",
    ),
    (
        "reuse the hoisted spec_req_idx for the accepted-token gather",
        """        num_accepted_tokens = num_accepted_tokens[
            spec_req_idx_cpu.to(num_accepted_tokens.device)
        ]
""",
        """        num_accepted_tokens = num_accepted_tokens[spec_req_idx]
""",
    ),
    (
        "reuse the hoisted non_spec_req_idx for num_computed_tokens",
        """            if non_spec_req_idx_cpu is not None:
                non_spec_req_idx = non_spec_req_idx_cpu.to(num_computed_tokens.device)
                num_computed_tokens = num_computed_tokens[non_spec_req_idx]
""",
        """            assert non_spec_req_idx is not None
            num_computed_tokens = num_computed_tokens[non_spec_req_idx]
""",
    ),
]

SENTINEL = "non_spec_req_idx: torch.Tensor | None = None"


def main() -> int:
    src = open(TARGET, encoding="utf-8").read()
    if SENTINEL in src:
        print("patch_short_conv_async_h2d.py: already applied", file=sys.stderr)
        return 0
    if "from vllm.utils.torch_utils import async_tensor_h2d" not in src:
        raise SystemExit(
            "short_conv_attn.py does not import async_tensor_h2d; refusing to patch"
        )

    for description, old, new in EDITS:
        count = src.count(old)
        if count != 1:
            raise SystemExit(
                f"patch_short_conv_async_h2d.py: {description!r} matched {count} "
                "times, expected exactly 1 - the base image has moved, re-verify "
                "against vllm-project/vllm#55054 before rebuilding"
            )
        src = src.replace(old, new, 1)

    # Post-conditions: none of the replaced blocking transfers may survive.
    for stale in (
        "spec_req_idx_cpu.to(",
        "decode_req_idx_cpu.to(",
        "non_spec_req_idx_cpu.to(",
    ):
        if stale in src:
            raise SystemExit(f"patch left a blocking transfer behind: {stale}")

    ast.parse(src)
    open(TARGET, "w", encoding="utf-8").write(src)
    print("patch_short_conv_async_h2d.py applied OK (vllm#55054)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
