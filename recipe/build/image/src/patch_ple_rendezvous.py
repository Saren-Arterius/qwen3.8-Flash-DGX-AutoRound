#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

import ast
import sys

PLE = "/usr/local/lib/python3.12/dist-packages/vllm_ple_mmap.py"

OLD_LINE = (
    '        ids_np = ids.detach().to("cpu", non_blocking=False)'
    ".numpy().reshape(-1)\n"
)
NEW_LINE = """        ids_np = (
            self._ids_to_cpu_async(ids)
            if _ASYNC_IDS
            else ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        )
"""

OLD_FORWARD = "    def forward(self, ids: torch.Tensor) -> torch.Tensor:\n"
NEW_FORWARD = '''    def _pinned_ids(self, n: int) -> "torch.Tensor | None":
        """Persistent pinned int64 staging buffer for the ids D2H (grown as needed)."""
        buf = getattr(self, "_pin_ids", None)
        if buf is None or buf.numel() < n:
            try:
                buf = torch.empty((max(n + n // 2, 4096),), dtype=torch.int64,
                                  pin_memory=True)
            except RuntimeError:  # no CUDA (CPU tests) or pinning unavailable
                buf = None
            self._pin_ids = buf
        return buf

    def _ids_to_cpu_async(self, ids: torch.Tensor) -> "np.ndarray":
        """R2: pinned async D2H + CUDA-event wait instead of a pageable D2H.

        Returns a numpy view of the pinned buffer holding exactly the same int64
        values, in the same order, as ``ids.to("cpu")`` would. The caller consumes it
        immediately (``np.unique``, which copies) and never retains it, so reusing
        the buffer across steps is safe.
        """
        src = ids.detach().reshape(-1)
        if src.device.type != "cuda" or src.dtype != torch.int64:
            return src.to("cpu", non_blocking=False).numpy()
        n = src.numel()
        buf = self._pinned_ids(n)
        if buf is None:
            return src.to("cpu", non_blocking=False).numpy()
        dst = buf[:n]
        dst.copy_(src, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        if _ASYNC_IDS_TRACE:
            import time as _t

            _t0 = _t.perf_counter()
            ev.synchronize()
            _STATS["async_ids_wait_s"] = (
                _STATS.get("async_ids_wait_s", 0.0) + (_t.perf_counter() - _t0)
            )
            _STATS["async_ids_n"] = _STATS.get("async_ids_n", 0) + 1
        else:
            ev.synchronize()
        return dst.numpy()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
'''

OLD_FLAG = "_MS_PATCHED = False\n"
NEW_FLAG = """_MS_PATCHED = False

# --- qwen38-flash-dgx R2: PLE ids rendezvous (VLLM_PLE_MMAP_ASYNC_IDS) ------
# 1 = pinned async D2H + CUDA-event wait in place of the pageable D2H at the
#     ids hand-off; 2 = the same, plus wall-clock accounting of the event wait
#     into _STATS so the event-wait cost remains visible.
# Unset/0 leaves the original line in place, byte for byte.
_ASYNC_IDS_MODE = os.environ.get("VLLM_PLE_MMAP_ASYNC_IDS", "0")
_ASYNC_IDS = _ASYNC_IDS_MODE in ("1", "2")
_ASYNC_IDS_TRACE = _ASYNC_IDS_MODE == "2"
# R2 is dormant, so there is deliberately NO install line here: its mode-2
# counters are unreadable. The disabled line prints like the other patches'
# so that a log can still prove the default-off state.
if not _ASYNC_IDS:
    logger.info("iter6 R2 disabled")
"""


def main() -> None:
    src = open(PLE).read()

    for anchor in (
        "class _MmapNgramEmbedding(nn.Module):",
        "def _pinned_buf(self, rows: int, row_bytes: int)",
        "uniq, inverse = np.unique(ids_np, return_inverse=True)",
        "_STATS",
        "import os",
    ):
        assert anchor in src, "vllm_ple_mmap.py: anchor moved -- %r" % anchor

    if "_ids_to_cpu_async" in src:
        print("patch_ple_rendezvous.py: already applied, skipping", file=sys.stderr)
        return

    assert OLD_LINE in src, (
        "vllm_ple_mmap.py: the pageable-D2H rendezvous line moved. Expected:\n"
        + OLD_LINE
    )
    assert src.count(OLD_LINE) == 1, "vllm_ple_mmap.py: rendezvous line is not unique"
    assert OLD_FORWARD in src, "vllm_ple_mmap.py: MmapPleEmbedding.forward moved"
    assert src.count(OLD_FORWARD) == 1, (
        "vllm_ple_mmap.py: more than one `def forward(self, ids: torch.Tensor)`"
    )
    assert OLD_FLAG in src, "vllm_ple_mmap.py: _MS_PATCHED anchor moved"

    src = src.replace(OLD_FLAG, NEW_FLAG, 1)
    src = src.replace(OLD_FORWARD, NEW_FORWARD, 1)
    src = src.replace(OLD_LINE, NEW_LINE, 1)

    open(PLE, "w").write(src)
    ast.parse(open(PLE).read())
    print("patch_ple_rendezvous.py applied OK", file=sys.stderr)


main()
