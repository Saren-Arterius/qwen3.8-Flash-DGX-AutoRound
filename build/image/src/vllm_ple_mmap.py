"""vllm_ple_mmap — serve the Qwen3.8-Flash-Next N-gram (PLE) table from NVMe via mmap.

Why: the 51B-parameter n-gram table is 44 GiB in FP8 and vLLM keeps it resident
(GPU, or pinned host RAM with VLLM_PLE_CPU_OFFLOAD). On a DGX Spark / GX10 the
host and the GPU share one 121 GiB pool, so neither fits next to the 78 GiB main
model. But a token only ever touches 16 rows x 160 bytes of that table, so the
table can live on disk and be served through the page cache — exactly what
llama.cpp does with its GGUF mmap.

How: with VLLM_PLE_MMAP=1 this module patches ``Qwen3_8FlashNextNGramEmbedding``:
  * ``__init__`` swaps the 44/95 GiB ``VocabParallelEmbedding`` for a tiny
    placeholder whose ``forward(ids)`` gathers rows from ``np.memmap`` views of the
    checkpoint's ``model-plefp8-*.safetensors`` shards (zero-copy, page-cache backed);
  * ``load_weights`` drops the 128 shard tensors on the floor, keeps the global FP8
    ``weight_scale`` (as ``_offload_weight_scale``, which the untouched
    ``Qwen3_8FlashNextPLELayer._dequantize_embeddings`` already consumes) and opens
    the memmaps.
  * ``forward_impl`` (hashing + lookup) is wrapped in a custom op
    ``vllm::ple_mmap_lookup`` so that (a) torch.compile treats it as opaque — the
    stock version trips an Inductor int64 indexing assert on sm_121 — and (b) it can
    be listed in ``-cc.splitting_ops`` and run OUTSIDE piecewise CUDA graphs: the
    gather is CPU work + a pageable H2D copy, which cannot live inside a capture.
    Use ``-cc.cudagraph_mode=PIECEWISE`` (not FULL*) with the splitting op list in
    serve-flashnext-vllm.sh, or ``--enforce-eager``.
Nothing else in vLLM changes: the n-gram hashing, the short-conv, the dequant path
are the stock ones.

Knobs (env):
  VLLM_PLE_MMAP=1            enable
  VLLM_PLE_MMAP_WORKERS=32   gather threads (page faults overlap across threads)
  VLLM_PLE_MMAP_CHUNK=2048   rows per gather task
  VLLM_PLE_MMAP_PREWARM=0    1 = stream the whole table once at load to fill the
                             page cache with whatever memory is free (harmless,
                             evictable; ~10 s at 4.7 GB/s)
  VLLM_PLE_MMAP_PREFETCH=0   EXPERIMENTAL: 1 = run the n-gram hash at batch
                             assembly and gather rows in a worker thread, so
                             the mid-layer-1 lookup op only copies ready rows
                             (hides the page-fault latency behind layer-0
                             compute). Off by default; lightly tested.
  VLLM_PLE_MMAP_FAST_PATH=0  1 = gather rows with posix_fadvise(WILLNEED) +
                             os.preadv instead of thread-pooled np.memmap
                             fancy indexing. Same bytes in the same order --
                             only the mechanism changes -- but ~6x less time
                             per decode-width gather, because the pool costs
                             more than the reads at that width and numpy takes
                             its page faults with the GIL held. Off by default:
                             unset, the module is byte-for-byte the old one.
  VLLM_PLE_MMAP_FAST_MAX_ROWS=8192
                             gathers wider than this keep the pooled path
                             (prefill: the per-row Python loop of the fast path
                             stops paying off once tasks are big enough to
                             amortise the pool).

Install: the Dockerfile copies this file next to vllm and appends
``_ple_mmap_apply(Qwen3_8FlashNextNGramEmbedding)`` to the end of
``vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py``. See the repo README.
"""

from __future__ import annotations

import glob
import json
import logging
import math
import os
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("vllm.ple_mmap")

ENV_ENABLE = "VLLM_PLE_MMAP"
ENV_FAST_PATH = "VLLM_PLE_MMAP_FAST_PATH"
_FP8_DTYPES = {
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}
# 16-bit tables need no weight_scale: the stock _dequantize_embeddings is a
# no-op for non-FP8 rows.
_TABLE_DTYPES = {
    **_FP8_DTYPES,
    "BF16": torch.bfloat16,
    "F16": torch.float16,
}


def enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0").lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# safetensors header parsing (no dependency on the safetensors package: we need
# raw file offsets, which its Python API does not expose)
# --------------------------------------------------------------------------- #
def parse_safetensors_header(path: str) -> tuple[dict, int]:
    """Return (header_dict, data_start_offset) of a safetensors file."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    header.pop("__metadata__", None)
    return header, 8 + header_len


# --------------------------------------------------------------------------- #
# VLLM_PLE_MMAP_FAST_PATH=1: syscall gather instead of pooled memmap faults
#
# Profile of the v13 pooled path at decode width (160 unique rows, the real
# 47.68 GiB / 128-shard table, WORKERS=32 CHUNK=8 FAST_ROWS=16 MADV_RANDOM=1):
#
#     np.unique                0.030 ms    0.7 %
#     shard/local/task prep    0.040 ms    0.9 %
#     pool.map (the gather)    4.276 ms   98.0 %
#     out[inverse] scatter     0.015 ms    0.3 %
#                             ---------
#                              4.364 ms/op  cold   (1.885 ms/op on warm rows)
#
# The caller (_MmapNgramEmbedding.forward) already deduplicates, so the ids
# arrive sorted; spread over 128 shards they form ~92 shard-groups of ~1.7 rows
# each, i.e. ~92 pool tasks per decode step. Dispatching 92 EMPTY tasks through
# that executor costs 0.53 ms on its own, and the same gather run serially takes
# 0.37 ms once the rows are cached: the pool is pure overhead on warm rows, and
# on cold ones it recovers only 3x of a possible 32x because np.memmap fancy
# indexing takes its page faults with the GIL held.
#
# So the fast path drops the pool and asks the kernel for the parallelism:
#   1. posix_fadvise(WILLNEED) over each row -- async, so one pass queues all
#      ~160 page-ins at block-layer depth instead of thread depth;
#   2. os.preadv straight into the output rows -- one syscall per row, GIL
#      released for the syscall, and the pages are already in flight by then.
# Same file, same offsets, same bytes, same order, same exceptions.
#
# Measured on the real table (this box, same id sets, ms/op):
#     rows        16     64    160    512   2048   8192
#     old cold  1.74   2.42   3.94   6.17  14.57  50.75
#     new cold  0.27   0.44   0.61   1.63   5.79  22.26
#     old warm  0.10   1.12   2.32   3.51   9.37  26.30
#     new warm  0.04   0.06   0.16   0.55   1.98   9.05
# The WILLNEED pass is what makes it work: serial preadv WITHOUT it is 9.15 ms
# cold at 160 rows (every read stalls on its own I/O), and pooled preadv WITH it
# is 1.94 ms (the pool overhead is back).
# --------------------------------------------------------------------------- #
_FADV_WILLNEED = getattr(os, "POSIX_FADV_WILLNEED", None)


def _fast_path_enabled() -> bool:
    return os.environ.get(ENV_FAST_PATH, "0").lower() in ("1", "true", "yes")


def _preadv_rest(fd: int, row, off: int, got: int, row_bytes: int) -> None:
    """Finish a short ``preadv``.

    A regular file below EOF does not normally return one, but a signal can cut
    a read short and a half-filled row would be silent corruption, so the fast
    path checks every read and lands here on the rare short one.
    """
    mv = memoryview(row)
    while got < row_bytes:
        n = os.preadv(fd, (mv[got:],), off + got)
        if n <= 0:
            raise OSError(
                f"PLE mmap: short read at offset {off} "
                f"({got}/{row_bytes} bytes)"
            )
        got += n


class MmapPleTable:
    """Row gather over a table split into ``split_ngram_parts`` shard files.

    ``shards``: {shard_index: (path, absolute_byte_offset, rows)}. Shard ``i``
    holds global rows ``[i*shard_size, i*shard_size + rows)`` (vLLM's
    ``copy_ple_embedding_shard_`` layout).
    """

    def __init__(
        self,
        shards: dict[int, tuple[str, int, int]],
        shard_size: int,
        row_bytes: int,
        torch_dtype: torch.dtype,
        workers: int = 32,
        chunk: int = 2048,
    ) -> None:
        if not shards:
            raise ValueError("no PLE shards")
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.chunk = max(1, int(chunk))
        self.paths: list[str | None] = [None] * (max(shards) + 1)
        self.mm: list[np.memmap | None] = [None] * (max(shards) + 1)
        self.rows_total = 0
        # VLLM_PLE_MMAP_MADV_RANDOM=1: advise the kernel the gathers are
        # random access, disabling readahead/fault-around on these mappings.
        # Worth it when the table can't fit in page cache (e.g. remote-RAM
        # backing on a memory-tight box) — each miss then pulls one page
        # instead of a readahead window of mostly-evicted-later pages.
        # Default off: with local NVMe and cache headroom, readahead helps.
        # Note it also makes VLLM_PLE_MMAP_PREWARM's sequential pass slower.
        madv_random = _env_int("VLLM_PLE_MMAP_MADV_RANDOM", 0)
        for idx, (path, offset, rows) in shards.items():
            self.paths[idx] = path
            self.mm[idx] = np.memmap(
                path, dtype=np.uint8, mode="r", offset=offset, shape=(rows, row_bytes)
            )
            if madv_random:
                import mmap as _mmap

                self.mm[idx]._mmap.madvise(_mmap.MADV_RANDOM)
            self.rows_total += rows
        self.pool = ThreadPoolExecutor(max_workers=max(1, int(workers)))
        self.fast_rows = _env_int("VLLM_PLE_MMAP_FAST_ROWS", 512)
        # VLLM_PLE_MMAP_FAST_PATH (see the comment block above this class).
        # Everything below is inert unless the variable is set.
        self._fast = False
        self._fast_fds: list[int] = []
        self._fd_of: np.ndarray | None = None
        self._base_of: np.ndarray | None = None
        self._fast_all_present = False
        self._fast_max_rows = _env_int("VLLM_PLE_MMAP_FAST_MAX_ROWS", 8192)
        if _fast_path_enabled():
            self._setup_fast_path()

    def _setup_fast_path(self) -> None:
        """Open one O_RDONLY fd per shard FILE and tabulate (fd, base offset)
        per shard index. ``np.memmap`` closes its own descriptor once the
        mapping exists, so the syscall path needs its own; there are 33 files
        behind the 128 shards, hence 33 fds for the life of the process."""
        if _FADV_WILLNEED is None or not hasattr(os, "preadv"):
            logger.warning(
                "PLE mmap: FAST_PATH asked for but posix_fadvise/preadv are "
                "unavailable; keeping the pooled path"
            )
            return
        n = len(self.mm)
        fd_of = np.full(n, -1, dtype=np.int64)
        base_of = np.zeros(n, dtype=np.int64)
        by_path: dict[str, int] = {}
        try:
            for idx, (path, mm) in enumerate(zip(self.paths, self.mm)):
                if path is None or mm is None:
                    continue
                fd = by_path.get(path)
                if fd is None:
                    fd = os.open(path, os.O_RDONLY)
                    by_path[path] = fd
                    self._fast_fds.append(fd)
                fd_of[idx] = fd
                base_of[idx] = int(mm.offset)
        except OSError as exc:
            logger.warning("PLE mmap: FAST_PATH setup failed (%s); keeping "
                           "the pooled path", exc)
            for fd in self._fast_fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fast_fds = []
            return
        self._fd_of = fd_of
        self._base_of = base_of
        self._fast_all_present = bool((fd_of >= 0).all())
        self._fast = True
        logger.info(
            "PLE mmap: FAST_PATH on (fadvise+preadv gather, %d fds, "
            "max_rows=%d)", len(self._fast_fds), self._fast_max_rows,
        )

    def gather(self, ids: np.ndarray) -> np.ndarray:
        """ids: int64 [N] global row ids -> uint8 [N, row_bytes] (a fresh array)."""
        import time as _time

        t0 = _time.perf_counter()
        try:
            return self._gather(ids)
        finally:
            _STATS["gather_ms"] += (_time.perf_counter() - t0) * 1e3
            _STATS["rows"] += int(np.asarray(ids).size)
            _STATS["bytes"] += int(np.asarray(ids).size) * self.row_bytes

    def _gather(self, ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        if self._fast and ids.size <= self._fast_max_rows:
            try:
                rows = self._gather_fast(ids)
            except OSError as exc:
                # fadvise/preadv stopped working (a closed fd, a full-blown I/O
                # error): never try again, fall through to the pooled path.
                self._fast = False
                logger.warning(
                    "PLE mmap: FAST_PATH disabled after %r; falling back to "
                    "the pooled path", exc,
                )
            else:
                _STATS["fast"] += 1
                return rows
        if ids.size <= self.fast_rows:
            # Decode-sized batches: thread-pool dispatch costs more than the
            # reads themselves (~50 tasks for ~65 rows). Gather inline instead.
            if ids.min() < 0 or ids.max() >= self.shard_size * len(self.mm):
                raise IndexError(
                    f"PLE row id out of range: [{ids.min()}, {ids.max()}] "
                    f"for {self.rows_total} rows"
                )
            shard = ids // self.shard_size
            local = ids - shard * self.shard_size
            out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
            for si in np.unique(shard):
                mask = shard == si
                out[mask] = self.mm[si][local[mask]]
            return out
        # Dedupe + sort: repeated n-grams are common, and sorted rows improve
        # locality inside a shard.
        uniq, inverse = np.unique(ids, return_inverse=True)
        if uniq[0] < 0 or uniq[-1] >= self.shard_size * len(self.mm):
            raise IndexError(
                f"PLE row id out of range: [{uniq[0]}, {uniq[-1]}] "
                f"for {self.rows_total} rows"
            )
        shard = uniq // self.shard_size
        local = uniq - shard * self.shard_size
        out = np.empty((uniq.size, self.row_bytes), dtype=np.uint8)

        bounds = np.flatnonzero(np.diff(shard)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [uniq.size]))
        tasks: list[tuple[int, int, int]] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            si = int(shard[s])
            for c in range(s, e, self.chunk):
                tasks.append((si, c, min(c + self.chunk, e)))

        def run(task: tuple[int, int, int]) -> None:
            si, a, b = task
            mm = self.mm[si]
            if mm is None:
                raise IndexError(f"PLE shard {si} missing")
            # Fancy indexing on a memmap: page faults do the I/O; NumPy releases
            # the GIL for the copy, so tasks overlap across threads.
            out[a:b] = mm[local[a:b]]

        if len(tasks) == 1:
            run(tasks[0])
        else:
            for _ in self.pool.map(run, tasks):
                pass
        return out[inverse]

    def _gather_fast(self, ids: np.ndarray) -> np.ndarray:
        """VLLM_PLE_MMAP_FAST_PATH=1 gather. Bit-identical to ``_gather``.

        ``ids`` is already contiguous int64 1-D and non-empty. The returned
        array is a fresh, writable, C-contiguous uint8 ``[ids.size, row_bytes]``
        -- exactly what the pooled path returns, and the range check raises the
        same ``IndexError`` with the same message, since ``ids.min()/max()``
        are ``uniq[0]/uniq[-1]``.
        """
        lo = int(ids.min())
        hi = int(ids.max())
        if lo < 0 or hi >= self.shard_size * len(self.mm):
            raise IndexError(
                f"PLE row id out of range: [{lo}, {hi}] "
                f"for {self.rows_total} rows"
            )
        # The caller deduplicates before us, so the common case is an already
        # strictly increasing array: then uniq is ids and inverse is arange,
        # and both np.unique and the out[inverse] scatter can be skipped. The
        # check is exact -- anything else goes through np.unique as before.
        inverse = None
        if ids.size > 1 and not bool((ids[1:] > ids[:-1]).all()):
            uniq, inverse = np.unique(ids, return_inverse=True)
        else:
            uniq = ids
        shard = uniq // self.shard_size
        fds_a = self._fd_of[shard]
        if not self._fast_all_present and int(fds_a.min()) < 0:
            raise IndexError(
                f"PLE shard {int(shard[int(np.argmin(fds_a))])} missing"
            )
        row_bytes = self.row_bytes
        offs = self._base_of[shard] + (uniq - shard * self.shard_size) * row_bytes
        fds = fds_a.tolist()
        offl = offs.tolist()
        out = np.empty((uniq.size, row_bytes), dtype=np.uint8)

        # Pass 1: hand the kernel every page at once. posix_fadvise(WILLNEED)
        # returns as soon as the read is queued, so this pass costs syscalls,
        # not I/O waits, and the device sees ~160 outstanding reads instead of
        # the 32 a thread pool can hold.
        fadvise = os.posix_fadvise
        willneed = _FADV_WILLNEED
        for fd, off in zip(fds, offl):
            fadvise(fd, off, row_bytes, willneed)
        # Pass 2: read into the output rows. os.preadv releases the GIL for the
        # syscall and writes straight into `out` -- no intermediate buffer.
        preadv = os.preadv
        for row, fd, off in zip(out, fds, offl):
            got = preadv(fd, (row,), off)
            if got != row_bytes:
                _preadv_rest(fd, row, off, got, row_bytes)
        return out if inverse is None else out[inverse]

    def prewarm(self) -> None:
        """Stream every shard once so the page cache holds as much as it can."""
        block = 64 << 20
        for path, mm in zip(self.paths, self.mm):
            if path is None or mm is None:
                continue
            start = mm.offset
            end = start + mm.shape[0] * mm.shape[1]
            with open(path, "rb", buffering=0) as f:
                pos = start
                while pos < end:
                    n = f.readinto(bytearray(min(block, end - pos)))  # noqa: F841
                    if not n:
                        break
                    pos += n


# --------------------------------------------------------------------------- #
# Placeholder that stands in for VocabParallelEmbedding
# --------------------------------------------------------------------------- #
class PrefetchingMmapTable:
    """EXPERIMENTAL (VLLM_PLE_MMAP_PREFETCH=1): wrap MmapPleTable with the
    prefetch pipeline — the n-gram ids are hashed at batch-assembly time (a
    hook on Qwen3_8FlashNextModelState.prepare_inputs), a worker thread does
    the mmap gather into a pinned buffer while the GPU runs layer 0, and the
    lookup op consumes ready rows (tiny H2D + reshape, no np.unique). Dummy/
    capture runs bypass the hook and fall back to the sync gather, which is
    also the safety net for any shape mismatch. Off by default."""

    _SLOT = 1 << 17  # 131072 rows = an 8192-token chunk x 16 heads

    def __init__(self, inner: MmapPleTable) -> None:
        import queue
        import threading

        self.inner = inner
        self.row_bytes = inner.row_bytes
        self.torch_dtype = inner.torch_dtype
        self.rows_total = inner.rows_total
        self.rows_t = torch.empty((2 * self._SLOT, inner.row_bytes),
                                  dtype=torch.uint8, pin_memory=True)
        self.ids_t = torch.empty((2 * self._SLOT,), dtype=torch.int64,
                                 pin_memory=True)
        self._q: "queue.Queue" = queue.Queue()
        self._jobs = [None, None]
        self._pending = None
        self._slot = 0
        threading.Thread(target=self._worker, daemon=True,
                         name="ple-mmap-pf").start()
        _patch_model_state_prefetch()

    def gather(self, ids: np.ndarray) -> np.ndarray:  # sync fallback path
        return self.inner.gather(ids)

    def prefetch(self, ids: torch.Tensor) -> None:
        import threading

        n = ids.numel()
        if n == 0 or n > self._SLOT:
            return
        slot = self._slot
        self._slot ^= 1
        prev = self._jobs[slot]
        if prev is not None and not prev["done"].wait(5.0):
            logger.error("PLE prefetch: slot %d stuck, skipping", slot)
            return
        base = slot * self._SLOT
        self.ids_t[base:base + n].copy_(ids.reshape(-1), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        job = {"ev": ev, "n": n, "base": base, "ok": False,
               "done": threading.Event()}
        self._jobs[slot] = job
        self._pending = job
        self._q.put(job)

    def _worker(self) -> None:
        ids_np = self.ids_t.numpy()
        rows_np = self.rows_t.numpy()
        while True:
            job = self._q.get()
            try:
                job["ev"].synchronize()  # hash + ids D2H done
                n, base = job["n"], job["base"]
                rows_np[base:base + n] = self.inner._gather(
                    ids_np[base:base + n])
                job["ok"] = True
            except Exception:
                logger.exception("PLE prefetch: worker error")
            finally:
                job["done"].set()

    def consume(self, want_rows: int, device) -> "torch.Tensor | None":
        job, self._pending = self._pending, None
        if job is None or job["n"] != want_rows or not job["done"].wait(5.0) \
                or not job["ok"]:
            _STATS["pf_miss"] = _STATS.get("pf_miss", 0) + 1
            return None
        _STATS["pf_hit"] = _STATS.get("pf_hit", 0) + 1
        rows = self.rows_t[job["base"]:job["base"] + job["n"]]
        return rows.to(device, non_blocking=True)


_MS_PATCHED = False


def _patch_model_state_prefetch() -> None:
    """Hook prepare_inputs (batch assembly, outside torch.compile) to run the
    n-gram hash early; the placeholder embedding sees _PREFETCH.on and starts
    the async gather instead of a sync one."""
    global _MS_PATCHED
    if _MS_PATCHED:
        return
    _MS_PATCHED = True
    try:
        from vllm.models.qwen3_8_flash_next.nvidia.model_state import (
            Qwen3_8FlashNextModelState,
        )
    except ImportError:
        logger.warning("PLE prefetch: model_state not found, prefetch disabled")
        return

    orig = Qwen3_8FlashNextModelState.prepare_inputs

    def prepare_inputs(self, input_batch, req_states):
        out = orig(self, input_batch, req_states)
        try:
            qsl = out.get("query_start_loc")
            ctx = out.get("ngram_context")
            ids = out.get("input_ids")
            if ids is None:
                ids = getattr(input_batch, "input_ids", None)
            if ids is not None and qsl is not None and ctx is not None:
                layers = list(_REGISTRY.values())
                if len(layers) == 1 and hasattr(
                        getattr(layers[0].ngram_embedding, "table", None),
                        "prefetch"):
                    _PREFETCH.on = True
                    try:
                        layers[0]._ple_mmap_orig_forward_impl(
                            None, ids, qsl, ctx)
                    finally:
                        _PREFETCH.on = False
        except Exception:
            logger.exception("PLE prefetch: hook error (sync fallback)")
        return out

    Qwen3_8FlashNextModelState.prepare_inputs = prepare_inputs
    logger.info("PLE prefetch: hook installed on prepare_inputs")


class _MmapNgramEmbedding(nn.Module):
    """Duck-types the bits of VocabParallelEmbedding the PLE code reads.

    No ``weight`` attribute on purpose: ``Qwen3_8FlashNextPLELayer`` then falls
    back to ``ple_embedding._offload_weight_scale`` for the FP8 scale.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.org_vocab_size = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.table: MmapPleTable | None = None
        self._zeros_dtype = torch.bfloat16

    def _pinned_buf(self, rows: int, row_bytes: int) -> torch.Tensor | None:
        """Persistent pinned staging buffer for async H2D (grown as needed)."""
        buf = getattr(self, "_pinned", None)
        if buf is None or buf.shape[0] < rows or buf.shape[1] != row_bytes:
            try:
                cap = max(rows + rows // 2, 4096)
                buf = torch.empty((cap, row_bytes), dtype=torch.uint8, pin_memory=True)
            except RuntimeError:  # no CUDA (CPU tests) or pinning unavailable
                buf = None
            self._pinned = buf
        return buf

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        table = self.table
        if getattr(_PREFETCH, "on", False) and hasattr(table, "prefetch"):
            # batch-assembly hash: kick off the async prefetch, return a
            # dummy — the real rows are consumed later by _lookup_impl.
            table.prefetch(ids)
            return torch.empty(
                (*ids.shape, self.embedding_dim),
                dtype=table.torch_dtype, device=ids.device,
            )
        if table is None:
            # Weights never loaded (e.g. --load-format dummy): keep the plumbing
            # alive with zeros so kernel tests can run without the 44 GiB table.
            return torch.zeros(
                (*ids.shape, self.embedding_dim),
                dtype=self._zeros_dtype,
                device=ids.device,
            )
        ids_np = ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)
        # Dedup on CPU, gather only unique rows, expand on the GPU: fewer disk
        # reads AND fewer H2D bytes (repeated n-grams are the common case).
        uniq, inverse = np.unique(ids_np, return_inverse=True)
        rows = table.gather(uniq)  # uint8 [U, row_bytes], fresh & writable
        u = rows.shape[0]
        buf = self._pinned_buf(u, table.row_bytes) if ids.device.type == "cuda" else None
        if buf is not None:
            buf[:u].numpy()[:] = rows
            dev = buf[:u].to(ids.device, non_blocking=True)
        else:
            dev = torch.from_numpy(rows).to(ids.device)
        inv = torch.from_numpy(inverse.reshape(-1)).to(ids.device, non_blocking=True)
        out = dev.view(table.torch_dtype)[inv]
        return out.reshape(*ids.shape, self.embedding_dim)


# --------------------------------------------------------------------------- #
# Patch
# --------------------------------------------------------------------------- #
def _find_shards(
    model_path: str, layer_idx: int
) -> tuple[dict[int, tuple[str, int, int]], str | None, tuple[str, int, int] | None]:
    """Locate ``layers.<idx>.ple.ple_embedding.ngram_embedding.shard_N.weight``.

    Returns (shards, dtype_str, scale_entry) where scale_entry is
    (path, abs_offset, nbytes) of ``ngram_embedding.weight_scale`` or None.
    """
    shard_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
    )
    scale_re = re.compile(
        rf"layers\.{layer_idx}\.ple\.ple_embedding\.ngram_embedding\.weight_scale$"
    )
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        files = sorted(
            {
                os.path.join(model_path, fn)
                for name, fn in weight_map.items()
                if shard_re.search(name) or scale_re.search(name)
            }
        )
    else:
        files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))

    shards: dict[int, tuple[str, int, int]] = {}
    dtype_str: str | None = None
    scale_entry: tuple[str, int, int] | None = None
    for path in files:
        header, data_start = parse_safetensors_header(path)
        for name, meta in header.items():
            m = shard_re.search(name)
            if m:
                start, end = meta["data_offsets"]
                rows, cols = meta["shape"]
                if dtype_str is None:
                    dtype_str = meta["dtype"]
                elif meta["dtype"] != dtype_str:
                    raise ValueError("PLE shards have mixed dtypes")
                if end - start != rows * cols * _itemsize(dtype_str):
                    raise ValueError(f"PLE shard {name}: size/shape mismatch")
                shards[int(m.group(1))] = (path, data_start + start, rows)
                shard_cols = cols
            elif scale_re.search(name):
                start, end = meta["data_offsets"]
                scale_entry = (path, data_start + start, end - start, meta["dtype"])  # type: ignore[assignment]
    if shards:
        # Return cols through dtype_str consumer; keep it simple: stash on dict.
        shards["__cols__"] = shard_cols  # type: ignore[index]
    return shards, dtype_str, scale_entry


def _itemsize(dtype_str: str) -> int:
    return {
        "F8_E4M3": 1,
        "F8_E5M2": 1,
        "U8": 1,
        "I8": 1,
        "BF16": 2,
        "F16": 2,
        "F32": 4,
    }[dtype_str]


def _read_scale(entry: tuple) -> torch.Tensor:
    path, offset, nbytes, dtype_str = entry
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(nbytes)
    if dtype_str == "F32":
        return torch.tensor(struct.unpack("<f", raw[:4])[0], dtype=torch.float32)
    if dtype_str == "BF16":
        u16 = struct.unpack("<H", raw[:2])[0]
        return torch.tensor(u16 << 16, dtype=torch.int32).view(torch.float32).squeeze()
    if dtype_str == "F16":
        return torch.frombuffer(bytearray(raw[:2]), dtype=torch.float16).clone().squeeze()
    raise ValueError(f"unsupported weight_scale dtype {dtype_str}")


_REGISTRY: dict[str, nn.Module] = {}
_OP_NAME = "ple_mmap_lookup"

# Aggregate gather-overhead stats, logged every VLLM_PLE_MMAP_STATS_SEC seconds
# (0 = off). op_ms covers hashing + gather + H2D; gather_ms just the disk reads.
_PREFETCH = __import__("threading").local()  # set during the batch-assembly hash
_STATS = {"calls": 0, "op_ms": 0.0, "gather_ms": 0.0, "rows": 0, "bytes": 0,
          "fast": 0}
_STATS_LAST = [0.0]
_STATS_SEC = _env_int("VLLM_PLE_MMAP_STATS_SEC", 30)


def _stats_log() -> None:
    import time as _time

    now = _time.monotonic()
    if _STATS_SEC <= 0 or now - _STATS_LAST[0] < _STATS_SEC:
        return
    elapsed = now - _STATS_LAST[0] if _STATS_LAST[0] else float(_STATS_SEC)
    _STATS_LAST[0] = now
    s = _STATS
    if not s["calls"]:
        return
    logger.info(
        "PLE mmap stats (last %.0fs): %d ops, op %.0f ms total (%.2f ms/op), "
        "gather %.0f ms total (%.2f ms/op), %d rows, %.1f MiB read, "
        "prefetch hit %d miss %d, fast %d",
        elapsed, s["calls"], s["op_ms"], s["op_ms"] / s["calls"],
        s["gather_ms"], s["gather_ms"] / s["calls"],
        s["rows"], s["bytes"] / 2**20,
        s.get("pf_hit", 0), s.get("pf_miss", 0), s.get("fast", 0),
    )
    s.update(calls=0, op_ms=0.0, gather_ms=0.0, rows=0, bytes=0, fast=0)


def _lookup_impl(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    import time as _time

    t0 = _time.perf_counter()
    layer = _REGISTRY[layer_name]
    table = getattr(layer.ngram_embedding, "table", None)
    consume = getattr(table, "consume", None)
    got = None
    if consume is not None and output.dtype.itemsize == 1:
        got = consume(output.numel() // table.row_bytes, output.device)
    if got is not None:
        output.copy_(got.view(output.dtype).reshape(output.shape))
    else:
        result = layer._ple_mmap_orig_forward_impl(
            None, input_ids, query_start_loc, ngram_context
        )
        output[: result.shape[0]].copy_(result.to(output.dtype))
    _STATS["calls"] += 1
    _STATS["op_ms"] += (_time.perf_counter() - t0) * 1e3
    _stats_log()


def _lookup_fake(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _register_op() -> None:
    if hasattr(torch.ops.vllm, _OP_NAME):
        return
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name=_OP_NAME,
        op_func=_lookup_impl,
        mutates_args=["output"],
        fake_impl=_lookup_fake,
    )


def apply(cls: type) -> None:
    """Patch ``Qwen3_8FlashNextNGramEmbedding`` (pass the class) when enabled."""
    if not enabled():
        return
    if getattr(cls, "_ple_mmap_patched", False):
        return
    mod = sys.modules[cls.__module__]
    orig_init = cls.__init__
    orig_load_weights = cls.load_weights

    def __init__(self, config, embedding_dim, ple_dense_layer_id, max_total_tokens,
                 max_num_reqs, prefix, quant_config=None, params_dtype=None):
        # Run the stock constructor (hash buffers, workspaces, ...) with the
        # embedding class swapped for our placeholder so nothing large is
        # allocated. quant_config=None keeps the stock code from selecting an
        # FP8 quant method that would create an FP8 weight parameter.
        real_embedding_cls = mod.VocabParallelEmbedding
        mod.VocabParallelEmbedding = lambda n, d, **_kw: _MmapNgramEmbedding(n, d)
        try:
            orig_init(self, config, embedding_dim, ple_dense_layer_id,
                      max_total_tokens, max_num_reqs, prefix,
                      quant_config=None, params_dtype=params_dtype)
        finally:
            mod.VocabParallelEmbedding = real_embedding_cls
        self._ple_mmap_prefix = prefix
        _REGISTRY[prefix] = self
        self._ple_mmap_model_path = None
        try:
            from vllm.config import get_current_vllm_config
            self._ple_mmap_model_path = get_current_vllm_config().model_config.model
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("PLE mmap: cannot read model path from vllm config: %s", exc)
        if params_dtype is not None:
            self.ngram_embedding._zeros_dtype = params_dtype
        logger.info(
            "PLE mmap: %s -> placeholder embedding (%d rows x %d), table will be mmapped",
            prefix, self.ngram_embedding.org_vocab_size, self.head_dim,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        rest: list[tuple[str, torch.Tensor]] = []
        for name, w in weights:
            if name.startswith("ngram_embedding.shard_") and name.endswith(".weight"):
                loaded.add(name)  # served from disk, never materialised
                continue
            if name == "ngram_embedding.weight_scale":
                self.register_buffer(
                    "_offload_weight_scale",
                    w.detach().to(device=torch.accelerator.current_accelerator()),
                    persistent=False,
                )
                loaded.add(name)
                continue
            rest.append((name, w))
        loaded.update(orig_load_weights(self, rest))
        _setup_table(self)
        return loaded

    def _setup_table(self) -> None:
        if self.ngram_embedding.table is not None:
            return
        # VLLM_PLE_MMAP_DIR: serve the table from a different directory than the
        # checkpoint (e.g. an FP8 copy of the table on local NVMe).
        model_path = os.environ.get("VLLM_PLE_MMAP_DIR") or self._ple_mmap_model_path
        if not model_path or not os.path.isdir(model_path):
            raise RuntimeError(
                f"PLE mmap: table path {model_path!r} is not a local directory; "
                "point --model at the downloaded snapshot or set VLLM_PLE_MMAP_DIR"
            )
        m = re.search(r"layers\.(\d+)\.", self._ple_mmap_prefix)
        if not m:
            raise RuntimeError(f"PLE mmap: cannot find layer index in {self._ple_mmap_prefix!r}")
        layer_idx = int(m.group(1))
        parts = int(self.split_ngram_parts)
        vocab = int(self.ngram_embedding.org_vocab_size)
        shard_size = math.ceil(vocab / parts)
        if True:  # (indentation kept to minimize the diff)
            shards, dtype_str, scale_entry = _find_shards(model_path, layer_idx)
            if not shards:
                raise RuntimeError(f"PLE mmap: no shard tensors for layer {layer_idx} under {model_path}")
            cols = shards.pop("__cols__")  # type: ignore[arg-type]
            if cols != self.head_dim:
                raise RuntimeError(f"PLE mmap: shard width {cols} != head_dim {self.head_dim}")
            if dtype_str not in _TABLE_DTYPES:
                raise RuntimeError(f"PLE mmap: unsupported shard dtype {dtype_str}")
            if dtype_str in _FP8_DTYPES and not hasattr(self, "_offload_weight_scale"):
                if scale_entry is None:
                    raise RuntimeError("PLE mmap: FP8 shards without ngram_embedding.weight_scale")
                self.register_buffer(
                    "_offload_weight_scale",
                    _read_scale(scale_entry).to(torch.accelerator.current_accelerator()),
                    persistent=False,
                )
            for idx, (_p, _o, rows) in shards.items():
                expected = max(0, min(shard_size, vocab - idx * shard_size))
                if rows != expected:
                    raise RuntimeError(
                        f"PLE mmap: shard {idx} has {rows} rows, expected {expected}"
                    )
            mmap_table = MmapPleTable(
                shards, shard_size, cols * _itemsize(dtype_str), _TABLE_DTYPES[dtype_str],
                workers=_env_int("VLLM_PLE_MMAP_WORKERS", 32),
                chunk=_env_int("VLLM_PLE_MMAP_CHUNK", 2048),
            )
        table = mmap_table
        if _env_int("VLLM_PLE_MMAP_PREWARM", 0):
            logger.info("PLE mmap: prewarming page cache (%.1f GiB)...", table.rows_total * table.row_bytes / 2**30)
            table.prewarm()
        if _env_int("VLLM_PLE_MMAP_PREFETCH", 0):
            table = PrefetchingMmapTable(mmap_table)
        self.ngram_embedding.table = table
        logger.info(
            "PLE table ready: layer %d, %d rows x %d B, backend %s",
            layer_idx, table.rows_total, table.row_bytes, type(table).__name__,
        )

    def forward_impl(self, hidden_states, input_ids, query_start_loc, ngram_context,
                     output_buffer=None):
        del hidden_states, output_buffer
        num_tokens = input_ids.reshape(-1).shape[0]
        table = self.ngram_embedding.table
        dtype = table.torch_dtype if table is not None else self.ngram_embedding._zeros_dtype
        output = torch.empty(
            (num_tokens, self.embedding_dim), dtype=dtype, device=input_ids.device
        )
        getattr(torch.ops.vllm, _OP_NAME)(
            input_ids, query_start_loc, ngram_context, output, self._ple_mmap_prefix
        )
        return output

    _register_op()
    cls._ple_mmap_orig_forward_impl = cls.forward_impl
    cls.forward_impl = forward_impl
    cls.__init__ = __init__
    cls.load_weights = load_weights
    cls._setup_table = _setup_table
    cls._ple_mmap_patched = True
    logger.info("PLE mmap patch applied to %s.%s", cls.__module__, cls.__name__)
