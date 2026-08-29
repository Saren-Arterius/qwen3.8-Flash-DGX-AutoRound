"""PLE table over RDMA — client side (magi), phase B.

The 47.7G n-gram table lives in wtako's RAM behind ple_rdma_server.py; rows
are fetched with one-sided RDMA READs (libple_rdma.so, RC QP, manual
exchange). This module provides:

- ctypes bindings + Client (READs land in a pinned, MR-registered buffer)
- RdmaPleTable: gather() (sync path, same duck-type as MmapPleTable) plus a
  prefetch pipeline: at batch-assembly time (GPUModelRunner._preprocess) the
  n-gram hash runs and row READs are posted from a worker thread; by the
  time the (eager, graph-split) ple_mmap_lookup op executes mid-layer-1, the
  rows are already in pinned memory — the op just does a tiny H2D + reshape.
  No dedup on this path: 64 duplicate 160-byte reads cost less than one
  np.unique.

Env: VLLM_PLE_RDMA=<host:port> (selection happens in vllm_ple_mmap),
VLLM_PLE_RDMA_DEV / _GID, VLLM_PLE_RDMA_PREFETCH=1 (default on),
VLLM_PLE_RDMA_VERIFY=1 (needs a local mmap table dir).
"""
import ctypes
import json
import os
import queue
import socket
import threading

import logging

import numpy as np
import torch

logger = logging.getLogger("vllm.ple_rdma")

MAX_ROWS = 1 << 18          # per-call cap; 8192-token chunk * 16 = 131072
_SLOT = MAX_ROWS // 2       # two prefetch slots of 131072 rows
_NREGIONS = 3               # slot A, slot B, sync-gather region

_lib = None


def _load_lib():
    global _lib
    if _lib is None:
        path = os.environ.get(
            "PLE_RDMA_LIB",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "libple_rdma.so"),
        )
        _lib = ctypes.CDLL(path)
        _lib.ple_dev_open.restype = ctypes.c_void_p
        _lib.ple_dev_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        _lib.ple_mr_reg.restype = ctypes.c_void_p
        _lib.ple_mr_reg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        _lib.ple_mr_rkey.restype = ctypes.c_uint32
        _lib.ple_mr_rkey.argtypes = [ctypes.c_void_p]
        _lib.ple_mr_lkey.restype = ctypes.c_uint32
        _lib.ple_mr_lkey.argtypes = [ctypes.c_void_p]
        _lib.ple_qp_create.restype = ctypes.c_void_p
        _lib.ple_qp_create.argtypes = [ctypes.c_void_p]
        _lib.ple_qp_local.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
        _lib.ple_qp_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        _lib.ple_read_rows.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_uint32, ctypes.c_void_p, ctypes.c_long, ctypes.c_int,
        ]
    return _lib


def _recv_json(sock):
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("ple rdma server closed during exchange")
        data += chunk
    return json.loads(data)


class Client:
    """RC QP to the wtako daemon; READs land in the caller's pinned buffer."""

    def __init__(self, host, port, dev_name, gid_index, rows_np: np.ndarray, row_bytes: int):
        L = _load_lib()
        self.L = L
        self.row_bytes = row_bytes
        self.dev = L.ple_dev_open(dev_name.encode(), 1, gid_index)
        if not self.dev:
            raise RuntimeError(f"ple rdma: cannot open device {dev_name}")
        self.rows_ptr = rows_np.ctypes.data
        self.mr = L.ple_mr_reg(self.dev, self.rows_ptr, rows_np.nbytes, 0)
        if not self.mr:
            raise RuntimeError("ple rdma: local MR registration failed (memlock ulimit?)")
        self.lkey = L.ple_mr_lkey(self.mr)

        self.sock = socket.create_connection((host, port), timeout=30)
        self.remote = _recv_json(self.sock)
        qp = L.ple_qp_create(self.dev)
        gid = ctypes.create_string_buffer(16)
        qpn = ctypes.c_uint32()
        L.ple_qp_local(qp, gid, ctypes.byref(qpn))
        self.sock.sendall(json.dumps({"gid": gid.raw.hex(), "qpn": qpn.value}).encode() + b"\n")
        if L.ple_qp_connect(qp, bytes.fromhex(self.remote["gid"]), self.remote["qpn"]) != 0:
            raise RuntimeError("ple rdma: QP connect failed")
        self.qp = qp
        self._lock = threading.Lock()  # one QP: serialize posters

    def read_rows_at(self, ids: np.ndarray, dst_row: int) -> None:
        """READ len(ids) rows into the pinned buffer starting at row dst_row."""
        ids = np.ascontiguousarray(ids, dtype=np.uint64)
        with self._lock:
            rc = self.L.ple_read_rows(
                self.qp, self.lkey,
                self.rows_ptr + dst_row * self.row_bytes,
                self.remote["addr"], self.remote["rkey"],
                ids.ctypes.data, ids.size, self.row_bytes,
            )
        if rc != 0:
            raise RuntimeError(f"ple rdma: read_rows rc={rc}")


class _Job:
    __slots__ = ("ev", "n", "base", "done", "ok")

    def __init__(self, ev, n, base):
        self.ev, self.n, self.base = ev, n, base
        self.done = threading.Event()
        self.ok = False


class RdmaPleTable:
    """Duck-types MmapPleTable (gather/row_bytes/torch_dtype/rows_total) and
    adds the prefetch pipeline. mmap_table (optional) backs subset/VERIFY."""

    def __init__(self, endpoint, shard_size, row_bytes, torch_dtype, rows_total,
                 mmap_table=None):
        host, port = endpoint.rsplit(":", 1)
        dev = os.environ.get("VLLM_PLE_RDMA_DEV", "roceP2p1s0f0")
        gid = int(os.environ.get("VLLM_PLE_RDMA_GID", "5"))
        self.shard_size = int(shard_size)
        self.row_bytes = int(row_bytes)
        self.torch_dtype = torch_dtype
        self.rows_total = int(rows_total)
        self.mmap_table = mmap_table

        self.rows_t = torch.empty((_NREGIONS * _SLOT, row_bytes),
                                  dtype=torch.uint8, pin_memory=True)
        self.rows_np = self.rows_t.numpy()
        self.ids_t = torch.empty((2 * _SLOT,), dtype=torch.int64, pin_memory=True)
        self.ids_np = self.ids_t.numpy()

        self.client = Client(host, int(port), dev, gid, self.rows_np, self.row_bytes)
        if self.client.remote["row_bytes"] != row_bytes:
            raise RuntimeError("ple rdma: row_bytes mismatch with server")
        self.loaded = np.asarray(sorted(self.client.remote["loaded_shards"]))
        self.full = self.loaded.size * self.shard_size >= self.rows_total
        self.verify = (os.environ.get("VLLM_PLE_RDMA_VERIFY", "0") == "1"
                       and mmap_table is not None)
        if not self.full and mmap_table is None:
            raise RuntimeError("ple rdma: server holds a subset and no mmap fallback")

        self._q: "queue.Queue[_Job]" = queue.Queue()
        self._jobs: list[_Job | None] = [None, None]
        self._pending: _Job | None = None
        self._slot = 0
        threading.Thread(target=self._worker, daemon=True, name="ple-rdma-pf").start()
        if os.environ.get("VLLM_PLE_RDMA_PREFETCH", "1") == "1":
            _patch_runner()
        logger.info(
            "PLE rdma: %s dev=%s gid=%d, %d/%d shards served%s, prefetch=%s",
            endpoint, dev, gid, self.loaded.size,
            -(-self.rows_total // self.shard_size),
            ", VERIFY mode" if self.verify else "",
            os.environ.get("VLLM_PLE_RDMA_PREFETCH", "1"),
        )

    # ---- prefetch pipeline -------------------------------------------------

    def prefetch(self, ids: torch.Tensor) -> None:
        """Called at batch-assembly time with the CUDA ngram_ids [T, heads]."""
        n = ids.numel()
        if n == 0 or n > _SLOT or not self.full:
            return
        _bump("pf_call")
        slot = self._slot
        self._slot ^= 1
        prev = self._jobs[slot]
        if prev is not None and not prev.done.wait(5.0):
            logger.error("PLE rdma: prefetch slot %d stuck, skipping", slot)
            return
        base = slot * _SLOT
        self.ids_t[base:base + n].copy_(ids.reshape(-1), non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        job = _Job(ev, n, base)
        self._jobs[slot] = job
        self._pending = job
        self._q.put(job)

    def _worker(self):
        while True:
            job = self._q.get()
            try:
                job.ev.synchronize()  # ids D2H (and the hash before it) done
                ids = self.ids_np[job.base:job.base + job.n]
                if ids.min() < 0 or ids.max() >= self.rows_total:
                    logger.error("PLE rdma: prefetch ids out of range, dropping")
                else:
                    self.client.read_rows_at(ids, job.base)
                    job.ok = True
            except Exception:
                logger.exception("PLE rdma: prefetch worker error")
            finally:
                job.done.set()

    def consume(self, want_rows: int, device) -> torch.Tensor | None:
        """Return prefetched rows as cuda uint8 [want_rows, row_bytes], or
        None -> caller takes the sync path."""
        job, self._pending = self._pending, None
        if job is None or job.n != want_rows:
            _bump("rdma_pf_miss")
            from vllm_ple_mmap import _STATS as S
            if S.get("rdma_pf_miss", 0) % 100 == 1:
                logger.info(
                    "PLE rdma: consume miss (%s); hook_fires=%d prefetch_calls=%d",
                    "no job" if job is None else f"job.n={job.n} want={want_rows}",
                    S.get("pf_hook", 0), S.get("pf_call", 0),
                )
            return None
        if not job.done.wait(5.0) or not job.ok:
            _bump("rdma_pf_miss")
            return None
        _bump("rdma_pf_hit")
        rows = self.rows_t[job.base:job.base + job.n]
        # NOTE next reuse of this slot is >=1 full step away; the async H2D
        # below is long complete by then (serial runner).
        return rows.to(device, non_blocking=True)

    # ---- sync path (MmapPleTable duck-type; used by fallback + VERIFY) -----

    def gather(self, ids: np.ndarray) -> np.ndarray:
        import time as _time
        from vllm_ple_mmap import _STATS

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
        if ids.min() < 0 or ids.max() >= self.rows_total:
            raise IndexError(f"PLE rdma: row id out of range [{ids.min()}, {ids.max()}]")
        served = self.full
        if not served:
            mask = np.isin(ids // self.shard_size, self.loaded)
            served = bool(mask.all())
        if not served:
            out = self.mmap_table._gather(ids)
            if mask.any():
                rdma_rows = self._read(ids[mask])
                if self.verify and not np.array_equal(rdma_rows, out[mask]):
                    logger.error("PLE rdma VERIFY MISMATCH (mixed batch)")
            return out
        out = self._read(ids)
        if self.verify:
            ref = self.mmap_table._gather(ids)
            if not np.array_equal(out, ref):
                bad = int((out != ref).any(axis=1).sum())
                logger.error("PLE rdma VERIFY MISMATCH: %d/%d rows", bad, ids.size)
                return ref
        return out

    def _read(self, ids: np.ndarray) -> np.ndarray:
        base = 2 * _SLOT  # dedicated sync region, untouched by the worker
        out = np.empty((ids.size, self.row_bytes), dtype=np.uint8)
        for a in range(0, ids.size, _SLOT):
            b = min(a + _SLOT, ids.size)
            self.client.read_rows_at(ids[a:b], base)
            out[a:b] = self.rows_np[base:base + (b - a)]
        return out


def _bump(key):
    from vllm_ple_mmap import _STATS
    _STATS[key] = _STATS.get(key, 0) + 1


# ---- runner hook: hash + post READs one model-forward ahead of use ---------

_PATCHED = False


def _patch_runner():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True
    # NOTE the active runner in this build is vllm.v1.worker.gpu.model_runner,
    # whose ngram kwargs come from the model_state plugin — hook there (the
    # separate vllm/v1/worker/gpu_model_runner.py is NOT what runs).
    from vllm.models.qwen3_8_flash_next.nvidia.model_state import (
        Qwen3_8FlashNextModelState,
    )

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
                _do_prefetch(ids, qsl, ctx)
        except Exception:
            logger.exception("PLE rdma: prefetch hook error (sync fallback)")
        return out

    Qwen3_8FlashNextModelState.prepare_inputs = prepare_inputs
    logger.info(
        "PLE rdma: prefetch hook installed on Qwen3_8FlashNextModelState.prepare_inputs")


def _do_prefetch(input_ids, query_start_loc, ngram_context):
    import vllm_ple_mmap as _pm

    _bump("pf_hook")
    layers = list(_pm._REGISTRY.values())
    if len(layers) != 1:
        return
    layer = layers[0]
    emb = layer.ngram_embedding
    if not hasattr(getattr(emb, "table", None), "prefetch"):
        return
    # Run the stock hash (GPU elementwise on token ids); the flag makes the
    # placeholder embedding start the async fetch and return a dummy.
    _pm._PREFETCH.on = True
    try:
        layer._ple_mmap_orig_forward_impl(None, input_ids, query_start_loc, ngram_context)
    finally:
        _pm._PREFETCH.on = False
