"""FP8 PLE rows over RDMA — sync client (magi-v4).

Same wire protocol as magi-v3's vllm_ple_rdma.py (fp8 ple_rdma_server,
160-byte rows, hello with addr/rkey/row_bytes/loaded_shards/weight_scale),
minus the prefetch pipeline (sync gather only; prefetch is a TODO).

No fallback: every connect and READ retries forever (2 s), the engine
stalls loudly until it succeeds. Shared libple_rdma.so verbs bindings.
"""
import ctypes
import json
import logging
import os
import socket
import threading
import time

import numpy as np
import torch

logger = logging.getLogger("vllm.ple_rdma_fp8")

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
        _lib.ple_qp_destroy.argtypes = [ctypes.c_void_p]
    return _lib


def _keepalive(sock, idle=5, interval=5, count=3):
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for opt, val in (("TCP_KEEPIDLE", idle), ("TCP_KEEPINTVL", interval), ("TCP_KEEPCNT", count)):
        if hasattr(socket, opt):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), val)


def find_gid_index(dev_name):
    """First IPv4-mapped RoCE v2 GID on port 1 — the index drifts across
    reboots (interface bring-up order), so never hardcode it."""
    base = f"/sys/class/infiniband/{dev_name}/ports/1"
    for i in range(16):
        try:
            typ = open(f"{base}/gid_attrs/types/{i}").read().strip()
            gid = open(f"{base}/gids/{i}").read().strip()
        except OSError:
            continue
        if typ == "RoCE v2" and gid.startswith("0000:0000:0000:0000:0000:ffff:"):
            return i
    raise RuntimeError(f"no IPv4 RoCE v2 gid on {dev_name}")


def _recv_json(sock):
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("ple rdma server closed during exchange")
        data += chunk
    return json.loads(data)


class Client:
    """One RC QP to the fp8 daemon over a shared device/MR."""

    def __init__(self, host, port, dev, lkey, rows_ptr: int, row_bytes: int):
        L = _load_lib()
        self.L = L
        self.dev, self.lkey, self.rows_ptr, self.row_bytes = dev, lkey, rows_ptr, row_bytes
        self.qp = None
        self._lock = threading.Lock()
        self.sock = socket.create_connection((host, port), timeout=15)
        try:
            _keepalive(self.sock)
            self.remote = _recv_json(self.sock)
            qp = L.ple_qp_create(dev)
            if not qp:
                raise RuntimeError("ple rdma: QP create failed")
            self.qp = qp
            gid = ctypes.create_string_buffer(16)
            qpn = ctypes.c_uint32()
            L.ple_qp_local(qp, gid, ctypes.byref(qpn))
            self.sock.sendall(json.dumps({"gid": gid.raw.hex(), "qpn": qpn.value}).encode() + b"\n")
            if L.ple_qp_connect(qp, bytes.fromhex(self.remote["gid"]), self.remote["qpn"]) != 0:
                raise RuntimeError("ple rdma: QP connect failed")
            self.sock.settimeout(None)
        except Exception:
            self.close()
            raise

    def close(self):
        with self._lock:
            try:
                self.sock.close()
            except OSError:
                pass
            if self.qp is not None:
                self.L.ple_qp_destroy(self.qp)
                self.qp = None

    def read_rows_at(self, ids: np.ndarray, dst_row: int) -> None:
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


class Fp8RdmaTable:
    """Sync row source for the mbx fp8 embedding path.

    gather(ids) -> fp8 rows on the caller's device. No local table, no
    fallback: connects and READs retry forever, stalling loudly.
    """

    MAX = 1 << 18

    def __init__(self, endpoint, row_bytes, rows_total):
        host, port = endpoint.rsplit(":", 1)
        dev = os.environ.get("VLLM_PLE_RDMA_DEV", "roceP2p1s0f0")
        gid_env = os.environ.get("VLLM_PLE_RDMA_GID", "auto")
        gid = find_gid_index(dev) if gid_env in ("auto", "") else int(gid_env)
        self.row_bytes = int(row_bytes)
        self.rows_total = int(rows_total)
        self.host, self.port = host, int(port)

        self.rows_t = torch.empty((self.MAX, self.row_bytes),
                                dtype=torch.uint8, pin_memory=True)
        self.rows_np = self.rows_t.numpy()

        L = _load_lib()
        self.dev = L.ple_dev_open(dev.encode(), 1, gid)
        if not self.dev:
            raise RuntimeError(f"ple rdma: cannot open device {dev}")
        self.mr = L.ple_mr_reg(self.dev, self.rows_np.ctypes.data, self.rows_np.nbytes, 0)
        if not self.mr:
            raise RuntimeError("ple rdma: local MR registration failed (memlock ulimit?)")
        self.lkey = L.ple_mr_lkey(self.mr)
        self._rlock = threading.Lock()
        self.client = self._connect_forever()
        logger.info("PLE fp8 rdma: %s dev=%s gid=%d, %d rows x %d B, scale %s",
                    endpoint, dev, gid, self.rows_total, self.row_bytes,
                    self.client.remote.get("weight_scale"))

    def _connect_forever(self) -> Client:
        attempt = 0
        while True:
            try:
                c = Client(self.host, self.port, self.dev, self.lkey,
                           self.rows_np.ctypes.data, self.row_bytes)
                r = c.remote
                if r.get("row_bytes") != self.row_bytes:
                    raise RuntimeError(f"row_bytes mismatch: server {r.get('row_bytes')} != {self.row_bytes}")
                if attempt:
                    logger.warning("PLE fp8 rdma: connected after %d retries", attempt)
                return c
            except Exception as e:
                attempt += 1
                logger.error("PLE fp8 rdma: connect to %s:%d failed (%s) — retry %d in 2s, stalling",
                             self.host, self.port, e, attempt)
                time.sleep(2.0)

    def _robust_read(self, ids: np.ndarray) -> np.ndarray:
        ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
        if ids.size == 0:
            return np.empty((0, self.row_bytes), dtype=np.uint8)
        if ids.min() < 0 or ids.max() >= self.rows_total:
            raise IndexError(f"PLE fp8 rdma: row id out of range [{ids.min()}, {ids.max()}]")
        while True:
            try:
                with self._rlock:
                    client = self.client
                client.read_rows_at(ids, 0)
                return np.array(self.rows_np[:ids.size], copy=True)
            except Exception as e:
                logger.error("PLE fp8 rdma: READ failed (%s) — reconnect in 2s, stalling", e)
                time.sleep(2.0)
                with self._rlock:
                    try:
                        client.close()
                    except OSError:
                        pass
                    self.client = self._connect_forever()

    def gather(self, ids: np.ndarray) -> np.ndarray:
        return self._robust_read(ids)
