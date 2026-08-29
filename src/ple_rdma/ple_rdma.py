#!/usr/bin/env python3
"""ctypes wrapper + QP-exchange protocol for libple_rdma.so.

Server (wtako): own the PLE table in RAM as one contiguous MR, hand out
(gid, qpn, rkey, addr, geometry) per TCP client, then idle — all data moves
as one-sided RDMA READs initiated by magi.
Client (magi): read_rows(ids) -> [n, row_bytes] uint8 view.
"""
import ctypes
import json
import mmap
import os
import socket
import struct

import numpy as np

ROW_BYTES = 160
SHARD_ROWS = 2_500_012
TENSOR_MARK = "ngram_embedding.shard_"

_lib = None


def lib():
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
        _lib.ple_mr_addr.restype = ctypes.c_uint64
        _lib.ple_mr_addr.argtypes = [ctypes.c_void_p]
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


def buf_addr(buf) -> int:
    return ctypes.addressof(ctypes.c_char.from_buffer(buf))


def iter_shard_tensors(table_dir):
    """Yield (shard_idx, file_path, abs_data_offset, nbytes) for every PLE shard."""
    for fn in sorted(os.listdir(table_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(table_dir, fn)
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
        for name, meta in header.items():
            if TENSOR_MARK in name and name.endswith(".weight"):
                idx = int(name.split(TENSOR_MARK)[1].split(".")[0])
                start, end = meta["data_offsets"]
                yield idx, path, 8 + hlen + start, end - start


def read_weight_scale(table_dir):
    """Find the global bf16 ngram_embedding.weight_scale in the source files."""
    for fn in sorted(os.listdir(table_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(table_dir, fn)
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(hlen))
            for name, meta in header.items():
                if name.endswith("ngram_embedding.weight_scale"):
                    f.seek(8 + hlen + meta["data_offsets"][0])
                    u16 = struct.unpack("<H", f.read(2))[0]
                    return struct.unpack("<f", struct.pack("<I", u16 << 16))[0]
    return None


def _send_json(sock, obj):
    sock.sendall(json.dumps(obj).encode() + b"\n")


def _recv_json(sock):
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("peer closed during exchange")
        data += chunk
    return json.loads(data)


def serve(table_dir, dev_name, gid_index, port, shards=None, region=None):
    """Load PLE shards into one region, register, serve QP exchanges forever.

    shards: optional iterable of shard indices (subset mode). region: optional
    preloaded mmap (skip load). Rows for unloaded shards read as garbage —
    subset mode is for testing only.
    """
    L = lib()
    want = None if shards is None else set(shards)
    plan = [t for t in iter_shard_tensors(table_dir) if want is None or t[0] in want]
    if not plan:
        raise RuntimeError(f"no PLE shards found under {table_dir}")
    n_shards = (max(t[0] for t in plan) + 1) if want else 128
    length = n_shards * SHARD_ROWS * ROW_BYTES

    if region is None:
        region = mmap.mmap(-1, length)
        view = memoryview(region)
        for i, (idx, path, off, nbytes) in enumerate(sorted(plan)):
            assert nbytes == SHARD_ROWS * ROW_BYTES, (idx, nbytes)
            with open(path, "rb") as f:
                f.seek(off)
                dst = view[idx * nbytes:(idx + 1) * nbytes]
                while len(dst):
                    got = f.readinto(dst[: 64 << 20])
                    dst = dst[got:]
            print(f"[{i + 1}/{len(plan)}] shard {idx} <- {os.path.basename(path)}", flush=True)

    dev = L.ple_dev_open(dev_name.encode(), 1, gid_index)
    assert dev, "dev open failed"
    addr = buf_addr(region)
    mr = L.ple_mr_reg(dev, addr, length, 1)
    assert mr, "mr reg failed (memlock ulimit?)"
    print(f"MR ready: {length >> 20} MiB, rkey={L.ple_mr_rkey(mr):#x}", flush=True)

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(4)
    print(f"listening :{port}", flush=True)
    while True:
        conn, peer = srv.accept()
        qp = L.ple_qp_create(dev)
        gid = ctypes.create_string_buffer(16)
        qpn = ctypes.c_uint32()
        L.ple_qp_local(qp, gid, ctypes.byref(qpn))
        _send_json(conn, {
            "gid": gid.raw.hex(), "qpn": qpn.value,
            "rkey": L.ple_mr_rkey(mr), "addr": L.ple_mr_addr(mr),
            "rows": n_shards * SHARD_ROWS, "row_bytes": ROW_BYTES,
            "loaded_shards": sorted(t[0] for t in plan),
            "weight_scale": read_weight_scale(table_dir),
        })
        try:
            peer_info = _recv_json(conn)
            rc = L.ple_qp_connect(qp, bytes.fromhex(peer_info["gid"]), peer_info["qpn"])
            print(f"client {peer} qp={qpn.value} connect rc={rc}", flush=True)
            while conn.recv(4096):  # idle until client disconnects
                pass
        except (ConnectionError, OSError) as e:
            print(f"client {peer}: {e}", flush=True)
        finally:
            conn.close()
            L.ple_qp_destroy(qp)
            print(f"client {peer} gone", flush=True)


class Client:
    def __init__(self, host, port, dev_name, gid_index, max_rows=1 << 18):
        L = lib()
        self.L = L
        self.dev = L.ple_dev_open(dev_name.encode(), 1, gid_index)
        assert self.dev, "dev open failed"
        self.buf = np.empty((max_rows, ROW_BYTES), dtype=np.uint8)
        self.mr = L.ple_mr_reg(self.dev, self.buf.ctypes.data, self.buf.nbytes, 0)
        assert self.mr, "client mr reg failed"
        self.max_rows = max_rows

        self.sock = socket.create_connection((host, port))
        self.remote = _recv_json(self.sock)
        qp = L.ple_qp_create(self.dev)
        gid = ctypes.create_string_buffer(16)
        qpn = ctypes.c_uint32()
        L.ple_qp_local(qp, gid, ctypes.byref(qpn))
        _send_json(self.sock, {"gid": gid.raw.hex(), "qpn": qpn.value})
        rc = L.ple_qp_connect(qp, bytes.fromhex(self.remote["gid"]), self.remote["qpn"])
        assert rc == 0, "qp connect failed"
        self.qp = qp
        self.lkey = None  # filled below

    def read_rows(self, ids: np.ndarray) -> np.ndarray:
        """ids: int64 [n] global row ids -> uint8 [n, ROW_BYTES] (view into self.buf)."""
        ids = np.ascontiguousarray(ids, dtype=np.uint64).reshape(-1)
        n = ids.size
        assert n <= self.max_rows, n
        rc = self.L.ple_read_rows(
            self.qp, self._lkey(), self.buf.ctypes.data,
            self.remote["addr"], self.remote["rkey"],
            ids.ctypes.data, n, ROW_BYTES,
        )
        if rc != 0:
            raise RuntimeError(f"ple_read_rows rc={rc}")
        return self.buf[:n]

    def _lkey(self):
        if self.lkey is None:
            self.lkey = self.L.ple_mr_lkey(self.mr)
        return self.lkey
