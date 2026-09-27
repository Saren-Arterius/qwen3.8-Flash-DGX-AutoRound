#!/usr/bin/env python3
"""NVFP4 PLE-table RDMA server (B1 test; prod fp8 server is ple_rdma_server.py).

Loads the hibrid48 `ple-nvfp4-*.safetensors` shards into one RAM region as
interleaved 90-byte rows (80 packed nibble bytes | 10 fp8-e4m3 scales), so a
row is one RDMA READ. Reuses ple_rdma's verbs lib + QP exchange unchanged.

usage: ple_nvfp4_rdma_server.py <table_dir> [--dev rocep1s0f0] [--gid N] [--port 18516]
"""
import argparse
import ctypes
import json
import mmap
import os
import re
import socket
import struct
import threading

import numpy as np

import ple_rdma as R

MARK = re.compile(r"ngram_embedding\.nvfp4_shard_(\d+)\.(packed|scales)$")


def scan(table_dir):
    """-> {shard: {kind: (path, abs_offset, rows, cols)}}, nvfp4_global float."""
    shards, glob = {}, None
    for fn in sorted(os.listdir(table_dir)):
        if not fn.endswith(".safetensors"):
            continue
        path = os.path.join(table_dir, fn)
        with open(path, "rb") as f:
            hlen = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(hlen))
            for name, m in hdr.items():
                if name == "__metadata__":
                    continue
                off = 8 + hlen + m["data_offsets"][0]
                if name.endswith("ngram_embedding.nvfp4_global"):
                    assert m["dtype"] == "F32", m
                    f.seek(off)
                    glob = struct.unpack("<f", f.read(4))[0]
                mm = MARK.search(name)
                if mm:
                    rows, cols = m["shape"]
                    shards.setdefault(int(mm.group(1)), {})[mm.group(2)] = (path, off, rows, cols)
    return shards, glob


def load(shards):
    ks = sorted(shards)
    assert ks == list(range(len(ks))), f"shard gap: {ks}"
    S = shards[0]["packed"][2]
    cp, cs = shards[0]["packed"][3], shards[0]["scales"][3]
    for k in ks:  # geometry must be uniform: row id = k * S + r
        assert shards[k]["packed"][2:] == (S, cp) and shards[k]["scales"][2:] == (S, cs), (k, shards[k])
    rb = cp + cs
    region = mmap.mmap(-1, len(ks) * S * rb)
    tbl = np.frombuffer(region, dtype=np.uint8).reshape(len(ks) * S, rb)
    CH = 1 << 20
    for k in ks:
        for kind, c0, c in (("packed", 0, cp), ("scales", cp, cs)):
            path, off, _, _ = shards[k][kind]
            with open(path, "rb") as f:
                for a in range(0, S, CH):
                    n = min(CH, S - a)
                    f.seek(off + a * c)
                    tbl[k * S + a:k * S + a + n, c0:c0 + c] = np.frombuffer(f.read(n * c), np.uint8).reshape(n, c)
        print(f"[{k + 1}/{len(ks)}] shard {k} loaded", flush=True)
    return region, len(ks), S, cp, cs


def serve(table_dir, dev_name, gid_index, port):
    L = R.lib()
    shards, glob = scan(table_dir)
    assert glob is not None, "nvfp4_global not found"
    region, ns, S, cp, cs = load(shards)
    if gid_index is None:
        gid_index = R.find_gid_index(dev_name)
    dev = L.ple_dev_open(dev_name.encode(), 1, gid_index)
    assert dev, "dev open failed"
    mr = L.ple_mr_reg(dev, R.buf_addr(region), len(region), 1)
    assert mr, "mr reg failed (memlock ulimit?)"
    hello = {
        "rkey": L.ple_mr_rkey(mr), "addr": L.ple_mr_addr(mr),
        "format": "nvfp4", "rows": ns * S, "shard_rows": S, "row_bytes": cp + cs,
        "packed_bytes": cp, "scales_bytes": cs, "nvfp4_global": glob,
        "loaded_shards": list(range(ns)),
    }
    print(f"MR ready: {len(region) >> 20} MiB, {ns}x{S} rows of {cp}+{cs} B, global {glob:.4e}", flush=True)

    def handle(conn, peer):  # same lifecycle as ple_rdma.serve's handler
        qp = None
        try:
            R.keepalive(conn)
            conn.settimeout(30)
            qp = L.ple_qp_create(dev)
            gid, qpn = ctypes.create_string_buffer(16), ctypes.c_uint32()
            L.ple_qp_local(qp, gid, ctypes.byref(qpn))
            R._send_json(conn, dict(hello, gid=gid.raw.hex(), qpn=qpn.value))
            p = R._recv_json(conn)
            rc = L.ple_qp_connect(qp, bytes.fromhex(p["gid"]), p["qpn"])
            print(f"client {peer} qp={qpn.value} connect rc={rc}", flush=True)
            conn.settimeout(None)
            while conn.recv(4096):
                pass
        except (ConnectionError, OSError, ValueError, KeyError) as e:
            print(f"client {peer}: {e}", flush=True)
        finally:
            conn.close()
            if qp:
                L.ple_qp_destroy(qp)
            print(f"client {peer} gone", flush=True)

    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(16)
    print(f"listening :{port}", flush=True)
    while True:
        c, peer = srv.accept()
        threading.Thread(target=handle, args=(c, peer), daemon=True).start()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("table_dir")
    ap.add_argument("--dev", default="rocep1s0f0")
    ap.add_argument("--gid", type=int, default=None)
    ap.add_argument("--port", type=int, default=18516)
    a = ap.parse_args()
    serve(a.table_dir, a.dev, a.gid, a.port)
