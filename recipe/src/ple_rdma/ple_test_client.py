#!/usr/bin/env python3
"""Phase-1 gate: random-row RDMA reads vs the safetensors source over NFS.
Usage: ple_test_client.py <server_host> <nfs_table_dir> [--dev roceP2p1s0f0] [--gid 5]
"""
import argparse
import time

import numpy as np

import ple_rdma

p = argparse.ArgumentParser()
p.add_argument("host")
p.add_argument("nfs_dir")
p.add_argument("--dev", default="roceP2p1s0f0")
p.add_argument("--gid", type=int, default=5)
p.add_argument("--port", type=int, default=18515)
a = p.parse_args()

cli = ple_rdma.Client(a.host, a.port, a.dev, a.gid)
loaded = cli.remote["loaded_shards"]
print(f"server: {cli.remote['rows']} rows, shards loaded: {loaded}")

src = {idx: (path, off) for idx, path, off, _ in ple_rdma.iter_shard_tensors(a.nfs_dir)}
rng = np.random.default_rng(7)

# correctness: 20 batches of 64 random rows from loaded shards, byte-compare
bad = 0
for it in range(20):
    sh = rng.choice(loaded, size=64)
    local = rng.integers(0, ple_rdma.SHARD_ROWS, size=64)
    ids = sh.astype(np.uint64) * ple_rdma.SHARD_ROWS + local
    got = cli.read_rows(ids).copy()
    for k in range(64):
        path, off = src[int(sh[k])]
        with open(path, "rb") as f:
            f.seek(off + int(local[k]) * ple_rdma.ROW_BYTES)
            ref = f.read(ple_rdma.ROW_BYTES)
        if got[k].tobytes() != ref:
            bad += 1
            print(f"MISMATCH it={it} shard={sh[k]} row={local[k]}")
print(f"correctness: {20 * 64 - bad}/{20 * 64} rows bit-exact")

# latency: 64-row batches (decode shape), then 8192-row (prefill-ish)
for n, iters in ((64, 2000), (8192, 50)):
    ts = []
    for _ in range(iters):
        sh = rng.choice(loaded, size=n)
        ids = sh.astype(np.uint64) * ple_rdma.SHARD_ROWS + rng.integers(
            0, ple_rdma.SHARD_ROWS, size=n)
        t0 = time.perf_counter()
        cli.read_rows(ids)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts = np.array(ts)
    print(f"batch={n}: p50 {np.percentile(ts, 50):.0f}us  p99 "
          f"{np.percentile(ts, 99):.0f}us  max {ts.max():.0f}us")

assert bad == 0, "bit-exactness FAILED"
print("PHASE 1 GATE: PASS")
