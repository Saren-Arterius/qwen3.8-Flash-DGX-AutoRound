#!/usr/bin/env python3
"""wtako PLE-table RDMA server. Usage:
  ple_rdma_server.py <table_dir> [--shards 0-3] [--dev rocep1s0f0] [--gid 3] [--port 18515]
Subset mode (--shards) is for testing; production loads all 128.
"""
import argparse

import ple_rdma

p = argparse.ArgumentParser()
p.add_argument("table_dir")
p.add_argument("--shards", default=None, help="e.g. 0-3 (test subset)")
p.add_argument("--dev", default="rocep1s0f0")
p.add_argument("--gid", type=int, default=3)
p.add_argument("--port", type=int, default=18515)
a = p.parse_args()

shards = None
if a.shards:
    lo, hi = (a.shards.split("-") + [a.shards])[:2]
    shards = range(int(lo), int(hi) + 1)

ple_rdma.serve(a.table_dir, a.dev, a.gid, a.port, shards=shards)
