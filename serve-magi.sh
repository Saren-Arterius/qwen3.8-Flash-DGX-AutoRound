#!/usr/bin/env bash
# magi's serving config — magi-v4 branch. Everything lives in v4/recipe.yaml:
# myllmbox recipe v5.2 stack (vLLM 0.30) + never-evict pin + FP8 PLE rows over
# RDMA from wtako's ple_rdma_server (:18515), serving the
# hybrid-mtpdense-g32-mtpint4 checkpoint with MTP=3 + 20 GiB KV as "qwen" in
# container qwen38-flash. Called by ~/magi-stack/ram-client-ple.sh.
# Rollback to magi-v3 (ultrafast stack): git checkout magi-v3 here (its
# serve-magi.sh waits on the same :18515).
set -e
cd "$(dirname "$(readlink -f "$0")")/v4"

# vLLM aborts at boot if the table daemon is unreachable — wait here instead
# so the supervisor's warm timeout isn't burned on a crash loop.
until timeout 2 bash -c '</dev/tcp/192.168.0.1/18515' 2>/dev/null; do
  echo "serve-magi.sh (magi-v4): waiting for ple_rdma_server on wtako:18515..."
  sleep 5
done

exec ./run.sh "$@"
