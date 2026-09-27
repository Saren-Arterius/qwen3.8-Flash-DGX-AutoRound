#!/usr/bin/env bash
# magi's serving config — magi-v2 branch. Everything lives in v2/recipe.yaml: bilikaz/qwen38-flash-next-recipe v4
# (hibrid48, vLLM 0.30) + NVFP4 PLE rows over RDMA from wtako's ple-nvfp4-rdma-server (:18516) + MTP=3 + never-evict
# pin + 20 GiB KV, served as "qwen" in container qwen38-flash. Called by ~/magi-stack/ram-client-ple.sh.
# Rollback to v1 (Intel AutoRound + fp8 table): git checkout magi here, revert the :18516 wait in ram-client-ple.sh,
# and on wtako: systemctl disable --now ple-nvfp4-rdma-server && systemctl enable --now ple-rdma-server.
exec "$(dirname "$(readlink -f "$0")")/v2/run.sh"
