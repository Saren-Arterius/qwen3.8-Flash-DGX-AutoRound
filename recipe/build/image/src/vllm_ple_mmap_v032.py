# SPDX-License-Identifier: Apache-2.0
"""PLE n-gram table from local SSD via mmap + CUDA host func (vLLM 0.32) — magi-v5-public.

Same graph-capturable pattern as vllm_ple_rdma_v032.py, but rows come from
local safetensors shards (~/ple-table-fp8: layer 1, 128 x 2500012 x 160 B)
memcpy'd from mmap'd files in a C host function (ple_mmap_graph.c,
cuLaunchHostFunc) instead of RDMA READs. No Python on the hot path, no
.cpu() during capture — fully capturable in FULL_AND_PIECEWISE.

Selection: VLLM_PLE_RDMA wins when set (RDMA adapter runs first).
Otherwise, when VLLM_PLE_MMAP_DIR points at the table dir, this swaps
both Qwen4Exp backends for Qwen4ExpPLEMmapEmbedding.

Env:
  VLLM_PLE_MMAP_DIR  table dir (e.g. /ple-table-fp8). Unset = dormant.
  VLLM_PLE_MMAP_PREWARM=1  populate page cache at load (madvise/touch).
"""
from __future__ import annotations

import ctypes
import json
import logging
import os
import re

import torch

logger = logging.getLogger("vllm.ple_mmap_v032")

MAX_ROWS = 1 << 18


def enabled() -> bool:
    if os.environ.get("VLLM_PLE_RDMA"):
        return False
    d = os.environ.get("VLLM_PLE_MMAP_DIR")
    return bool(d) and os.path.isdir(d)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


class _MmapGraphTable:
    """C-side mmap + pinned id/row buffers with CUDA views."""

    def __init__(self, table_dir: str, layer_idx: int, row_bytes: int) -> None:
        import sys as _sys  # noqa: F401  (keep import local: image COPY time has no deps)
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from vllm_ple_mmap import (
                _find_shards,
                _read_scale,
                _TABLE_DTYPES,
                _FP8_DTYPES,
            )
        finally:
            _sys.path.remove(os.path.dirname(os.path.abspath(__file__)))

        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        shards, dtype_str, scale_entry = _find_shards(table_dir, layer_idx)
        if not shards:
            raise RuntimeError(
                f"PLE mmap: no shard tensors for layer {layer_idx} under {table_dir}"
            )
        cols = shards.pop("__cols__")
        if dtype_str not in _TABLE_DTYPES:
            raise RuntimeError(f"PLE mmap: unsupported shard dtype {dtype_str}")
        if dtype_str not in _FP8_DTYPES:
            raise RuntimeError(
                f"PLE mmap (public): only FP8 tables supported, got {dtype_str}"
            )
        idxs = sorted(shards)
        if idxs != list(range(len(idxs))):
            raise RuntimeError(f"PLE mmap: non-contiguous shards {idxs[:8]}...")
        nshards = len(idxs)
        rows_list = [shards[i][2] for i in idxs]
        shard_size = max(rows_list)
        rows_total = sum(rows_list)
        if scale_entry is None:
            raise RuntimeError("PLE mmap: FP8 shards without weight_scale")
        scale = float(_read_scale(scale_entry))

        here = os.path.dirname(os.path.abspath(__file__))
        libpath = os.path.join(here, "libple_mmap_graph.so")
        L = ctypes.CDLL(libpath)
        L.ple_mg_new.restype = ctypes.c_void_p
        L.ple_mg_new.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int,
            ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_double, ctypes.c_int,
        ]
        L.ple_mg_hello.restype = ctypes.c_char_p
        L.ple_mg_hello.argtypes = [ctypes.c_void_p]
        L.ple_mg_call.restype = ctypes.c_void_p
        L.ple_mg_call.argtypes = [ctypes.c_void_p, ctypes.c_long]
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuLaunchHostFunc.restype = ctypes.c_int
        cu.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self.row_bytes, self.rows_total = int(row_bytes), int(rows_total)
        self.rows_h = torch.empty((MAX_ROWS, row_bytes), dtype=torch.uint8,
                                  device="cpu", pin_memory=True)
        self.ids_h = torch.empty((MAX_ROWS,), dtype=torch.int64,
                                 device="cpu", pin_memory=True)
        paths = (ctypes.c_char_p * nshards)(
            *[shards[i][0].encode() for i in idxs])
        offsets = (ctypes.c_int64 * nshards)(
            *[shards[i][1] for i in idxs])
        prewarm = _env_int("VLLM_PLE_MMAP_PREWARM", 0)
        self.g = L.ple_mg_new(
            ctypes.c_void_p(self.rows_h.data_ptr()),
            ctypes.c_void_p(self.ids_h.data_ptr()),
            MAX_ROWS, int(row_bytes), int(rows_total),
            int(shard_size), int(nshards), paths, offsets,
            float(scale), int(prewarm))
        if not self.g:
            raise RuntimeError("PLE mmap graph: ple_mg_new failed (open/mmap?)")
        self.remote = json.loads(L.ple_mg_hello(self.g))
        self.rows_d = get_accelerator_view_from_cpu_tensor(self.rows_h)
        self.ids_d = get_accelerator_view_from_cpu_tensor(self.ids_h)
        self.L, self.cu = L, cu
        self.fn = ctypes.cast(L.ple_mg_host_fn, ctypes.c_void_p)
        self.calls: dict[int, int] = {}
        self.shard_size = int(shard_size)
        logger.info("PLE mmap graph: %s layer %d, %d shards x %d rows, scale %.4e",
                    table_dir, layer_idx, nshards, shard_size, scale)

    def lookup(self, ids: torch.Tensor, out_u8: torch.Tensor) -> None:
        n = ids.numel()
        if n > MAX_ROWS:
            raise ValueError(f"PLE mmap graph: {n} rows > {MAX_ROWS}")
        call = self.calls.get(n)
        if call is None:
            call = self.calls[n] = self.L.ple_mg_call(self.g, n)
        self.ids_d[:n].copy_(ids, non_blocking=True)
        rc = self.cu.cuLaunchHostFunc(torch.cuda.current_stream().cuda_stream,
                                      self.fn, call)
        if rc:
            raise RuntimeError(f"PLE mmap graph: cuLaunchHostFunc rc={rc}")
        out_u8.copy_(self.rows_d[:n], non_blocking=True)


def apply(ngram_module) -> None:
    """Patch the 0.32 Qwen4Exp n-gram embedding module (pass the module)."""
    if not enabled() or getattr(ngram_module, "_ple_mmap_v032", False):
        return
    from vllm.models.qwen4_exp.common import ngram_embedding as common

    class Qwen4ExpPLEMmapEmbedding(common.Qwen4ExpPLEEmbedding):
        """PLE rows from local SSD mmap; holds no table weight."""

        def __init__(self, num_embeddings, embedding_dim, **kw) -> None:
            prefix = str(kw.get("prefix", ""))
            m = re.search(r"layers\.(\d+)\.", prefix)
            if not m:
                raise RuntimeError(
                    f"PLE mmap: cannot find layer index in {prefix!r}")
            layer_idx = int(m.group(1))
            super().__init__(num_embeddings, embedding_dim, **kw)
            if self.tp_size != 1:
                raise RuntimeError("PLE mmap: only ETP=1 is supported")
            table_dir = os.environ["VLLM_PLE_MMAP_DIR"]
            with torch.device("cpu"):
                self.table = _MmapGraphTable(
                    table_dir, layer_idx, int(self.embedding_dim))
            scale = self.table.remote.get("weight_scale")
            if scale is None:
                raise RuntimeError("PLE mmap: no weight_scale in table hello")
            self.weight_scale.data.fill_(float(scale))
            logger.info("PLE mmap: %d rows x %d B local SSD, scale %.4e",
                        self.table.rows_total, int(self.embedding_dim),
                        float(scale))

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            return torch.empty(0, embedding_dim, dtype=dtype)

        def start_prefetch(self, ngram_ids):
            return None

        def finalize_prefetch(self, total_tokens):
            raise RuntimeError("mmap PLE embedding does not support prefetch")

        def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            out = torch.empty((*ngram_ids.shape, self.embedding_dim),
                              dtype=torch.float8_e4m3fn, device=ngram_ids.device)
            if ngram_ids.numel():
                self.table.lookup(ngram_ids.reshape(-1).long(),
                                  out.view(torch.uint8).reshape(-1, self.embedding_dim))
            return out

    orig_from_quant_config = common.Qwen4ExpPLEEmbeddingMethod.from_quant_config

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        if prefix.endswith("ngram_embedding"):
            return common.Qwen4ExpPLEFp8EmbeddingMethod()
        return orig_from_quant_config(quant_config, prefix, embedding_dtype)

    common.Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(
        from_quant_config)
    ngram_module.Qwen4ExpPLEDeviceEmbedding = Qwen4ExpPLEMmapEmbedding
    ngram_module.Qwen4ExpPLEPinnedHostEmbedding = Qwen4ExpPLEMmapEmbedding
    ngram_module._ple_mmap_v032 = True
    logger.info("PLE mmap (0.32) patch applied to %s", ngram_module.__name__)
