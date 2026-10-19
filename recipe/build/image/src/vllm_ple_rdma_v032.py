"""PLE n-gram table over RDMA for vLLM 0.32 (Qwen4Exp) — magi-v3 port.

vLLM 0.32 picks the PLE table backend in Qwen4ExpNGramEmbedding.__init__
(``Qwen4ExpPLEDeviceEmbedding``, or ``Qwen4ExpPLEPinnedHostEmbedding`` when the
auto-enabled Engram cpu_offload is on; both looked up from module globals at
call time). With VLLM_PLE_RDMA=<host:port> this module swaps both for
``Qwen4ExpPLERdmaEmbedding``: no table storage at all, rows come from wtako's
fp8 ple_rdma_server.

VLLM_PLE_RDMA_GRAPH=1 (default): graph-capturable lookup. The GPU copies the
ids into a pinned host buffer, ``ple_g_host_fn`` (libple_rdma_graph.so,
recorded with cuLaunchHostFunc) READs the rows into a pinned MR buffer, and the
GPU copies them out — capturable in FULL and PIECEWISE CUDA graphs. Reconnect
and stall-forever live in the C library.
VLLM_PLE_RDMA_GRAPH=0: the eager-break Python path (RdmaPleTable, unchanged
vllm_ple_rdma.py), which forces PIECEWISE graphs.

The checkpoint is table-stripped GPTQ, for which upstream's
``from_quant_config`` raises; in RDMA mode the fp8 PLE method is forced and its
global scale comes from the server hello.

Applied by appending ``apply()`` to vllm/models/qwen4_exp/nvidia/ngram_embedding.py.
"""
from __future__ import annotations

import ctypes
import json
import logging
import math
import os

import numpy as np
import torch

logger = logging.getLogger("vllm.ple_rdma_v032")
MAX_ROWS = 1 << 18  # 8192 batched tokens x 16 n-gram heads = 131072


def enabled() -> bool:
    return bool(os.environ.get("VLLM_PLE_RDMA"))


class _GraphTable:
    """C-side connection + pinned id/row buffers with CUDA views."""

    def __init__(self, endpoint: str, row_bytes: int, rows_total: int) -> None:
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
        from vllm_ple_rdma import find_gid_index

        host, port = endpoint.rsplit(":", 1)
        dev = os.environ.get("VLLM_PLE_RDMA_DEV", "roceP2p1s0f0")
        gid_env = os.environ.get("VLLM_PLE_RDMA_GID", "auto")
        gid = find_gid_index(dev) if gid_env in ("auto", "") else int(gid_env)
        here = os.path.dirname(os.path.abspath(__file__))
        L = ctypes.CDLL(os.path.join(here, "libple_rdma_graph.so"))
        L.ple_g_new.restype = ctypes.c_void_p
        L.ple_g_new.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int,
                                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long, ctypes.c_int,
                                ctypes.c_uint64]
        L.ple_g_hello.restype = ctypes.c_char_p
        L.ple_g_hello.argtypes = [ctypes.c_void_p]
        L.ple_g_call.restype = ctypes.c_void_p
        L.ple_g_call.argtypes = [ctypes.c_void_p, ctypes.c_long]
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuLaunchHostFunc.restype = ctypes.c_int
        cu.cuLaunchHostFunc.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]

        self.row_bytes, self.rows_total = int(row_bytes), int(rows_total)
        self.rows_h = torch.empty((MAX_ROWS, row_bytes), dtype=torch.uint8, device="cpu",
                                  pin_memory=True)
        self.ids_h = torch.empty((MAX_ROWS,), dtype=torch.int64, device="cpu", pin_memory=True)
        self.g = L.ple_g_new(host.encode(), port.encode(), dev.encode(), gid,
                             self.rows_h.data_ptr(), self.ids_h.data_ptr(), MAX_ROWS,
                             self.row_bytes, self.rows_total)
        if not self.g:
            raise RuntimeError(f"PLE rdma graph: cannot open {dev} / register MR")
        self.remote = json.loads(L.ple_g_hello(self.g))
        self.rows_d = get_accelerator_view_from_cpu_tensor(self.rows_h)
        self.ids_d = get_accelerator_view_from_cpu_tensor(self.ids_h)
        self.L, self.cu = L, cu
        self.fn = ctypes.cast(L.ple_g_host_fn, ctypes.c_void_p)
        self.calls: dict[int, int] = {}
        logger.info("PLE rdma graph: %s dev=%s gid=%d, host-func lookup (capturable)",
                    endpoint, dev, gid)

    def lookup(self, ids: torch.Tensor, out_u8: torch.Tensor) -> None:
        """ids: int64 [n] on GPU; out_u8: uint8 [n, row_bytes] on GPU. Stream-ordered."""
        n = ids.numel()
        if n > MAX_ROWS:
            raise ValueError(f"PLE rdma graph: {n} rows > {MAX_ROWS}")
        call = self.calls.get(n)
        if call is None:
            call = self.calls[n] = self.L.ple_g_call(self.g, n)
        self.ids_d[:n].copy_(ids, non_blocking=True)
        rc = self.cu.cuLaunchHostFunc(torch.cuda.current_stream().cuda_stream, self.fn, call)
        if rc:
            raise RuntimeError(f"PLE rdma graph: cuLaunchHostFunc rc={rc}")
        out_u8.copy_(self.rows_d[:n], non_blocking=True)


def apply(ngram_module) -> None:
    """Patch the 0.32 Qwen4Exp n-gram embedding module (pass the module)."""
    if not enabled() or getattr(ngram_module, "_ple_rdma_v032", False):
        return
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    from vllm.models.qwen4_exp.common import ngram_embedding as common

    graph_mode = os.environ.get("VLLM_PLE_RDMA_GRAPH", "1") != "0"

    class Qwen4ExpPLERdmaEmbedding(common.Qwen4ExpPLEEmbedding):
        """PLE rows over RDMA; holds no table."""

        def __init__(self, num_embeddings, embedding_dim, **kw) -> None:
            super().__init__(num_embeddings, embedding_dim, **kw)
            if self.tp_size != 1:
                raise RuntimeError("PLE rdma: only ETP=1 is supported")
            vocab = int(self.org_vocab_size)
            parts = int(os.environ.get("VLLM_PLE_RDMA_SPLIT_PARTS", "128"))
            ep = os.environ["VLLM_PLE_RDMA"]
            with torch.device("cpu"):  # model init runs under a cuda default device
                if graph_mode:
                    self.table = _GraphTable(ep, int(self.embedding_dim), vocab)
                else:
                    from vllm_ple_rdma import RdmaPleTable
                    self.table = RdmaPleTable(ep, math.ceil(vocab / parts),
                                              int(self.embedding_dim), torch.float8_e4m3fn, vocab)
            remote = self.table.remote if graph_mode else self.table.client.remote
            shards = remote.get("loaded_shards")
            if shards is not None and len(shards) * math.ceil(vocab / parts) < vocab:
                raise RuntimeError("PLE rdma: server holds a subset of the table")
            scale = remote.get("weight_scale")
            if scale is None:
                raise RuntimeError("PLE rdma: server hello has no weight_scale")
            self.weight_scale.data.fill_(float(scale))
            self._pinned = None
            logger.info("PLE rdma: %s rows x %d B over RDMA, scale %.4e, graph=%s",
                        vocab, int(self.embedding_dim), float(scale), graph_mode)

        def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
            return torch.empty(0, embedding_dim, dtype=dtype)  # table lives on wtako

        def start_prefetch(self, ngram_ids):
            return None

        def finalize_prefetch(self, total_tokens):
            raise RuntimeError("RDMA PLE embedding does not support prefetch")

        def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            out = torch.empty((*ngram_ids.shape, self.embedding_dim),
                              dtype=torch.float8_e4m3fn, device=ngram_ids.device)
            if graph_mode:
                if ngram_ids.numel():
                    self.table.lookup(ngram_ids.reshape(-1).long(),
                                      out.view(torch.uint8).reshape(-1, self.embedding_dim))
            else:
                self._lookup(ngram_ids, out)
            return out

        @eager_break_during_capture
        def _lookup(self, ngram_ids: torch.Tensor, out: torch.Tensor) -> None:
            ids = ngram_ids.reshape(-1).cpu().numpy()
            if ids.size == 0:
                return
            # Out-of-range ids (dummy/uninitialised warm-up and capture inputs)
            # become zero rows, as the upstream pinned-host kernel does.
            bad = (ids < 0) | (ids >= self.table.rows_total)
            if bad.any():
                ids = np.where(bad, 0, ids)
            # Dedup on CPU, read unique rows, expand on the GPU (as the
            # magi-v3 mmap path does: repeated n-grams are the common case).
            uniq, inverse = np.unique(ids, return_inverse=True)
            rows = self.table.gather(uniq)
            if bad.any():
                rows = rows[inverse].copy()
                rows[bad] = 0
                inverse = np.arange(ids.size)
            u = rows.shape[0]
            buf = self._pinned
            if buf is None or buf.shape[0] < u:
                buf = torch.empty((max(u + u // 2, 4096), rows.shape[1]),
                                  dtype=torch.uint8, device="cpu", pin_memory=True)
                self._pinned = buf
            buf[:u].numpy()[:] = rows
            dev = buf[:u].to(out.device, non_blocking=True)
            inv = torch.from_numpy(inverse.reshape(-1)).to(out.device, non_blocking=True)
            out.view(torch.uint8).reshape(-1, rows.shape[1]).copy_(dev[inv])
            # ponytail: sync so the pinned buffer can't be overwritten by the
            # next call before this H2D lands; drop with a double buffer if it shows.
            torch.cuda.current_stream().synchronize()

    orig_from_quant_config = common.Qwen4ExpPLEEmbeddingMethod.from_quant_config

    def from_quant_config(quant_config, prefix, embedding_dtype=None):
        if prefix.endswith("ngram_embedding"):
            return common.Qwen4ExpPLEFp8EmbeddingMethod()  # rows are fp8 on the server
        return orig_from_quant_config(quant_config, prefix, embedding_dtype)

    common.Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(from_quant_config)
    ngram_module.Qwen4ExpPLEDeviceEmbedding = Qwen4ExpPLERdmaEmbedding
    # 0.32 auto-enables EngramConfig(cpu_offload=True) for Qwen4Exp, which picks the
    # pinned-host class instead: it would pin the whole table in host RAM (OOM on GB10).
    ngram_module.Qwen4ExpPLEPinnedHostEmbedding = Qwen4ExpPLERdmaEmbedding
    ngram_module._ple_rdma_v032 = True
    logger.info("PLE rdma (0.32) patch applied to %s (graph=%s)", ngram_module.__name__, graph_mode)
