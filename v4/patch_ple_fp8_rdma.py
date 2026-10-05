# --- magi-v4: FP8 PLE rows over RDMA, appended to ngram_embedding.py ---
# MBX_PLE_RDMA_FP8=host:port: fp8 n-gram rows via one-sided RDMA READs from the
# fp8 ple_rdma_server (wtako :18515) — the same server/rows as magi-v3
# production (160-byte fp8 rows). Selection happens in from_quant_config, so
# the graph-split points are unchanged: same pieces as the DiskMethod path.
#
# Design mirrors _MbxPLEFp8DiskMethod (zero-resident weight, sync eager
# gather), with the shard files replaced by the RDMA table:
#   - create_weights registers a (0, D) fp8 stand-in — nothing resident, the
#     loader has no table tensors to fill (this checkpoint family carries
#     neither table shards nor a weight_scale tensor);
#   - process_weights_after_loading takes the global fp8 scale from the
#     server hello (like magi-v3) instead of the shard files;
#   - embedding() READs rows over the shared client (sync path; no prefetch).
# No fallback: failed connect/READs retry forever, the engine stalls loudly.
import os as _fp8_os


def _fp8_rdma_endpoint() -> str:
    return _fp8_os.environ.get("MBX_PLE_RDMA_FP8", "")


class _MbxPLEFp8RdmaMethod(_MbxPLEFp8DiskMethod):
    """fp8 rows over RDMA; same shapes as the disk method, no local files."""

    def create_weights(self, layer, input_size_per_partition, output_partition_sizes,
                       input_size, output_size, params_dtype, **extra_weight_attrs):
        del input_size, output_size, extra_weight_attrs
        n = int(sum(output_partition_sizes))
        D = int(input_size_per_partition)
        layer.register_parameter("weight", nn.Parameter(torch.empty((0, D), dtype=torch.float8_e4m3fn),
                                                        requires_grad=False))
        layer.register_parameter("weight_scale", nn.Parameter(torch.ones((), dtype=torch.float32),
                                                              requires_grad=False))
        layer._mbx_mmap = True
        layer._mbx_mmap_rows = n
        layer._mbx_fp8_D = D
        layer._mbx_fp8_rdma = True
        layer._mbx_out_dtype = params_dtype
        print(f"MBX fp8-rdma: partition {n} rows x {D}, rows from {_fp8_rdma_endpoint()} (0 allocated)",
              flush=True)

    def process_weights_after_loading(self, layer) -> None:
        import time as _time
        t0 = _time.time()
        ep = _fp8_rdma_endpoint()
        n = int(layer._mbx_mmap_rows)
        D = int(layer._mbx_fp8_D)
        from vllm.models.qwen4_exp.nvidia.vllm_ple_rdma_fp8 import Fp8RdmaTable
        table = Fp8RdmaTable(ep, D, n)
        scale = table.client.remote.get("weight_scale")
        if scale is None:
            raise RuntimeError("MBX fp8-rdma: no weight_scale in server hello")
        layer.weight_scale.data.fill_(float(scale))
        layer._mbx_mm_table = table
        layer._mbx_mm_S = n
        layer._mbx_mm_n = 1
        layer._mbx_mm_cols_p = D
        layer._mbx_mm_cols_s = 0
        print(f"PLE: fp8 table over RDMA from {ep} ({n} rows x {D}, scale {float(scale):.4e}, "
              f"{_time.time() - t0:.1f}s)", flush=True)

    def embedding(self, layer, input_: torch.Tensor) -> torch.Tensor:
        shape = input_.shape
        import numpy as _np
        ids = input_.reshape(-1).to("cpu", dtype=torch.int64).numpy()
        rows = layer._mbx_mm_table.gather(_np.ascontiguousarray(ids))
        out = torch.from_numpy(rows).to(input_.device).view(torch.float8_e4m3fn)
        return out.view(*shape, -1)


_orig_fp8_from_quant_config = Qwen4ExpPLEEmbeddingMethod.from_quant_config


def _fp8_rdma_from_quant_config(quant_config, prefix, embedding_dtype=None):
    if _fp8_rdma_endpoint():
        # RDMA is exclusive (like magi-v3): whatever the checkpoint's PLE
        # quant resolves to, rows come from the server. Unconditional here so
        # a checkpoint without table/scale tensors can never fall through to
        # a resident-weight path and OOM on a 47 GiB alloc.
        print(f"MBX fp8-rdma: PLE method -> _MbxPLEFp8RdmaMethod ({prefix})", flush=True)
        return _MbxPLEFp8RdmaMethod()
    return _orig_fp8_from_quant_config(quant_config, prefix, embedding_dtype)


Qwen4ExpPLEEmbeddingMethod.from_quant_config = staticmethod(_fp8_rdma_from_quant_config)
print("MBX fp8-rdma hook installed (MBX_PLE_RDMA_FP8)", flush=True)
