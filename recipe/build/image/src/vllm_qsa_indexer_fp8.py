"""Block-FP8 the QSA indexer projection at load time (vLLM 0.32, VLLM_FP8_HYBRID=1).

vLLM 0.32 fuses ``indexer.index_qk_proj`` into ``self_attn.qkv_proj``
(packed_modules_mapping), and refuses a fused layer whose shards differ in
precision. The magi hybrid checkpoints store q/k/v as blockwise FP8 but
index_qk_proj as BF16, so every QSA layer failed to build. This quantises the
small BF16 indexer weight (640 x 2560 per QSA layer) to 128x128-block FP8
e4m3 on the fly, with a matching ``weight_scale_inv``, and marks it as an FP8
layer for the hybrid dispatch, so the fused layer is all-FP8. No checkpoint
change.
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger("vllm.qsa_indexer_fp8")
BLOCK = 128
FP8_MAX = 448.0


def enabled() -> bool:
    return os.environ.get("VLLM_FP8_HYBRID", "0").lower() in ("1", "true", "yes")


def quant_block_fp8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] -> (fp8 [N, K], fp32 scale_inv [ceil(N/128), ceil(K/128)])."""
    n, k = w.shape
    bn, bk = -(-n // BLOCK), -(-k // BLOCK)
    p = torch.zeros(bn * BLOCK, bk * BLOCK, dtype=torch.float32, device=w.device)
    p[:n, :k] = w.float()
    b = p.view(bn, BLOCK, bk, BLOCK)
    s = b.abs().amax(dim=(1, 3)).clamp(min=1e-12) / FP8_MAX
    q = (b / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).view(bn * BLOCK, bk * BLOCK)
    return q[:n, :k].to(torch.float8_e4m3fn).contiguous(), s.contiguous()


def apply_config() -> None:
    """Call after vllm_fp8_hybrid.apply(): add index_qk_proj to the FP8 layer set."""
    if not enabled():
        return
    from vllm.model_executor.layers.quantization import auto_gptq as m

    cls = m.AutoGPTQConfig
    if getattr(cls, "_qsa_indexer_fp8", False):
        return
    orig = cls.maybe_update_config

    def maybe_update_config(self, *a, **kw):
        orig(self, *a, **kw)
        fl = getattr(self, "fp8_layers", None)
        if fl:
            add = {l[: -len("q_proj")] + "indexer.index_qk_proj"
                   for l in fl if l.endswith("self_attn.q_proj")}
            fl |= add
            logger.info("qsa indexer fp8: %d index_qk_proj layers quantised at load", len(add))

    cls.maybe_update_config = maybe_update_config
    cls._qsa_indexer_fp8 = True


def apply_model(model_module) -> None:
    """Append to qwen4_exp/nvidia/model.py: convert the weights in the load stream."""
    if not enabled():
        return
    Model = model_module.Qwen4ExpModel
    if getattr(Model, "_qsa_indexer_fp8", False):
        return
    orig = Model.load_weights

    def load_weights(self, weights):
        def stream():
            for name, w in weights:
                if name.endswith("indexer.index_qk_proj.weight") and w.dtype == torch.bfloat16:
                    q, s = quant_block_fp8(w)
                    yield name, q
                    yield name[: -len("weight")] + "weight_scale_inv", s
                else:
                    yield name, w
        return orig(self, stream())

    Model.load_weights = load_weights
    Model._qsa_indexer_fp8 = True


if __name__ == "__main__":  # self-check: round-trip error stays small
    w = torch.randn(640, 2560) * 0.02
    q, s = quant_block_fp8(w)
    assert q.shape == w.shape and s.shape == (5, 20)
    deq = (q.float().view(5, 128, 20, 128) * s[:, None, :, None]).view(640, 2560)
    rel = ((deq - w).norm() / w.norm()).item()
    assert rel < 0.05, rel
    print(f"ok, rel err {rel:.4f}")
