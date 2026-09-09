#!/usr/bin/env python3
"""Build-time patch (qwen38-flash-dgx): reduced draft vocabulary for the MTP drafter.

vLLM shares the target model's lm_head with the MTP draft (llm_base_proposer._maybe_share_lm_head),
so every draft step scores all 248,320 vocabulary rows: a 1.27 GiB bf16 read per drafted token,
on a decode step that is memory-bandwidth bound. With VLLM_MTP_DRAFT_VOCAB=<ids.npy> the draft
scores only those rows (a private, sliced copy of the head; the target's lm_head is untouched)
and the logits of every other id are -inf, so the proposer's argmax/sampling code is unchanged.
The target still verifies every drafted token, so outputs are identical to full-vocabulary
drafting; only the acceptance rate can move (down, when the target wants an id outside the set).

Fork note (Saren-Arterius/qwen3.8-Flash-DGX-AutoRound): this fork's lm_head is int8
GPTQ-Marlin, so there is no dense .weight to slice; the hook then dequantizes the
selected rows from the checkpoint's lm_head.qweight/qzeros/scales (GPTQ v1, g128) at
first use (VLLM_MTP_DRAFT_VOCAB_CKPT, default /model) into a bf16 slice.

usage: patch_mtp_draft_vocab.py <path to vllm/models/qwen3_8_flash_next/nvidia/mtp.py>
Inert unless VLLM_MTP_DRAFT_VOCAB is set at runtime.
"""
import sys

TARGET = sys.argv[1]
MARK = "qwen38-flash-dgx: reduced draft vocabulary"
HOOK = '''

# --- qwen38-flash-dgx: reduced draft vocabulary (VLLM_MTP_DRAFT_VOCAB=<ids.npy>) -------------
import os as _dv_os
from vllm.logger import init_logger as _dv_init_logger

_dv_logger = _dv_init_logger(__name__)


def _dv_compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
    st = getattr(self, "_dv_state", None)
    if st is None:
        import numpy as _np

        head = self.lm_head  # the target's lm_head, shared in by the proposer
        w = getattr(head, "weight", None)
        vocab = int(getattr(self.logits_processor, "org_vocab_size", 0)
                    or getattr(self.logits_processor, "vocab_size", 0)
                    or (w.shape[0] if w is not None else head.org_vocab_size))
        ids = torch.from_numpy(_np.load(_dv_os.environ["VLLM_MTP_DRAFT_VOCAB"]).astype(_np.int64))
        ids = ids[(ids >= 0) & (ids < vocab)].to(hidden_states.device)
        if w is not None and w.dim() == 2 and w.shape[0] >= vocab:
            wk = w.index_select(0, ids).contiguous()  # dense (bf16) head
            full_bytes = w.shape[0] * w.shape[1] * w.element_size()
        else:
            wk, full_bytes = _dv_gptq_head_rows(ids.cpu().numpy(), hidden_states.device)
        st = self._dv_state = (ids, wk, vocab)
        _dv_logger.info(
            "MTP reduced draft vocabulary: %d of %d rows (%.0f -> %.0f MiB per draft step)",
            ids.numel(), vocab, full_bytes / 2**20, wk.numel() * wk.element_size() / 2**20,
        )
    ids, wk, vocab = st
    red = torch.nn.functional.linear(hidden_states.to(wk.dtype), wk)
    full = red.new_full((red.shape[0], vocab), float("-inf"))
    full.index_copy_(1, ids, red)
    return full


def _dv_gptq_head_rows(ids, device):
    """Dequantize the selected lm_head rows from the checkpoint's GPTQ tensors
    (int8/int4 v1 layout: qweight [in/pack, out] packed along in, qzeros stores zp-1,
    scales [in/group, out]) -> bf16 [len(ids), in] on `device`."""
    import json as _json
    import numpy as _np
    from safetensors import safe_open as _safe_open

    ckpt = _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB_CKPT", "/model")
    wm = _json.load(open(_dv_os.path.join(ckpt, "model.safetensors.index.json")))["weight_map"]
    names = ("lm_head.qweight", "lm_head.qzeros", "lm_head.scales")
    ts = {}
    for f in {wm[n] for n in names}:
        with _safe_open(_dv_os.path.join(ckpt, f), "pt", device="cpu") as sf:
            for n in names:
                if n in sf.keys():
                    ts[n] = sf.get_tensor(n)
    qw, qz, sc = (ts[n] for n in names)
    groups, out_f = sc.shape
    # pack = number of values per int32 along in; in = groups * group_size, group_size from qzeros/scales
    # geometry: qzeros [groups, out/pack] -> pack = out / qzeros.shape[1]
    pack = out_f // qz.shape[1]
    bits = 32 // pack
    in_f = qw.shape[0] * pack
    group = in_f // groups
    mask = (1 << bits) - 1
    q = qw.numpy().astype(_np.uint32)[:, ids]                       # [in/pack, k]
    nib = _np.stack([(q >> (i * bits)) & mask for i in range(pack)], axis=1).reshape(in_f, len(ids))
    zp = int(qz.numpy().astype(_np.uint32)[0, 0] & mask) + 1
    scale = sc.to(torch.float32).numpy()[:, ids]                    # [groups, k]
    wf = (nib.astype(_np.float32) - zp).reshape(groups, group, len(ids)) * scale[:, None, :]
    wk = torch.from_numpy(_np.ascontiguousarray(wf.reshape(in_f, len(ids)).T)).to(torch.bfloat16)
    return wk.to(device), qw.numel() * 4 + sc.numel() * sc.element_size()


if _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB"):
    Qwen3_8FlashNextMTP.compute_logits = _dv_compute_logits  # type: ignore[method-assign]
    _dv_logger.info("MTP reduced draft vocabulary enabled: %s", _dv_os.environ["VLLM_MTP_DRAFT_VOCAB"])
'''

src = open(TARGET).read()
if MARK in src:
    print("  draft-vocab hook already installed"); sys.exit(0)
assert "class Qwen3_8FlashNextMTP(" in src, "MTP class not found"
open(TARGET, "w").write(src.rstrip("\n") + HOOK)
import ast; ast.parse(open(TARGET).read())
print("  draft-vocab hook INSTALLED in", TARGET, "(inert unless VLLM_MTP_DRAFT_VOCAB is set)")
