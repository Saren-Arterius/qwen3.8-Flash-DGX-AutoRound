#!/usr/bin/env python3
"""Build-time patch (qwen38-flash-dgx): private MTP draft head — reduced vocabulary and/or int4.

vLLM shares the target model's lm_head with the MTP draft (llm_base_proposer._maybe_share_lm_head),
so every draft step reads the whole head: 248,320 rows x 2560 int8 = 616 MiB per drafted token,
on a decode step that is memory-bandwidth bound. Two independent knobs shrink that read; the
target still verifies every drafted token, so outputs are identical to full-precision,
full-vocabulary drafting and only the acceptance rate can move:

  VLLM_MTP_DRAFT_VOCAB=<ids.npy>  the draft scores only those rows (a private slice of the head)
                                  and the logits of every other id are -inf. Acceptance drops for
                                  text whose tokens fall outside the set (e.g. CJK with an
                                  English/code-weighted set). Idea: upstream blazux / MiaAI-Lab.
  VLLM_MTP_DRAFT_HEAD=int4        the draft uses a private int4 g128 RTN GPTQ-Marlin copy of the
                                  head (320 MB, built at first use, ~half the bytes of int8) over
                                  the FULL vocabulary, so every language keeps its acceptance.
                                  Combined with DRAFT_VOCAB the slice itself is int4 (~85 MB).

Fork note (Saren-Arterius/qwen3.8-Flash-DGX-AutoRound): this fork's lm_head is int8
GPTQ-Marlin, so there is no dense .weight to slice; rows are dequantized from the checkpoint's
lm_head.qweight/qzeros/scales (GPTQ v1, g128; VLLM_MTP_DRAFT_VOCAB_CKPT, default /model) in
16k-row chunks, then either kept as a bf16 slice or re-quantized to int4 with vLLM's own
marlin_quantize (chunked packing is bit-identical to whole-tensor packing).

usage: patch_mtp_draft_vocab.py <path to vllm/models/qwen3_8_flash_next/nvidia/mtp.py>
Inert unless VLLM_MTP_DRAFT_VOCAB or VLLM_MTP_DRAFT_HEAD is set at runtime.
"""
import sys

TARGET = sys.argv[1]
MARK = "qwen38-flash-dgx: private MTP draft head"
HOOK = '''

# --- qwen38-flash-dgx: private MTP draft head (VLLM_MTP_DRAFT_VOCAB=<ids.npy>, VLLM_MTP_DRAFT_HEAD=int4)
import os as _dv_os
from vllm.logger import init_logger as _dv_init_logger

_dv_logger = _dv_init_logger(__name__)
_DV_CHUNK = 16384


def _dv_compute_logits(self, hidden_states: torch.Tensor, spec_step_idx: int = 0):
    st = getattr(self, "_dv_state", None)
    if st is None:
        st = self._dv_state = _dv_build(self, hidden_states)
    ids, head, vocab = st
    red = head(hidden_states)
    if ids is None:
        return red[:, :vocab] if red.shape[1] != vocab else red
    full = red.new_full((red.shape[0], vocab), float("-inf"))
    full.index_copy_(1, ids, red)
    return full


def _dv_build(self, hidden_states):
    import time as _time
    import numpy as _np

    t0 = _time.time()
    dev = hidden_states.device
    head = self.lm_head  # the target's lm_head, shared in by the proposer
    w = getattr(head, "weight", None)
    dense = w is not None and w.dim() == 2
    vocab = int(getattr(self.logits_processor, "org_vocab_size", 0)
                or getattr(self.logits_processor, "vocab_size", 0)
                or (w.shape[0] if dense else head.org_vocab_size))
    if dense:
        out_f, in_f = w.shape
        rows = lambda sel: w.index_select(0, torch.as_tensor(sel, device=w.device)).to(torch.bfloat16)
        full_bytes = w.numel() * w.element_size()
    else:
        rows, out_f, in_f, full_bytes = _dv_gptq_rows()

    ids_path = _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB")
    int4 = _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD", "").lower() == "int4"
    if ids_path:
        ids = _np.load(ids_path).astype(_np.int64)
        ids = ids[(ids >= 0) & (ids < vocab)]
        if int4 and len(ids) % 64:  # Marlin needs N % 64 == 0; duplicates write identical values
            ids = _np.concatenate([ids, _np.repeat(ids[-1], 64 - len(ids) % 64)])
        sel = ids
    else:
        sel = _np.arange(out_f)
        ids = None
    n = len(sel)

    if int4:
        from vllm.scalar_type import scalar_types as _st
        from vllm.model_executor.layers.quantization.utils.marlin_utils_test import marlin_quantize as _mq
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            marlin_make_workspace_new as _mws, apply_gptq_marlin_linear as _mapply)
        qws, scs = [], []
        for i in range(0, n, _DV_CHUNK):
            wt = rows(sel[i:i + _DV_CHUNK]).to(dev).t().contiguous()      # [in, chunk] bf16
            _, qw, s, _, _, _ = _mq(wt, _st.uint4b8, 128, False)
            qws.append(qw); scs.append(s)
            del wt
        qw = torch.cat(qws, 1).contiguous(); sc = torch.cat(scs, 1).contiguous()
        del qws, scs
        ws = _mws(dev)
        e = torch.empty(0, dtype=torch.int, device=dev)

        def apply(h, qw=qw, sc=sc, ws=ws, e=e, n=n, k=in_f):
            return _mapply(h.to(sc.dtype), qw, sc, e, e, e, ws, _st.uint4b8, n, k, True)
        bytes_ = qw.numel() * 4 + sc.numel() * sc.element_size()
        kind = "int4 g128 RTN GPTQ-Marlin"
    else:
        wk = torch.cat([rows(sel[i:i + _DV_CHUNK]).to(dev) for i in range(0, n, _DV_CHUNK)]).contiguous()

        def apply(h, wk=wk):
            return torch.nn.functional.linear(h.to(wk.dtype), wk)
        bytes_ = wk.numel() * wk.element_size()
        kind = "bf16"

    # self-check on the first real hidden states: private head vs the shared target head
    try:
        ref = self.logits_processor(self.lm_head, hidden_states)
        if ref is not None:
            ref = ref[:, :vocab].float()
            got = apply(hidden_states)[:, :ids.shape[0] if ids is not None else vocab].float()
            if ids is not None:
                ref = ref[:, torch.as_tensor(ids, device=ref.device)]
            else:
                got = got[:, :vocab]
            top1 = (ref.argmax(1) == got.argmax(1)).float().mean().item()
            rel = ((got - ref).norm() / ref.norm()).item()
            _dv_logger.info("MTP private draft head self-check: top-1 agreement %.3f, rel err %.4f (T=%d)",
                            top1, rel, ref.shape[0])
    except Exception as ex:  # never let a diagnostic kill the drafter
        _dv_logger.warning("MTP private draft head self-check skipped: %r", ex)

    _dv_logger.info("MTP private draft head: %s, %s of %d rows, %.0f -> %.0f MiB per draft step, built in %.1fs",
                    kind, ("%d" % ids.shape[0]) if ids is not None else "all", vocab,
                    full_bytes / 2**20, bytes_ / 2**20, _time.time() - t0)
    ids_t = torch.as_tensor(ids, device=dev) if ids is not None else None
    return ids_t, apply, vocab


def _dv_gptq_rows():
    """Row dequantizer for the checkpoint's GPTQ lm_head (int8/int4 v1 layout: qweight [in/pack, out]
    packed along in, qzeros stores zp-1, scales [in/group, out]). rows(sel) -> bf16 [len(sel), in]."""
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
    pack = out_f // qz.shape[1]
    bits = 32 // pack
    in_f = qw.shape[0] * pack
    group = in_f // groups
    mask = (1 << bits) - 1
    qwn = qw.numpy().view(_np.uint32)
    zp = int(qz.numpy().view(_np.uint32)[0, 0] & mask) + 1
    scn = sc.to(torch.float32).numpy()

    def rows(sel):
        q = qwn[:, sel]                                                  # [in/pack, k]
        nib = _np.stack([(q >> (i * bits)) & mask for i in range(pack)], axis=1).reshape(in_f, len(sel))
        wf = (nib.astype(_np.float32) - zp).reshape(groups, group, len(sel)) * scn[:, None, sel]
        return torch.from_numpy(_np.ascontiguousarray(wf.reshape(in_f, len(sel)).T)).to(torch.bfloat16)
    return rows, out_f, in_f, qw.numel() * 4 + sc.numel() * sc.element_size()


if _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB") or _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD"):
    Qwen3_8FlashNextMTP.compute_logits = _dv_compute_logits  # type: ignore[method-assign]
    _dv_logger.info("MTP private draft head enabled: vocab=%s head=%s",
                    _dv_os.environ.get("VLLM_MTP_DRAFT_VOCAB"), _dv_os.environ.get("VLLM_MTP_DRAFT_HEAD", "shared"))
'''

src = open(TARGET).read()
if MARK in src:
    print("  draft-head hook already installed"); sys.exit(0)
assert "class Qwen3_8FlashNextMTP(" in src, "MTP class not found"
open(TARGET, "w").write(src.rstrip("\n") + HOOK)
import ast; ast.parse(open(TARGET).read())
print("  draft-head hook INSTALLED in", TARGET, "(inert unless VLLM_MTP_DRAFT_VOCAB / VLLM_MTP_DRAFT_HEAD is set)")
