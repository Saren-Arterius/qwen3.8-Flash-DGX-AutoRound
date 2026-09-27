# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vllm_mtp_draft_vocab - reduced draft vocabulary for the Qwen3.8-Flash-Next MTP head.

WHY
---
Per MTP-3 engine step the int8 GPTQ-Marlin ``lm_head`` (qweight I32
[640, 248320] + scales [20, 248320] ~= 646 MB) is read about 3.8 times: once to
verify and three times to draft. The three draft reads only need to produce a
proposal, and rejection sampling stays exact for ANY proposal distribution q as
long as q is positive at the token it drew. So the draft passes can run against
a K-row slice of the head and the emitted distribution does not move; only the
mean accepted length can.

WHAT THIS DOES
--------------
With ``VLLM_MTP_DRAFT_VOCAB=/path/to/ids.txt`` set, this module

  1. wraps ``Qwen3_8FlashNextMTP.load_weights`` and, once the drafter's weights
     are in, builds a second ``ParallelLMHead`` holding only the K columns named
     by the id file, and
  2. wraps ``Qwen3_8FlashNextMTP.compute_logits`` so the drafter runs that
     sliced head and scatters the K logits into a full-width row that is -inf
     everywhere else.

With the variable unset it does nothing at all: neither wrapper is installed.

WHY THE SLICE IS TAKEN BEFORE THE MARLIN REPACK
-----------------------------------------------
A masked full-width matmul would save zero bytes - the whole point is to read
fewer. The slice therefore has to be a real, repacked Marlin head. It cannot be
cut out of the repacked tensors: ``MarlinLinearKernel.process_weights_after_loading``
(vllm/model_executor/kernels/linear/mixed_precision/marlin.py:127-140) runs
``ops.gptq_marlin_repack``, which folds 16 output columns into each packed
group (``marlin_utils.marlin_repacked_nk`` :246-252 recovers
``size_n = qweight.size(1) * pack_factor // 16``), and :142-158 permutes the
scale columns within blocks of 64 (``marlin_utils.get_scale_perms`` :460-467).
Neither is a gather an arbitrary id set survives.

What IS a plain gather is the *checkpoint* layout, before the repack:
``AutoGPTQLinearMethod.create_weights`` (auto_gptq.py:381-392) allocates
``qweight`` as [in/pack_factor, out] with ``packed_dim=0``, i.e. each int32 packs
four values along the INPUT dim for a single output column - so output columns
are independent int32 lanes. ``scales`` is [groups, out] (:412-419). ``qzeros``
never matters here: the checkpoint is ``sym: true``, so
``MPLinearLayerConfig.zero_points`` is False (auto_gptq.py:350) and the kernel
overwrites the parameter with an empty tensor (marlin.py:213-214). Likewise
``desc_act: false`` means ``has_g_idx`` is False and g_idx is discarded
(marlin.py:183-185).

So we slice ``lm_head.qweight`` / ``lm_head.scales`` by column while they are
still in checkpoint layout, hand them to a fresh ``ParallelLMHead``, and let
vLLM's own ``process_weights_after_loading`` do the repack. No Marlin
permutation is reimplemented here.

WHERE THE TENSORS COME FROM
---------------------------
Not from disk. ``Qwen3_8FlashNextMTP`` builds its own ``ParallelLMHead``
(mtp.py:391-396) and ``_remap_mtp_weight_name`` passes ``lm_head.*`` straight
through (mtp.py:101-102), so at the moment ``load_weights`` returns, the
drafter's own head holds exactly the tensors the target's head holds, in
checkpoint layout - ``process_weights_after_loading`` has not run yet. We slice
those. That is what makes the K == vocab_size case bit-identical to the
unpatched path by construction.

THE DRAFTER DOES NOT OWN ITS HEAD FOR LONG
------------------------------------------
Immediately after loading, ``LLMBaseProposer._maybe_share_lm_head``
(vllm/v1/spec_decode/llm_base_proposer.py:1566-1577) takes the "MTP model"
branch, which is unconditional: ``share_lm_head = True``, then
``del self.model.lm_head; self.model.lm_head = target_language_model.lm_head``.
The drafter therefore SHARES the target's head object. That is exactly why the
slice must be a separate module owned by the drafter: mutating ``self.lm_head``
would corrupt the target. This module never touches ``self.lm_head``; it keeps
the sliced head in ``self.__dict__["_draft_vocab"]``, deliberately outside
``nn.Module``'s registries so it is invisible to ``named_modules()``,
``named_parameters()``, the loader's post-load ``process_weights_after_loading``
sweep, and torch.compile.

EXACTNESS
---------
The emitted distribution is unchanged. ``_sample_draft_tokens``
(llm_base_proposer.py:480-509) feeds these logits to
``compute_probs_and_sample_next_token`` (:1859-1897), which divides by the
per-request temperature in place and softmaxes: -inf/T stays -inf and
exp(-inf - max) is exactly 0, so out-of-slice probabilities are exactly zero and
``argmax(probs / exponential_noise)`` can never select one. The resulting
``draft_probs`` is a restricted-support q; block rejection sampling preserves
the target distribution for any q, and its Triton kernels already handle -inf
tails (``rejection_sampler_utils.py:10-17`` guards ``max > -inf`` before the
sumexp). Only mean accepted length can move.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("vllm.mtp_draft_vocab")

ENV_PATH = "VLLM_MTP_DRAFT_VOCAB"
_SENTINEL = "_mtp_draft_vocab_patched"
_STATE = "_draft_vocab"


def apply(mtp_cls) -> None:
    """Install the reduced-draft-vocabulary hooks on the MTP model class.

    No-op unless VLLM_MTP_DRAFT_VOCAB names a readable id-list file.
    """
    path = os.environ.get(ENV_PATH, "").strip()
    if not path:
        return
    if getattr(mtp_cls, _SENTINEL, False):
        return

    orig_load_weights = mtp_cls.load_weights
    orig_compute_logits = mtp_cls.compute_logits

    def load_weights(self, weights):
        loaded = orig_load_weights(self, weights)
        _build_sliced_head(self, loaded, path)
        return loaded

    def compute_logits(self, hidden_states, spec_step_idx: int = 0):
        state = self.__dict__.get(_STATE)
        if state is None:
            return orig_compute_logits(self, hidden_states, spec_step_idx)
        return _sliced_compute_logits(state, hidden_states)

    load_weights.__doc__ = orig_load_weights.__doc__
    compute_logits.__doc__ = orig_compute_logits.__doc__
    mtp_cls.load_weights = load_weights
    mtp_cls.compute_logits = compute_logits
    setattr(mtp_cls, _SENTINEL, True)
    logger.info("mtp draft vocab: hooks installed, id list %s", path)


# ------------------------------------------------------------------ build


def _build_sliced_head(model, loaded_params, path: str) -> None:
    # The structural checks below need neither torch nor vLLM, and running them
    # first means a misconfigured server fails with the reason rather than with
    # an import error.
    if model.__dict__.get(_STATE) is not None:
        return  # already built (e.g. a sleep/wake reload)

    head = getattr(model, "lm_head", None)
    if head is None or not hasattr(head, "qweight") or not hasattr(head, "scales"):
        raise RuntimeError(
            f"{ENV_PATH} is set but the MTP drafter's lm_head is not a GPTQ head "
            "(no qweight/scales). This patch is written for the int8 GPTQ-Marlin "
            "head produced by tools/quantize_lm_head_int8.py."
        )

    # A slice built from uninitialised memory would look like a working server
    # and score as "the lever does nothing", so never guess: either the drafter
    # demonstrably loaded the head, or we re-read those two tensors from the
    # checkpoint ourselves.
    missing = {"lm_head.qweight", "lm_head.scales"} - set(loaded_params or ())
    from_checkpoint = bool(missing)
    if from_checkpoint:
        logger.warning(
            "mtp draft vocab: the drafter did not report loading %s; falling back "
            "to reading them from the checkpoint",
            sorted(missing),
        )

    _check_logits_path(model, head)

    import torch

    from draft_vocab_common import parse_id_list, validate_slice_size
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    vocab_size = int(model.config.vocab_size)
    hidden_size = int(model.config.hidden_size)
    with open(path, "r", encoding="utf-8") as fh:
        ids = parse_id_list(fh.read(), vocab_size)
    k = len(ids)
    validate_slice_size(k, vocab_size)
    if k == vocab_size:
        logger.warning(
            "mtp draft vocab: id list covers the whole %d-token vocabulary; the "
            "sliced head is a bit-identical copy of the full head and saves no "
            "bytes. This is the self-test configuration, not a serving one.",
            vocab_size,
        )

    qweight = head.qweight
    scales = head.scales
    device = qweight.device
    if qweight.shape[1] != vocab_size or scales.shape[1] != vocab_size:
        raise RuntimeError(
            "unexpected head shapes: qweight %s scales %s for vocab %d"
            % (tuple(qweight.shape), tuple(scales.shape), vocab_size)
        )

    index = torch.as_tensor(ids, dtype=torch.long, device=device)
    if from_checkpoint:
        sliced_qweight, sliced_scales = _slice_head_from_checkpoint(
            model, ids, device, qweight.dtype, scales.dtype, vocab_size
        )
    else:
        sliced_qweight = qweight.data.index_select(1, index).contiguous()
        sliced_scales = scales.data.index_select(1, index).contiguous()

    with torch.device(device):
        sliced_head = ParallelLMHead(
            k,
            hidden_size,
            params_dtype=scales.dtype,
            quant_config=model.quant_config,
            # The same prefix the real head uses, so the AutoGPTQ dynamic rule
            # "+:.*lm_head$" -> {"bits": 8} and get_marlin_input_dtype() resolve
            # identically (gptq_utils.py:133-162, marlin_utils.py:666-685).
            prefix="lm_head",
        )

    if not hasattr(sliced_head, "qweight"):
        raise RuntimeError(
            "the sliced ParallelLMHead was not built by AutoGPTQLinearMethod "
            "(no qweight). Check that quantization_config still has "
            '"lm_head": true and the "+:.*lm_head$" dynamic rule.'
        )
    if tuple(sliced_head.qweight.shape) != tuple(sliced_qweight.shape):
        raise RuntimeError(
            "sliced head qweight shape %s != sliced source %s"
            % (tuple(sliced_head.qweight.shape), tuple(sliced_qweight.shape))
        )
    if tuple(sliced_head.scales.shape) != tuple(sliced_scales.shape):
        raise RuntimeError(
            "sliced head scales shape %s != sliced source %s"
            % (tuple(sliced_head.scales.shape), tuple(sliced_scales.shape))
        )

    sliced_head.qweight.data.copy_(sliced_qweight)
    sliced_head.scales.data.copy_(sliced_scales)
    # qzeros and g_idx are never read on this path (sym=true -> zero_points
    # False, desc_act=false -> has_g_idx False; marlin.py:183-185 and :213-214
    # replace both with empty tensors). Zero them anyway so nothing downstream
    # can consume torch.empty garbage.
    for name in ("qzeros", "g_idx"):
        param = getattr(sliced_head, name, None)
        if param is not None and hasattr(param, "data"):
            param.data.zero_()

    del sliced_qweight, sliced_scales

    sliced_head.quant_method.process_weights_after_loading(sliced_head)

    head_bytes = _module_weight_bytes(sliced_head)
    full_bytes = _module_weight_bytes(head)
    model.__dict__[_STATE] = {
        "head": sliced_head,
        "index": index,
        "k": k,
        "vocab_size": vocab_size,
        "path": path,
    }
    logger.info(
        "mtp draft vocab: K=%d of %d (%.2f%%), sliced head %.1f MiB vs full head "
        "%.1f MiB, saving %.1f MiB per draft pass; id list %s",
        k,
        vocab_size,
        100.0 * k / vocab_size,
        head_bytes / (1 << 20),
        full_bytes / (1 << 20),
        (full_bytes - head_bytes) / (1 << 20),
        path,
    )


def _slice_head_from_checkpoint(model, ids, device, qweight_dtype, scales_dtype,
                                vocab_size: int):
    """Fallback: gather the head's columns straight out of the safetensors shard.

    Only used when the drafter's ``load_weights`` did not report loading
    ``lm_head.*`` - i.e. when the draft model's weight stream is filtered to the
    ``mtp.*`` prefix. Reading ``lm_head.qweight`` whole would be 606 MiB, so the
    gather is done in contiguous column blocks: safetensors slices those without
    materialising the rest.
    """
    import json
    import os

    import torch
    from safetensors import safe_open

    from draft_vocab_common import plan_column_chunks

    model_dir = None
    cfg = getattr(model, "vllm_config", None)
    for attr in ("speculative_config", "model_config"):
        holder = getattr(cfg, attr, None) if cfg is not None else None
        if attr == "speculative_config" and holder is not None:
            holder = getattr(holder, "draft_model_config", None)
        candidate = getattr(holder, "model", None) if holder is not None else None
        if isinstance(candidate, str) and os.path.isdir(candidate):
            model_dir = candidate
            break
    if model_dir is None:
        raise RuntimeError(
            f"{ENV_PATH}: the drafter did not load lm_head.* and the checkpoint "
            "directory could not be located from vllm_config; cannot build the "
            "draft-vocabulary slice."
        )

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]

    out = {}
    # 32768 source columns of qweight is 640 * 32768 * 4 B = 80 MiB per read.
    plan = plan_column_chunks(ids, vocab_size, 32768)
    for name, dtype in (("lm_head.qweight", qweight_dtype),
                        ("lm_head.scales", scales_dtype)):
        shard = weight_map.get(name)
        if shard is None:
            raise RuntimeError(f"{name} is not in {index_path}")
        path = os.path.join(model_dir, shard)
        with safe_open(path, framework="pt", device="cpu") as fh:
            sl = fh.get_slice(name)
            rows = sl.get_shape()[0]
            dest = torch.empty((rows, len(ids)), dtype=dtype)
            for lo, hi, offset, local in plan:
                block = sl[:, lo:hi]
                take = torch.as_tensor(local, dtype=torch.long)
                dest[:, offset:offset + len(local)] = block.index_select(1, take).to(
                    dtype
                )
                del block
        out[name] = dest.to(device)
    logger.info(
        "mtp draft vocab: sliced the head from %s in %d column blocks",
        model_dir,
        len(plan),
    )
    return out["lm_head.qweight"].contiguous(), out["lm_head.scales"].contiguous()


def _check_logits_path(model, head) -> None:
    """Refuse to run if the drafter's logits path is not the one we replicate.

    ``compute_logits`` normally calls ``LogitsProcessor.forward`` ->
    ``_get_logits`` -> ``_apply_head`` -> ``lm_head.quant_method.apply`` and then
    truncates to ``org_vocab_size`` (logits_processor.py:63-153). We call
    ``quant_method.apply`` directly, which is equivalent only when the scale is
    1.0, there is no soft cap, the input really is hidden states, head_dtype is
    not overridden, and TP is 1 (no gather). Any other configuration must fail
    loudly at startup rather than silently emit different logits.
    """
    lp = getattr(model, "logits_processor", None)
    if lp is None:
        raise RuntimeError("MTP model has no logits_processor")
    problems = []
    if getattr(lp, "scale", 1.0) != 1.0:
        problems.append(f"logits scale is {lp.scale}, not 1.0")
    if getattr(lp, "soft_cap", None) is not None:
        problems.append(f"soft_cap is set ({lp.soft_cap})")
    if getattr(lp, "logits_as_input", False):
        problems.append("logits_as_input is True")
    head_dtype = getattr(lp, "head_dtype", None)
    if head_dtype is not None and head_dtype != getattr(head, "params_dtype", None):
        problems.append(f"head_dtype override {head_dtype}")
    if int(getattr(head, "tp_size", 1)) != 1:
        problems.append(f"tensor parallel size is {head.tp_size}, not 1")
    if problems:
        raise RuntimeError(
            f"{ENV_PATH} cannot be used with this logits configuration: "
            + "; ".join(problems)
        )


def _module_weight_bytes(module) -> int:
    total = 0
    for name in ("qweight", "scales", "qzeros", "weight"):
        param = getattr(module, name, None)
        data = getattr(param, "data", None)
        if data is not None:
            total += data.numel() * data.element_size()
    return total


# ---------------------------------------------------------------- forward


def _sliced_compute_logits(state, hidden_states):
    head = state["head"]
    index = state["index"]
    k = state["k"]
    vocab_size = state["vocab_size"]

    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    # Same call LogitsProcessor._apply_head makes; TP is 1 so no gather, and the
    # slice is exactly k wide so the org_vocab_size truncation is a no-op. The
    # guard below keeps that assumption checkable rather than assumed.
    slice_logits = head.quant_method.apply(head, flat, bias=None)
    if slice_logits.shape[-1] != k:
        slice_logits = slice_logits[..., :k]

    full = slice_logits.new_full((slice_logits.shape[0], vocab_size), float("-inf"))
    # index_copy_ rather than advanced indexing: one kernel, no intermediate
    # index tensor, and it is a plain scatter so the result is exact.
    full.index_copy_(1, index, slice_logits)
    if hidden_states.dim() == 2:
        return full
    return full.reshape(*hidden_states.shape[:-1], vocab_size)
