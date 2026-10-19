"""DFlash/DFlash2 target-feature capture for Qwen4Exp (Qwen3.8-Flash-Next), vLLM 0.32.

vLLM's EAGLE3/DFlash plumbing asks the target for aux hidden states at layer
ids L = target_layer_id + 1 ("the stream entering layer L" = the completed
output of layer L-1). Qwen4Exp has no such hook upstream. This ports the
model card's SGLang capture patch (incoai/Qwen3.8-Flash-Next-DFlash2,
patches/qwen38-dflash2-capture.patch): DFlash2 was trained on the completed,
zero-based layer outputs averaged over the hc_count hyper-connection streams.

vLLM 0.32 runs a delayed HC combine between layers, so a layer's output is
(pre-combine state, block_output, injection). For capture only, the pending
combine is materialised with that layer's own ``mlp_hyper_connection.combine``
(the same op the PP path uses); the delayed tuple handed to the next layer is
untouched. Validated layout: TP1, no PP, no sequence parallelism.

Applied by appending ``apply(sys.modules[__name__])`` to
vllm/models/qwen4_exp/nvidia/model.py.
"""
from __future__ import annotations

import logging

logger = logging.getLogger("vllm.qwen4exp_dflash_capture")


def apply(model_module) -> None:
    if getattr(model_module, "_dflash_capture", False):
        return
    Model = model_module.Qwen4ExpModel
    Layer = model_module.Qwen4ExpDecoderLayer

    orig_layer_forward = Layer.forward

    def layer_forward(self, hidden_states, prev_block_output, prev_injection, positions, **kw):
        out = orig_layer_forward(self, hidden_states, prev_block_output, prev_injection,
                                 positions, **kw)
        sink = getattr(self, "_dflash_sink", None)
        if sink is not None:
            h, block_out, inj = out
            done = self.mlp_hyper_connection.combine(h, block_out, inj)
            hc = self.config_hc_count
            sink.append(done.unflatten(-1, (hc, done.shape[-1] // hc)).mean(dim=-2))
        return out

    orig_model_forward = Model.forward

    def model_forward(self, *a, **kw):
        layers = getattr(self, "_dflash_layers", ())
        if not layers:
            return orig_model_forward(self, *a, **kw)
        sink: list = []
        for i in layers:
            self.layers[i]._dflash_sink = sink
        try:
            out = orig_model_forward(self, *a, **kw)
        finally:
            for i in layers:
                self.layers[i]._dflash_sink = None
        if not isinstance(out, model_module.torch.Tensor):  # PP intermediate tensors
            return out
        return out, sink

    def set_aux_hidden_state_layers(self, layers) -> None:
        inner = getattr(self, "language_model", self).model
        # vLLM's ids are capture-before-layer-L; hook the end of layer L-1.
        ids = tuple(sorted(int(l) - 1 for l in layers))
        assert ids and all(0 <= i < len(inner.layers) for i in ids), ids
        if getattr(inner, "use_hc_sequence_parallel", False):
            raise RuntimeError("DFlash capture is validated without HC sequence parallelism")
        hc = int(inner.config.hc_count)
        for i in ids:
            inner.layers[i].config_hc_count = hc
        inner._dflash_layers = ids
        logger.info("DFlash capture: completed outputs of layers %s, mean over %d HC streams",
                    list(ids), hc)

    def get_eagle3_default_aux_hidden_state_layers(self):
        raise RuntimeError("Qwen4Exp has no default aux layers; the draft config must name them")

    Layer.forward = layer_forward
    Model.forward = model_forward
    for cls in (model_module.Qwen4ExpForConditionalGeneration, model_module.Qwen4ExpForCausalLM):
        cls.supports_eagle3 = True
        cls.has_own_lm_head = getattr(cls, "has_own_lm_head", False)
        cls.has_own_embed_tokens = getattr(cls, "has_own_embed_tokens", False)
        cls.set_aux_hidden_state_layers = set_aux_hidden_state_layers
        cls.get_eagle3_default_aux_hidden_state_layers = get_eagle3_default_aux_hidden_state_layers
    model_module._dflash_capture = True
    logger.info("DFlash capture patch applied to %s", model_module.__name__)
