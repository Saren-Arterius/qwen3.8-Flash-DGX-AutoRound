"""vllm_fp8_hybrid — int4(GPTQ-Marlin) + blockwise-fp8 hybrid dispatch.

The checkpoint's routed experts + lm_head are GPTQ int4/int8 (served by
AutoGPTQConfig), while dense side-layers (GDN in_proj/out_proj, QSA q/k/v/o,
shared experts) are stored as blockwise FP8-e4m3 (`weight` fp8 +
`weight_scale_inv`, block 128x128). Stock AutoGPTQConfig would init those as
unquantized bf16 and then fail to load fp8 tensors.

This patch (enabled by VLLM_FP8_HYBRID=1, no-op otherwise):
  * after ``maybe_update_config``, scans the checkpoint's safetensors metadata
    for F8_E4M3 weights that have a ``.weight_scale_inv`` sibling and records
    those layer names;
  * maps them through the model's hf->vllm mapper alongside
    ``modules_in_block_to_quantize``;
  * in ``get_quant_method``, dispatches matching LinearBase layers (including
    fused qkv_proj / gate_up_proj via ``packed_modules_mapping``) to a shared
    blockwise ``Fp8Config`` instead of the GPTQ/unquantized path.

Port of Saren's spark-dflash-hybrid-fp8 (vLLM 0.23 INCConfig) onto the
qwen38next build's AutoGPTQConfig.

SECOND, INDEPENDENT SWITCH: VLLM_DRAFTER_EXPERTS_FP8
----------------------------------------------------
The MTP drafter's 512 routed experts are the last BF16 tensors of any size in
the served model (``mtp.layers.0.mlp.experts.{0..511}.*``, ~5.03 GB resident,
~0.30-0.35 GB of the per-step byte budget), because the checkpoint's
quantization_config excludes them with the dynamic rule
``"-:.*layers\\.48\\..*"``. ``get_moe_quant_method`` (auto_gptq.py:71-94)
therefore returns ``UnquantizedFusedMoEMethod`` for them, and the startup log
shows the unquantized FlashInfer CUTLASS MoE path.

Quantising a DRAFTER weight is exact on the emitted distribution — the target
verifies every proposed token under block rejection sampling — so only mean
accepted length can move. NVIDIA's own NVFP4 checkpoint quantises exactly these
tensors (to 128x128 block-scaled FP8) while keeping the main attention and
shared experts at BF16, which is the precedent for the precision level.

No checkpoint change is needed: vLLM in this image can quantise a BF16 MoE at
load time. ``Fp8Config.get_quant_method`` (fp8.py:215-222) routes a
``RoutedExperts`` layer to ``Fp8PerTensorOnlineMoEMethod`` when the checkpoint is
not fp8-serialised, and that method's ``process_weights_after_loading``
(online/fp8.py:565-606) writes fp8 tensors and ``replace_parameter``s the bf16
ones away (:505-506), so resident bytes really do halve. Note the *offline*
``Fp8MoEMethod`` cannot be used here at all: ``create_weights`` asserts
``is_checkpoint_fp8_serialized`` (fp8.py:563).

Values:
    VLLM_DRAFTER_EXPERTS_FP8 unset/0  no change (default; the BF16 arm)
    1 | pertensor                     per-tensor online fp8 (matches the path
                                      vLLM's own Fp8Config picks)
    block                             128x128 blockwise online fp8 (the NVIDIA
                                      precision level; higher backend-fallback
                                      risk on SM121)
    list                              change nothing, but log every RoutedExperts
                                      prefix seen — the preflight that tells us
                                      what VLLM_DRAFTER_EXPERTS_FP8_PREFIX must
                                      match

Guards, so this can never touch the target's int4 experts:
    * only ``RoutedExperts`` layers,
    * only prefixes matching VLLM_DRAFTER_EXPERTS_FP8_PREFIX (default: an mtp/
      layer-48 pattern),
    * only layers for which the unmodified AutoGPTQ path returned
      ``UnquantizedFusedMoEMethod`` or None. The target's experts return a real
      GPTQ MoE method and are left alone.

The backend that will actually serve these experts is decided at runtime by
``select_fp8_moe_backend`` (fused_moe/oracle/fp8.py:271-425), which tries
FLASHINFER_TRTLLM, then FLASHINFER_CUTLASS, then DEEPGEMM (removed here — we run
VLLM_USE_DEEP_GEMM=0), then VLLM_CUTLASS (removed unless allow_vllm_cutlass),
then TRITON, then MARLIN, and logs
``"Using <X> Fp8 MoE backend out of potential backends: [...]"`` once. That log
line is the acceptance gate for this arm: vLLM issue #43507 reports the CUTLASS
MoE backend unavailable on SM_120/SM_121 for tensor/token-scaled FP8, so a
TRITON or MARLIN fallback is a real possibility and a slower kernel would eat
the byte saving. Read the line before trusting the arm.
"""
import logging
import os
import re

logger = logging.getLogger("vllm.fp8_hybrid")

_SENTINEL = "_fp8_hybrid_patched"

_DRAFTER_ENV = "VLLM_DRAFTER_EXPERTS_FP8"
_DRAFTER_PREFIX_ENV = "VLLM_DRAFTER_EXPERTS_FP8_PREFIX"
# The MTP layer is layer 48 of the flattened model and lives under an "mtp."
# path in the checkpoint; both spellings are accepted so the preflight does not
# have to be run before the first real attempt.
_DRAFTER_PREFIX_DEFAULT = r"(^|\.)mtp(\.|$)|\.layers\.48(\.|$)|(^|\.)draft(\.|$)"

_OFF = ("", "0", "off", "false", "no")
_seen_moe_prefixes: set = set()


def _drafter_mode() -> str:
    return os.environ.get(_DRAFTER_ENV, "").strip().lower()


def apply() -> None:
    if os.environ.get("VLLM_FP8_HYBRID", "0").lower() not in ("1", "true", "yes"):
        return
    from vllm.model_executor.layers.quantization import auto_gptq as m

    cfg_cls = m.AutoGPTQConfig
    if getattr(cfg_cls, _SENTINEL, False):
        return

    from vllm.model_executor.layers.linear import LinearBase
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.transformers_utils.config import get_safetensors_params_metadata

    orig_update = cfg_cls.maybe_update_config
    orig_mapper = cfg_cls.apply_vllm_mapper
    orig_gqm = cfg_cls.get_quant_method

    def maybe_update_config(self, model_name, hf_config=None, revision=None):
        orig_update(self, model_name, hf_config=hf_config, revision=revision)
        md = get_safetensors_params_metadata(model_name, revision=revision)
        self.fp8_layers = {
            name[: -len(".weight")]
            for name, info in md.items()
            if name.endswith(".weight")
            and info.get("dtype") == "F8_E4M3"
            and name[: -len(".weight")] + ".weight_scale_inv" in md
        }
        if self.fp8_layers:
            logger.info(
                "fp8 hybrid: %d blockwise-fp8 layers detected", len(self.fp8_layers)
            )

    def apply_vllm_mapper(self, hf_to_vllm_mapper):
        orig_mapper(self, hf_to_vllm_mapper)
        if getattr(self, "fp8_layers", None):
            self.fp8_layers = set(hf_to_vllm_mapper.apply_list(list(self.fp8_layers)))

    def _is_fp8_layer(self, prefix: str) -> bool:
        fp8_layers = getattr(self, "fp8_layers", None)
        if not fp8_layers:
            return False
        head, _, proj = prefix.rpartition(".")
        fused = self.packed_modules_mapping.get(proj)
        names = [f"{head}.{p}" for p in fused] if fused and head else [prefix]
        return all(any(l in n for l in fp8_layers) for n in names)

    def get_quant_method(self, layer, prefix):
        if isinstance(layer, LinearBase) and self._is_fp8_layer(prefix):
            fp8_cfg = getattr(self, "_fp8_cfg", None)
            if fp8_cfg is None:
                fp8_cfg = Fp8Config(
                    is_checkpoint_fp8_serialized=True,
                    activation_scheme="dynamic",
                    weight_block_size=[128, 128],
                )
                fp8_cfg.packed_modules_mapping = self.packed_modules_mapping
                self._fp8_cfg = fp8_cfg
            logger.debug("fp8 hybrid: %s -> Fp8LinearMethod", prefix)
            return fp8_cfg.get_quant_method(layer, prefix)
        base = orig_gqm(self, layer, prefix)
        drafter = _maybe_drafter_moe_fp8(layer, prefix, base)
        return base if drafter is None else drafter

    cfg_cls.maybe_update_config = maybe_update_config
    cfg_cls.apply_vllm_mapper = apply_vllm_mapper
    cfg_cls._is_fp8_layer = _is_fp8_layer
    cfg_cls.get_quant_method = get_quant_method
    setattr(cfg_cls, _SENTINEL, True)
    logger.info("fp8 hybrid patch applied to AutoGPTQConfig")
    if _drafter_mode() not in _OFF:
        logger.info(
            "fp8 hybrid: drafter-experts arm armed, %s=%s prefix=%r",
            _DRAFTER_ENV,
            _drafter_mode(),
            os.environ.get(_DRAFTER_PREFIX_ENV, _DRAFTER_PREFIX_DEFAULT),
        )


def _maybe_drafter_moe_fp8(layer, prefix, base_method):
    """Return an online-fp8 MoE method for the DRAFT model's routed experts.

    Returns None (leave the base method alone) unless every guard passes.
    """
    mode = _drafter_mode()
    if mode in _OFF:
        return None

    from vllm.model_executor.layers.fused_moe import (
        RoutedExperts,
        UnquantizedFusedMoEMethod,
    )

    if not isinstance(layer, RoutedExperts):
        return None

    if prefix not in _seen_moe_prefixes:
        _seen_moe_prefixes.add(prefix)
        logger.info(
            "fp8 hybrid: RoutedExperts prefix %r -> %s",
            prefix,
            type(base_method).__name__,
        )
    if mode == "list":
        return None

    pattern = os.environ.get(_DRAFTER_PREFIX_ENV, _DRAFTER_PREFIX_DEFAULT)
    if not re.search(pattern, prefix):
        return None

    # Only take over experts the base config left unquantized. The target's int4
    # experts return a real GPTQ MoE method and must never reach this branch.
    if base_method is not None and not isinstance(
        base_method, UnquantizedFusedMoEMethod
    ):
        logger.warning(
            "fp8 hybrid: %s matches the drafter prefix but is already served by "
            "%s; leaving it alone",
            prefix,
            type(base_method).__name__,
        )
        return None

    if mode in ("block", "blockwise"):
        from vllm.model_executor.layers.quantization.online.fp8 import (
            Fp8PerBlockOnlineMoEMethod,
        )

        method = Fp8PerBlockOnlineMoEMethod(layer=layer)
    else:
        from vllm.model_executor.layers.quantization.online.fp8 import (
            Fp8PerTensorOnlineMoEMethod,
        )

        method = Fp8PerTensorOnlineMoEMethod(layer=layer)

    logger.info(
        "fp8 hybrid: drafter experts %s -> %s (backend %s)",
        prefix,
        type(method).__name__,
        getattr(getattr(method, "fp8_backend", None), "value", "?"),
    )
    return method
