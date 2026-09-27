#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build support for the pinned GB10 serving recipe."""

import ast
import sys

LLG = (
    "/usr/local/lib/python3.12/dist-packages/vllm/models/"
    "qwen3_8_flash_next/nvidia/low_latency_gemm.py"
)
SKINNY = (
    "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/kernels/"
    "linear/cute_dsl/skinny_gemm.py"
)

# --------------------------------------------------------------------------- #
# Part 1: low_latency_gemm.py -- the SM12x gate and the TP=1 plan entries
# --------------------------------------------------------------------------- #
src = open(LLG).read()
assert "_is_sm103" in src, "low_latency_gemm.py: gate helper moved"
if "qwen38next_low_latency_sm12x" in src:
    print("patch_low_latency_gemm_sm12x.py: already applied, skipping", file=sys.stderr)
else:
    OLD_GATE = """def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))"""
    NEW_GATE = '''def _is_sm103() -> bool:
    return current_platform.is_device_capability((10, 3))


# --- qwen38next_low_latency_sm12x (Iteration 6 / R3) ----------------------
import os as _llg_os

from vllm.logger import init_logger as _llg_init_logger

_llg_logger = _llg_init_logger(__name__)

_LLG_PREFETCH_TILES = 2  # _skinny_gemm.py:11
# Whether QWEN38NEXT_LOW_LATENCY_GEMM=1 was seen, as distinct from whether the
# box is SM12x. Set by _llg_sm12x_enabled(); read by _llg_signal().
_LLG_KNOB_SET = None
_LLG_SIGNALLED = False


def _llg_signal(count: int, shapes) -> None:
    """Emit EXACTLY ONE INFO line saying what R3 did at install time.

    An unattended run must distinguish a zero measured benefit from a patch
    that never installed. ``count`` is the number of dispatch
    attachments actually made and ``shapes`` the (N, K) pairs they were made on,
    so ``iter6 R3 installed: 0 dispatch attachments`` is exactly as loud as a
    real install and a log-grep acceptance gate (EXPECT_LOG_RE in the variant
    file) can require a non-zero count. With the knob unset the single line is
    ``iter6 R3 disabled``, so the log proves either state.
    """
    global _LLG_SIGNALLED
    if _LLG_SIGNALLED:
        return
    _LLG_SIGNALLED = True
    if not _LLG_KNOB_SET:
        _llg_logger.info("iter6 R3 disabled")
        return
    pdl = "off" if _llg_os.environ.get("QWEN38NEXT_LLG_PDL", "1") == "0" else "on"
    _llg_logger.info(
        "iter6 R3 installed: %d dispatch attachments, accepted shapes %s, PDL=%s",
        count,
        sorted(shapes),
        pdl,
    )


def _llg_sm12x_enabled() -> bool:
    """Opt-in: run the CuTe-DSL skinny GEMM on SM12x (GB10) too."""
    global _LLG_KNOB_SET
    _LLG_KNOB_SET = _llg_os.environ.get("QWEN38NEXT_LOW_LATENCY_GEMM", "0") == "1"
    if not _LLG_KNOB_SET:
        return False
    return current_platform.is_device_capability_family(120)


def _llg_validate(cfg: SkinnyGemmConfig, n: int, k: int) -> str | None:
    """Reproduce CuteSkinnyGemm.__init__'s contract (_skinny_gemm.py:38-44) plus
    the K-loop's floor divide (:198, no tail loop). Returns a reason string when
    the kernel raises ValueError on that shape -- skinny_gemm.py:250-255 makes
    the same two tests on every call -- so the refusal happens at build time,
    else None."""
    if cfg.block_size % 32:
        return "block_size %d is not a multiple of the warp size" % cfg.block_size
    k_tile = cfg.block_size * cfg.vector_width
    if k % k_tile:
        return (
            "K=%d is not a multiple of block_size*vector_width=%d; the kernel "
            "raises ValueError on that shape (skinny_gemm.py:250-255) and "
            "_skinny_gemm.py:198 floor-divides with no tail loop -- this "
            "validator moves the refusal to build time"
            % (k, k_tile)
        )
    if k < _LLG_PREFETCH_TILES * k_tile:
        return "K=%d is fewer than two complete %d-element tiles" % (k, k_tile)
    if n % cfg.outputs_per_block:
        return "N=%d is not a multiple of outputs_per_block=%d" % (
            n,
            cfg.outputs_per_block,
        )
    return None


def _llg_add_tp1_plans() -> None:
    """Plan entries for the TP=1 shapes the shipped TP=4 table misses.

    Configs come from ShapeDynamicSkinnyGemm._config(), the module's own
    heuristic -- nothing is invented here -- and every emitted entry is checked
    against the kernel's own constructor contract first.

    (10240, 320) -- the hyper-connection input_mix_weight_up -- is deliberately
    ABSENT: _config() returns block_size=32 / vector_width=4 for it, k_tile=128,
    and 320 % 128 == 64. The kernel raises ValueError on that shape
    (skinny_gemm.py:250-255); the validator moves the refusal to build time.
    See this file's module docstring, CORRECTION 1.
    """
    extra = [
        (320, 10240),   # final-mixer input_mix_weight_down
        (96, 2560),     # fused GDN in_proj_ba
    ]
    for n, k in extra:
        plan = QWEN38NEXT_GEMM_PLANS.setdefault((n, k), {})
        for m in (1, 2, 4, 8, 16):
            if m in plan:
                continue
            base = shape_dynamic_skinny_gemm._config(m, n, k)
            cfg = SkinnyGemmConfig(
                m,
                base.block_size,
                base.outputs_per_block,
                base.k_unroll,
                base.vector_width,
                base.static_k,
            )
            why = _llg_validate(cfg, n, k)
            if why is not None:
                raise RuntimeError(
                    "qwen38next_low_latency_sm12x: refusing plan entry "
                    "(N=%d, K=%d, M=%d) %r -- %s" % (n, k, m, cfg, why)
                )
            plan[m] = cfg'''
    assert OLD_GATE in src, "low_latency_gemm.py: _is_sm103 text moved"
    src = src.replace(OLD_GATE, NEW_GATE, 1)

    OLD_ENABLE = """    if dtype != torch.bfloat16 or not _is_sm103():
        return
    if not shape_dynamic_skinny_gemm.is_available():
        return"""
    NEW_ENABLE = """    _llg_sm12x = _llg_sm12x_enabled()
    if dtype != torch.bfloat16 or not (_is_sm103() or _llg_sm12x):
        _llg_signal(0, ())
        return
    if not shape_dynamic_skinny_gemm.is_available():
        _llg_signal(0, ())
        return
    if _llg_sm12x:
        _llg_add_tp1_plans()"""
    assert OLD_ENABLE in src, "low_latency_gemm.py: enable() gate text moved"
    src = src.replace(OLD_ENABLE, NEW_ENABLE, 1)

    # iter6b: count what actually attached, and say so in one INFO line (F7).
    OLD_LOOP = """    warmup_configs: set[SkinnyGemmConfig] = set()
    for child in module.modules():"""
    NEW_LOOP = """    warmup_configs: set[SkinnyGemmConfig] = set()
    _llg_attached: list = []  # iter6b/F7: the (N, K) of every attachment made
    for child in module.modules():"""
    assert OLD_LOOP in src, "low_latency_gemm.py: dispatch loop head moved"
    src = src.replace(OLD_LOOP, NEW_LOOP, 1)

    OLD_TAIL = """        warmup_configs.update(plan.values())

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)"""
    NEW_TAIL = """        warmup_configs.update(plan.values())
        _llg_attached.append((weight.shape[0], weight.shape[1]))

    if warmup_configs:
        shape_dynamic_skinny_gemm.request_warmup_configs(dtype, warmup_configs)

    # Emit exactly one assertable INFO line for the installed state.
    _llg_signal(len(_llg_attached), set(_llg_attached))"""
    assert OLD_TAIL in src, "low_latency_gemm.py: dispatch loop tail moved"
    src = src.replace(OLD_TAIL, NEW_TAIL, 1)

    open(LLG, "w").write(src)
    ast.parse(open(LLG).read())
    print("patch_low_latency_gemm_sm12x.py: low_latency_gemm.py OK", file=sys.stderr)

# --------------------------------------------------------------------------- #
# Part 2: skinny_gemm.py -- the QWEN38NEXT_LLG_PDL=0 variant knob
# --------------------------------------------------------------------------- #
ssrc = open(SKINNY).read()
if "qwen38next_llg_pdl" in ssrc:
    print("patch_low_latency_gemm_sm12x.py: PDL knob already applied", file=sys.stderr)
else:
    OLD_PDL = """    @staticmethod
    def _use_pdl() -> bool:
        from vllm.platforms import current_platform

        return current_platform.is_arch_support_pdl()"""
    NEW_PDL = '''    @staticmethod
    def _use_pdl() -> bool:
        # --- qwen38next_llg_pdl (Iteration 6 / R3) ------------------------
        # QWEN38NEXT_LLG_PDL=0 turns off Programmatic Dependent Launch for the
        # skinny GEMM, so "is griddepcontrol_wait() safe inside a piecewise
        # CUDA graph?" can be measured as a one-bit arm. The value is LATCHED
        # on first call because the compile cache key in _compile() is
        # (dtype, config, has_residual) and does not include use_pdl -- a
        # mid-run flip would silently reuse a kernel compiled the other way.
        global _QWEN38NEXT_LLG_PDL
        if _QWEN38NEXT_LLG_PDL is None:
            import os as _pdl_os

            _QWEN38NEXT_LLG_PDL = _pdl_os.environ.get("QWEN38NEXT_LLG_PDL", "1") != "0"
            if not _QWEN38NEXT_LLG_PDL:
                logger.info("qwen38next_llg_pdl: PDL disabled by QWEN38NEXT_LLG_PDL=0")
        if not _QWEN38NEXT_LLG_PDL:
            return False

        from vllm.platforms import current_platform

        return current_platform.is_arch_support_pdl()'''
    assert OLD_PDL in ssrc, "skinny_gemm.py: _use_pdl text moved"
    ssrc = ssrc.replace(OLD_PDL, NEW_PDL, 1)

    OLD_FLAG = "_cutedsl_available: bool | None = None"
    NEW_FLAG = (
        "_cutedsl_available: bool | None = None\n"
        "# qwen38next_llg_pdl: latched QWEN38NEXT_LLG_PDL, see _use_pdl below.\n"
        "_QWEN38NEXT_LLG_PDL: bool | None = None"
    )
    assert OLD_FLAG in ssrc, "skinny_gemm.py: module flag anchor moved"
    ssrc = ssrc.replace(OLD_FLAG, NEW_FLAG, 1)

    open(SKINNY, "w").write(ssrc)
    ast.parse(open(SKINNY).read())
    print("patch_low_latency_gemm_sm12x.py: skinny_gemm.py PDL knob OK", file=sys.stderr)

print("patch_low_latency_gemm_sm12x.py applied OK", file=sys.stderr)
