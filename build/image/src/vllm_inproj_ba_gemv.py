# SPDX-License-Identifier: Apache-2.0
"""Iteration 6 / R7: a small-N Triton GEMV for the GDN ``in_proj_ba`` projection.

*** KNOWN DEFECT -- BLOCK_M = 16 PRODUCES WRONG ANSWERS (iteration 6b) ***
Measured 2026-09-09 13:25 UTC against qwen38-flash-dgx:iter6b-20260909.
``test_in_proj_ba_gemv_gpu.py`` FAILS G1 and G2 at M = 16: max deviation
**202 / 110.6 / 156 bf16 ulp** at scale 1 / 0.05 / 8 against a 1-ulp gate, and an
exact-match fraction of **0.562 / 0.587 / 0.578** against a >= 0.99 gate. A direct
sweep isolates it to ``BLOCK_M`` alone:

    BLOCK_M in {1, 2, 4, 8} and BLOCK_M = 32  -- clean (<= 1 ulp) at every
        BLOCK_K in {32, 64, 128, 256, 512} and every BLOCK_N in {1, 2, 4}
    BLOCK_M = 16                              -- WRONG at all of them:
        ~42 % of elements differ, deviations 97 ulp to 4.7e27 ulp

Since ``BLOCK_M = next_power_of_2(m)``, that is **every batch of 9 to 16 rows** --
which includes concurrency 4's M = 16 and any CUDA-graph capture size in that
range. This is a wrong-answer bug, not a rounding difference, and it is inside the
default ``VLLM_INPROJ_BA_GEMV_MAX_M = 16``.

CONTAINED AT ITERATION 6c, NOT FIXED. The default ``VLLM_INPROJ_BA_GEMV_MAX_M`` is
now ``_DEFAULT_MAX_M = 8`` (was 16), so ``BLOCK_M`` can only ever be 1, 2, 4 or 8 --
every value the sweep above found clean -- and **every batch of 9 rows or more takes
the ``F.linear`` fallback instead**, which gate G4 verifies is bit-identical. The
kernel itself is untouched: the ``[16, 1, BLOCK_K]``
broadcast-and-``tl.sum(axis=2)`` codegen fault is still there and still reachable by
setting ``VLLM_INPROJ_BA_GEMV_MAX_M`` to 9 or more by hand. Do not do that.
``test_in_proj_ba_gemv_gpu.py``'s G7 sweeps M = 1..32 end to end against the
shipped default and must pass before using this kernel in a serving image.

WHY
---
The measured ``cutlass::Kernel2 (8,1,1)`` runs **36 times per step,
18.9 us each, to read 0.47 MiB** -- 26 GB/s, 0.679 ms/step, the worst-behaved kernel in
the model by bandwidth, of which ~0.60 ms is pure launch/latency floor (the measured
launch floor on this box is 3.4 us). The tensor is the fused GDN ``in_proj_ba``:

    checkpoint headers (iter4/tensor-headers.json):
        model.language_model.layers.<0..35>.linear_attn.in_proj_a.weight  BF16 [48, 2560]
        model.language_model.layers.<0..35>.linear_attn.in_proj_b.weight  BF16 [48, 2560]
      -> 72 tensors, 36 layers x 2

    vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:562-584
        create_ba_proj() builds a MergedColumnParallelLinear with
        output_sizes = [num_v_heads] * 2 = [48, 48]  (gqa_interleaved_layout=False,
        set at models/qwen3_8_flash_next/nvidia/model.py:208-212), input_size = 2560,
        bias=False (:580)
      -> one bf16 weight of [96, 2560]: the ``(96, 2560)`` shape the decomposition
         attributes to that kernel.

    Call sites in the live path:
        qwen_gdn_linear_attn.py:890   forward_cuda   ``ba, _ = self.in_proj_ba(...)``
        (the HIP/XPU/CPU twins at :848, :969, :1009 are not reached on this box)

    It stays **bf16 and unquantised**: qwen_gdn_linear_attn.py:419 says "ba_proj
    doesn't support blockwise fp8 quantization", the headers are BF16, and the
    decomposition puts the kernel in the skinny-bf16 family. So its quant_method is
    ``UnquantizedLinearMethod`` and ``apply()`` is exactly ``F.linear(x, W, None)``
    (linear.py:204-212).

WHAT
----
``_in_proj_ba_gemv_kernel`` -- **one CTA per output column** (``BLOCK_N=1`` by default,
so 96 CTAs over the box's 48 SMs = two clean waves), 128 threads (4 warps) strided over
K, contiguous unmasked K loads, fp32 accumulate, single bf16 round on store.

LOAD WIDTH. The earlier "128-bit loads" claim is
**withdrawn**. At the production config -- ``BLOCK_M=1, BLOCK_N=1, BLOCK_K=512,
num_warps=4`` -- 512 bf16 spread over 128 lanes is **4 bf16 per thread = 64-bit loads**.
Eight bf16 per thread would need ``BLOCK_K >= 1024``, which ``_MAX_BLOCK_K = 512``
forbids. This is a performance claim only; correctness is unaffected. Verifying it
either way means dumping the TTGIR/PTX and looking for ``ld.global.v8.b16`` / ``.b128``,
and raising ``_MAX_BLOCK_K`` is worth one kernel-sweep data point. Each CTA reads
1 x 2560 x 2 B = 5 KiB of weight, so the 0.47 MiB weight slice is read exactly once from
HBM; the activation ([M, 2560] bf16, 5-40 KiB) is re-read by every CTA but stays
L2-resident.

Registered as ``torch.ops.vllm.qwen_gdn_in_proj_ba_gemv`` via
``direct_register_custom_op`` with a fake kernel, so ``torch.compile`` sees an opaque
node with a known output meta and CUDA-graph capture records **one** node -- the same
shape of registration ``low_latency_gemm.py:146-150`` uses.

EXACTNESS
---------
**Not** bit-identical to ``F.linear``, and cannot be: cuBLAS/CUTLASS picks its own fp32
accumulation order over K=2560 and that order is not reproducible from outside. This
kernel's order is fixed and deterministic (tree reduction inside each ``BLOCK_K`` chunk,
sequential across chunks); both paths accumulate in fp32 from the same bf16 inputs and
round once to bf16 on store. ``test_in_proj_ba_gemv_gpu.py`` therefore asserts **within
1 ulp of bf16**, not bit-identity. Concretely: fp32 accumulation over 2560 terms carries
~1e-6 relative error while one bf16 ulp is ~4e-3 relative, so the two paths differ only
when the exact value lands within ~1e-6 relative of a bf16 rounding boundary -- of order
1e-4 of elements. The test reports the exact-match fraction beside the 1-ulp bound.

Determinism is **per capture size**, not across them:
``BLOCK_M = next_power_of_2(m)`` feeds ``choose_block_k``, so BLOCK_K is 512 for M <= 8
and 256 for M = 16, a different fp32 reduction tree and therefore possibly a different
last bf16 ulp for the same row at a different M. ``F.linear`` has the same property, so
this is inside the declared class, but it is not "bit-identical across batch sizes".

Label: **near-lossless** (fp-associativity only), the same class as R3 and
``max_autotune``.

FALLBACKS -- each returns ``F.linear``:
  * ``VLLM_INPROJ_BA_GEMV`` != 1  (the method is never installed at all)
  * Triton unavailable, or not CUDA
  * dtype is not bf16, or x/W are not 2-D with a unit last stride
  * ``M > VLLM_INPROJ_BA_GEMV_MAX_M`` (default 8 since iteration 6c; was 16)
    -- prefill, concurrency 4's M = 16, and every wide graph capture
  * a bias is present, or ``VLLM_BATCH_INVARIANT``
The ``M`` test is a *shape* branch. Under CUDA-graph capture the shape is fixed per
capture, so whichever branch runs at capture time is the one recorded; there is no
data-dependent control flow inside a graph.
"""

from __future__ import annotations

import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover - the image always has triton
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False

ENV_ENABLE = "VLLM_INPROJ_BA_GEMV"
ENV_MAX_M = "VLLM_INPROJ_BA_GEMV_MAX_M"
ENV_BLOCK_N = "VLLM_INPROJ_BA_GEMV_BLOCK_N"
ENV_BLOCK_K = "VLLM_INPROJ_BA_GEMV_BLOCK_K"

# Iteration 6c: the default row cap. 8, not 16, because BLOCK_M = next_power_of_2(m)
# and BLOCK_M = 16 is a wrong-answer defect (see the banner at the top). At 8 the only
# reachable BLOCK_M values are 1, 2, 4 and 8, all measured clean; M >= 9 falls through
# to the bit-identical F.linear path that gate G4 verifies. Raising it by hand
# re-opens the defect.
_DEFAULT_MAX_M = 8

# Keep the [BLOCK_M, BLOCK_N, BLOCK_K] fp32 product tile at <= 32 registers per
# thread across 4 warps. 4 warps * 32 lanes * 32 = 4096 fp32.
_TILE_BUDGET = 4096
_MAX_BLOCK_K = 512
_MIN_BLOCK_K = 32


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = int(raw)
    except ValueError:
        logger.warning("%s=%r is not an integer; using %d", name, raw, default)
        return default
    return v if v > 0 else default


def is_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "0") == "1"


def choose_block_k(k: int, block_m: int, block_n: int) -> tuple[int, bool]:
    """Largest power-of-two BLOCK_K that fits the register budget and divides K.

    Returns ``(block_k, even_k)``. ``even_k`` says the K loop needs no mask, which
    is what lets Triton vectorise the contiguous axis at all. It does NOT by itself
    buy 128-bit loads: at the default BLOCK_K=512 over 4 warps each thread takes
    4 bf16, i.e. 64 bit. See the module docstring, LOAD WIDTH (D6/F9).
    K = 2560 = 2**9 * 5, so every power of two up to 512 divides it and the
    common case always returns ``even_k=True``.
    """
    forced = _env_int(ENV_BLOCK_K, 0)
    if forced:
        return forced, (k % forced == 0)
    budget = max(_MIN_BLOCK_K, _TILE_BUDGET // max(1, block_m * block_n))
    bk = min(_MAX_BLOCK_K, 1 << (max(budget, _MIN_BLOCK_K).bit_length() - 1))
    while bk > _MIN_BLOCK_K and k % bk:
        bk //= 2
    return bk, (k % bk == 0)


if HAS_TRITON:

    @triton.jit
    def _in_proj_ba_gemv_kernel(
        X,  # [M, K] bf16
        W,  # [N, K] bf16
        Y,  # [M, N] bf16
        M,
        stride_xm,
        stride_wn,
        stride_ym,
        K: tl.constexpr,
        N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        EVEN_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = tl.arange(0, BLOCK_M)
        m_mask = offs_m < M
        n_mask = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            if EVEN_K:
                # No K mask: the contiguous BLOCK_K run is what lets Triton
                # vectorise the load at all -- 4 bf16 (64 bit) per thread at the
                # default BLOCK_K=512 over 4 warps, not 128-bit (D6/F9). Masks on
                # M and N are on the outer axis and do not inhibit that.
                x = tl.load(
                    X + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=m_mask[:, None],
                    other=0.0,
                )
                w = tl.load(
                    W + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=n_mask[:, None],
                    other=0.0,
                )
            else:
                k_mask = offs_k < K
                x = tl.load(
                    X + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
                w = tl.load(
                    W + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=n_mask[:, None] & k_mask[None, :],
                    other=0.0,
                )
            acc += tl.sum(
                x.to(tl.float32)[:, None, :] * w.to(tl.float32)[None, :, :], axis=2
            )

        tl.store(
            Y + offs_m[:, None] * stride_ym + offs_n[None, :],
            acc.to(Y.dtype.element_ty),
            mask=m_mask[:, None] & n_mask[None, :],
        )


def _shapes_ok(x: torch.Tensor, w: torch.Tensor) -> bool:
    return (
        x.dim() == 2
        and w.dim() == 2
        and x.dtype == torch.bfloat16
        and w.dtype == torch.bfloat16
        and x.is_cuda
        and w.is_cuda
        and x.device == w.device
        and x.shape[1] == w.shape[1]
        and x.stride(1) == 1
        and w.stride(1) == 1
    )


def gemv(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The op body, also importable directly by the unit test."""
    if (
        not HAS_TRITON
        or envs.VLLM_BATCH_INVARIANT
        or not _shapes_ok(x, weight)
        or x.shape[0] > _env_int(ENV_MAX_M, _DEFAULT_MAX_M)
    ):
        return torch.nn.functional.linear(x, weight)

    m, k = x.shape
    n = weight.shape[0]
    block_n = _env_int(ENV_BLOCK_N, 1)
    block_m = max(1, triton.next_power_of_2(m))
    block_k, even_k = choose_block_k(k, block_m, block_n)

    out = x.new_empty((m, n))
    grid = (triton.cdiv(n, block_n),)
    _in_proj_ba_gemv_kernel[grid](
        x,
        weight,
        out,
        m,
        x.stride(0),
        weight.stride(0),
        out.stride(0),
        K=k,
        N=n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        EVEN_K=even_k,
        num_warps=4,
        num_stages=2,
    )
    return out


def _gemv_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


_OP_REGISTERED = False


def _register_op() -> None:
    global _OP_REGISTERED
    if _OP_REGISTERED or hasattr(torch.ops.vllm, "qwen_gdn_in_proj_ba_gemv"):
        _OP_REGISTERED = True
        return
    direct_register_custom_op(
        op_name="qwen_gdn_in_proj_ba_gemv",
        op_func=gemv,
        fake_impl=_gemv_fake,
    )
    _OP_REGISTERED = True


class InProjBaGemvLinearMethod(UnquantizedLinearMethod):
    """UnquantizedLinearMethod with apply() routed to the Triton GEMV.

    Only ``apply`` is overridden, so ``create_weights`` and
    ``process_weights_after_loading`` behave exactly as before -- the same shape of
    override ``low_latency_gemm.py:104-119`` uses.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # First point at which installed_count() is final (see signal_installed).
        signal_installed()
        return super().process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        signal_installed()  # net, in case process_weights_after_loading is skipped
        if bias is None and not envs.VLLM_BATCH_INVARIANT:
            return torch.ops.vllm.qwen_gdn_in_proj_ba_gemv(x, layer.weight)
        return super().apply(layer, x, bias)


_INSTALLED = 0
_SKIPPED: list[str] = []
_SIGNALLED = False
_FIRST_PREFIX = "<none>"
_FIRST_SHAPE: tuple = ()


def installed_count() -> int:
    return _INSTALLED


def signal_installed() -> None:
    """Emit EXACTLY ONE INFO line naming ``installed_count()``, at install time.

    ``installed_count()`` existed but **no caller read it**, so if zero of the
    36 layers installed, a benchmark would measure the control and
    report a clean sub-noise null -- indistinguishable from "the lever does not
    help". This is the caller.

    Timing: every ``_maybe_install()`` runs from the wrapped
    ``QwenGatedDeltaNetAttention.__init__``, i.e. during model construction, so by
    the time vLLM walks the layers to finalise weights
    (``process_weights_after_loading``) the count is already final. That is the
    trigger, with the first ``apply()`` as a net. Both are well inside the
    harness's ready timeout, so a variant file can gate on this line
    (``EXPECT_LOG_RE``).

    If the knob is set and **zero** layers install, ``InProjBaGemvLinearMethod`` is
    never instantiated, neither trigger fires and no ``iter6 R7`` line appears at
    all -- which is exactly what makes the arm fail its EXPECT_LOG_RE rather than
    return a false null. The per-reason ``logger.warning`` in ``_maybe_install``
    says why.
    """
    global _SIGNALLED
    if _SIGNALLED:
        return
    _SIGNALLED = True
    logger.info(
        "iter6 R7 installed: %d in_proj_ba layers on the Triton GEMV "
        "(first %s, weight %s, MAX_M=%d (iter6c default cap %d; rows above it take "
        "the bit-identical F.linear fallback, because BLOCK_M=16 is defective), "
        "BLOCK_N=%d, skipped: %s)",
        installed_count(),
        _FIRST_PREFIX,
        _FIRST_SHAPE,
        _env_int(ENV_MAX_M, _DEFAULT_MAX_M),
        _DEFAULT_MAX_M,
        _env_int(ENV_BLOCK_N, 1),
        "; ".join(_SKIPPED) or "none",
    )


def _maybe_install(mod) -> None:
    proj = getattr(mod, "in_proj_ba", None)
    if proj is None:
        return
    global _INSTALLED
    qm = getattr(proj, "quant_method", None)
    why = None
    if type(qm) is not UnquantizedLinearMethod:
        why = "quant_method is %s, not UnquantizedLinearMethod" % type(qm).__name__
    elif getattr(proj, "bias", None) is not None:
        why = "layer has a bias"
    elif getattr(proj, "skip_bias_add", False):
        why = "skip_bias_add is set"
    else:
        w = getattr(proj, "weight", None)
        if w is None or w.dim() != 2:
            why = "weight is missing or not 2-D"
    if why is not None:
        if why not in _SKIPPED:
            _SKIPPED.append(why)
            logger.warning(
                "in_proj_ba GEMV: skipping %s -- %s",
                getattr(proj, "prefix", "<unknown>"),
                why,
            )
        return
    proj.quant_method = InProjBaGemvLinearMethod()
    _INSTALLED += 1
    if _INSTALLED == 1:
        # Recorded, not logged: the single install line is emitted once the whole
        # model is built, so that it can carry the final installed_count() (F8).
        global _FIRST_PREFIX, _FIRST_SHAPE
        _FIRST_PREFIX = getattr(proj, "prefix", "<unknown>")
        _FIRST_SHAPE = tuple(proj.weight.shape)


def apply(cls) -> None:
    """Install the GEMV on every QwenGatedDeltaNetAttention's ``in_proj_ba``.

    With ``VLLM_INPROJ_BA_GEMV`` unset this returns before touching anything: the
    class is not wrapped, no op is registered, and every forward is byte-identical
    to the stock image.

    The swap happens in ``__init__``, which runs before
    ``models/qwen3_8_flash_next/nvidia/model.py:636`` calls
    ``enable_qwen38next_low_latency_gemm``; that function's
    ``type(child.quant_method) is UnquantizedLinearMethod`` test
    (low_latency_gemm.py:166) then skips ``in_proj_ba``, so R7 takes precedence over
    R3's ``(96, 2560)`` plan entry with no double-install.
    """
    if not is_enabled():
        logger.info("iter6 R7 disabled")
        return
    if getattr(cls, "_in_proj_ba_gemv_patched", False):
        return
    if not HAS_TRITON:
        logger.warning(
            "%s=1 but triton is not importable; leaving in_proj_ba on F.linear",
            ENV_ENABLE,
        )
        return
    _register_op()

    _orig_init = cls.__init__

    def __init__(self, *args, **kwargs):  # noqa: N807
        _orig_init(self, *args, **kwargs)
        _maybe_install(self)

    cls.__init__ = __init__
    cls._in_proj_ba_gemv_patched = True
    # No log line here on purpose: the single `iter6 R7 installed: <count> ...`
    # line is emitted by signal_installed() once the count is final (F7/F8).
