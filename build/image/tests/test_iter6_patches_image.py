#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""CPU image regression checks."""

import ast
import hashlib
import os
import subprocess
import sys

SP = "/usr/local/lib/python3.12/dist-packages"
PATCH_DIR = os.environ.get("ITER6_PATCH_DIR", "/selftest/src")

LLG = SP + "/vllm/models/qwen3_8_flash_next/nvidia/low_latency_gemm.py"
SKINNY = SP + "/vllm/model_executor/kernels/linear/cute_dsl/skinny_gemm.py"
GDN = SP + "/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
GEMV_MOD = SP + "/vllm_inproj_ba_gemv.py"
PLE = SP + "/vllm_ple_mmap.py"
TOPK_OPS = SP + "/vllm/v1/sample/ops/topk_topp_sampler.py"
STATES = SP + "/vllm/v1/worker/gpu/sample/states.py"
GPU_SAMPLER = SP + "/vllm/v1/worker/gpu/sample/sampler.py"
REJ = SP + "/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"

FAILURES = []


def check(cond, msg):
    if cond:
        print("  ok   %s" % msg)
    else:
        print("  FAIL %s" % msg)
        FAILURES.append(msg)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def parses(path):
    ast.parse(open(path).read())
    return True


def rerun(script, targets):
    """Re-run a patch script; assert it is a no-op on every target file."""
    before = {p: sha(p) for p in targets}
    proc = subprocess.run(
        [sys.executable, os.path.join(PATCH_DIR, script)],
        capture_output=True,
        text=True,
    )
    check(proc.returncode == 0, "%s re-run exits 0 (stderr: %s)" % (
        script, proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""))
    check(
        "already applied" in proc.stderr,
        "%s re-run reports 'already applied'" % script,
    )
    for p in targets:
        check(sha(p) == before[p], "%s unchanged by %s re-run" % (
            p.rsplit("/", 1)[-1], script))


# --------------------------------------------------------------------------- #
# Patch 1 -- R3, low-latency GEMM on SM12x + the PDL knob
# --------------------------------------------------------------------------- #
def test_llg():
    print("[1] patch_low_latency_gemm_sm12x.py")
    src = open(LLG).read()
    check(parses(LLG), "low_latency_gemm.py parses")
    check("qwen38next_low_latency_sm12x" in src, "SM12x block present")
    check("QWEN38NEXT_LOW_LATENCY_GEMM" in src, "env gate present")
    check("_llg_validate" in src, "plan validator present")
    emitted = src.split("def _llg_add_tp1_plans")[1].split("extra = [")[1].split("]")[0]
    check(
        "(10240, 320)" not in emitted,
        "(10240, 320) is NOT emitted: K mod 128 == 64, and the kernel raises "
        "ValueError on that shape (the validator moves the refusal to build time)",
    )
    for shape in ("(320, 10240)", "(96, 2560)"):
        check(shape in emitted, "%s plan entry emitted" % shape)
    gate = src.split("def _llg_sm12x_enabled")[1].split("def _llg_validate")[0]
    check(
        '"QWEN38NEXT_LOW_LATENCY_GEMM", "0"' in gate.replace("'", '"'),
        "gate reads QWEN38NEXT_LOW_LATENCY_GEMM defaulting to 0",
    )
    check(
        'is_device_capability_family(120)' in gate,
        "gate widens to the SM 12.x family only",
    )
    check(
        "raise RuntimeError" in src.split("def _llg_add_tp1_plans")[1],
        "an invalid plan entry raises rather than being emitted",
    )
    # iter6b F7: exactly one assertable install line, either way.
    check(src.count('_llg_logger.info("iter6 R3 disabled")') == 1,
          "R3 emits exactly one 'iter6 R3 disabled' line")
    check(src.count('"iter6 R3 installed: %d dispatch attachments, '
                    'accepted shapes %s, PDL=%s"') == 1,
          "R3 emits exactly one 'iter6 R3 installed: <count> ...' line")
    check(src.count("_llg_signal(0, ())") == 2 and
          src.count("_llg_signal(len(_llg_attached), set(_llg_attached))") == 1,
          "every exit of enable_qwen38next_low_latency_gemm signals exactly once")
    check("_llg_attached.append((weight.shape[0], weight.shape[1]))" in src,
          "the dispatch attachments are counted, with their (N, K)")
    check("_llg_init_logger(__name__)" in src,
          "R3 logs on the engine-core logger (vllm init_logger)")

    ssrc = open(SKINNY).read()
    check(parses(SKINNY), "skinny_gemm.py parses")
    check("qwen38next_llg_pdl" in ssrc, "PDL knob present")
    check("_QWEN38NEXT_LLG_PDL: bool | None = None" in ssrc, "PDL latch declared")
    check(
        '"QWEN38NEXT_LLG_PDL", "1"' in ssrc.replace("'", '"'),
        "PDL knob defaults to 1 (stock behaviour) and only 0 disables",
    )
    rerun("patch_low_latency_gemm_sm12x.py", [LLG, SKINNY])


# --------------------------------------------------------------------------- #
# Patch 2 -- R7, in_proj_ba GEMV
# --------------------------------------------------------------------------- #
def test_gemv():
    print("[2] patch_in_proj_ba_gemv.py")
    check(os.path.exists(GEMV_MOD), "vllm_inproj_ba_gemv.py installed")
    check(parses(GEMV_MOD), "vllm_inproj_ba_gemv.py parses")
    src = open(GDN).read()
    check(parses(GDN), "qwen_gdn_linear_attn.py parses")
    check("_in_proj_ba_gemv_apply" in src, "hook block appended")
    head = src.split("# --- qwen38-flash-dgx: in_proj_ba small-N GEMV")[0]
    check(
        "VLLM_INPROJ_BA_GEMV" not in head,
        "the guard did not leak outside the appended block",
    )
    check(
        "ba, _ = self.in_proj_ba(hidden_states)" in head,
        "forward_cuda call site is UNTOUCHED (quant_method swap, not a rewrite)",
    )
    # Default-off contract: apply()'s first executable statement must be the
    # is_enabled() short circuit, so nothing at all happens with the env unset.
    mod = open(GEMV_MOD).read()
    tree = ast.parse(mod)
    fn = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "apply"
        ),
        None,
    )
    check(fn is not None, "vllm_inproj_ba_gemv.apply() is a module-level function")
    stmts = [s for s in (fn.body if fn else []) if not (
        isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
        and isinstance(s.value.value, str))]
    first = stmts[0] if stmts else None
    ok = (
        isinstance(first, ast.If)
        and isinstance(first.test, ast.UnaryOp)
        and isinstance(first.test.op, ast.Not)
        and isinstance(first.test.operand, ast.Call)
        and getattr(first.test.operand.func, "id", "") == "is_enabled"
        # iter6b: the body is now `logger.info("iter6 R7 disabled"); return` --
        # one log line and the return, and still nothing else.
        and len(first.body) == 2
        and isinstance(first.body[0], ast.Expr)
        and isinstance(first.body[1], ast.Return)
    )
    check(ok, "apply()'s first statement is the is_enabled() short circuit, whose "
              "whole body is the disabled log line plus `return`")
    check('ENV_ENABLE = "VLLM_INPROJ_BA_GEMV"' in mod,
          "env name is VLLM_INPROJ_BA_GEMV")
    check("_register_op()" in mod and mod.index("if not is_enabled()")
          < mod.index("    _register_op()"),
          "the custom op is registered only after the enable check")
    # iter6b F7/F8: installed_count() finally has a caller, and it prints.
    check(mod.count('logger.info("iter6 R7 disabled")') == 1,
          "R7 emits exactly one 'iter6 R7 disabled' line")
    check(mod.count('"iter6 R7 installed: %d in_proj_ba layers on the Triton GEMV "')
          == 1,
          "R7 emits exactly one 'iter6 R7 installed: <count> ...' line")
    check("def signal_installed()" in mod
          and "        installed_count()," in mod,
          "installed_count() is actually called and printed (F8)")
    sig = mod.split("def signal_installed()")[1]
    check("_env_int(ENV_MAX_M, _DEFAULT_MAX_M)," in sig
          and "_DEFAULT_MAX_M," in sig
          and "iter6c default cap %d" in sig,
          "the install line carries the M threshold AND the iter6c default cap")
    # iter6c: the BLOCK_M = 16 containment. The default cap is the whole fix --
    # if it drifts back to 16 the wrong-answer path is reachable again.
    check("_DEFAULT_MAX_M = 8" in mod,
          "iter6c: the default MAX_M is 8, so BLOCK_M can only be 1, 2, 4 or 8")
    check("_env_int(ENV_MAX_M, 16)" not in mod,
          "iter6c: no MAX_M default of 16 survives anywhere in the module")
    check("or x.shape[0] > _env_int(ENV_MAX_M, _DEFAULT_MAX_M)" in mod,
          "iter6c: gemv() takes the F.linear fallback above the capped default")
    check("KNOWN DEFECT -- BLOCK_M = 16 PRODUCES WRONG ANSWERS" in mod,
          "the BLOCK_M = 16 defect banner is still there (contained, not fixed)")
    check("def process_weights_after_loading" in mod
          and "signal_installed()" in mod.split(
              "def process_weights_after_loading")[1],
          "the count is signalled once the model is built, not at patch time")
    check("in_proj_ba GEMV enabled (" not in mod,
          "the old free-text enable line is gone -- exactly one install line")
    rerun("patch_in_proj_ba_gemv.py", [GDN])


# --------------------------------------------------------------------------- #
# Patch 3 -- R2, PLE rendezvous
# --------------------------------------------------------------------------- #
SYNC_TOKENS = (
    ".item()",
    "non_blocking=False",
    ".cpu()",
    ".tolist()",
    "torch.cuda.synchronize",
    "stream.synchronize",
)


def test_ple():
    print("[3] patch_ple_rendezvous.py")
    src = open(PLE).read()
    check(parses(PLE), "vllm_ple_mmap.py parses")
    check("_ids_to_cpu_async" in src, "async ids helper present")
    check("VLLM_PLE_MMAP_ASYNC_IDS" in src, "env gate present")
    check(
        '_ASYNC_IDS_MODE in ("1", "2")' in src,
        "gate accepts 1 (async) and 2 (async + wait accounting)",
    )
    # The original synchronous line must still be present as the OFF branch.
    check(
        'ids.detach().to("cpu", non_blocking=False).numpy().reshape(-1)' in src,
        "the original pageable-D2H line survives as the default-off branch",
    )
    check(
        "if _ASYNC_IDS" in src,
        "the async path is selected by a latched module-level flag",
    )
    # No sync op may survive on the ENABLED branch.
    helper = src.split("def _ids_to_cpu_async")[1].split("\n    def forward")[0]
    # The two documented fallbacks (non-CUDA / no pinned memory) legitimately use
    # the synchronous copy; strip them before scanning.
    scan = "\n".join(
        ln
        for ln in helper.splitlines()
        if 'return src.to("cpu", non_blocking=False).numpy()' not in ln
    )
    for tok in SYNC_TOKENS:
        check(tok not in scan, "no %r on the enabled step path" % tok)
    check("non_blocking=True" in helper, "the staged copy is non_blocking=True")
    check("torch.cuda.Event()" in helper and "ev.record()" in helper,
          "a CUDA event is recorded")
    check("ev.synchronize()" in helper, "the consumer waits on the event")
    check("pin_memory=True" in src.split("def _pinned_ids")[1].split("def ")[0],
          "the staging buffer is pinned")
    # R2 is dormant. It prints the disabled line and deliberately has no
    # `installed:` line because its mode-2 counters are unreadable.
    check(src.count('logger.info("iter6 R2 disabled")') == 1,
          "R2 emits exactly one 'iter6 R2 disabled' line")
    check("iter6 R2 installed" not in src,
          "R2 has NO install line while its mode-2 counters are unreadable")
    rerun("patch_ple_rendezvous.py", [PLE])


# --------------------------------------------------------------------------- #
# Patch 4 -- L5a, verify-path truncation
# --------------------------------------------------------------------------- #
def test_topk():
    print("[4] patch_verify_topk_pivot.py")
    for p in (TOPK_OPS, STATES, GPU_SAMPLER, REJ):
        check(parses(p), "%s parses" % p.rsplit("/", 1)[-1])
    ops = open(TOPK_OPS).read()
    check("_verify_topk_triton" in ops, "env latch present in topk_topp_sampler.py")
    check(
        '"VLLM_VERIFY_TOPK_TRITON", "0"' in ops.replace("'", '"'),
        "gate defaults to 0",
    )
    check(
        "if HAS_TRITON and logits.shape[0] >= 8:" in ops,
        "the ordinary decode path's >= 8 heuristic is UNCHANGED",
    )
    check(
        "prefer_triton and _verify_topk_triton" in ops,
        "prefer_triton is honoured only when the env is set",
    )
    check("logits.dtype == torch.float32" in ops, "fp32 precondition asserted")
    check("prefer_triton: bool = False" in open(STATES).read(),
          "states.py threads prefer_triton")
    check("prefer_triton: bool = False" in open(GPU_SAMPLER).read(),
          "gpu/sample/sampler.py threads prefer_triton")
    check("prefer_triton=True" in open(REJ).read(),
          "rejection_sampler.py:_verify passes prefer_triton=True")
    # iter6b F7.
    check(ops.count('logger.info("iter6 L5a disabled")') == 1,
          "L5a emits exactly one 'iter6 L5a disabled' line")
    check(ops.count('"iter6 L5a installed: 1 sampler hook on apply_top_k_top_p "')
          == 1,
          "L5a emits exactly one 'iter6 L5a installed: 1 sampler hook ...' line")
    sig = ops.split("iter6 L5a installed")[1].split("else:")[0]
    check("fp32" in sig,
          "the L5a install line names the fp32 precondition")
    rerun("patch_verify_topk_pivot.py", [TOPK_OPS, STATES, GPU_SAMPLER, REJ])


def main():
    for name in (
        "QWEN38NEXT_LOW_LATENCY_GEMM",
        "QWEN38NEXT_LLG_PDL",
        "VLLM_INPROJ_BA_GEMV",
        "VLLM_PLE_MMAP_ASYNC_IDS",
        "VLLM_VERIFY_TOPK_TRITON",
    ):
        os.environ.pop(name, None)
    test_llg()
    test_gemv()
    test_ple()
    test_topk()
    if FAILURES:
        print("\n%d FAILURES:" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        sys.exit(1)
    print("\nall iteration-6 in-image patch checks passed")


main()
