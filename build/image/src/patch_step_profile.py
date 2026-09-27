#!/usr/bin/env python3
"""On-demand torch.profiler around GPUModelRunner.execute_model.

This build has no VLLM_TORCH_PROFILER_DIR / /start_profile. With
VLLM_STEP_PROFILE=1, touching /tmp/profile_trigger inside the container makes
the next 24 engine steps run under torch.profiler (CPU+CUDA) and writes a
chrome trace to /tmp/step_profile_<n>.json (docker cp it out). Zero cost when
idle: one os.path.exists per step.
"""

import ast
import sys

MR = "/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/model_runner.py"

src = open(MR).read()
assert "class GPUModelRunner" in src and "_step_profile" not in src

src += '''

# --- appended by patch_step_profile.py (VLLM_STEP_PROFILE=1) ---------------
import os as _sp_os

if _sp_os.environ.get("VLLM_STEP_PROFILE", "0") == "1":
    _sp_orig_execute = GPUModelRunner.execute_model
    _sp_state = {"prof": None, "steps": 0, "n": 0}
    _SP_TRIGGER = "/tmp/profile_trigger"
    _SP_STEPS = int(_sp_os.environ.get("VLLM_STEP_PROFILE_STEPS", "24"))

    def _step_profile_execute(self, *args, **kwargs):
        st = _sp_state
        if st["prof"] is None and _sp_os.path.exists(_SP_TRIGGER):
            try:
                _sp_os.remove(_SP_TRIGGER)
            except OSError:
                pass
            import torch.profiler as _tp

            st["prof"] = _tp.profile(
                activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA]
            )
            st["prof"].__enter__()
            st["steps"] = 0
            logger.info("[step-profile] capturing %d steps", _SP_STEPS)
        out = _sp_orig_execute(self, *args, **kwargs)
        if st["prof"] is not None:
            st["steps"] += 1
            if st["steps"] >= _SP_STEPS:
                st["prof"].__exit__(None, None, None)
                st["n"] += 1
                path = f"/tmp/step_profile_{st['n']}.json"
                st["prof"].export_chrome_trace(path)
                logger.info("[step-profile] trace written to %s", path)
                st["prof"] = None
        return out

    GPUModelRunner.execute_model = _step_profile_execute
'''
open(MR, "w").write(src)
ast.parse(open(MR).read())
print("patch_step_profile.py applied OK", file=sys.stderr)
