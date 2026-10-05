#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Upstream per-request dynamic draft depth (myllmbox v5.2 mbx_depth), installer.

Mirrors the five hook sites from myllmbox/qwen38-flash-next-vllm:v5.2 onto
this tree (hook text copied from that image's vllm):
  base speculator load_model -> mbx_depth.init(self)
  model_runner execute_model -> trim after update_requests/apply_staged_writes
  model_runner execute_model -> observe after self.sample(...)
  autoregressive _multi_step_decode + _generate_fused_drafts -> depth() bounds
  structured_outputs apply_grammar_bitmask -> trim_bitmask
Policy math is untouched in /opt/mbx/lib/libmbx_mtp.so. With mode off (or no
.so / no JSON) every hook is a pass-through.
"""
import ast
import sys

SP = "/usr/local/lib/python3.12/dist-packages/vllm"
BASE = SP + "/v1/worker/gpu/spec_decode/speculator.py"
AUTO = SP + "/v1/worker/gpu/spec_decode/autoregressive/speculator.py"
RUNNER = SP + "/v1/worker/gpu/model_runner.py"
STRUCT = SP + "/v1/worker/gpu/structured_outputs.py"

IMPORT = "\ntry:  # MBX depth\n    import mbx_depth as _mbx_depth\nexcept ImportError:\n    _mbx_depth = None\n"


def edit(path, old, new, count=1):
    src = open(path).read()
    n = src.count(old)
    assert n == count, f"{path}: anchor found {n} times (want {count}):\n{old[:200]}"
    open(path, "w").write(src.replace(old, new))


def need_import(path):
    if "import mbx_depth as _mbx_depth" not in open(path).read():
        with open(path, "a") as fh:
            fh.write(IMPORT)


# --- base speculator: init -------------------------------------------------------
edit(
    BASE,
    "self.model = self.load_draft_model(target_model, target_attn_layer_names)\n"
    "        self._validate_local_argmax_reduction()",
    "self.model = self.load_draft_model(target_model, target_attn_layer_names)\n"
    "        self._validate_local_argmax_reduction()\n"
    "        if _mbx_depth is not None:  # MBX depth\n"
    "            _mbx_depth.init(self)",
)
need_import(BASE)

# --- model_runner: trim ------------------------------------------------------------
edit(
    RUNNER,
    "self.update_requests(scheduler_output)\n"
    "            self.block_tables.apply_staged_writes()",
    "self.update_requests(scheduler_output)\n"
    "            self.block_tables.apply_staged_writes()\n"
    "            if _mbx_depth is not None and _mbx_depth.D.on:  # MBX depth\n"
    "                _mbx_depth.trim(scheduler_output, self)",
)

# --- model_runner: observe -----------------------------------------------------------
edit(
    RUNNER,
    "sampler_output, num_sampled, num_rejected = self.sample(\n"
    "            hidden_states, input_batch, grammar_output\n"
    "        )",
    "sampler_output, num_sampled, num_rejected = self.sample(\n"
    "            hidden_states, input_batch, grammar_output\n"
    "        )\n"
    "        if _mbx_depth is not None and _mbx_depth.D.on:  # MBX depth\n"
    "            _mbx_depth.observe(input_batch.idx_mapping, num_sampled, num_rejected)",
)
need_import(RUNNER)

# --- autoregressive: _multi_step_decode bound ------------------------------------------
edit(
    AUTO,
    "slot_mappings_by_layer = None\n"
    "        for step in range(1, self.num_speculative_steps):\n"
    "            # Rebuild every step when positions advance, or just once",
    "slot_mappings_by_layer = None\n"
    "        _mbx_d = (_mbx_depth.depth(self.num_speculative_steps)  # MBX depth\n"
    "                  if _mbx_depth is not None else self.num_speculative_steps)\n"
    "        for step in range(1, _mbx_d):\n"
    "            # Rebuild every step when positions advance, or just once",
)
edit(
    AUTO,
    "num_tokens_across_dp=num_tokens_across_dp,\n"
    "                    cudagraph_runtime_mode=batch_desc.cg_mode,\n"
    "                )",
    "num_tokens_across_dp=num_tokens_across_dp,\n"
    "                    cudagraph_runtime_mode=batch_desc.cg_mode,\n"
    "                )\n"
    "        if _mbx_d < self.num_speculative_steps:  # MBX depth\n"
    "            _mbx_depth.fill(self, num_reqs, _mbx_d)",
)

# --- autoregressive: _generate_fused_drafts bound ---------------------------------------
edit(
    AUTO,
    "for step in range(1, self.num_speculative_steps):\n"
    "            self.current_draft_step.fill_(step)",
    "_mbx_d = (_mbx_depth.depth(self.num_speculative_steps, cudagraph_runtime_mode)  # MBX depth\n"
    "                  if _mbx_depth is not None else self.num_speculative_steps)\n"
    "        for step in range(1, _mbx_d):\n"
    "            self.current_draft_step.fill_(step)",
)
edit(
    AUTO,
    "step < self.num_speculative_steps - 1",
    "step < _mbx_d - 1",
)
edit(
    AUTO,
    "for attn_group in attn_groups:\n"
    "                    attn_group.update_draft_decode_metadata(attn_metadata)",
    "for attn_group in attn_groups:\n"
    "                    attn_group.update_draft_decode_metadata(attn_metadata)\n"
    "        if _mbx_d < self.num_speculative_steps:  # MBX depth\n"
    "            _mbx_depth.fill(self, num_reqs, _mbx_d)",
)
need_import(AUTO)

# --- structured outputs: bitmask trim ----------------------------------------------------
edit(
    STRUCT,
    "if not grammar_req_ids:\n"
    "            return",
    "if not grammar_req_ids:\n"
    "            return\n"
    "        if _mbx_depth is not None and _mbx_depth.D.on:  # MBX depth: rows of the drafts the trim removed\n"
    "            grammar_bitmask = _mbx_depth.trim_bitmask(grammar_req_ids, grammar_bitmask)",
)
need_import(STRUCT)

for path in (BASE, AUTO, RUNNER, STRUCT):
    ast.parse(open(path).read())
print("patch_mbx_depth.py applied OK")
sys.exit(0)
