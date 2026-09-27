#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""M1-prime (counterfactual draft-mask) + M4 (distinct experts) instrumentation.

Dockerfile-applied patch, in the style of the other src/patch_*.py scripts in
this recipe. It installs one new module, ``vllm_measure_draft_mass``, into
site-packages and appends two guarded hook blocks:

  1. ``vllm/v1/worker/gpu/spec_decode/rejection_sampler.py`` -- wraps
     ``RejectionSampler._verify`` and, per draft position, accumulates the
     acceptance counterfactuals for candidate draft-vocabulary masks.
  2. ``vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`` -- rebinds
     the module-global ``moe_align_block_size`` that ``fused_marlin_moe`` calls
     at :340, so the returned ``num_tokens_post_padded`` device tensor of every
     Marlin MoE layer is registered and sampled once per engine step (M4).

Both blocks are gated on ``VLLM_MEASURE_DRAFT_MASS=1``. With the variable unset
or 0 the appended code is one ``os.environ.get`` at module import time and
nothing else: no wrapper, no allocation, no kernel, byte-identical behaviour.

Apply:  python3 patch_measure_draft_mass.py            (inside the image build)
Test:   the generated module source is exposed as MODULE_SRC for inspection.
"""

import ast
import os
import sys

SP = "/usr/local/lib/python3.12/dist-packages"
MODULE_PATH = SP + "/vllm_measure_draft_mass.py"
RS_PATH = SP + "/vllm/v1/worker/gpu/spec_decode/rejection_sampler.py"
MM_PATH = SP + "/vllm/model_executor/layers/fused_moe/experts/marlin_moe.py"

# Anchors that must be present verbatim in the base image, or the build fails.
RS_ANCHORS = (
    "class RejectionSampler:",
    "processed_logits = self.sampler.apply_sampling_params(",
    "return processed_logits, sampled, num_sampled",
)
MM_ANCHORS = (
    "sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(",
    "def fused_marlin_moe(",
)

SENTINEL = "vllm_measure_draft_mass"


# ---------------------------------------------------------------------------
# The runtime module. Kept as one string so this patch script is the single
# shipped artefact and a host test can exec exactly what the image runs.
# ---------------------------------------------------------------------------
MODULE_SRC = r'''# SPDX-License-Identifier: Apache-2.0
"""Draft-mass / expert-count instrumentation (VLLM_MEASURE_DRAFT_MASS=1).

WHAT IS MEASURED (DRAFT-MASK COUNTERFACTUALS)
------------------------------------------------
Inside ``RejectionSampler._verify`` both distributions of the rejection test are
already materialised, so the counterfactual "what would a proxy draft mask have
done" is available offline, with no change to sampling:

  p  = softmax(processed_logits[row])
       ``processed_logits`` is the tensor returned by
       ``Sampler.apply_sampling_params`` at
       vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:146-154 and fed
       straight into ``rejection_sample`` at :155.  It is an fp32 copy
       (sampler.py:196) carrying logit-bias, penalties, bad-words, thinking
       budget, temperature (sampler.py:233), min_p (:238) and top_k/top_p
       (:244), i.e. -inf outside the target's kept set S_p.
       It is *the same object* the sampler's own target-side logsumexp is taken
       from (rejection_sampler_utils.py:216-217 target_logits_ptr).

  q  = softmax(draft_logits[req_state_idx, k, :] / temperature[req_state_idx])
       ``draft_logits`` is the speculator's cache
       (speculator.py:132-141, dtype = model_config.head_dtype), written by the
       very ``gumbel_sample`` call that draws the proposal
       (speculator.py:329-339, logits_cache=draft_logits), storing the logits
       *before* temperature (gumbel.py:107-123).  The rejection sampler divides
       by the same temperature on load
       (rejection_sampler_utils.py:163-173 and :279-290), which is reproduced
       here exactly.  Under VLLM_MTP_DRAFT_VOCAB the drafter's out-of-slice
       entries are -inf (vllm_mtp_draft_vocab.py:403-406), so q is supported on
       the sliced 65,536-id set and renormalised over it by the softmax -- the
       same q that ``gumbel_sample`` consumed.

  row -> (req_state_idx, k) mapping is the sampler's own:
       k   = expanded_local_pos[row]   (>= num_speculative_steps => bonus row)
       req = expanded_idx_mapping[row]
       exactly as rejection_sampler_utils.py:227-234 does it.

Per row (draft position k = 1..depth), accumulated into preallocated device
tensors -- no ``.item()``, no host sync inside the step:

  A0      = sum(min(p, q))                    expected acceptance at k
  delta   = q(S_p^c) = 1 - sum_{S_p} q        the old M1 number
  deficit = sum_{q < p} q                     A3(iii): the mass a rescale can win
  H       = -sum p log p                      target entropy, the bucket key
  A_T     = sum(min(p, q_T))                  q masked to T and renormalised
  loss_T  = q(S_p and not T)                  target-kept mass the mask deletes
  mass_T  = q(T)                              the mask's own retained mass

for the proxy masks T in {top_k 20, top_k 40, top_k 64, min_p 0.02, min_p 0.05},
all defined on q's *own* order statistics (the implementable lever), and all
bucketed by the target entropy decile.

WHAT IS MEASURED (M4)
---------------------
``fused_marlin_moe`` (marlin_moe.py:340) calls ``moe_align_block_size`` and gets
``num_tokens_post_padded``; ``num_tokens_post_padded / block_size_m`` is the
number of (expert, block) pairs the Marlin grid runs, i.e. the distinct experts
touched by that layer at that verify width.  The wrapper does not read it (that
would sync, and under CUDA-graph replay the Python body never runs).  It
*registers the device tensor*, which is a static graph-pool buffer that every
replay rewrites; the eager ``_verify`` hook then samples every registered slot
once per step with two kernels.

SLOT KEYING (ordinal reset at each pass boundary)
------------------------------------------------------------------
A slot is keyed ``(M, topk, block_size, E, ordinal-within-pass)``. The ordinal
restarts at every verify and at every change of M -- every MoE layer in one
forward pass sees the same token count, so a new M is necessarily a new pass,
and an M change is the only pass boundary that exists during the profile run
and the cudagraph capture, where ``_verify`` is never called. The original
version reset the ordinal only at a verify, so every warm-up registration took
a unique slot, the 256-slot table filled with M = 8192/64/56/48 before the
first decode step, and the verify-width slot never existed. Registrations wider
than ``MOE_MAX_M`` (default 64, this box's ``max_cudagraph_capture_size``) are
skipped and counted separately: the profile run and every prefill chunk are not
what M4 measures. ``overflow`` therefore means one thing only -- the table was
genuinely too small -- and is a gate that must read 0.

READING A SLOT. A slot registered before the first verify
(``before_first_verify``) holds a CUDA-graph pool buffer: it is rewritten by
every replay of that capture size, and if that size is never replayed it keeps
whatever the capture wrote with the profiler's dummy input. A ``blocks_hist``
with a single key and ``samples == steps`` is that case, and is NOT a decode
measurement.

COST -- counted, not guessed
----------------------------
An instrumented dispatch-count run of one observe() under TorchDispatchMode
found 135 dispatches, of which 38 touch a full-width [n, V]
tensor (1 gt, 2 softmax, 1 index gather of the draft row, 3 div, 3 minimum,
10 sum, 8 where, 1 entr, 1 lt, 1 topk, 2 gather, 1 amax, 2 ge, 2 and) and 97 are
[n, 64] or [n] scalars plus the 3 index_add_ into the accumulators.
Zero syncing ops: no item / _local_scalar_dense / nonzero anywhere in the
hot path (asserted by the test).
Summing reads+writes those 38 come to ~61 passes over an [n, V] fp32 tensor.
At n = 4 (one request, depth 3: three draft rows + one bonus row) and
V = 248320 one pass is 3.97 MB, so ~240 MB/step -> ~1.0 ms at 240 GB/s, plus
135 launches at ~2.4 us of eager gap ~= 0.3 ms.  Budget ~1.3 ms/step, i.e.
about 2.4 % of a 55 ms step (~ -1.3 tok/s).  This arm exists to measure
acceptance, not throughput; set VLLM_MEASURE_DRAFT_MASS_EVERY=N to observe one
verify in N and divide the cost by N if a clean tok/s is also wanted.
M4 adds exactly 2 kernels per step (one ``torch._foreach_copy_`` gather of the
registered scalars, one row copy into the ring), independent of the number of
MoE layers AND of the number of slots: the per-slot 0-d views are built once
and cached, and are rebuilt only when a slot is added or its tensor replaced.
Every VLLM_MEASURE_DRAFT_MASS_FLUSH verify calls (default 500) the accumulators
are copied to host and the JSON is rewritten -- that flush *does* sync, once per
~500 steps (~28 s at 55 ms/step).
"""

import atexit
import json
import math
import os
import time

ENV_ON = "VLLM_MEASURE_DRAFT_MASS"
ENV_DIR = "VLLM_MEASURE_DRAFT_MASS_DIR"
ENV_EVERY = "VLLM_MEASURE_DRAFT_MASS_EVERY"
ENV_FLUSH = "VLLM_MEASURE_DRAFT_MASS_FLUSH"
ENV_MOE_MAX_M = "VLLM_MEASURE_DRAFT_MASS_MOE_MAX_M"

SCHEMA = "draft_mass/v1"

# Proxy masks on q's own order statistics. (kind, parameter).
MASKS = (
    ("top_k", 20),
    ("top_k", 40),
    ("top_k", 64),
    ("min_p", 0.02),
    ("min_p", 0.05),
)
MASK_NAMES = ["top_k20", "top_k40", "top_k64", "min_p0.02", "min_p0.05"]
TOPK_MAX = 64

STAT_NAMES = (
    ["A0", "delta", "H", "deficit", "sp_mass"]
    + ["A_" + m for m in MASK_NAMES]
    + ["loss_" + m for m in MASK_NAMES]
    + ["mass_" + m for m in MASK_NAMES]
)
NSTAT = len(STAT_NAMES)  # 20

# Fixed entropy bins. With the harness's top_k = 20 the support of p has at most
# 20 atoms, so H(p) is in [0, ln 20]; ten equal-width bins over that range are
# the "decile" axis. The 64-bin fine histogram is dumped alongside so a reducer
# can recut true empirical deciles offline without re-running.
NBUCKET = 10
H_MAX = math.log(20.0)
NFINE = 64

# Trash rows: bonus / placeholder-draft-token / greedy (temperature == 0).
TRASH_BONUS, TRASH_PLACEHOLDER, TRASH_GREEDY = 0, 1, 2
NTRASH = 3

RING = 1024        # per-step MoE snapshots held on device between flushes
# M4 slot table: the ordinal resets at each pass boundary.
# The key is (M, topk, block_size, E, ordinal-within-pass) and the ordinal now
# restarts at every verify AND at every change of M, so the same
# (capture size, layer, kernel) always lands in the same slot -- including
# during the profile run and the cudagraph capture, where no verify ever runs.
# The population that must fit is n_capture_sizes x 2 x n_layers: 13 x 2 x 48 =
# 1 248 for 13 capture sizes, 1 344 for 14. MAX_SLOTS is a
# BACKSTOP, not a working size -- MOE_MAX_M is what keeps the table small, by
# refusing the profile run and every prefill chunk. Hitting MAX_SLOTS is
# `overflow`, which is a pre-registered GATE and must read 0.
MAX_SLOTS = 4096
# Registrations wider than this are skipped and counted (NOT overflow): M4 is
# about the VERIFY width, and every distinct prefill width would otherwise mint
# a fresh set of 96 slots. 64 is this box's max_cudagraph_capture_size.
MOE_MAX_M_DEFAULT = 64
# A second bound, for a workload that produces many small widths. Also counted,
# also not overflow. 24 > the 14 sizes of the widest capture list in use.
MOE_MAX_M_VALUES = 24
TINY = 1e-30

P_SOURCE = (
    "processed_logits from Sampler.apply_sampling_params, "
    "rejection_sampler.py:146-154 (fp32 copy, sampler.py:196; "
    "temperature :233, min_p :238, top_k/top_p :244)"
)
Q_SOURCE = (
    "softmax(draft_logits[req_state_idx, k] / temperature[req_state_idx]); "
    "draft_logits is the speculator cache speculator.py:132-141 written by "
    "gumbel_sample(logits_cache=...) speculator.py:329-339, stored "
    "pre-temperature gumbel.py:107-123 and divided by the same temperature by "
    "rejection_sampler_utils.py:163-173 / :279-290"
)


def enabled():
    return os.environ.get(ENV_ON, "0") == "1"


def _int_env(name, default):
    try:
        v = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return v if v > 0 else default


class _Logger(object):
    def __init__(self):
        try:
            from vllm.logger import init_logger

            self._log = init_logger("vllm.measure_draft_mass")
        except Exception:
            self._log = None

    def info(self, msg, *a):
        if self._log is not None:
            self._log.info(msg, *a)
        else:
            print("[measure-draft-mass] " + (msg % a if a else msg))

    def warning(self, msg, *a):
        if self._log is not None:
            self._log.warning(msg, *a)
        else:
            print("[measure-draft-mass] WARN " + (msg % a if a else msg))


LOG = _Logger()


class State(object):
    """All accumulators. One instance per process."""

    def __init__(self):
        self.acc = None      # [S*NBUCKET + NTRASH, NSTAT] float64, device
        self.cnt = None      # [S*NBUCKET + NTRASH]        float64, device
        self.hfine = None    # [S*NFINE + 1]               float64, device
        self.moe_ring = None  # [RING, MAX_SLOTS] int32,    device
        self.depth = 0
        self.vocab_size = 0
        self.device = None

        self.verify_calls = 0
        self.observed_calls = 0
        self.rows_seen = 0
        self.identity_processed_calls = 0
        self.no_draft_logits_calls = 0
        self.errors = []

        self.every = _int_env(ENV_EVERY, 1)
        self.flush_every = _int_env(ENV_FLUSH, 500)
        self.out_dir = os.environ.get(ENV_DIR, "/tmp")
        self.path = os.path.join(self.out_dir, "draft_mass_%d.json" % os.getpid())

        # M4
        self.moe_slots = []       # list of dicts, index == slot
        self.moe_slot_of = {}     # key -> slot
        self.moe_ordinal = 0      # WITHIN-PASS index: reset at every verify AND
        self.moe_last_m = None    # at every change of M (see register_moe)
        self.moe_overflow = 0     # MAX_SLOTS was hit -- a GATE, must be 0
        self.moe_ring_pos = 0
        self.moe_ring_skipped = 0
        self.moe_steps = 0
        self.moe_hist = []        # per slot: {blocks: count}
        self.moe_sum = []         # per slot: running sum of blocks
        self.moe_n = []           # per slot: running count
        self.moe_max_m = _int_env(ENV_MOE_MAX_M, MOE_MAX_M_DEFAULT)
        self.moe_m_values = set()      # the distinct M values that took slots
        self.moe_skipped_large_m = 0   # M > moe_max_m: prefill and the profile run
        self.moe_skipped_new_m = 0     # a new M beyond MOE_MAX_M_VALUES
        self.moe_m_switches = 0        # pass boundaries detected by an M change
        self.moe_row = None       # [MAX_SLOTS] int32 gather buffer, device
        self._moe_srcs = None     # cached 0-d views of the registered tensors
        self._moe_dst = None      # cached 0-d views into moe_row
        self._moe_views_dirty = True
        self._moe_foreach = True  # torch._foreach_copy_ available (latched)

        self.t0 = time.time()
        self.dumps = 0
        self.dead = False
        atexit.register(self._atexit)

    # -- allocation ---------------------------------------------------------
    def ensure(self, depth, vocab_size, device):
        import torch

        if self.acc is not None:
            return
        self.depth = int(depth)
        self.vocab_size = int(vocab_size)
        self.device = device
        n = self.depth * NBUCKET + NTRASH
        self.acc = torch.zeros(n, NSTAT, dtype=torch.float64, device=device)
        self.cnt = torch.zeros(n, dtype=torch.float64, device=device)
        self.hfine = torch.zeros(
            self.depth * NFINE + 1, dtype=torch.float64, device=device
        )
        self.moe_ring = torch.zeros(
            RING, MAX_SLOTS, dtype=torch.int32, device=device
        )
        self.moe_row = torch.zeros(MAX_SLOTS, dtype=torch.int32, device=device)
        LOG.info(
            "measure-draft-mass: armed. depth=%d vocab=%d every=%d flush=%d out=%s",
            self.depth,
            self.vocab_size,
            self.every,
            self.flush_every,
            self.path,
        )

    # -- M4: registration and per-step snapshot -----------------------------
    def moe_pass_boundary(self):
        """Start a new registration pass. The ordinal is a WITHIN-PASS index."""
        self.moe_ordinal = 0
        self.moe_last_m = None

    def register_moe(self, topk_ids, block_size, num_experts, npad):
        m = int(topk_ids.shape[0])
        topk = int(topk_ids.shape[1]) if topk_ids.dim() > 1 else 1
        if m > self.moe_max_m:
            # The profile run (M = max_num_batched_tokens) and every prefill
            # chunk. M4 is about the VERIFY width; without this bound each
            # distinct prefill width would mint a fresh set of 2 x n_layers
            # slots. Counted, and deliberately NOT `overflow`, which is a gate.
            self.moe_skipped_large_m += 1
            return
        # The ordinal is a
        # WITHIN-PASS index: it restarts at every verify (moe_pass_boundary,
        # called from snapshot_moe) AND at every change of M. Every MoE layer in
        # one forward pass sees the same token count, so a new M is necessarily
        # a new pass -- and an M change is the ONLY step boundary that exists
        # during the profile run and the whole cudagraph capture, where no
        # verify is ever called. Before this the ordinal was reset only at a
        # verify, so every warm-up registration took a unique slot and the table
        # was full of M = 8192/64/56/48 before the first decode step.
        if self.moe_last_m is not None and m != self.moe_last_m:
            self.moe_ordinal = 0
            self.moe_m_switches += 1
        self.moe_last_m = m
        key = (m, topk, int(block_size), int(num_experts), self.moe_ordinal)
        self.moe_ordinal += 1
        slot = self.moe_slot_of.get(key)
        if slot is None:
            if m not in self.moe_m_values:
                if len(self.moe_m_values) >= MOE_MAX_M_VALUES:
                    self.moe_skipped_new_m += 1
                    return
                self.moe_m_values.add(m)
            if len(self.moe_slots) >= MAX_SLOTS:
                self.moe_overflow += 1
                return
            slot = len(self.moe_slots)
            self.moe_slot_of[key] = slot
            self.moe_slots.append(
                {
                    "slot": slot,
                    "M": m,
                    "topk": topk,
                    "block_size_m": int(block_size),
                    "E": int(num_experts),
                    "ordinal": key[4],
                    "registrations": 0,
                    # Warm-up slots (profile run, cudagraph capture) are the ONLY
                    # ones a replayed graph can ever write to, so this is not a
                    # defect marker -- it is how a reader tells a capture-pool
                    # buffer from an eager one.
                    "before_first_verify": self.verify_calls == 0,
                    # The binding is deliberately NOT pinned: the last tensor
                    # registered for a slot is the one sampled, exactly as
                    # before the fix. These two fields make that visible, so a
                    # degenerate reading can be diagnosed instead of guessed --
                    # a warm-up slot whose pointer moved after the first verify
                    # is an eager pass at a captured width, and its samples
                    # after that point are of a transient allocation.
                    "bindings": 1,
                    "last_bind_at_verify": self.verify_calls,
                    "tensor": npad,
                }
            )
            self.moe_hist.append({})
            self.moe_sum.append(0)
            self.moe_n.append(0)
            self._moe_views_dirty = True
        s = self.moe_slots[slot]
        if s["tensor"] is not npad:
            s["tensor"] = npad
            s["bindings"] += 1
            s["last_bind_at_verify"] = self.verify_calls
            self._moe_views_dirty = True
        s["registrations"] += 1

    def _moe_rebuild_views(self):
        """Cache one 0-d view per slot. Rebuilt only when the slot set moves."""
        self._moe_srcs = [s["tensor"].reshape(-1)[0] for s in self.moe_slots]
        self._moe_dst = [self.moe_row[i] for i in range(len(self.moe_slots))]
        self._moe_views_dirty = False

    def snapshot_moe(self):
        import torch

        self.moe_pass_boundary()
        slots = self.moe_slots
        if not slots or self.moe_ring is None:
            return
        if self.moe_ring_pos >= RING:
            self.moe_ring_skipped += 1
            return
        # TWO KERNELS PER STEP, INDEPENDENT OF THE SLOT COUNT. The per-slot
        # scalar views are built once and cached -- rebuilding 1 200+ of them
        # every step would cost more than the whole M1' accumulation this arm
        # exists for -- then one foreach copy gathers them and one row copy
        # files them in the ring.
        if self._moe_views_dirty or self._moe_srcs is None:
            self._moe_rebuild_views()
        n = len(slots)
        if self._moe_foreach:
            try:
                torch._foreach_copy_(self._moe_dst, self._moe_srcs)
            except Exception:
                self._moe_foreach = False
        if not self._moe_foreach:
            self.moe_row[:n].copy_(torch.stack(self._moe_srcs))
        self.moe_ring[self.moe_ring_pos, :n].copy_(self.moe_row[:n])
        self.moe_ring_pos += 1
        self.moe_steps += 1

    def _drain_moe(self):
        """Host-side; only called from dump(), where a sync is allowed.

        Vectorised: this runs INSIDE a step, every ENV_FLUSH verifies, and the
        old row-by-row python loop is O(ring_pos x slots) -- 500 x 1 248 after
        the slot-table fix.
        """
        if self.moe_ring is None or self.moe_ring_pos == 0:
            return
        n = len(self.moe_slots)
        block = self.moe_ring[: self.moe_ring_pos, :n].cpu()
        try:
            import numpy as np
        except ImportError:
            np = None
        if np is not None:
            arr = block.numpy()
            for slot in range(n):
                bsm = self.moe_slots[slot]["block_size_m"] or 1
                col = arr[:, slot] // bsm
                vals, counts = np.unique(col, return_counts=True)
                h = self.moe_hist[slot]
                for v, c in zip(vals.tolist(), counts.tolist()):
                    h[v] = h.get(v, 0) + c
                self.moe_sum[slot] += int(col.sum(dtype=np.int64))
                self.moe_n[slot] += int(col.shape[0])
        else:
            for row in block.tolist():
                for slot, npad in enumerate(row):
                    bsm = self.moe_slots[slot]["block_size_m"] or 1
                    blocks = int(npad) // int(bsm)
                    h = self.moe_hist[slot]
                    h[blocks] = h.get(blocks, 0) + 1
                    self.moe_sum[slot] += blocks
                    self.moe_n[slot] += 1
        self.moe_ring_pos = 0

    # -- M1-prime: the per-position accumulation ----------------------------
    def observe(
        self,
        processed_logits,
        raw_logits,
        draft_logits,
        draft_sampled,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        depth,
    ):
        import torch

        pl = processed_logits
        if pl.dtype != torch.float32:
            pl = pl.float()
        # rejection_sample uses vocab = min(target V, draft V)
        # (rejection_sampler_utils.py:962); mirror that so the two rows line up
        # even on checkpoints where the target head is padded wider.
        vocab = min(int(pl.shape[1]), int(draft_logits.shape[-1]))
        if vocab != int(pl.shape[1]):
            pl = pl[:, :vocab]
        n = int(pl.shape[0])
        dev = pl.device
        self.ensure(depth, vocab, dev)
        S = self.depth
        if processed_logits is raw_logits:
            self.identity_processed_calls += 1

        # Row -> (req_state_idx, draft step). Same mapping the sampler uses.
        req = expanded_idx_mapping.to(torch.int64)
        pos = expanded_local_pos.to(torch.int64)
        is_bonus = pos >= S
        pos_c = pos.clamp(0, S - 1)
        req_c = req.clamp_min(0)

        temp_row = temperature.to(torch.float32).index_select(0, req_c)
        is_greedy = temp_row <= 0.0

        # A -1 draft token is a placeholder that can never be accepted
        # (rejection_sampler_utils.py:353-355). Row i's draft token is
        # draft_sampled[i + 1] inside the chunk.
        if draft_sampled is not None and int(draft_sampled.shape[0]) == n:
            nxt = torch.cat(
                [draft_sampled[1:], draft_sampled.new_zeros(1)]
            )
            is_ph = nxt < 0
        else:
            is_ph = torch.zeros(n, dtype=torch.bool, device=dev)

        neg_inf = float("-inf")
        keep = pl > neg_inf                       # S_p
        p = torch.softmax(pl, dim=-1)

        dl = draft_logits[req_c, pos_c]           # [n, V], head_dtype
        if int(dl.shape[1]) != vocab:
            dl = dl[:, :vocab]
        q = torch.softmax(
            dl.to(torch.float32) / temp_row.clamp_min(1e-6).unsqueeze(1), dim=-1
        )

        zero = torch.zeros((), dtype=torch.float32, device=dev)

        A0 = torch.minimum(p, q).sum(-1)
        sp_mass = torch.where(keep, q, zero).sum(-1)
        delta = 1.0 - sp_mass
        try:
            H = torch.special.entr(p).sum(-1)
        except AttributeError:                    # pragma: no cover
            H = -(p * torch.log(p.clamp_min(TINY))).sum(-1)
        deficit = torch.where(q < p, q, zero).sum(-1)

        k_max = min(TOPK_MAX, vocab)
        topv, topi = torch.topk(q, k_max, dim=-1)
        p_top = p.gather(1, topi)
        keep_top = keep.gather(1, topi).to(torch.float32)
        qcum = topv.cumsum(-1)
        qkeep_cum = (topv * keep_top).cumsum(-1)
        qmax = q.amax(dim=-1, keepdim=True)

        A_list, loss_list, mass_list = [], [], []
        for kind, param in MASKS:
            if kind == "top_k":
                k = min(int(param), k_max)
                mass = qcum[:, k - 1 : k]                       # [n, 1]
                qT = topv[:, :k] / mass.clamp_min(TINY)
                A = torch.minimum(p_top[:, :k], qT).sum(-1)
                loss = sp_mass - qkeep_cum[:, k - 1]
                massv = mass.squeeze(1)
            else:
                mT = q >= (float(param) * qmax)
                massv = torch.where(mT, q, zero).sum(-1)
                qT = q / massv.clamp_min(TINY).unsqueeze(1)
                A = torch.where(mT, torch.minimum(p, qT), zero).sum(-1)
                loss = sp_mass - torch.where(mT & keep, q, zero).sum(-1)
            A_list.append(A)
            loss_list.append(loss)
            mass_list.append(massv)

        valid = (~is_bonus) & (~is_greedy) & (~is_ph) & (req >= 0)
        bucket = (H * (NBUCKET / H_MAX)).floor().to(torch.int64).clamp(0, NBUCKET - 1)
        base = S * NBUCKET
        trash = torch.where(
            is_bonus,
            base + TRASH_BONUS,
            torch.where(is_ph, base + TRASH_PLACEHOLDER, base + TRASH_GREEDY),
        )
        idx = torch.where(valid, pos_c * NBUCKET + bucket, trash)

        vals = torch.stack(
            [A0, delta, H, deficit, sp_mass] + A_list + loss_list + mass_list, dim=1
        ).to(torch.float64)
        vals = torch.where(
            valid.unsqueeze(1), vals, torch.zeros((), dtype=torch.float64, device=dev)
        )
        ones = torch.ones(n, dtype=torch.float64, device=dev)
        self.acc.index_add_(0, idx, vals)
        self.cnt.index_add_(0, idx, ones)

        fbin = (H * (NFINE / H_MAX)).floor().to(torch.int64).clamp(0, NFINE - 1)
        fidx = torch.where(valid, pos_c * NFINE + fbin, S * NFINE)
        self.hfine.index_add_(0, fidx, ones)

        self.rows_seen += n
        self.observed_calls += 1

    # -- dump ---------------------------------------------------------------
    def to_dict(self):
        acc = self.acc.cpu().tolist() if self.acc is not None else []
        cnt = self.cnt.cpu().tolist() if self.cnt is not None else []
        hfine = self.hfine.cpu().tolist() if self.hfine is not None else []
        S = self.depth
        base = S * NBUCKET
        edges = [i * H_MAX / NBUCKET for i in range(NBUCKET + 1)]

        per_pos = []
        for k in range(S):
            rows = []
            for b in range(NBUCKET):
                i = k * NBUCKET + b
                rows.append({"bucket": b, "n": cnt[i], "sum": acc[i]})
            per_pos.append(
                {
                    "position": k + 1,
                    "buckets": rows,
                    "h_fine": hfine[k * NFINE : (k + 1) * NFINE],
                }
            )

        moe_slots = []
        for s in self.moe_slots:
            slot = s["slot"]
            n = self.moe_n[slot]
            moe_slots.append(
                {
                    "slot": slot,
                    "M": s["M"],
                    "topk": s["topk"],
                    "block_size_m": s["block_size_m"],
                    "E": s["E"],
                    "ordinal": s["ordinal"],
                    "registrations": s["registrations"],
                    "before_first_verify": s.get("before_first_verify"),
                    "bindings": s.get("bindings"),
                    "last_bind_at_verify": s.get("last_bind_at_verify"),
                    "samples": n,
                    "blocks_mean": (self.moe_sum[slot] / n) if n else None,
                    "blocks_hist": {str(k): v for k, v in sorted(
                        self.moe_hist[slot].items())},
                }
            )

        return {
            "schema": SCHEMA,
            "pid": os.getpid(),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "uptime_s": round(time.time() - self.t0, 3),
            "dumps": self.dumps,
            "config": {
                "depth": S,
                "vocab_size": self.vocab_size,
                "every": self.every,
                "flush_every": self.flush_every,
                "masks": MASK_NAMES,
                "stats": STAT_NAMES,
                "entropy_bins": {
                    "n": NBUCKET,
                    "lo": 0.0,
                    "hi": H_MAX,
                    "edges": edges,
                    "fine_bins": NFINE,
                },
                "p_source": P_SOURCE,
                "q_source": Q_SOURCE,
            },
            "counters": {
                "verify_calls": self.verify_calls,
                "observed_calls": self.observed_calls,
                "rows_seen": self.rows_seen,
                "identity_processed_calls": self.identity_processed_calls,
                "no_draft_logits_calls": self.no_draft_logits_calls,
                "n_bonus": cnt[base + TRASH_BONUS] if cnt else 0,
                "n_placeholder": cnt[base + TRASH_PLACEHOLDER] if cnt else 0,
                "n_greedy": cnt[base + TRASH_GREEDY] if cnt else 0,
                "h_fine_trash": hfine[S * NFINE] if hfine else 0,
                "errors": self.errors,
            },
            "positions": per_pos,
            "moe": {
                "steps": self.moe_steps,
                "slots": moe_slots,
                "overflow": self.moe_overflow,
                "ring_skipped": self.moe_ring_skipped,
                "max_slots": MAX_SLOTS,
                "moe_max_m": self.moe_max_m,
                "max_m_values": MOE_MAX_M_VALUES,
                "m_values": sorted(self.moe_m_values),
                "skipped_large_m": self.moe_skipped_large_m,
                "skipped_new_m": self.moe_skipped_new_m,
                "m_switches": self.moe_m_switches,
                "note": (
                    "blocks = num_tokens_post_padded // block_size_m, the number "
                    "of (expert, block) pairs the Marlin grid runs for that layer "
                    "= distinct experts touched at that verify width "
                    "(marlin_moe.py:340)"
                ),
            },
        }

    def dump(self, path=None):
        if self.acc is None:
            return None
        self._drain_moe()
        self.dumps += 1
        payload = self.to_dict()
        path = path or self.path
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        return path

    def _atexit(self):
        try:
            if self.acc is not None:
                p = self.dump()
                LOG.info("measure-draft-mass: final dump %s", p)
        except Exception as exc:            # pragma: no cover
            LOG.warning("measure-draft-mass: atexit dump failed: %r", exc)


STATE = State()


def on_verify(
    sampler_obj,
    raw_logits,
    processed_logits,
    draft_logits,
    draft_sampled,
    expanded_idx_mapping,
    expanded_local_pos,
    depth,
):
    st = STATE
    if st.dead:
        return
    st.verify_calls += 1
    try:
        if draft_logits is None:
            st.no_draft_logits_calls += 1
        elif (st.verify_calls % st.every) == 0:
            temperature = sampler_obj.sampler.sampling_states.temperature.gpu
            st.observe(
                processed_logits,
                raw_logits,
                draft_logits,
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                depth,
            )
        st.snapshot_moe()
        if st.flush_every and (st.verify_calls % st.flush_every) == 0:
            st.dump()
    except Exception as exc:
        st.dead = True
        st.errors.append(repr(exc))
        LOG.warning(
            "measure-draft-mass: disabled after error on verify %d: %r",
            st.verify_calls,
            exc,
        )
        try:
            st.dump()
        except Exception:
            pass


def install_rejection_sampler(cls):
    """Wrap RejectionSampler._verify. Idempotent."""
    if getattr(cls, "_measure_draft_mass_patched", False):
        return cls
    orig = cls._verify

    def _verify(
        self,
        logits,
        draft_logits,
        draft_sampled,
        pos,
        cu_num_logits,
        idx_mapping,
        idx_mapping_np,
        expanded_idx_mapping,
        expanded_local_pos,
    ):
        processed_logits, sampled, num_sampled = orig(
            self,
            logits,
            draft_logits,
            draft_sampled,
            pos,
            cu_num_logits,
            idx_mapping,
            idx_mapping_np,
            expanded_idx_mapping,
            expanded_local_pos,
        )
        on_verify(
            self,
            logits,
            processed_logits,
            draft_logits,
            draft_sampled,
            expanded_idx_mapping,
            expanded_local_pos,
            self.num_speculative_steps,
        )
        return processed_logits, sampled, num_sampled

    cls._verify = _verify
    cls._measure_draft_mass_patched = True
    LOG.info("measure-draft-mass: RejectionSampler._verify wrapped")
    return cls


def wrap_moe_align(fn):
    """Wrap the module-global moe_align_block_size that fused_marlin_moe calls.

    Registers the returned num_tokens_post_padded tensor; never reads it here
    (no sync, and under CUDA-graph replay this body does not run at all -- the
    registered tensor is the graph-pool buffer every replay rewrites).

    NOTE: only the non-batched path is wrapped. BatchedMarlinExperts
    (marlin_moe.py:924, :1025) goes through batched_moe_align_block_size and is
    NOT observed -- it is the EP/DP activation format, not this box's TP=1
    decode path. If it were ever taken, the dump's moe.slots would simply be
    empty, which the arm's gate treats as a failure rather than a zero.
    """
    if getattr(fn, "_measure_draft_mass_wrapped", False):
        return fn

    def wrapped(*args, **kwargs):
        out = fn(*args, **kwargs)
        st = STATE
        if not st.dead:
            try:
                topk_ids = args[0] if args else kwargs.get("topk_ids")
                block_size = (
                    args[1] if len(args) > 1
                    else kwargs.get("block_size", kwargs.get("block_size_m"))
                )
                num_experts = (
                    args[2] if len(args) > 2 else kwargs.get("num_experts")
                )
                st.register_moe(topk_ids, block_size, num_experts, out[2])
            except Exception as exc:
                st.errors.append("register_moe: " + repr(exc))
        return out

    wrapped._measure_draft_mass_wrapped = True
    wrapped.__name__ = getattr(fn, "__name__", "moe_align_block_size")
    wrapped.__doc__ = getattr(fn, "__doc__", None)
    # THE INSTALL LINE. An M4 arm asserts on it (EXPECT_LOG_RE), so that an arm
    # cannot silently use the older slot table, which produced zero-valued
    # measurements and 4 256 overflows before the first decode step.
    LOG.info(
        "measure M4 fixed: slots keyed by step -- key (M, topk, block_size, E, "
        "ordinal-within-pass), ordinal reset at every verify and at every "
        "change of M; MAX_SLOTS=%d moe_max_m=%d max_m_values=%d",
        MAX_SLOTS,
        STATE.moe_max_m,
        MOE_MAX_M_VALUES,
    )
    LOG.info("measure-draft-mass: moe_align_block_size wrapped for Marlin MoE")
    return wrapped
'''


RS_HOOK = '''

# --- appended by patch_measure_draft_mass.py (VLLM_MEASURE_DRAFT_MASS=1) ---
# M1-prime: per-draft-position counterfactual acceptance under proxy draft
# masks, bucketed by target entropy. No-op unless the env var is 1.
import os as _mdm_os

if _mdm_os.environ.get("VLLM_MEASURE_DRAFT_MASS", "0") == "1":
    from vllm_measure_draft_mass import (
        install_rejection_sampler as _mdm_install_rs,
    )

    _mdm_install_rs(RejectionSampler)
'''

MM_HOOK = '''

# --- appended by patch_measure_draft_mass.py (VLLM_MEASURE_DRAFT_MASS=1) ---
# M4: register the num_tokens_post_padded tensor of every Marlin MoE layer.
# fused_marlin_moe resolves moe_align_block_size through this module global, so
# rebinding it here (at import, before any consumer imports the function) is
# enough. No-op unless the env var is 1.
import os as _mdm_os

if _mdm_os.environ.get("VLLM_MEASURE_DRAFT_MASS", "0") == "1":
    from vllm_measure_draft_mass import wrap_moe_align as _mdm_wrap_align

    moe_align_block_size = _mdm_wrap_align(moe_align_block_size)
'''


def _patch(path, anchors, hook):
    with open(path) as fh:
        src = fh.read()
    for a in anchors:
        assert a in src, "anchor missing in %s: %r" % (path, a)
    if SENTINEL in src:
        print("already patched: %s" % path, file=sys.stderr)
        return
    src += hook
    with open(path, "w") as fh:
        fh.write(src)
    ast.parse(open(path).read())
    print("patched %s" % path, file=sys.stderr)


def main():
    with open(MODULE_PATH, "w") as fh:
        fh.write(MODULE_SRC)
    ast.parse(MODULE_SRC)
    print("wrote %s (%d bytes)" % (MODULE_PATH, len(MODULE_SRC)), file=sys.stderr)

    _patch(RS_PATH, RS_ANCHORS, RS_HOOK)
    _patch(MM_PATH, MM_ANCHORS, MM_HOOK)

    # The default-off contract: importing either patched module with the
    # variable unset must not import the measurement module at all.
    assert os.environ.get(ENV_ON_CHECK, "0") != "1", (
        "do not build with VLLM_MEASURE_DRAFT_MASS=1 in the build environment"
    )
    print("patch_measure_draft_mass.py applied OK", file=sys.stderr)


ENV_ON_CHECK = "VLLM_MEASURE_DRAFT_MASS"

if __name__ == "__main__":
    main()
