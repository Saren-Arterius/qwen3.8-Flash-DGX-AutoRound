"""mbx_depth -- per-request dynamic draft depth (upstream: myllmbox/qwen38-flash-next-vllm:v5.2).

Verbatim from /usr/local/lib/python3.12/dist-packages/mbx_depth.py in that
image except the CFG default: upstream reads /cache/mbx-depth.json (their
container mounts a cache dir); this tree has no /cache mount and vLLM spawns
EngineCore/workers with a scrubbed env, so the default is the file serve.sh
writes and mounts, /tmp/mtp_depth.json (MBX_DEPTH_CFG still overrides).
Policy math lives in /opt/mbx/lib/libmbx_mtp.so (same image, copied with it):
per-request depth in [min, K] from the deepest position's acceptance rate over
`window` steps (+2/+1 promote, -1/-2 demote); each step's batch drafts the
requests' average depth; the JSON is re-read live (mtime poll).

Magi deltas from upstream (documented at each site): the promote bar sits at
the configured 70/55 instead of 60/45, and requests start at the configured
min instead of mbx_d_start(). Step magnitudes (+2/+1/-1/-2), demote side,
batch averaging, trim/observe, and the live config schema are untouched.
"""
import ctypes, json, os
import torch
from vllm.logger import init_logger

log = init_logger("vllm.mbx_depth")
CFG = os.environ.get("MBX_DEPTH_CFG", "/tmp/mtp_depth.json")
LIB = os.environ.get("MBX_DEPTH_LIB", "/opt/mbx/lib/libmbx_mtp.so")
_POLL = 4


class _State:
    on = False


D = _State()
_mtime = [None]
_L = None


def _lib():
    global _L
    if _L is None:
        _L = ctypes.CDLL(LIB)
        _L.mbx_d_set.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int] + [ctypes.c_double] * 4
        _L.mbx_d_ready.argtypes = [ctypes.c_long, ctypes.c_long]
        _L.mbx_d_next.argtypes = [ctypes.c_int, ctypes.c_double]
        _L.mbx_d_batch.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    return _L


def _read():
    try:
        m = os.stat(CFG).st_mtime
    except OSError:
        return None
    if m == _mtime[0]:
        return None
    _mtime[0] = m
    try:
        return json.load(open(CFG))
    except (OSError, ValueError) as e:
        log.warning("mbx depth: bad %s (%s), keeping the previous settings", CFG, e)
        return None


def _tp():
    try:
        from vllm.distributed.parallel_state import get_tp_group
        g = get_tp_group()
        return g if g.world_size > 1 else None
    except Exception:
        return None


def _apply(c):
    D.mode = str(c.get("mode", "off")).lower()
    D.log = bool(c.get("log", False))
    up = c.get("promote", [60, 45]); dn = c.get("demote", [25, 15])
    _lib().mbx_d_set(int(c.get("min", 3)), D.K, int(c.get("window", 48)),
                     float(up[0]), float(up[1]), float(dn[0]), float(dn[1]))
    D.min, D.up, D.dn = int(c.get("min", 3)), (float(up[0]), float(up[1])), (float(dn[0]), float(dn[1]))
    D.active = D.mode == "dynamic"
    log.info("mbx depth: %s · min %s · max %d · window %s · promote %s · demote %s · log %s",
             D.mode, c.get("min", 3), D.K, c.get("window", 48), up, dn, "on" if D.log else "off")


def _cfg(sync):
    """Every tensor-parallel rank must decide the same: rank 0 reads, the group takes its values (sync calls happen at the
    same step on every rank)."""
    tp = _tp()
    if tp is None:
        c = _read()
    elif sync:
        c = tp.broadcast_object(_read() if tp.rank_in_group == 0 else None, src=0)
    else:
        return
    if c is not None:
        _apply(c)


def init(spec):
    K = int(spec.num_speculative_steps)
    if K < 2 or not os.path.exists(LIB):
        return
    try:
        dev, R = spec.device, int(spec.max_num_reqs)
        D.K, D.R, D.steps, D.active, D.log, D.mode = K, R, 0, False, False, "off"
        D.min, D.up, D.dn = 3, (60.0, 45.0), (25.0, 15.0)
        _cfg(sync=True)
        ring = _lib().mbx_d_ring()
        D.ring = ring
        D.depth = [D.min] * R  # magi delta: start at configured min (upstream: mbx_d_start())
        D.own = torch.full((R,), D.min, dtype=torch.int32, device=dev)
        D.batch = D.drafted = K
        D.held = False
        D.acc = torch.zeros(R, ring, K, dtype=torch.float32, device=dev)
        D.ptr = torch.zeros(R, dtype=torch.long, device=dev)
        D.last = [0] * R
        D.pos = torch.arange(1, K + 1, device=dev, dtype=torch.int32)
        D.nspec, D.cut_d = {}, K
        D.on = True
    except Exception as e:
        D.on = False
        log.warning("mbx depth off: %r", e)


def depth(K, cg_mode=None):
    if not D.on or not D.active or K != D.K or torch.cuda.is_current_stream_capturing():
        return K
    if cg_mode is not None and getattr(cg_mode, "name", "") == "FULL":
        return K
    D.drafted = D.batch
    return D.batch


def fill(spec, num_reqs, d):
    return


def note_tokens(idx_mapping, sampled, num_sampled):
    return


def _set(i, k):
    D.depth[i] = k
    D.own[i] = k


def _reset(i):
    D.ptr[i] = 0
    D.last[i] = 0


def trim(so, runner=None):
    if not D.on:
        return
    idx = getattr(getattr(runner, "req_states", None), "req_id_to_index", None)
    if not D.active:
        D.batch = D.drafted = D.K
        return
    if idx is not None:
        for r in getattr(so, "scheduled_new_reqs", ()) or ():
            i = idx.get(r.req_id)
            if i is not None:
                _set(i, _lib().mbx_d_start())
                _reset(i)
        live = [idx[r] for r in so.num_scheduled_tokens if r in idx]
        held = ctypes.c_int(0)
        arr = (ctypes.c_int * max(len(live), 1))(*[D.depth[i] for i in live])
        b = _lib().mbx_d_batch(len(live), arr, ctypes.byref(held))
        if D.held and not held.value:
            for i in live:
                _reset(i)
        D.held, D.batch = bool(held.value), b
    D.nspec = {}
    d, spec = D.drafted, so.scheduled_spec_decode_tokens
    if d >= D.K or not spec:
        return
    D.nspec = {rid: len(t) for rid, t in spec.items()}
    D.cut_d = d
    cut = 0
    for rid, toks in spec.items():
        n = len(toks)
        if n > d:
            spec[rid] = toks[:d]
            so.num_scheduled_tokens[rid] -= n - d
            cut += n - d
    so.total_num_scheduled_tokens -= cut


def trim_bitmask(grammar_req_ids, bitmask):
    if not D.on or not D.active or not D.nspec:
        return bitmask
    keep, off, d = [], 0, D.cut_d
    for rid in grammar_req_ids:
        n = D.nspec.get(rid, 0)
        keep.extend(range(off, off + 1 + min(n, d)))
        off += 1 + n
    if off != bitmask.shape[0] or len(keep) == off:
        return bitmask
    return bitmask[keep]


def observe(idx_mapping, num_sampled, num_rejected):
    if not D.on:
        return
    D.steps += 1
    if D.steps % 64 == 0:
        _cfg(sync=True)
    if not D.active:
        return
    acc = (num_sampled.to(torch.int32) - 1).clamp(min=0)
    rows = idx_mapping.long()
    # a request drafted shallower than its own depth has no reading at that depth: those steps don't count
    live = (acc + num_rejected.to(torch.int32) > 0) & (D.own[rows] <= D.drafted)
    col = D.ptr[rows] % D.ring
    D.acc[rows, col] = ((acc[:, None] >= D.pos[None, :]) & live[:, None]).to(torch.float32)
    D.ptr[rows] += live.long()
    if D.steps % _POLL == 0 and not D.held:
        _decide()


def _decide():
    L = _lib()
    W = L.mbx_d_window()
    ptr = D.ptr.tolist()
    for i, n in enumerate(ptr):
        if not L.mbx_d_ready(n, D.last[i]):
            continue
        k = D.depth[i]
        idx = torch.tensor([(n - 1 - j) % D.ring for j in range(W)], device=D.acc.device)
        rate = 100.0 * float(D.acc[i].index_select(0, idx)[:, k - 1].sum()) / W
        D.last[i] = n
        # No magi delta at this site: mbx_d_next resolves the configured bars
        # (70/55/35/25) with upstream magnitudes; the clamp is belt-and-braces
        # (the .so floors at max(configured min, 2)).
        new = L.mbx_d_next(k, rate)
        new = max(D.min, min(D.K, new))
        if D.log:
            log.info("mbx depth · req %d · depth %d → %d · position %d accepted %.1f %% of the last %d steps",
                     i, k, new, k, rate, W)
        if new != k:
            _set(i, new)
            _reset(i)
