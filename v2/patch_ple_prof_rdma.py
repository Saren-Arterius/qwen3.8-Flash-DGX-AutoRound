# --- recipe-bench (magi): PLE read profiler + RDMA row source, appended to ngram_embedding.py ---
# MBX_PLE_PROF=1        : time the row reads (the part RDMA would replace), log every MBX_PLE_PROF_EVERY s
# MBX_PLE_RDMA=host:port: rows via one-sided RDMA READs from wtako's ple_nvfp4_rdma_server (no local
#                         table reads at all: no nvme map, capture-time lookups return zeros)
# Hooks module globals that _mbx_t_gather_impl / _mbx_mmap_attach look up at call time, so the
# breakable-cudagraph split point is unchanged: same graph pieces as stock nvme mode.
import os as _rb_os, threading as _rb_th, time as _rb_time

_RB = {"calls": 0, "ids": 0, "rows": 0, "read_s": 0.0, "gather_s": 0.0, "t_log": _rb_time.time()}
_RB_LOCK = _rb_th.Lock()
_RB_SRC = "rdma" if _rb_os.environ.get("MBX_PLE_RDMA") else "nvme"


def _rb_note(n_ids, n_rows, read_s, gather_s):
    if _rb_os.environ.get("MBX_PLE_PROF") != "1":
        return
    with _RB_LOCK:
        _RB["calls"] += 1; _RB["ids"] += n_ids; _RB["rows"] += n_rows
        _RB["read_s"] += read_s; _RB["gather_s"] += gather_s
        now = _rb_time.time()
        if now - _RB["t_log"] >= float(_rb_os.environ.get("MBX_PLE_PROF_EVERY", "30")):
            c = max(_RB["calls"], 1)
            print(f"PLE-PROF src={_RB_SRC} calls={_RB['calls']} ids={_RB['ids']} uniq_rows={_RB['rows']} "
                  f"read_total={_RB['read_s']:.3f}s gather_total={_RB['gather_s']:.3f}s "
                  f"read_avg={_RB['read_s'] / c * 1e3:.3f}ms gather_avg={_RB['gather_s'] / c * 1e3:.3f}ms "
                  f"rows/call={_RB['rows'] / c:.0f}", flush=True)
            _RB["t_log"] = now


class _RbRdma:
    """One QP to the NVFP4 server; READs land in a pinned MR buffer of 90-byte rows.

    Like v1's client: a failed READ (server restart, link drop) never raises. The dead session is closed, the
    client redials every 2 s until the server is back, and the READ is retried. The engine stalls loudly
    meanwhile, with no fallback to local rows. The initial connect waits the same way."""

    MAX = 1 << 18

    def __init__(self, endpoint):
        import ctypes
        self.host, port = endpoint.rsplit(":", 1); self.port = int(port)
        L = ctypes.CDLL(_rb_os.environ.get("PLE_RDMA_LIB", "/opt/rb/libple_rdma.so"))
        L.ple_dev_open.restype = ctypes.c_void_p; L.ple_dev_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        L.ple_mr_reg.restype = ctypes.c_void_p; L.ple_mr_reg.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        L.ple_mr_lkey.restype = ctypes.c_uint32; L.ple_mr_lkey.argtypes = [ctypes.c_void_p]
        L.ple_qp_create.restype = ctypes.c_void_p; L.ple_qp_create.argtypes = [ctypes.c_void_p]
        L.ple_qp_local.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint32)]
        L.ple_qp_connect.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
        L.ple_qp_destroy.argtypes = [ctypes.c_void_p]
        L.ple_read_rows.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint64,
                                    ctypes.c_uint32, ctypes.c_void_p, ctypes.c_long, ctypes.c_int]
        self.dev_name = _rb_os.environ.get("VLLM_PLE_RDMA_DEV", "roceP2p1s0f0")
        base = f"/sys/class/infiniband/{self.dev_name}/ports/1"
        self.gid_i = next(i for i in range(16) if _rb_os.path.exists(f"{base}/gids/{i}")
                          and open(f"{base}/gid_attrs/types/{i}").read().strip() == "RoCE v2"
                          and open(f"{base}/gids/{i}").read().startswith("0000:0000:0000:0000:0000:ffff:"))
        self.dev = L.ple_dev_open(self.dev_name.encode(), 1, self.gid_i)
        assert self.dev, f"PLE rdma: cannot open {self.dev_name}"
        self.L, self.buf, self.lkey, self.qp, self.sock = L, None, None, None, None
        self.lock = _rb_th.Lock()
        self._connect_forever()

    def _connect(self):
        """TCP exchange + fresh QP. The MR buffer is registered once, on the first successful connect."""
        import ctypes, json, socket
        import numpy as np
        s = socket.create_connection((self.host, self.port), timeout=15)
        try:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    raise ConnectionError("server closed during exchange")
                buf += chunk
            r = json.loads(buf)
            if r.get("format") != "nvfp4":
                raise RuntimeError(f"server format {r.get('format')!r}, want nvfp4")
            rb = int(r["row_bytes"])
            if self.buf is None:
                self.buf = np.empty((self.MAX, rb), dtype=np.uint8)
                mr = self.L.ple_mr_reg(self.dev, self.buf.ctypes.data, self.buf.nbytes, 0)
                assert mr, "PLE rdma: MR reg failed (memlock ulimit?)"
                self.rb, self.lkey = rb, self.L.ple_mr_lkey(mr)
            elif rb != self.rb:
                raise RuntimeError(f"server row_bytes {rb} != {self.rb}")
            qp = self.L.ple_qp_create(self.dev)
            if not qp:
                raise RuntimeError("QP create failed")
            gid, qpn = ctypes.create_string_buffer(16), ctypes.c_uint32()
            self.L.ple_qp_local(qp, gid, ctypes.byref(qpn))
            s.sendall(json.dumps({"gid": gid.raw.hex(), "qpn": qpn.value}).encode() + b"\n")
            if self.L.ple_qp_connect(qp, bytes.fromhex(r["gid"]), r["qpn"]) != 0:
                self.L.ple_qp_destroy(qp)
                raise RuntimeError("QP connect failed")
            s.settimeout(None)
        except BaseException:
            s.close()
            raise
        self.remote, self.qp, self.sock = r, qp, s  # sock kept open = session alive

    def _close(self):
        """Drop the dead session first, so the server sees it end and takes the new one."""
        if self.qp:
            self.L.ple_qp_destroy(self.qp); self.qp = None
        if self.sock:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def _connect_forever(self):
        attempt = 0
        while True:
            try:
                self._connect()
                break
            except Exception as e:  # noqa: BLE001
                attempt += 1
                print(f"PLE rdma: connect to {self.host}:{self.port} failed ({e}) — retry {attempt} in 2s, "
                      "stalling (no fallback)", flush=True)
                _rb_time.sleep(2.0)
        r = self.remote
        print(f"PLE: rows over RDMA from {self.host}:{self.port} ({r['rows']} rows x {self.rb} B, "
              f"dev {self.dev_name} gid {self.gid_i})" + (f" after {attempt} retries" if attempt else ""), flush=True)

    def read(self, rows):
        """rows: contiguous int64 [n] -> uint8 [n, row_bytes] view of the MR buffer. Blocks until it succeeds."""
        import numpy as np
        n = rows.size
        assert n <= self.MAX, n
        ids = np.ascontiguousarray(rows, dtype=np.uint64)
        attempt = 0
        with self.lock:
            while True:
                rc = self.L.ple_read_rows(self.qp, self.lkey, self.buf.ctypes.data, self.remote["addr"],
                                          self.remote["rkey"], ids.ctypes.data, n, self.rb)
                if rc == 0:
                    if attempt:
                        print(f"PLE rdma: READ recovered after {attempt} retries", flush=True)
                    return self.buf[:n]
                attempt += 1
                print(f"PLE rdma: READ failed (rc={rc}) — reconnect {attempt} in 2s, stalling (no fallback)", flush=True)
                self._close()
                _rb_time.sleep(2.0)
                self._connect_forever()


_RB_CLIENT = []


def _rb_client():
    if not _RB_CLIENT:
        _RB_CLIENT.append(_RbRdma(_rb_os.environ["MBX_PLE_RDMA"]))
    return _RB_CLIENT[0]


def _mbx_t_gather(layer, ids):  # noqa: F811 — replaces the nvme reader above
    import numpy as _np
    t0 = _rb_time.perf_counter()
    uniq, inv = _np.unique(ids.detach().cpu().numpy().astype(_np.int64), return_inverse=True)
    cp, cs = layer._mbx_mm_cols_p, layer._mbx_mm_cols_s
    rows = _np.ascontiguousarray(uniq)
    t1 = _rb_time.perf_counter()
    if _RB_SRC == "rdma":
        c = _rb_client()
        assert (c.remote["packed_bytes"], c.remote["scales_bytes"]) == (cp, cs), (c.remote, cp, cs)
        got = c.read(rows)
        op, os_ = got[:, :cp], got[:, cp:]
    else:
        lib = _mbx_t_lib()
        n = len(rows)
        op = _np.empty((n, cp), dtype=_np.uint8); os_ = _np.empty((n, cs), dtype=_np.uint8)
        rc = lib.mbx_t_read(layer._mbx_t_fd, rows.ctypes.data, n, op.ctypes.data, os_.ctypes.data, cp, cs,
                            0 if _MBX_T.get("q") else int(_mbx_os.environ.get("MBX_PLE_NVME_THREADS", "32")))
        if rc:
            raise RuntimeError(f"PLE: table read failed ({rc})")
    t2 = _rb_time.perf_counter()
    out = torch.from_numpy(op[inv]), torch.from_numpy(os_[inv]).view(torch.float8_e4m3fn)  # fancy index = copy
    _rb_note(int(ids.numel()), len(rows), t2 - t1, _rb_time.perf_counter() - t0)
    return out


if _RB_SRC == "rdma":
    _rb_orig_setup = _mbx_t_setup

    def _mbx_t_setup(layer):  # noqa: F811 — no nvme map: nothing is read from local table files
        c = _rb_client()
        S, NS = layer._mbx_mm_S, layer._mbx_mm_n
        assert (c.remote["shard_rows"], c.remote["rows"]) == (S, S * NS), (c.remote, S, NS)
        g = float(layer.nvfp4_global)
        assert abs(c.remote["nvfp4_global"] - g) <= 1e-6 * abs(g), (c.remote["nvfp4_global"], g)
        print("PLE: nvme map skipped (MBX_PLE_RDMA)", flush=True)

    _rb_orig_mmap_gather = _mbx_mmap_gather

    def _mbx_mmap_gather(layer, ids):  # noqa: F811 — only reached during graph capture in rdma mode
        n = ids.shape[0]
        return (torch.zeros((n, layer._mbx_mm_cols_p), dtype=torch.uint8, device=ids.device),
                torch.zeros((n, layer._mbx_mm_cols_s), dtype=torch.float8_e4m3fn, device=ids.device))

    def _mbx_mmap_attach(layer):  # noqa: F811 — rdma: no table mmap, no prewarm thread; geometry from config
        cfg = _mbx_ple_nvfp4_cfg()
        if int(layer._mbx_mmap_rows) < int(cfg["rows"]):
            raise RuntimeError(f"PLE rdma needs the whole table on this rank ({layer._mbx_mmap_rows} < {cfg['rows']})")
        layer._mbx_mm_S = int(cfg["shard_rows"]); layer._mbx_mm_n = int(cfg["shards"])
        layer._mbx_mm_cols_p = int(layer.nvfp4_packed.shape[1]); layer._mbx_mm_cols_s = int(layer.nvfp4_scales.shape[1])
        layer._mbx_name = f"t{len(_MBX_T_LAYERS)}"
        _MBX_T_LAYERS[layer._mbx_name] = layer
        _mbx_t_setup(layer)
        print("PLE: table files not opened (MBX_PLE_RDMA)", flush=True)

    # Loader: never read the ple-nvfp4-* files. The table tensors only feed a row count in mmap mode, so yield
    # zero-storage stand-ins with the right shape[0]; nvfp4_global comes from the server (same value, asserted
    # against config geometry in _mbx_t_setup / _mbx_t_gather).
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader as _RbDML
    _rb_orig_prepare, _rb_orig_iter = _RbDML._prepare_weights, _RbDML._get_weights_iterator

    def _rb_prepare(self, *a, **kw):
        import json
        folder, files, use_st = _rb_orig_prepare(self, *a, **kw)
        by_file = {}
        for name, f in json.load(open(_rb_os.path.join(folder, "model.safetensors.index.json")))["weight_map"].items():
            by_file.setdefault(f, []).append(name)
        table = {f for f, names in by_file.items() if all(".nvfp4_" in n for n in names)}
        self._rb_skipped = sorted(n for f in table for n in by_file[f])
        kept = [p for p in files if _rb_os.path.basename(p) not in table]
        print(f"PLE: loader skips {len(files) - len(kept)} table files (MBX_PLE_RDMA)", flush=True)
        return folder, kept, use_st

    def _rb_iter(self, source):
        yield from _rb_orig_iter(self, source)
        S = int(_mbx_ple_nvfp4_cfg()["shard_rows"])
        for name in self.__dict__.pop("_rb_skipped", ()):
            if name.endswith("nvfp4_global"):
                t = torch.tensor(float(_rb_client().remote["nvfp4_global"]), dtype=torch.float32)
            else:
                t = torch.empty((1, 1), dtype=torch.uint8).expand(S, 1)
            yield source.prefix + name, t

    _RbDML._prepare_weights, _RbDML._get_weights_iterator = _rb_prepare, _rb_iter


# --- magi-v2: give back torch's cached pinned host memory after load and after warm-up ---------------------------
# torch's CachingHostAllocator rounds pinned requests up to a power of two and never returns freed blocks to the OS;
# on GB10 that is plain system RAM (a 4 GiB block seen resident in the worker after boot). Flush it once the one-off
# users (weight load, warm-up/capture) are done; logs host stats + the process's shared-anon RSS before/after.
def _rb_shared_rss_mib():
    t, cur = 0, False
    with open("/proc/self/smaps") as f:
        for ln in f:
            if ln[0] in "0123456789abcdef" and "-" in ln.split(" ", 1)[0]:
                cur = ln.rstrip().endswith("/dev/zero (deleted)")
            elif cur and ln.startswith("Rss:"):
                t += int(ln.split()[1])
    return t // 1024


def _rb_host_flush(why):
    s = torch.cuda.host_memory_stats()
    a, r = s.get("allocated_bytes.current", 0) >> 20, s.get("reserved_bytes.current", 0) >> 20
    before = _rb_shared_rss_mib()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()
    print(f"PLE-HOST: {why}: torch pinned allocated {a} MiB / reserved {r} MiB; "
          f"shared-anon RSS {before} -> {_rb_shared_rss_mib()} MiB after flush", flush=True)


try:
    from vllm.v1.worker import gpu_worker as _rb_gw

    for _rb_m in ("load_model", "compile_or_warm_up_model"):
        _rb_f = getattr(_rb_gw.Worker, _rb_m)
        if not getattr(_rb_f, "_rb_flush", False):
            def _rb_wrap(self, *a, _f=_rb_f, _m=_rb_m, **kw):
                out = _f(self, *a, **kw)
                _rb_host_flush(f"after {_m}")
                return out
            _rb_wrap._rb_flush = True
            setattr(_rb_gw.Worker, _rb_m, _rb_wrap)
except Exception as _rb_e:  # noqa: BLE001
    print(f"PLE-HOST: cannot hook worker ({_rb_e})", flush=True)
