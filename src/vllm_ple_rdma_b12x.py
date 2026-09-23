"""PLE table over RDMA for the b12x vLLM image (VLLM_PLE_RDMA=<host:port>, VLLM_PLE_TABLE_MEMORY=disk).

b12x's disk mode hashes the n-gram ids on the GPU, fills a CUDA-mapped host cache with those
rows (native io_uring reader) and decodes them with its fp8 lookup kernel, all synchronously in
model_state.prepare_inputs -> prepare_disk. Here the rows come from RdmaPleTable instead
(vllm_ple_rdma.py: same retry-forever / no-fallback client as the magi image), and the
transaction is split so the READ overlaps compute the way the magi image's prefetch does:

  issue    (prepare_disk, batch assembly): hash + ids D2H on the stream, READ on a worker thread
  complete (wait for the rows, launch b12x's lookup kernel):
             PIECEWISE / eager steps -> at the PLE layer, through b12x's eager split op
                                        (vllm::qwen3_8_flash_next_ple_embedding): the READ
                                        overlaps layer 0, like production
             FULL cudagraph steps    -> right before the replay (no host hook inside a full
                                        graph); these are decode batches, <=1024 rows ~0.25 ms

Hashing, lookup and dequant stay b12x's kernels. The table is not in the checkpoint index:
the fp8 weight_scale comes from the server hello (its parameter becomes a buffer the strict
loader does not expect) and the shard coverage checks are satisfied up front. b12x's io_uring
reader is never created. Inert unless VLLM_PLE_RDMA is set.
"""
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

logger = logging.getLogger("vllm.ple_rdma_b12x")

_LIVE: list = []  # RDMA-backed embeddings (one per PLE layer)
_STATS = {"steps": 0, "read_ms": 0.0, "wait_ms": 0.0, "last": 0.0, "stale": 0}


def apply(emb_cls: type) -> None:
    endpoint = os.environ.get("VLLM_PLE_RDMA")
    if not endpoint or getattr(emb_cls, "_ple_rdma_patched", False):
        return
    from b12x.sequence._shared.disk_table import DiskRowCache
    from b12x.sequence.ple_embedding._kernels import _launch_fp8_lookup, _launch_hash

    # ---- DiskRowCache: no io_uring reader, no file shards --------------------------------
    orig_cache_init = DiskRowCache.__init__
    orig_require = DiskRowCache.require_complete

    class _NoUring:
        """b12x's native loader minus the io_uring reader, which RDMA mode never uses (and
        Docker's default seccomp profile refuses: 'io_uring initialization failed')."""

        def __init__(self, native):
            self._native = native

        def ple_reader(self, *args):
            return None

        def __getattr__(self, name):
            return getattr(self._native, name)

    def cache_init(self, *args, **kwargs):
        import b12x.loader._native as native

        real_load = native.load
        native.load = lambda: _NoUring(real_load())
        try:
            orig_cache_init(self, *args, **kwargs)
        finally:
            native.load = real_load

    def require_complete(self):
        if getattr(self, "_rdma", None) is None:
            orig_require(self)

    DiskRowCache.__init__ = cache_init
    DiskRowCache.require_complete = require_complete

    # ---- the embedding: issue / complete ---------------------------------------------------
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ple-rdma")

    def fetch(cache, ready: torch.cuda.Event, n: int) -> float:
        ready.synchronize()  # hash + ids D2H done, and every earlier lookup of this row cache
        t0 = time.perf_counter()
        if not n:
            return 0.0
        ids = cache.ids_host[:n].numpy()
        rows = cache.weight_host[:n].numpy()
        # The lookup kernel zeroes any id outside [shard_start, shard_end) (the hash emits -1
        # for positions without an n-gram) and never reads its row: fetch only in-range ids.
        live = (ids >= cache.shard_start) & (ids < cache.shard_end)
        if live.all():
            rows[:] = cache._rdma._gather(ids)
        elif live.any():
            where = np.flatnonzero(live)
            rows[where] = cache._rdma._gather(ids[where])
        return (time.perf_counter() - t0) * 1e3

    def prepare_disk(self, input_ids, query_start_loc, ngram_context):
        if torch.compiler.is_compiling() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Disk PLE preparation must run outside CUDA graphs")
        if self._rdma_job is not None:
            # The previous step's forward never reached a completion point (PLE split op or
            # run_fullgraph), so it ran on stale embeddings: a bug, never expected.
            _STATS["stale"] += 1
            logger.error("PLE rdma: step ran without completing its READ (stale embeddings), %d so far",
                         _STATS["stale"])
            self._rdma_complete()
        self._disk_prepared_tokens = -1
        self._disk_prepared = False
        self._validate_embedding_loaded()
        token_count = self._prepare_inputs(input_ids, query_start_loc, ngram_context)
        b = self._bind_embedding()
        cache, plan = b.disk_table._cache, b.plan
        caps = plan.caps
        stream = torch.cuda.current_stream()
        if cache._cache_used:
            stream.wait_event(cache._cache_done)
        if cache._rdma_scale is not None:
            target, value = cache._rdma_scale
            target.fill_(value)
            cache._rdma_scale = None
        _launch_hash(
            b.token_ids, b.query_start_loc, b.committed_history, b.num_seqs, b.num_tokens,
            plan.multipliers, plan.prime_sizes, plan.table_offsets, b._ids,
            b._hash_binding.request_ids, b.error_code, caps.eos_token_id, caps.vocab_size,
            caps.max_order, caps.heads_per_order, caps.max_seqs, caps.max_tokens, token_count,
        )
        n = token_count * plan.head_count
        cache.ids_host[:n].copy_(b._ids.view(-1)[:n], non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(stream)
        self._rdma_job = (pool.submit(fetch, cache, ready, n), b, token_count)
        self._disk_prepared_tokens = input_ids.numel()
        self._disk_prepared = True

    def _rdma_complete(self):
        job, self._rdma_job = self._rdma_job, None
        if job is None:
            return
        future, b, token_count = job
        t0 = time.perf_counter()
        while True:
            try:
                read_ms = future.result(timeout=5.0)  # a worker error re-raises here: loud, no fallback
                break
            except TimeoutError:
                logger.error("PLE rdma: prefetched READ still pending after 5s -- stalling "
                             "the step until it lands (no fallback)")
        _stats(read_ms, (time.perf_counter() - t0) * 1e3)
        cache, plan = b.disk_table._cache, b.plan
        caps = plan.caps
        if token_count:
            _launch_fp8_lookup(
                b.disk_table.weight, b.weight_scale, b._ids, b.num_tokens, b.out[:token_count],
                caps.max_tokens, plan.head_count, plan.head_dim, caps.embedding_dim,
                plan.table_vocab_size, plan.shard_start, plan.shard_end, compact_rows=True,
            )
        cache._cache_done.record(torch.cuda.current_stream())
        cache._cache_used = True

    orig_run_embedding = emb_cls._run_embedding

    def _run_embedding(self, *args):
        # Reached through b12x's eager split op (see forward): PIECEWISE/eager completion.
        if self._rdma:
            self._rdma_complete()
            return
        return orig_run_embedding(self, *args)

    orig_forward = emb_cls.forward

    def forward(self, input_ids, query_start_loc, ngram_context, *, wait_for=None):
        if self._rdma:
            # A split point at the PLE layer, as the magi image has: in PIECEWISE/eager runs
            # this waits for the READ issued at batch assembly, after layer 0 was launched.
            # Captured into a FULL graph it is a no-op (dummy runs issue nothing); replays
            # complete in run_fullgraph instead.
            torch.ops.vllm.qwen3_8_flash_next_ple_embedding(
                input_ids.reshape(-1), query_start_loc, ngram_context,
                self._embedding_out, self.owner_prefix,
            )
        return orig_forward(self, input_ids, query_start_loc, ngram_context, wait_for=wait_for)

    orig_init = emb_cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if not self.requires_disk_preparation:
            raise RuntimeError("VLLM_PLE_RDMA needs VLLM_PLE_TABLE_MEMORY=disk (b12x io_uring table mode)")
        if self._quant_mode != "fp8_e4m3_per_tensor":
            raise RuntimeError(f"VLLM_PLE_RDMA serves an fp8 table, config says {self._quant_mode}")
        storage = self.ngram_embedding
        cache = storage.disk_table._cache
        # vllm_ple_rdma's own prefetch hook targets the magi image's runner; this module
        # issues/completes the READs itself.
        os.environ["VLLM_PLE_RDMA_PREFETCH"] = "0"
        from vllm_ple_rdma import RdmaPleTable

        with torch.device("cpu"):  # model init runs under a CUDA default device; the READ buffers are pinned host
            table = RdmaPleTable(endpoint, cache.shard_rows, cache.weight_row_bytes,
                                 torch.float8_e4m3fn, cache.table_rows)
        remote = table.client.remote
        if remote["rows"] != self.split_ngram_parts * cache.shard_rows:
            raise RuntimeError(f"PLE rdma: server holds {remote['rows']} rows, b12x plans "
                               f"{self.split_ngram_parts} x {cache.shard_rows} -- row ids would not line up")
        if remote.get("weight_scale") is None:
            raise RuntimeError("PLE rdma: server sent no weight_scale")
        scale = storage.weight_scale
        del storage._parameters["weight_scale"]
        storage.register_buffer("weight_scale", scale.data, persistent=False)
        cache._rdma = table
        cache._rdma_scale = (storage.weight_scale, float(remote["weight_scale"]))
        self._embedding_load_ranges.add((self._plan.shard_start, self._plan.shard_end))
        self._weight_scale_loaded = True
        self._rdma = True
        self._rdma_job = None
        _LIVE.append(self)
        _patch_fullgraph()
        logger.info("PLE rdma (b12x): %s rows [%d, %d) x %d B, shard_rows %d, scale %g",
                    endpoint, self._plan.shard_start, self._plan.shard_end,
                    cache.weight_row_bytes, cache.shard_rows, remote["weight_scale"])

    emb_cls._rdma = False
    emb_cls._rdma_job = None
    emb_cls.__init__ = __init__
    emb_cls.prepare_disk = prepare_disk
    emb_cls._rdma_complete = _rdma_complete
    emb_cls._run_embedding = _run_embedding
    emb_cls.forward = forward
    emb_cls._ple_rdma_patched = True
    logger.info("PLE rdma patch applied to b12x DiskRowCache + %s", emb_cls.__name__)


def _patch_fullgraph() -> None:
    """FULL cudagraph replays run no Python inside the model: complete the READ first."""
    from vllm.v1.worker.gpu.cudagraph_utils import ModelCudaGraphManager

    if getattr(ModelCudaGraphManager, "_ple_rdma_patched", False):
        return
    orig = ModelCudaGraphManager.run_fullgraph

    def run_fullgraph(self, *args, **kwargs):
        for emb in _LIVE:
            emb._rdma_complete()
        return orig(self, *args, **kwargs)

    ModelCudaGraphManager.run_fullgraph = run_fullgraph
    ModelCudaGraphManager._ple_rdma_patched = True


def _stats(read_ms: float, wait_ms: float) -> None:
    """read = RDMA READ time; wait = host time blocked at completion (includes waiting for the
    previous step's GPU work, which the hash is queued behind -- not RDMA latency)."""
    s = _STATS
    s["steps"] += 1
    s["read_ms"] += read_ms
    s["wait_ms"] += wait_ms
    now = time.monotonic()
    if now - s["last"] >= 30.0:
        if s["last"]:
            n = max(1, s["steps"])
            logger.info("PLE rdma stats (last %.0fs): %d steps, READ %.3f ms/step, host wait at "
                        "completion %.2f ms/step, stale steps %d", now - s["last"], s["steps"],
                        s["read_ms"] / n, s["wait_ms"] / n, s["stale"])
        s.update(steps=0, read_ms=0.0, wait_ms=0.0, last=now)
