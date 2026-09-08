#!/usr/bin/env python3
"""Per-step prefill metrics (build-time patch of vllm/v1/metrics/loggers.py).

vLLM credits vllm:prompt_tokens_total only when a request's prefill FINISHES, so
a 100k-token prompt shows 0 tok/s for a minute and then a spike. The engine can
attach per-iteration SchedulerIterationDetails (num_ctx_tokens = prefill tokens
scheduled this step) behind --enable-logging-iteration-details, but the stock
consumer is a one-INFO-line-per-step text logger. This patch:
  * mutes that per-step log line;
  * adds Prometheus counters fed from the same details every engine step:
      vllm:scheduled_ctx_tokens_total   prefill tokens scheduled  (rate = live pp tok/s)
      vllm:scheduled_iterations_total   engine steps observed
Inert unless --enable-logging-iteration-details is passed (ITER_DETAILS=1 in
scripts/serve-intel-ar.sh). Watch with bench/ppwatch.sh.
"""
SP = "/usr/local/lib/python3.12/dist-packages"
F = f"{SP}/vllm/v1/metrics/loggers.py"
src = open(F).read()
assert "class PrometheusStatLogger" in src and "def _log_iteration_details" in src
assert "_q38_prefill_metrics" not in src, "already patched"
src += '''

# --- qwen38-flash-dgx: per-step prefill metrics (--enable-logging-iteration-details) ---
def _q38_prefill_metrics():
    _orig_init = PrometheusStatLogger.__init__
    _orig_record = PrometheusStatLogger.record

    def __init__(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        names = ["model_name", "engine"]
        ctx = self._counter_cls(
            name="vllm:scheduled_ctx_tokens",
            documentation="Prefill (context) tokens scheduled per engine step; "
                          "rate() = live prefill tok/s. Needs --enable-logging-iteration-details.",
            labelnames=names)
        its = self._counter_cls(
            name="vllm:scheduled_iterations",
            documentation="Engine steps observed via iteration details.",
            labelnames=names)
        self._q38_ctx = {i: ctx.labels(*v) for i, v in self.per_engine_labelvalues.items()}
        self._q38_iter = {i: its.labels(*v) for i, v in self.per_engine_labelvalues.items()}

    def _record_details(self, scheduler_stats, engine_idx=0):
        d = getattr(scheduler_stats, "iteration_details", None) if scheduler_stats else None
        if d is not None and not d.is_dummy and engine_idx in self._q38_ctx:
            self._q38_ctx[engine_idx].inc(d.num_ctx_tokens)
            self._q38_iter[engine_idx].inc()

    def record(self, scheduler_stats, iteration_stats, mm_cache_stats=None, engine_idx=0):
        _orig_record(self, scheduler_stats, iteration_stats, mm_cache_stats, engine_idx)
        self._q38_record_details(scheduler_stats, engine_idx)

    PrometheusStatLogger.__init__ = __init__
    PrometheusStatLogger._q38_record_details = _record_details
    PrometheusStatLogger.record = record
    # the counters carry the data; the per-step INFO line is pure noise
    LoggingStatLogger._log_iteration_details = lambda self, scheduler_stats, engine_idx: None


_q38_prefill_metrics()
'''
open(F, "w").write(src)
import ast; ast.parse(src)
print("loggers.py: per-step prefill metrics added OK")
