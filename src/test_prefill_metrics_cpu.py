#!/usr/bin/env python3
"""CPU check for patch_prefill_metrics.py (no GPU): PrometheusStatLogger._q38_record_details()
(called by the wrapped record()) feeds the per-step counters from iteration_details, and the per-step text log is muted.
    docker run --rm -v "$PWD/src:/t" -w /t --entrypoint python3 qwen38-flash-dgx test_prefill_metrics_cpu.py
"""
from vllm.v1.metrics import loggers as L
from vllm.v1.metrics.stats import SchedulerStats, SchedulerIterationDetails

names = ["model_name", "engine"]
ctx = L.Counter("vllm:scheduled_ctx_tokens_test", "t", names)
its = L.Counter("vllm:scheduled_iterations_test", "t", names)
class Stub: pass
me = Stub()  # only what _q38_record_details() touches
me._q38_ctx = {0: ctx.labels("qwen", "0")}
me._q38_iter = {0: its.labels("qwen", "0")}

def step(n_ctx, dummy=False):
    st = SchedulerStats()
    st.iteration_details = SchedulerIterationDetails(
        iteration_index=1, num_ctx_requests=1, num_ctx_tokens=n_ctx,
        num_generation_requests=2, num_generation_tokens=8, elapsed_ms=3.0, is_dummy=dummy)
    L.PrometheusStatLogger._q38_record_details(me, st, 0)

step(8000); step(1600); step(5000, dummy=True)          # dummy steps must not count
L.PrometheusStatLogger._q38_record_details(me, SchedulerStats(), 0)   # no details -> no-op
L.PrometheusStatLogger._q38_record_details(me, None, 0)               # None stats -> no-op
got = ctx.labels("qwen", "0")._value.get()
assert got == 9600, got
assert its.labels("qwen", "0")._value.get() == 2
assert L.LoggingStatLogger._log_iteration_details(None, None, 0) is None
print("prefill metrics: OK (9600 ctx tokens over 2 real steps; dummy/None ignored; per-step log muted)")
