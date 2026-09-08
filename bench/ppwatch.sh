#!/usr/bin/env bash
# Live prefill throughput from vllm:scheduled_ctx_tokens_total (needs the image's
# prefill-metrics patch and the server started with ITER_DETAILS=1).
#   bench/ppwatch.sh [http://localhost:8000] [interval_s]
BASE="${1:-http://localhost:8000}"; DT="${2:-1}"
get() { curl -s "$BASE/metrics" | awk '/^vllm:scheduled_ctx_tokens_total/{c=$NF} /^vllm:scheduled_iterations_total/{i=$NF} END{print c+0, i+0}'; }
read -r c0 i0 < <(get); t0=$(date +%s.%N)
while sleep "$DT"; do
  read -r c1 i1 < <(get); t1=$(date +%s.%N)
  awk -v c0="$c0" -v c1="$c1" -v i0="$i0" -v i1="$i1" -v t0="$t0" -v t1="$t1" \
    'BEGIN{dt=t1-t0; printf "%s  prefill %6.0f tok/s  (%4.1f steps/s, %5.0f tok/step)\n", strftime("%H:%M:%S"), (c1-c0)/dt, (i1-i0)/dt, (i1>i0)?(c1-c0)/(i1-i0):0}'
  c0=$c1; i0=$i1; t0=$t1
done
