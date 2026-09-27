#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Quick check that the server is up, coherent, and measure prefill + decode.
#   bash recipe/scripts/smoke-test.sh [host:port]
set -euo pipefail
EP="${1:-localhost:8000}"
BASE="http://$EP"

echo ">> health"
if curl -sf -m 5 "$BASE/health" >/dev/null; then
  echo "   OK"
else
  echo "   not ready"
  exit 1
fi

echo ">> coherence"
curl -s -m 120 "$BASE/v1/completions" -H 'Content-Type: application/json' -d \
  '{"model":"qwen","prompt":"The capital of France is","max_tokens":12,"temperature":0}' \
  | python3 -c 'import json,sys;print("  ",repr(json.load(sys.stdin)["choices"][0]["text"]))'

echo ">> prefill (TTFT on a ~8k-token prompt)"
python3 - "$BASE" <<'PY'
import json,sys,time,urllib.request
base=sys.argv[1]; prompt="word "*8000
t=time.time()
req=urllib.request.Request(base+"/v1/completions",
    data=json.dumps({"model":"qwen","prompt":prompt,"max_tokens":1,"temperature":0}).encode(),
    headers={"Content-Type":"application/json"})
u=json.load(urllib.request.urlopen(req,timeout=300))["usage"]; dt=time.time()-t
print(f"   {u['prompt_tokens']} tok in {dt:.2f}s  =>  {u['prompt_tokens']/dt:.0f} tok/s prefill")
PY

echo ">> decode (256 tokens, short prompt)"
python3 - "$BASE" <<'PY'
import json,sys,time,urllib.request
base=sys.argv[1]
t=time.time()
req=urllib.request.Request(base+"/v1/completions",
    data=json.dumps({"model":"qwen","prompt":"Hello","max_tokens":256,"temperature":0,"ignore_eos":True}).encode(),
    headers={"Content-Type":"application/json"})
n=json.load(urllib.request.urlopen(req,timeout=300))["usage"]["completion_tokens"]; dt=time.time()-t
print(f"   {n} tok in {dt:.2f}s  =>  {n/dt:.1f} tok/s decode")
PY
