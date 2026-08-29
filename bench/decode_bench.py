#!/usr/bin/env python3
"""Batch-1 decode benchmark for the qwen38-flash stack.

W1: unique short prompt, 1000 forced greedy tokens -> decode tok/s.
W2: PIN-prefixed ~8k prompt (prefix hit) + short question, 256 tokens
    -> TTFT + tok/s.
Prints spec-decode acceptance (accepted tokens/step, per-position rate)
derived from /metrics deltas around each workload.

Usage: decode_bench.py <tag> [runs=3]
"""
import json, re, subprocess, sys, time

BASE = "http://localhost:8000"
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
RUNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PIN = 'You are "Magi AI", a smart home AI (via Home Assistant) and general knowledge assistant.'


def metrics():
    out = subprocess.run(["curl", "-s", f"{BASE}/metrics"], capture_output=True).stdout.decode()
    m = {}
    for k in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        r = re.search(rf"vllm:spec_decode_{k}_total{{[^}}]*}} ([0-9.e+]+)", out)
        m[k] = float(r.group(1)) if r else 0.0
    m["pos"] = {int(p): float(v) for p, v in re.findall(
        r'vllm:spec_decode_num_accepted_tokens_per_pos_total{[^}]*position="(\d+)"[^}]*} ([0-9.e+]+)', out)}
    return m


def complete(prompt, max_tokens, stream=False):
    payload = json.dumps({"model": "qwen", "prompt": prompt, "max_tokens": max_tokens,
                          "temperature": 0, "ignore_eos": True, "stream": stream})
    t0 = time.perf_counter()
    if not stream:
        subprocess.run(["curl", "-s", f"{BASE}/v1/completions", "-H",
                        "Content-Type: application/json", "-d", "@-"],
                       input=payload.encode(), capture_output=True)
        return time.perf_counter() - t0, None
    ttft = None
    p = subprocess.Popen(["curl", "-sN", f"{BASE}/v1/completions", "-H",
                          "Content-Type: application/json", "-d", "@-"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    p.stdin.write(payload.encode()); p.stdin.close()
    for line in p.stdout:
        if line.startswith(b"data: ") and b"[DONE]" not in line and ttft is None:
            ttft = time.perf_counter() - t0
    return time.perf_counter() - t0, ttft


def spec_report(a, b):
    drafts = b["num_drafts"] - a["num_drafts"]
    drafted = b["num_draft_tokens"] - a["num_draft_tokens"]
    accepted = b["num_accepted_tokens"] - a["num_accepted_tokens"]
    if drafts <= 0:
        return "no spec data"
    per_pos = {p: (b["pos"].get(p, 0) - a["pos"].get(p, 0)) / drafts
               for p in sorted(b["pos"])}
    pp = " ".join(f"p{p}:{v:.2f}" for p, v in per_pos.items())
    return (f"steps {drafts:.0f}, tok/step {1 + accepted / drafts:.2f} "
            f"(accept {accepted / drafted * 100 if drafted else 0:.0f}%), pos-rate {pp}")


print(f"=== {TAG} ===", flush=True)
w1 = []
for i in range(RUNS):
    m0 = metrics()
    prompt = f"[{TAG}-w1-{i}-{time.time():.0f}] Write a very long detailed essay about the history of computing. "
    total, ttft = complete(prompt, 1000, stream=True)
    tg = 999 / (total - ttft)
    w1.append(tg)
    print(f"W1 run{i}: ttft {ttft:.2f}s total {total:.2f}s tg {tg:.1f} tok/s | {spec_report(m0, metrics())}", flush=True)
print(f"W1 median: {sorted(w1)[len(w1) // 2]:.1f} tok/s", flush=True)

pad = " The home has many rooms and devices." * 220  # ~8k tokens with PIN
w2p = PIN + pad + "\nUser: what can you do?\nAssistant:"
complete(w2p, 8)  # ensure cached
w2 = []
for i in range(RUNS):
    m0 = metrics()
    total, ttft = complete(w2p, 256, stream=True)
    tg = 255 / (total - ttft)
    w2.append((ttft, tg))
    print(f"W2 run{i}: ttft {ttft:.2f}s tg {tg:.1f} tok/s | {spec_report(m0, metrics())}", flush=True)
w2.sort(key=lambda x: x[1])
print(f"W2 median: ttft {w2[len(w2) // 2][0]:.2f}s tg {w2[len(w2) // 2][1]:.1f} tok/s", flush=True)
