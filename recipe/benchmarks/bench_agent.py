#!/usr/bin/env python3
# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""Agent-shaped decode benchmark for the Qwen3.8-Flash vLLM server on one GB10.

Sends streaming chat completions built from ``prompts.json`` (system + tools +
a list of user tasks) and records, per request: TTFT, total wall time,
prompt/completion/cached token counts, decode tok/s, and the vLLM speculative
decoding counters that moved while the request was in flight.

Standard library only. Read-only against the server: it never restarts or
reconfigures anything, it only issues chat completions and GETs /metrics.

Usage:
    python3 bench_agent.py --base http://localhost:8000 --model qwen \\
        --repeats 3 --max-tokens 1500 --effort xhigh --out V1.agent.json \\
        --concurrency-probe 4 --warmup 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

# Prometheus sample line: name{label="v",...} 1.23e+04
_SAMPLE_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)\s*$')
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

SPEC_DRAFTS = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_ACCEPTED_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"


# --------------------------------------------------------------------------
# /metrics
# --------------------------------------------------------------------------
def parse_metrics(text: str) -> dict:
    """Parse Prometheus text exposition into {name: [(labels, float), ...]}."""
    out: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _SAMPLE_RE.match(line)
        if not m:
            continue
        try:
            value = float(m.group("value"))
        except ValueError:
            continue  # NaN / +Inf / malformed -- not useful as a counter
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        out.setdefault(m.group("name"), []).append((labels, value))
    return out


def fetch_metrics(base: str, timeout: float = 10.0) -> dict:
    """GET /metrics and parse it; returns {} on any failure."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=timeout) as resp:
            return parse_metrics(resp.read().decode("utf-8", "replace"))
    except Exception:
        return {}


def metric_value(metrics: dict, name: str, **want) -> float:
    """Sum the samples of ``name`` whose labels match every kwarg in ``want``."""
    total = 0.0
    for labels, value in metrics.get(name, []):
        if all(labels.get(k) == v for k, v in want.items()):
            total += value
    return total


def spec_delta(before: dict, after: dict) -> dict:
    """Speculative-decoding counters that moved between two /metrics scrapes."""
    if not before or not after:
        return {}
    drafts = metric_value(after, SPEC_DRAFTS) - metric_value(before, SPEC_DRAFTS)
    draft_tokens = metric_value(after, SPEC_DRAFT_TOKENS) - metric_value(before, SPEC_DRAFT_TOKENS)
    accepted = metric_value(after, SPEC_ACCEPTED) - metric_value(before, SPEC_ACCEPTED)
    per_pos = []
    for pos in ("0", "1", "2"):
        per_pos.append(
            metric_value(after, SPEC_ACCEPTED_POS, position=pos)
            - metric_value(before, SPEC_ACCEPTED_POS, position=pos)
        )
    out = {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "accepted_per_pos": per_pos,
        "prefix_hits": metric_value(after, PREFIX_HITS) - metric_value(before, PREFIX_HITS),
        "prefix_queries": metric_value(after, PREFIX_QUERIES) - metric_value(before, PREFIX_QUERIES),
    }
    if drafts > 0:
        out["tokens_per_step"] = 1.0 + accepted / drafts
        out["accept_pos_rate"] = [p / drafts for p in per_pos]
    else:
        out["tokens_per_step"] = None
        out["accept_pos_rate"] = [None, None, None]
    return out


# --------------------------------------------------------------------------
# one streaming request
# --------------------------------------------------------------------------
def build_body(cfg, task_user: str) -> dict:
    """Assemble the chat-completions request body for one task."""
    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": cfg.system},
            {"role": "user", "content": task_user},
        ],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": cfg.max_tokens,
        "temperature": 1.0,
        "top_p": 0.95,
        "top_k": 20,  # vLLM accepts top_k at the top level of the body
        "chat_template_kwargs": {"reasoning_effort": cfg.effort},
    }
    if cfg.tools:
        body["tools"] = cfg.tools
        body["tool_choice"] = "auto"
    return body


def stream_chat(cfg, task_id: str, task_user: str, collect_metrics: bool = True) -> dict:
    """Run one streaming chat completion and return a sample dict.

    Never raises: transport and server errors are recorded in the sample with
    ``ok=False`` so the caller can keep going.
    """
    sample = {"task": task_id, "ok": False}
    before = fetch_metrics(cfg.base) if collect_metrics else {}

    req = urllib.request.Request(
        cfg.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(build_body(cfg, task_user)).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )

    ttft = None
    usage = None
    reasoning_chars = 0
    content_chars = 0
    chunks = 0
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=cfg.request_timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except ValueError:
                    continue
                chunks += 1
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices") or []:
                    delta = choice.get("delta") or {}
                    r = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    c = delta.get("content") or ""
                    if (r or c) and ttft is None:
                        ttft = time.perf_counter() - t0
                    reasoning_chars += len(r)
                    content_chars += len(c)
        sample["ok"] = True
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", "replace")[:500]
        except Exception:
            pass
        sample["error"] = "HTTP %s: %s" % (exc.code, body)
    except Exception as exc:  # URLError, socket timeout, incomplete read...
        sample["error"] = "%s: %s" % (type(exc).__name__, exc)

    total = time.perf_counter() - t0
    after = fetch_metrics(cfg.base) if collect_metrics else {}

    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    completion_tokens = usage.get("completion_tokens")
    sample.update(
        {
            "ttft_s": ttft,
            "total_s": total,
            "chunks": chunks,
            "reasoning_chars": reasoning_chars,
            "content_chars": content_chars,
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": completion_tokens,
            "cached_tokens": details.get("cached_tokens"),
            "decode_tok_s": None,
        }
    )
    if completion_tokens and ttft is not None and total > ttft:
        sample["decode_tok_s"] = completion_tokens / (total - ttft)
    if collect_metrics:
        sample["spec"] = spec_delta(before, after)
    return sample


# --------------------------------------------------------------------------
# stats
# --------------------------------------------------------------------------
def summarize(values) -> dict:
    """Median / IQR summary of a list of numbers, ignoring None."""
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return {"n": 0, "median": None, "iqr": None, "q1": None, "q3": None,
                "min": None, "max": None}
    out = {"n": len(vals), "median": statistics.median(vals),
           "min": vals[0], "max": vals[-1]}
    if len(vals) >= 2:
        q1, _, q3 = statistics.quantiles(vals, n=4, method="inclusive")
        out.update({"q1": q1, "q3": q3, "iqr": q3 - q1})
    else:
        out.update({"q1": vals[0], "q3": vals[0], "iqr": 0.0})
    return out


def concurrency_probe(cfg, tasks, n: int) -> dict:
    """One round of ``n`` parallel requests over the first ``n`` tasks."""
    picked = tasks[:n]
    if not picked:
        return {"n": 0}
    results: list = [None] * len(picked)
    before = fetch_metrics(cfg.base)

    def worker(idx, task):
        results[idx] = stream_chat(cfg, task["id"], task["user"], collect_metrics=False)

    threads = [threading.Thread(target=worker, args=(i, t), daemon=True)
               for i, t in enumerate(picked)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=cfg.request_timeout + 60)
    wall = time.perf_counter() - t0
    after = fetch_metrics(cfg.base)

    samples = [r for r in results if r]
    tokens = sum(s.get("completion_tokens") or 0 for s in samples)
    per_stream = [s.get("decode_tok_s") for s in samples]
    return {
        "n": len(picked),
        "wall_s": wall,
        "total_completion_tokens": tokens,
        "aggregate_tok_s": (tokens / wall) if wall > 0 else None,
        "per_stream_tok_s": per_stream,
        "per_stream_tok_s_summary": summarize(per_stream),
        "spec": spec_delta(before, after),
        "samples": samples,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      "prompts.json"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--out", default="bench_agent.json")
    ap.add_argument("--concurrency-probe", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--label", default="", help="free-form tag copied into the output")
    cfg = ap.parse_args(argv)

    try:
        with open(cfg.prompts, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception as exc:
        print("bench_agent: cannot read %s: %s" % (cfg.prompts, exc), file=sys.stderr)
        return 2

    cfg.system = spec.get("system") or "You are a helpful assistant."
    cfg.tools = spec.get("tools") or []
    tasks = spec.get("tasks") or []
    if not tasks:
        print("bench_agent: prompts.json has no tasks", file=sys.stderr)
        return 2

    started = time.time()
    print("bench_agent: %d tasks x %d repeats, effort=%s" % (len(tasks), cfg.repeats, cfg.effort),
          file=sys.stderr)

    # Warm-up: not recorded. Gets the prefix cache and any lazy kernels hot so
    # the first recorded sample is not an outlier.
    for i in range(max(0, cfg.warmup)):
        t = tasks[i % len(tasks)]
        s = stream_chat(cfg, "warmup-%d" % i, t["user"], collect_metrics=False)
        print("  warmup %d: ok=%s %.1fs" % (i, s["ok"], s["total_s"]), file=sys.stderr)

    samples = []
    for rep in range(max(1, cfg.repeats)):
        for t in tasks:
            s = stream_chat(cfg, t["id"], t["user"])
            s["repeat"] = rep
            samples.append(s)
            print("  rep%d %-18s ok=%s ttft=%s tok/s=%s ct=%s" % (
                rep, t["id"], s["ok"],
                ("%.2f" % s["ttft_s"]) if s["ttft_s"] else "-",
                ("%.1f" % s["decode_tok_s"]) if s["decode_tok_s"] else "-",
                s["completion_tokens"]), file=sys.stderr)

    good = [s for s in samples if s["ok"]]
    probe = concurrency_probe(cfg, tasks, cfg.concurrency_probe) if cfg.concurrency_probe > 0 else {}

    tps = [s["spec"].get("tokens_per_step") for s in good if s.get("spec")]
    pos_rates = [[], [], []]
    for s in good:
        rates = (s.get("spec") or {}).get("accept_pos_rate") or []
        for i, r in enumerate(rates[:3]):
            if isinstance(r, (int, float)):
                pos_rates[i].append(r)

    result = {
        "label": cfg.label,
        "base": cfg.base,
        "model": cfg.model,
        "effort": cfg.effort,
        "max_tokens": cfg.max_tokens,
        "repeats": cfg.repeats,
        "warmup": cfg.warmup,
        "started_epoch": started,
        "elapsed_s": time.time() - started,
        "n_samples": len(samples),
        "n_ok": len(good),
        "n_failed": len(samples) - len(good),
        "errors": [s.get("error") for s in samples if not s["ok"]][:20],
        "samples": samples,
        "summary": {
            "decode_tok_s": summarize([s["decode_tok_s"] for s in good]),
            "ttft_s": summarize([s["ttft_s"] for s in good]),
            "total_s": summarize([s["total_s"] for s in good]),
            "completion_tokens": summarize([s["completion_tokens"] for s in good]),
            "prompt_tokens": summarize([s["prompt_tokens"] for s in good]),
            "cached_tokens": summarize([s["cached_tokens"] for s in good]),
            "tokens_per_step": summarize(tps),
            "accept_pos_rate": [summarize(p) for p in pos_rates],
        },
        "concurrency_probe": probe,
    }

    try:
        with open(cfg.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(result, fh, indent=2, sort_keys=False)
            fh.write("\n")
    except Exception as exc:
        print("bench_agent: cannot write %s: %s" % (cfg.out, exc), file=sys.stderr)
        return 2

    med = result["summary"]["decode_tok_s"]["median"]
    print("bench_agent: median decode %s tok/s over %d ok samples -> %s" % (
        ("%.2f" % med) if med else "n/a", len(good), cfg.out), file=sys.stderr)
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
