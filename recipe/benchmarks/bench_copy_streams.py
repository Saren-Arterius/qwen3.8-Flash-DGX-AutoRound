#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Measure copy-heavy decode throughput at one through eight concurrent streams.

Three rounds run at each stream count, after one warmup. A generated public
copy workload is bundled below; --prompts accepts a JSON file with system,
tools and copy tasks. The metric is completion tokens within the all-decoding
window (latest first token to earliest finish), divided by that window.
The runner also records per-stream rates, time to first token, acceptance
deltas, scheduler running/waiting peaks and ordinary total-tokens-per-wall-time
for each round. It checks host available memory before each stream count.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

_SAMPLE_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)\s*$')
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

SPEC_DRAFTS = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_ACCEPTED_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
GAUGE_RUNNING = "vllm:num_requests_running"
GAUGE_WAITING = "vllm:num_requests_waiting"
REQ_SUCCESS = "vllm:request_success_total"
SPEC_POSITIONS = ("0", "1", "2", "3", "4")

def generic_copy_spec() -> dict:
    """Build a public copy workload with a long shared prefix and three tasks."""
    guidance = (
        "Preserve source code exactly when asked to copy it. Keep indentation, "
        "identifiers, literals, comments and line order unchanged. Do not add "
        "explanations, markdown fences or tool calls."
    )
    system = "You are a coding assistant.\n" + "\n".join(
        f"Reference instruction {index:03d}: {guidance}"
        for index in range(1, 116)
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_reference",
            "description": "Look up a public programming-language reference entry.",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string"}},
                "required": ["topic"],
            },
        },
    }]
    tasks = []
    for variant in range(1, 4):
        code = "\n".join(
            f"def transform_{variant}_{line:03d}(value): return (value + {line}) * {variant}"
            for line in range(1, 91)
        )
        tasks.append({
            "id": f"copy{variant}",
            "category": "copy",
            "user": "Copy the following Python module exactly as plain text:\n" + code,
        })
    return {"system": system, "tools": tools, "tasks": tasks}


# --------------------------------------------------------------------------
# /metrics
# --------------------------------------------------------------------------
def parse_metrics(text: str) -> dict:
    out = {}
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
            continue
        labels = dict(_LABEL_RE.findall(m.group("labels") or ""))
        out.setdefault(m.group("name"), []).append((labels, value))
    return out


def fetch_metrics(base: str, timeout: float = 10.0) -> dict:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/metrics", timeout=timeout) as resp:
            return parse_metrics(resp.read().decode("utf-8", "replace"))
    except Exception:
        return {}


def metric_value(metrics: dict, name: str, **want) -> float:
    total = 0.0
    for labels, value in metrics.get(name, []):
        if all(labels.get(k) == v for k, v in want.items()):
            total += value
    return total


def metric_present(metrics: dict, name: str, **want) -> bool:
    for labels, _value in metrics.get(name, []):
        if all(labels.get(k) == v for k, v in want.items()):
            return True
    return False


def spec_delta(before: dict, after: dict) -> dict:
    if not before or not after:
        return {}
    drafts = metric_value(after, SPEC_DRAFTS) - metric_value(before, SPEC_DRAFTS)
    accepted = metric_value(after, SPEC_ACCEPTED) - metric_value(before, SPEC_ACCEPTED)
    per_pos = []
    for pos in SPEC_POSITIONS:
        if not metric_present(after, SPEC_ACCEPTED_POS, position=pos):
            per_pos.append(None)
            continue
        per_pos.append(
            metric_value(after, SPEC_ACCEPTED_POS, position=pos)
            - metric_value(before, SPEC_ACCEPTED_POS, position=pos)
        )
    out = {
        "drafts": drafts,
        "draft_tokens": metric_value(after, SPEC_DRAFT_TOKENS) - metric_value(before, SPEC_DRAFT_TOKENS),
        "accepted_tokens": accepted,
        "accepted_per_pos": per_pos,
        "prefix_hits": metric_value(after, PREFIX_HITS) - metric_value(before, PREFIX_HITS),
        "prefix_queries": metric_value(after, PREFIX_QUERIES) - metric_value(before, PREFIX_QUERIES),
        "request_success_delta": metric_value(after, REQ_SUCCESS) - metric_value(before, REQ_SUCCESS),
    }
    out["tokens_per_step"] = (1.0 + accepted / drafts) if drafts > 0 else None
    return out


def gauges(metrics: dict) -> dict:
    return {"running": metric_value(metrics, GAUGE_RUNNING),
            "waiting": metric_value(metrics, GAUGE_WAITING),
            "request_success_total": metric_value(metrics, REQ_SUCCESS)}


def monitor_gauges(base: str, stop: threading.Event, samples: list) -> None:
    while not stop.is_set():
        reading = gauges(fetch_metrics(base))
        samples.append((time.time(), reading["running"], reading["waiting"]))
        stop.wait(1.0)


def gauge_peaks(samples: list, start: float, end: float) -> dict:
    readings = [(running, waiting) for stamp, running, waiting in samples
                if start <= stamp <= end]
    return {
        "running_peak": max((item[0] for item in readings), default=None),
        "waiting_peak": max((item[1] for item in readings), default=None),
        "samples": len(readings),
    }


def mem_available_gib() -> float:
    with open("/proc/meminfo", encoding="utf-8") as file:
        for line in file:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / (1024 * 1024)
    raise RuntimeError("MemAvailable missing from /proc/meminfo")


# --------------------------------------------------------------------------
# one streaming request
# --------------------------------------------------------------------------
def build_body(cfg, task_user: str) -> dict:
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
        "top_k": 20,
        "chat_template_kwargs": {"reasoning_effort": cfg.effort},
    }
    if cfg.tools:
        body["tools"] = cfg.tools
        body["tool_choice"] = "auto"
    return body


def stream_chat(cfg, task_id: str, task_user: str, t_launch: float, barrier: threading.Barrier) -> dict:
    sample = {"task": task_id, "ok": False}
    barrier.wait()                      # all N released at the same instant
    t0 = time.perf_counter()
    sample["launch_offset_ms"] = (t0 - t_launch) * 1000.0

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
    timeline = []          # (time since t0, cumulative generated chars)
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
                    if r or c:
                        timeline.append((time.perf_counter() - t0, reasoning_chars + content_chars))
        sample["ok"] = True
    except urllib.error.HTTPError as exc:
        sample["error"] = "HTTP %s" % exc.code
    except Exception as exc:
        sample["error"] = type(exc).__name__

    t_finish = time.perf_counter() - t0
    usage = usage or {}
    details = usage.get("prompt_tokens_details") or {}
    sample.update({
        "ttft_s": ttft,
        "finish_rel_s": t_finish,          # relative to this stream's t0
        "chunks": chunks,
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": details.get("cached_tokens"),
        "decode_tok_s": None,
        "timeline": timeline,
    })
    ct = usage.get("completion_tokens")
    if ct and ttft is not None and t_finish > ttft:
        sample["decode_tok_s"] = ct / (t_finish - ttft)
    return sample


# --------------------------------------------------------------------------
# window arithmetic
# --------------------------------------------------------------------------
def chars_at(timeline, t: float) -> float:
    """Cumulative chars at time t (linear between chunk points)."""
    prev_t, prev_c = 0.0, 0.0
    for tt, cc in timeline:
        if tt >= t:
            if tt > prev_t:
                frac = (t - prev_t) / (tt - prev_t)
                return prev_c + (cc - prev_c) * frac
            return cc
        prev_t, prev_c = tt, cc
    return prev_c


def run_round(cfg, n: int, round_idx: int, tasks: list, monitor_samples: list) -> dict:
    """One round: n simultaneous streams, snapshot /metrics around it."""
    picked = [tasks[(round_idx + i) % len(tasks)] for i in range(n)]

    # quiet server before the round (also the redo gate after foreign traffic)
    deadline = time.time() + 180
    while True:
        g = gauges(fetch_metrics(cfg.base))
        if g["running"] == 0 and g["waiting"] == 0:
            break
        if time.time() > deadline:
            break
        time.sleep(2.0)
    time.sleep(cfg.settle_s)
    before = fetch_metrics(cfg.base)
    utc_start = time.time()

    barrier = threading.Barrier(n + 1)
    t_launch = time.perf_counter()
    results = [None] * n

    def worker(idx, task):
        try:
            results[idx] = stream_chat(cfg, task["id"], task["user"], t_launch, barrier)
        except threading.BrokenBarrierError:
            results[idx] = {"task": task["id"], "ok": False, "error": "barrier broken"}
        except Exception as exc:
            results[idx] = {"task": task["id"], "ok": False, "error": type(exc).__name__}

    threads = [threading.Thread(target=worker, args=(i, t)) for i, t in enumerate(picked)]
    for th in threads:
        th.start()
    barrier.wait()
    for th in threads:
        th.join(timeout=cfg.request_timeout + 60)
    wall = time.perf_counter() - t_launch
    after = fetch_metrics(cfg.base)
    utc_end = time.time()

    samples = [r for r in results if r]
    ok = [s for s in samples if s.get("ok")]

    # Common clock: every sample's times are relative to its own t0; shift by
    # launch_offset_ms so first-token and finish times line up on one axis.
    def rel_common(s, key):
        off = (s.get("launch_offset_ms") or 0.0) / 1000.0
        return off + s[key]

    ttfts = [rel_common(s, "ttft_s") for s in ok if s.get("ttft_s") is not None]
    finishes = [rel_common(s, "finish_rel_s") for s in ok]
    window = {"n_ok": len(ok), "start_s": None, "end_s": None, "duration_s": None,
              "tokens": None, "tok_s": None}
    if ttfts and finishes:
        w_start = max(ttfts)             # all N decoding from here
        w_end = min(finishes)            # until the earliest stream finishes
        tokens_in_window = 0.0
        for s in ok:
            ct = s.get("completion_tokens") or 0
            total_chars = s.get("reasoning_chars", 0) + s.get("content_chars", 0)
            if total_chars <= 0:
                continue
            off = (s.get("launch_offset_ms") or 0.0) / 1000.0
            c0 = chars_at(s["timeline"], max(0.0, w_start - off))
            c1 = chars_at(s["timeline"], max(0.0, w_end - off))
            tokens_in_window += ct * max(0.0, c1 - c0) / total_chars
        window = {"n_ok": len(ok), "start_s": w_start, "end_s": w_end,
                  "duration_s": w_end - w_start,
                  "tokens": tokens_in_window,
                  "tok_s": (tokens_in_window / (w_end - w_start)) if w_end > w_start else None}

    total_tokens = sum(s.get("completion_tokens") or 0 for s in ok)
    spec = spec_delta(before, after)
    round_rec = {
        "n": n,
        "round": round_idx,
        "tasks": [t["id"] for t in picked],
        "utc_start": utc_start,
        "utc_end": utc_end,
        "wall_s": wall,
        "simple": {"total_completion_tokens": total_tokens,
                   "wall_s": wall,
                   "tok_s": (total_tokens / wall) if wall > 0 else None},
        "window": window,
        "per_stream_tok_s": [s.get("decode_tok_s") for s in ok],
        "samples": samples,
        "spec": spec,
        "gauges_peak": gauge_peaks(monitor_samples, utc_start, utc_end),
        "foreign": bool(spec.get("request_success_delta", 0) > n),
        "n_failed": len(samples) - len(ok),
        "gauges_before": gauges(before),
    }
    return round_rec


# --------------------------------------------------------------------------
# pre-test idle check (120 seconds unchanged)
# --------------------------------------------------------------------------
def idle_check(cfg) -> bool:
    a = fetch_metrics(cfg.base)
    ga = gauges(a)
    suc_a = ga["request_success_total"]
    time.sleep(120)
    b = fetch_metrics(cfg.base)
    gb = gauges(b)
    ok = (ga["running"] == 0 and ga["waiting"] == 0
          and gb["running"] == 0 and gb["waiting"] == 0
          and gb["request_success_total"] == suc_a)
    print("idle-check: t0 running=%s waiting=%s success=%s | t120 running=%s waiting=%s success=%s -> %s"
          % (ga["running"], ga["waiting"], suc_a, gb["running"], gb["waiting"],
             gb["request_success_total"], "PASS" if ok else "BUSY"), flush=True)
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--prompts", type=Path,
                    help="optional JSON workload with system, tools and three copy tasks")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--streams", type=int, nargs="+", default=list(range(1, 9)),
                    help="stream counts to measure, each from 1 through 8")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--settle-s", type=float, default=5.0)
    ap.add_argument("--min-mem-gib", type=float, default=12.0,
                    help="stop before a stream count if host MemAvailable is below this")
    ap.add_argument("--out", type=Path, default=Path("copy-streams.json"))
    ap.add_argument("--skip-idle-check", action="store_true",
                    help="resume path: skip the 120 s pre-test gate")
    cfg = ap.parse_args(argv)
    if not cfg.streams or any(n < 1 or n > 8 for n in cfg.streams) or len(set(cfg.streams)) != len(cfg.streams):
        ap.error("--streams requires distinct counts from 1 through 8")

    if cfg.prompts:
        with cfg.prompts.open(encoding="utf-8") as fh:
            spec = json.load(fh)
    else:
        spec = generic_copy_spec()
    cfg.system = spec.get("system") or "You are a helpful assistant."
    cfg.tools = spec.get("tools") or []
    tasks = [t for t in (spec.get("tasks") or []) if t.get("category") == "copy"] or spec.get("tasks")
    if len(tasks) != 3 or any(not t.get("user") for t in tasks):
        raise SystemExit("workload needs three copy tasks with user fields")
    tasks = [dict(task, id=f"copy{index}") for index, task in enumerate(tasks, 1)]
    print("workload: %d copy tasks, effort=%s max_tokens=%d"
          % (len(tasks), cfg.effort, cfg.max_tokens), flush=True)

    if not cfg.skip_idle_check:
        deadline = time.time() + 2 * 3600
        while True:
            if idle_check(cfg):
                break
            if time.time() > deadline:
                print("server busy after 2 h of rechecks -- stopping without measuring.", flush=True)
                return 3
            print("box busy -- waiting 300 s and re-checking", flush=True)
            time.sleep(300)

    utc_test_start = time.time()
    print("TEST-UTC-START %s (%.3f)" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(utc_test_start)),
                                        utc_test_start), flush=True)

    # Warm the shared prefix with the first task; do not record this request.
    print("warmup: launching 1 (not recorded)", flush=True)
    b = threading.Barrier(2)
    t_launch = time.perf_counter()
    th = threading.Thread(target=lambda: stream_chat(cfg, "warmup-0", tasks[0]["user"], t_launch, b))
    th.start()
    b.wait()
    th.join(timeout=cfg.request_timeout + 60)
    print("warmup: done", flush=True)

    rounds = []
    monitor_samples = []
    monitor_stop = threading.Event()
    monitor_thread = threading.Thread(target=monitor_gauges,
                                      args=(cfg.base, monitor_stop, monitor_samples), daemon=True)
    monitor_thread.start()
    for n in cfg.streams:
        available = mem_available_gib()
        print("N=%d MemAvailable-before %.2f GiB" % (n, available), flush=True)
        if available < cfg.min_mem_gib:
            print("STOP: MemAvailable %.2f GiB below %.2f GiB" % (available, cfg.min_mem_gib), flush=True)
            break
        for r in (0, 1, 2):
            rec = run_round(cfg, n, r, tasks, monitor_samples)
            if rec["foreign"]:
                # protocol-mandated redo: discard, fresh idle check, redo once
                print("round N=%d r=%d: FOREIGN traffic (success delta %s > %d) -- discard + redo"
                      % (n, r, rec["spec"].get("request_success_delta"), n), flush=True)
                rec["discarded_foreign"] = True
                rounds.append(rec)
                deadline = time.time() + 180
                while True:
                    a = fetch_metrics(cfg.base)
                    g = gauges(a)
                    if g["running"] == 0 and g["waiting"] == 0:
                        break
                    if time.time() > deadline:
                        break
                    time.sleep(2.0)
                suc_a = gauges(fetch_metrics(cfg.base))["request_success_total"]
                time.sleep(60)
                suc_b = gauges(fetch_metrics(cfg.base))["request_success_total"]
                if suc_b != suc_a:
                    print("still busy after foreign round -- stopping.", flush=True)
                    rounds[-1]["redo_aborted_busy"] = True
                    break
                rec = run_round(cfg, n, r, tasks, monitor_samples)
                rec["redo_of_foreign"] = True
            rounds.append(rec)
            w = rec["window"]
            print("round N=%d r=%d tasks=%s ok=%d/%d window=%.3fs tokens=%.1f -> %.2f tok/s | simple %.2f tok/s | tps %s"
                  % (n, r, ",".join(rec["tasks"]), w["n_ok"], n,
                     w["duration_s"] or 0.0, w["tokens"] or 0.0, w["tok_s"] or 0.0,
                     rec["simple"]["tok_s"] or 0.0,
                     ("%.3f" % rec["spec"]["tokens_per_step"]) if rec["spec"].get("tokens_per_step") else "n/a"),
                  flush=True)
            with cfg.out.with_suffix(cfg.out.suffix + ".partial").open("w", encoding="utf-8", newline="\n") as fh:
                json.dump({"rounds": rounds}, fh, indent=1)
        print("N=%d MemAvailable-after %.2f GiB" % (n, mem_available_gib()), flush=True)

    monitor_stop.set()
    monitor_thread.join(timeout=2.0)

    utc_test_end = time.time()
    print("TEST-UTC-END %s (%.3f)" % (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(utc_test_end)),
                                      utc_test_end), flush=True)

    # summary per N over kept rounds
    summary = {}
    for n in cfg.streams:
        kept = [r for r in rounds if r["n"] == n and not r.get("discarded_foreign")]
        agg = [r["window"]["tok_s"] for r in kept if r["window"]["tok_s"]]
        simple = [r["simple"]["tok_s"] for r in kept if r["simple"]["tok_s"]]
        ps = [v for r in kept for v in (r["per_stream_tok_s"] or []) if v]
        ttfts = [s["ttft_s"] for r in kept for s in r["samples"] if s.get("ok") and s.get("ttft_s") is not None]
        tps = [r["spec"]["tokens_per_step"] for r in kept if r["spec"].get("tokens_per_step")]
        running_peaks = [r["gauges_peak"]["running_peak"] for r in kept
                         if r["gauges_peak"]["running_peak"] is not None]
        waiting_peaks = [r["gauges_peak"]["waiting_peak"] for r in kept
                         if r["gauges_peak"]["waiting_peak"] is not None]
        summary[str(n)] = {
            "rounds_kept": len(kept),
            "window_tok_s": {"peak": max(agg) if agg else None,
                             "median": statistics.median(agg) if agg else None,
                             "values": agg},
            "simple_tok_s": {"median": statistics.median(simple) if simple else None, "values": simple},
            "per_stream_tok_s": {"median": statistics.median(ps) if ps else None,
                                 "min": min(ps) if ps else None, "max": max(ps) if ps else None},
            "tokens_per_step": {"median": statistics.median(tps) if tps else None},
            "ttft_s": {"median": statistics.median(ttfts) if ttfts else None,
                       "max": max(ttfts) if ttfts else None},
            "running_peak": max(running_peaks) if running_peaks else None,
            "waiting_peak": max(waiting_peaks) if waiting_peaks else None,
        }

    level_off = None
    for previous, current in zip(cfg.streams, cfg.streams[1:]):
        prior_peak = summary[str(previous)]["window_tok_s"]["peak"]
        current_peak = summary[str(current)]["window_tok_s"]["peak"]
        if prior_peak and current_peak and (current_peak - prior_peak) / prior_peak < 0.05:
            level_off = current
            break

    out = {
        "label": "copy-heavy decode; three rounds per configured stream count",
        "model": cfg.model,
        "effort": cfg.effort,
        "max_tokens": cfg.max_tokens,
        "workload": "user-supplied" if cfg.prompts else "generated public copy tasks",
        "utc_test_start": utc_test_start,
        "utc_test_end": utc_test_end,
        "rounds": rounds,
        "summary": summary,
        "level_off_n": level_off,
    }
    with cfg.out.open("w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1)
        fh.write("\n")
    for n in cfg.streams:
        s = summary[str(n)]
        print("N=%d peak=%.2f median=%.2f per-stream=%.2f tps=%s ttft=%s"
              % (n, s["window_tok_s"]["peak"] or 0, s["window_tok_s"]["median"] or 0,
                 s["per_stream_tok_s"]["median"] or 0,
                 ("%.3f" % s["tokens_per_step"]["median"]) if s["tokens_per_step"]["median"] else "n/a",
                 ("%.3f" % s["ttft_s"]["median"]) if s["ttft_s"]["median"] else "n/a"), flush=True)
    print("wrote %s" % cfg.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
