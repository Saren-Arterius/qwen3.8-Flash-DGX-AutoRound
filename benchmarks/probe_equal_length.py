# SPDX-License-Identifier: LicenseRef-PolyForm-Noncommercial-1.0.0
"""Equal-length concurrent decode probe; requires an explicit guard file."""
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

# --- bench_agent.py conventions, verbatim ---------------------------------
_SAMPLE_RE = re.compile(r'^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[^\s]+)\s*$')
_LABEL_RE = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')

SPEC_DRAFTS = "vllm:spec_decode_num_drafts_total"
SPEC_DRAFT_TOKENS = "vllm:spec_decode_num_draft_tokens_total"
SPEC_ACCEPTED = "vllm:spec_decode_num_accepted_tokens_total"
SPEC_ACCEPTED_POS = "vllm:spec_decode_num_accepted_tokens_per_pos_total"
PREFIX_HITS = "vllm:prefix_cache_hits_total"
PREFIX_QUERIES = "vllm:prefix_cache_queries_total"
SPEC_POSITIONS = ("0", "1", "2", "3", "4")

# --- request-occupancy gauges needed by the equal-length probe --------------
RUNNING = "vllm:num_requests_running"
WAITING = "vllm:num_requests_waiting"
# ms/step, the same route MiaAI's bench/sweep.py takes: d(sum)/d(count)
ITL_SUM = "vllm:inter_token_latency_seconds_sum"
ITL_COUNT = "vllm:inter_token_latency_seconds_count"

DEFAULT_GUARD = ""

# A neutral, self-contained prompt. Short on purpose: this probe is about the
# SCHEDULER, and a long prompt would put prefix-cache behaviour and prefill
# chunking into a decode measurement. It is also identical across streams, so
# every stream hits the same cached prefix and no stream pays a different TTFT.
DEFAULT_PROMPT = (
    "Write a short, plain-language explanation of how a hash map resolves "
    "collisions. Do not use bullet points."
)


# --------------------------------------------------------------------------
# /metrics
# --------------------------------------------------------------------------
def parse_metrics(text: str) -> dict:
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
    for labels, _v in metrics.get(name, []):
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
        per_pos.append(metric_value(after, SPEC_ACCEPTED_POS, position=pos)
                       - metric_value(before, SPEC_ACCEPTED_POS, position=pos))
    out = {
        "drafts": drafts,
        "draft_tokens": metric_value(after, SPEC_DRAFT_TOKENS) - metric_value(before, SPEC_DRAFT_TOKENS),
        "accepted_tokens": accepted,
        "accepted_per_pos": per_pos,
        "prefix_hits": metric_value(after, PREFIX_HITS) - metric_value(before, PREFIX_HITS),
        "prefix_queries": metric_value(after, PREFIX_QUERIES) - metric_value(before, PREFIX_QUERIES),
    }
    if drafts > 0:
        out["tokens_per_step"] = 1.0 + accepted / drafts
        out["accept_pos_rate"] = [None if p is None else p / drafts for p in per_pos]
    else:
        out["tokens_per_step"] = None
        out["accept_pos_rate"] = [None] * len(SPEC_POSITIONS)
    return out


def itl_ms_per_step(before: dict, after: dict):
    """ms per engine step from d(sum)/d(count) -- MiaAI's own derivation."""
    if not before or not after:
        return None
    ds = metric_value(after, ITL_SUM) - metric_value(before, ITL_SUM)
    dc = metric_value(after, ITL_COUNT) - metric_value(before, ITL_COUNT)
    return (1000.0 * ds / dc) if dc > 0 else None


# --------------------------------------------------------------------------
# refusals (probe_determinism.py conventions)
# --------------------------------------------------------------------------
def require_guard(path: str) -> None:
    if not path:
        raise SystemExit("refusing to start: --server-guard-file is empty")
    if not os.path.exists(path):
        raise SystemExit(
            "refusing to start: guard file %s does not exist.\n"
            "A measurement run owns the server until it drops that marker." % path)
    print("guard ok: %s" % path)


def require_no_measurement_process() -> None:
    """Second, independent refusal: no measurement process may be alive.

    The guard file says a sweep FINISHED; this says nothing is running RIGHT
    NOW -- including a chain member that has not written its DONE yet.
    """
    import subprocess
    pat = r'[r]un_tuning.py|[b]ench_agent|[b]ench_longctx|[p]robe_determinism|[r]un_gate'
    try:
        out = subprocess.run(["pgrep", "-af", pat], capture_output=True, text=True).stdout.strip()
    except Exception:
        out = ""
    if out:
        raise SystemExit("refusing to start: a measurement process owns the box:\n" + out)
    print("no measurement process running: ok")


def require_idle(base: str, seconds: int, poll: float = 2.0) -> None:
    if seconds <= 0:
        raise SystemExit("refusing to start: --idle-seconds must be > 0")
    print("idle check: %s must report running == 0 and waiting == 0 for %ds ..."
          % (base.rstrip("/") + "/metrics", seconds))
    start = time.time()
    while True:
        m = fetch_metrics(base)
        if not m:
            raise SystemExit("refusing to start: cannot read /metrics")
        r, w = metric_value(m, RUNNING), metric_value(m, WAITING)
        if r > 0 or w > 0:
            raise SystemExit("refusing to start: server is busy (running=%g waiting=%g). "
                             "Another client is using the server." % (r, w))
        if time.time() - start >= seconds:
            break
        time.sleep(poll)
    print("idle check passed (%ds quiet)" % seconds)


# --------------------------------------------------------------------------
# Gauge sampler for live request occupancy, which logs cannot reconstruct
# --------------------------------------------------------------------------
class GaugeSampler(threading.Thread):
    def __init__(self, base: str, hz: float):
        super().__init__(daemon=True)
        self.base, self.period = base, 1.0 / max(0.1, hz)
        self.samples: list = []
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            t = time.perf_counter()
            m = fetch_metrics(self.base, timeout=3.0)
            if m:
                self.samples.append({
                    "t": t,
                    "running": metric_value(m, RUNNING),
                    "waiting": metric_value(m, WAITING),
                })
            self._stop.wait(self.period)

    def stop(self) -> None:
        self._stop.set()


def summarize_gauges(samples: list, t_lo: float, t_hi: float) -> dict:
    """Mean/max running and waiting over [t_lo, t_hi], plus the running histogram."""
    win = [s for s in samples if t_lo <= s["t"] <= t_hi]
    if not win:
        return {"n": 0}
    run = [s["running"] for s in win]
    wait = [s["waiting"] for s in win]
    hist: dict = {}
    for v in run:
        hist[int(round(v))] = hist.get(int(round(v)), 0) + 1
    n = float(len(win))
    return {
        "n": len(win),
        "running_mean": sum(run) / n,
        "running_max": max(run),
        "running_median": statistics.median(run),
        "waiting_mean": sum(wait) / n,
        "waiting_max": max(wait),
        "running_histogram_pct": {k: 100.0 * v / n for k, v in sorted(hist.items())},
    }


# --------------------------------------------------------------------------
# one equal-length streaming request
# --------------------------------------------------------------------------
def build_body(cfg, prompt: str) -> dict:
    body = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        # THE POINT OF THE PROBE: every stream decodes exactly cfg.tokens tokens,
        # so all N start and finish together and there is no ragged tail to
        # correct for. This is MiaAI's DecodeBench.js:111-116, 280-281 shape.
        "max_tokens": cfg.tokens,
        "min_tokens": cfg.tokens,
        "ignore_eos": True,
        "chat_template_kwargs": {"reasoning_effort": cfg.effort,
                                 "enable_thinking": not cfg.no_thinking},
    }
    if cfg.greedy:
        body.update({"temperature": 0.0, "top_p": 1.0})
    else:
        body.update({"temperature": 1.0, "top_p": 0.95, "top_k": 20})
    return body


def stream_one(cfg, prompt: str, tag: str) -> dict:
    sample = {"stream": tag, "ok": False}
    req = urllib.request.Request(
        cfg.base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(build_body(cfg, prompt)).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    ttft = None
    usage = None
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
        sample["ok"] = True
    except urllib.error.HTTPError as exc:
        try:
            sample["error"] = "HTTP %s: %s" % (exc.code, exc.read().decode("utf-8", "replace")[:500])
        except Exception:
            sample["error"] = "HTTP %s" % exc.code
    except Exception as exc:
        sample["error"] = "%s: %s" % (type(exc).__name__, exc)

    total = time.perf_counter() - t0
    usage = usage or {}
    ct = usage.get("completion_tokens")
    sample.update({
        "t_start": t0, "t_end": t0 + total,
        "ttft_s": ttft, "total_s": total, "chunks": chunks,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": ct,
        "cached_tokens": ((usage.get("prompt_tokens_details") or {}).get("cached_tokens")),
        "decode_tok_s": (ct / (total - ttft)) if (ct and ttft is not None and total > ttft) else None,
    })
    return sample


# --------------------------------------------------------------------------
# one concurrency level
# --------------------------------------------------------------------------
def run_level(cfg, n: int, sampler: GaugeSampler) -> dict:
    results: list = [None] * n
    before = fetch_metrics(cfg.base)

    def worker(i):
        results[i] = stream_one(cfg, cfg.prompt, "s%d" % i)

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=cfg.request_timeout + 60)
    wall = time.perf_counter() - t0
    after = fetch_metrics(cfg.base)

    samples = [r for r in results if r]
    ok = [s for s in samples if s["ok"]]
    tokens = sum(s.get("completion_tokens") or 0 for s in ok)

    # the decode window: every stream is the same length, so it is simply
    # [max ttft, min end]. With ignore_eos + min_tokens this is nearly the whole
    # wall clock, which is the entire point of the probe.
    lo = max((s["t_start"] + (s["ttft_s"] or 0.0)) for s in ok) if ok else t0
    hi = min(s["t_end"] for s in ok) if ok else t0 + wall

    lens = [s.get("completion_tokens") for s in ok]
    equal = bool(lens) and len(set(lens)) == 1 and lens[0] == cfg.tokens

    return {
        "n": n,
        "wall_s": wall,
        "decode_window_s": max(0.0, hi - lo),
        "equal_length": equal,
        "completion_tokens": lens,
        "total_completion_tokens": tokens,
        "aggregate_tok_s": (tokens / wall) if wall > 0 else None,
        "aggregate_tok_s_decode_window": (n * cfg.tokens / (hi - lo)) if hi > lo else None,
        "per_stream_tok_s": [s.get("decode_tok_s") for s in ok],
        "ttft_s": [s.get("ttft_s") for s in ok],
        "ms_per_step": itl_ms_per_step(before, after),
        "spec": spec_delta(before, after),
        "gauges": summarize_gauges(sampler.samples, lo, hi),
        "samples": samples,
        "n_failed": len(samples) - len(ok),
        "errors": [s.get("error") for s in samples if not s["ok"]][:8],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="http://localhost:8000")
    ap.add_argument("--model", default="qwen")
    ap.add_argument("--tag", default="", help="free-form label copied into the output")
    ap.add_argument("--out", default="c4_equal_length.json")
    ap.add_argument("--streams", type=int, nargs="+", default=[1, 2, 3, 4, 5],
                    help="concurrency levels to run, in order (default 1 2 3 4 5). "
                         "3, 4 and 5 are the levels vllm#55533 turns on.")
    ap.add_argument("--tokens", type=int, default=600,
                    help="forced output tokens per stream (min_tokens == max_tokens)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--effort", default="xhigh")
    ap.add_argument("--no-thinking", action="store_true",
                    help="chat_template_kwargs.enable_thinking=false, MiaAI's shape")
    ap.add_argument("--greedy", action="store_true",
                    help="temperature 0 / top_p 1, MiaAI's sampling. Off by default: "
                         "this probe measures OUR scheduler through OUR sampler path.")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--poll-hz", type=float, default=2.0,
                    help="gauge sampling rate for num_requests_running/waiting")
    ap.add_argument("--request-timeout", type=float, default=900.0)
    ap.add_argument("--server-guard-file", default=DEFAULT_GUARD)
    ap.add_argument("--idle-seconds", type=int, default=30)
    ap.add_argument("--settle-s", type=float, default=10.0,
                    help="quiet time between concurrency levels")
    cfg = ap.parse_args(argv)

    require_guard(cfg.server_guard_file)
    require_no_measurement_process()
    require_idle(cfg.base, cfg.idle_seconds)

    m0 = fetch_metrics(cfg.base)
    if not metric_present(m0, RUNNING) or not metric_present(m0, WAITING):
        raise SystemExit("refusing to start: this server does not expose %s / %s -- "
                         "the whole point of the probe" % (RUNNING, WAITING))

    sampler = GaugeSampler(cfg.base, cfg.poll_hz)
    sampler.start()

    started = time.time()
    for i in range(max(0, cfg.warmup)):
        s = stream_one(cfg, cfg.prompt, "warmup-%d" % i)
        print("  warmup %d: ok=%s %.1fs" % (i, s["ok"], s["total_s"]), file=sys.stderr)

    levels = []
    for n in cfg.streams:
        print("  level n=%d ..." % n, file=sys.stderr)
        lv = run_level(cfg, n, sampler)
        g = lv["gauges"]
        print("    aggregate %.2f tok/s | ms/step %s | L %s | running mean %s max %s | waiting mean %s"
              % (lv["aggregate_tok_s"] or 0,
                 ("%.2f" % lv["ms_per_step"]) if lv["ms_per_step"] else "-",
                 ("%.4f" % lv["spec"]["tokens_per_step"]) if lv["spec"].get("tokens_per_step") else "-",
                 ("%.2f" % g["running_mean"]) if g.get("n") else "-",
                 ("%.0f" % g["running_max"]) if g.get("n") else "-",
                 ("%.2f" % g["waiting_mean"]) if g.get("n") else "-"),
              file=sys.stderr)
        levels.append(lv)
        time.sleep(cfg.settle_s)

    sampler.stop()

    # The two readings the probe exists for.
    scaling = {}
    by_n = {lv["n"]: lv for lv in levels}
    if 1 in by_n and 4 in by_n and by_n[1]["ms_per_step"] and by_n[4]["ms_per_step"]:
        scaling["step_growth_1_to_4"] = by_n[4]["ms_per_step"] / by_n[1]["ms_per_step"]
        scaling["note"] = ("compare with MiaAI's 1.67x (their ignore_eos, equal-length, "
                           "30-token prompts) and with our 2.13x, which was DERIVED from "
                           "a ragged-tail agent aggregate, not measured")
    cap = {}
    for lv in levels:
        g = lv["gauges"]
        if g.get("n"):
            cap[lv["n"]] = {"running_mean": g["running_mean"], "running_max": g["running_max"],
                            "waiting_mean": g["waiting_mean"]}
    scaling["running_by_level"] = cap
    scaling["c6_verdict"] = None
    if 4 in cap:
        scaling["c6_verdict"] = ("CAPPED (vllm#55533 reproduces here)"
                                 if cap[4]["running_mean"] < 3.5 else
                                 "NOT CAPPED at n=4")
        scaling["verify_width_at_4"] = 4 * cap[4]["running_mean"]

    result = {
        "tag": cfg.tag, "base": cfg.base, "model": cfg.model,
        "tokens_per_stream": cfg.tokens, "greedy": cfg.greedy,
        "thinking": not cfg.no_thinking, "effort": cfg.effort,
        "prompt": cfg.prompt,
        "started_epoch": started, "elapsed_s": time.time() - started,
        "levels": levels,
        "scaling": scaling,
        "gauge_series": sampler.samples,
        "caveat": ("ignore_eos output past EOS is degenerate BY CONSTRUCTION. "
                   "It must not be scored for quality, compared to any eval, or "
                   "read as a decode result for the recipe -- it is a scheduler "
                   "measurement only. This is the same caveat that makes MiaAI's "
                   "61.5 tok/s a shape rather than a stack."),
    }
    with open(cfg.out, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")
    print("probe_c4_equal_length: wrote %s" % cfg.out, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
