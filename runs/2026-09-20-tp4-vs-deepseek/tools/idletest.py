#!/usr/bin/env python3
"""Idle-vs-busy step test (root on Reddie). Tony's question: does the GPU drop into the slow state
when idle? Streams short requests and measures every speculative step from chunk timings:
  A count  after the server sat idle
  B count  right after A,  C code right after B
  (idle IDLE_S seconds)
  D count  after idle,     E code right after D
Per request: throughput (tokens after the first / time after the first), median step ms,
first-10 vs last-10 step medians, and the step series."""
import json
import os
import time
import urllib.request

U = "http://127.0.0.1:8000/v1/chat/completions"
M = __import__("os").environ.get("BENCH_MODEL", "deepseek-v4.1-flash")
IDLE_S = int(os.environ.get("IDLE_S", "45"))
COUNT = "Count from 1 to 100, separated by spaces. Output only the numbers."
CODE = ("Write a Python function merge_intervals(intervals) that merges overlapping intervals and returns them "
        "sorted. Include a one-line docstring and two example calls.")


def med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else float("nan")


def run(tag, prompt, max_tokens):
    body = {"model": M, "messages": [{"role": "user", "content": f"[{tag}] {prompt}"}], "max_tokens": max_tokens,
            "temperature": 0, "stream": True, "stream_options": {"include_usage": True, "continuous_usage_stats": True}}
    req = urllib.request.Request(U, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    pts = []  # (t, cumulative completion tokens)
    with urllib.request.urlopen(req, timeout=600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            u = d.get("usage") or {}
            ct = u.get("completion_tokens")
            if ct is not None and (not pts or ct > pts[-1][1]):
                pts.append((time.time() - t0, ct))
    if len(pts) < 3:
        print(f"{tag}: too few chunks ({pts})", flush=True)
        return
    steps = [(pts[i][0] - pts[i - 1][0]) * 1000 for i in range(2, len(pts))]  # skip prefill -> first token
    toks = pts[-1][1] - pts[1][1]
    tps = toks / (pts[-1][0] - pts[1][0])
    print(f"{tag}: {pts[-1][1]} tok, TTFT {pts[0][0]:.2f}s, {tps:5.1f} tok/s after first token, "
          f"{toks / len(steps):.2f} tok/step, step ms p50 {med(steps):5.1f} (first10 {med(steps[:10]):5.1f}, "
          f"last10 {med(steps[-10:]):5.1f}) | {' '.join(str(round(s)) for s in steps)}", flush=True)


print(f"start {time.strftime('%H:%M:%S', time.gmtime())} UTC", flush=True)
run("A count after idle", COUNT, 256)
run("B count back-to-back", COUNT, 256)
run("C code back-to-back", CODE, 256)
print(f"idle {IDLE_S}s", flush=True)
time.sleep(IDLE_S)
run("D count after idle", COUNT, 256)
run("E code back-to-back", CODE, 256)
