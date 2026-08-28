#!/usr/bin/env python3
"""bench_c1c6.py — decode-throughput bench at concurrency 1..6 for a vLLM endpoint.

Usage: python3 bench_c1c6.py [--url http://localhost:8000] [--rounds 3] [--max-tokens 700]
For each concurrency level c in 1..6: run `rounds` waves of c parallel requests,
report aggregate output tok/s, per-stream decode tok/s, and TTFT. Scrapes
vLLM spec-decode acceptance from /metrics when available.
Prompts are code/reasoning-flavored (realistic acceptance rates), unique per
request to defeat prefix caching.
"""
import argparse, json, threading, time, urllib.request, random, re

PROMPTS = [
    "Write a Python function that parses an nginx access log line into a dict, with a regex, and explain each group.",
    "Implement a rate limiter class in Python using the token bucket algorithm, then show example usage.",
    "A warehouse ships 340 orders/day growing 6% weekly. Model 8 weeks of volume in a Python list comprehension and explain.",
    "Write a SQL query for the top 5 customers by 90-day revenue, then rewrite it as a window function version.",
    "Explain the difference between TCP slow start and congestion avoidance, then pseudocode both.",
    "Implement binary search in Python, then walk through the trace on [2,5,8,12,17,23] searching for 17.",
]

def post(url, payload, timeout=900):
    req = urllib.request.Request(url + "/v1/chat/completions",
        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def one_request(url, prompt, max_tokens, out, idx):
    salt = f"[run {random.randint(1,10**9)}] "
    p = {"model": "glm-5.3-flash",
         "messages": [{"role": "user", "content": salt + prompt}],
         "max_tokens": max_tokens, "temperature": 1.0, "top_p": 0.95}
    t0 = time.time()
    try:
        r = post(url, p)
        dt = time.time() - t0
        ct = r["usage"]["completion_tokens"]
        out[idx] = {"ok": True, "seconds": dt, "completion_tokens": ct}
    except Exception as e:
        out[idx] = {"ok": False, "err": str(e)[:80]}

def metrics_accept(url):
    try:
        txt = urllib.request.urlopen(url + "/metrics", timeout=10).read().decode()
        drafted = accepted = None
        for line in txt.splitlines():
            if line.startswith("vllm:spec_decode_num_draft_tokens_total"):
                drafted = float(line.split()[-1])
            if line.startswith("vllm:spec_decode_num_accepted_tokens_total"):
                accepted = float(line.split()[-1])
        if drafted and accepted is not None:
            return drafted, accepted
    except Exception:
        pass
    return None, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--levels", default="1,2,3,4,5,6")
    a = ap.parse_args()

    # warmup
    one_request(a.url, PROMPTS[0], 64, {}, 0)

    print(f"{'c':>2} {'reqs':>4} {'agg_tok/s':>10} {'per_stream':>10} {'mean_s':>7} {'fails':>5}  accept_len")
    results = {}
    for c in [int(x) for x in a.levels.split(",")]:
        d0, a0 = metrics_accept(a.url)
        agg_toks, agg_time, per_stream, fails, nreq = 0, 0.0, [], 0, 0
        for _ in range(a.rounds):
            out = {}
            ts = [threading.Thread(target=one_request,
                    args=(a.url, PROMPTS[(i * 7 + c) % len(PROMPTS)], a.max_tokens, out, i))
                  for i in range(c)]
            t0 = time.time()
            [t.start() for t in ts]; [t.join() for t in ts]
            wall = time.time() - t0
            ok = [v for v in out.values() if v.get("ok")]
            fails += c - len(ok)
            nreq += c
            toks = sum(v["completion_tokens"] for v in ok)
            agg_toks += toks; agg_time += wall
            per_stream += [v["completion_tokens"] / v["seconds"] for v in ok]
        d1, a1 = metrics_accept(a.url)
        acc = ""
        if d0 is not None and d1 is not None and d1 > d0:
            # accepted-per-drafted ratio over this level's window; accept_len ~= 1 + accepted/steps
            acc = f"ratio={ (a1 - a0) / (d1 - d0):.3f}"
        agg = agg_toks / agg_time if agg_time else 0
        ps = sum(per_stream) / len(per_stream) if per_stream else 0
        ms = agg_time / a.rounds
        results[c] = {"agg_toks_per_s": round(agg, 1), "per_stream_toks_per_s": round(ps, 1),
                      "mean_wall_s": round(ms, 1), "fails": fails, "accept": acc}
        print(f"{c:>2} {nreq:>4} {agg:>10.1f} {ps:>10.1f} {ms:>7.1f} {fails:>5}  {acc}")
    print(json.dumps(results))

if __name__ == "__main__":
    main()
