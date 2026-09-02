#!/usr/bin/env python3
"""bench_glm53_tp4.py — config-comparison bench for GLM-5.3-Flash on the 4-Spark TP4 fleet.

Why not probes/bench_c1c6.py: that harness runs temperature=1.0 (content variance is
acceptance variance on a spec-decode engine), reports means not medians, and is
non-streaming so it cannot separate prefill from decode at all. It also has no
count-to-100 prompt, which is our PEAK number.

This one:
  * fixed prompt set, temperature 0 -> the same tokens every run, so a config delta
    is a config delta and not a content delta
  * streaming, so TTFT / prefill tok/s / decode tok/s / e2e tok/s split out
  * MEDIAN + P90 + peak across repeats (Tony's rule), not pooled means
  * C1..C6 aggregate sweep
  * long-context arm (32K prompt, 1 vs 2 concurrent) -- the regime issue #14 says
    short-prompt sweeps structurally cannot see
  * spec-decode acceptance scraped from /metrics as BOTH accepted/drafted ratio and
    mean accepted length (1 + accepted/drafts), plus per-position when exposed

Usage:
  python3 bench_glm53_tp4.py --label baseline-eager-k7 --suite all
  python3 bench_glm53_tp4.py --label nccl-tree-ll --suite single,sweep
Results append to results/<label>.json and print a table.
"""
import argparse, json, os, statistics, threading, time, urllib.request, urllib.error

URL = "http://100.113.138.96:8000"
MODEL = "glm-5.3-flash"

# ---- fixed prompt set -------------------------------------------------------
# count100 is the PEAK probe: maximally predictable output -> highest draft
# acceptance -> the ceiling this engine can hit. code/prose bracket real traffic.
PROMPTS = {
    "count100": ("Count from 1 to 100. Output only the numbers, one per line, nothing else.", 400),
    "code":     ("Write a complete Python implementation of an LRU cache with get and put in "
                 "O(1), using a dict plus a doubly linked list. Include the class, full method "
                 "bodies, and a short docstring for each method.", 700),
    "prose":    ("Explain, in flowing prose with no code, no lists and no headings, how a "
                 "modern CPU's branch predictor works and why mispredictions are expensive on "
                 "a deeply pipelined machine.", 700),
}


def post_stream(prompt, max_tokens, temperature=0.0, timeout=900):
    """One streaming request -> timing split. Returns dict or {'ok': False}."""
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft = None
    completion = prompt_toks = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                ch = ev.get("choices") or []
                if ch and ttft is None:
                    d = ch[0].get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        ttft = time.perf_counter()
                u = ev.get("usage")
                if u:
                    completion = u.get("completion_tokens")
                    prompt_toks = u.get("prompt_tokens")
        t1 = time.perf_counter()
    except Exception as e:
        return {"ok": False, "err": str(e)[:120]}
    if not completion:
        return {"ok": False, "err": "no usage in stream"}
    ttft = ttft or t1
    decode_s = max(t1 - ttft, 1e-9)
    return {
        "ok": True,
        "ttft_s": ttft - t0,
        "total_s": t1 - t0,
        "prompt_tokens": prompt_toks,
        "completion_tokens": completion,
        # prefill rate: prompt tokens digested per second before first token
        "prefill_tok_s": (prompt_toks / (ttft - t0)) if prompt_toks and (ttft - t0) > 0 else None,
        "decode_tok_s": (completion - 1) / decode_s,
        "e2e_tok_s": completion / (t1 - t0),
    }


# ---- spec-decode metrics ----------------------------------------------------
def spec_metrics():
    """Scrape vLLM spec-decode counters. Returns dict of name -> float."""
    out = {}
    try:
        txt = urllib.request.urlopen(URL + "/metrics", timeout=15).read().decode()
    except Exception:
        return out
    for line in txt.splitlines():
        if line.startswith("#") or "spec_decode" not in line:
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            pass
    return out


def spec_delta(before, after):
    """accepted/drafted ratio + mean accepted length + per-position acceptance."""
    def g(d, key):
        for k, v in d.items():
            if k.startswith(key):
                return v
        return None

    dd = (g(after, "vllm:spec_decode_num_draft_tokens_total") or 0) - \
         (g(before, "vllm:spec_decode_num_draft_tokens_total") or 0)
    da = (g(after, "vllm:spec_decode_num_accepted_tokens_total") or 0) - \
         (g(before, "vllm:spec_decode_num_accepted_tokens_total") or 0)
    dn = (g(after, "vllm:spec_decode_num_drafts_total") or 0) - \
         (g(before, "vllm:spec_decode_num_drafts_total") or 0)
    res = {}
    if dd > 0:
        res["accept_ratio"] = round(da / dd, 4)
    if dn > 0:
        # mean accepted length per step, counting the always-free bonus token
        res["mean_accept_len"] = round(1 + da / dn, 3)
    # per-position acceptance, when the build exposes the histogram
    pos = {}
    for k, v in after.items():
        if "accepted_tokens_per_pos" in k:
            dv = v - before.get(k, 0.0)
            if dv:
                pos[k.split("{")[-1].rstrip("}")] = dv
    if pos and dn > 0:
        res["per_pos"] = {p: round(c / dn, 3) for p, c in sorted(pos.items())}
    return res


def stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    s = sorted(vals)
    return {
        "median": round(statistics.median(s), 2),
        "p90": round(s[min(len(s) - 1, int(0.9 * len(s)))], 2),
        "peak": round(max(s), 2),
        "n": len(s),
    }


# ---- suites -----------------------------------------------------------------
def suite_single(reps):
    """Fixed prompts, single stream, repeated. The headline + PEAK numbers."""
    res = {}
    for name, (prompt, mt) in PROMPTS.items():
        before = spec_metrics()
        runs = []
        for _ in range(reps):
            r = post_stream(prompt, mt)
            if r.get("ok"):
                runs.append(r)
        after = spec_metrics()
        if not runs:
            res[name] = {"error": "all runs failed"}
            continue
        res[name] = {
            "decode_tok_s": stats([r["decode_tok_s"] for r in runs]),
            "e2e_tok_s": stats([r["e2e_tok_s"] for r in runs]),
            "ttft_s": stats([r["ttft_s"] for r in runs]),
            "completion_tokens": runs[0]["completion_tokens"],
            "prompt_tokens": runs[0]["prompt_tokens"],
            "spec": spec_delta(before, after),
        }
        print(f"  {name:9s} decode med={res[name]['decode_tok_s'].get('median')} "
              f"peak={res[name]['decode_tok_s'].get('peak')} "
              f"ttft={res[name]['ttft_s'].get('median')}s "
              f"spec={res[name]['spec']}", flush=True)
    return res


def suite_prefill(sizes=(4096, 16384, 32768)):
    """Prefill throughput vs prompt length. Unique filler defeats any prefix reuse."""
    res = {}
    for n in sizes:
        # ~1 token per word for this filler; salt keeps each run distinct
        filler = " ".join(f"w{i}" for i in range(n))
        prompt = f"[{time.time()}] Here is a corpus:\n{filler}\n\nReply with exactly: DONE"
        r = post_stream(prompt, 16)
        if r.get("ok"):
            res[f"{n}"] = {
                "prompt_tokens": r["prompt_tokens"],
                "ttft_s": round(r["ttft_s"], 3),
                "prefill_tok_s": round(r["prefill_tok_s"], 1) if r["prefill_tok_s"] else None,
            }
            print(f"  prefill~{n:6d} tokens={r['prompt_tokens']} ttft={r['ttft_s']:.2f}s "
                  f"rate={r['prefill_tok_s']:.0f} tok/s", flush=True)
        else:
            res[f"{n}"] = {"error": r.get("err")}
            print(f"  prefill~{n:6d} FAILED {r.get('err')}", flush=True)
    return res


def _wave(prompt, mt, c, out):
    ths = []
    for i in range(c):
        def go(i=i):
            out[i] = post_stream(f"[w{i}] " + prompt, mt)
        t = threading.Thread(target=go)
        ths.append(t)
    t0 = time.perf_counter()
    [t.start() for t in ths]
    [t.join() for t in ths]
    return time.perf_counter() - t0


def suite_sweep(levels, rounds):
    """C1..CN aggregate throughput. Aggregate = total output tokens / wall."""
    res = {}
    prompt, mt = PROMPTS["code"]
    for c in levels:
        before = spec_metrics()
        aggs, walls = [], []
        for _ in range(rounds):
            out = {}
            wall = _wave(prompt, mt, c, out)
            ok = [v for v in out.values() if v and v.get("ok")]
            toks = sum(v["completion_tokens"] for v in ok)
            if wall > 0 and ok:
                aggs.append(toks / wall)
                walls.append(wall)
        after = spec_metrics()
        res[f"c{c}"] = {
            "agg_tok_s": stats(aggs),
            "wall_s": stats(walls),
            "spec": spec_delta(before, after),
        }
        print(f"  C{c} agg med={res[f'c{c}']['agg_tok_s'].get('median')} "
              f"peak={res[f'c{c}']['agg_tok_s'].get('peak')} "
              f"wall={res[f'c{c}']['wall_s'].get('median')}s", flush=True)
    return res


def suite_longctx(ctx_tokens=32768, levels=(1, 2)):
    """The regime issue #14 says short-prompt sweeps cannot see: long prompts,
    2 sequences in decode at once. Reported collapse: ~4 tok/s aggregate."""
    res = {}
    filler = " ".join(f"tok{i}" for i in range(ctx_tokens))
    prompt = (f"Below is a log corpus.\n{filler}\n\n"
              "Summarize what kind of data this is in about 150 words.")
    for c in levels:
        out = {}
        wall = _wave(prompt, 250, c, out)
        ok = [v for v in out.values() if v and v.get("ok")]
        toks = sum(v["completion_tokens"] for v in ok)
        res[f"c{c}"] = {
            "agg_tok_s": round(toks / wall, 2) if wall and ok else None,
            "per_stream_tok_s": round(
                sum(v["decode_tok_s"] for v in ok) / len(ok), 2) if ok else None,
            "wall_s": round(wall, 1),
            "prompt_tokens": ok[0]["prompt_tokens"] if ok else None,
            "fails": c - len(ok),
        }
        print(f"  longctx C{c} agg={res[f'c{c}']['agg_tok_s']} "
              f"per_stream={res[f'c{c}']['per_stream_tok_s']} "
              f"prompt_tokens={res[f'c{c}']['prompt_tokens']} "
              f"fails={res[f'c{c}']['fails']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="config name under test")
    ap.add_argument("--suite", default="single,sweep",
                    help="comma list: single,prefill,sweep,longctx,all")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--levels", default="1,2,3,4,5,6")
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "results"))
    a = ap.parse_args()

    suites = {s.strip() for s in a.suite.split(",")}
    if "all" in suites:
        suites = {"single", "prefill", "sweep", "longctx"}
    os.makedirs(a.outdir, exist_ok=True)

    print(f"== {a.label} ==", flush=True)
    print("  warming up...", flush=True)
    post_stream("hello", 8)

    out = {"label": a.label, "started": time.strftime("%Y-%m-%d %H:%M:%S"), "url": URL}
    if "single" in suites:
        print(" [single]", flush=True); out["single"] = suite_single(a.reps)
    if "prefill" in suites:
        print(" [prefill]", flush=True); out["prefill"] = suite_prefill()
    if "sweep" in suites:
        print(" [sweep]", flush=True)
        out["sweep"] = suite_sweep([int(x) for x in a.levels.split(",")], a.rounds)
    if "longctx" in suites:
        print(" [longctx]", flush=True); out["longctx"] = suite_longctx()

    path = os.path.join(a.outdir, f"{a.label}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  -> {path}", flush=True)


if __name__ == "__main__":
    main()
