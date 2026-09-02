#!/usr/bin/env python3
"""compare.py — table every results/*.json against the baseline.

Usage: python3 compare.py [--baseline 00-baseline-eager-k7-seqs6] [--md]
"""
import argparse, glob, json, os

def pct(new, base):
    if base in (None, 0) or new is None:
        return ""
    d = (new - base) / base * 100
    return f"{d:+.1f}%"

def get(d, *path, default=None):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="00-baseline-eager-k7-seqs6")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "results"))
    ap.add_argument("--md", action="store_true", help="emit markdown tables")
    a = ap.parse_args()

    runs = {}
    for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
        runs[os.path.basename(f)[:-5]] = json.load(open(f))
    if not runs:
        print("no results yet"); return
    base = runs.get(a.baseline)

    sep = "|" if a.md else " "
    def row(cells, widths):
        if a.md:
            return "| " + " | ".join(str(c) for c in cells) + " |"
        return "  ".join(str(c).ljust(w) for c, w in zip(cells, widths))

    # ---- single-stream decode (the headline + PEAK) ----
    print("\n### Single-stream decode, tok/s (median, temp 0)\n")
    hdr = ["config", "count100 PEAK", "code", "prose", "accept(code)", "accept_len(code)"]
    w = [34, 15, 15, 15, 13, 16]
    print(row(hdr, w))
    if a.md: print("|" + "|".join("---" for _ in hdr) + "|")
    for name, r in runs.items():
        cells = [name]
        for p in ("count100", "code", "prose"):
            v = get(r, "single", p, "decode_tok_s", "median")
            b = get(base, "single", p, "decode_tok_s", "median") if base else None
            cells.append(f"{v}" + (f" ({pct(v,b)})" if base and name != a.baseline and v else ""))
        cells.append(get(r, "single", "code", "spec", "accept_ratio", default="-"))
        cells.append(get(r, "single", "code", "spec", "mean_accept_len", default="-"))
        print(row(cells, w))

    # ---- aggregate sweep ----
    print("\n### Aggregate throughput by concurrency, tok/s (median)\n")
    levels = sorted({k for r in runs.values() for k in get(r, "sweep", default={})},
                    key=lambda s: int(s[1:]))
    hdr = ["config"] + levels + ["best"]
    w = [34] + [11] * len(levels) + [11]
    print(row(hdr, w))
    if a.md: print("|" + "|".join("---" for _ in hdr) + "|")
    for name, r in runs.items():
        vals, cells = [], [name]
        for lv in levels:
            v = get(r, "sweep", lv, "agg_tok_s", "median")
            b = get(base, "sweep", lv, "agg_tok_s", "median") if base else None
            vals.append(v)
            cells.append(f"{v}" + (f" ({pct(v,b)})" if base and name != a.baseline and v else "")
                         if v is not None else "-")
        good = [v for v in vals if v is not None]
        bb = [get(base, "sweep", lv, "agg_tok_s", "median") for lv in levels] if base else []
        bb = [v for v in bb if v is not None]
        cells.append(f"{max(good)}" + (f" ({pct(max(good), max(bb))})"
                     if bb and name != a.baseline else "") if good else "-")
        print(row(cells, w))

    # ---- prefill ----
    print("\n### Prefill, tok/s\n")
    sizes = sorted({k for r in runs.values() for k in get(r, "prefill", default={})}, key=int)
    hdr = ["config"] + [f"~{s}w" for s in sizes]
    w = [34] + [16] * len(sizes)
    print(row(hdr, w))
    if a.md: print("|" + "|".join("---" for _ in hdr) + "|")
    for name, r in runs.items():
        cells = [name]
        for s in sizes:
            v = get(r, "prefill", s, "prefill_tok_s")
            b = get(base, "prefill", s, "prefill_tok_s") if base else None
            cells.append(f"{v}" + (f" ({pct(v,b)})" if base and name != a.baseline and v else "")
                         if v is not None else "-")
        print(row(cells, w))

    # ---- long context ----
    print("\n### Long context (~114K prompt), per-stream decode tok/s\n")
    hdr = ["config", "C1 per-stream", "C2 per-stream", "C1 wall", "C2 wall"]
    w = [34, 15, 15, 10, 10]
    print(row(hdr, w))
    if a.md: print("|" + "|".join("---" for _ in hdr) + "|")
    for name, r in runs.items():
        if not get(r, "longctx"): continue
        print(row([name,
                   get(r, "longctx", "c1", "per_stream_tok_s", default="-"),
                   get(r, "longctx", "c2", "per_stream_tok_s", default="-"),
                   get(r, "longctx", "c1", "wall_s", default="-"),
                   get(r, "longctx", "c2", "wall_s", default="-")], w))
    print()

if __name__ == "__main__":
    main()
