#!/usr/bin/env python3
"""Compare the lanes against the baseline already in the repo, and judge each difference against the
cell's own measured spread rather than against a fixed threshold.

A delta smaller than the spread of the cells being compared is not resolvable by this harness and is
reported as "within noise" rather than as a win or a loss.
"""
import json,os,sys,statistics as st
G="/var/tmp/boot-results/glm53"
LANES=[("baseline (LibertAI+NVFP4)","glm-500k-sweep"),("Lane A (nvidia)","nvidia-laneA"),("Lane B (nvidia+ablit)","nvidia-laneB")]
CATS=["coding","json","math","prose","format","ceiling_count","reasoning","summary","narrative"]
NAME={"coding":"code","json":"JSON","math":"math","prose":"prose","format":"structure",
      "ceiling_count":"counting","reasoning":"reasoning","summary":"summary","narrative":"narrative"}
def load(lbl):
    runs=[]
    for i in (1,2,3):
        for p in ("%s/%s/bench-%s-r%d.json"%(G,lbl,lbl,i),):
            if os.path.exists(p):
                try: runs.append(json.load(open(p)))
                except Exception: pass
    if not runs: return None
    cells={};pref1={};tt={}
    for n,d in enumerate(runs):
        for b in d["batches"]:
            cells.setdefault((b["category"],b["c"]),[]).append(b)
            tt.setdefault(b["c"],[]).append(b["ttft_mean_s"])
        for p in (d.get("prefill") or []):
            if n==0: pref1[p["target"]]=(p["prefill_tok_s"],p["ttft_s"],p["prompt_tokens"])
    ps={k:st.median([b["per_stream_tok_s"] for b in v]) for k,v in cells.items()}
    sp={k:(max(b["per_stream_tok_s"] for b in v)/max(.01,min(b["per_stream_tok_s"] for b in v))) for k,v in cells.items()}
    agg={}
    for C in sorted({c for _,c in cells}):
        vals=[st.median([b["agg_tok_s"] for b in cells[(c,C)]]) for c in CATS if c!="ceiling_count" and (c,C) in cells]
        if vals: agg[C]=st.mean(vals)
    return dict(ps=ps,sp=sp,agg=agg,pref1=pref1,tt={k:st.median(v) for k,v in tt.items()},n=len(runs))
data=[(n,load(l)) for n,l in LANES]
have=[(n,d) for n,d in data if d]
if not have: print("no data yet"); sys.exit(0)
base=have[0][1]
print("reps loaded: "+", ".join("%s=%d"%(n,d["n"]) for n,d in have))
print("\n## C1 per-stream tok/s (medians), and whether the difference is resolvable\n")
hdr="| category | "+" | ".join(n for n,_ in have)+" |"
print(hdr); print("|"+"---|"*(len(have)+1))
for c in CATS:
    row=["**%s**"%NAME[c]]
    for i,(n,d) in enumerate(have):
        v=d["ps"].get((c,1)); s=d["sp"].get((c,1),1.0)
        if v is None: row.append("-"); continue
        if i==0: row.append("%.1f (%.2fx)"%(v,s))
        else:
            b=base["ps"].get((c,1)); delta=(v/b-1)*100 if b else 0
            band=max(s,base["sp"].get((c,1),1.0))-1
            verdict="within noise" if abs(delta)/100 < band else ("**%+.0f%%**"%delta)
            row.append("%.1f (%.2fx) %s"%(v,s,verdict if verdict=="within noise" else verdict))
    print("| "+" | ".join(row)+" |")
print("\n## aggregate by concurrency, and TTFT\n")
print("| level | "+" | ".join(n for n,_ in have)+" |"); print("|"+"---|"*(len(have)+1))
for C in sorted(set().union(*[set(d["agg"]) for _,d in have])):
    row=["C%d"%C]
    for i,(n,d) in enumerate(have):
        a=d["agg"].get(C); t=d["tt"].get(C)
        row.append("-" if a is None else "%.1f (ttft %.3f)"%(a,t))
    print("| "+" | ".join(row)+" |")
print("\n## cold prefill, rep 1 only\n")
print("| target | "+" | ".join(n for n,_ in have)+" |"); print("|"+"---|"*(len(have)+1))
for t in sorted(set().union(*[set(d["pref1"]) for _,d in have])):
    row=["%d"%t]
    for n,d in have:
        v=d["pref1"].get(t)
        row.append("-" if not v else "%.0f tok/s (%d tok, ttft %.1fs)"%(v[0],v[2],v[1]))
    print("| "+" | ".join(row)+" |")
for lbl,short in (("Lane A","nvidia-laneA"),("Lane B","nvidia-laneB")):
    for f,tag in (("hiconc.log","C8-C16"),("hiconc2.log","C24-C32"),("specacc.txt","acceptance")):
        p="%s/%s/%s"%(G,short,f)
        if os.path.exists(p):
            out=[l.strip() for l in open(p) if l.strip().startswith("C") or "acceptance" in l or "tokens per step" in l]
            if out: print("\n%s %s: %s"%(lbl,tag," | ".join(out[-3:])))
