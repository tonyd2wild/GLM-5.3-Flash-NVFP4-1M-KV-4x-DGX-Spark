#!/usr/bin/env python3
# mktable.py <glm_dir> <glm_label> <ds_dir> <ds_label> [reps]
# Emits the markdown comparison tables from the 3-rep median JSONs of both lanes.
import json,sys,os,statistics as st
def load(d,l,reps):
    runs=[]
    for i in range(1,reps+1):
        p=f"{d}/bench-{l}-r{i}.json"
        if os.path.exists(p):
            try: runs.append(json.load(open(p)))
            except Exception: pass
    return runs
def digest(runs):
    cells={}; pref1={}; prefm={}
    for n,d in enumerate(runs):
        for b in d["batches"]: cells.setdefault((b["category"],b["c"]),[]).append(b)
        for p in (d.get("prefill") or []):
            prefm.setdefault(p["target"],[]).append(p["prefill_tok_s"])
            if n==0: pref1[p["target"]]=(p["prefill_tok_s"],p.get("ttft_s"),p.get("prompt_tokens"))
    ps={k:st.median([b["per_stream_tok_s"] for b in v]) for k,v in cells.items()}
    cps={}
    for k,v in cells.items():
        vals=[]
        for b in v:
            rq=b.get("requests") or []
            ch=sum(r.get("chars",0) for r in rq); tk=sum(r.get("completion_tokens",0) for r in rq)
            if tk: vals.append(b["per_stream_tok_s"]*ch/tk)
        if vals: cps[k]=st.median(vals)
    sp={k:(max(b["per_stream_tok_s"] for b in v)/max(.01,min(b["per_stream_tok_s"] for b in v))) for k,v in cells.items()}
    cats=[c for c in ["coding","json","math","prose","format","ceiling_count","reasoning","summary","narrative"]]
    agg={}
    for C in sorted({c for _,c in cells}):
        vals=[st.median([b["agg_tok_s"] for b in cells[(cat,C)]]) for cat in cats if cat!="ceiling_count" and (cat,C) in cells]
        if vals: agg[C]=st.mean(vals)
    return ps,agg,pref1,prefm,sp,len(runs),cps
gd,gl,dd,dl=sys.argv[1:5]; R=int(sys.argv[5]) if len(sys.argv)>5 else 3
G=digest(load(gd,gl,R)); D=digest(load(dd,dl,R))
NAMES=[("coding","code"),("json","JSON"),("math","math"),("prose","prose"),("ceiling_count","counting"),("format","structure (tables)"),("reasoning","reasoning"),("summary","summary"),("narrative","narrative")]
print(f"GLM reps={G[5]}  DS reps={D[5]}\n")
print("| C1 per-stream tok/s | DeepSeek-V4.1-Flash | GLM-5.3-Flash | GLM vs DS |")
print("|---|---|---|---|")
for k,n in NAMES:
    g=G[0].get((k,1)); d=D[0].get((k,1))
    if g is None or d is None: continue
    print(f"| {n} | {d:.1f} | {g:.1f} | {(g/d-1)*100:+.0f}% |")
print()
print("| aggregate tok/s | DeepSeek | GLM | GLM vs DS |")
print("|---|---|---|---|")
for C in sorted(set(G[1])&set(D[1])):
    print(f"| C{C} | {D[1][C]:.1f} | {G[1][C]:.1f} | {(G[1][C]/D[1][C]-1)*100:+.0f}% |")
print()
print("| cold prefill, rep 1 only | DeepSeek | GLM | tok/s delta | wall-clock TTFT delta |")
print("|---|---|---|---|---|")
for t in sorted(set(G[2])&set(D[2])):
    g=G[2][t]; d=D[2][t]
    print(f"| ~{t} tok target | {d[0]:.0f} tok/s ({d[2]} tok, ttft {d[1]:.1f} s) | {g[0]:.0f} tok/s ({g[2]} tok, ttft {g[1]:.1f} s) | {(g[0]/d[0]-1)*100:+.0f}% | {(g[1]/d[1]-1)*100:+.0f}% |")
print()
print()
print("| C1 chars/s (tokenizer-neutral) | DeepSeek | GLM | GLM vs DS |")
print("|---|---|---|---|")
for k,n in NAMES:
    g=G[6].get((k,1)); d=D[6].get((k,1))
    if g is None or d is None: continue
    print(f"| {n} | {d:.0f} | {g:.0f} | {(g/d-1)*100:+.0f}% |")
print()
print("spread (max/min over reps), GLM: " + " ".join(f"{n} {G[4][(k,1)]:.2f}x" for k,n in NAMES if (k,1) in G[4]))
print("spread (max/min over reps), DS:  " + " ".join(f"{n} {D[4][(k,1)]:.2f}x" for k,n in NAMES if (k,1) in D[4]))
