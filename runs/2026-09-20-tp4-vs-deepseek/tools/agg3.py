import json,statistics as st
O="/var/tmp/boot-results/glm53/g18-nvfp4attn-patched"
runs=[]
for i in (1,2,3):
    for pat in ("%s/bench-g18-nvfp4attn-patched-r%d.json"%(O,i),):
        try: runs.append(json.load(open(pat)))
        except Exception as e: print("miss",i)
cells={}
for d in runs:
    for b in d["batches"]: cells.setdefault((b["category"],b["c"]),[]).append(b)
cats=["coding","json","math","prose","format","reasoning","summary","narrative"]
for C in sorted({c for _,c in cells}):
    vals=[st.median([b["agg_tok_s"] for b in cells[(cat,C)]]) for cat in cats if (cat,C) in cells]
    if vals: print("C%d aggregate mean-of-medians %.1f  (n=%d reps)"%(C,st.mean(vals),len(runs)))
ps={}
for k,v in cells.items():
    if k[1]==1: ps[k[0]]=st.median([b["per_stream_tok_s"] for b in v])
chars={}
for k,v in cells.items():
    if k[1]!=1: continue
    vs=[]
    for b in v:
        rq=b.get("requests") or []
        ch=sum(r.get("chars",0) for r in rq); tk=sum(r.get("completion_tokens",0) for r in rq)
        if tk: vs.append(b["per_stream_tok_s"]*ch/tk)
    if vs: chars[k[0]]=st.median(vs)
DS={"coding":285.4,"json":226.1,"math":216.5,"prose":188.6,"format":221.5}
print("\nC1 chars/s (tokenizer-neutral) vs DeepSeek:")
for k in ("coding","json","math","prose","format"):
    if k in chars: print("  %-8s %6.1f  vs DS %6.1f  (%+.0f%%)"%(k,chars[k],DS[k],(chars[k]/DS[k]-1)*100))
