import json,statistics as st
O="/var/tmp/boot-results/glm53/glm-500k-sweep"
runs=[json.load(open("%s/bench-glm-500k-sweep-r%d.json"%(O,i))) for i in (1,2,3)]
cells={}
for d in runs:
    for b in d["batches"]: cells.setdefault((b["category"],b["c"]),[]).append(b)
DS={"coding":90.7,"json":75.8,"math":87.5,"prose":39.0,"format":98.6,"ceiling_count":113.0,"reasoning":75.1,"summary":41.1,"narrative":31.4}
NAME={"coding":"code","json":"JSON","math":"math","prose":"prose","format":"structure","ceiling_count":"counting","reasoning":"reasoning","summary":"summary","narrative":"narrative"}
print("C1 per-stream, medians of 3, vs DeepSeek:")
w=0
for k in ("format","summary","math","prose","ceiling_count","coding","narrative","json","reasoning"):
    v=[b["per_stream_tok_s"] for b in cells[(k,1)]]
    m=st.median(v); sp=max(v)/max(.01,min(v)); d=(m/DS[k]-1)*100
    if d>0: w+=1
    print("  %-10s %6.1f  vs %6.1f  %+6.1f%%   spread %.2fx  %s"%(NAME[k],m,DS[k],d,sp,"WIN" if d>0 else ""))
print("  -> GLM wins %d of 9"%w)
DSagg={1:61.3,2:102.3,3:128.9,4:153.4,5:174.3,6:189.3}
print("\naggregate by concurrency:")
cats=[c for c in DS if c!="ceiling_count"]
for C in sorted({c for _,c in cells}):
    agg=st.mean([st.median([b["agg_tok_s"] for b in cells[(c,C)]]) for c in cats if (c,C) in cells])
    tt=st.median([b["ttft_mean_s"] for b in cells[("coding",C)]])
    print("  C%d  %6.1f  vs %6.1f  %+5.0f%%   ttft %.3f s"%(C,agg,DSagg[C],(agg/DSagg[C]-1)*100,tt))
