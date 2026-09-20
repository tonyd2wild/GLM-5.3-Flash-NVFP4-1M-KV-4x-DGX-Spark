#!/bin/bash
# Full characterisation of the SERVING 500K NVFP4 lane: the numbers quoted so far came from the 1M lane at
# C1/C3 only. This covers C1-C6 with prefill, then C8/C12/C16, then C24/C32.
# One discarded warm-up pass, three measured passes, per-cell medians and spread. Only rep 1's prefill is cold.
G=/var/tmp/boot-results/glm53; L=glm-500k-sweep; O=$G/$L; mkdir -p $O
export BENCH_MODEL=glm-5.3-flash
say(){ echo "$(date -u +%T) SWEEP $*" >> $G/status.txt; }
say "start (C1-C6 + prefill, then C8-C32)"
docker logs vllm_glm53 2>&1 | grep -aE "GPU KV cache size|Maximum concurrency|quant_algo|MarlinNvFp4|non-default args" | sed -E 's/^.*\] //' > $O/kv-context.txt
python3 /root/sr_quality.py $O > $O/quality.txt 2>&1; say "quality $(grep -aoE 'PASS|FAIL' $O/quality.txt | head -1)"
python3 /root/idletest.py > $O/idletest.txt 2>&1
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $L-warm --out $O/warm --levels 1 --prefill "" > $O/warm.txt 2>&1 || true
say "warm-up done, 3 measured passes C1-C6"
for i in 1 2 3; do
  python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $L-r$i \
    --out $O --levels 1,2,3,4,5,6 --prefill 2000,8000,32000,64000 --notes "GLM NVFP4 500K sweep rep $i" > $O/bench-r$i.txt 2>&1
  say "rep $i done"
done
python3 - "$O" "$L" <<'PY'
import json,sys,statistics as st
O,L=sys.argv[1],sys.argv[2]
runs=[]
for i in (1,2,3):
    try: runs.append(json.load(open("%s/bench-%s-r%d.json"%(O,L,i))))
    except Exception as e: print("missing rep",i)
cells={}; tt={}; pref1={}
for n,d in enumerate(runs):
    for b in d["batches"]:
        cells.setdefault((b["category"],b["c"]),[]).append(b)
        tt.setdefault(b["c"],[]).append(b["ttft_mean_s"])
    for p in (d.get("prefill") or []):
        if n==0: pref1[p["target"]]=(p["prefill_tok_s"],p["ttft_s"],p["prompt_tokens"])
cats=["coding","json","math","prose","format","ceiling_count","reasoning","summary","narrative"]
print("MEDIANS of %d reps"%len(runs))
print("C1 per-stream: "+" ".join("%s %.1f"%(c[:5],st.median([b["per_stream_tok_s"] for b in cells[(c,1)]])) for c in cats if (c,1) in cells))
for C in sorted({c for _,c in cells}):
    agg=[st.median([b["agg_tok_s"] for b in cells[(cat,C)]]) for cat in cats if cat!="ceiling_count" and (cat,C) in cells]
    ps=[st.median([b["per_stream_tok_s"] for b in cells[(cat,C)]]) for cat in cats if cat!="ceiling_count" and (cat,C) in cells]
    print("C%d: aggregate %.1f | per-stream %.1f | TTFT %.3f s"%(C,st.mean(agg),st.mean(ps),st.median(tt[C])))
print("cold prefill (rep 1): "+" | ".join("%d->%.0f tok/s (%d tok, ttft %.1fs)"%(t,v[0],v[2],v[1]) for t,v in sorted(pref1.items())))
print("C1 spread: "+" ".join("%s %.2fx"%(c[:5],max(b["per_stream_tok_s"] for b in cells[(c,1)])/max(.01,min(b["per_stream_tok_s"] for b in cells[(c,1)]))) for c in cats if (c,1) in cells))
PY
python3 - "$O" "$L" >> $G/status.txt <<'PY'
import json,sys,statistics as st
O,L=sys.argv[1],sys.argv[2]
runs=[]
for i in (1,2,3):
    try: runs.append(json.load(open("%s/bench-%s-r%d.json"%(O,L,i))))
    except Exception: pass
cells={}
for d in runs:
    for b in d["batches"]: cells.setdefault((b["category"],b["c"]),[]).append(b)
cats=["coding","json","math","prose","format","ceiling_count","reasoning","summary","narrative"]
print("  SWEEP C1 per-stream: "+" ".join("%s %.1f"%(c[:5],st.median([b["per_stream_tok_s"] for b in cells[(c,1)]])) for c in cats if (c,1) in cells))
for C in sorted({c for _,c in cells}):
    agg=[st.median([b["agg_tok_s"] for b in cells[(cat,C)]]) for cat in cats if cat!="ceiling_count" and (cat,C) in cells]
    print("  SWEEP C%d aggregate %.1f"%(C,st.mean(agg)))
PY
say "C1-C6 measured, starting C8/C12/C16"
bash /root/glm_hiconc.sh $L >> $O/hiconc.log 2>&1
grep -aE "^  C" $O/hiconc.log | tail -3 >> $G/status.txt
say "C8-C16 done, starting C24/C32"
bash /root/glm_hiconc2.sh $L >> $O/hiconc2.log 2>&1
grep -aE "^  C" $O/hiconc2.log | tail -2 >> $G/status.txt
bash /root/specacc.sh $O/specacc.txt >> $O/specacc.log 2>&1
grep -aE "acceptance|tokens per step" $O/specacc.txt | tail -2 >> $G/status.txt
say "ALL DONE"
