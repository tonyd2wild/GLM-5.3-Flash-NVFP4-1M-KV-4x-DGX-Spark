#!/bin/bash
# ds_screen3.sh <label> [reps] [levels] [prefill]  (root on Reddie): the SAME 3-rep median protocol
# glm_screen3.sh runs, pointed at the DeepSeek lane, so the GLM-vs-DeepSeek comparison is medians
# against medians instead of medians against single runs.
LBL=${1:?label}; REPS=${2:-3}; LV=${3:-1,3,6}; PF=${4:-2000,8000,32000,64000}
O=/var/tmp/boot-results/speedrun/$LBL; mkdir -p $O
echo "=== ds_screen3 $LBL reps=$REPS $(date -u +%T)"
docker logs vllm_dsv41 2>&1 | grep -aE "GPU KV cache size|Maximum concurrency|Graph capturing finished|non-default args" | sed -E 's/^.*\] //' > $O/kv-context.txt
docker inspect vllm_dsv41 --format '{{json .Args}}' > $O/args.json
python3 /root/sr_quality.py $O > $O/quality.txt 2>&1; cat $O/quality.txt
python3 /root/idletest.py > $O/idletest.txt 2>&1; grep -aE "tok/s after first" $O/idletest.txt | cut -c1-95
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label $LBL-warm --out $O/warm --levels 1 --prefill "" > $O/warm.txt 2>&1 || true
for i in $(seq 1 $REPS); do
  python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model deepseek-v4.1-flash --label $LBL-r$i --out $O --levels $LV --prefill $PF --notes "DS4.1F TP4 rep $i" > $O/bench-r$i.txt 2>&1
  echo "  rep $i done $(date -u +%T)"
done
python3 - "$O" "$LBL" "$REPS" <<'PY'
import json,sys,statistics as st
O,L,R=sys.argv[1],sys.argv[2],int(sys.argv[3])
runs=[]
for i in range(1,R+1):
    try: runs.append(json.load(open(f"{O}/bench-{L}-r{i}.json")))
    except Exception as e: print("missing rep",i,e)
if not runs: raise SystemExit
cells={}; pref={}; pref1={}
for n,d in enumerate(runs):
    for b in d["batches"]: cells.setdefault((b["category"],b["c"]),[]).append(b)
    for p in (d.get("prefill") or []):
        pref.setdefault(p["target"],[]).append(p["prefill_tok_s"])
        if n==0: pref1[p["target"]]=(p["prefill_tok_s"],p.get("ttft_s"),p.get("prompt_tok"))
med=lambda v: st.median(v)
cats=["coding","json","math","prose","format","ceiling_count","reasoning","summary","narrative"]
print(f"  MEDIAN of {len(runs)} reps")
print("  C1 per-stream: " + " ".join(f"{c[:5]} {med([b['per_stream_tok_s'] for b in cells[(c,1)]]):.1f}" for c in cats if (c,1) in cells))
for C in sorted({c for _,c in cells}):
    agg=[med([b["agg_tok_s"] for b in cells[(cat,C)]]) for cat in cats if cat!="ceiling_count" and (cat,C) in cells]
    print(f"  C{C} aggregate mean-of-medians {st.mean(agg):.1f}")
print("  prefill median (CACHE-CONTAMINATED past rep1): " + " ".join(f"{t} {med(v):.0f}" for t,v in sorted(pref.items())))
print("  prefill rep1 COLD: " + " ".join(f"{t}->{v[0]:.0f} (ttft {v[1]:.1f}s, {v[2]} tok)" for t,v in sorted(pref1.items())))
print("  spread per C1 cell (max/min): " + " ".join(f"{c[:5]} {max(b['per_stream_tok_s'] for b in cells[(c,1)])/max(0.01,min(b['per_stream_tok_s'] for b in cells[(c,1)])):.2f}x" for c in cats if (c,1) in cells))
PY
echo "=== ds_screen3 $LBL done $(date -u +%T)"
