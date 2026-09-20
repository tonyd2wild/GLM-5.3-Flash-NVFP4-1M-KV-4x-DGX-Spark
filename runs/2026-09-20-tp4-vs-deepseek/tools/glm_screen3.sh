#!/bin/bash
# glm_screen3.sh <label> [reps] [levels] [prefill]  (root on Reddie): the DeepSeek bench run N times on the
# serving GLM lane, then per-cell MEDIANS. Needed because single-run category cells on this lane have a
# +/-25% run-to-run band (measured: g00 vs g09-rep7, identical config).
LBL=${1:?label}; REPS=${2:-3}; LV=${3:-1,3,6}; PF=${4:-8000,32000}
export BENCH_MODEL=glm-5.3-flash
O=/var/tmp/boot-results/glm53/$LBL; mkdir -p $O
echo "=== glm_screen3 $LBL reps=$REPS $(date -u +%T)"
docker logs vllm_glm53 2>&1 | grep -aE "GPU KV cache size|Maximum concurrency|Graph capturing|RoCEnante|non-default args" | sed -E 's/^.*\] //' > $O/kv-context.txt
docker inspect vllm_glm53 --format '{{json .Args}}' > $O/args.json
python3 /root/sr_quality.py $O > $O/quality.txt 2>&1; cat $O/quality.txt
python3 /root/idletest.py > $O/idletest.txt 2>&1; grep -aE "tok/s after first" $O/idletest.txt | cut -c1-95
# discard one warm-up pass, then REPS measured passes
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $LBL-warm --out $O/warm --levels 1 --prefill "" > $O/warm.txt 2>&1 || true
for i in $(seq 1 $REPS); do
  python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $LBL-r$i --out $O --levels $LV --prefill $PF --notes "GLM TP4 rep $i" > $O/bench-r$i.txt 2>&1
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
cells={}; pref={}
for d in runs:
    for b in d["batches"]: cells.setdefault((b["category"],b["c"]),[]).append(b)
    for p in (d.get("prefill") or []): pref.setdefault(p["target"],[]).append(p["prefill_tok_s"])
med=lambda v: st.median(v)
cats=["coding","json","math","prose","format","ceiling_count","reasoning","summary","narrative"]
print(f"  MEDIAN of {len(runs)} reps")
print("  C1 per-stream: " + " ".join(f"{c[:5]} {med([b['per_stream_tok_s'] for b in cells[(c,1)]]):.1f}" for c in cats if (c,1) in cells))
for C in sorted({c for _,c in cells}):
    agg=[med([b["agg_tok_s"] for b in cells[(cat,C)]]) for cat in cats if cat!="ceiling_count" and (cat,C) in cells]
    print(f"  C{C} aggregate mean-of-medians {st.mean(agg):.1f}")
print("  prefill: " + " ".join(f"{t} {med(v):.0f}" for t,v in sorted(pref.items())))
print("  spread per C1 cell (max/min): " + " ".join(f"{c[:5]} {max(b['per_stream_tok_s'] for b in cells[(c,1)])/max(0.01,min(b['per_stream_tok_s'] for b in cells[(c,1)])):.2f}x" for c in cats if (c,1) in cells))
PY
echo "=== glm_screen3 $LBL done $(date -u +%T)"
