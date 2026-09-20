#!/bin/bash
# glm_hiconc.sh <label>  (root on Reddie): high-concurrency sweep on the serving GLM lane, 2 reps, medians.
# GLM serves max-num-seqs 64 vs DeepSeek's 16, so this is the axis where GLM should win.
LBL=${1:?label}
export BENCH_MODEL=glm-5.3-flash
O=/var/tmp/boot-results/glm53/$LBL-hiconc; mkdir -p $O
echo "=== hiconc $LBL $(date -u +%T)"
for i in 1 2; do
  python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $LBL-hc$i \
    --out $O --levels 8,12,16 --prefill "" --notes "GLM TP4 high concurrency rep $i" > $O/hc$i.txt 2>&1
  echo "  rep $i done $(date -u +%T)"
done
python3 - "$O" "$LBL" <<'PY'
import json,sys,statistics as st
O,L=sys.argv[1],sys.argv[2]
runs=[]
for i in (1,2):
    try: runs.append(json.load(open(f"{O}/bench-{L}-hc{i}.json")))
    except Exception as e: print("missing",i,e)
cells={}
for d in runs:
    for b in d["batches"]: cells.setdefault((b["category"],b["c"]),[]).append(b)
for C in sorted({c for _,c in cells}):
    rows=[(cat,st.median([b["agg_tok_s"] for b in v])) for (cat,c),v in cells.items() if c==C]
    rows.sort(key=lambda r:-r[1])
    tot=st.mean([v for _,v in rows if _!="ceiling_count"])
    print(f"  C{C}: mean-of-medians {tot:.1f} | " + " ".join(f"{c[:5]} {v:.0f}" for c,v in rows[:4]))
PY
echo "=== hiconc $LBL done $(date -u +%T)"
