#!/bin/bash
# glm_hiconc2.sh <label>: GLM-only concurrency levels that the DeepSeek lane cannot reach at all
# (it serves max-num-seqs 16; GLM serves 64). One pass, coding + counting + json only, to keep it cheap.
LBL=${1:?label}
export BENCH_MODEL=glm-5.3-flash
O=/var/tmp/boot-results/glm53/$LBL-hiconc2; mkdir -p $O
echo "=== hiconc2 $LBL $(date -u +%T)"
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $LBL-hc2 \
  --out $O --levels 24,32 --prefill "" --notes "GLM TP4 concurrency beyond the DeepSeek lane's ceiling" > $O/hc2.txt 2>&1
python3 - "$O" "$LBL" <<'PY'
import json,sys
O,L=sys.argv[1],sys.argv[2]
d=json.load(open(f"{O}/bench-{L}-hc2.json"))
rows={}
for b in d["batches"]: rows.setdefault(b["c"],[]).append((b["category"],b["agg_tok_s"],b["per_stream_tok_s"]))
for C in sorted(rows):
    tot=[v for k,v,_ in rows[C] if k!="ceiling_count"]
    print(f"  C{C}: mean agg {sum(tot)/len(tot):.1f} | " + " ".join(f"{k[:5]} {v:.0f} ({p:.1f}/stream)" for k,v,p in sorted(rows[C],key=lambda r:-r[1])))
PY
echo "=== hiconc2 $LBL done $(date -u +%T)"
