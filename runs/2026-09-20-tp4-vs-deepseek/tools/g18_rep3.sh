#!/bin/bash
# One more measured pass on the NVFP4 lane: the headline claim rests on medians, and 2 reps is thinner than the
# 3-rep protocol used on the bf16 lane. Runs after the needle so the two do not overlap.
D=/var/tmp/boot-results/glm53; O=$D/g18-nvfp4attn-patched
for i in $(seq 1 40); do pgrep -f "^bash /root/g18_needle.sh" >/dev/null || break; sleep 15; done
export BENCH_MODEL=glm-5.3-flash
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label g18-nvfp4attn-patched-r3 \
  --out $O --levels 1,3 --prefill "" --notes "NVFP4 attn rep 3" > $O/bench-r3.txt 2>&1
python3 - "$O" <<'PY'
import json,sys,statistics as st
O=sys.argv[1]; runs=[]
for i in (1,2,3):
    try: runs.append(json.load(open("%s/bench-g18-nvfp4attn-patched-r%d.json"%(O,i))))
    except Exception: pass
cells={}
for d in runs:
    for b in d["batches"]: cells.setdefault((b["category"],b["c"]),[]).append(b["per_stream_tok_s"])
cats=["coding","json","math","prose","format","ceiling_count","reasoning","summary","narrative"]
print("  MEDIAN of %d reps"%len(runs))
print("  C1 per-stream: "+" ".join("%s %.1f"%(c[:5],st.median(cells[(c,1)])) for c in cats if (c,1) in cells))
print("  spread: "+" ".join("%s %.2fx"%(c[:5],max(cells[(c,1)])/max(.01,min(cells[(c,1)]))) for c in cats if (c,1) in cells))
PY
