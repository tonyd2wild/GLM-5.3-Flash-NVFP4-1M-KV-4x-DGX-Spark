#!/bin/bash
# glm_screen.sh <label> [levels] [prefill]  (root on Reddie): the DeepSeek SCREEN protocol pointed at
# the GLM endpoint, so every number is directly comparable to the DeepSeek speed-run tables (same
# prompts, same harness, temperature 0, streaming, after-first-token decode).
LBL=${1:?label}; LV=${2:-1,3,6}; PF=${3:-8000,32000}
export BENCH_MODEL=glm-5.3-flash
O=/var/tmp/boot-results/glm53/$LBL; mkdir -p $O
echo "=== glm_screen $LBL $(date -u +%T) levels=$LV prefill=$PF"
docker logs vllm_glm53 2>&1 | grep -aE "GPU KV cache size|Maximum concurrency|Graph capturing finished|RoCEnante|non-default args|Loading weights took" | sed -E 's/^.*\] //' > $O/kv-context.txt
docker inspect vllm_glm53 --format '{{json .Args}}' > $O/args.json
docker inspect vllm_glm53 --format '{{range .Config.Env}}{{println .}}{{end}}' | grep -E "NCCL|VLLM|PYTORCH|B12X|TORCH" > $O/env.txt
python3 /root/sr_quality.py $O > $O/quality.txt 2>&1; cat $O/quality.txt
python3 /root/idletest.py > $O/idletest.txt 2>&1; grep -aE "tok/s after first" $O/idletest.txt | cut -c1-95
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $LBL-warm --out $O/warm --levels 1 --prefill "" > $O/warm.txt 2>&1 || true
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $LBL --out $O --levels $LV --prefill $PF --notes "GLM-5.3 TP4 speed run 2026-09-20 $LBL" > $O/bench.txt 2>&1
echo "bench exit $? $(date -u +%T)"
grep -aE "prefill +[0-9]+:" $O/bench.txt
python3 - "$O" "$LBL" <<'PY'
import json,sys,statistics as st
O,L=sys.argv[1],sys.argv[2]
d=json.load(open(f"{O}/bench-{L}.json"))
a={};p={}
for b in d["batches"]:
    if b["category"]=="ceiling_count": continue
    a.setdefault(b["c"],[]).append(b["agg_tok_s"]); p.setdefault(b["c"],[]).append(b["per_stream_tok_s"])
print("  agg mean:", " ".join(f"C{c} {st.mean(v):.1f}" for c,v in sorted(a.items())))
print("  per-stream mean:", " ".join(f"C{c} {st.mean(v):.1f}" for c,v in sorted(p.items())))
m={(b["category"],b["c"]):b for b in d["batches"]}
cats=["coding","json","math","prose","format","ceiling_count","reasoning","summary","narrative"]
print("  C1 per-stream:", " ".join(f"{c[:5]} {m[(c,1)]['per_stream_tok_s']:.1f}" for c in cats if (c,1) in m))
PY
echo "=== glm_screen $LBL done $(date -u +%T)"
