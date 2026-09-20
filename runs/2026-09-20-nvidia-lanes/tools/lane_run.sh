#!/bin/bash
# lane_run.sh <label> <MODEL_DIR>: settle, boot, then the full characterisation the repo needs:
# quality gate, idle probe, C1-C6 with cold prefill (3 reps, medians), C8/C12/C16, C24/C32, acceptance.
LBL=${1:?label}; MD=${2:?model dir}
G=/var/tmp/boot-results/glm53; O=$G/$LBL; mkdir -p $O
export MODEL_DIR=$MD NVFP4_PATCH=1 MNBT=8192 SPEC_K=7 MAXLEN=500000
export VLLM_EXTRA="--limit-mm-per-prompt {\"image\":4,\"video\":0}"
export BENCH_MODEL=glm-5.3-flash
say(){ echo "$(date -u +%T) $LBL $*" >> $G/status.txt; }
say "SETTLE"; bash /root/glm_settle.sh >> $O/settle.log 2>&1
say "BOOT ($MD)"
bash /root/glm_boot.sh $LBL || { say "BOOT FAILED"; exit 2; }
say "SERVING"
docker logs vllm_glm53 2>&1 | grep -aE "GPU KV cache size|Maximum concurrency|quant_algo|MarlinNvFp4|Model loading took" | sed -E 's/^.*\] //' | tail -5 > $O/kv-context.txt
cat $O/kv-context.txt >> $G/status.txt
python3 /root/sr_quality.py $O > $O/quality.txt 2>&1; say "QUALITY $(grep -aoE 'PASS|FAIL' $O/quality.txt | head -1)"
python3 /root/idletest.py > $O/idletest.txt 2>&1
grep -aE "tok/s after first" $O/idletest.txt | cut -c1-95 >> $G/status.txt
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $LBL-warm --out $O/warm --levels 1 --prefill "" > $O/warm.txt 2>&1 || true
for i in 1 2 3; do
  python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label $LBL-r$i \
    --out $O --levels 1,2,3,4,5,6 --prefill 2000,8000,32000,64000 --notes "$LBL rep $i" > $O/bench-r$i.txt 2>&1
  say "rep $i done"
done
say "C1-C6 done, concurrency next"
bash /root/glm_hiconc.sh $LBL >> $O/hiconc.log 2>&1; grep -aE "^  C" $O/hiconc.log | tail -3 >> $G/status.txt
bash /root/glm_hiconc2.sh $LBL >> $O/hiconc2.log 2>&1; grep -aE "^  C" $O/hiconc2.log | tail -2 >> $G/status.txt
bash /root/specacc.sh $O/specacc.txt >> $O/specacc.log 2>&1
grep -aE "acceptance|tokens per step" $O/specacc.txt | tail -2 >> $G/status.txt
say "ALL DONE"
