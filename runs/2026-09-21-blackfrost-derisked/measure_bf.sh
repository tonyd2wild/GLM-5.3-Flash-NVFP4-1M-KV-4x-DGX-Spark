#!/bin/bash
# Full characterisation of the ALREADY-SERVING Blackfrost lane: the measurement half of lane_run.sh, unchanged,
# so the numbers compare 1:1 with nvidia-laneA (same harness, same prompts, same protocol).
# Quality, idle probe, discarded warm-up, 3 measured reps of C1-C6 + cold prefill 2K/8K/32K/64K, C8/C12/C16,
# C24/C32, draft acceptance. One measurement at a time; nothing else may touch the endpoint meanwhile.
LBL=blackfrost-derisked-bench
G=/var/tmp/boot-results/glm53; O=$G/$LBL; mkdir -p $O
export BENCH_MODEL=glm-5.3-flash
say(){ echo "$(date -u +%T) $LBL $*" >> $G/status.txt; }
say "MEASURE START (lane already serving, booted 23:45:52)"
docker logs vllm_glm53 2>&1 | grep -aE "GPU KV cache size|Maximum concurrency|quant_algo|MarlinNvFp4|Model loading took" | sed -E 's/^.*\] //' | tail -5 > $O/kv-context.txt
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
