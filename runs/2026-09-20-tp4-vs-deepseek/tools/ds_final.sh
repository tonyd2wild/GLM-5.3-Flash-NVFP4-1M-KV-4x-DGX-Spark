#!/bin/bash
# ds_final.sh <label>: restore the DeepSeek-V4.1-Flash speed-run-2 final lane, then measure it the SAME way
# GLM was measured (3 reps + medians, then the high-concurrency sweep). This lane STAYS SERVING afterwards.
LBL=${1:-dsrestore}
S=/var/tmp/boot-results/speedrun/s2-status.txt
echo "$(date -u +%T) DS-RESTORE-BOOT $LBL" >> $S
bash /root/sr_boot.sh sr2-e12-final500k-s16-go.sh $LBL; rc=$?
if [ $rc != 0 ]; then echo "$(date -u +%T) DS-RESTORE-FAILED rc=$rc" >> $S; exit $rc; fi
echo "$(date -u +%T) DS-SERVING $LBL" >> $S
bash /root/ds_screen3.sh $LBL 3 1,3,6 2000,8000,32000,64000 > /var/tmp/boot-results/speedrun/$LBL-screen3.log 2>&1
echo "$(date -u +%T) DS-MEASURED $LBL" >> $S
bash /root/ds_hiconc.sh $LBL > /var/tmp/boot-results/speedrun/$LBL-hiconc.log 2>&1
echo "$(date -u +%T) DS-HICONC-DONE $LBL" >> $S
curl -s -m 20 -X POST http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","messages":[{"role":"user","content":"ok"}],"max_tokens":4}' \
  | head -c 200 >> $S; echo "" >> $S
echo "$(date -u +%T) DS-LIVE-CONFIRMED" >> $S
