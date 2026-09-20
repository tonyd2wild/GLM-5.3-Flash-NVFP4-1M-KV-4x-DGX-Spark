#!/bin/bash
# glm_final.sh <label> [KNOB=val ...]  (root on Reddie, detached): boot a GLM TP4 config and measure it
# properly (3 measured bench passes + medians + per-cell spread), for the shipping candidate.
LBL=${1:?label}; shift
D=/var/tmp/boot-results/glm53
for kv in "$@"; do export "$kv"; done
export VLLM_EXTRA="${VLLM_EXTRA:---limit-mm-per-prompt {\"image\":4,\"video\":0}}"
echo "$(date -u +%T) FINAL-BOOT $LBL ($*)" >> $D/status.txt
bash /root/glm_boot.sh "$LBL" || { echo "$(date -u +%T) FINAL-BOOT-FAILED $LBL" >> $D/status.txt; exit 2; }
echo "$(date -u +%T) FINAL-SERVING $LBL" >> $D/status.txt
bash /root/glm_screen3.sh "$LBL" 3 1,3,6 2000,8000,32000,64000 > $D/screen3-$LBL.log 2>&1
echo "$(date -u +%T) FINAL-MEASURED $LBL" >> $D/status.txt
grep -aE "MEDIAN|C1 per-stream|aggregate mean-of-medians|prefill:|spread" $D/screen3-$LBL.log | tail -10
