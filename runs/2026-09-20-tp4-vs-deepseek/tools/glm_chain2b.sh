#!/bin/bash
# chain2b: after chain2's C8/C12/C16 sweep, run the GLM-only C24/C32 probe, then clear the HOLD flag so
# chain3 hands the fleet back to DeepSeek. Never edits a running script; chain3 polls the flag.
G=/var/tmp/boot-results/glm53
for i in $(seq 1 120); do
  pgrep -f "^bash /root/glm_chain2.sh" >/dev/null || break
  sleep 30
done
if grep -q "FINAL-SERVING g12-ship9" $G/status.txt && docker ps --format '{{.Names}}' | grep -q vllm_glm53; then
  bash /root/specacc.sh $G/g12-specacc.txt >> $G/g12-hiconc2.log 2>&1
  bash /root/glm_hiconc2.sh g12-ship9 >> $G/g12-hiconc2.log 2>&1
  echo "$(date -u +%T) CHAIN2B hiconc2 done" >> $G/status.txt
else
  echo "$(date -u +%T) CHAIN2B skipped, GLM not serving" >> $G/status.txt
fi
rm -f $G/HOLD_DS
echo "$(date -u +%T) CHAIN2B hold cleared, DS handback released" >> $G/status.txt
