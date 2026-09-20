#!/bin/bash
# chain3: hand the fleet back to DeepSeek. Waits for chain2 (GLM hiconc) to finish and for the HOLD flag to be
# cleared, but boots DeepSeek no later than the hard deadline regardless, because DS must be serving by 13:30 UTC.
G=/var/tmp/boot-results/glm53
DEADLINE=$(date -u -d "2026-09-20 12:10:00" +%s)
while :; do
  now=$(date -u +%s)
  [ $now -ge $DEADLINE ] && { echo "$(date -u +%T) CHAIN3 deadline reached, booting DS" >> $G/status.txt; break; }
  if ! pgrep -f "^bash /root/glm_chain2.sh" >/dev/null && ! pgrep -f "^bash /root/glm_hiconc.sh" >/dev/null; then
    [ -f $G/HOLD_DS ] || { echo "$(date -u +%T) CHAIN3 chain2 done and no hold, booting DS" >> $G/status.txt; break; }
  fi
  sleep 60
done
bash /root/ds_final.sh dsrestore >> $G/ds_final.log 2>&1
echo "$(date -u +%T) CHAIN3 done rc=$?" >> $G/status.txt
