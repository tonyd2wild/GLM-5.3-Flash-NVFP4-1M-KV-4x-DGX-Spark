#!/bin/bash
# chain5: the armed handback. Waits for the g14 experiment to finish or for the hard deadline, then runs
# ds_handback.sh (which includes the memory-settle step). Retries once if the boot refuses again.
G=/var/tmp/boot-results/glm53
DEADLINE=$(date -u -d "2026-09-20 12:45:00" +%s)
while :; do
  [ $(date -u +%s) -ge $DEADLINE ] && { echo "$(date -u +%T) CHAIN5 deadline, handing back" >> $G/status.txt; break; }
  if ! pgrep -f "^bash /root/g14_run.sh" >/dev/null && ! pgrep -f "^bash /root/glm_boot.sh" >/dev/null && ! pgrep -f "^bash /root/glm_screen3.sh" >/dev/null; then
    echo "$(date -u +%T) CHAIN5 g14 work finished, handing back" >> $G/status.txt; break
  fi
  sleep 30
done
bash /root/ds_handback.sh || { echo "$(date -u +%T) CHAIN5 first handback failed, retrying once" >> $G/status.txt; sleep 60; bash /root/ds_handback.sh; }
echo "$(date -u +%T) CHAIN5 done rc=$?" >> $G/status.txt
