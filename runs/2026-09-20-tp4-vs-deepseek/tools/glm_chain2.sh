#!/bin/bash
# after g12's 3-rep measurement finishes, run the high-concurrency sweep on the same serving lane
S=/var/tmp/boot-results/glm53/status.txt
for i in $(seq 1 240); do
  pgrep -f "^bash /root/glm_final.sh" >/dev/null || break
  sleep 30
done
echo "$(date -u +%T) CHAIN2 final done, hiconc next" >> $S
grep -q "FINAL-SERVING g12-ship9" $S || { echo "$(date -u +%T) CHAIN2 skip: g12 never served" >> $S; exit 0; }
bash /root/glm_hiconc.sh g12-ship9 >> /var/tmp/boot-results/glm53/g12-hiconc.log 2>&1
echo "$(date -u +%T) CHAIN2 hiconc done" >> $S
