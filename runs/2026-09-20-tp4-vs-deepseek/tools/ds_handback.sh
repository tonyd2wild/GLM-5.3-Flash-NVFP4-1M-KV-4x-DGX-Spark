#!/bin/bash
# ds_handback.sh: hand the fleet back to DeepSeek, with the settle step the 11:27 failure proved is needed.
# That attempt died with "MemAvailable 15 GiB < 100 GiB, refusing to boot" on Bluey because the DS launcher
# checks free memory seconds after a 100 GiB GLM lane is stopped, before the fleet has reclaimed it. Exit 4
# is a refusal, not damage, but it costs a boot. So: stop, drop caches, then WAIT for the memory to come back
# on every node before handing over.
G=/var/tmp/boot-results/glm53
S=/var/tmp/boot-results/speedrun/s2-status.txt
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
echo "$(date -u +%T) HANDBACK stopping GLM on all four" >> $S
docker rm -f vllm_glm53 >/dev/null 2>&1
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  $J -n $h "docker rm -f vllm_glm53 >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null" </dev/null
done
sync; echo 3 > /proc/sys/vm/drop_caches
for i in $(seq 1 20); do
  sleep 15
  lo=999
  for h in "" tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
    if [ -z "$h" ]; then m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    else m=$($J -n $h "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" </dev/null 2>/dev/null); fi
    [ -n "$m" ] && [ "$m" -lt "$lo" ] && lo=$m
  done
  echo "$(date -u +%T) HANDBACK min MemAvailable across nodes: ${lo} GiB" >> $S
  [ "$lo" -ge 100 ] && break
done
echo "$(date -u +%T) HANDBACK booting DeepSeek (min mem ${lo} GiB)" >> $S
bash /root/ds_final.sh dsrestore >> $G/ds_final.log 2>&1
rc=$?
echo "$(date -u +%T) HANDBACK ds_final rc=$rc" >> $S
exit $rc
