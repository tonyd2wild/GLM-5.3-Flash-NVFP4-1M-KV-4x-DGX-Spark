#!/bin/bash
# Stop every lane, drop caches, and WAIT for the fleet to actually reclaim before the next boot allocates.
# glm_boot.sh drops caches then launches immediately; on a fleet coming off a ~100 GiB lane that races the
# reclaim and vLLM refuses with "Free memory on device cuda:0 ... less than desired GPU memory utilization".
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
NEED=${NEED:-105}
docker rm -f vllm_glm53 vllm_dsv41 >/dev/null 2>&1
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  $J -n $h "docker rm -f vllm_glm53 vllm_dsv41 >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null" </dev/null
done
sync; echo 3 > /proc/sys/vm/drop_caches
for i in $(seq 1 24); do
  sleep 15
  lo=999
  for h in "" tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
    if [ -z "$h" ]; then m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    else m=$($J -n $h "awk '/MemAvailable/{print int(\$2/1048576)}' /proc/meminfo" </dev/null 2>/dev/null); fi
    [ -n "$m" ] && [ "$m" -lt "$lo" ] && lo=$m
  done
  echo "$(date -u +%T) settle: min MemAvailable ${lo} GiB (need ${NEED})"
  [ "$lo" -ge "$NEED" ] && { echo "$(date -u +%T) settled"; exit 0; }
done
echo "$(date -u +%T) WARNING settle timed out at ${lo} GiB"; exit 1
