#!/bin/bash
# glm_boot.sh <label>  (root on Reddie): boot GLM-5.3 TP4 worker-first with the env knobs already
# exported, then poll /health until the engine is really up. Exit 0 serving, 2 boot failed, 3 timeout.
LBL=${1:?label}
D=/var/tmp/boot-results/glm53; mkdir -p $D
L=$D/boot-$LBL.log; exec > >(tee -a "$L") 2>&1
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
KNOBS="PATCH_HOME SPEC VLLM_EXTRA_EP IMAGE MODEL_DIR EXP_NAME GMU MAXLEN SEQS MNBT BLOCK MOE_BACKEND SPEC_K KV_MEM KV_DTYPE CG ROCE PREFIX_FIX VLLM_EXTRA NCCL_EXTRA"
ENVS=""; for k in $KNOBS; do v=$(eval echo "\${$k:-}"); [ -n "$v" ] && ENVS="$ENVS $k='$v'"; done   # single quotes: values carry JSON double quotes
echo "=== glm_boot $LBL $(date -u +%T) knobs:$ENVS"
T0=$(date -u +%s)
# stop everything first (a worker joining a live head hangs distributed init)
docker rm -f vllm_glm53 vllm_dsv41 >/dev/null 2>&1
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  $J -n $h "docker rm -f vllm_glm53 vllm_dsv41 >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null" </dev/null
done
sync; echo 3 > /proc/sys/vm/drop_caches
echo "--- stopped all, caches dropped $(date -u +%T)"
# worker-first: 3 Bluey, 2 Asusi, 1 Spark4, then head 0
for p in 3:tonyspark1@192.168.192.1:/home/tonyspark1 2:tonyspark3@192.168.192.3:/home/tonyspark3 1:tonyspark4@192.168.192.4:/home/tonyspark4; do
  r=$(echo "$p" | cut -d: -f1); h=$(echo "$p" | cut -d: -f2); ph=$(echo "$p" | cut -d: -f3)
  $J -n $h "sudo -n env PATCH_HOME=$ph $ENVS bash /root/glm53_tp4.sh $r" </dev/null | tail -2 || { echo "rank $r launch failed"; exit 2; }
  sleep 20
done
eval "env PATCH_HOME=/home/tonyspark2 $ENVS bash /root/glm53_tp4.sh 0" | tail -2 || { echo "head launch failed"; exit 2; }
echo "--- all ranks launched $(date -u +%T), polling /health"
for i in $(seq 1 120); do
  sleep 15
  st=$(docker inspect vllm_glm53 --format '{{.State.Status}}' 2>/dev/null)
  [ "$st" != running ] && { echo "HEAD EXITED after $(( $(date -u +%s) - T0 ))s"; docker logs --tail 400 vllm_glm53 > /var/tmp/boot-results/glm53/headlog-$LBL.txt 2>&1; echo "  (head log saved to headlog-$LBL.txt)"; grep -iE "error|Traceback|raise|Exception|NO_MEMORY|not support|Unsupported" /var/tmp/boot-results/glm53/headlog-$LBL.txt | tail -10; exit 2; }
  code=$(curl -s -o /dev/null -w "%{http_code}" -m 5 http://127.0.0.1:8000/health 2>/dev/null)
  if [ "$code" = 200 ]; then
    echo "SERVING $(date -u +%T) after $(( $(date -u +%s) - T0 ))s"
    docker logs vllm_glm53 2>&1 | grep -aoE "GPU KV cache size: [0-9,]+ tokens|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|RoCEnante all-reduce is live[^\"]*|Graph capturing finished in [0-9]+ secs, took [0-9.]+ GiB" | sed 's/^/  /' | sort -u | head -6
    exit 0
  fi
  [ $((i % 8)) = 0 ] && echo "  $(date -u +%T) waiting: head=$st http=$code shards=$(docker logs vllm_glm53 2>&1 | grep -aoE 'Loading safetensors checkpoint shards: +[0-9]+%' | tail -1)"
done
echo "TIMEOUT after $(( $(date -u +%s) - T0 ))s"; exit 3
