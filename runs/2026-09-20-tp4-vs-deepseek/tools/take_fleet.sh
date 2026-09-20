#!/bin/bash
# Stop the DeepSeek safety-net boot and free the GPUs for the NVFP4 conversion + GLM attempt 2.
# DeepSeek is restored automatically afterwards by chain6, settle step included.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
for pat in "^bash /root/glm_chain5.sh" "^bash /root/ds_handback.sh" "^bash /root/ds_final.sh" "^bash /root/sr_boot.sh"; do
  pgrep -f "$pat" | while read -r p; do kill "$p" 2>/dev/null && echo "killed $pat pid $p"; done
done
sleep 1
docker rm -f vllm_dsv41 vllm_glm53 >/dev/null 2>&1
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  $J -n $h "docker rm -f vllm_dsv41 vllm_glm53 >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null" </dev/null
done
sync; echo 3 > /proc/sys/vm/drop_caches
echo "fleet free $(date -u +%T); MemAvailable $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) GiB"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
