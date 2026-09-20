#!/bin/bash
# glm_takeover.sh: DeepSeek -> GLM-5.3-Flash TP4, NVFP4-attention build (the fastest measured config).
# Mirrors ds_handback.sh: stop, drop caches, and WAIT for the fleet to actually reclaim the outgoing lane's
# ~100 GiB before the incoming workers allocate. glm53_tp4.sh has no memory guard of its own, so skipping the
# wait risks an allocation failure rather than a clean refusal.
G=/var/tmp/boot-results/glm53
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
echo "$(date -u +%T) TAKEOVER stopping DeepSeek measurement and lane" >> $G/status.txt
for pat in "^bash /root/ds_final.sh" "^bash /root/ds_screen3.sh" "^bash /root/ds_hiconc.sh" "^bash /root/sr_boot.sh" "^bash /root/ds_handback.sh" "^bash /root/glm_chain"; do
  pgrep -f "$pat" | while read -r p; do kill "$p" 2>/dev/null && echo "  killed $pat pid $p"; done
done
sleep 2
docker rm -f vllm_dsv41 vllm_glm53 >/dev/null 2>&1
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  $J -n $h "docker rm -f vllm_dsv41 vllm_glm53 >/dev/null 2>&1; sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null" </dev/null
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
  echo "$(date -u +%T) TAKEOVER min MemAvailable: ${lo} GiB" >> $G/status.txt
  [ "$lo" -ge 100 ] && break
done
# The measured-best config: NVFP4 attention projections (W4A16) + the glm5next patch that lets them load.
export MODEL_DIR=keys-glm53-nvfp4-attn3 NVFP4_PATCH=1 MNBT=8192 SPEC_K=7
export VLLM_EXTRA="--limit-mm-per-prompt {\"image\":4,\"video\":0}"
echo "$(date -u +%T) TAKEOVER booting GLM (attn3 + NVFP4_PATCH=1, MNBT=8192 SPEC_K=7)" >> $G/status.txt
bash /root/glm_boot.sh glm-serve-nvfp4; rc=$?
if [ $rc != 0 ]; then echo "$(date -u +%T) TAKEOVER BOOT FAILED rc=$rc" >> $G/status.txt; exit $rc; fi
echo "$(date -u +%T) TAKEOVER GLM SERVING" >> $G/status.txt
python3 /root/sr_quality.py $G/glm-serve-nvfp4 > $G/glm-serve-quality.txt 2>&1
head -1 $G/glm-serve-quality.txt >> $G/status.txt
docker logs vllm_glm53 2>&1 | grep -aE "GPU KV cache size|quant_algo|MarlinNvFp4|Model loading took" | sed -E 's/^.*\] //' | tail -4 >> $G/status.txt
echo "$(date -u +%T) TAKEOVER done" >> $G/status.txt
