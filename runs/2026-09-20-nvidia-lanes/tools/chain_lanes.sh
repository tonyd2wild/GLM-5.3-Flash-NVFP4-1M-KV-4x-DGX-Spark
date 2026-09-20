#!/bin/bash
# Unattended: Lane A (official nvidia + NVFP4 attn), then build and run Lane B (same + dealignai o_proj).
# Each lane gets one retry, because the only failure so far was a memory race that the settle step now covers.
# If Lane B cannot be built or booted, Lane A is left serving rather than nothing.
G=/var/tmp/boot-results/glm53
say(){ echo "$(date -u +%T) CHAIN $*" >> $G/status.txt; }
for i in $(seq 1 40); do docker ps --format '{{.Names}}' | grep -q fetch_dealign || break; sleep 15; done
say "fetch done: $(docker logs fetch_dealign 2>&1 | grep -c '^  layers') tensors"

# ---- Lane A -----------------------------------------------------------------
bash /root/lane_run.sh nvidia-laneA nvidia-glm53-attn
if ! grep -q "nvidia-laneA ALL DONE" $G/status.txt; then
  say "Lane A attempt 2"
  bash /root/lane_run.sh nvidia-laneA2 nvidia-glm53-attn
fi

# ---- build Lane B -----------------------------------------------------------
say "building Lane B (dealignai o_proj substituted before quantization)"
mkdir -p /var/tmp/glm53-vllm-cache/nv-ablit
docker rm -f conv_nvB build_nvB >/dev/null 2>&1
docker run --rm --gpus all -e OUT=/cache/nv-ablit -e SRC=/models/src \
  -e SUBST=/donor/dealign_oproj.safetensors \
  -v /root/mknvfp4b.py:/tmp/s.py:ro \
  -v /var/tmp/models/GLM-5.3-Flash-NVFP4-nvidia:/models/src:ro \
  -v /var/tmp/glm53-vllm-cache/dealign:/donor:ro \
  -v /var/tmp/glm53-vllm-cache:/cache \
  --entrypoint python3 ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 /tmp/s.py > $G/nvB-convert.log 2>&1
say "convert rc=$? donor-used=$(grep -oE 'donor tensors used: [0-9]+' $G/nvB-convert.log | tail -1)"
docker run --rm --entrypoint python3 -e SRC=/out/GLM-5.3-Flash-NVFP4-nvidia -e DST=/out/nvidia-glm53-ablit-attn \
  -e NEWDIR=/newshard -v /root/build_nv.py:/tmp/s.py:ro \
  -v /var/tmp/glm53-vllm-cache/nv-ablit:/newshard:ro -v /var/tmp/models:/out \
  ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2 /tmp/s.py > $G/nvB-build.log 2>&1
say "build rc=$? $(grep -aE 'VERIFY' $G/nvB-build.log | tail -1)"

# ---- Lane B -----------------------------------------------------------------
if [ -f /var/tmp/models/nvidia-glm53-ablit-attn/config.json ]; then
  bash /root/lane_run.sh nvidia-laneB nvidia-glm53-ablit-attn
  if ! grep -q "nvidia-laneB ALL DONE" $G/status.txt; then
    say "Lane B attempt 2"; bash /root/lane_run.sh nvidia-laneB2 nvidia-glm53-ablit-attn
  fi
else
  say "Lane B build produced no config.json; leaving Lane A serving"
  bash /root/lane_run.sh nvidia-laneA-restore nvidia-glm53-attn
fi
say "CHAIN COMPLETE"
df -h / | tail -1 >> $G/status.txt
