#!/bin/bash
# g14: the NVFP4-attention build. 334 non-expert projections (13.98 GiB bf16 -> 3.93 GiB NVFP4), W4A16 so no
# activation scales are needed. The fitted step model predicts 67.6 -> 57.2 ms, i.e. every category x1.18.
D=/var/tmp/boot-results/glm53
export MODEL_DIR=keys-glm53-nvfp4-attn MNBT=8192 SPEC_K=7
export VLLM_EXTRA="${VLLM_EXTRA:---limit-mm-per-prompt {\"image\":4,\"video\":0}}"
echo "$(date -u +%T) FINAL-BOOT g14-nvfp4attn (MODEL_DIR=$MODEL_DIR MNBT=8192 SPEC_K=7)" >> $D/status.txt
bash /root/glm_boot.sh g14-nvfp4attn || { echo "$(date -u +%T) FINAL-BOOT-FAILED g14-nvfp4attn" >> $D/status.txt; exit 2; }
echo "$(date -u +%T) FINAL-SERVING g14-nvfp4attn" >> $D/status.txt
bash /root/glm_screen3.sh g14-nvfp4attn 2 1,3,6 2000,8000 > $D/screen3-g14-nvfp4attn.log 2>&1
echo "$(date -u +%T) FINAL-MEASURED g14-nvfp4attn" >> $D/status.txt
bash /root/specacc.sh $D/g14-specacc.txt >> $D/g14.log 2>&1
grep -aE "MEDIAN|C1 per-stream|aggregate mean-of-medians|prefill:|spread" $D/screen3-g14-nvfp4attn.log | tail -8 >> $D/status.txt
echo "$(date -u +%T) g14 done" >> $D/status.txt
