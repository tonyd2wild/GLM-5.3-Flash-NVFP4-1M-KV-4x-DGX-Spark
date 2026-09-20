#!/bin/bash
# Re-boot the NVFP4-attention lane at 500K max context (matching DeepSeek's window), then verify long-context
# retrieval properly - including the exact case that failed at 1M: 131K at depth 0.3.
G=/var/tmp/boot-results/glm53
export MODEL_DIR=keys-glm53-nvfp4-attn3 NVFP4_PATCH=1 MNBT=8192 SPEC_K=7 MAXLEN=500000
export VLLM_EXTRA="--limit-mm-per-prompt {\"image\":4,\"video\":0}"
export BENCH_MODEL=glm-5.3-flash      # the takeover script omitted this, which is why the gate threw
echo "$(date -u +%T) 500K BOOT (attn3 + NVFP4_PATCH=1, MAXLEN=500000, MNBT=8192 SPEC_K=7)" >> $G/status.txt
bash /root/glm_boot.sh glm-500k; rc=$?
if [ $rc != 0 ]; then echo "$(date -u +%T) 500K BOOT FAILED rc=$rc" >> $G/status.txt; exit $rc; fi
echo "$(date -u +%T) 500K SERVING" >> $G/status.txt
docker logs vllm_glm53 2>&1 | grep -aE "GPU KV cache size|Maximum concurrency|quant_algo|MarlinNvFp4|Model loading took" | sed -E 's/^.*\] //' | tail -5 >> $G/status.txt
mkdir -p $G/glm-500k
python3 /root/sr_quality.py $G/glm-500k > $G/glm-500k/quality.txt 2>&1
echo "$(date -u +%T) 500K QUALITY: $(head -1 $G/glm-500k/quality.txt | cut -c1-120)" >> $G/status.txt
python3 /root/idletest.py > $G/glm-500k/idletest.txt 2>&1
grep -aE "tok/s after first" $G/glm-500k/idletest.txt | cut -c1-95 >> $G/status.txt
# Needles: depth 0.3 is the one that dropped a digit at 131K on the 1M lane, so it is repeated first.
python3 /root/v41needle.py --targets 65536,131072,262144 --depth 0.3 --out $G/glm-500k/needle-d03.json > $G/glm-500k/needle-d03.txt 2>&1
echo "$(date -u +%T) 500K NEEDLES depth 0.3:" >> $G/status.txt
grep -ah prompt_tokens $G/glm-500k/needle-d03.txt | cut -c1-190 >> $G/status.txt
python3 /root/v41needle.py --targets 131072,450000 --depth 0.6 --out $G/glm-500k/needle-d06.json > $G/glm-500k/needle-d06.txt 2>&1
echo "$(date -u +%T) 500K NEEDLES depth 0.6:" >> $G/status.txt
grep -ah prompt_tokens $G/glm-500k/needle-d06.txt | cut -c1-190 >> $G/status.txt
echo "$(date -u +%T) 500K done" >> $G/status.txt
