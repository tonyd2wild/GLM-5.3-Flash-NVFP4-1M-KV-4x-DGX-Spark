#!/bin/bash
# Boot Lane A (official nvidia/GLM-5.3-Flash-NVFP4 + our NVFP4 attention, NO abliteration) for the operator's
# own corruption test. Every knob is deliberately IDENTICAL to boot_keys.sh so the only variable between the
# two lanes is the checkpoint: same endpoint (:8000, glm-5.3-flash), 500K context, MNBT 8192, k=7, image 16.
# That makes the operator's side-by-side a single-variable comparison.
# No sweep, no probes after it comes up: the lane is left quiet for the operator.
G=/var/tmp/boot-results/glm53
mkdir -p $G/blackfrost-derisked
export MODEL_DIR=blackfrost-glm53-derisked-attn NVFP4_PATCH=1 MNBT=8192 SPEC_K=7 MAXLEN=500000
export VLLM_EXTRA='--limit-mm-per-prompt {"image":16,"video":0}'
export BENCH_MODEL=glm-5.3-flash
say(){ echo "$(date -u +%T) BLACKFROST $*" | tee -a $G/status.txt; }
say "SETTLE (stop all, drop caches, wait for >=105 GiB MemAvailable on every node)"
bash /root/glm_settle.sh >> $G/blackfrost-derisked/settle.log 2>&1
say "settle rc=$?"
say "BOOT blackfrost-glm53-derisked-attn (Blackfrost DERISKED, 500K, image 16)"
if ! bash /root/glm_boot.sh blackfrost-derisked; then say "BOOT FAILED - see docker logs vllm_glm53"; exit 2; fi
say "SERVING"
docker logs vllm_glm53 2>&1 | grep -aE "GPU KV cache size|quant_algo|MarlinNvFp4|Model loading took|limit-mm" \
  | sed -E 's/^.*\] //' | tail -5 | tee $G/blackfrost-derisked/kv-context.txt
python3 /root/sr_quality.py $G/blackfrost-derisked > $G/blackfrost-derisked/quality.txt 2>&1
say "QUALITY $(grep -aoE 'PASS|FAIL' $G/blackfrost-derisked/quality.txt | head -1)"
say "LANE A QUIET AND READY FOR THE OPERATOR TEST"
