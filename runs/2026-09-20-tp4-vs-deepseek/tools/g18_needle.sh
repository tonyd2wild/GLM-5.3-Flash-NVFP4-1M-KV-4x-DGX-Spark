#!/bin/bash
# The NVFP4 build quantized q/k/v/o and the MLA a/b projections. If that degraded attention numerics, long-context
# retrieval fails before a short-prompt quality gate would notice. So: needle at 65K and 130K.
D=/var/tmp/boot-results/glm53
for i in $(seq 1 60); do pgrep -f "^bash /root/glm_screen3.sh" >/dev/null || break; sleep 20; done
export BENCH_MODEL=glm-5.3-flash
python3 /root/v41needle.py --targets 65536,131072 --depth 0.3 --out $D/g18-needle.json > $D/g18-needle.txt 2>&1
echo "$(date -u +%T) g18 NEEDLE: $(grep -aoE "PASS|FAIL" $D/g18-needle.txt | tr '\n' ' ')" >> $D/status.txt
tail -6 $D/g18-needle.txt >> $D/status.txt
