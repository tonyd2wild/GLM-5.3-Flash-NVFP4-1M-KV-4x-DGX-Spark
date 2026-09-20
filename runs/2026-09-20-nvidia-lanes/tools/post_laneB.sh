#!/bin/bash
# After the chain finishes: one more CLEAN bench rep so Lane B has three (rep 1 was contaminated by a probe
# running concurrently), then a cold prefill probe on a quiet lane. Strictly sequential: nothing else runs
# against the endpoint while either is measuring.
G=/var/tmp/boot-results/glm53; O=$G/nvidia-laneB
export BENCH_MODEL=glm-5.3-flash
say(){ echo "$(date -u +%T) POST $*" >> $G/status.txt; }
for i in $(seq 1 60); do
  grep -q "CHAIN COMPLETE" $G/status.txt && break
  pgrep -f "^bash /root/chain_lanes.sh" >/dev/null || break
  sleep 30
done
sleep 20
say "clean rep 4 for Lane B"
python3 /root/v41bench.py --base http://127.0.0.1:8000/v1 --model glm-5.3-flash --label nvidia-laneB-r4 \
  --out $O --levels 1,2,3,4,5,6 --prefill "" --notes "Lane B clean rep 4" > $O/bench-r4.txt 2>&1
say "rep 4 done"
sleep 10
say "cold prefill probe on a quiet lane"
bash /root/probe_prefill.sh laneB-quiet > $O/probe.log 2>&1
say "prefill: $(cat /var/tmp/boot-results/speedrun/probe-prefill-laneB-quiet/request.txt 2>/dev/null)"
sleep 10
say "longer corruption run"
python3 /root/corrupt_probe.py > $O/corrupt2.txt 2>&1
say "corruption: $(grep -a TOTAL $O/corrupt2.txt | tail -1)"
say "ALL POST DONE"
