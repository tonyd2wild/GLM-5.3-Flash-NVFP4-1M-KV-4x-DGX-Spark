#!/bin/bash
# V4.1 boot poll. Runs as root on Reddie (rank 0 head). Prints one summary line,
# then terminal markers: STARTUP-COMPLETE / HEAD-EXITED / WORKER-DOWN:<nodes>.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"
REM='echo "st=$(docker ps -a --filter name=vllm_dsv41 --format {{.Names}}:{{.State}} | grep ^vllm_dsv41: | cut -d: -f2)"; echo "av=$(( $(grep MemAvailable /proc/meminfo | tr -s " " | cut -d" " -f2) / 1048576 ))"'
o=$(bash -c "$REM"); R=$(printf '%s\n' "$o" | grep '^st=' | cut -d= -f2); R=${R:-none}
minav=$(printf '%s\n' "$o" | grep '^av=' | cut -d= -f2)
line="R=$R"; down=""
for pair in S4:tonyspark4@192.168.192.4 A:tonyspark3@192.168.192.3 B:tonyspark1@192.168.192.1; do
  k=${pair%%:*}; h=${pair#*:}
  o=$($J "$h" "$REM" 2>/dev/null)
  if [ -z "$o" ]; then ws=unreach; av=""; else
    ws=$(printf '%s\n' "$o" | grep '^st=' | cut -d= -f2); ws=${ws:-none}
    av=$(printf '%s\n' "$o" | grep '^av=' | cut -d= -f2); fi
  line="$line $k=$ws"
  case "$ws" in exited|dead) down="$down $k";; esac
  [ -n "$av" ] && [ "$av" -lt "$minav" ] && minav=$av
done
b=$(( minav / 5 * 5 ))
sh=""; ms=""; er=""; jit=""
if [ "$R" != none ]; then
  L=$(docker logs --tail 4000 vllm_dsv41 2>&1)
  sh=$(printf '%s\n' "$L" | grep -oE 'checkpoint shards: +[0-9]+%' | tail -1 | grep -oE '[0-9]+')
  [ -n "$sh" ] && sh="shards~$(( sh / 25 * 25 ))%"
  ms=$(printf '%s\n' "$L" | grep -oE 'Loading weights took [0-9.]+ seconds|Model loading took [0-9.]+ GiB|GPU KV cache size: [0-9,]+ tokens|Application startup complete' | tail -1)
  jit=$(printf '%s\n' "$L" | grep -oE 'Building JIT module [A-Za-z0-9_]+' | tail -1)
  er=$(printf '%s\n' "$L" | grep -E ' ERROR |Traceback|No available shared memory|Under memory|EngineCore failed|RuntimeError|ValueError' | grep -v ' INFO ' | tail -1 | cut -c1-220)
fi
echo "$line | minAvail~${b}GiB | ${sh:-no-shards-yet} | ${ms:-no-milestone} | ${jit:+RUNTIME-JIT: $jit | }err: ${er:-none}"
case "$R" in exited|dead) echo "HEAD-EXITED";; esac
[ -n "$down" ] && echo "WORKER-DOWN:$down"
[ "$R" != none ] && printf '%s\n' "$L" | grep -q 'Application startup complete' && echo "STARTUP-COMPLETE"
exit 0
