#!/bin/bash
# glm_dq.sh  (root on Reddie, detached): dynamic queue for the GLM TP4 speed run.
# Each line of /var/tmp/boot-results/glm53/queue.txt is "<label> KNOB=val [KNOB=val ...]" and is run
# with glm_run.sh. Editable while running (write .new then mv). Stops on empty queue, queue.stop, or
# past STOP_AT (default 12:40 UTC) so there is time to boot the final GLM config and then restore DeepSeek.
D=/var/tmp/boot-results/glm53
STOP_AT="${STOP_AT:-12:40}"
fails=0
while true; do
  [ -f $D/queue.stop ] && { echo "$(date -u +%T) dq: stop file" >> $D/status.txt; break; }
  [ "$(date -u +%H:%M)" \> "$STOP_AT" ] && { echo "$(date -u +%T) dq: past STOP_AT $STOP_AT" >> $D/status.txt; break; }
  L=$(grep -m1 -vE '^\s*(#|$)' $D/queue.txt 2>/dev/null)
  [ -z "$L" ] && { echo "$(date -u +%T) dq: queue empty" >> $D/status.txt; break; }
  grep -vxF "$L" $D/queue.txt > $D/queue.txt.pop; mv $D/queue.txt.pop $D/queue.txt
  echo "$L" >> $D/queue.done
  bash /root/glm_run.sh $L > $D/run-$(echo "$L" | awk '{print $1}').log 2>&1; rc=$?
  if [ $rc != 0 ]; then fails=$((fails+1)); [ $fails -ge 2 ] && { echo "$(date -u +%T) dq: two boot failures, stopping" >> $D/status.txt; break; }; else fails=0; fi
done
echo "$(date -u +%T) dq exited" >> $D/status.txt
