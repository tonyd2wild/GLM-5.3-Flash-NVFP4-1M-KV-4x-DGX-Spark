#!/bin/bash
# One owner at a time. Kill every chain and boot script, report what was alive, leave the fleet stopped.
for pat in "^bash /root/glm_chain" "^bash /root/ds_handback.sh" "^bash /root/ds_final.sh" "^bash /root/sr_boot.sh" "^bash /root/g1[4-9]_run.sh" "^bash /root/glm_boot.sh" "^bash /root/glm_final.sh"; do
  pgrep -af "$pat" | while read -r p rest; do kill "$p" 2>/dev/null && echo "killed: $p $rest"; done
done
sleep 2
echo "--- still alive:"; pgrep -af "^bash /root/(glm_chain|ds_|sr_boot|g1[4-9]_run|glm_boot|glm_final)" || echo "  none"
echo "--- containers:"; docker ps -a --format '{{.Names}} {{.Status}}' | head -5
