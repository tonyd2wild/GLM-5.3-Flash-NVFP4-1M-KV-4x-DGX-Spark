#!/bin/bash
# (no kill here: this script is run while boots are in flight)
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
for h in tonyspark4@192.168.192.4 tonyspark3@192.168.192.3 tonyspark1@192.168.192.1; do
  $J $h "sudo -n tee /root/glm53_tp4.sh >/dev/null" < /root/glm53_tp4.sh
  echo "$h md5=$($J -n $h 'sudo -n md5sum /root/glm53_tp4.sh | cut -c1-8') syntax=$($J -n $h 'sudo -n bash -n /root/glm53_tp4.sh && echo ok')"
done
echo "REDDIE md5=$(md5sum /root/glm53_tp4.sh | cut -c1-8)"
