#!/bin/bash
# Put the two patched glm5next files in each node's tonyspark home under patches/nvfp4/.
J="sudo -u tonyspark2 ssh -i /home/tonyspark2/.ssh/id_ed25519_shared -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15"
install -d -o tonyspark2 -g tonyspark2 /home/tonyspark2/patches/nvfp4
for f in kda.py model.py; do install -o tonyspark2 -g tonyspark2 -m 644 /root/patches-nvfp4/$f /home/tonyspark2/patches/nvfp4/$f; done
echo "reddie $(md5sum /home/tonyspark2/patches/nvfp4/kda.py | cut -c1-8) $(md5sum /home/tonyspark2/patches/nvfp4/model.py | cut -c1-8)"
for hp in tonyspark4@192.168.192.4:/home/tonyspark4 tonyspark3@192.168.192.3:/home/tonyspark3 tonyspark1@192.168.192.1:/home/tonyspark1; do
  h=${hp%%:*}; ph=${hp##*:}
  $J -n $h "mkdir -p $ph/patches/nvfp4" </dev/null
  for f in kda.py model.py; do $J $h "cat > $ph/patches/nvfp4/$f" < /root/patches-nvfp4/$f; done
  echo "$h $($J -n $h "md5sum $ph/patches/nvfp4/kda.py | cut -c1-8; md5sum $ph/patches/nvfp4/model.py | cut -c1-8" </dev/null | tr '\n' ' ')"
done
