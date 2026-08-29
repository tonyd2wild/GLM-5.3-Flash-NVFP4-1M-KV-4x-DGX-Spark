#!/usr/bin/env bash
# Unconditional drop_caches every 60s for the whole boot window.
# MUST be unconditional. A threshold-triggered flusher (flush only when Cached > N)
# can sit below its threshold and still leave the NVRM allocator short, which shows up
# as the SAME command booting or OOMing depending on the moment. This is the single
# change that made 24 GiB/rank pass where it died on 2026-08-27.
# Run on EVERY node, started before the launcher, for the full boot.
end=$((SECONDS+3600))
while [ $SECONDS -lt $end ]; do
  sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null
  sleep 60
done
