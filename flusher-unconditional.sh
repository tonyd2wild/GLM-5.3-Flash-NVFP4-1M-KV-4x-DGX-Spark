#!/usr/bin/env bash
# Unconditional drop_caches for the whole boot window.
#
# MUST be unconditional. A threshold-triggered flusher (flush only when Cached > N)
# can sit below its threshold and still leave the NVRM allocator short, which shows up
# as the SAME command booting or OOMing depending on the moment. This is the single
# change that made 24 GiB/rank pass where it died on 2026-08-27.
#
# Run on EVERY node, started BEFORE the launcher, and leave it running for the full boot.
# Stop it once the engine is serving: pkill -f flusher-unconditional.sh
set -u
DURATION="${1:-5400}"   # seconds; default 90 min. A cold NFS worker boot can exceed an hour.

if ! sudo -n true 2>/dev/null; then
  echo "FATAL: passwordless sudo is required (this loop runs 'sudo tee /proc/sys/vm/drop_caches')." >&2
  echo "       Without it the flusher fails silently and you reproduce the OOM it exists to prevent." >&2
  exit 1
fi

echo "flusher: starting, unconditional, every 60s for ${DURATION}s (pid $$)"
end=$((SECONDS+DURATION))
while [ $SECONDS -lt $end ]; do
  sync
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo "flusher: WARN drop_caches failed"
  sleep 60
done
echo "flusher: window elapsed after ${DURATION}s -- exiting. If the engine is still booting, restart it."
