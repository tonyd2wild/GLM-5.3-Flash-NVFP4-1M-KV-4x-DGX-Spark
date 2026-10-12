#!/usr/bin/env bash
# fleet_watchdog.sh -- auto-recovery for the GLM-5.3 vLLM multi-node fleet.
# Runs on Reddie (head, rank 0). Each probe checks that every rank's container is
# running, then /health, then a one-token canary request. On N consecutive failures
# it tears down ALL containers, runs the GB10 memory ritual, relaunches
# workers-first (rank 3 -> 2 -> 1) then the head (rank 0), waits for ready.
#
# Why more than /health: in a failover test on DeepSeek-V4.1-Flash (same vLLM
# multi-node mp executor, four GX10), killing one worker left the head at /health 200
# with nothing in its log, and client requests hung with no reply until an NCCL timeout
# took the head down ~6 min later. The engine blocks in a collective and stops logging,
# so neither /health nor log parsing sees it. Not reproduced on GLM-5.3 itself; the
# mechanism is the same.
#
# vLLM v1 CANNOT recover a dead engine core. Docker restart policies are unsafe
# here: headless workers exit 0 on head death (on-failure never fires) and the
# dead head often never exits at all. Full orchestrated relaunch is the only cure.
set -u

### ---- config -------------------------------------------------------------
HEALTH_URL="http://127.0.0.1:8000/health"   # NOT /v1/models: that returns 200
                                            # even with a dead engine. /health
                                            # returns 503 on EngineDeadError.
CHECK_INTERVAL=60          # seconds between probes
FAIL_THRESHOLD=3           # consecutive failures before recovery fires
CURL_TIMEOUT=15            # per-probe timeout
READY_TIMEOUT=3600         # matches VLLM_ENGINE_READY_TIMEOUT_S in launch script
CANARY_URL="http://127.0.0.1:8000/v1/completions"
CANARY_MODEL="glm-5.3-flash"   # --served-model-name in the launch script
CANARY_TIMEOUT=90          # chunked prefill interleaves it with long prompts
MAX_RECOVERIES=3           # failed relaunches in a row before giving up
CONTAINER="vllm_glm53"
LAUNCH_SCRIPT='~/launch-glm53-tp4-24g.sh'   # same path on every node
SSH_KEY="$HOME/.ssh/id_ed25519_shared"
SSH_OPTS=(-i "$SSH_KEY" -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
LOCKFILE="$HOME/.fleet_watchdog.lock"
PAUSE_FLAG="$HOME/.fleet_watchdog.pause"    # touch to pause (maintenance); rm to resume
GIVEUP_FLAG="$HOME/.fleet_watchdog.gaveup"  # written after MAX_RECOVERIES; rm to re-arm
LOGFILE="$HOME/fleet_watchdog.log"
POST_TEARDOWN_SLEEP=10     # let master-port TIME_WAIT / NVRM settle
INTER_WORKER_SLEEP=5

# rank -> ssh target; empty string = local (head). Launch order is the
# ARRAY ORDER below: workers 3,2,1 first, head 0 last.
RANK_ORDER=(3 2 1 0)
declare -A NODE=(
  [3]="tonyspark1@192.168.192.1"
  [2]="tonyspark3@192.168.192.3"
  [1]="tonyspark4@192.168.192.4"
  [0]=""
)
### -------------------------------------------------------------------------

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOGFILE"; }

run_on() {  # run_on <rank> <command string>
  local rank="$1"; shift
  local target="${NODE[$rank]}"
  if [[ -z "$target" ]]; then
    bash -lc "$*" >> "$LOGFILE" 2>&1
  else
    ssh "${SSH_OPTS[@]}" "$target" "$*" >> "$LOGFILE" 2>&1
  fi
}

healthy() { curl -sf -m "$CURL_TIMEOUT" -o /dev/null "$HEALTH_URL"; }

# probe: prints the reason and returns 1 on the first failed check.
probe() {
  local r target state
  for r in "${RANK_ORDER[@]}"; do
    target="${NODE[$r]}"
    if [[ -z "$target" ]]; then
      state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER" 2>/dev/null || echo missing)
    else
      state=$(ssh "${SSH_OPTS[@]}" "$target" "docker inspect -f '{{.State.Status}}' $CONTAINER 2>/dev/null || echo missing" 2>/dev/null || echo unreachable)
    fi
    [[ "$state" == running ]] || { echo "rank $r container: $state"; return 1; }
  done
  healthy || { echo "health"; return 1; }
  curl -sf -m "$CANARY_TIMEOUT" -o /dev/null "$CANARY_URL" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$CANARY_MODEL\",\"prompt\":\"1\",\"max_tokens\":1,\"temperature\":0}" \
    || { echo "canary"; return 1; }
  return 0
}

# The unconditional flusher MUST be running on EVERY node for the whole relaunch,
# not just a one-shot drop_caches. A threshold-triggered flusher can sit below its
# threshold and still starve the NVRM allocator, which is what made 24 GiB/rank fail
# on 2026-08-27. See README, 'The flusher is the whole trick'.
start_flusher() {
  local rank="$1"
  run_on "$rank" 'pkill -f flusher-unconditional.sh 2>/dev/null; setsid nohup bash "$HOME/flusher-unconditional.sh" >/tmp/flusher-unconditional.log 2>&1 < /dev/null & sleep 1; pgrep -f flusher-unconditional.sh >/dev/null && echo "flusher:RUNNING" || echo "flusher:FAILED"'
}

mem_ritual() {  # GB10 NVRM allocator hygiene (launch script requires it)
  local rank="$1"
  run_on "$rank" 'sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || echo "WARN: drop_caches failed (sudo -n?)"; echo 1 | sudo -n tee /proc/sys/vm/compact_memory >/dev/null || echo "WARN: compact_memory failed"'
}

recover() {
  log "=== RECOVERY START: $FAIL_THRESHOLD consecutive health failures ==="

  # 1. Tear down EVERYTHING first. A worker must never start while the old
  #    head is dying: it joins the stale TCPStore rendezvous, exits 0
  #    when the old head finally dies, and wedges the new head's rendezvous.
  for r in "${RANK_ORDER[@]}"; do
    log "teardown: docker rm -f $CONTAINER on rank $r (${NODE[$r]:-local})"
    run_on "$r" "docker rm -f $CONTAINER 2>/dev/null || true"
  done
  sleep "$POST_TEARDOWN_SLEEP"

  # 2. Memory ritual on all nodes AFTER teardown, BEFORE relaunch, then start the
  #    unconditional flusher and leave it running through the entire boot.
  for r in "${RANK_ORDER[@]}"; do
    log "mem ritual on rank $r"
    mem_ritual "$r"
    log "starting unconditional flusher on rank $r"
    start_flusher "$r"
  done

  # 3. Relaunch: workers rank 3 -> 2 -> 1, then head rank 0.
  for r in "${RANK_ORDER[@]}"; do
    log "launch rank $r on ${NODE[$r]:-local}"
    if ! run_on "$r" "$LAUNCH_SCRIPT $r"; then
      log "ERROR: launch of rank $r reported failure; continuing (head may still rendezvous)"
    fi
    [[ "$r" != "0" ]] && sleep "$INTER_WORKER_SLEEP"
  done

  # 4. Wait for the engine to come up (TP4 load takes many minutes).
  log "waiting up to ${READY_TIMEOUT}s for $HEALTH_URL"
  local waited=0
  until healthy; do
    sleep 30; waited=$((waited + 30))
    if (( waited >= READY_TIMEOUT )); then
      log "ERROR: fleet did not become healthy within ${READY_TIMEOUT}s -- will retry via main loop"
      return 1
    fi
  done
  log "=== RECOVERY COMPLETE: healthy after ${waited}s ==="
  return 0
}

### ---- main ---------------------------------------------------------------
exec 9>"$LOCKFILE"
if ! flock -n 9; then
  echo "fleet_watchdog already running (lock: $LOCKFILE)" >&2
  exit 1
fi
log "watchdog started (pid $$, interval ${CHECK_INTERVAL}s, threshold $FAIL_THRESHOLD)"

fails=0
failed_recoveries=0
while true; do
  if [[ -f "$GIVEUP_FLAG" || -f "$PAUSE_FLAG" ]]; then
    fails=0; sleep "$CHECK_INTERVAL"; continue
  fi
  if why=$(probe); then
    (( fails > 0 )) && log "probe OK again after $fails failure(s)"
    fails=0
    failed_recoveries=0
  else
    fails=$((fails + 1))
    log "probe FAIL ($fails/$FAIL_THRESHOLD): $why"
    if (( fails >= FAIL_THRESHOLD )); then
      if recover; then
        failed_recoveries=0
      else
        failed_recoveries=$((failed_recoveries + 1))
        log "recovery attempt failed ($failed_recoveries/$MAX_RECOVERIES)"
        # A boot that keeps failing needs a person; relaunching every few minutes hides the cause.
        if (( failed_recoveries >= MAX_RECOVERIES )); then
          echo "gave up $(date '+%F %T') after $failed_recoveries failed recoveries" > "$GIVEUP_FLAG"
          log "=== GIVING UP; rm $GIVEUP_FLAG to re-arm ==="
        fi
      fi
      fails=0
    fi
  fi
  sleep "$CHECK_INTERVAL"
done
