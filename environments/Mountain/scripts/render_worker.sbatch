#!/bin/bash
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=260G
#SBATCH --time=3-00:00:00
#SBATCH --exclude=gpu05,gpu06
# Long-lived render worker (user 2026-10-23: "if the queue gives us a GPU, keep it until all simulations are done"):
# holds one GPU and renders the missing slices of flown Easy Mountain runs, up to 3 at a time (the same x500_job.sbatch
# slices mode the manager used per job), one group after another, without going back to the queue in between.
# Claims are atomic (mkdir lock per slice); a lock whose worker job is gone is taken over. The manager (RENDER_BY_WORKERS=1)
# keeps flying and publishing. Exits when nothing is left to render for 30 minutes or < 2 h of its time remain.
set -uo pipefail
CB=/mnt/weka/sgrigoryan/dataset_codebase
HERE=$CB/Mountain
JOB=$CB/common/scripts/x500_job.sbatch
ME=$SLURM_JOB_ID
# compute nodes have no squeue: the job's end time comes from Slurm's environment, liveness from heartbeat files
END=${SLURM_JOB_END_TIME:-$(( $(date +%s) + 3 * 86400 ))}
HB=$HERE/_workers; mkdir -p "$HB"
( while true; do touch "$HB/$ME.alive"; sleep 60; done ) & HBPID=$!
trap 'kill $HBPID 2>/dev/null; rm -f "$HB/$ME.alive"' EXIT
log() { echo "$(date '+%F %T') [worker $ME $(hostname)] $*"; }
# a job id is alive if it is a worker with a fresh heartbeat, or a (manager) job whose render logs changed < 15 min ago
alive() {
  local j=$1 run=$2
  [ -n "$j" ] || return 1
  [ -f "$HB/$j.alive" ] && [ $(( $(date +%s) - $(stat -c %Y "$HB/$j.alive") )) -lt 300 ] && return 0
  [ -n "$(find "$run/tmp_$j" -maxdepth 1 -name 'attempt_*.log' -mmin -15 2>/dev/null | head -1)" ] && return 0
  return 1
}

claim() {  # claim up to 3 missing slices of one run; prints "<run> <m> <list>" or nothing
  local run m n SL s e name lock owner got=()
  for run in $(ls -d "$HERE"/_runs/easy_mountain_* | sort); do
    m=$(basename "$run")
    [ -f "$run/flight/FLIGHT_OK" ] || continue
    [ -f "$run/PUBLISHED" ] && continue
    [ -f "$run/manager/FAILED" ] && continue
    [ -f "$run/manager/HOLD_REST" ] && continue
    n=$(cat "$run/flight/FLIGHT_OK"); SL=$(cat "$run/manager/slice_size" 2>/dev/null || echo 1800)
    [ -n "$SL" ] || SL=1800
    got=()
    for ((s = 0; s < n; s += SL)); do
      e=$((s + SL < n ? s + SL : n)); name=slice_$(printf %06d $s)
      [ -f "$run/slices/$name.mp4" ] && continue
      sj=$(cat "$run/manager/$name.jid" 2>/dev/null || echo "")
      alive "$sj" "$run" && [ "$sj" != "$ME" ] && continue          # a manager job or another worker has it
      lock="$run/manager/lock_$name"
      if ! mkdir "$lock" 2>/dev/null; then
        owner=$(cat "$lock/owner" 2>/dev/null || echo "")
        alive "$owner" "$run" && continue
        rm -rf "$lock"; mkdir "$lock" 2>/dev/null || continue   # stale lock of a finished worker
      fi
      echo "$ME" > "$lock/owner"; echo "$ME" > "$run/manager/$name.jid"
      got+=("$s:$e")
      [ ${#got[@]} -ge ${MAXS:-3} ] && break
    done
    if [ ${#got[@]} -gt 0 ]; then echo "$run $m $(IFS=,; echo "${got[*]}")"; return; fi
  done
}

idle_since=$(date +%s)
log "start, time left $(( (END - $(date +%s)) / 3600 )) h"
while [ $(( END - $(date +%s) )) -gt ${MARGIN:-7200} ]; do
  c=$(claim)
  if [ -z "$c" ]; then
    [ $(( $(date +%s) - idle_since )) -gt 1800 ] && { log "nothing to render for 30 min -- done"; exit 0; }
    sleep 60; continue
  fi
  read -r run m list <<< "$c"
  log "render $m [$list]"
  t0=$(date +%s)
  timeout ${RENDER_TIMEOUT:-8h} bash "$JOB" Mountain slices "$m" "$list"
  rc=$?
  for se in ${list//,/ }; do rm -rf "$run/manager/lock_slice_$(printf %06d "${se%%:*}")"; done
  log "done $m [$list] rc=$rc in $(( ($(date +%s) - t0) / 60 )) min"
  [ $rc -eq 76 ] && { log "bad GPU -- leaving the GPU"; exit 76; }
  if [ $rc -eq 75 ]; then  # Isaac Sim crashed at start-up on both attempts: this GPU is bad -- record it and give it back
    U=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=uuid --format=csv,noheader 2>/dev/null); U=${U#GPU-}
    echo "${U:0:8} $(hostname) Isaac Sim start-up crash on both attempts, render worker $ME, $(date +%F)" >> $CB/common/scripts/bad_gpus.txt
    log "start-up crash on GPU ${U:0:8} -- recorded as bad, leaving the GPU"; exit 76
  fi
  idle_since=$(date +%s)
done
log "less than 2 h left -- done"
