#!/bin/bash
# Drives the Forest X500 missions through flight -> parallel render slices -> stitch + publish.
# Runs on the login node (compute nodes cannot submit jobs):
#     nohup bash manage_missions.sh > manager.log 2>&1 &
#
# Per mission, every minute:
#   1. flight   one GPU job flies the mission with real PX4 (physics only)       -> _runs/<m>/flight/FLIGHT_OK
#   2. slices   frames are split into SLICE-frame chunks, one GPU job each      -> _runs/<m>/slices/slice_*.mp4
#   3. publish  finalize.py stitches + verifies + copies 4 files to All_Simulations -> _runs/<m>/PUBLISHED
# Start-up GPU crashes and hangs (no log output for HANG_MIN minutes after start-up) are resubmitted,
# at most MAX_SUBMITS times per job. A real flight failure (the X500 flew but did not land/disarm) is
# not retried; it is reported in this log.
set -u
ENV=Forest
CB=/mnt/weka/sgrigoryan/dataset_codebase
HERE=$CB/$ENV
JOB=$CB/common/scripts/x500_job.sbatch
LOGS=$HERE/scripts/logs
A=/mnt/weka/sgrigoryan/All_Simulations
SLICE=900
MAX_SUBMITS=25
HANG_MIN=12
mkdir -p "$LOGS"

MISSIONS="easy_marker:Easy:1:0 easy_footbridge:Easy:2:1 easy_lookout:Easy:3:2 easy_cairn:Easy:4:3
medium_bridge_cairn:Medium:1:4 medium_river_lookout:Medium:2:5 medium_waterfall_rockslide:Medium:3:6
hard_bridge_lookout_cairn:Hard:1:7 hard_river_patrol:Hard:2:8 hard_five_stage:Hard:3:9"

log() { echo "$(date '+%F %T') $*"; }

# unit_step <run> <unit-name> <started-regex> <sbatch args...>
#   keeps one GPU job alive for a unit (flight or one slice); returns 0 if a job is running/queued,
#   2 if it gave up.
unit_step() {
  local run=$1 unit=$2 started=$3; shift 3
  local st=$run/manager; mkdir -p "$st"
  local jid; jid=$(cat "$st/$unit.jid" 2>/dev/null || echo "")
  local n; n=$(cat "$st/$unit.count" 2>/dev/null || echo 0)
  if [ -n "$jid" ] && squeue -j "$jid" -h 2>/dev/null | grep -q .; then
    # hang = the job's attempt log has not grown for HANG_MIN minutes, in ANY phase (the PX4/Pegasus
    # lockstep freeze also happens before arming, not only after recording starts)
    local logs=("$run"/tmp_"$jid"/attempt_*.log)
    if [ -e "${logs[0]}" ]; then
      local newest; newest=$(stat -c %Y "${logs[@]}" | sort -n | tail -1)
      local age=$(( $(date +%s) - newest ))
      if [ "$age" -gt $((HANG_MIN * 60)) ]; then
        log "HANG $(basename "$run")/$unit job $jid (no output for ${age}s) -> cancel"; scancel "$jid"
      fi
    fi
    return 0
  fi
  if [ "$n" -ge "$MAX_SUBMITS" ]; then return 2; fi
  local new; new=$(sbatch --parsable --job-name="x500_$(basename "$run")_$unit" -o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err" "$JOB" "$@")
  echo "$new" > "$st/$unit.jid"; echo $((n + 1)) > "$st/$unit.count"
  log "SUBMIT $(basename "$run")/$unit job $new (try $((n + 1))/$MAX_SUBMITS${jid:+, previous $jid})"
  return 0
}

while true; do
  remaining=0
  for spec in $MISSIONS; do
    IFS=: read -r m d k vid <<< "$spec"
    run=$HERE/_runs/$m; dest=$A/$d/$ENV/Simulation_$k
    mkdir -p "$run/manager"
    [ -f "$run/PUBLISHED" ] && continue
    [ -f "$run/manager/FAILED" ] && continue
    remaining=$((remaining + 1))

    if [ ! -f "$run/flight/FLIGHT_OK" ]; then
      fj=$(cat "$run/manager/flight.jid" 2>/dev/null || echo "")
      if [ -n "$fj" ] && ! squeue -j "$fj" -h 2>/dev/null | grep -q . && grep -qs "FLIGHT FAILED" "$run"/tmp_"$fj"/attempt_*.log; then
        log "FLIGHT FAILURE $m: $(grep -h -m1 RESULT "$run"/tmp_"$fj"/attempt_*.log | cut -c1-400)"
        touch "$run/manager/FAILED"; continue
      fi
      unit_step "$run" flight "recording starts" "$ENV" flight "$m" "$vid" || { log "GIVE UP $m flight"; touch "$run/manager/FAILED"; }
      continue
    fi

    n=$(cat "$run/flight/FLIGHT_OK")
    all=1
    for ((s = 0; s < n; s += SLICE)); do
      e=$((s + SLICE < n ? s + SLICE : n))
      name=slice_$(printf %06d $s)
      [ -f "$run/slices/$name.mp4" ] && continue
      all=0
      unit_step "$run" "$name" "\[slice\] booted" "$ENV" slice "$m" "$s" "$e" || { log "GIVE UP $m $name"; touch "$run/manager/FAILED"; break; }
    done
    if [ "$all" -eq 1 ]; then
      if out=$(python3 $CB/common/pipeline/finalize.py --run "$run" --dest "$dest" 2>&1); then
        log "PUBLISHED $m -> $dest ($out)"
      else
        log "FINALIZE FAILED $m: $out"; touch "$run/manager/FAILED"
      fi
    fi
  done
  [ "$remaining" -eq 0 ] && { log "all missions finished"; exit 0; }
  sleep 60
done
