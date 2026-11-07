#!/bin/bash
# Forest Easy v2 (100 single-task missions, own props layout each, river v2) -- copy of manage_missions.sh.
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
LOGS=$HERE/scripts/logs_easy_v2
A=/mnt/weka/sgrigoryan/All_Simulations
SLICE=900
MAX_SUBMITS=25
HANG_MIN=12
mkdir -p "$LOGS"

# easy_v2_001..100 -> All_Simulations/Easy/Forest/Simulation_5..104 (Simulation_1..4 are the earlier Easy runs)
MISSIONS=$(for i in $(seq 1 100); do printf "easy_v2_%03d:Easy:%d:x " $i $((i + 4)); done)
MAX_ACTIVE=${MAX_ACTIVE:-40}     # our queued + running jobs at most (keeps the queue fair)
MAX_FLIGHTS=${MAX_FLIGHTS:-8}    # concurrent flight jobs
# PX4 SITL only has distinct MAVLink ports for instances 0..9 (instance >= 10 sends to 14549 and the runner never
# hears a heartbeat -- flights 012/013/016, 2026-10-11). Each flight gets a vehicle id 0..9 not used by any of
# our other active flights.
free_vid() {
  local used="" j m
  for j in $(squeue -u "$USER" -h -o %j | grep '^x500_.*_flight$'); do
    m=${j#x500_}; m=${m%_flight}
    used="$used $(cat "$HERE/_runs/$m/manager/flight.vid" 2>/dev/null)"
  done
  for v in 0 1 2 3 4 5 6 7 8 9; do
    case " $used " in *" $v "*) ;; *) echo $v; return;; esac
  done
  echo 9
}

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
  if [ "$(squeue -u "$USER" -h -n "x500_$(basename "$run")_$unit" 2>/dev/null | wc -l)" -gt 0 ]; then return 0; fi
  if [ "$(squeue -u "$USER" -h -o %j | grep -c '^x500_')" -ge "$MAX_ACTIVE" ]; then return 0; fi
  if [ "$unit" = flight ] && [ "$(squeue -u "$USER" -h -o %j | grep -c '^x500_.*_flight$')" -ge "$MAX_FLIGHTS" ]; then return 0; fi
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
      vid=$(free_vid); before=$(cat "$run/manager/flight.jid" 2>/dev/null)
      unit_step "$run" flight "recording starts" "$ENV" flight "$m" "$vid" || { log "GIVE UP $m flight"; touch "$run/manager/FAILED"; }
      [ "$(cat "$run/manager/flight.jid" 2>/dev/null)" != "$before" ] && echo "$vid" > "$run/manager/flight.vid"
      continue
    fi

    [ "${FLIGHTS_ONLY:-0}" = 1 ] && continue   # render later (scene look still being reviewed)
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
  [ -f "$HERE/scripts/STOP_EASY_V2" ] && { log "stop file found, exiting"; exit 0; }
  sleep 60
done
