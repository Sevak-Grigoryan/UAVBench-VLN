#!/bin/bash
# Coastal Easy (100 single-task missions, own coast each) -- copy of City/scripts/manage_easy_city.sh.
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
ENV=Coastal
CB=/mnt/weka/sgrigoryan/dataset_codebase
HERE=$CB/$ENV
JOB=$CB/common/scripts/x500_job.sbatch
LOGS=$HERE/scripts/logs_easy_coastal
A=/mnt/weka/sgrigoryan/All_Simulations
SLICE=900
MAX_SUBMITS=25
GROUP=3            # slices rendered concurrently in one GPU job
HANG_MIN=12
mkdir -p "$LOGS"

# easy_coastal_001..100 -> All_Simulations/Easy/Coastal/Simulation_1..100
MISSIONS=$(for i in $(seq 1 100); do printf "easy_coastal_%03d:Easy:%d:x " $i $i; done)
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
  # a job that only found a known-bad GPU (x500_job.sbatch, exit 76) does not count as a try; its retry avoids that
  # node, otherwise it lands on the same idle bad GPU again (only bad GPUs were freeing up, 2026-10-15: 30 of 40 jobs)
  local excl=""
  if [ -n "$jid" ] && grep -qs "^BAD GPU" "$LOGS"/*_"$jid".out; then
    n=$((n - 1)); excl="--exclude=gpu02,gpu05,gpu09,$(grep -hs "^BAD GPU" "$LOGS"/*_"$jid".out | head -1 | awk '{print $3}')"
  fi
  if [ "$n" -ge "$MAX_SUBMITS" ]; then return 2; fi
  if [ "$(squeue -u "$USER" -h -n "x500_$(basename "$run")_$unit" 2>/dev/null | wc -l)" -gt 0 ]; then return 0; fi
  if [ "$(squeue -u "$USER" -h -o %j | grep -c '^x500_')" -ge "$MAX_ACTIVE" ]; then return 0; fi
  if [ "$unit" = flight ] && [ "$(squeue -u "$USER" -h -o %j | grep -c '^x500_.*_flight$')" -ge "$MAX_FLIGHTS" ]; then return 0; fi
  # time limit close to the real need (flight ~11-20 min, 1800-frame slice <= ~42 min): short jobs backfill into the
  # gaps the scheduler keeps while it collects 8 free GPUs for big jobs; 1:30 was too long to fit most gaps
  local tl=01:10:00; [ "$unit" = flight ] && tl=00:40:00
  local new; new=$(sbatch --parsable --time=$tl $excl --job-name="x500_$(basename "$run")_$unit" -o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err" "$JOB" "$@")
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

    # every movable prop must stay put during the flight; re-fly (max twice) if one slid or tipped
    if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$run/flight/meta.json'))['props_physics_check']['all_at_rest'] else 1)"; then
      rr=$(cat "$run/manager/rest_rejects" 2>/dev/null || echo 0)
      if [ "$rr" -lt 2 ]; then
        mv "$run/flight" "$run/flight_notrest_$rr"; echo $((rr + 1)) > "$run/manager/rest_rejects"
        rm -f "$run/manager/flight.jid" "$run/manager/flight.count"
        log "NOT AT REST $m: flight moved to flight_notrest_$rr, re-flying"; continue
      fi
      [ -f "$run/manager/HOLD_REST" ] || log "NOT AT REST $m after $rr re-flights -- on hold until the quay physics fix"
      touch "$run/manager/HOLD_REST"; continue
    fi
    [ "${FLIGHTS_ONLY:-0}" = 1 ] && continue   # render later (scene look still being reviewed)
    n=$(cat "$run/flight/FLIGHT_OK")
    all=1
    # slice size is fixed per run: 1800 frames for runs that start rendering now (half the Isaac start-ups, ~17 % less
    # GPU time), 900 for runs that already have 900-frame slices
    SL=$(cat "$run/manager/slice_size" 2>/dev/null)
    if [ -z "$SL" ]; then
      if ls "$run/manager"/slice_*.jid >/dev/null 2>&1; then SL=900; else SL=1800; fi
      echo "$SL" > "$run/manager/slice_size"
    fi
    # missing slices not covered by a queued/running job are rendered GROUP slices at a time in ONE GPU job
    # (concurrent renders; a single render keeps an H100 only ~37 % busy, 2026-10-15)
    free=() excl="" gave_up=0
    for ((s = 0; s < n; s += SL)); do
      e=$((s + SL < n ? s + SL : n))
      name=slice_$(printf %06d $s)
      [ -f "$run/slices/$name.mp4" ] && continue
      all=0
      sj=$(cat "$run/manager/$name.jid" 2>/dev/null || echo "")
      if [ -n "$sj" ] && squeue -j "$sj" -h 2>/dev/null | grep -q .; then
        hl=("$run"/tmp_"$sj"/attempt_*.log)
        if [ -e "${hl[0]}" ]; then
          age=$(( $(date +%s) - $(stat -c %Y "${hl[@]}" | sort -n | tail -1) ))
          [ "$age" -gt $((HANG_MIN * 60)) ] && { log "HANG $m/$name job $sj (no output for ${age}s) -> cancel"; scancel "$sj"; }
        fi
        continue
      fi
      c=$(cat "$run/manager/$name.count" 2>/dev/null || echo 0)
      if [ -n "$sj" ] && grep -qs "^BAD GPU" "$LOGS"/*_"$sj".out; then
        c=$((c - 1)); excl="--exclude=gpu02,gpu05,gpu09,$(grep -hs "^BAD GPU" "$LOGS"/*_"$sj".out | head -1 | awk '{print $3}')"
        echo "$c" > "$run/manager/$name.count"
      fi
      [ "$c" -ge "$MAX_SUBMITS" ] && { log "GIVE UP $m $name"; gave_up=1; break; }
      free+=("$s:$e")
    done
    [ "$gave_up" = 1 ] && { touch "$run/manager/FAILED"; continue; }
    while [ ${#free[@]} -gt 0 ]; do
      [ "$(squeue -u "$USER" -h -o %j | grep -c '^x500_')" -ge "$MAX_ACTIVE" ] && break
      grp=("${free[@]:0:$GROUP}"); free=("${free[@]:$GROUP}")
      list=$(IFS=,; echo "${grp[*]}"); first=$(printf %06d "${grp[0]%%:*}"); k=${#grp[@]}
      new=$(sbatch --parsable --time=01:40:00 --cpus-per-task=$((4 * k + 2)) --mem=$((80 * k + 20))G $excl \
            --job-name="x500_${m}_render_$first" -o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err" "$JOB" "$ENV" slices "$m" "$list")
      for se in "${grp[@]}"; do
        nm=slice_$(printf %06d "${se%%:*}"); c=$(cat "$run/manager/$nm.count" 2>/dev/null || echo 0)
        echo "$new" > "$run/manager/$nm.jid"; echo $((c + 1)) > "$run/manager/$nm.count"
      done
      log "SUBMIT $m/render [$list] job $new ($k slices on one GPU)"
    done
    if [ "$all" -eq 1 ]; then
      # publish only once the LLM paraphrases of this mission's instruction exist (Easy-data review item #1)
      python3 $HERE/has_paraphrases.py "$m" || continue
      if out=$(python3 $CB/common/pipeline/finalize.py --run "$run" --dest "$dest" 2>&1); then
        log "PUBLISHED $m -> $dest ($out)"
      else
        log "FINALIZE FAILED $m: $out"; touch "$run/manager/FAILED"
      fi
    fi
  done
  [ "$remaining" -eq 0 ] && { log "all missions finished"; exit 0; }
  [ -f "$HERE/scripts/STOP_EASY_COASTAL" ] && { log "stop file found, exiting"; exit 0; }
  sleep 60
done
