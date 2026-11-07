#!/bin/bash
# Drives ONE benchmark episode end to end, on the login node:  flight (GPU job) -> render the episode frames only (GPU job)
# -> publish to /mnt/weka/sgrigoryan/Benchmark -> score the expert flight -> checker self-test.
#   nohup bash run_episode.sh 13 Simulation_1 >> logs/run_013.log 2>&1 &
# K = the published Easy Mountain Simulation_K (an inspect mission) whose scene is used; it is only read.
set -u
K=${1:?K}; SIM=${2:?Simulation_N}
HERE=/home/sgrigoryan/Desktop/Benchmark_codebase
M=vlnbench_mountain_$(printf %03d "$K")
WORK=/mnt/weka/sgrigoryan/Benchmark_work/$M
DEST=/mnt/weka/sgrigoryan/Benchmark/Easy/Mountain/$SIM
LOGS=$HERE/logs; EXCL=gpu05,gpu06,gpu09
NAME=VLNBench-Mountain-$(printf %03d "$K")
log() { echo "$(date '+%F %T') $*"; }
mkdir -p "$WORK" "$LOGS"

# run_unit <suffix> <time> <extra sbatch args> -- <job args...>: submit, wait, resubmit after a bad GPU / start-up crash
run_unit() {
  local suf=$1 tl=$2 extra=$3; shift 4
  local tries=0 j st out node
  while [ $tries -lt 12 ]; do
    tries=$((tries + 1))
    j=$(sbatch --parsable -p research --time=$tl --exclude=$EXCL $extra -J $NAME-$suf -o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err" $HERE/job.sbatch "$@")
    log "submit $suf job $j (try $tries, exclude $EXCL)"
    while squeue -j "$j" -h 2>/dev/null | grep -q .; do sleep 60; done
    sleep 5
    out="$LOGS/$NAME-${suf}_$j.out"
    st=$(sacct -j "$j" -X -n -o ExitCode | tr -d ' ')
    if grep -q "^BAD GPU" "$out" 2>/dev/null; then
      node=$(grep -h "^BAD GPU" "$out" | head -1 | awk '{print $3}'); EXCL="$EXCL,$node"; log "bad GPU on $node -> resubmit"; continue
    fi
    if [ "$st" = "75:0" ]; then log "Isaac Sim start-up crash on both attempts -> resubmit"; continue; fi
    log "$suf job $j ended, exit $st"; return 0
  done
  log "GIVE UP $suf"; return 1
}

if [ ! -f "$WORK/flight/FLIGHT_OK" ]; then
  run_unit Flight 00:40:00 "" -- flight "$M" "$WORK" 0
  [ -f "$WORK/flight/FLIGHT_OK" ] || { log "FLIGHT FAILED: $(grep -h -m1 RESULT "$WORK"/tmp_*/attempt_*.log | cut -c1-400)"; exit 1; }
fi
log "flight OK: $(cat "$WORK/flight/FLIGHT_OK") frames"
read F0 F1 < <(python3 $HERE/window.py "$WORK/flight")
log "episode window: frames [$F0, $F1) = $((F1 - F0)) frames"
list=""; for ((s = F0; s < F1; s += 900)); do e=$((s + 900 < F1 ? s + 900 : F1)); [ -f "$WORK/slices/slice_$(printf %06d $s).mp4" ] || list="$list,$s:$e"; done
list=${list#,}
if [ -n "$list" ]; then
  k=$(echo "$list" | tr ',' '\n' | wc -l)
  run_unit Render 02:30:00 "--cpus-per-task=$((4 * k + 2)) --mem=$((60 * k + 20))G" -- slices "$M" "$WORK" "$list"
fi
for ((s = F0; s < F1; s += 900)); do [ -f "$WORK/slices/slice_$(printf %06d $s).mp4" ] || { log "slice $s missing after render -- stop"; exit 1; }; done
python3 $HERE/publish_episode.py --work "$WORK" --dest "$DEST" || { log "PUBLISH FAILED"; exit 1; }
python3 $HERE/evaluate.py "$DEST" --telemetry "$DEST/reference/telemetry.json" --actions "$DEST/reference/action.json" \
  -o "$DEST/results/reference_score.json" | head -30
python3 $HERE/selftest.py "$DEST" -o "$DEST/results/checker_selftest.json"
log "done -> $DEST"
