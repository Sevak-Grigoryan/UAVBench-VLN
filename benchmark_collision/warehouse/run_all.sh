#!/bin/bash
# Login-node driver of the collision-aware Forest benchmark:
#   GPU job (job.sbatch: episode flight + every collision test + episode render + debug video; resubmitted after a bad GPU /
#   start-up crash, continuing where it stopped) -> publish the episode -> score the expert -> checker self-test -> test report.
#   nohup bash run_all.sh 37 Simulation_1 >> logs/run_all.log 2>&1 &
set -u
IDX=${1:?Forest Easy v2 index}; SIM=${2:?Simulation_N}
HERE=/home/sgrigoryan/Desktop/Benchmark_collision_codebase
SCENE=easy_v2_$(printf %03d "$IDX"); EP=vlncol_forest_$(printf %03d "$IDX")
ROOT=/mnt/weka/sgrigoryan/Benchmark_collision_work
DEST=/mnt/weka/sgrigoryan/Benchmark_collision/Easy/Forest/$SIM
TESTS="wall rotor_ceiling tree thin_fast wrong_ground pad_ok pad_tip debug_sideways"
LOGS=$HERE/logs; EXCL=gpu05,gpu06,gpu09
NAME=VLNCol-Forest-$(printf %03d "$IDX")
log() { echo "$(date '+%F %T') $*"; }
mkdir -p "$ROOT" "$LOGS"

complete() {
  [ -f "$ROOT/$EP/flight/FLIGHT_DONE" ] || return 1
  for n in $TESTS; do [ -f "$ROOT/colltest_$n/flight/FLIGHT_DONE" ] || return 1; done
  [ -f "$ROOT/colltest_debug_sideways/debug/debug_external_onboard.mp4" ] || return 1
  for n in $TESTS; do
    [ "$n" = debug_sideways ] && continue
    [ -f "$ROOT/colltest_$n/flight/telemetry.json" ] || continue
    [ -f "$ROOT/colltest_$n/video/collision_test.mp4" ] || return 1
  done
  if [ -f "$ROOT/$EP/flight/FLIGHT_OK" ]; then
    read F0 F1 < <(python3 $HERE/window.py "$ROOT/$EP/flight")
    for ((s = F0; s < F1; s += 900)); do [ -f "$ROOT/$EP/slices/slice_$(printf %06d $s).mp4" ] || return 1; done
  fi
  return 0
}

tries=0
while ! complete && [ $tries -lt 6 ]; do
  tries=$((tries + 1))
  j=$(sbatch --parsable -p research --exclude=$EXCL -J $NAME -o "$LOGS/%x_%j.out" -e "$LOGS/%x_%j.err" \
      $HERE/job.sbatch "$SCENE" "$EP" "$ROOT" "$TESTS")
  log "submit job $j (try $tries, exclude $EXCL)"
  while squeue -j "$j" -h 2>/dev/null | grep -q .; do sleep 60; done
  sleep 5
  out="$LOGS/${NAME}_$j.out"
  if grep -q "^BAD GPU" "$out" 2>/dev/null; then
    node=$(grep -h "^BAD GPU" "$out" | head -1 | awk '{print $3}'); EXCL="$EXCL,$node"; log "bad GPU on $node -> resubmit"; continue
  fi
  log "job $j ended, exit $(sacct -j "$j" -X -n -o ExitCode | tr -d ' ')"
  grep -h "attempt\|episode window\|job done" "$out" | sed 's/^/    /'
done
complete || log "WARNING: not every unit finished after $tries jobs"

if [ -f "$ROOT/$EP/flight/FLIGHT_OK" ] && [ ! -f "$DEST/task.json" ]; then
  python3 $HERE/publish_episode.py --work "$ROOT/$EP" --dest "$DEST" || log "PUBLISH FAILED"
fi
if [ -f "$DEST/task.json" ]; then
  python3 $HERE/evaluate.py "$DEST" --telemetry "$DEST/reference/telemetry.json" --actions "$DEST/reference/action.json" \
    --collision "$DEST/reference/collision.json" -o "$DEST/results/reference_score.json" | head -40
  python3 $HERE/selftest.py "$DEST" -o "$DEST/results/checker_selftest.json"
fi
python3 $HERE/tests/check_tests.py --root "$ROOT" --episode "$DEST"
log "done"
