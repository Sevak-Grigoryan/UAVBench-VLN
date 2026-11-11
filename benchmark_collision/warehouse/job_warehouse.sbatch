#!/bin/bash
# Warehouse collision tests (scenarios_warehouse.py): fly each test (run_flight.py --env Warehouse), then render its video
# (render_debug.py --camera side: outside camera | onboard camera), one Isaac Sim process at a time.
#   sbatch job_warehouse.sbatch <work root> "<test names>"
# Tests already flown / rendered are skipped, so a resubmitted job continues.
#SBATCH --partition=research
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --exclude=gpu05,gpu06,gpu09
#SBATCH --job-name=VLNColWH-tests
#SBATCH --output=/home/sgrigoryan/Desktop/Benchmark_collision_warehouse_codebase/logs/VLNColWH-tests_%j.out
set -uo pipefail
ROOT="$1"; TESTS="$2"; SCENE=warehouse_multiple_shelves
HERE=/home/sgrigoryan/Desktop/Benchmark_collision_warehouse_codebase
CB=/mnt/weka/sgrigoryan/dataset_codebase
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate behavior_h100
export PATH="$CONDA_PREFIX/bin:$PATH"   # a self-resubmitted job inherits an already-active env: conda activate then skips it and base python would come first
export OMNI_KIT_ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
TMPD="$ROOT/tmp_${SLURM_JOB_ID}"; mkdir -p "$TMPD"
export XDG_CACHE_HOME="$TMPD/cache" OPTIX_CACHE_PATH="$TMPD/optix-cache"
export HOME="$TMPD/home" CUDA_CACHE_PATH="$TMPD/cuda-cache" __GL_SHADER_DISK_CACHE_PATH="$TMPD/gl-cache"
mkdir -p "$HOME" "$CUDA_CACHE_PATH" "$__GL_SHADER_DISK_CACHE_PATH"
cd $CB/common/pipeline

BAD_GPUS="3d4343b0 3f93ab95 5763f019 ae9778d4 f42b4bec 0ee7e8c1 2221bcb1 47d88006 b832b697 3e1d756a 5781045f 72e28f11 a1823aac ab6b5970 c8a12c7f 578925d3"
BAD_GPUS="$BAD_GPUS $(awk '{print $1}' $CB/common/scripts/bad_gpus.txt 2>/dev/null)"
UUIDS=$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null)
if [ "$(echo "$UUIDS" | grep -c GPU-)" -gt 1 ]; then UUIDS=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES%%,*}" --query-gpu=uuid --format=csv,noheader 2>/dev/null); fi
GPU_UUID=${UUIDS#GPU-}
echo "gpu $(hostname) $GPU_UUID"
for b in $BAD_GPUS; do case "$GPU_UUID" in "$b"*) echo "BAD GPU $(hostname) $GPU_UUID -- resubmitted as $(sbatch --parsable "$HERE/job_warehouse.sbatch" "$@"); holding this GPU 4 min so the new job gets another one"; sleep 240; exit 76;; esac; done

vid=1
for n in $TESTS; do
  TW="$ROOT/colltest_$n"; mkdir -p "$TW"
  if [ ! -f "$TW/flight/FLIGHT_DONE" ]; then
    for att in 1 2; do
      L="$TW/flight_attempt_${SLURM_JOB_ID}_${att}.log"
      X500_NO_BAKE=1 python $HERE/run_flight.py --env Warehouse --scene $SCENE --mission "colltest_$n" --out "$TW/flight" \
        --vehicle-id $vid > "$L" 2>&1
      rc=$?
      pkill -u "$USER" -f "px4_sitl_default/bin/px4 .* -i $vid " 2>/dev/null
      echo "colltest_$n attempt $att rc=$rc: $(grep -h -m1 '\[x500\] RESULT' "$L" | cut -c1-300)"
      if [ -f "$TW/flight/FLIGHT_DONE" ] || grep -q "recording starts" "$L"; then break; fi
    done
  fi
  vid=$(( vid % 9 + 1 ))
  [ -f "$TW/flight/collision.json" ] || { echo "colltest_$n: no flight record, no video"; continue; }
  [ -f "$TW/video/collision_test.mp4" ] && continue
  for att in 1 2; do
    L="$TW/video_${SLURM_JOB_ID}_${att}.log"
    python $HERE/render_debug.py --env Warehouse --scene $SCENE --flight "$TW/flight" --out "$TW/video" --camera side \
      --max-seconds 12 --stride 2 --hold-s 2 --bank-samples 0 --name collision_test > "$L" 2>&1 && break
    grep -q "\[debug\] booted" "$L" && break
  done
  echo "colltest_$n video: $(ls "$TW/video/collision_test.mp4" 2>/dev/null || echo MISSING) $(grep -h 'side camera' "$L")"
done
echo "job done"
