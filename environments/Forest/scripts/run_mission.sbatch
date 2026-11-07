#!/bin/bash
# usage: sbatch run_mission.sbatch <mission_id> <dest_dir> <vehicle_id>
#SBATCH --job-name=x500_forest
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/Forest/scripts/logs/%x_%j.out
#SBATCH --error=/mnt/weka/sgrigoryan/dataset_codebase/Forest/scripts/logs/%x_%j.err
#SBATCH --partition=research8
#SBATCH --exclude=gpu06
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=140G
#SBATCH --time=16:00:00
set -euo pipefail
MISSION="$1"; DEST="$2"; VID="${3:-0}"
export PATH="/mnt/weka/shared-cache/miniforge3/bin:$PATH"
source /mnt/weka/shared-cache/miniforge3/etc/profile.d/conda.sh
conda activate behavior_h100
export OMNI_KIT_ACCEPT_EULA=Y PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="/mnt/weka/shared-cache/miniforge3/envs/flowr_root/lib:${LD_LIBRARY_PATH:-}"
RUN_DIR="/mnt/weka/sgrigoryan/dataset_codebase/Forest/_runs/${MISSION}_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
export XDG_CACHE_HOME="$RUN_DIR/cache" OPTIX_CACHE_PATH="$RUN_DIR/optix-cache"
cd /mnt/weka/sgrigoryan/dataset_codebase/common/pipeline
# Isaac Sim on this cluster intermittently dies at start-up with Vulkan ERROR_DEVICE_LOST / a segfault
# before any of our code runs (hit gpu02, gpu06, gpu07, gpu08 on 2026-10-09). Retry only that case:
# a failure after PX4 has armed is a real mission failure and is not retried.
set +e
# Start-up GPU crashes (ERROR_DEVICE_LOST / segfault before PX4 arms) are retried once here; anything
# beyond that is handled by scripts/manage_missions.sh on the login node, which resubmits the mission
# to a fresh allocation (sbatch is not available on compute nodes).
for attempt in 1 2; do
  sleep $((RANDOM % 60))  # de-synchronise Isaac Sim start-ups of jobs landing on the same node
  ATT_LOG="$RUN_DIR/attempt_${attempt}.log"
  python run_x500_mission.py --env Forest --mission "$MISSION" --dest "$DEST" --vehicle-id "$VID" \
      --run-dir "$RUN_DIR" 2>&1 | tee "$ATT_LOG"
  rc=${PIPESTATUS[0]}
  if [ $rc -eq 0 ] || grep -q "recording starts" "$ATT_LOG"; then
    echo "attempt $attempt finished (rc=$rc)"; exit $rc
  fi
  if [ $rc -eq 139 ] || grep -qE "ERROR_DEVICE_LOST|GPU crash|Segmentation fault" "$ATT_LOG"; then
    echo "attempt $attempt: Isaac Sim start-up GPU crash"; continue
  fi
  echo "attempt $attempt failed (rc=$rc) for a non-retryable reason"; exit $rc
done
exit 75
