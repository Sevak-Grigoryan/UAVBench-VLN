#!/bin/bash
#SBATCH --job-name=x500_px4_build
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/common/scripts/logs/%x_%j.out
#SBATCH --error=/mnt/weka/sgrigoryan/dataset_codebase/common/scripts/logs/%x_%j.err
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=24G
#SBATCH --time=01:00:00
set -euo pipefail
# PX4 v1.14.3 needs empy 3.3.4 and CMake 3.x (the shared env has empy 4 / CMake 4.4, which is what
# broke Simulations_0_0_2's build job 255829), so it builds from its own venv.
. /mnt/weka/sgrigoryan/dataset_codebase/common/px4_build_env/venv/bin/activate
cd /mnt/weka/sgrigoryan/dataset_codebase/common/px4_env/PX4-Autopilot
# GCC 13 vs PX4 1.14: several sources rely on <cstdint> arriving transitively, which it no longer does.
# GCC 13 also raises known false-positive diagnostics in PX4 1.14 (matrix::inv<1> array-bounds etc.)
# that PX4 turns into errors; keep them as warnings.
export CXXFLAGS="-include cstdint -Wno-error=array-bounds -Wno-error=stringop-overflow -Wno-error=dangling-pointer -Wno-error=maybe-uninitialized -Wno-error=use-after-free -Wno-error=dangling-reference -Wno-error=restrict"
export CFLAGS="-Wno-error=array-bounds -Wno-error=stringop-overflow"
make px4_sitl_default
ls -la build/px4_sitl_default/bin/px4
echo "PX4_BUILD_DONE"
