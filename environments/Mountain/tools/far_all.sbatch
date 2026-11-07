#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --array=0-24
#SBATCH --job-name=X500-Mountain-FarTexture
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_offline/far_%a.out
cd /mnt/weka/sgrigoryan/dataset_codebase/Mountain/tools
python3 - <<PY
import sys, json
from pathlib import Path
CB = Path("/mnt/weka/sgrigoryan/dataset_codebase")
sys.path[:0] = [str(CB / "common/pipeline"), str(CB / "Mountain"), str(CB / "Mountain/scene")]
import mountain_v2 as mv, mountain_build as mb
for k in range(1, 101):
    if (k - 1) % 25 != $SLURM_ARRAY_TASK_ID:
        continue
    f = CB / f"Mountain/_offline/{k:03d}.json"
    if not f.exists():
        continue
    r = json.loads(f.read_text())
    pl = mv.plan(k, r["attempt"])
    print(k, mb.far_texture(pl), flush=True)
PY
