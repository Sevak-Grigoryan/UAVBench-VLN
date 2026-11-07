#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-49
#SBATCH --job-name=X500-Mountain-Plan-Bake
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_offline/log_%a.out
cd /mnt/weka/sgrigoryan/dataset_codebase/Mountain/tools
IDS=$(python3 -c "print(' '.join(str(k) for k in range(1,101) if (k-1)%50==$SLURM_ARRAY_TASK_ID and k not in (1,7,28)))")
python3 offline_all.py $IDS --bake
