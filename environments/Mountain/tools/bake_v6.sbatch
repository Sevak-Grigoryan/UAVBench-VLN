#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --array=0-95
#SBATCH --job-name=X500-Mountain-Bake
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_offline/log_bake_v6_%a.out
cd /mnt/weka/sgrigoryan/dataset_codebase/Mountain/tools
I=$(python3 -c "print([k for k in range(1,101) if k not in (1,7,28,39)][$SLURM_ARRAY_TASK_ID])")
python3 offline_all.py $I --bake
