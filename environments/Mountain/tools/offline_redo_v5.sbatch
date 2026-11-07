#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-11
#SBATCH --job-name=X500-Mountain-Plan-Bake
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_offline/log_redo_v5_%a.out
cd /mnt/weka/sgrigoryan/dataset_codebase/Mountain/tools
L=(55 56 58 59 60 61 62 63 65 66 67 68); IDS=${L[$SLURM_ARRAY_TASK_ID]}
python3 offline_all.py $IDS --bake
