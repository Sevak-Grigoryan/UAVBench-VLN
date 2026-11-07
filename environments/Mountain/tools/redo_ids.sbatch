#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --job-name=X500-Mountain-Plan-Bake
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_offline/redo_%a.out
# usage: sbatch --array=0-(n-1) redo_ids.sbatch "4 9 15 ..."   (one id per array task)
cd /mnt/weka/sgrigoryan/dataset_codebase/Mountain/tools
IDS=($1)
python3 offline_all.py ${IDS[$SLURM_ARRAY_TASK_ID]} --bake
