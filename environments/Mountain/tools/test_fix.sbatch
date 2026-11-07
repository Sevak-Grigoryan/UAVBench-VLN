#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-5
#SBATCH --job-name=X500-Mountain-Physics-Test
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_audit/log_test_%a.out
cd /mnt/weka/sgrigoryan/dataset_codebase/Mountain/tools
L=(39 11 13 21 2 3); I=${L[$SLURM_ARRAY_TASK_ID]}
python3 offline_all.py $I && python3 physics_audit.py $I
