#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-48
#SBATCH --job-name=X500-Mountain-Replan-Audit
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_audit/log_replan_%a.out
cd /mnt/weka/sgrigoryan/dataset_codebase/Mountain/tools
IDS=$(python3 -c "print(' '.join(str(k) for k in [k for k in range(1,101) if k not in (1,7,28)][$SLURM_ARRAY_TASK_ID::49]))")
for I in $IDS; do python3 offline_all.py $I && python3 physics_audit.py $I; done
