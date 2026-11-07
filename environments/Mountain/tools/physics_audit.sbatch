#!/bin/bash
#SBATCH --partition=research_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-24
#SBATCH --job-name=X500-Mountain-Physics-Audit
#SBATCH --output=/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_audit/log_physics_%a.out
cd /mnt/weka/sgrigoryan/dataset_codebase/Mountain/tools
IDS=$(python3 -c "print(' '.join(str(k) for k in ([39]+[k for k in range(1,101) if k not in (1,7,28,39)]) [$SLURM_ARRAY_TASK_ID::25] if __import__('os').path.exists(f'../_offline/{k:03d}.json')))")
python3 physics_audit.py $IDS
