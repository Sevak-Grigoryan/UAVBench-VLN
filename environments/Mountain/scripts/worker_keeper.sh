#!/bin/bash
# Keeps N render workers queued/running (login node; compute nodes cannot submit jobs). A worker that leaves a bad GPU
# (exit 76) is replaced at once, and for an hour replacements avoid the node that gave out the bad GPU: Slurm hands
# the same free bad GPU to the next waiting job (9 workers lost on gpu03 in 2 minutes, 2026-10-24).
#   nohup bash worker_keeper.sh >> worker_keeper.log 2>&1 &      stop: touch STOP_WORKER_KEEPER
cd /mnt/weka/sgrigoryan/dataset_codebase/Mountain/scripts
N=${N:-25}
BASE_EXCL=gpu05,gpu06
log() { echo "$(date '+%F %T') $*"; }
while [ ! -f STOP_WORKER_KEEPER ]; do
  # nodes that gave out a bad GPU in the last hour
  recent=$(find logs_render_workers -name 'worker_*.out' -mmin -60 -print0 2>/dev/null | xargs -0 grep -h "bad GPU -- leaving\|recorded as bad" 2>/dev/null \
           | sed -n 's/.*\[worker [0-9]* \(gpu[0-9]*\)\].*/\1/p' | sort -u | paste -sd,)
  excl=$BASE_EXCL${recent:+,$recent}
  have=$(squeue -u "$USER" -h -n X500-Mountain-Render-Worker | wc -l)
  left=$(ls -d ../_runs/easy_mountain_* | while read r; do [ -f "$r/PUBLISHED" ] || echo x; done | wc -l)
  [ "$left" -eq 0 ] && { log "all published -- keeper exits"; exit 0; }
  while [ "$have" -lt "$N" ]; do
    P=$([ $((RANDOM % 2)) = 0 ] && echo research8 || echo research)
    j=$(sbatch --parsable -p $P --exclude="$excl" -J X500-Mountain-Render-Worker -o logs_render_workers/worker_%j.out render_worker.sbatch)
    log "submitted worker $j on $P (exclude $excl)"
    have=$((have + 1))
  done
  sleep 120
done
log "stop file -- keeper exits"
