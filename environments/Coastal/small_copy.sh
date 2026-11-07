#!/bin/bash
# Compressed (<30 MB) copy of a published Easy v2 video for sending to phone/web; kept in _runs, not All_Simulations.
K=$1; m=$(printf easy_coastal_%03d $K)
D=/mnt/weka/sgrigoryan/All_Simulations/Easy/Coastal/Simulation_$K
O=/mnt/weka/sgrigoryan/dataset_codebase/Coastal/_runs/$m/Simulation_${K}_preview_small.mp4
/mnt/weka/shared-cache/miniforge3/bin/ffmpeg -v error -y -i $D/trajectory.mp4 -c:v libx264 -crf 31 -preset veryfast -an $O
python3 -c "
import json;m=json.load(open('$D/meta.json'));v=m['verification'];p=m.get('props_physics_check') or {}
print(m['instruction']);print(f\"{m['duration_s']:.0f} s, {m['actions']} stick changes, touchdown {v['touchdown_error_m']} m, props at rest: {p.get('all_at_rest')} ({p.get('bodies')} bodies)\")"
ls -la $O | awk '{print $5, $9}'
