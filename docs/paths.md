# Cluster-specific paths

The code was written for one Slurm cluster and refers to absolute paths. Before running it elsewhere, replace the
roots below. The scorer (`benchmark/evaluate.py`, `selftest.py`, `geometry.py`) contains none of them.

| Root on the original cluster | Meaning | Replace with |
|---|---|---|
| `/mnt/weka/sgrigoryan/dataset_codebase` | this repository's `environments/` folder | `<repo>/environments` |
| `/home/sgrigoryan/Desktop/Benchmark_codebase` | this repository's `benchmark/` folder | `<repo>/benchmark` |
| `/mnt/weka/sgrigoryan/All_Simulations` | published dataset | your dataset output folder |
| `/mnt/weka/sgrigoryan/Benchmark`, `/mnt/weka/sgrigoryan/Benchmark_work` | published benchmark episodes and their work files | your benchmark output and scratch folders |
| `/mnt/weka/sgrigoryan/assets` | downloaded Poly Haven assets | your asset cache (filled by `fetch_*_assets.py`) |
| `/mnt/weka/shared-cache/miniforge3` | conda install with the Isaac Sim environment and ffmpeg | your conda / ffmpeg install |
| `/home/sgrigoryan/.conda/envs` | user conda environments | your conda envs |
| `/mnt/frtn/uav-sim/assets` | asset cache of the earlier `vln_simulation` generator | your asset cache |
| `/mnt/weka/sgrigoryan/Simulation_0_0_*`, `Simulations_0_0_*` | outputs of earlier single-sortie versions (only used by the older `scene/run_*_mission.py` runners) | — |

Slurm settings (partitions `research`, `research8`, `research_cpu`, `rtx`; node names `gpuNN`; `bad_gpus.txt`) are
also specific to that cluster.

## Files that contain absolute paths

| Occurrences | File |
|---|---|
| 1 | `benchmark/episode_mission.py` |
| 5 | `benchmark/job.sbatch` |
| 4 | `benchmark/publish_episode.py` |
| 4 | `benchmark/run_episode.sh` |
| 3 | `environments/City/scene/city_bridge.py` |
| 1 | `environments/City/scene/encode_video.py` |
| 1 | `environments/City/scene/fetch_city_assets.py` |
| 2 | `environments/City/scene/hq_bridge.py` |
| 2 | `environments/City/scene/probe_city_asset.py` |
| 2 | `environments/City/scene/probe_city_view.py` |
| 4 | `environments/City/scene/probe_materials.py` |
| 2 | `environments/City/scene/run_city_mission.py` |
| 2 | `environments/City/scripts/manage_easy_city.sh` |
| 3 | `environments/City/small_copy.sh` |
| 1 | `environments/Coastal/enrich.py` |
| 3 | `environments/Coastal/scene/coastal_bridge.py` |
| 3 | `environments/Coastal/scene/coastal_bridge_v2.py` |
| 1 | `environments/Coastal/scene/encode_video.py` |
| 1 | `environments/Coastal/scene/fetch_coastal_assets.py` |
| 2 | `environments/Coastal/scene/fetch_coastal_v2_assets.py` |
| 2 | `environments/Coastal/scene/hq_bridge.py` |
| 2 | `environments/Coastal/scene/probe_coastal_trajectory.py` |
| 2 | `environments/Coastal/scene/probe_water_isolate.py` |
| 2 | `environments/Coastal/scene/run_coastal_mission.py` |
| 2 | `environments/Coastal/scripts/manage_easy_coastal.sh` |
| 3 | `environments/Coastal/small_copy.sh` |
| 3 | `environments/Coastal/tools/probe.sbatch` |
| 3 | `environments/Coastal/tools/probe_rtx.sbatch` |
| 1 | `environments/Forest/scene/encode_video.py` |
| 2 | `environments/Forest/scene/forest_bridge.py` |
| 2 | `environments/Forest/scene/hq_bridge.py` |
| 3 | `environments/Forest/scene/run_forest_mission.py` |
| 2 | `environments/Forest/scripts/manage_easy_v2.sh` |
| 2 | `environments/Forest/scripts/manage_missions.sh` |
| 7 | `environments/Forest/scripts/run_mission.sbatch` |
| 1 | `environments/Forest/tools/check_grounding.py` |
| 1 | `environments/Forest/tools/easy_v2_stage_check.py` |
| 3 | `environments/Forest/tools/small_copy.sh` |
| 1 | `environments/Mountain/enrich.py` |
| 1 | `environments/Mountain/scene/encode_video.py` |
| 1 | `environments/Mountain/scene/fetch_mountain_assets.py` |
| 2 | `environments/Mountain/scene/fetch_mountain_v2_assets.py` |
| 2 | `environments/Mountain/scene/hq_bridge.py` |
| 2 | `environments/Mountain/scene/mountain_bake.py` |
| 3 | `environments/Mountain/scene/mountain_bridge.py` |
| 2 | `environments/Mountain/scene/mountain_bridge_v2.py` |
| 2 | `environments/Mountain/scene/probe_chain_wide.py` |
| 2 | `environments/Mountain/scene/probe_mountain_trajectory.py` |
| 2 | `environments/Mountain/scene/probe_mountain_view.py` |
| 2 | `environments/Mountain/scene/run_mountain_mission.py` |
| 2 | `environments/Mountain/scripts/manage_easy_mountain.sh` |
| 1 | `environments/Mountain/scripts/render_worker.sbatch` |
| 1 | `environments/Mountain/scripts/worker_keeper.sh` |
| 3 | `environments/Mountain/small_copy.sh` |
| 2 | `environments/Mountain/tools/bake_v6.sbatch` |
| 2 | `environments/Mountain/tools/bake_v7.sbatch` |
| 3 | `environments/Mountain/tools/far_all.sbatch` |
| 5 | `environments/Mountain/tools/fix_pass_point.py` |
| 2 | `environments/Mountain/tools/offline_all.sbatch` |
| 2 | `environments/Mountain/tools/offline_check.sbatch` |
| 2 | `environments/Mountain/tools/offline_redo_v5.sbatch` |
| 2 | `environments/Mountain/tools/physics_audit.sbatch` |
| 2 | `environments/Mountain/tools/redo_ids.sbatch` |
| 2 | `environments/Mountain/tools/replan_audit.sbatch` |
| 2 | `environments/Mountain/tools/test_fix.sbatch` |
| 2 | `environments/Mountain/tools/verify_pub.py` |
| 1 | `environments/common/pipeline/encode_video.py` |
| 2 | `environments/common/pipeline/finalize.py` |
| 1 | `environments/common/pipeline/render_slice.py` |
| 4 | `environments/common/scripts/build_px4.sbatch` |
| 4 | `environments/common/scripts/preview.sbatch` |
| 7 | `environments/common/scripts/smoke_h100.sbatch` |
| 4 | `environments/common/scripts/x500_job.sbatch` |
| 1 | `environments/common/vln_simulation/fetch_assets.py` |
