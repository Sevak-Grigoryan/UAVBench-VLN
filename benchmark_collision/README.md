# benchmark_collision: collision-aware benchmark (v0.2)

Version `vln-x500-bench-collision v0.2`, collision rule `collision-v1`: the same episodes as `../benchmark` (v0.1), but an
episode **ends and fails at the drone's first prohibited contact**. Contacts come from Isaac Sim's physics engine (PhysX),
checked after every 240 Hz physics step. The v0.1 benchmark and `../environments` are not changed: they are *run* from here,
and the flight runner is a modified copy (`run_flight.py`).

| Folder | What it is | Simulator |
|---|---|---|
| [`forest/`](forest/README.md) | **Main version.** Forest episode builder, contact monitor, collision proxies, publisher, evaluator, self-test and the 8 Forest collision tests. Built `Easy/Forest/Simulation_1`. | Isaac Sim 5.1 |
| [`warehouse/`](warehouse/README.md) | Collision tests in a closed room (Isaac Sim's warehouse template): shelf, wall, propeller strike, ceiling, and the post-contact demos with PX4's Failure Detector. | Isaac Sim 5.1 |
| [`forest_isaac62/`](forest_isaac62/README.md) | Port of `forest/` to Isaac Sim 6.2 (Python 3.12, Pegasus `dev_6.0.1`), used for `Easy/Forest/Simulation_3`. `dataset_port/` is a ported copy of the `environments/common` + `Forest` files it needs. | Isaac Sim 6.2 |

Each folder is self-contained: the shared files (`evaluate.py`, `geometry.py`, `collision_record.py`, ...) are copies, and
the README of each folder lists what differs.

## How one episode is made (`forest/run_all.sh <index> Simulation_N`)

1. **Scene**: an Easy Forest "inspect" mission of the dataset (`environments/Forest/missions.py`, e.g. `easy_v2_037`):
   the same valley, props and target. It is only read.
2. **Plan** (`episode_forest.py`): pre-roll (take off, turn 35-100° away from the target, settle), **episode** (fly to
   the target, stop about D m from it, face it, STOP), post-roll (land). Only the episode is published.
3. **Flight** (GPU, `job.sbatch` → `run_flight.py`): Isaac Sim + Pegasus + PX4 v1.14.3 SITL; the dataset stick pilot
   (`environments/common/pipeline/pilot.py`, via `pilot_ext.py`) sends MAVLink `MANUAL_CONTROL` every 0.5 s; the contact
   monitor (`contact_monitor.py`) judges every contact after every physics step; telemetry, actions and `collision.json`
   are recorded. The same job flies the collision tests (`scenarios.py`).
4. **Render**: the episode frames with the dataset renderer (`render_slice.py`); test videos with `render_debug.py`.
5. **Publish** (`publish_episode.py`): `task.json`, hidden `goal.json`, `reference/` (video, actions, telemetry,
   `collision.json`). It refuses if the expert touched anything or the contact monitor was not verified.
6. **Score** (`evaluate.py`: SR, OSR, NE, SPL; a contact fails the episode, missing collision data makes the flight
   INVALID), **self-test** (`selftest.py`) and the **test report** (`tests/check_tests.py`).

## Success rule (v0.2)
The v0.1 rule (STOP in time, distance `D ± max(10, 0.3·D)` m, target in frame with a 10 % margin, never tilted beyond 60°,
never below 2 m) **plus**: no part of the drone (rotors, airframe, landing gear) touched anything during the episode.

## Examples
Published data without the videos: [`../examples/collision_benchmark`](../examples/collision_benchmark/README.md).

## PX4 Failure Detector
PX4 itself has no collision sensor. Its Failure Detector only reacts to the attitude (roll or pitch above 60° for 0.3 s).
The warehouse post-contact demos switch it on (`CBRK_FLIGHTTERM = 0`) and record what PX4 reports after a contact. The
official documentation and source are in [`../docs/px4_failure_detector`](../docs/px4_failure_detector/README.md).
