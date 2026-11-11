# Benchmark_collision codebase: collision-aware X500 VLN benchmark (v0.2, Forest)

> **Isaac Sim 6.2 variant.** Same code as `../forest`, ported to Isaac Sim 6.2.0 (pip install, Python 3.12) for
> `Easy/Forest/Simulation_3` (scene `easy_v2_069`). Port-specific files: `isaac62_compat.py` (stand-in for the removed
> `omni.isaac.dynamic_control`, Pegasus loaded without its UI extension, physics callbacks revived after start-up),
> `smoke62.py` / `smoke62*.sbatch` (smoke test), and `dataset_port/` (ported copy of the `environments/common` + `Forest`
> files). Paths and job names in `job.sbatch` / `run_all.sh` point at the 6.2 environment. The text below is the Forest
> README.

This version of the benchmark (`vln-x500-bench-collision v0.2`, rule `collision-v1`) ends and fails an episode at the drone's
first prohibited contact. Contacts are detected by PhysX at every 240 Hz physics step. The benchmark is built in the Forest
scene of `All_Simulations/Easy/Forest/Simulation_41` (Easy v2 mission `easy_v2_037`).

- **Data:** `/mnt/weka/sgrigoryan/Benchmark_collision/Easy/Forest/Simulation_N`
- **Work files:** `/mnt/weka/sgrigoryan/Benchmark_collision_work`

**Nothing else is changed:**
- the v0.1 benchmark (`Benchmark/`, `Benchmark_codebase/`);
- the dataset (`All_Simulations/`);
- `dataset_codebase/`. Its scene builders, PX4, Pegasus, pilot and slice renderer are *run* from here. The flight runner is
  a modified **copy** (`run_flight.py`).

**What the approximations mean and what is not modelled:** [docs/collision_and_sim2real.md](docs/collision_and_sim2real.md)
**Test results (real simulated flights):** [tests/RESULTS.md](tests/RESULTS.md)
**The task in Armenian:** [docs/task_hy.md](docs/task_hy.md)

## Files

| File | What it does |
|---|---|
| `run_flight.py` | Copy of the dataset flight runner, with: the contact monitor after every physics step; collision proxies; CCD; termination right after the first prohibited contact's step; clean PX4 force-disarm; telemetry with physics step, phase and attitude quaternion, plus a terminal record; `collision.json` and `proxies.json`. Forest only (other environments are refused). |
| `contact_monitor.py` | PhysX contact capture: the callback only copies. The rule is applied in `drain()`: phases, landing exception, latch. |
| `collision_proxies.py` | Static rotor-disc proxies on the X500 body; Forest tree, rock and fallen-wood proxies from the instancer bounding boxes; CCD; coverage audit. |
| `pilot_ext.py` | The dataset pilot plus a `sticks` beat (hold given sticks for N s), used by the tests. |
| `episode_forest.py` | Forest inspect episode: pre-roll / EPISODE / post-roll. Same design as the Mountain v0.1 episode. |
| `scenarios.py` | The 8 collision test scenarios and their expected results. |
| `collision_record.py` | Converts the flight's `collision.json` into the episode record the evaluator reads (episode-relative times and steps). |
| `publish_episode.py` | Publishes `task.json`, `goal.json`, `reference/` (video, actions, telemetry, **collision.json**) and `results/`. Refuses if the monitor isn't verified or the expert touched anything. |
| `evaluate.py` | SR / OSR / NE / SPL. A prohibited contact fails the episode. A missing or unverified collision record makes the flight **INVALID**. |
| `selftest.py` | Checker self-test on edited in-memory copies, now including contact cases. |
| `render_debug.py` | **Opt-in** external-camera debug video: fixed world camera, the X500 rendered with its full recorded attitude, overlay, side by side with the onboard view, and a bank-angle check. It is written to the work dir, never to the benchmark. With `--camera side` it also makes one video per collision test: the external camera follows the drone from the side, the test obstacles are drawn in orange, and the first prohibited contact is marked (red banner and circle). |
| `job.sbatch` | One GPU job: episode flight, all tests, episode render slices, the debug video, and a video of each collision test (`colltest_<name>/video/collision_test.mp4`: last 12 s up to the contact). Finished units are skipped. |
| `run_all.sh` | Login-node driver: job (resubmitted after a bad GPU), then publish, score, self-test and the test report. |
| `tests/check_tests.py` | Checks the real test flights and writes `tests/results.json` and `tests/RESULTS.md`. |
| `tests/unit_monitor.py` | Offline unit test of the rule logic (fake reports). The real behaviour is checked only by the flights. |

## Run

```
nohup bash run_all.sh 37 Simulation_1 >> logs/run_all.log 2>&1 &
```
