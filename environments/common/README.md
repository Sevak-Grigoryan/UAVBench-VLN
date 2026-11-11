# common: shared pipeline

Code used by every environment. The vehicle, the autopilot link, the pilot, the renderer and the publisher are
identical across City, Coastal, Forest and Mountain.

## `pipeline/`

| File | Role |
|---|---|
| `x500.py` | Holybro X500 V2 as a Pegasus Simulator vehicle, flown by PX4 v1.14.3 SITL (Pegasus ships only Iris). |
| `build_x500_usd.py` | Converts PX4's official X500 model (SDF + meshes) into the USD vehicle `x500.py` loads. |
| `smoke_x500.py` | End-to-end smoke test of Isaac Sim + Pegasus + PX4 with the X500. |
| `envs.py` | `boot(env_name, kit_dir, mission_id)`: builds the scene of any environment. Shared by flight and render so both see exactly the same scene. |
| `pilot.py` | Stick pilot: flies a mission's beats as a human would in PX4 Position mode, moving the sticks only when needed (change-only actions). |
| `run_x500_mission.py` | **Flight phase**: real PX4 SITL flies the mission with physics only and records telemetry, actions and camera poses. |
| `render_slice.py` | **Render phase**: renders frames `[start, end)` from the recorded camera poses (RealTimePathTracing). Many run in parallel. |
| `finalize.py` | **Publish phase**: stitches slices into `trajectory.mp4`, verifies, normalises and writes the sample. |
| `meta_schema.py` | One common `meta.json` structure for every published simulation. |
| `preview_stills.py` | A few still frames of a scene for review, without flying. |
| `encode_video.py` | Streaming mp4 encoder (frames piped to ffmpeg). |

## `scripts/`

| File | Role |
|---|---|
| `x500_job.sbatch` | One GPU job of the pipeline: `flight` or `slice` mode, with start-up guards. |
| `bad_gpus.txt` | GPU UUIDs known to fail. A job that lands on one exits with code 76 and is resubmitted. |
| `build_px4.sbatch` | Builds PX4 v1.14.3 SITL into `px4_env/` (not in the repo). |
| `preview.sbatch`, `smoke_h100.sbatch` | Preview stills and the smoke test as Slurm jobs. |

## `px4_patches/`

Our only changes to the external PX4 and Pegasus checkouts: the airframe `10050_isaac_x500` and
`pegasus_simulator.patch`. See [`px4_patches/README.md`](px4_patches/README.md).

## `vln_simulation/`

The earlier generator, used for the first natural-valley proof of concept and still imported by the Forest scene.

| File | Role |
|---|---|
| `natural_valley.py` | High-fidelity natural river valley scene. The Forest environment is built on it. |
| `generate_episode.py`, `postprocess_episode.py` | Proof-of-concept episode generation and finalisation. |
| `mission_contract.py`, `validation_contract.py`, `validate_dataset.py` | Navigation contracts and dataset validation from recorded state. |
| `recording.py`, `contact_recorder.py` | Bounded recording and PhysX contact reports. |
| `dataset_splits.py` | Split assignment by world ancestry. |
| `fetch_assets.py` | Downloads the pinned CC0 assets. |
