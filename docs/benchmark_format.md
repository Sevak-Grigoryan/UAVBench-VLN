# Benchmark format (v0.1)

The dataset produced by `environments/` is **training data**: perfect expert flights with everything recorded. A
**benchmark episode** is an evaluation task that an agent flies by itself and that is scored by a fixed rule.

## Why a separate format

| Issue in the dataset format | Benchmark episode |
|---|---|
| Success was not defined. Every simulation is `completed`, and the only check compared the flight with the expert's own plan. | `goal.json` has a written **success rule with tolerances** that scores *any* flight. |
| `meta.json` mixes model inputs, the answer (beats, waypoints, target coordinates) and simulator logs. | **Split in two.** `task.json` is all the agent gets. `goal.json` is hidden and read only by the evaluator. Simulator logs are not published. |
| 70–80 % of each flight was take-off, transit and landing; the task was a small part. | The episode is **only the task**. It starts in the air and ends when the agent says **STOP**. |
| Fields differed between environments. | One fixed format for every episode. |
| Some expert runs barely saw the target. | The expert flight is **scored by the same checker** and must succeed before the episode is kept. |
| The expert path was the only "right answer". | The expert path is one valid solution. Only the end state is judged. |

## Episode layout

```
<Difficulty>/<Environment>/Simulation_<N>/
  task.json                 AGENT INPUT
  goal.json                 HIDDEN, evaluator only
  reference/
    trajectory.mp4          expert onboard video, episode frames only (1280x720, 30 fps)
    action.json             expert actions, one entry per change, last entry {"stop": true}
    telemetry.json          every frame: pose, velocity, AGL, camera pose + extrinsics, PX4 state
  results/
    reference_score.json    expert flight scored by evaluate.py (must be a success)
    checker_selftest.json   checker on edited copies of the expert flight
```

### `task.json` (agent input)

| Key | Content |
|---|---|
| `instruction`, `instruction_paraphrases`, `instruction_short` | the task in natural language; 3 paraphrases (similarity < 0.8 to the canonical text); a short form |
| `scene` | environment, scene id, source dataset simulation, landscape, season, sky |
| `start` | `xyz`, `yaw_deg`, `camera_pitch_deg`; the drone is armed and hovering in PX4 Position mode |
| `time_limit_s` | `ceil(2 × expert duration / 10) × 10` |
| `observation` | RGB 1280×720 at 30 fps, camera intrinsics, own pose |
| `action` | decision interval 0.5 s; sticks `x1` yaw, `y1` throttle, `x2` roll, `y2` pitch in [-1, 1] (step 0.1); gimbal pitch 0–80° (step 5°); `stop` |
| `coordinate_frame` | ENU metres (x east, y north, z up); yaw 0 = +x, counter-clockwise positive; camera pitch positive = down |

### `goal.json` (hidden)

| Key | Content |
|---|---|
| `target` | name, description, kind, `xy`, `point_xyz`, base/top height, `in_view_at_start` |
| `success` | rule text, `standoff_m`, `standoff_tolerance_m`, `frame_margin`, `max_tilt_deg`, `min_agl_m`, `time_limit_s` |
| `metrics` | definitions of SR, OSR, NE, SPL, heading error |
| `reference` | expert path length, duration, frames, actions, stop position, provenance |
| `instruction_checks` | paraphrase similarity scores |

### `action.json`

```json
{"index": 20, "frame": 705, "t": 23.5, "x1": 0.0, "y1": 0.0, "x2": 0.0, "y2": 0.7,
 "gimbal_pitch_deg": 15.0, "hold_frames": 300, "hold_s": 10.0}
...
{"index": 53, "frame": 1860, "t": 62.0, "stop": true}
```

### `telemetry.json` (one entry per frame)

```json
{"frame": 1859, "t": 61.97, "action_index": 52,
 "pose": {"x": 337.87, "y": 184.43, "z": 83.67, "roll_deg": 0.96, "pitch_deg": -1.71, "yaw_deg": 115.38},
 "velocity": {"vx": -0.19, "vy": 0.16, "vz": 0.01}, "agl_m": 55.16,
 "camera": {"x": 337.82, "y": 184.54, "z": 83.60, "yaw_deg": 115.38, "pitch_deg": 37.90},
 "camera_extrinsics": {"R_wc": [[...], [...], [...]], "t_wc": [...]},
 "px4": {"armed": true, "mode": "POSCTL", "landed_state": "IN_AIR"}}
```

Camera axes are OpenCV (rows of `R_wc` = right, down, forward). Pitch is positive looking down.

## Success rule (inspect)

The agent says STOP within `time_limit_s`, and at that moment:

1. its horizontal distance to `target.xy` is `standoff_m ± standoff_tolerance_m`, where the tolerance is
   `max(10, 0.3 · standoff_m)`;
2. `target.point_xyz` projects inside the image with `frame_margin` (10 %) clear at every edge;
3. during the whole episode it never tilted beyond `max_tilt_deg` (60°) and never came below `min_agl_m` (2 m).

Route, speed, height and hover time are not judged.

## Metrics

| Metric | Definition |
|---|---|
| **SR** | success rate (rule above) |
| **OSR** | oracle success: at some frame the drone was in the goal state (distance band + target in frame), even if it did not stop there |
| **NE** | navigation error at STOP = \|horizontal distance to target − `standoff_m`\| |
| **SPL** | SR × reference path / max(own path, reference path) |
| heading error | angle between the drone heading and the direction to the target at STOP |
| target-in-frame fraction | share of frames with the target in frame |

They follow the CityNav / AerialVLN / OpenFly protocol. Here the goal is a *state* (a distance band plus the target in
view) rather than a single point.

## Evaluation protocol

Evaluation is closed loop. The agent receives `task.json`, is placed at `start`, receives live camera frames from the
simulator, and sends an action every 0.5 s until it says STOP or the time limit is reached. The simulator records the
flight as telemetry and actions in the `reference/` format, and `benchmark/evaluate.py` scores it against `goal.json`.

Train/validation/test splits are left to the benchmark user. If a split is defined, a benchmark episode and the
dataset simulation it was built from must not be on different sides of it.
