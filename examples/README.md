# Examples

## `mountain_episode_001`

A complete published benchmark episode: `Easy/Mountain/Simulation_1`, built in the scene of dataset simulation
`Easy/Mountain/Simulation_13`.

> "Fly to the mountain hut with the flag and stop about 50 m from it, facing it."

| | |
|---|---|
| Start | hovering 30 m above the valley meadow, about 300 m from the hut, turned 63° away from it (hut not in view) |
| Success | STOP within 130 s, 50 ± 15 m from the hut, hut in frame, no crash, never below 2 m |
| Expert | 62 s, 287 m path, 54 actions; stops 49.4 m from the hut (NE 0.6 m, SPL 1.00) |
| Checker self-test | 9/9 cases pass |

| File | Content |
|---|---|
| `task.json` | Agent input: instruction, 3 paraphrases, short instruction, scene, start pose, observation and action spec, time limit |
| `goal.json` | Hidden: target, success rule, metric definitions, reference statistics |
| `reference/action.json` | Expert actions: sticks + gimbal pitch, one entry per change, last entry `{"stop": true}` |
| `reference/telemetry.json` | Expert flight, 1,860 frames: pose, velocity, AGL, camera pose + extrinsics, PX4 state |
| `results/reference_score.json` | The expert flight scored by `benchmark/evaluate.py` |
| `results/checker_selftest.json` | `benchmark/selftest.py` on the expert flight and 8 edited copies |

`reference/trajectory.mp4` (the onboard video, 1280×720, 30 fps) is not included.

`goal.json → reference.flown` records the cluster work directory the episode was built from. It's provenance only,
and nothing reads it.
