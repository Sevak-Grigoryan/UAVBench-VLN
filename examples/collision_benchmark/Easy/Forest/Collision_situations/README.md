# Collision situations (Forest, same scene as Simulation_1)

Test flights that deliberately touch something, saved with the **same layout as a benchmark episode**, so you can see how
a contact shows up in the data. They are tests, not episodes: there is no instruction and no target, so `task.json`
describes the test and `goal.json` holds the expected verdict. Flown 2026-11-09 with Isaac Sim 5.1 (Slurm job 378816).
Made by `~/Desktop/Benchmark_collision_codebase/package_collision_tests.py` from `Benchmark_collision_work/colltest_*`.

```
<N>_<test>/
  task.json                   the test, start pose, test obstacles, which frames the video shows
  goal.json                   expected verdict
  reference/
    trajectory.mp4            left: outside camera following the drone | right: onboard camera
    action.json               sticks + gimbal, one entry per change; the last entry says how the flight ended (no STOP)
    telemetry.json            every frame (episode format) + contacts_touching, contact_events, terminal
    collision.json            the contact monitor's record (same format as Simulation_1/reference/collision.json)
  results/
    collision_verdict.json    how the evaluator reads it: valid / collided / episode end / matches_expected
```

| Folder | Situation | Verdict |
|---|---|---|
| `1_wall` | fly forward at 3 m/s into a wall | COLLISION at t = 3.77 s: rotor 0 touched the wall |
| `2_rotor_ceiling` | climb slowly into a plate above the drone | COLLISION at t = 1.68 s: rotor 1 touched the ceiling |
| `3_wrong_ground` | land 8 m away from the landing clearing | COLLISION at t = 43.30 s: right skid touched the ground off the pad |
| `4_pad_ok` | land on the landing clearing | no prohibited contact: the skid contact is an allowed landing |

All four match their expected verdict. Times are from the start of the evaluated part of each test.

## How a collision shows up in the data (example `1_wall`)

1. **`telemetry.json`**, ordinary frames: `contact_events` is empty and `contacts_touching` is 0.
2. **One frame before the hit** (frame 113): the rotor comes within PhysX's 4 cm contact offset of the wall. It shows up as
   `contact_events` with verdict **`proximity`** and impulse 0. These events are recorded, but they never fail a flight.
3. **The hit** (physics step 905, between video frames 113 and 114): the first event with verdict **`prohibited`**
   (rotor `rotor_disc_0` vs `/World/TestFixtures/wall`, impulse 4.46 N·s). The run stops at that physics step. The last
   telemetry record is the **terminal record**: `"terminal": true, "termination": "prohibited contact"`, with the pose at
   the contact step. No frame is recorded after it.
4. **`collision.json`**: `first_prohibited` holds the same contact with its contact points, normals, impulses and
   separations; `termination` holds the stop; `contacts` lists every judged contact of the evaluated part.
5. **`action.json`**: the last entry is `"end": "prohibited contact"` instead of a STOP.
6. **`results/collision_verdict.json`**: `collided: true` and `episode_end: "COLLISION at t = ..."`. For an agent's flight
   this means the episode is failed, whatever the distance to the target.

The allowed case (`4_pad_ok`): the skids' ground contact appears as `proximity` events, then as `allowed_landing`
(`collision.json` `first_allowed_landing`, `collision_verdict.json` `allowed_landing_contact`). The flight is not stopped,
and `contacts_touching` stays above 0 after touchdown.

## Not included yet
- **tree** and **thin_fast**: the drone missed the obstacle (the test setup has to be fixed), so there was no contact.
- **pad_tip**: the simulator crashed; it is being flown again (job 380471).

## Videos
The videos of `1_wall` and `2_rotor_ceiling` were re-rendered on 2026-11-10 (job 380774). The outside camera is now placed
so that the test obstacle does not block its view of the drone. The first versions had the camera behind the wall. They are
kept in `Benchmark_collision_work/_replaced_videos_2026-11-10/`. A negative first frame in `task.json` `video.frames` means the
video starts a little before the evaluated part.
