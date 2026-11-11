# Collision situations (Warehouse)

Test flights that deliberately touch something inside a warehouse, saved with the **same layout as a benchmark episode** and as
`../../Forest/Collision_situations`, so you can see how a contact shows up in the data. The racks and walls make the contacts
easy to see; the Forest folder keeps the floor (landing) cases. They are tests, not episodes: there is no instruction and no
target, so `task.json` describes the test and `goal.json` holds the expected verdict.

Scene: Isaac Sim's warehouse template `Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd` (Isaac Sim 5.1
assets), no people. Every one of its 1878 meshes is a PhysX collider; nothing is added for the tests. Flown 2026-11-10 with
Isaac Sim 5.1, PX4 v1.14.3 SITL, Position mode (Slurm jobs 381687, 381707, 381797).
Made by `~/Desktop/Benchmark_collision_warehouse_codebase/package_collision_tests.py` from
`Benchmark_collision_work_warehouse/tests/colltest_*` (scenarios: `scenarios_warehouse.py`).

```
<N>_<test>/
  task.json                   the test, start pose, which frames the video shows
  goal.json                   expected verdict
  reference/
    trajectory.mp4            left: outside camera following the drone (+ close-up inset) | right: onboard camera
    action.json               the sticks of the evaluated part; the last entry says how the flight ended (no STOP)
    telemetry.json            every frame (episode format) + contacts_touching, contact_events, terminal
    collision.json            the contact monitor's record (same format as Forest/Simulation_1/reference/collision.json)
  results/
    collision_verdict.json    how the evaluator reads it: valid / collided / episode end / matches_expected
```

| Folder | Situation | Verdict |
|---|---|---|
| `1_shelf` | face rack Shelf_0 from the aisle at 2.35 m, fly forward at ~1.3 m/s into the boxes on it | COLLISION at t = 4.98 s: left skid touched a cardboard box on Shelf_0 (impulse 0.59 N·s) |
| `2_wall` | fly forward at ~1.35 m/s along a free corridor into the east wall | COLLISION at t = 5.50 s: rotor 2 touched the wall (impulse 1.05 N·s) |
| `3_propeller_strike` | hover beside rack Shelf_1 at the height of its shelf beam (2.62 m), slide right at ~0.4 m/s | COLLISION at t = 6.01 s: rotor 0 touched the beam of the second shelf level (impulse 0.03 N·s) |

All three match their expected verdict. Times are from the start of the evaluated part (take-off and settling are flown
before it and are not recorded here).

## What to look at
- **`1_shelf`**: the boxes sit a little lower than the drone, so the lowest part, the **landing gear**, hits first.
  `drone_part: landing_gear`. The onboard camera ends filled with cardboard.
- **`2_wall`**: a flat wall at body height: the front **rotor** reaches it first (`rotor_disc_2`, the 14 cm disc proxy
  around the propeller). The outside camera is behind the drone and shows it stopping at the bricks.
- **`3_propeller_strike`**: a slow sideways slide. The propellers stick out 0.31 m to the side, further than any other part,
  so the **propeller itself** (`/World/x500/rotor0/collision`) touches the shelf beam while the airframe is still 0.13 m
  away. The impulse is tiny (0.03 N·s): any touch counts, however soft. Blade breakage is not simulated: the contact is
  detected and the run stops.

## How a collision shows up in the data (example `2_wall`)
1. **`telemetry.json`**, ordinary frames: `contact_events` is empty and `contacts_touching` is 0.
2. **Frame 165**, the last one: first the rotor comes within PhysX's contact offset of the wall: `contact_events` with
   verdict **`proximity`** (recorded, never a failure). Then, at physics step 1319 of the evaluated part, the first
   **`prohibited`** event: rotor `rotor_disc_2` vs `/Root/SM_WallA_6M15/SM_WallA_6M/Section0`. The run stops at that
   physics step; the record has `"terminal": true, "termination": "prohibited contact"`. Nothing after it is recorded.
3. **`collision.json`**: `first_prohibited` holds the same contact with contact points, normals, impulses and separations;
   `termination` holds the stop; `contacts` lists every judged contact of the evaluated part.
4. **`action.json`**: the last entry is `"end": "prohibited contact"` instead of a STOP.
5. **`results/collision_verdict.json`**: `collided: true`, `episode_end: "COLLISION at t = ..."`. For an agent's flight
   this means the episode is failed, whatever the distance to the target.

## Earlier attempts (kept, not packaged)
In `Benchmark_collision_work_warehouse/tests/colltest_propeller_strike/`:
- `flight_v1_leg_first`: hovering at 2.38 m the rotors passed 1 mm under the beam and a **leg** touched a box first.
- `flight_v2_pallet_3m`: a higher take-off overshot to 3.03 m and the **top plate** touched a pallet.
In `colltest_shelf/video_v1_lamps_blocked`: the first video, with the outside camera among the ceiling lamps.

## Videos
The outside camera follows the drone at a fixed offset, 0.5 m above it and 4.5 m away, chosen so that racks and walls do
not block the view (it is kept below the ceiling lamps). In `1_shelf` a box of the rack covers the left part of the outside
view near the end; the close-up inset still shows the contact. A negative first frame in `task.json` `video.frames` means
the video starts 1 s before the evaluated part.
