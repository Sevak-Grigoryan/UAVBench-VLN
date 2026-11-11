# collision_benchmark: published v0.2 data (without videos)

A copy of the published collision-aware benchmark folder (`Benchmark_collision/`), laid out as it is published. The
`trajectory.mp4` videos are not included (too large for git).

```
Easy/
  Forest/
    Simulation_1/              benchmark episode (v0.2): task.json, goal.json, reference/, results/
    Collision_situations/      Forest collision tests saved like episodes: 1_wall, 2_rotor_ceiling, 3_wrong_ground, 4_pad_ok
  Warehouse/
    Collision_situations/      warehouse collision tests: 1_shelf, 2_wall, 3_propeller_strike
```

## `Easy/Forest/Simulation_1`

> "Fly to the red barn with the grain silo beside it, south of the river, and stop about 28 m from it, facing it."

Scene `easy_v2_037` (dataset Easy Forest). Time limit 60 s. The expert flight scores SR 1, NE 0.83 m, SPL 1.00, ends
27.2 m from the barn with it in frame, and touched nothing; the checker self-test passes every case.
`reference/collision.json` is the contact monitor's record of the episode, which every scored flight must come with.

Score it:
```bash
python benchmark_collision/forest/evaluate.py examples/collision_benchmark/Easy/Forest/Simulation_1 \
    --telemetry  examples/collision_benchmark/Easy/Forest/Simulation_1/reference/telemetry.json \
    --actions    examples/collision_benchmark/Easy/Forest/Simulation_1/reference/action.json \
    --collision  examples/collision_benchmark/Easy/Forest/Simulation_1/reference/collision.json
```

## `Collision_situations`

Test flights that deliberately touch something, saved with the same layout as an episode so you can see how a contact
appears in the data (`contact_events` in the telemetry, the terminal record, `collision.json → first_prohibited`, the
last action `"end": "prohibited contact"`, `results/collision_verdict.json`). Each folder has its own README with the
details of every test.
