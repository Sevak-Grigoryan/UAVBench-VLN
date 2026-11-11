# warehouse: collision tests in a closed room

The `../forest` code, extended with a second environment: Isaac Sim's warehouse template
`Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd` (Isaac Sim 5.1 assets, no people). Every warehouse
mesh is made a PhysX collider; nothing is added for the tests. Racks, walls and the ceiling make contacts easy to see.

Flown with `run_flight.py --env Warehouse` (the Forest episode path is unchanged). Published results:
`../../examples/collision_benchmark/Easy/Warehouse/Collision_situations`.

## Files that differ from `../forest`

| File | What it does |
|---|---|
| `warehouse_env.py` | Opens the warehouse as the root stage, uses the dataset's renderer settings and camera, de-instances the shelving and adds colliders (`make_solid()`). |
| `warehouse_layout.json` | Positions of racks, walls, pillars, lamps and roof beams, written by `probe_warehouse.py` (GPU probe). |
| `probe_warehouse.py`, `probe_warehouse.sbatch` | The GPU probe that measures the layout. |
| `scenarios_warehouse.py` | The warehouse tests and their expected verdicts: `shelf`, `wall`, `propeller_strike`, `ceiling`, `wall_post`, `wall_fast_post`. |
| `job_warehouse.sbatch` | One GPU job: flies the given warehouse tests and renders one video per test. |
| `run_flight.py` | Adds `--env Warehouse`; and the **post-contact window** (mission field `post_contact_s`): after the first prohibited contact the episode is failed, but physics and PX4 keep running for that many seconds and are recorded (`phase: post-contact`, `episode_failed`, `s_after_first_contact`, `tilt_deg`). With `px4_failure_detector` it sets `CBRK_FLIGHTTERM = 0` before arming, reads the `FD_FAIL_*` parameters back, and records every `HEARTBEAT.system_status` change and `STATUSTEXT` (`px4_events.json`). |
| `contact_monitor.py` | Verdict `post_contact` for contacts inside the post-contact window. |
| `render_debug.py` | Side camera kept below the lamps (or below the drone near the ceiling); post-contact frames get an "AFTER CONTACT" bar. |
| `package_collision_tests.py` | Packages the warehouse tests (`--env Warehouse`). |

The post-contact demos (`wall_post`, `wall_fast_post`) and the `ceiling` test were added after the last published
warehouse run; their first flights are pending.
