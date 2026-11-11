# Local changes to PX4 and Pegasus

The pipeline uses two external projects, which are not copied into this repository:

| Project | Version used | Link |
|---|---|---|
| PX4 Autopilot (SITL) | v1.14.3 | https://github.com/PX4/PX4-Autopilot/tree/v1.14.3 |
| Pegasus Simulator | 5.1.0 (`main`, commit `e13dc65`); `dev_6.0.1` for Isaac Sim 6.2 | https://github.com/PegasusSimulator/PegasusSimulator |

Both are checked out under `environments/common/px4_env/` (`build_px4.sbatch`). The files here are the **only** changes we
made to them. Apply them after checking out the versions above.

## PX4: airframe `10050_isaac_x500`

1. Copy [`10050_isaac_x500`](10050_isaac_x500) to `PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/`.
2. Add the line `10050_isaac_x500` to the airframe list in
   `PX4-Autopilot/ROMFS/px4fmu_common/init.d-posix/airframes/CMakeLists.txt` (after the other `1xxxx` airframes).
3. Build SITL (`make px4_sitl_default`).

It is PX4's own `4001_gz_x500` (same rotor geometry and hover thrust) driven over MAVLink HIL the way Pegasus talks to
PX4, plus the stick settings used for every flight: linear sticks (no expo), 8 m/s horizontal, 3 m/s up, 2 m/s down,
60 °/s yaw at full stick, a 0.02 stick deadzone and a 3 s stick-loss timeout. The comments in the file give the reason
for each value.

## Pegasus: [`pegasus_simulator.patch`](pegasus_simulator.patch)

`git apply pegasus_simulator.patch` inside the PegasusSimulator checkout:
- `config/configs.yaml`: default airframe `isaac_x500` and the PX4 directory (a cluster path, adapt it);
- `px4_mavlink_backend.py`: waits at most 0.5 s for PX4's `HIL_ACTUATOR_CONTROLS` reply instead of forever. Without it
  PX4 and Pegasus could both wait for each other until the job timed out (seen on 2026-10-09); each such event is counted.

The Isaac Sim 6.2 variant (`benchmark_collision/forest_isaac62`) imports Pegasus' Python package from the same patched
checkout (same commit as the `dev_6.0.1` branch) without enabling its UI extension; the other 6.x fixes are in
`isaac62_compat.py`.
