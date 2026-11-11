# PX4 Failure Detector: official documentation and source

The collision demos `wall_post` and `wall_fast_post` (Warehouse) turn on PX4's own **Failure Detector**. This folder keeps the
official material it is based on, unchanged, for the PX4 version we fly (**v1.14.3**).

| File | What it is | Source |
|---|---|---|
| `px4_docs_v1.14_config_safety.md` | PX4 User Guide v1.14, *Safety Configuration (Failsafes)*. The section **"Failure Detector"** (and "Attitude Trigger") is the one that matters | https://docs.px4.io/v1.14/en/config/safety.html#failure-detector (raw: github.com/PX4/PX4-user_guide, branch v1.14, `en/config/safety.md`) |
| `px4_docs_v1.14_flight_termination.md` | PX4 User Guide v1.14, *Flight Termination Configuration*: what PX4 does when the detector fires | https://docs.px4.io/v1.14/en/advanced_config/flight_termination.html |
| `failure_detector_params.c` | the parameter definitions (`FD_FAIL_R`, `FD_FAIL_P`, `FD_FAIL_*_TTRI`, ...) with their official descriptions | PX4-Autopilot v1.14.3, `src/modules/commander/failure_detector/` (our checkout: `dataset_codebase/common/px4_env/PX4-Autopilot`) |
| `FailureDetector.cpp` | the detector itself: lines ~211-253 compare roll/pitch with `FD_FAIL_R`/`FD_FAIL_P` and hold them for `FD_FAIL_*_TTRI` s | same |

## What it does (from the files above)
- **Trigger:** roll > `FD_FAIL_R` (default 60 deg) or pitch > `FD_FAIL_P` (default 60 deg) for longer than `FD_FAIL_R_TTRI` /
  `FD_FAIL_P_TTRI` (default 0.3 s). It runs in every mode where attitude is controlled (Position mode included); not in Acro.
- **During take-off** it always disarms the motors ("lockdown"), whatever `CBRK_FLIGHTTERM` is.
- **In flight** it only acts if `CBRK_FLIGHTTERM = 0` (default 121212 = off). Then PX4 runs **flight termination**: all
  controllers off, outputs to their failsafe values (motors stop), no recovery. With the default it only warns.
  (PX4 v1.14.3 `src/modules/commander/failsafe/failsafe.cpp`, the three `fd_critical_failure` lines.)
- PX4 reports it over MAVLink: `HEARTBEAT.system_status` becomes `FLIGHT_TERMINATION` (`MAV_STATE` 8), plus `STATUSTEXT` messages.

## How our runner uses it
`run_flight.py` (Benchmark_collision_warehouse_codebase), mission field `px4_failure_detector`:
1. before arming it sets `CBRK_FLIGHTTERM = 0` over MAVLink (`PARAM_SET`) and reads back `CBRK_FLIGHTTERM` and the `FD_FAIL_*`
   values; it does not arm until PX4 confirms 0;
2. every `HEARTBEAT.system_status` change and every `STATUSTEXT` is stored with its time
   (`px4_events.json`, and `collision.json` -> `px4_failure_detector`), and every telemetry frame carries `px4.system_status`.
Nothing in PX4 is modified: the thresholds are PX4's defaults.
