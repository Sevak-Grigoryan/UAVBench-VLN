# Collision termination, approximations and the sim-to-real gap

Scope: benchmark `vln-x500-bench-collision v0.2` (rule `collision-v1`), Forest environment, X500 flown by PX4 v1.14.3 SITL
through Pegasus 5.1.0 in Isaac Sim 5.1 (PhysX, 240 Hz). The published v0.1 benchmark and the dataset are unchanged.

Each statement below has one of three tags:
- **[verified]**: checked by a real simulated test in `tests/` (see `tests/RESULTS.md`).
- **[configured]**: a value we set or inherited. Nobody has measured it against a real aircraft.
- **[not modelled]**: the simulator does not do this.

## 1. What the simulator does on contact

| Item | Behaviour | Status |
|---|---|---|
| Capture | `PhysxContactReportAPI` (threshold 0) on every X500 rigid body: body and the four rotors. `subscribe_contact_report_events` callback copies raw headers and points only. | configured |
| Judging | `drain()` runs after **every** 240 Hz physics step, outside the callback. It is independent of the 30 fps frames and the 0.5 s decisions. | verified (`between_frames`, `between_decisions` in the results) |
| Termination | The first prohibited contact is latched. The run stops right after that physics step. A terminal telemetry record is written at that step's time. No later frame is recorded. | verified (`terminated_at_contact_step`, `terminal_record_at_step_time`, `no_frame_after_contact`) |
| PX4 shutdown | Force-disarm (MAV_CMD_COMPONENT_ARM_DISARM, 21196). Physics keeps stepping (not recorded) until PX4 reports disarmed or 3 s pass, then PX4 is killed. | verified (`px4_disarmed_after`) |
| Saved | `collision.json`: monitor configuration, coverage audit, phases, every contact pair (first and last step, count, max impulse), the first 3 full records per pair (sim time, physics step, colliders, actors, instancer proto index, contact points, normals, impulses, separations), the first prohibited, invalid and allowed contact, the termination and the PX4 shutdown. | verified |
| Internal pairs | Drone-drone pairs are ignored. Body-rotor pairs are jointed, so PhysX does not pair them anyway. | unit test |
| Touch vs proximity | A pair counts as touching if any point has separation ≤ 5 mm or a non-zero impulse. Pairs that are only inside the 4 cm contact offset count as "proximity" and never fail an episode. | unit test |
| Missing data | No `collision.json`, or a monitor that saw no contact report during initialization: the evaluator returns **INVALID**, never "collision-free". | verified (`evaluator_invalid_without_data`) + self-test |

### Phases

1. **Initialization**: from spawn to lift-off, i.e. the first physics step with no touching contact once the skids are 3 cm above the spawn height. Landing gear resting on the take-off clearing is expected; this is also what proves the monitor works (`monitor_verified`). An airframe or rotor contact here makes the run *invalid*.
2. **Pre-roll** (take-off, turning to the start heading): runs before the evaluated episode. Any contact makes the episode **invalid** and ends the run. It is never exempted silently.
3. **Episode** (evaluated): every touching contact is **prohibited**, with one exception (next paragraph).
4. **Post-roll** (the expert's flight to the landing clearing): contacts are recorded only. Nothing here is evaluated.

**Landing exception** (only when the evaluated episode itself ends in a landing):
- contact is between **landing-gear** colliders (`skid_left`, `skid_right`, `leg_left`, `leg_right`) and a designated surface collider (`/World/Terrain`);
- it happens during the `land` beat of the designated pad;
- **every** contact point lies within 2.5 m of the pad centre.

Forest clearings are unmarked spots on the shared terrain mesh. That is why the point-in-footprint check is required: a bare collider check would exempt all ground contact. Painted markings (City, Coastal and Mountain pads) have their colliders disabled by the runner and can never be a landing surface. Rotor and airframe contact stays a failure even on the pad.

The Forest inspect episode ends in the air, so its landing exception is `null`: any contact fails it.

## 2. Collision coverage (what can be hit)

| Geometry | Collider | Approximation | Status |
|---|---|---|---|
| Terrain (`/World/Terrain`) | triangle mesh | This is the coarse terrain. Inside the river corridor the *visible* banks (`/World/River`, visual only) differ slightly from the collider. | configured |
| Buildings, props, landmarks | triangle meshes or USD primitives (dataset runner) | Movable props are rigid bodies. | configured |
| Trees (fir, pine saplings; jacaranda) | **new proxies**: a cone per conifer; a trunk cylinder plus canopy cylinder per broadleaf tree | Sized from the prototype bounding box: radius = mean half-extent, full height. The visual foliage has gaps, so proxies are solid where the real tree is partly empty, and miss single branches that stick out past the mean extent. | tree test verifies contact with a proxy |
| Rocks, stumps, logs | **new proxies**: oriented boxes from the bounding box | Boxes are larger than rounded shapes. | configured |
| Grass, ferns, shrubs (Groundcover) | **none** | Foliage below about 2 m. The episode's 2 m minimum-height rule applies instead. | not modelled |
| X500 airframe and landing gear | boxes from PX4's `model.sdf` (top plate, 2 legs, 2 skids) | Arms, motors and the GPS mast are covered only by the top plate box. | configured |
| X500 rotors | dataset: the blade itself, a 0.28 × 0.017 × 0.0008 m box spinning with the rotor; **new**: a static disc (r = 0.14 m, 2 cm thick) on the body per rotor | The disc covers the whole swept area at every instant. A real spinning blade is not a solid disc. The disc changes no dynamics, because the body's mass, inertia and centre of mass are set explicitly. | rotor_ceiling test |

Proxies are invisible and exist only in the flight job. They sit in a collision group that does not collide with the movable props.

City, Coastal and Mountain still leave tree and prop groups without colliders. `run_flight.py` **refuses** them for collision-aware evaluation.

Numerical settings:
- 240 Hz step, GPU dynamics;
- contact offset 4 cm on drone colliders, rest offset 0;
- CCD enabled on the scene and on every drone body.

At 8 m/s the drone moves 3.3 cm per step. The thin_fast test, a 2 cm pole at 8 m/s, is the check for missed contacts. CCD does not cover every case: rotational motion, articulation links, and very thin static triangle meshes can still tunnel.

## 3. What "failure" means

**"First prohibited contact = failure"** is a conservative **task rule**, not a prediction of physical damage. Real multirotors survive many light touches: a leaf, a skid scuff, even a prop strike sometimes. The simulator stops at the first contact, before modelling any of that. It does not model:
- prop or arm breakage;
- loss of thrust;
- tumbling;
- a recovery afterwards.

## 4. Vehicle and sensing models

| Aspect | What the code does | Evidence |
|---|---|---|
| Thrust | `T = 8.54858e-6 · ω²` per rotor, fixed coefficient (Pegasus QuadraticThrustCurve). Yaw moment `0.016 · T`. | Values copied from PX4's gz `x500/model.sdf`. **Not** measured on a real X500 V2 (KV920 motors, 1045 props). |
| Motor dynamics | **not modelled**: rotor speed follows the PX4 command within the step (no time constant, no ESC model, no voltage sag). | Pegasus source. |
| Drag | Linear, horizontal only: 0.248 N per m/s on x and y, zero on z. No rotor drag or blade flapping. | Derived from the sdf `rotorDragCoefficient`. Configured. |
| Wind, gusts, turbulence | **not modelled** (Pegasus marks it "TODO"). | Pegasus source. |
| Rotor inflow, ground effect, vortex ring | **not modelled**. | |
| Battery | **not modelled**. | |
| Damage | **not modelled** (see §3). | |
| Mass and inertia | 2.064 kg, Ixx = Iyy = 0.0217, Izz = 0.04 kg·m². | From the PX4 sdf. Not weighed or measured. |
| Friction and restitution | PhysX default material: no physics material is authored on the X500 or the terrain. | Configured by default. Not validated. |
| Sensors | Pegasus IMU, GPS, barometer and magnetometer with its default Gazebo-style noise and bias models (e.g. GPS 2e-4 m/√Hz). | Default values. Not fitted to a real Pixhawk 6C or GPS module. |
| Latency | PX4 runs in lockstep with physics, so there is no communication delay. Sticks are sent once per 30 fps frame. The agent decides every 0.5 s. | Real RC, telemetry and compute latencies are not modelled. |
| Controller | PX4 v1.14.3 Position mode, airframe 10050 (the gains of 4001_gz_x500; COM_DISARM_LAND 2 s). | Same as PX4's SITL reference. Not tuned on hardware. |

## 5. Onboard camera

The onboard camera is idealized:
- **Mount:** 0.12 m forward of and 0.07 m below the body centre, yaw locked to the airframe. Roll is held at exactly 0. Pitch is the commanded gimbal angle, slewed at 35°/s.
- **No gimbal dynamics:** no stabilisation error, lag, vibration or jitter, and the gimbal has no mechanical limits beyond 0–80° pitch.
- **Ideal optics:** pinhole model (24 mm lens, 20.955 mm aperture) with no lens distortion, rolling shutter, motion blur from the drone's motion, auto-exposure dynamics or sensor noise. Path tracing runs with 16 subframes.

Rendering happens after the flight, from the recorded poses. A static scene therefore renders correctly, but moving props are frozen at their authored poses. The opt-in external debug video (`render_debug.py`) shows the real attitude (roll and pitch) next to this stabilised view.

## 6. The sim-to-real gap is unmeasured

There are no matched real-world flights. No accuracy number, percentage or "realism score" can honestly be given. Validation measurements that would close the gap:

1. **Hover and step response:** log the real X500 V2 (PX4 ulog) for hover thrust, attitude and velocity steps (e.g. 0 → 3 m/s on x2), yaw-rate steps and climb steps. Compare rise time, overshoot, peak tilt and settling time with the same sticks in SITL.
2. **Motor and thrust stand:** measure thrust and torque against ω for the 2216 KV920 + 1045 prop, and the motor time constant. Replace the gz constants and add a first-order motor lag.
3. **Mass properties:** weigh the real airframe with battery and payload. Measure inertia with a bifilar pendulum.
4. **Drag:** steady-state tilt at constant speeds of 2–8 m/s in calm air gives the drag curve.
5. **Wind:** fly with measured wind (anemometer on a mast) and compare position-hold error.
6. **Ground effect:** hover thrust at 0.2–1.5 m above the ground.
7. **Sensors:** Allan variance of the real IMU, and the GPS noise and latency. Fit the Pegasus noise and bias parameters.
8. **Latency:** time from a stick command to the attitude response, through the real radio and companion-computer path.
9. **Gimbal and camera:** measured roll/pitch residual and jitter of the real gimbal, lens calibration (distortion), rolling-shutter time.
10. **Contact:** controlled low-speed contacts with foam and branch-like obstacles: what the drone survives and what the proxies miss (compare against high-speed video).

Each of these yields a per-quantity error, e.g. "step-response rise time within X %". That is the only defensible way to state a sim-to-real figure.
