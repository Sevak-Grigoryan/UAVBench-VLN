"""Joystick-style flight controller, ported from Simulations_0_0_2 (same schema/logic), now with
an optional per-mission orbit_radius/height_above_target override for Simulations_0_0_3's larger
city landmarks (e.g. a bus station shelter needs more clearance than a car or hydrant).

Implements the original spec: drone flies to the target, moves around it (orbit), then comes
back -- with actions computed LIVE each simulated frame from the current state (a proportional
controller mapping "desired motion" to stick values), not scripted/derived after the fact.

Stick convention (Mode-2-style, documented per-mission in meta.json's sticks_legend so it's never
ambiguous downstream):
    left stick:  x1 = yaw rate (-1..1, +1 = turn right), y1 = vertical throttle (-1..1, +1 = up)
    right stick: x2 = lateral/roll (-1..1, +1 = strafe right, body frame),
                 y2 = forward/pitch (-1..1, +1 = forward, body frame)

NO-OVERSHOOT FIX (the actual cause of Simulations_0_0_1's infinite-loop/OOM crash): naive
"always command max speed toward the target" diverges once you're within one timestep's travel
distance of the goal -- deceleration lag causes overshoot, then correction overshoots back the
other way, forever. move_towards() below clamps commanded speed to `remaining / dt` so a step
never travels further than the remaining distance, matching the fix already applied and proven in
that project.
"""

import math
from dataclasses import dataclass

FPS = 30
DT = 1.0 / FPS
MAX_HORIZONTAL_SPEED = 14.0  # m/s -- raised from the city's 8.0 for Simulation_0_0_6. The only
                             # flat, landable valley ground on this mountain sits ~390-560m from
                             # the summit (steeper ground below that fails the offline flatness
                             # check -- verified, not assumed), so a realistic pad-to-peak transit
                             # here is inherently several times the city's ~130m. 14 m/s (~50km/h)
                             # is a normal cruise speed for a real inspection/mapping multirotor
                             # away from obstacles, not a shortcut around realism -- acceleration
                             # is still rate-limited below, so the ramp stays smooth.
MAX_VERTICAL_SPEED = 7.0    # m/s -- the summit is ~186m above the valley pads, so a realistic
                            # fast-climb rate matters here more than it did for the city's ~52m
                            # cruise altitude. 7 m/s is within a real multirotor's sport-mode climb
                            # rate.
MAX_YAW_RATE_DEG = 45.0  # deg/s -- was 60, lowered 2026-09-11 for a gentler peak turn speed
HEIGHT_ABOVE_TARGET = 2.0  # m -- flight altitude is TARGET-relative (target_z + this), not a
                            # fixed cruise altitude: with a fixed altitude, small low objects (a
                            # ~0.7m car) end up meters below the orbit path, so 3D distance-to-
                            # target during "orbit" is dominated by vertical offset rather than the
                            # intended close orbit radius -- defeats "drone must be near the object"
ORBIT_RADIUS = 6.0  # m -- "drone must be near the object" per the original spec
ORBIT_SPEED_DEG = 25.0  # deg/s default; pass orbit_speed_deg per mission. Tangential speed is
                        # radius * omega, so a big orbit radius at a high angular rate demands an
                        # unrealistic ground speed (38m at 25deg/s = 16.6 m/s). Slow it for wide orbits.
ORBIT_SWEEP_DEG = 300.0  # how far around the target to travel before returning
GOAL_TOLERANCE_M = 0.15

STICKS_LEGEND = {
    "x1": "yaw rate, -1..1, +1=turn right",
    "y1": "vertical throttle, -1..1, +1=up",
    "x2": "lateral/roll, -1..1, +1=strafe right (body frame)",
    "y2": "forward/pitch, -1..1, +1=forward (body frame)",
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _angle_diff_deg(a, b):
    """Shortest signed difference a-b in degrees, wrapped to [-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d


@dataclass
class DroneState:
    x: float
    y: float
    z: float
    yaw_deg: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    yaw_rate: float = 0.0  # deg/s, persisted between steps so yaw ACCELERATION can be rate-limited
                            # (see move_towards) -- the previous bang-bang controller had no memory
                            # of this, so it could only ever be 0 or +-max_yaw_rate, snapping
                            # instantly between them instead of ramping smoothly


# deg/s of commanded rate per degree of heading error (proportional control -- NOT error/dt,
# which made the old controller bang-bang: at 30fps, error/dt has a gain of 30, so any nonzero
# error saturated to max_yaw_rate immediately). Lowered 3.0 -> 1.8 on 2026-09-11 (user: "needs
# more smoother") for a softer response to large heading errors.
# Translational acceleration limits. Measured on the first city render, the drone was hitting
# 386 m/s^2 (~39g) because only YAW was rate-limited -- horizontal/vertical velocity jumped
# straight to the commanded value each frame, so the drone snapped from rest to full speed and
# stopped dead. Real multirotors accelerate at ~2-4 m/s^2. Limiting this is what makes the motion
# read as a real aircraft rather than a camera on rails.
MAX_ACCEL_H = 3.0   # m/s^2 horizontal -- slightly above the city's 2.5, in proportion to the
                    # higher cruise speed above; still well inside a real multirotor's envelope
MAX_ACCEL_V = 1.8   # m/s^2 vertical

YAW_RATE_GAIN = 1.8
# deg/s^2 -- caps how fast yaw_rate itself can change per frame, so a turn ramps up and eases down
# instead of snapping to/from full speed. Lowered 150 -> 40 on 2026-09-11, same request: at 40,
# reaching the (now 45deg/s) max rate from a stop takes ~1.1s instead of ~0.3s.
MAX_YAW_ACCEL_DEG = 40.0


def move_towards(state: DroneState, target_xyz, target_yaw_deg=None, max_h_speed=MAX_HORIZONTAL_SPEED,
                  max_v_speed=MAX_VERTICAL_SPEED, max_yaw_rate=MAX_YAW_RATE_DEG, dt=DT):
    """One control step toward target_xyz (and optionally a target heading). Returns
    (x1, y1, x2, y2) stick values already clamped to the no-overshoot rule; caller integrates
    state using the SAME values (so recorded actions and actual motion never diverge)."""
    tx, ty, tz = target_xyz
    dx, dy, dz = tx - state.x, ty - state.y, tz - state.z
    horiz_remaining = math.hypot(dx, dy)
    vert_remaining = abs(dz)

    # desired heading: face the direction of travel unless a target_yaw_deg is explicitly given
    desired_yaw = target_yaw_deg if target_yaw_deg is not None else (
        math.degrees(math.atan2(dy, dx)) if horiz_remaining > 1e-6 else state.yaw_deg
    )
    yaw_err = _angle_diff_deg(desired_yaw, state.yaw_deg)
    # proportional target rate (eases toward 0 as the error shrinks, instead of holding max speed
    # right up until a hard 2deg deadband), then rate-limited against last frame's yaw_rate so the
    # rate itself can't jump instantly -- this is what actually makes the rotation look smooth.
    desired_yaw_rate = _clamp(YAW_RATE_GAIN * yaw_err, -max_yaw_rate, max_yaw_rate)
    max_delta = MAX_YAW_ACCEL_DEG * dt
    yaw_rate = _clamp(desired_yaw_rate, state.yaw_rate - max_delta, state.yaw_rate + max_delta)
    x1 = _clamp(yaw_rate / max_yaw_rate, -1.0, 1.0) if max_yaw_rate > 0 else 0.0

    # vertical: speed capped so the drone can still DECELERATE to a stop at the target
    # (v = sqrt(2*a*s)), then the change in velocity itself is acceleration-limited.
    if vert_remaining > GOAL_TOLERANCE_M:
        v_stop = math.sqrt(2.0 * MAX_ACCEL_V * vert_remaining)
        v_target = math.copysign(min(max_v_speed, v_stop, vert_remaining / dt), dz)
    else:
        v_target = 0.0
    vz = _clamp(v_target, state.vz - MAX_ACCEL_V * dt, state.vz + MAX_ACCEL_V * dt)
    y1 = _clamp(vz / max_v_speed, -1.0, 1.0) if max_v_speed > 0 else 0.0

    # horizontal: same treatment -- stopping-distance-limited target speed, then the velocity
    # VECTOR change is acceleration-limited, so turns and starts/stops ramp instead of snapping.
    if horiz_remaining > GOAL_TOLERANCE_M:
        v_stop = math.sqrt(2.0 * MAX_ACCEL_H * horiz_remaining)
        h_speed = min(max_h_speed, v_stop, horiz_remaining / dt)
        target_vx = h_speed * dx / horiz_remaining
        target_vy = h_speed * dy / horiz_remaining
    else:
        target_vx = target_vy = 0.0
    dvx = target_vx - state.vx
    dvy = target_vy - state.vy
    dv = math.hypot(dvx, dvy)
    max_dv = MAX_ACCEL_H * dt
    if dv > max_dv and dv > 1e-9:
        dvx *= max_dv / dv
        dvy *= max_dv / dv
    world_vx = state.vx + dvx
    world_vy = state.vy + dvy
    yaw_rad = math.radians(state.yaw_deg)
    cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
    body_forward = world_vx * cos_y + world_vy * sin_y
    body_right = -world_vx * sin_y + world_vy * cos_y
    x2 = _clamp(body_right / max_h_speed, -1.0, 1.0) if max_h_speed > 0 else 0.0
    y2 = _clamp(body_forward / max_h_speed, -1.0, 1.0) if max_h_speed > 0 else 0.0

    return x1, y1, x2, y2, yaw_rate, vz, world_vx, world_vy


def apply_sticks(state: DroneState, x1, y1, x2, y2, yaw_rate, vz, world_vx, world_vy, dt=DT):
    """Integrate state forward using the SAME derived values move_towards() returned -- keeps
    recorded actions and actual resulting motion exactly consistent (a lesson learned the hard
    way in Simulations_1_0_0's pre/post-integration position bug: never let two different
    computations of 'where does this step end up' disagree)."""
    state.yaw_deg = (state.yaw_deg + yaw_rate * dt) % 360.0
    state.x += world_vx * dt
    state.y += world_vy * dt
    state.z += vz * dt
    state.vx, state.vy, state.vz = world_vx, world_vy, vz
    state.yaw_rate = yaw_rate  # persisted so next frame's move_towards() can rate-limit against it


def simulate_mission(home_xyz, target_xyz, rng=None, max_seconds=90.0,
                      orbit_radius=None, height_above_target=None, land_xyz=None,
                      orbit_speed_deg=None):
    """Runs one full takeoff -> approach -> orbit -> return -> land mission, recording every
    frame's stick values, pose, velocity, phase, and distance-to-target (the extra data point the
    original spec called out by name) as it happens -- not derived afterward from a static path.

    orbit_radius/height_above_target default to the module constants (tuned for car/hydrant-scale
    targets); pass larger values for bigger structures -- e.g. Simulations_0_0_3's bus_station
    shelter, whose roof/support poles extend further than the default 6m radius clears safely.

    Returns (records, meta) where records is a list of per-frame dicts matching Simulations_0_0_1's
    action.json schema plus a distance_to_target_m field, and meta carries the summary fields for
    that project's meta.json (home/target/start/final pose, fps, num_frames, duration_sec).
    """
    orbit_radius = ORBIT_RADIUS if orbit_radius is None else orbit_radius
    orbit_speed_deg = ORBIT_SPEED_DEG if orbit_speed_deg is None else orbit_speed_deg
    height_above_target = HEIGHT_ABOVE_TARGET if height_above_target is None else height_above_target
    hx, hy, hz = home_xyz
    tx, ty, tz = target_xyz
    # land_xyz defaults to the start pad (the classic there-and-back mission). Simulations_0_0_4
    # passes a DIFFERENT pad so the flight ends somewhere else, as a real delivery/transit sortie
    # would. Its z is the GROUND height of that pad -- see the note on hz below.
    lx, ly, lz = land_xyz if land_xyz is not None else (hx, hy, hz)
    lands_elsewhere = math.hypot(lx - hx, ly - hy) > 1.0
    FLIGHT_ALTITUDE = tz + height_above_target  # target-relative, see module docstring
    # Start ON THE GROUND at the start pad. This used to be hardcoded z=0.0, which silently assumed
    # every pad sits at exactly world z=0; on sloped terrain that starts/ends the drone under or
    # above the surface (the 0_0_4 end pad's ground is at +1.33m, so z=0 would bury it).
    state = DroneState(x=hx, y=hy, z=hz, yaw_deg=0.0)
    records = []
    frame = 0
    max_frames = int(max_seconds * FPS)

    def log(phase, x1, y1, x2, y2):
        dist = math.sqrt((state.x - tx) ** 2 + (state.y - ty) ** 2 + (state.z - tz) ** 2)
        records.append({
            "frame": frame, "t": round(frame * DT, 4), "phase": phase,
            "x1": round(x1, 4), "y1": round(y1, 4), "x2": round(x2, 4), "y2": round(y2, 4),
            "pose": {"x": round(state.x, 4), "y": round(state.y, 4), "z": round(state.z, 4),
                     "yaw_deg": round(state.yaw_deg, 3)},
            "velocity": {"vx": round(state.vx, 4), "vy": round(state.vy, 4), "vz": round(state.vz, 4)},
            "distance_to_target_m": round(dist, 4),
        })

    # ---- phase 1: takeoff (ascend at home position to cruise altitude) ----
    while state.z < FLIGHT_ALTITUDE - GOAL_TOLERANCE_M and frame < max_frames:
        x1, y1, x2, y2, yaw_rate, vz, wvx, wvy = move_towards(
            state, (hx, hy, FLIGHT_ALTITUDE), target_yaw_deg=state.yaw_deg)
        apply_sticks(state, x1, y1, x2, y2, yaw_rate, vz, wvx, wvy)
        log("takeoff", x1, y1, x2, y2)
        frame += 1

    # ---- phase 2: approach (fly to a point orbit_radius from the target, facing it) ----
    approach_yaw = math.degrees(math.atan2(ty - hy, tx - hx))
    approach_point = (tx - orbit_radius * math.cos(math.radians(approach_yaw)),
                       ty - orbit_radius * math.sin(math.radians(approach_yaw)),
                       FLIGHT_ALTITUDE)
    while frame < max_frames:
        remaining = math.hypot(approach_point[0] - state.x, approach_point[1] - state.y)
        if remaining < GOAL_TOLERANCE_M:
            break
        target_yaw = math.degrees(math.atan2(ty - state.y, tx - state.x))
        x1, y1, x2, y2, yaw_rate, vz, wvx, wvy = move_towards(state, approach_point, target_yaw_deg=target_yaw)
        apply_sticks(state, x1, y1, x2, y2, yaw_rate, vz, wvx, wvy)
        log("approach", x1, y1, x2, y2)
        frame += 1

    # ---- phase 3: orbit (circle the target at close range, always facing it) ----
    start_angle = math.degrees(math.atan2(state.y - ty, state.x - tx))
    swept = 0.0
    while swept < ORBIT_SWEEP_DEG and frame < max_frames:
        swept += orbit_speed_deg * DT
        angle = math.radians(start_angle + swept)
        wp = (tx + orbit_radius * math.cos(angle), ty + orbit_radius * math.sin(angle), FLIGHT_ALTITUDE)
        target_yaw = math.degrees(math.atan2(ty - state.y, tx - state.x))
        x1, y1, x2, y2, yaw_rate, vz, wvx, wvy = move_towards(state, wp, target_yaw_deg=target_yaw,
                                                                max_h_speed=min(MAX_HORIZONTAL_SPEED,
                                                                                orbit_radius * math.radians(orbit_speed_deg) * 1.35))
        apply_sticks(state, x1, y1, x2, y2, yaw_rate, vz, wvx, wvy)
        log("orbit", x1, y1, x2, y2)
        frame += 1

    # ---- phase 4: fly to the landing pad (cruise altitude) ----
    # phase label is "transit" when the landing pad differs from the start pad, "return" when it's
    # the classic there-and-back mission -- so 0_0_2/0_0_3's recorded semantics stay unchanged.
    egress_phase = "transit" if lands_elsewhere else "return"
    while frame < max_frames:
        remaining = math.hypot(lx - state.x, ly - state.y)
        if remaining < GOAL_TOLERANCE_M:
            break
        target_yaw = math.degrees(math.atan2(ly - state.y, lx - state.x))
        x1, y1, x2, y2, yaw_rate, vz, wvx, wvy = move_towards(state, (lx, ly, FLIGHT_ALTITUDE), target_yaw_deg=target_yaw)
        apply_sticks(state, x1, y1, x2, y2, yaw_rate, vz, wvx, wvy)
        log(egress_phase, x1, y1, x2, y2)
        frame += 1

    # ---- phase 5: land (descend to the GROUND HEIGHT of the landing pad, not to z=0) ----
    while state.z > lz + GOAL_TOLERANCE_M and frame < max_frames:
        x1, y1, x2, y2, yaw_rate, vz, wvx, wvy = move_towards(state, (lx, ly, lz), target_yaw_deg=state.yaw_deg)
        apply_sticks(state, x1, y1, x2, y2, yaw_rate, vz, wvx, wvy)
        log("land", x1, y1, x2, y2)
        frame += 1

    # ---- phase 6: stop (final hold frame) ----
    log("stop", 0.0, 0.0, 0.0, 0.0)
    frame += 1

    meta = {
        "home_position": {"x": hx, "y": hy, "z": hz},
        "land_position": {"x": lx, "y": ly, "z": lz},
        "lands_elsewhere": lands_elsewhere,
        "target_position": {"x": tx, "y": ty, "z": tz},
        "start_pose": records[0]["pose"],
        "final_pose": records[-1]["pose"],
        "fps": FPS,
        "num_frames": len(records),
        "duration_sec": round(len(records) * DT, 3),
        "sticks_legend": STICKS_LEGEND,
        "limits": {
            "max_horizontal_speed_mps": MAX_HORIZONTAL_SPEED,
            "max_vertical_speed_mps": MAX_VERTICAL_SPEED,
            "max_yaw_rate_deg": MAX_YAW_RATE_DEG,
            "orbit_radius_m": orbit_radius,
            "orbit_speed_deg": orbit_speed_deg,
            "max_accel_h_mps2": MAX_ACCEL_H,
            "max_accel_v_mps2": MAX_ACCEL_V,
        },
        "hit_max_frames": frame >= max_frames,
    }
    return records, meta


if __name__ == "__main__":
    recs, meta = simulate_mission((0.0, 0.0, 0.0), (25.0, 15.0, 1.0))
    print(f"{meta['num_frames']} frames, {meta['duration_sec']}s, hit_max_frames={meta['hit_max_frames']}")
    phases = {}
    for r in recs:
        phases[r["phase"]] = phases.get(r["phase"], 0) + 1
    print("phase frame counts:", phases)
    print("min distance to target:", min(r["distance_to_target_m"] for r in recs))
    print("final pose:", meta["final_pose"], "home:", meta["home_position"])
