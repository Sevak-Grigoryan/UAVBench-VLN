"""
FLIGHT phase of one Holybro X500 V2 mission: real PX4 v1.14.3 SITL (Pegasus) flies the X500 in the
Isaac Sim environment with physics only -- nothing is rendered here. Every frame's camera pose is
recorded; render_slice.py renders those poses later, many slices in parallel, and finalize.py stitches
the video and publishes the sample.

    python run_x500_mission.py --env Forest --mission easy_footbridge --out <run>/flight --vehicle-id 0

Loop, per 30 fps video frame:
    8 x PhysX step @ 240 Hz      Pegasus applies X500 rotor thrust/drag, streams HIL_SENSOR to PX4,
                                 PX4 (lockstep) answers with motor commands
    pilot.update()               stick decision (held / quantised, see pilot.py) -> MANUAL_CONTROL
    gimbal camera pose           front-mounted, yaw-locked to the airframe, stabilised pitch -> recorded

Why the split: rendering inside this loop (pause timeline, 16 path-tracing subframes, resume) made a
9000-frame mission take 3.5 h on one GPU and repeatedly froze the Pegasus/PX4 lockstep at the first
rendered frame (forest jobs 319314/16/21/22). Flying alone takes minutes; the static scene renders
identically from the recorded poses.

Writes to --out: action.json, telemetry.json, meta.json, px4.ulg, statustext.json, and FLIGHT_OK
only if the mission completed and the X500 landed and disarmed.
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(line_buffering=True)

COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON / "pipeline"))
import envs

FPS = 30
PHYSICS_HZ = 240
SUBSTEPS = PHYSICS_HZ // FPS


def log(*a):
    print("[x500]", *a, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--env", required=True, choices=envs.ENV_NAMES)
ap.add_argument("--mission", required=True)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--vehicle-id", type=int, default=0)
ap.add_argument("--max-sim-s", type=float, default=600.0)
args = ap.parse_args()

out = args.out
out.mkdir(parents=True, exist_ok=True)
(out / "FLIGHT_OK").unlink(missing_ok=True)

# ---- environment (boots SimulationApp through the env's own HQ bridge) -------------------------
env = envs.boot(args.env, out.parent / f"kit_flight_{os.environ.get('SLURM_JOB_ID', 'local')}", mission_id=args.mission)
bridge, ground, missions_mod = env.bridge, env.ground, env.missions_mod
MISSIONS, LANDMARKS, env_info, intrinsics = env.missions, env.landmarks, env.info, env.intrinsics
mission = MISSIONS[args.mission]
log(f"env={args.env} mission={args.mission} ({mission['difficulty']}) -- flight only, no rendering")

# ---- Pegasus + PX4 -----------------------------------------------------------------------------
import numpy as np
import omni.kit.app
import omni.timeline
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ext = omni.kit.app.get_app().get_extension_manager()
ext.add_path(str(COMMON / "px4_env/PegasusSimulator/extensions"))
ext.set_extension_enabled_immediate("pegasus.simulator", True)
bridge.app.update()

from isaacsim.core.api.world import World
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pymavlink import mavutil
from scipy.spatial.transform import Rotation

import pilot as pilot_mod
import x500

stage = bridge.stage
pg = PegasusInterface()
pg.set_world_settings(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / FPS, stage_units_in_meters=1.0)
pg._world = World(**pg._world_settings)
world = pg.world
# NOTE: switching PhysX to CPU here (enable_gpu_dynamics(False) + MBP) froze the Pegasus/PX4 lockstep
# at the first rendered frame in all four forest jobs 319314/16/21/22; the default GPU solver is what
# flew job 318542 end to end, so the default is kept.
physics_ctx = world.get_physics_context()
log(f"physics: gpu_dynamics={physics_ctx.is_gpu_dynamics_enabled()}")

# Static colliders: everything solid the X500 could touch (terrain already has one). Vegetation is
# instanced visual-only scatter and stays non-colliding; missions keep >= 13 m above the canopy.
n_col = 0
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
    if prim.IsInstanceProxy() or prim.IsA(UsdGeom.PointInstancer):
        continue
    path = str(prim.GetPath())
    # instanced street furniture / trees: PhysX instantiates prototype colliders at EVERY instance, so a street
    # tree's canopy over the parking lane shoved parked trucks 4 m and pedestrians up to 6.5 m (City 002-005)
    if "/Prototypes/" in path or any(s in path for s in (
            "/Nature", "/DenseNature", "/EgoCamera", "/Looks", "/Materials", "/World/River", "/World/Far",
            "/World/City/Marks", "/World/City/Zebra", "/World/City/Props", "/World/City/Trees", "/World/City/ParkTrees",
            "/World/Ocean", "/World/DistantTerrain")):
        continue  # /World/River: visual corridor over the (colliding) coarse terrain, 270k triangles
    if prim.HasAttribute("x500:noCollision"):
        continue  # decorative prop detail
    if prim.IsA(UsdGeom.Gprim) and not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        n_col += 1
log(f"static colliders added: {n_col}")
# Painted road markings, zebra stripes and pad markings are paint, not solid: no collision (people standing on a
# zebra crossing were pushed up by 3 cm stripe "blocks", City flight 002).
n_paint = 0
for root in ("/World/City/Marks", "/World/City/Zebra", "/World/City/Pads", "/World/Coast/Pads", "/World/Mtn/Pads"):
    rp = stage.GetPrimAtPath(root)
    if not rp:
        continue
    for prim in Usd.PrimRange(rp):
        nm = prim.GetName()
        if prim.HasAPI(UsdPhysics.CollisionAPI) and (not root.endswith("/Pads") or nm.startswith(("ring", "H"))):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            n_paint += 1
if n_paint:
    log(f"collision disabled on {n_paint} painted markings")
RIGID = getattr(env, "rigid_bodies", []) or []
if RIGID:
    if getattr(env, "make_dynamic", None) is not None:
        n_dyn = env.make_dynamic(stage, RIGID)
    else:
        import easy_v2

        n_dyn = easy_v2.make_dynamic(stage, RIGID)
    log(f"movable props as PhysX rigid bodies under gravity: {n_dyn}")


def rigid_poses():
    dc = drone.get_dc_interface()
    out = {}
    for r in RIGID:
        b = dc.get_rigid_body(r["path"])
        if not b:
            continue
        p = dc.get_rigid_body_pose(b)
        out[r["name"]] = ([float(p.p[0]), float(p.p[1]), float(p.p[2])], [float(v) for v in p.r])
    return out

sx, sy = mission["start_xy"]
start_ground = ground(sx, sy)
yaw0 = math.radians(mission["yaw_deg"])
q = Rotation.from_euler("z", yaw0).as_quat()
cfg = x500.X500Config(vehicle_id=args.vehicle_id)
drone = x500.X500("/World/x500", vehicle_id=args.vehicle_id, init_pos=(sx, sy, start_ground + x500.SPAWN_Z + 0.02),
                  init_orientation=tuple(q), config=cfg)
world.reset()
physics_steps = [0]
world.add_physics_callback("x500_step_counter", lambda dt: physics_steps.__setitem__(0, physics_steps[0] + 1))

mav = mavutil.mavlink_connection(f"udpin:0.0.0.0:{14540 + args.vehicle_id}", source_system=255)
PX4_MAIN_MODE = {1: "MANUAL", 2: "ALTCTL", 3: "POSCTL", 4: "AUTO", 5: "ACRO", 6: "OFFBOARD", 7: "STABILIZED"}
px4 = {"armed": False, "mode": None, "landed": None, "hb": False, "yaw_ned": None, "sysid": None}
statustext = []


def pump():
    while True:
        m = mav.recv_match(blocking=False)
        if m is None:
            return
        t = m.get_type()
        if t == "HEARTBEAT" and m.type != mavutil.mavlink.MAV_TYPE_GCS:
            px4["hb"] = True
            # PX4 instance N runs as MAVLink system N+1; address it by the id it actually reports
            # (commands sent to system 1 were silently ignored by every instance but vehicle 0).
            px4["sysid"] = m.get_srcSystem()
            px4["armed"] = bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            px4["mode"] = PX4_MAIN_MODE.get((m.custom_mode >> 16) & 0xFF, str(m.custom_mode))
        elif t == "EXTENDED_SYS_STATE":
            px4["landed"] = {1: "ON_GROUND", 2: "IN_AIR", 3: "TAKEOFF", 4: "LANDING"}.get(m.landed_state, "UNDEFINED")
        elif t == "ATTITUDE":
            px4["yaw_ned"] = m.yaw
        elif t == "STATUSTEXT":
            statustext.append(m.text)
            log("PX4:", m.text)


def send_sticks(x1, y1, x2, y2):
    mav.mav.manual_control_send(px4["sysid"] or 1, int(1000 * y2), int(1000 * x2), int(500 + 500 * y1), int(1000 * x1), 0)


# ---- fly ---------------------------------------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
timeline.play()
pilot = pilot_mod.Pilot(mission["beats"], ground)

actions, telemetry = [], []
rec_frame, rec_t0 = 0, None
step_i = 0
last_arm_try = -10.0
outcome = "timeout"
done_at = None
max_tilt = 0.0
max_heading_err = 0.0
hb_sim_t = None
wall0 = time.time()

import faulthandler
while bridge.app.is_running():
    # watchdog: if one frame takes > 3 min, dump every thread's Python stack to stderr
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(180, repeat=True)
    for _ in range(SUBSTEPS):
        world.step(render=False)
    step_i += 1
    sim_t = step_i / FPS
    pump()
    if step_i % FPS == 0:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    st = drone.state
    pos = [float(v) for v in st.position]
    rot = Rotation.from_quat(st.attitude)
    yaw, pitch_b, roll_b = [float(v) for v in rot.as_euler("ZYX")]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, rot.apply([0, 0, 1])[2]))))

    if rec_t0 is None:  # boot: get PX4 into Position mode and armed, nothing recorded yet
        send_sticks(0.0, -1.0, 0.0, 0.0)
        # Without rendering, physics runs far faster than PX4 boots (~70 s wall). Don't let sim time
        # race ahead while PX4 is still starting, and time the arming wait from PX4's first heartbeat
        # (the 90 s sim-time limit fired before PX4 was even up and killed most forest flights).
        if not px4["hb"]:
            time.sleep(0.02)
            if time.time() - wall0 > 600:
                outcome = "px4_no_heartbeat"
                break
            continue
        if hb_sim_t is None:
            hb_sim_t = sim_t
        if px4["hb"] and sim_t - last_arm_try > 2.0:
            mav.mav.command_long_send(px4["sysid"], 1, mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                                      mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 3, 0, 0, 0, 0, 0)
            mav.mav.command_long_send(px4["sysid"], 1, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
            last_arm_try = sim_t
        if px4["armed"] and px4["mode"] == "POSCTL":
            rec_t0 = sim_t
            rigid_start = rigid_poses() if RIGID else {}
            log(f"armed in POSCTL at sim {sim_t:.2f}s -- recording starts")
        elif sim_t - hb_sim_t > 120.0:
            outcome = "px4_never_armed"
            break
        continue

    t = sim_t - rec_t0
    s = dict(x=pos[0], y=pos[1], z=pos[2], yaw=yaw, vx=float(st.linear_velocity[0]), vy=float(st.linear_velocity[1]),
             vz=float(st.linear_velocity[2]), landed=px4["landed"] == "ON_GROUND", armed=px4["armed"])
    sticks, gimbal, stage_label = pilot.update(t, s, 1.0 / FPS)
    send_sticks(*sticks)
    max_tilt = max(max_tilt, tilt)

    cx, cy, cz, cyaw, cpitch = envs.camera_pose(pos, yaw, gimbal)

    cam = {"x": round(cx, 4), "y": round(cy, 4), "z": round(cz, 4), "yaw_deg": round(cyaw, 3), "pitch_deg": round(cpitch, 3)}
    pose = {"x": round(pos[0], 4), "y": round(pos[1], 4), "z": round(pos[2], 4), "roll_deg": round(math.degrees(roll_b), 3),
            "pitch_deg": round(math.degrees(pitch_b), 3), "yaw_deg": round(math.degrees(yaw), 3)}
    if not actions or tuple(actions[-1][k] for k in ("x1", "y1", "x2", "y2")) != tuple(sticks):
        if actions:
            actions[-1]["hold_frames"] = rec_frame - actions[-1]["frame"]
            actions[-1]["hold_s"] = round(actions[-1]["hold_frames"] / FPS, 4)
        actions.append({"index": len(actions), "frame": rec_frame, "t": round(t, 4),
                        "x1": sticks[0], "y1": sticks[1], "x2": sticks[2], "y2": sticks[3],
                        "stage": stage_label, "pose": pose, "camera": cam})
    telemetry.append({"frame": rec_frame, "t": round(t, 4), "stage": stage_label, "action_index": len(actions) - 1,
                      "pose": pose,
                      "velocity": {"vx": round(s["vx"], 4), "vy": round(s["vy"], 4), "vz": round(s["vz"], 4)},
                      "agl_m": round(pos[2] - x500.SPAWN_Z - ground(pos[0], pos[1]), 3),
                      "camera": cam,
                      "rotor_speed_rad_s": [round(float(v), 1) for v in drone._thrusters._velocity],
                      "px4": {"armed": px4["armed"], "mode": px4["mode"], "landed_state": px4["landed"]}})
    if rec_frame % 150 == 0:
        log(f"t={t:7.2f} frame={rec_frame} stage={stage_label:18s} pos=({pos[0]:7.1f},{pos[1]:7.1f},{pos[2]:6.1f}) "
            f"sticks={sticks} gimbal={gimbal:4.1f} armed={px4['armed']} landed={px4['landed']} "
            f"physics_steps={physics_steps[0]} rotors={[round(float(v)) for v in drone._thrusters._velocity]} "
            f"wall={time.time() - wall0:.0f}s")
    rec_frame += 1

    if rec_frame == 300 and physics_steps[0] < rec_frame * SUBSTEPS:
        outcome = "physics_not_stepping"
        break
    if px4["yaw_ned"] is not None:
        heading_err = math.degrees(abs(((math.pi / 2 - yaw) - px4["yaw_ned"] + math.pi) % (2 * math.pi) - math.pi))
        max_heading_err = max(max_heading_err, heading_err)
        # ignore the first 3 s: at arming PX4's EKF heading is still converging (easy_v2_054 aborted at frame 1
        # with 16.8 deg); a real mismatch persists and is caught afterwards
        if heading_err > 15.0 and t > 3.0:
            outcome = "px4_heading_mismatch"
            log(f"ABORT: PX4 heading differs from true heading by {heading_err:.1f} deg")
            break
    if tilt > 70.0 and not s["landed"]:
        outcome = "crash_attitude"
        break
    if pilot.failed:
        outcome = "stuck"
        log(f"ABORT: {pilot.failed}")
        break
    if pilot.done and done_at is None:
        done_at = t
        outcome = "completed"
    if done_at is not None and t - done_at >= 1.0:
        break
    if t > args.max_sim_s:
        outcome = "timeout"
        break

if actions:
    actions[-1]["hold_frames"] = rec_frame - actions[-1]["frame"]
    actions[-1]["hold_s"] = round(actions[-1]["hold_frames"] / FPS, 4)

# ---- physics check of the movable props: simulated with gravity for the whole flight -------------
physics_check = None
if RIGID and rec_t0 is not None:
    end = rigid_poses()
    rows = []
    for r in RIGID:
        if r["name"] not in end or r["name"] not in rigid_start:
            rows.append({"name": r["name"], "error": "no PhysX body"})
            continue
        p1, q1 = end[r["name"]]
        p0, q0 = rigid_start[r["name"]]
        dot = abs(sum(a * b for a, b in zip(q0, q1))) / max(1e-9, math.sqrt(sum(a * a for a in q0) * sum(b * b for b in q1)))
        rows.append({"name": r["name"], "kind": r["kind"], "mass_kg": r["mass_kg"],
                     "settle_from_authored_m": round(math.dist(r["pos"], p0), 4),
                     "moved_during_flight_m": round(math.dist(p0, p1), 4),
                     "rotated_during_flight_deg": round(math.degrees(2 * math.acos(min(1.0, dot))), 3)})
    ok = [x for x in rows if "error" not in x]
    physics_check = {
        "method": "every movable prop is a PhysX rigid body with its real mass under gravity (9.81 m/s^2) on the "
                  "terrain collider for the whole flight; it must stay where it was placed",
        "bodies": len(RIGID), "measured": len(ok),
        "max_settle_from_authored_m": max((x["settle_from_authored_m"] for x in ok), default=None),
        "max_moved_during_flight_m": max((x["moved_during_flight_m"] for x in ok), default=None),
        "max_rotated_during_flight_deg": max((x["rotated_during_flight_deg"] for x in ok), default=None),
        "all_at_rest": bool(ok) and all(x["moved_during_flight_m"] < 0.05 and x["rotated_during_flight_deg"] < 2.0 for x in ok),
        "bodies_detail": rows,
    }
    log("PHYSICS CHECK", json.dumps({k: v for k, v in physics_check.items() if k != "bodies_detail"}))

# ---- PX4 flight log (kept in the project, not in the dataset) ----------------------------------
ulog = None
tool = getattr(drone._backends[0], "px4_tool", None)
if tool is not None:
    logs = sorted(Path(tool.root_fs.name).rglob("*.ulg"))
    if logs:
        ulog = out / "px4.ulg"
        shutil.copy(logs[-1], ulog)

# ---- meta --------------------------------------------------------------------------------------
text, steps = missions_mod.instruction(mission, LANDMARKS, ground)
land_ev = next((e for e in pilot.events if e["type"] == "land"), None)
visited = []
for name, lm in LANDMARKS.items():
    if any(b.get("landmark") == name for b in mission["beats"]):
        dmin = min((math.hypot(f["pose"]["x"] - lm["xy"][0], f["pose"]["y"] - lm["xy"][1]) for f in telemetry),
                   default=float("nan"))
        visited.append({"name": name, "description": lm["desc"], "position_xy": [round(v, 3) for v in lm["xy"]],
                        "top_z": round(lm["top_z"], 3), "closest_horizontal_approach_m": round(dmin, 2)})
airborne = [f["agl_m"] for f in telemetry if f["px4"]["landed_state"] == "IN_AIR" and f["stage"] not in ("take off",)
            and not f["stage"].startswith("land")]
meta = {
    "simulation_id": f"x500_{args.env}_{args.mission}",
    "difficulty": mission["difficulty"],
    "task_count": len(mission.get("tasks", [])),
    "tasks": mission.get("tasks", []),
    "difficulty_rule": getattr(missions_mod, "DIFFICULTY_RULE", None),
    "task_type": mission.get("task_type"),
    "environment": env_info,
    "instruction": text,
    "instruction_steps": steps,
    "mission": {"id": args.mission, "start_pad": mission["start"], "start_xy": list(mission["start_xy"]),
                "start_heading_deg": mission["yaw_deg"],
                "landing_pad": land_ev and next(b["pad"] for b in mission["beats"] if b["type"] == "land"),
                "beats": mission["beats"], "landmarks": visited},
    "drone": {**x500.KIT, "physics": x500.physics_summary(), "usd": "dataset_codebase/common/assets/x500/x500.usd"},
    "autopilot": {"firmware": "PX4 v1.14.3 SITL (lockstep)", "airframe": "10050_isaac_x500",
                  "flight_mode": "POSCTL (Position mode)", "control_input": "MAVLink MANUAL_CONTROL = the recorded sticks",
                  "stick_response": {"horizontal_mps_at_full": 8.0, "climb_mps_at_full": 3.0, "descent_mps_at_full": 2.0,
                                     "yaw_dps_at_full": 60.0, "expo": 0.0},
                  "flight_log": str(ulog) if ulog else None},
    "action_format": {"file": "action.json", "recorded": "only when a stick value changes",
                      "sticks": {"x1": "yaw, + = rotate clockwise (right)", "y1": "throttle, + = climb, - = descend",
                                 "x2": "roll, + = move right", "y2": "pitch, + = move forward"},
                      "range": [-1.0, 1.0], "quantisation": pilot_mod.STICK_STEP,
                      "min_decision_interval_s": pilot_mod.DECIDE_EVERY_S,
                      "hold": "each action holds until the next entry; hold_frames / hold_s give its duration"},
    "telemetry_format": {"file": "telemetry.json", "recorded": "every video frame (pose, velocity, camera, PX4 state)"},
    "camera_intrinsics": intrinsics,
    "frame_size": list(envs.FRAME_SIZE), "fps": FPS, "physics_hz": PHYSICS_HZ,
    "frames": rec_frame, "duration_s": round(rec_frame / FPS, 3), "actions": len(actions),
    "events": pilot.events,
    "verification": {
        "outcome": outcome,
        "landed_and_disarmed": bool(land_ev) and not px4["armed"],
        "touchdown_error_m": land_ev and land_ev.get("touchdown_error_m"),
        "max_tilt_deg": round(max_tilt, 2),
        "max_px4_heading_error_deg": round(max_heading_err, 2),
        "pegasus_lockstep_timeouts": getattr(drone._backends[0], "lockstep_timeouts", 0),
        "min_agl_in_transit_m": round(min(airborne), 2) if airborne else None,
        "canopy_height_max_m": 12.7,
    },
    "props_physics_check": physics_check,
    "flight_wall_time_s": round(time.time() - wall0, 1),
    "simulator": "NVIDIA Isaac Sim 5.1 + Pegasus Simulator 5.1.0",
}
(out / "action.json").write_text(json.dumps({"actions": actions}, indent=1))
(out / "telemetry.json").write_text(json.dumps(telemetry))
(out / "meta.json").write_text(json.dumps(meta, indent=1))
(out / "statustext.json").write_text(json.dumps(statustext))
log("RESULT", json.dumps({"outcome": outcome, "frames": rec_frame, "actions": len(actions), **meta["verification"]}))
if outcome == "completed" and meta["verification"]["landed_and_disarmed"]:
    (out / "FLIGHT_OK").write_text(str(rec_frame))
    log(f"FLIGHT OK: {rec_frame} frames -> {out}")
else:
    log(f"FLIGHT FAILED (outcome={outcome}); outputs left in {out}")

# Kill PX4 explicitly: an orphaned PX4 keeps the job's stdout open and makes a finished job look hung.
if tool is not None:
    try:
        tool.kill_px4()
    except Exception as e:
        log(f"kill_px4 failed: {e}")
timeline.stop()
bridge.shutdown()
