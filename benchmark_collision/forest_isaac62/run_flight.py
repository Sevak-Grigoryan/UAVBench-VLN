"""
FLIGHT with collision-aware termination (Benchmark_collision): a copy of the dataset flight runner
(dataset_codebase/common/pipeline/run_x500_mission.py, which stays unchanged and keeps building the dataset) plus

  * PhysX contact monitoring of every X500 body, judged after EVERY 240 Hz physics step (contact_monitor.py);
  * rotor-disc and Forest vegetation collision proxies, CCD (collision_proxies.py);
  * termination right after the physics step of the first prohibited contact (no wait for the next frame / decision), then
    a clean PX4 shutdown (force-disarm, PX4 killed only after it disarmed or 3 s of sim time);
  * per-frame telemetry with the physics-step index, the phase and the full attitude quaternion, plus a terminal record at
    the exact termination step;
  * collision.json (monitor configuration, coverage, every contact pair, first prohibited / invalid / allowed contact,
    termination) and proxies.json.

    python run_flight.py --env Forest --scene easy_v2_037 --mission vlncol_forest_037 --out <work>/flight --vehicle-id 0
    python run_flight.py --env Forest --scene easy_v2_037 --mission colltest_wall      --out <work>/flight --vehicle-id 1

Only Forest is supported for collision-aware evaluation: its trees, rocks and fallen wood get proxies here. City, Coastal and
Mountain leave tree / prop groups without colliders in the dataset runner and are refused (see docs/collision_and_sim2real.md).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.dont_write_bytecode = True
sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent

def _isaac_version():
    import importlib.metadata as md
    return md.version("isaacsim")

CB = Path(__file__).resolve().parent / "dataset_port"
COMMON = CB / "common"
sys.path.insert(0, str(COMMON / "pipeline"))
sys.path.insert(0, str(HERE))
import envs  # noqa: E402

FPS = 30
PHYSICS_HZ = 240
SUBSTEPS = PHYSICS_HZ // FPS
SUPPORTED_ENVS = ("Forest",)
SHUTDOWN_MAX_S = 3.0
DRONE = "/World/x500"


def log(*a):
    print("[x500]", *a, flush=True)


ap = argparse.ArgumentParser()
ap.add_argument("--env", required=True, choices=envs.ENV_NAMES)
ap.add_argument("--scene", required=True, help="dataset mission whose scene is built, e.g. easy_v2_037")
ap.add_argument("--mission", required=True, help="vlncol_forest_<idx> (benchmark episode) or colltest_<name> (test)")
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--vehicle-id", type=int, default=0)
ap.add_argument("--max-sim-s", type=float, default=600.0)
args = ap.parse_args()
if args.env not in SUPPORTED_ENVS:
    sys.exit(f"{args.env}: not supported for collision-aware evaluation (vegetation / props without colliders); "
             f"supported: {SUPPORTED_ENVS}")

out = args.out
out.mkdir(parents=True, exist_ok=True)
for f in ("FLIGHT_OK", "FLIGHT_DONE"):
    (out / f).unlink(missing_ok=True)

# ---- environment -------------------------------------------------------------------------------
env = envs.boot(args.env, out.parent / f"kit_flight_{os.environ.get('SLURM_JOB_ID', 'local')}", mission_id=args.scene)
env.scene_mission = args.scene
bridge, ground, missions_mod = env.bridge, env.ground, env.missions_mod
LANDMARKS, env_info, intrinsics = env.landmarks, env.info, env.intrinsics
hooks = None
if args.mission.startswith("vlncol_forest_"):
    import episode_forest

    mission = episode_forest.build(int(args.mission[len("vlncol_forest_"):]), env)
    if episode_forest.scene_of(args.mission) != args.scene:
        sys.exit(f"{args.mission} belongs to scene {episode_forest.scene_of(args.mission)}, not {args.scene}")
elif args.mission.startswith("colltest_"):
    import scenarios

    mission, hooks = scenarios.build(args.mission[len("colltest_"):], env)
else:
    sys.exit(f"unknown mission {args.mission}")
log(f"env={args.env} scene={args.scene} mission={args.mission} -- flight only, collision-aware")

# ---- Pegasus + PX4 -----------------------------------------------------------------------------
import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

import isaac62_compat  # noqa: E402

compat_info = isaac62_compat.install()     # dynamic_control stand-in + Pegasus package (Isaac Sim 6.2 removed both paths)
log(f"[isaac62] {compat_info}")
bridge.app.update()

from isaacsim.core.api.world import World  # noqa: E402
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface  # noqa: E402
from pymavlink import mavutil  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

import collision_proxies as cp  # noqa: E402
import contact_monitor as cmon  # noqa: E402
import pilot_ext as pilot_mod  # noqa: E402
import x500  # noqa: E402

stage = bridge.stage
pg = PegasusInterface()
pg.set_world_settings(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / FPS, stage_units_in_meters=1.0)
pg._world = World(**pg._world_settings)
world = pg.world
physics_ctx = world.get_physics_context()
log(f"physics: gpu_dynamics={physics_ctx.is_gpu_dynamics_enabled()}")

# Static colliders: same pass as the dataset runner (everything solid except instanced scatter and paint)
EXCLUDED = ("/World/Nature", "/World/DenseNature", "/World/EgoCamera", "/World/Looks", "/World/Materials", "/World/River",
            "/World/Far", "/World/City/Marks", "/World/City/Zebra", "/World/City/Props", "/World/City/Trees",
            "/World/City/ParkTrees", "/World/Ocean", "/World/DistantTerrain")
n_col = 0
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
    if prim.IsInstanceProxy() or prim.IsA(UsdGeom.PointInstancer):
        continue
    path = str(prim.GetPath())
    if "/Prototypes/" in path or any(s in path for s in EXCLUDED):
        continue
    if prim.HasAttribute("x500:noCollision"):
        continue
    if prim.IsA(UsdGeom.Gprim) and not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        n_col += 1
log(f"static colliders added: {n_col}")
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
RIGID = getattr(env, "rigid_bodies", []) or []
if RIGID:
    if getattr(env, "make_dynamic", None) is not None:
        n_dyn = env.make_dynamic(stage, RIGID)
    else:
        import easy_v2

        n_dyn = easy_v2.make_dynamic(stage, RIGID)
    log(f"movable props as PhysX rigid bodies under gravity: {n_dyn}")
proxies = cp.add_vegetation_proxies(stage, log=log)
coverage = cp.audit(stage, EXCLUDED)


def rigid_poses():
    dc = drone.get_dc_interface()
    res = {}
    for r in RIGID:
        b = dc.get_rigid_body(r["path"])
        if not b:
            continue
        p = dc.get_rigid_body_pose(b)
        res[r["name"]] = ([float(p.p[0]), float(p.p[1]), float(p.p[2])], [float(v) for v in p.r])
    return res


sx, sy = mission["start_xy"]
start_ground = ground(sx, sy)
yaw0 = math.radians(mission["yaw_deg"])
q = Rotation.from_euler("z", yaw0).as_quat()
cfg = x500.X500Config(vehicle_id=args.vehicle_id)
drone = x500.X500(DRONE, vehicle_id=args.vehicle_id, init_pos=(sx, sy, start_ground + x500.SPAWN_Z + 0.02),
                  init_orientation=tuple(q), config=cfg)
rotor_discs = cp.add_rotor_discs(stage, DRONE)
monitor = cmon.ContactMonitor(stage, DRONE, landing=mission.get("landing"), log=log)
ccd = cp.enable_ccd(stage, DRONE)
log(f"contact monitor on {monitor.bodies}; colliders {monitor.colliders}; ccd {ccd}")
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


def apply_body_force(f):
    """World-frame force (N) at the body origin for the next physics step (omni.physx IPhysx.apply_force_at_pos)."""
    import carb
    import omni.usd
    from omni.physx import get_physx_interface
    from pxr import PhysicsSchemaTools

    p = drone.state.position
    get_physx_interface().apply_force_at_pos(omni.usd.get_context().get_stage_id(),
                                             PhysicsSchemaTools.sdfPathToInt(f"{DRONE}/body"),
                                             carb.Float3(*[float(v) for v in f]), carb.Float3(float(p[0]), float(p[1]), float(p[2])),
                                             "Force")


def drone_record(t, frame, stage_label, extra=None):
    st = drone.state
    pos = [float(v) for v in st.position]
    rot = Rotation.from_quat(st.attitude)
    yaw, pitch_b, roll_b = [float(v) for v in rot.as_euler("ZYX")]
    r = {"frame": frame, "t": round(t, 6), "physics_step": gstep, "phase": phase, "stage": stage_label,
         "pose": {"x": round(pos[0], 4), "y": round(pos[1], 4), "z": round(pos[2], 4), "roll_deg": round(math.degrees(roll_b), 3),
                  "pitch_deg": round(math.degrees(pitch_b), 3), "yaw_deg": round(math.degrees(yaw), 3)},
         "attitude_quat_xyzw": [round(float(v), 6) for v in st.attitude],
         "velocity": {"vx": round(float(st.linear_velocity[0]), 4), "vy": round(float(st.linear_velocity[1]), 4),
                      "vz": round(float(st.linear_velocity[2]), 4)},
         "agl_m": round(pos[2] - x500.SPAWN_Z - ground(pos[0], pos[1]), 3)}
    cx, cy, cz, cyaw, cpitch = envs.camera_pose(pos, yaw, gimbal)
    r["camera"] = {"x": round(cx, 4), "y": round(cy, 4), "z": round(cz, 4), "yaw_deg": round(cyaw, 3), "pitch_deg": round(cpitch, 3)}
    if extra:
        r.update(extra)
    return r


# ---- fly ---------------------------------------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
timeline.play()
pilot = pilot_mod.Pilot(mission["beats"], ground)

actions, telemetry = [], []
rec_frame, rec_t0 = 0, None
step_i = 0
gstep = 0                    # physics steps since the timeline started (240 Hz)
rec_step0 = None             # gstep at the start of recording
last_arm_try = -10.0
outcome = "timeout"
done_at = None
max_tilt = 0.0
max_heading_err = 0.0
hb_sim_t = None
wall0 = time.time()
init_done = False
phase = "initialization"
phase_log = [{"phase": "initialization", "step": 0, "t": None}]
terminated = None            # {"reason", "step", "t", "record"}
stage_label = "boot"
gimbal = pilot_mod.base.GIMBAL_GROUND_DEG


def t_of(step):
    return (step - rec_step0) / PHYSICS_HZ if rec_step0 is not None else None


def set_phase(p):
    global phase
    if p != phase:
        phase = p
        phase_log.append({"phase": p, "step": gstep, "t": None if rec_step0 is None else round(t_of(gstep), 6)})
        log(f"phase -> {p} at step {gstep}")


def current_phase():
    if not init_done:
        return "initialization"
    if pilot.done:
        return "post-roll"
    return pilot.beat.get("phase", "episode")


import faulthandler  # noqa: E402
while bridge.app.is_running():
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(180, repeat=True)
    for _ in range(SUBSTEPS):
        world.step(render=False)
        gstep += 1
        if gstep == 3:                    # Isaac Sim 6.2 drops the Pegasus vehicle callbacks after 2 calls: re-subscribe them
            isaac62_compat.revive_physics_callbacks(world, log=log)
        touching = monitor.drain(gstep, t_of(gstep), {"phase": phase, "beat": pilot.beat, "beat_index": pilot.i})
        if not init_done and rec_t0 is not None and not touching and float(drone.state.position[2]) > start_ground + x500.SPAWN_Z + 0.03:
            init_done = True              # lift-off: the initialization phase (resting on the take-off clearing) is over
            set_phase(current_phase())
        if monitor.latched is not None or monitor.invalid is not None:
            r = monitor.latched or monitor.invalid
            terminated = {"reason": "prohibited contact" if monitor.latched else r["verdict"], "step": gstep,
                          "t": t_of(gstep), "contact_step": r["step"]}
            break
        if hooks is not None and rec_t0 is not None:
            hooks.on_step(SimpleNamespace(t=t_of(gstep), phase=phase, apply_body_force=apply_body_force,
                                          yaw=float(Rotation.from_quat(drone.state.attitude).as_euler("ZYX")[0])))
    step_i += 1
    sim_t = step_i / FPS
    pump()
    if step_i % FPS == 0:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    if terminated is not None:
        rec = drone_record(terminated["t"], rec_frame, stage_label,
                           {"terminal": True, "termination": terminated["reason"],
                            "px4": {"armed": px4["armed"], "mode": px4["mode"], "landed_state": px4["landed"]}})
        terminated["record"] = rec
        telemetry.append(rec)
        outcome = "collision" if monitor.latched else f"invalid_{terminated['reason']}"
        log(f"TERMINATED at physics step {gstep} (t={terminated['t']:.4f}s, frame {rec_frame}): {terminated['reason']}")
        break

    st = drone.state
    pos = [float(v) for v in st.position]
    rot = Rotation.from_quat(st.attitude)
    yaw, pitch_b, roll_b = [float(v) for v in rot.as_euler("ZYX")]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, rot.apply([0, 0, 1])[2]))))

    if rec_t0 is None:
        send_sticks(0.0, -1.0, 0.0, 0.0)
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
            rec_step0 = gstep
            rigid_start = rigid_poses() if RIGID else {}
            log(f"armed in POSCTL at sim {sim_t:.2f}s (physics step {gstep}) -- recording starts")
        elif sim_t - hb_sim_t > 120.0:
            outcome = "px4_never_armed"
            break
        continue

    t = sim_t - rec_t0
    s = dict(x=pos[0], y=pos[1], z=pos[2], yaw=yaw, vx=float(st.linear_velocity[0]), vy=float(st.linear_velocity[1]),
             vz=float(st.linear_velocity[2]), landed=px4["landed"] == "ON_GROUND", armed=px4["armed"])
    if hooks is not None:
        hooks.on_frame(SimpleNamespace(t=t, phase=phase, pos=pos, yaw=yaw, stage=stage, pilot=pilot, monitor=monitor,
                                       proxies=proxies["instances"], log=log, ground=ground))
    sticks, gimbal, stage_label = pilot.update(t, s, 1.0 / FPS)
    send_sticks(*sticks)
    if init_done:
        set_phase(current_phase())
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
    telemetry.append({"frame": rec_frame, "t": round(t, 4), "physics_step": gstep, "phase": phase, "stage": stage_label,
                      "action_index": len(actions) - 1, "pose": pose,
                      "attitude_quat_xyzw": [round(float(v), 6) for v in st.attitude],
                      "velocity": {"vx": round(s["vx"], 4), "vy": round(s["vy"], 4), "vz": round(s["vz"], 4)},
                      "agl_m": round(pos[2] - x500.SPAWN_Z - ground(pos[0], pos[1]), 3),
                      "camera": cam, "gimbal_pitch_deg": round(gimbal, 3),
                      "sticks": list(sticks), "contacts_touching": monitor.touching_now,
                      "rotor_speed_rad_s": [round(float(v), 1) for v in drone._thrusters._velocity],
                      "px4": {"armed": px4["armed"], "mode": px4["mode"], "landed_state": px4["landed"]}})
    if rec_frame % 150 == 0:
        log(f"t={t:7.2f} frame={rec_frame} phase={phase:14s} stage={stage_label:18s} pos=({pos[0]:7.1f},{pos[1]:7.1f},{pos[2]:6.1f}) "
            f"sticks={sticks} armed={px4['armed']} landed={px4['landed']} touching={monitor.touching_now} "
            f"wall={time.time() - wall0:.0f}s")
    rec_frame += 1

    if rec_frame == 300 and physics_steps[0] < rec_frame * SUBSTEPS:
        outcome = "physics_not_stepping"
        break
    if px4["yaw_ned"] is not None:
        heading_err = math.degrees(abs(((math.pi / 2 - yaw) - px4["yaw_ned"] + math.pi) % (2 * math.pi) - math.pi))
        max_heading_err = max(max_heading_err, heading_err)
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

# ---- clean PX4 shutdown after a termination: force-disarm, keep PX4 in lockstep until it reports disarmed ---------------
shutdown = None
if terminated is not None:
    set_phase("post-termination")
    t_sd = time.time()
    n_sd, sent = 0, -1.0
    while n_sd < int(SHUTDOWN_MAX_S * PHYSICS_HZ):
        if n_sd / PHYSICS_HZ - sent >= 0.25:
            # MAV_CMD_COMPONENT_ARM_DISARM param1 = 0 (disarm), param2 = 21196 (force, also in the air)
            mav.mav.command_long_send(px4["sysid"] or 1, 1, mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 0, 21196,
                                      0, 0, 0, 0, 0)
            sent = n_sd / PHYSICS_HZ
        world.step(render=False)
        gstep += 1
        n_sd += 1
        monitor.drain(gstep, t_of(gstep), {"phase": phase, "beat": pilot.beat, "beat_index": pilot.i})
        if n_sd % SUBSTEPS == 0:
            pump()
            if not px4["armed"]:
                break
    shutdown = {"force_disarm_sent": True, "px4_disarmed": not px4["armed"], "sim_s_after_termination": round(n_sd / PHYSICS_HZ, 3),
                "wall_s": round(time.time() - t_sd, 1),
                "note": "physics kept stepping only so PX4 (lockstep) could process the disarm and close its log; "
                        "nothing after the termination step is recorded or evaluated"}
    log(f"PX4 shutdown: {shutdown}")

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
                     "moved_during_flight_m": round(math.dist(p0, p1), 4),
                     "rotated_during_flight_deg": round(math.degrees(2 * math.acos(min(1.0, dot))), 3)})
    ok = [x for x in rows if "error" not in x]
    physics_check = {"bodies": len(RIGID), "measured": len(ok),
                     "max_moved_during_flight_m": max((x["moved_during_flight_m"] for x in ok), default=None),
                     "all_at_rest": bool(ok) and all(x["moved_during_flight_m"] < 0.05 and x["rotated_during_flight_deg"] < 2.0 for x in ok)}

ulog = None
tool = getattr(drone._backends[0], "px4_tool", None)
if tool is not None:
    logs = sorted(Path(tool.root_fs.name).rglob("*.ulg"))
    if logs:
        ulog = out / "px4.ulg"
        shutil.copy(logs[-1], ulog)

# ---- outputs -----------------------------------------------------------------------------------
land_ev = next((e for e in pilot.events if e["type"] == "land"), None)
ep_phase = [p for p in phase_log if p["phase"] == "episode"]
summary = monitor.summary(step_to_t=lambda s: (s - rec_step0) / PHYSICS_HZ if rec_step0 is not None else float("nan"))
collision = {
    **summary,
    "mission": args.mission, "scene": args.scene, "environment": args.env,
    "physics": {"hz": PHYSICS_HZ, "substeps_per_frame": SUBSTEPS, "gpu_dynamics": physics_ctx.is_gpu_dynamics_enabled(),
                "ccd": ccd, "judged": "after every physics step, outside the PhysX callback"},
    "uav_coverage": {"rotor_disc_proxies": rotor_discs, "rotor_disc_radius_m": cp.ROTOR_DISC_RADIUS_M,
                     "rotor_disc_height_m": cp.ROTOR_DISC_HEIGHT_M},
    "environment_coverage": {"excluded_by_dataset_runner": list(EXCLUDED), "audit": coverage,
                             "vegetation_proxies": {k: v for k, v in proxies.items() if k != "instances"},
                             "proxy_file": "proxies.json"},
    "phases": phase_log,
    "episode_start": ep_phase[0] if ep_phase else None,
    "recording_start_step": rec_step0,
    "termination": terminated, "px4_shutdown": shutdown,
    "outcome": outcome,
    "test": mission.get("test"), "test_notes": getattr(hooks, "notes", None),
}
(out / "collision.json").write_text(json.dumps(collision, indent=1))
(out / "proxies.json").write_text(json.dumps(proxies["instances"]))
meta = {
    "simulation_id": f"x500_{args.env}_{args.mission}",
    "mission": {"id": args.mission, "scene": args.scene, "start_pad": mission["start"], "start_xy": list(mission["start_xy"]),
                "start_heading_deg": mission["yaw_deg"], "beats": mission["beats"], "bench": mission.get("bench"),
                "landing": mission.get("landing"), "instruction": mission.get("instruction"),
                "landing_pad": land_ev and next((b["pad"] for b in mission["beats"] if b["type"] == "land"), None)},
    "environment": env_info, "drone": {**x500.KIT, "physics": x500.physics_summary()},
    "camera_intrinsics": intrinsics, "frame_size": list(envs.FRAME_SIZE), "fps": FPS, "physics_hz": PHYSICS_HZ,
    "frames": rec_frame, "duration_s": round(rec_frame / FPS, 3), "actions": len(actions), "events": pilot.events,
    "verification": {
        "outcome": outcome,
        "landed_and_disarmed": bool(land_ev) and not px4["armed"],
        "touchdown_error_m": land_ev and land_ev.get("touchdown_error_m"),
        "max_tilt_deg": round(max_tilt, 2),
        "max_px4_heading_error_deg": round(max_heading_err, 2),
        "collision_monitor_verified": summary["monitor_verified"],
        "first_prohibited_contact": summary["first_prohibited"] and {k: summary["first_prohibited"][k] for k in (
            "step", "t", "phase", "drone_part", "drone_collider", "other_collider")},
        "terminated": terminated and {k: v for k, v in terminated.items() if k != "record"},
    },
    "props_physics_check": physics_check,
    "flight_wall_time_s": round(time.time() - wall0, 1),
    "simulator": f"NVIDIA Isaac Sim {_isaac_version()} + Pegasus Simulator dev_6.0.1 branch (+ PhysX contact monitor, Benchmark_collision, Isaac Sim 6.2 port)",
}
(out / "action.json").write_text(json.dumps({"actions": actions}, indent=1))
(out / "telemetry.json").write_text(json.dumps(telemetry))
(out / "meta.json").write_text(json.dumps(meta, indent=1))
(out / "statustext.json").write_text(json.dumps(statustext))
log("RESULT", json.dumps({"outcome": outcome, "frames": rec_frame, **{k: v for k, v in meta["verification"].items()}}))
(out / "FLIGHT_DONE").write_text(outcome)
if outcome == "completed" and meta["verification"]["landed_and_disarmed"] and summary["monitor_verified"] and terminated is None:
    (out / "FLIGHT_OK").write_text(str(rec_frame))
    log(f"FLIGHT OK: {rec_frame} frames -> {out}")
else:
    log(f"FLIGHT ENDED (outcome={outcome}); outputs kept in {out}")

monitor.close()
if tool is not None:
    try:
        tool.kill_px4()
    except Exception as e:
        log(f"kill_px4 failed: {e}")
timeline.stop()
bridge.shutdown()
