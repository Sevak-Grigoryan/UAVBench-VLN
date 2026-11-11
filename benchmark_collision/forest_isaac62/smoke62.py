"""
[ISAAC SIM 6.2 PORT, smoke62.py] loads Pegasus through isaac62_compat (dynamic_control stand-in) and also checks that PhysX
contact reports arrive (skids on the ground before take-off and after landing).

Smoke test: the Holybro X500 V2 flown by real PX4 v1.14.3 SITL inside Isaac Sim 5.1 via Pegasus.

Proves the whole chain end to end before any environment/mission work is built on it:
    x500.usd loads and renders  ->  Pegasus applies X500 thrust/drag  ->  PX4 (airframe
    10050_isaac_x500) boots in lockstep, gets a GPS/EKF fix, accepts MAVLink MANUAL_CONTROL
    sticks, arms in Position mode, flies, lands on its skids, auto-disarms.

The drone is commanded ONLY with sticks (MAVLink MANUAL_CONTROL, PX4 Position mode) -- the same
four channels (x1 yaw, y1 throttle, x2 roll, y2 pitch) every earlier project records in
action.json -- so the recorded actions are the real control input PX4 received, not a log
derived after the fact.

Flight plan (sim time after arming):
    climb 5 s -> hover 2 s -> forward 4 s -> yaw left 3 s -> hover 2 s -> descend until landed

Timing: physics 240 Hz, one rendered frame per 8 physics steps = 30 fps video. PX4 runs in
lockstep with physics through Pegasus' HIL_SENSOR stream, so slow rendering cannot desync it.

Outputs in --out: trajectory.mp4 (chase camera), action.json, meta.json, px4.ulg, stills/.
"""
import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

from isaacsim import SimulationApp

FRAME_SIZE = (1280, 720)
FPS = 30
PHYSICS_HZ = 240

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("--max-sim-s", type=float, default=150.0)
ap.add_argument("--hq-loop", action="store_true", help="step/render exactly like run_x500_mission.py")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

# multi_gpu False: on a shared node Kit otherwise renders across every GPU it can see, including other
# users' busy ones -- that is what produced the start-up ERROR_DEVICE_LOST / Vulkan segfaults.
app = SimulationApp({"headless": True, "width": FRAME_SIZE[0], "height": FRAME_SIZE[1], "renderer": "RayTracedLighting",
                     "multi_gpu": False})

import carb
import numpy as np
import omni.kit.app
import omni.replicator.core as rep
import omni.timeline
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade
from scipy.spatial.transform import Rotation

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "dataset_port/common/pipeline"
PROJECT = PIPELINE.parent
sys.path.insert(0, str(HERE))
import isaac62_compat

compat_info = isaac62_compat.install()
print("[smoke] compat", compat_info, flush=True)
app.update()

from isaacsim.core.api.world import World
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pymavlink import mavutil

sys.path.insert(0, str(PIPELINE))
import x500
from encode_video import VideoWriter


def log(*a):
    print("[smoke]", *a, flush=True)


# ---- world ------------------------------------------------------------------------------------
pg = PegasusInterface()
pg.set_world_settings(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / FPS, stage_units_in_meters=1.0)
pg._world = World(**pg._world_settings)
world = pg.world
stage = omni.usd.get_context().get_stage()

ground = UsdGeom.Cube.Define(stage, "/World/ground")
ground.CreateSizeAttr(1.0)
gx = UsdGeom.Xformable(ground)
gx.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.05))
gx.AddScaleOp().Set(Gf.Vec3f(200, 200, 0.1))
UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
gmat = UsdShade.Material.Define(stage, "/World/Looks/ground")
gsh = UsdShade.Shader.Define(stage, "/World/Looks/ground/Shader")
gsh.CreateIdAttr("UsdPreviewSurface")
gsh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.35, 0.42, 0.3))
gsh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
gmat.CreateSurfaceOutput().ConnectToSource(gsh.ConnectableAPI(), "surface")
UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(gmat)
# a few markers so forward motion and yaw read in the video
for i, (mx, my) in enumerate([(6, 0), (12, 2), (18, -2), (6, 6), (0, -6)]):
    c = UsdGeom.Cube.Define(stage, f"/World/marker_{i}")
    c.CreateSizeAttr(1.0)
    UsdGeom.Xformable(c).AddTranslateOp().Set(Gf.Vec3d(mx, my, 0.5))
    c.CreateDisplayColorAttr([Gf.Vec3f(0.8, 0.25 + 0.1 * i, 0.15)])

UsdLux.DomeLight.Define(stage, "/World/sky").CreateIntensityAttr(900)
sun = UsdLux.DistantLight.Define(stage, "/World/sun")
sun.CreateIntensityAttr(2500)
UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, 0, 30))

# ---- vehicle ----------------------------------------------------------------------------------
# diagnostics: count every Pegasus callback / backend call and print (not swallow) the first exceptions, because
# Isaac Sim 6.2 silently drops a physics callback that raised once
import collections  # noqa: E402
import traceback  # noqa: E402

from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend  # noqa: E402
from pegasus.simulator.logic.vehicles.multirotor import Multirotor  # noqa: E402
from pegasus.simulator.logic.vehicles.vehicle import Vehicle  # noqa: E402

DIAG = collections.Counter()


def _wrap(cls, name):
    f = getattr(cls, name)

    def w(self, *a, **k):
        DIAG[f"{cls.__name__}.{name}"] += 1
        try:
            return f(self, *a, **k)
        except Exception:
            DIAG[f"{cls.__name__}.{name}.EXC"] += 1
            if DIAG[f"{cls.__name__}.{name}.EXC"] <= 2:
                print(f"[smoke] EXCEPTION in {cls.__name__}.{name}:\n{traceback.format_exc()}", flush=True)
            raise

    setattr(cls, name, w)


for _c, _n in ((Vehicle, "update_state"), (Vehicle, "update_sensors"), (Vehicle, "update_sim_state"),
               (Vehicle, "sim_start_stop"), (Multirotor, "update"), (PX4MavlinkBackend, "update"),
               (PX4MavlinkBackend, "start"), (PX4MavlinkBackend, "wait_for_first_hearbeat"),
               (PX4MavlinkBackend, "send_sensor_msgs"), (PX4MavlinkBackend, "update_sensor"), (x500.X500, "update_state")):
    if hasattr(_c, _n):
        _wrap(_c, _n)
drone = x500.X500("/World/x500", vehicle_id=0, init_pos=(0.0, 0.0, x500.SPAWN_Z))
backend = drone._backends[0]

# ---- contact reports (same API as contact_monitor.py) --------------------------------------------
from omni.physx import get_physx_simulation_interface
from pxr import PhysicsSchemaTools, PhysxSchema, Usd

for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/x500")):
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
        cr.CreateThresholdAttr().Set(0.0)
contacts = {"reports": 0, "pairs": {}, "first_t": None, "last_t": None}
sim_t_now = [0.0]


def _on_contact(headers, data):
    for h in headers:
        a0, a1 = str(PhysicsSchemaTools.intToSdfPath(h.collider0)), str(PhysicsSchemaTools.intToSdfPath(h.collider1))
        if not (a0.startswith("/World/x500") or a1.startswith("/World/x500")):
            continue
        contacts["reports"] += 1
        key = f"{a0} | {a1}"
        contacts["pairs"][key] = contacts["pairs"].get(key, 0) + 1
        if contacts["first_t"] is None:
            contacts["first_t"] = sim_t_now[0]
        contacts["last_t"] = sim_t_now[0]


_contact_sub = get_physx_simulation_interface().subscribe_contact_report_events(_on_contact)

# ---- chase camera -----------------------------------------------------------------------------
cam = UsdGeom.Camera.Define(stage, "/World/chase_cam")
cam.CreateFocalLengthAttr(24.0)
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(20.955 * FRAME_SIZE[1] / FRAME_SIZE[0])
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 2000.0))
cam_op = UsdGeom.Xformable(cam).AddTransformOp()


def aim_chase(pos, yaw):
    back, up = 2.6, 1.0
    eye = Gf.Vec3d(pos[0] - back * math.cos(yaw), pos[1] - back * math.sin(yaw), pos[2] + up)
    target = Gf.Vec3d(*pos)
    cam_op.Set(Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1)).GetInverse())
    return eye


world.reset()
rep.orchestrator.set_capture_on_play(False)
rp = rep.create.render_product(str(cam.GetPath()), FRAME_SIZE)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(rp)

# ---- MAVLink ----------------------------------------------------------------------------------
mav = mavutil.mavlink_connection("udpin:0.0.0.0:14540", source_system=255)
PX4_MAIN_MODE = {1: "MANUAL", 2: "ALTCTL", 3: "POSCTL", 4: "AUTO", 5: "ACRO", 6: "OFFBOARD", 7: "STABILIZED"}
px4 = {"armed": False, "mode": None, "landed_state": None, "heartbeat": False, "attitude": None, "lpos": None}
statustext = []


def pump_mavlink():
    while True:
        m = mav.recv_match(blocking=False)
        if m is None:
            return
        t = m.get_type()
        if t == "HEARTBEAT" and m.get_srcSystem() == 1 and m.type != mavutil.mavlink.MAV_TYPE_GCS:
            px4["heartbeat"] = True
            px4["armed"] = bool(m.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            px4["mode"] = PX4_MAIN_MODE.get((m.custom_mode >> 16) & 0xFF, str(m.custom_mode))
        elif t == "EXTENDED_SYS_STATE":
            px4["landed_state"] = {0: "UNDEFINED", 1: "ON_GROUND", 2: "IN_AIR", 3: "TAKEOFF", 4: "LANDING"}.get(m.landed_state)
        elif t == "ATTITUDE":
            px4["attitude"] = [round(math.degrees(m.roll), 1), round(math.degrees(m.pitch), 1), round(math.degrees(m.yaw), 1)]
        elif t == "LOCAL_POSITION_NED":
            px4["lpos"] = [m.x, m.y, m.z, m.vx, m.vy, m.vz]
        elif t == "STATUSTEXT":
            statustext.append(m.text)
            log("PX4:", m.text)


def send_sticks(x1, y1, x2, y2):
    """Our normalised sticks (-1..1) -> MAVLink MANUAL_CONTROL (x pitch, y roll, z throttle 0..1000, r yaw)."""
    mav.mav.manual_control_send(1, int(1000 * y2), int(1000 * x2), int(500 + 500 * y1), int(1000 * x1), 0)


def command(cmd, *params):
    p = list(params) + [0] * (7 - len(params))
    mav.mav.command_long_send(1, 1, cmd, 0, *p)


# ---- run --------------------------------------------------------------------------------------
timeline = omni.timeline.get_timeline_interface()
timeline.play()
video = VideoWriter(args.out / "trajectory.mp4", FRAME_SIZE[0], FRAME_SIZE[1], fps=FPS)
(args.out / "stills").mkdir(exist_ok=True)

frames = []
phase, phase_t0 = "boot", 0.0
armed_at = None
last_arm_try = -10.0
sim_t = 0.0
wall0 = time.time()
FLIGHT_PLAN = [("climb", 5.0, (0, 0.6, 0, 0)), ("hover", 2.0, (0, 0, 0, 0)), ("forward", 4.0, (0, 0, 0, 0.6)),
               ("yaw_left", 3.0, (-0.5, 0, 0, 0)), ("forward2", 4.0, (0, 0, 0.3, 0.6)),
               ("yaw_right", 6.0, (0.5, 0, 0, 0)), ("forward3", 5.0, (0.2, 0, -0.3, 0.7)), ("hover2", 2.0, (0, 0, 0, 0)), ("descend", 999.0, (0, -0.5, 0, 0))]
plan_i = 0
outcome = "timeout"

import faulthandler
frame_i = 0
while app.is_running() and sim_t < args.max_sim_s:
    # watchdog: if one frame takes > 3 min, dump every thread's Python stack to stderr
    faulthandler.cancel_dump_traceback_later()
    faulthandler.dump_traceback_later(180, repeat=True)
    if args.hq_loop:
        for _ in range(PHYSICS_HZ // FPS):
            world.step(render=False)
    else:
        world.step(render=True)
    if frame_i == 3:
        isaac62_compat.revive_physics_callbacks(world, log=log)
    sim_t = frame_i / FPS
    sim_t_now[0] = sim_t
    pump_mavlink()
    if sim_t - getattr(pump_mavlink, "hb", -1) >= 1.0:
        mav.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
        pump_mavlink.hb = sim_t

    sticks = (0.0, -1.0, 0.0, 0.0)  # throttle low on the ground
    if phase == "boot":
        if px4["heartbeat"] and sim_t - last_arm_try > 2.0:
            command(mavutil.mavlink.MAV_CMD_DO_SET_MODE, mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, 3)
            command(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1)
            last_arm_try = sim_t
        if px4["armed"] and px4["mode"] == "POSCTL":
            armed_at = sim_t
            phase, phase_t0 = FLIGHT_PLAN[0][0], sim_t
            log(f"ARMED in POSCTL at sim t={sim_t:.2f}s")
    elif phase != "done":
        name, dur, s = FLIGHT_PLAN[plan_i]
        sticks = s
        if sim_t - phase_t0 >= dur and plan_i + 1 < len(FLIGHT_PLAN):
            plan_i += 1
            phase, phase_t0 = FLIGHT_PLAN[plan_i][0], sim_t
            log(f"phase -> {phase} at {sim_t:.2f}s, z={drone.state.position[2]:.2f}")
        if phase == "descend" and armed_at is not None and not px4["armed"]:
            phase, phase_t0 = "done", sim_t
            outcome = "landed_and_disarmed"
            log(f"DISARMED after landing at {sim_t:.2f}s")
        if phase == "descend" and px4["landed_state"] == "ON_GROUND":
            sticks = (0, -1, 0, 0)
    else:
        if sim_t - phase_t0 > 3.0:
            break
    send_sticks(*sticks)

    st = drone.state
    pos = [float(v) for v in st.position]
    yaw = float(Rotation.from_quat(st.attitude).as_euler("ZYX")[0])
    eye = aim_chase(pos, yaw)
    if args.hq_loop:
        timeline.pause(); timeline.commit()
        rep.orchestrator.step(rt_subframes=1, pause_timeline=False)
        timeline.play(); timeline.commit()
    try:
        data = rgb.get_data()
    except TypeError:  # first frames before the render product has produced anything
        data = None
    if data is not None and getattr(data, "size", 0):
        img = np.asarray(data)[..., :3]
        video.write_frame(img)
        if frame_i % 150 == 0:
            from PIL import Image
            Image.fromarray(img).save(args.out / "stills" / f"f{frame_i:05d}.png")

    roll, pitch = [float(v) for v in Rotation.from_quat(st.attitude).as_euler("ZYX")[2:0:-1]]
    frames.append({
        "frame": frame_i, "t": round(sim_t, 4), "phase": phase,
        "x1": sticks[0], "y1": sticks[1], "x2": sticks[2], "y2": sticks[3],
        "pose": {"x": pos[0], "y": pos[1], "z": pos[2], "roll_deg": math.degrees(roll),
                 "pitch_deg": math.degrees(pitch), "yaw_deg": math.degrees(yaw)},
        "velocity": dict(zip(("vx", "vy", "vz"), [float(v) for v in st.linear_velocity])),
        "rotor_speed_rad_s": [float(v) for v in drone._thrusters._velocity],
        "contact_reports_so_far": contacts["reports"],
        "camera": {"x": eye[0], "y": eye[1], "z": eye[2], "kind": "chase"},
        "px4": {"armed": px4["armed"], "mode": px4["mode"], "landed_state": px4["landed_state"],
                "attitude_deg": px4["attitude"], "local_position_ned": px4["lpos"]},
    })
    if frame_i % 30 == 0:
        log(f"t={sim_t:6.2f} phase={phase:9s} z={pos[2]:6.3f} armed={px4['armed']} mode={px4['mode']} "
            f"landed={px4['landed_state']} true_yaw_ned={math.degrees(math.pi/2-yaw):7.1f} px4_yaw_ned={px4['attitude'][2] if px4['attitude'] else None} rotors={[round(v) for v in drone._thrusters._velocity]} wall={time.time()-wall0:.0f}s")
    if frame_i % 60 == 0:
        sd = getattr(backend, "_sensor_data", None)
        log(f"diag t={sim_t:.1f} cbs={sorted(getattr(world, '_physics_callback_functions', {}))} calls={dict(DIAG)} playing={world.is_playing()} conn={getattr(backend, '_connection', None) is not None} "
            f"hb={getattr(backend, '_received_first_hearbeat', None)} running={getattr(backend, '_is_running', None)} "
            f"first_imu={getattr(sd, 'received_first_imu', None)} new_imu={getattr(sd, 'new_imu_data', None)} "
            f"first_act={getattr(backend, '_received_first_actuator', None)} lockstep_timeouts={getattr(backend, 'lockstep_timeouts', 0)}")
    frame_i += 1

video.close()

# ---- PX4 flight log ---------------------------------------------------------------------------
ulogs = []
tool = getattr(backend, "px4_tool", None)
if tool is not None:
    for f in Path(tool.root_fs.name).rglob("*.ulg"):
        dst = args.out / ("px4.ulg" if not ulogs else f"px4_{len(ulogs)}.ulg")
        shutil.copy(f, dst)
        ulogs.append(dst.name)

z = [f["pose"]["z"] for f in frames]
meta = {
    "simulation_id": "smoke_x500_0",
    "drone": x500.KIT,
    "drone_physics": x500.physics_summary(),
    "px4": {"version": "v1.14.3", "airframe": "10050_isaac_x500", "control": "MAVLink MANUAL_CONTROL in POSCTL",
            "ulog": ulogs, "statustext": statustext[-50:]},
    "outcome": outcome,
    "armed_at_s": armed_at,
    "max_altitude_m": max(z) if z else None,
    "final_z_m": z[-1] if z else None,
    "frames": len(frames), "fps": FPS, "physics_hz": PHYSICS_HZ, "frame_size": list(FRAME_SIZE),
    "wall_time_s": round(time.time() - wall0, 1),
    "isaac62": {"isaacsim": __import__("importlib.metadata").metadata.version("isaacsim"), "compat": compat_info,
                "dc_calls": isaac62_compat.acquire_dynamic_control_interface().calls, "contacts": contacts},
}
(args.out / "action.json").write_text(json.dumps(frames))
(args.out / "meta.json").write_text(json.dumps(meta, indent=1))
log("RESULT", json.dumps({k: meta[k] for k in ("outcome", "armed_at_s", "max_altitude_m", "final_z_m", "frames", "wall_time_s", "isaac62")}))

timeline.stop()
app.close()
