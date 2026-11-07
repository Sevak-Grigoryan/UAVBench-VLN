"""Render a few still frames of a scene for review (no flight). Views are picked from the mission's layout:
the river (bridge, a bend, the waterfall) and each props group from a drone-like oblique view.

    python preview_stills.py --env Forest --mission easy_v2_001 --out <dir>
"""
import argparse
import math
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import envs

ap = argparse.ArgumentParser()
ap.add_argument("--env", default="Forest")
ap.add_argument("--mission", required=True)
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
env = envs.boot(a.env, a.out / f"kit_{os.environ.get('SLURM_JOB_ID', 'local')}", mission_id=a.mission)
print("[preview] booted", flush=True)
from PIL import Image

nv = env.bridge.natural_valley
g = env.ground


def look(name, tx, ty, tz, dist, height, bearing_deg):
    b = math.radians(bearing_deg)
    cx, cy = tx + dist * math.cos(b), ty + dist * math.sin(b)
    cz = tz + height
    yaw = math.degrees(math.atan2(ty - cy, tx - cx))
    pitch = math.degrees(math.atan2(cz - tz, math.hypot(tx - cx, ty - cy)))
    env.bridge.set_camera_pose(cx, cy, cz, yaw, pitch)
    for _ in range(2):
        f = env.bridge.process_camera_data()
    Image.fromarray(f[:, :, :3]).save(a.out / f"{name}.png")
    print(f"[preview] {name}", flush=True)


views = []
if a.env == "Coastal":
    pl = env.layout
    for st in pl["sites"]:
        x, y = st["xy"]
        z = max(g(x, y), 0.0)
        views.append((f"site_{st['name']}", x, y, z, 70.0, 45.0, math.degrees(st["yaw"]) - 90 - 35))
        views.append((f"site_{st['name']}_close", x, y, z + 1.0, 22.0, 9.0, math.degrees(st["yaw"]) - 90 + 25))
    for sh in pl["ships"]:
        if sh.get("moored") or sh["L"] < 9:
            continue
        x, y = sh["xy"]
        views.append((f"ship_{sh['name']}", x, y, sh["F"], max(25.0, sh["L"] * 0.9), max(10.0, sh["L"] * 0.35), math.degrees(sh["yaw"]) + 50))
    for k, p in pl["pads_all"].items():
        if p.get("ship"):
            views.append((f"helideck_{k}", p["xy"][0], p["xy"][1], p["z"], 14.0, 9.0, 200.0))
    for k, p in pl.get("spots", {}).items():
        views.append((f"spot_{k}_{p['spot']}", p["xy"][0], p["xy"][1], p["z"], 16.0, 8.0, 200.0))
    sk = env.info.get("build", {})
    if sk.get("sun_azimuth_deg") is not None:  # look straight at where the sun light comes from: the sky's sun must be there
        env.bridge.set_camera_pose(0.0, 0.0, 80.0, sk["sun_azimuth_deg"], -min(sk["sun_elevation_deg"], 30.0))
        for _ in range(2):
            f = env.bridge.process_camera_data()
        Image.fromarray(f[:, :, :3]).save(a.out / "sun_direction.png")
        print(f"[preview] sun_direction {sk.get('sky')} az {sk['sun_azimuth_deg']} el {sk['sun_elevation_deg']}", flush=True)
    views.append(("overview", 0.0, pl["coast"]["A1"] * 0.0, 0.0, 420.0, 230.0, 250.0))
    views.append(("beach_low", *next(((s["xy"][0], s["xy"][1]) for s in pl["sites"] if s["kind"] == "beach"), (0.0, 0.0)), 1.0, 12.0, 2.5, 280.0))
    for v in views:
        look(*v)
    print("[preview] DONE", flush=True)
    env.bridge.shutdown()
    raise SystemExit(0)
if a.env == "Mountain":
    pl = env.layout
    import mountain_v2 as _mv
    for st in pl["sites"]:
        x, y = st["xy"]
        z = st["base_z"]
        views.append((f"site_{st['name']}", x, y, z, 80.0 + st["size_r"], 45.0 + 0.4 * st["size_r"], math.degrees(st.get("yaw", 0.0)) - 70))
        views.append((f"site_{st['name']}_close", x, y, z + 1.0, 18.0 + 0.3 * st["size_r"], 8.0, math.degrees(st.get("yaw", 0.0)) + 200))
    for p in pl["summits"][:3]:
        views.append((f"summit_{p['name']}", p["x"], p["y"], p["z"], 260.0, 60.0, 200.0))
    for k, p in list(pl["pads_all"].items())[:4]:
        views.append((f"place_{k}", p["xy"][0], p["xy"][1], p["z"], 16.0, 7.0, 210.0))
    v = pl["valley"]
    for xx, nm in ((-450.0, "valley_west"), (0.0, "valley_mid"), (450.0, "valley_east")):
        yv = float(_mv.valley_y(xx, v))
        views.append((nm, xx, yv, float(_mv.floor_z(xx, v)) + 30.0, 260.0, 40.0, 180.0 if nm != "valley_east" else 0.0))
    views.append(("river_low", 120.0, float(_mv.river_y(120.0, v)), float(_mv.river_level(120.0, v)), 30.0, 12.0, 160.0))
    views.append(("overview", 0.0, 0.0, 60.0, 750.0, 380.0, 240.0))
    sk = env.info.get("build", {})
    if sk.get("sun_azimuth_deg") is not None:
        env.bridge.set_camera_pose(0.0, 0.0, 200.0, sk["sun_azimuth_deg"], -min(sk["sun_elevation_deg"], 30.0))
        for _ in range(2):
            f = env.bridge.process_camera_data()
        Image.fromarray(f[:, :, :3]).save(a.out / "sun_direction.png")
    for v_ in views:
        look(*v_)
    print("[preview] DONE", flush=True)
    env.bridge.shutdown()
    raise SystemExit(0)
if a.env == "City":
    pl = env.layout
    for (bx, by), f in pl["features"].items():
        views.append((f"{f['name']}", bx, by, 0.0, 34.0, 30.0, 225.0))
        views.append((f"{f['name']}_close", bx, by, 0.0, 16.0, 9.0, 200.0))
    views.append(("street_1", 0.0, 62.0 * 1.5, 0.0, 25.0, 14.0, 250.0))
    views.append(("street_2", 62.0, -40.0, 0.0, 18.0, 8.0, 160.0))
    views.append(("tower", 31.0, 31.0, 30.0, 60.0, 25.0, 200.0))
    views.append(("overview", 0.0, 0.0, 0.0, 160.0, 90.0, 220.0))
    for v in views:
        look(*v)
    print("[preview] DONE", flush=True)
    env.bridge.shutdown()
    raise SystemExit(0)
for x, nm in ((72.0, "river_bridge"), (10.0, "river_bend"), (150.0, "river_mid")):
    y = nv.river_y(x)
    views.append((nm, x, y, g(x, y), 30.0, 16.0, 200.0))
views.append(("river_low", 30.0, nv.river_y(30.0), g(30.0, nv.river_y(30.0)), 14.0, 5.0, 160.0))
views.append(("waterfall", 262.0, nv.river_y(262.0), g(262.0, nv.river_y(262.0)), 28.0, 12.0, 205.0))
for grp in env.layout["groups"]:
    x, y = grp["xy"]
    r = grp["clear_r"]
    views.append((f"group{grp['gid']}_{grp['type']}", x, y, g(x, y) + 1.0, r + 14.0, 18.0, 35.0 * grp["gid"]))
    o = grp["objects"][0]
    views.append((f"group{grp['gid']}_{grp['type']}_close", o["xy"][0], o["xy"][1], g(*o["xy"]) + 1.0, 14.0, 6.0, 35.0 * grp["gid"] + 20))
for v in views:
    look(*v)
print("[preview] DONE", flush=True)
env.bridge.shutdown()
