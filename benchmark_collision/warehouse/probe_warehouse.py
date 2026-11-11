"""
GPU probe of the warehouse template: where are the floor, walls, ceiling and shelving racks, and are they solid?
Writes <out>/warehouse_probe.json (every mesh's world bbox, grouped) and a few rendered views (<out>/view_*.png), so the
collision scenarios can be placed on the real geometry.

    python probe_warehouse.py --out /mnt/weka/sgrigoryan/Benchmark_collision_work_warehouse/probe
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import warehouse_env  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
a = ap.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
env = warehouse_env.boot(a.out / f"kit_{os.environ.get('SLURM_JOB_ID', 'local')}")
bridge, stage = env.bridge, env.bridge.stage
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
meshes, n_inst_proxy, n_col_proxy = [], 0, 0
for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
    if not prim.IsA(UsdGeom.Gprim):
        continue
    if prim.IsInstanceProxy():
        n_inst_proxy += 1
    has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
    n_col_proxy += has_col
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if r.IsEmpty():
        continue
    lo, hi = r.GetMin(), r.GetMax()
    meshes.append({"path": str(prim.GetPath()), "type": prim.GetTypeName(), "instance_proxy": prim.IsInstanceProxy(),
                   "collision": has_col, "visible": UsdGeom.Imageable(prim).ComputeVisibility() != UsdGeom.Tokens.invisible,
                   "min": [round(v, 3) for v in lo], "max": [round(v, 3) for v in hi]})
print(f"[probe] {len(meshes)} gprims, {n_inst_proxy} instance proxies, {n_col_proxy} with CollisionAPI", flush=True)

# group by the first two path components below the default prim, to find racks / walls / floor
groups = {}
for m in meshes:
    parts = m["path"].strip("/").split("/")
    key = "/" + "/".join(parts[:3])
    g = groups.setdefault(key, {"n": 0, "collision": 0, "min": [1e9] * 3, "max": [-1e9] * 3, "names": set()})
    g["n"] += 1
    g["collision"] += m["collision"]
    g["names"].add(parts[-1])
    for i in range(3):
        g["min"][i] = min(g["min"][i], m["min"][i])
        g["max"][i] = max(g["max"][i], m["max"][i])
for g in groups.values():
    g["names"] = sorted(g["names"])[:12]
allmin = [min(m["min"][i] for m in meshes) for i in range(3)]
allmax = [max(m["max"][i] for m in meshes) for i in range(3)]
big = sorted(meshes, key=lambda m: -(m["max"][0] - m["min"][0]) * (m["max"][1] - m["min"][1]))[:15]
summary = {"stage_meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage), "up_axis": UsdGeom.GetStageUpAxis(stage),
           "default_prim": str(stage.GetDefaultPrim().GetPath()) if stage.GetDefaultPrim() else None,
           "bbox_min": allmin, "bbox_max": allmax, "n_gprims": len(meshes), "n_instance_proxies": n_inst_proxy,
           "n_with_collision": n_col_proxy, "largest_footprint": big,
           "groups": {k: v for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["n"])},
           "top_level": [str(p.GetPath()) for p in stage.GetPseudoRoot().GetChildren()]}
(a.out / "warehouse_probe.json").write_text(json.dumps(summary, indent=1, default=list))
(a.out / "warehouse_meshes.json").write_text(json.dumps(meshes))
print(f"[probe] bbox {allmin} .. {allmax}", flush=True)

# a few views from inside: centre at 2 / 6 m, looking along +x, -x, +y, -y, and a high oblique overview
from PIL import Image  # noqa: E402

cx, cy = (allmin[0] + allmax[0]) / 2, (allmin[1] + allmax[1]) / 2
views = [("c2_px", cx, cy, 2.0, 0, 5), ("c2_mx", cx, cy, 2.0, 180, 5), ("c2_py", cx, cy, 2.0, 90, 5),
         ("c2_my", cx, cy, 2.0, -90, 5), ("c6_px", cx, cy, 6.0, 0, 20),
         ("corner_high", allmin[0] + 1.5, allmin[1] + 1.5, min(allmax[2] - 1.0, 8.0),
          math.degrees(math.atan2(cy - allmin[1], cx - allmin[0])), 25)]
for name, x, y, z, yaw, pitch in views:
    bridge.set_camera_pose(x, y, z, yaw, pitch)
    for _ in range(3):
        bridge.app.update()
    img = bridge.process_camera_data()
    Image.fromarray(img[..., :3]).save(a.out / f"view_{name}.png")
    print(f"[probe] view {name} at ({x:.1f},{y:.1f},{z:.1f}) yaw {yaw:.0f} pitch {pitch}", flush=True)
print("[probe] done", flush=True)
bridge.shutdown()
