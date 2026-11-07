"""Physics-only check of one coastal plan: build the scene exactly as the flight does (same colliders, same rigid
bodies), simulate N seconds at 240 Hz without PX4 and without rendering, and report which props move.
    python physics_probe.py --out DIR --mission easy_coastal_020 [--slab cube|convex|mesh] [--secs 60] [--cpu]"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--mission", required=True)
ap.add_argument("--slab", default="cube")
ap.add_argument("--secs", type=float, default=60.0)
ap.add_argument("--cpu", action="store_true")
ap.add_argument("--drop", default="", help="comma-separated path substrings: deactivate matching prims (bisect)")
ap.add_argument("--tag", default="")
ap.add_argument("--barrelbox", action="store_true")
ap.add_argument("--friction", action="store_true")
ap.add_argument("--sleep", action="store_true")
ap.add_argument("--iters", type=int, default=0)
ap.add_argument("--no-terrain-col", action="store_true")
args = ap.parse_args()
os.environ["COASTAL_SLAB_COLLIDER"] = args.slab
sys.dont_write_bytecode = True
COMMON = Path(__file__).resolve().parents[2] / "common"
sys.path.insert(0, str(COMMON / "pipeline"))
import envs  # noqa: E402

env = envs.boot("Coastal", args.out / f"kit_{os.environ.get('SLURM_JOB_ID', 'local')}", mission_id=args.mission)
from isaacsim.core.api.world import World  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

stage = env.bridge.stage
world = World(physics_dt=1 / 240, rendering_dt=1 / 30, stage_units_in_meters=1.0)
ctx = world.get_physics_context()
if args.cpu:
    ctx.enable_gpu_dynamics(False)
print("gpu_dynamics", ctx.is_gpu_dynamics_enabled(), flush=True)
drops = [d for d in args.drop.split(",") if d]
if drops:
    kill = [str(pr.GetPath()) for pr in stage.Traverse() if any(d in str(pr.GetPath()) for d in drops)]
    kill = [k for k in kill if not any(k.startswith(o + "/") for o in kill)]
    for k in kill:
        stage.GetPrimAtPath(k).SetActive(False)
    print("deactivated", len(kill), kill[:40], flush=True)
# --- same static-collider pass as run_x500_mission.py
for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
    if prim.IsInstanceProxy() or prim.IsA(UsdGeom.PointInstancer):
        continue
    path = str(prim.GetPath())
    if "/Prototypes/" in path or any(s in path for s in ("/Nature", "/EgoCamera", "/Looks", "/Materials", "/World/Ocean", "/World/DistantTerrain")):
        continue
    if prim.HasAttribute("x500:noCollision"):
        continue
    if prim.IsA(UsdGeom.Gprim) and not prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
rp = stage.GetPrimAtPath("/World/Coast/Pads")
if rp:
    for prim in Usd.PrimRange(rp):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetName().startswith(("ring", "H")):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
RIGID = [r for r in env.rigid_bodies if stage.GetPrimAtPath(r["path"]) and stage.GetPrimAtPath(r["path"]).IsActive()]
if args.no_terrain_col:
    tp = stage.GetPrimAtPath("/World/Terrain")
    UsdPhysics.CollisionAPI.Apply(tp).CreateCollisionEnabledAttr(False)
env.make_dynamic(stage, RIGID)
from pxr import Gf, PhysxSchema, Sdf, UsdShade  # noqa: E402
if args.barrelbox:
    for r in RIGID:
        if r["kind"] != "barrel":
            continue
        for pr in Usd.PrimRange(stage.GetPrimAtPath(r["path"])):
            if pr.IsA(UsdGeom.Cylinder) and pr.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(pr).CreateCollisionEnabledAttr(False)
        cb = UsdGeom.Cube.Define(stage, r["path"] + "/col_box")
        cb.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(cb)
        xf.SetTranslate(Gf.Vec3d(0, 0, 0.45))
        xf.SetScale(Gf.Vec3f(0.41, 0.41, 0.9))
        cb.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
        UsdPhysics.CollisionAPI.Apply(cb.GetPrim())
if args.friction:
    mp = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/HighFriction")
    pm = UsdPhysics.MaterialAPI.Apply(mp.GetPrim())
    pm.CreateStaticFrictionAttr(1.0); pm.CreateDynamicFrictionAttr(0.9); pm.CreateRestitutionAttr(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(mp.GetPrim()).CreateFrictionCombineModeAttr("max")
    for root in ("/World/Props", "/World/Coast", "/World/Terrain"):
        rp_ = stage.GetPrimAtPath(root)
        if rp_:
            for pr in Usd.PrimRange(rp_):
                if pr.HasAPI(UsdPhysics.CollisionAPI):
                    UsdShade.MaterialBindingAPI.Apply(pr).Bind(mp, UsdShade.Tokens.weakerThanDescendants, "physics")
for r in RIGID:
    if not (args.sleep or args.iters):
        break
    pa = PhysxSchema.PhysxRigidBodyAPI.Apply(stage.GetPrimAtPath(r["path"]))
    if args.sleep:
        pa.CreateSleepThresholdAttr(0.005); pa.CreateStabilizationThresholdAttr(0.002)
        pa.CreateLinearDampingAttr(0.2); pa.CreateAngularDampingAttr(0.5)
    if args.iters:
        pa.CreateSolverPositionIterationCountAttr(args.iters); pa.CreateSolverVelocityIterationCountAttr(4)
world.reset()
from omni.isaac.dynamic_control import _dynamic_control  # noqa: E402

dc = _dynamic_control.acquire_dynamic_control_interface()


def poses():
    out = {}
    for r in RIGID:
        b = dc.get_rigid_body(r["path"])
        if b:
            p = dc.get_rigid_body_pose(b)
            out[r["name"]] = ([float(p.p[0]), float(p.p[1]), float(p.p[2])], [float(v) for v in p.r])
    return out


for _ in range(240 * 2):
    world.step(render=False)
p0 = poses()
track = {k: [] for k in p0}
n = int(240 * args.secs)
for i in range(n):
    world.step(render=False)
    if i % 240 == 0:
        for k, v in poses().items():
            track[k].append([round(c, 3) for c in v[0]])
p1 = poses()
rows = []
for r in RIGID:
    k = r["name"]
    if k not in p0:
        rows.append({"name": k, "error": "no body"})
        continue
    (a, qa), (b, qb) = p0[k], p1[k]
    dot = min(1.0, abs(sum(x * y for x, y in zip(qa, qb))))
    rows.append({"name": k, "kind": r["kind"], "settle": round(math.dist(r["pos"], a), 4), "moved": round(math.dist(a, b), 4),
                 "rot": round(math.degrees(2 * math.acos(dot)), 3), "track": track[k][::5]})
bad = [x for x in rows if "error" in x or x["moved"] >= 0.05 or x["rot"] >= 2.0 or x["settle"] > 0.05]
res = {"mission": args.mission, "slab": args.slab, "cpu": args.cpu, "secs": args.secs, "bodies": len(rows), "bad": len(bad),
       "worst": sorted(bad, key=lambda x: -x.get("moved", 99))[:25]}
(args.out / f"probe_{args.mission}_{args.slab}{'_cpu' if args.cpu else ''}{args.tag}.json").write_text(json.dumps(res, indent=1))
print("RESULT", args.tag, args.mission, args.slab, "cpu" if args.cpu else "gpu", "bodies", len(rows), "bad", len(bad), flush=True)
for x in res["worst"][:12]:
    print("  ", x["name"], x.get("kind"), x.get("settle"), x.get("moved"), x.get("rot"), flush=True)
env.bridge.app.close() if hasattr(env.bridge, "app") else None
