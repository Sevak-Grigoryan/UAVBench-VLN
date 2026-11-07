"""Build an Easy v2 stage offline (usd-core, no Isaac Sim) and check every prop: lowest point vs the visible
ground under it (floating / buried), and that the river v2 prims exist. Usage: easy_v2_stage_check.py <idx>..."""
import math, sys, time
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
from pxr import Usd, UsdGeom, Gf
import natural_valley as nv, forest_scene as fs, missions, easy_v2
missions.apply_pads(fs)
SEED, ROOT = 5200, "/mnt/weka/sgrigoryan/assets/polyhaven-v2"
fs.patch_terrain(nv, SEED); fs.relax_clearings(nv)
for idx in [int(a) for a in sys.argv[1:]] or [1]:
    t0 = time.time()
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    w = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(w.GetPrim())
    nv.build_environment(stage, SEED, ROOT)
    fs.add_dense_vegetation(stage, nv, SEED, ROOT)
    fs.rebuild_lookout(stage, nv, SEED)
    fs.build_river_v2(stage, nv, SEED, ROOT)
    fs.rebuild_footbridge(stage, nv, SEED)
    lay = easy_v2.layout(idx, nv, fs)
    info = easy_v2.build(stage, nv, lay)
    cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    worst_float, worst_bury, n = [], [], 0
    for g in lay["groups"]:
        for o in g["objects"]:
            if o.get("world"):
                continue
            prim = stage.GetPrimAtPath(f"/World/Props/G{g['gid']}_{g['type']}/{o['name']}")
            r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            c = r.GetMidpoint()
            gz = fs.visible_ground(nv, c[0], c[1], SEED)
            n += 1
            if "matrix" in o:  # movable: every support point vs ground under it
                M = Gf.Matrix4d(*[v for row in o["matrix"] for v in row])
                for a, b in o["prop"]["support"]:
                    pw = M.Transform(Gf.Vec3d(a, b, 0))
                    gap = pw[2] - fs.visible_ground(nv, pw[0], pw[1], SEED)
                    if gap > 0.03: worst_float.append((round(gap, 3), o["name"]))
                    if gap < -0.03: worst_bury.append((round(gap, 3), o["name"]))
            else:
                gap = r.GetMin()[2] - gz
                if gap > 0.0: worst_float.append((round(gap, 3), o["name"]))
    print(f"sim {idx}: {n} objects, {info['props_primitives']} primitives, rigid {len(info['rigid_bodies'])}, "
          f"veg removed {info['vegetation_removed']}, floating {sorted(worst_float, reverse=True)[:5]}, "
          f"buried {sorted(worst_bury)[:5]}, river prims {[p.GetName() for p in stage.GetPrimAtPath('/World/River').GetChildren()]} "
          f"({time.time()-t0:.0f}s)")
    if len(sys.argv) > 1 and sys.argv[-1] == str(idx) and "--save" in sys.argv[0]:
        pass
