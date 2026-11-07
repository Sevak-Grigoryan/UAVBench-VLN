"""Build the X500 forest stage offline (usd-core, no Isaac Sim) and report every object whose lowest
point is more than THRESH metres above the terrain directly beneath it (i.e. floating)."""
import math, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
import numpy as np
from pxr import Usd, UsdGeom, Gf
import natural_valley as nv, forest_scene as fs, missions
missions.apply_pads(fs)
SEED = 5200
fs.patch_terrain(nv, SEED); fs.relax_clearings(nv)
stage = Usd.Stage.CreateInMemory()
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z); UsdGeom.SetStageMetersPerUnit(stage, 1.0)
w = UsdGeom.Xform.Define(stage, "/World"); stage.SetDefaultPrim(w.GetPrim())
ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-v2"
nv.build_environment(stage, SEED, ROOT)
fs.add_dense_vegetation(stage, nv, SEED, ROOT)
fs.rebuild_lookout(stage, nv, SEED)
fs.build_waterfall(stage, nv, SEED, ROOT)
fs.rebuild_footbridge(stage, nv, SEED)
ground = lambda x, y: nv.terrain_height(x, y, SEED)
THRESH = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
report = {}
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.PointInstancer):
        pi = UsdGeom.PointInstancer(prim)
        xf = pi.ComputeInstanceTransformsAtTime(0, 0)
        protos = pi.GetPrototypesRel().GetTargets()
        idx = pi.GetProtoIndicesAttr().Get()
        pbox = [cache.ComputeLocalBound(stage.GetPrimAtPath(p)).ComputeAlignedRange() for p in protos]
        bad = []
        for m, k in zip(xf, idx):
            r = pbox[k]
            corners = [Gf.Vec3d(x, y, z) for x in (r.GetMin()[0], r.GetMax()[0]) for y in (r.GetMin()[1], r.GetMax()[1]) for z in (r.GetMin()[2], r.GetMax()[2])]
            pts = [m.Transform(c) for c in corners]
            zmin = min(p[2] for p in pts); px, py = m.ExtractTranslation()[0], m.ExtractTranslation()[1]
            gap = zmin - ground(px, py)
            if gap > THRESH: bad.append((round(gap, 2), round(px, 1), round(py, 1)))
        report[str(prim.GetPath())] = (len(xf), sorted(bad, reverse=True))
    elif prim.IsA(UsdGeom.Gprim) and "/World/Landmarks" in str(prim.GetPath()) or (prim.IsA(UsdGeom.Xform) and str(prim.GetPath()).startswith("/World/Landmarks/CliffGate/C")):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if r.IsEmpty(): continue
        c = r.GetMidpoint(); gap = r.GetMin()[2] - ground(c[0], c[1])
        if gap > THRESH: report.setdefault("LANDMARK " + str(prim.GetPath()), (1, [(round(gap, 2), round(c[0], 1), round(c[1], 1))]))
for k, (n, bad) in report.items():
    if bad: print(f"{k}: {len(bad)}/{n} floating > {THRESH} m; worst {bad[:5]}")
print("checked", len(report), "groups")
