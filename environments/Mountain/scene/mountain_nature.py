"""
Mountain v2 vegetation and rocks: a conifer forest belt (mature firs and pines, young trees, saplings toward the tree
line), broadleaf trees and shrubs on the valley floor, boulders on scree and meadows. Pure numpy planning (plan_vegetation:
the instance list + a 10 m canopy-height grid used by the mission clearance check), then build_vegetation authors one
PointInstancer per model variant (visual only: under /World/Mtn/Nature, skipped by the flight job's collider pass).

Poly Haven tree files hold three variants side by side (fir_tree_01 a / b / c at x = 0, 6, 12 m): every variant is its
own prototype, recentred, so a forest is single trees in natural spacing, not repeated clumps of three.
"""
import math

import numpy as np

import mountain_v2 as mv

# prototype key -> (model, sub-prim or None, centre offset (x, y), real height m, kind)
PROTOS = {
    "fir_a": ("fir_tree_01", "/root/fir_tree_01_a_LOD0", (0.0, 0.3), 18.9, "tree"),
    "fir_b": ("fir_tree_01", "/root/fir_tree_01_b_LOD0", (6.2, 0.1), 14.1, "tree"),
    "fir_c": ("fir_tree_01", "/root/fir_tree_01_c_LOD0", (12.4, 0.0), 14.5, "tree"),
    "pine_a": ("pine_tree_01", "/root/pine_tree_01_a_LOD0", (-5.7, 0.3), 20.4, "tree"),
    "pine_b": ("pine_tree_01", "/root/pine_tree_01_b_LOD0", (1.1, -0.3), 14.9, "tree"),
    "pine_c": ("pine_tree_01", "/root/pine_tree_01_c_LOD0", (8.5, 0.2), 17.5, "tree"),
    "yfir_a": ("fir_sapling_medium", "/root/fir_sapling_medium_a_LOD0", (-0.1, -0.1), 8.8, "young"),
    "yfir_b": ("fir_sapling_medium", "/root/fir_sapling_medium_b_LOD0", (5.1, 0.0), 7.8, "young"),
    "yfir_c": ("fir_sapling_medium", "/root/fir_sapling_medium_c_LOD0", (10.1, -0.1), 5.9, "young"),
    "ypine_a": ("pine_sapling_medium", "/root/pine_sapling_medium_a_LOD0", (0.0, -0.2), 11.5, "young"),
    "ypine_b": ("pine_sapling_medium", "/root/pine_sapling_medium_b_LOD0", (6.0, -0.1), 9.1, "young"),
    "sap_fir": ("fir_sapling", None, (1.0, 0.0), 1.3, "sapling"),
    "sap_pine": ("pine_sapling_small", None, (1.0, 0.0), 1.3, "sapling"),
    "broadleaf": ("tree_small_02", None, (0.1, -0.8), 4.6, "broadleaf"),
    "shrub": ("shrub_02", None, (0.0, 0.0), 2.1, "shrub"),
    "fern": ("fern_02", None, (0.0, 0.0), 0.4, "ground"),
    "grass": ("grass_bermuda_01", None, (0.0, 0.0), 0.16, "ground"),
    "boulder": ("boulder_01", None, (0.0, 0.0), 1.0, "rock"),
    "boulder2": ("rock_moss_set_02", None, (0.0, 0.0), 1.4, "rock"),
    "boulder3": ("boulder_01", None, (0.0, 0.0), 1.0, "rock"),
    "boulder4": ("rock_moss_set_01", None, (0.0, 0.0), 1.8, "rock"),
    "mossrocks": ("rock_moss_set_02", None, (0.0, 0.0), 1.4, "rock"),
    "deadtrunk": ("dead_tree_trunk_02", None, (0.0, 0.0), 1.0, "ground"),
    "stump": ("tree_stump_01", None, (0.0, 0.0), 0.6, "ground"),
}
DENSITY = {"alpine valley": 1.0, "glacial lake valley": 0.95, "gorge": 1.1, "dolomite towers": 0.9, "rolling highlands": 0.35, "high cirque": 0.4}


def plan_vegetation(pl):
    """Instances [(proto, x, y, z, yaw, scale)] and the canopy grid. Own rng: the rest of the plan does not depend on it."""
    rng = np.random.default_rng(31_337 * pl["index"] + 7 * pl["attempt"] + 3)
    v = pl["valley"]
    land = pl["landscape"]
    dens = DENSITY[land]
    blocked = [(s["xy"][0], s["xy"][1], s["size_r"] + 14) for s in pl["sites"]]
    blocked += [(p["xy"][0], p["xy"][1], 20.0) for p in pl["pads"].values()] + [(p["xy"][0], p["xy"][1], 16.0) for p in pl.get("spots", {}).values()]
    # every road densified to 2 m steps; a tree keeps its trunk + 2.5 m off the road edge (a 1-in-120 subsample let trees
    # stand on the road, audit 2026-10-21)
    rp, rr = [], []
    for r in pl["roads"]:
        pts = np.asarray(r["pts"], float)
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.hypot(*(b - a)) / 2.0) + 1)
            for t in np.linspace(0.0, 1.0, n):
                rp.append(a + (b - a) * t)
                rr.append((r["w"] / 2 + 3.0) ** 2)
    road_xy = np.array(rp) if rp else None
    road_r2 = np.array(rr) if rr else None
    sol = [sd for sd in pl["solids"] if sd[0] == "rect"]
    sol_xy = np.array([(sd[1], sd[2], math.hypot(sd[4], sd[5])) for sd in sol]) if sol else np.zeros((0, 3))

    def solids_near(x, y):
        if not len(sol_xy):
            return []
        return [sol[i] for i in np.where(np.hypot(sol_xy[:, 0] - x, sol_xy[:, 1] - y) < sol_xy[:, 2] + 3.0)[0]]
    cab = [seg for c in pl["cables"] for seg in c["segs"]]
    N = int(65000 * dens)
    X = rng.uniform(-mv.EXTENT + 4, mv.EXTENT - 4, N)
    Y = rng.uniform(-mv.EXTENT + 4, mv.EXTENT - 4, N)
    Z = mv.terrain_heights(X, Y, pl)
    e = 3.0
    gx = (mv.terrain_heights(X + e, Y, pl) - mv.terrain_heights(X - e, Y, pl)) / (2 * e)
    gy = (mv.terrain_heights(X, Y + e, pl) - mv.terrain_heights(X, Y - e, pl)) / (2 * e)
    S = np.hypot(gx, gy)
    zf = mv.floor_z(X, v)
    rel = Z - zf
    forest = mv.fbm(X / 140.0, Y / 140.0, v["seed"] + 301, 4)  # patchy forest with clearings and avalanche paths
    tl = pl["treeline"] + 25.0 * mv.fbm(X / 90.0, Y / 90.0, v["seed"] + 307, 2)
    rd = np.abs(Y - mv.river_y(X, v))
    dval = np.abs(Y - mv.valley_y(X, v)) - mv.valley_w(X, v)
    inst = []
    for i in range(N):
        x, y, z, s, r_ = float(X[i]), float(Y[i]), float(Z[i]), float(S[i]), float(rel[i])
        if s > 1.05 or rd[i] < v["rw"] + 4:
            continue
        if mv.water_level(x, y, pl) is not None:
            continue
        if any(math.hypot(x - a, y - b) < rr for a, b, rr in blocked):
            continue
        if road_xy is not None and float(np.min((road_xy[:, 0] - x) ** 2 + (road_xy[:, 1] - y) ** 2 - road_r2)) < 0.0:
            continue
        if any(sd[0] == "rect" and sd[7] - sd[6] > 0.3 and mv._solid_hit(sd, x, y, 1.5) for sd in solids_near(x, y)):
            continue
        if cab and any(_seg_dist(x, y, sg) < 16.0 for sg in cab):
            continue
        snow = r_ > pl["snowline"] + 10 * float(forest[i])
        u = rng.random()
        f = float(forest[i])
        if r_ < tl[i] - 30 and not snow:
            if dval[i] < -10 and land != "gorge":  # valley floor: meadows with a few broadleaf trees and shrubs
                if f > 0.25 and u < 0.35:
                    k = "broadleaf" if u < 0.2 else "shrub"
                elif u < 0.06:
                    k = str(rng.choice(["shrub", "boulder4", "grass"]))
                else:
                    continue
            elif f > -0.12:          # forest
                if u < 0.62:
                    k = str(rng.choice(["fir_a", "fir_b", "fir_c", "pine_a", "pine_b", "pine_c"], p=[0.22, 0.2, 0.18, 0.14, 0.13, 0.13]))
                elif u < 0.82:
                    k = str(rng.choice(["yfir_a", "yfir_b", "yfir_c", "ypine_a", "ypine_b"]))
                elif u < 0.88:
                    k = str(rng.choice(["fern", "deadtrunk", "stump", "mossrocks"]))
                else:
                    continue
            elif u < 0.05:
                k = str(rng.choice(["yfir_a", "yfir_c", "shrub", "boulder"]))
            else:
                continue
        elif r_ < tl[i] + 20 and not snow:   # tree line: stunted young trees, saplings, boulders
            if u < 0.18 and f > -0.3:
                k = str(rng.choice(["yfir_b", "yfir_c", "sap_fir", "sap_pine", "sap_fir"]))
            elif u < 0.26:
                k = str(rng.choice(["boulder", "boulder2", "boulder3", "mossrocks"]))
            else:
                continue
        else:                                 # alpine zone: rocks only
            if u < 0.06 and s < 0.9:
                k = str(rng.choice(["boulder", "boulder2", "boulder3", "boulder4"]))
            else:
                continue
        kind = PROTOS[k][4]
        sc = {"tree": float(rng.uniform(0.8, 1.2)), "young": float(rng.uniform(0.75, 1.2)), "sapling": float(rng.uniform(1.0, 2.2)),
              "broadleaf": float(rng.uniform(1.0, 2.0)), "shrub": float(rng.uniform(0.6, 1.2)), "ground": float(rng.uniform(0.8, 1.4)),
              "rock": float(rng.uniform(0.8, 3.0))}[kind]
        sink = {"tree": 0.3, "young": 0.2, "rock": 0.15 * sc}.get(kind, 0.05)
        inst.append((k, x, y, z - sink, float(rng.uniform(0, 2 * math.pi)), sc))
    # canopy-height grid (10 m cells): highest tree top per cell + 1 cell margin
    n = int(2 * mv.EXTENT // 10) + 1
    cg = np.full((n, n), -1e9)
    for k, x, y, z, yaw, sc in inst:
        kind = PROTOS[k][4]
        if kind not in ("tree", "young", "broadleaf"):
            continue
        top = z + PROTOS[k][3] * sc
        i, j = int((x + mv.EXTENT) // 10), int((y + mv.EXTENT) // 10)
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if 0 <= i + di < n and 0 <= j + dj < n and cg[j + dj, i + di] < top:
                    cg[j + dj, i + di] = top
    pl["_canopy"] = cg
    pl["vegetation"] = inst
    pl["vegetation_counts"] = {k: sum(1 for t in inst if t[0] == k) for k in PROTOS if any(t[0] == k for t in inst)}


def _seg_dist(x, y, sg):
    (ax, ay), (bx, by) = sg
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
    return math.hypot(x - ax - t * dx, y - ay - t * dy)


def build_vegetation(stage, nv, pl, root):
    from pathlib import Path

    from pxr import Gf, Sdf, UsdGeom

    by = {}
    for k, x, y, z, yaw, sc in pl["vegetation"]:
        by.setdefault(k, []).append((0, (x, y, z), yaw, (sc, sc, sc)))
    out = {}
    single_mats = {}
    for k, items in sorted(by.items()):
        model, sub, (ox, oy), _, kind = PROTOS[k]
        asset = str(Path(root) / "models" / model / f"{model}_1k.usdc")
        if (Path(root) / "models" / model / f"{model}_1k_st.usdc").exists():
            # copy with the UVs as 'st' (fir_tree_01 / fir_sapling_medium / pine_sapling_medium store them as float3 'UVMap'
            # and their needles came out fully transparent: bare-trunk forests, preview 2026-10-20)
            asset = str(Path(root) / "models" / model / f"{model}_1k_st.usdc")
        n_diff = len(list((Path(root) / "models" / model / "textures").glob("*_diff_*")))
        if n_diff == 1 and model not in single_mats:
            single_mats[model] = nv.create_asset_material(stage, f"/World/Materials/MtnVeg_{model}", root, model)
        mat = single_mats.get(model)
        if n_diff > 1:
            mat = _tree_materials(stage, nv, root, model)  # dict: source material name -> our material
        # own instancer per prototype; the prototype = an offset Xform that references the variant sub-prim
        base = f"/World/Mtn/Nature/{k}"
        _instancer(stage, nv, base, asset, sub, (ox, oy), mat, items)
        out[f"veg_{k}"] = len(items)
    return out


_TREE_MATS = {}


def _tree_materials(stage, nv, root, model):
    """UsdPreviewSurface materials for every material of a multi-material Poly Haven tree: <model>_<part> uses
    textures/<model>_<part>_diff_1k (+ _alpha_ -> cut-out opacity); parts without their own texture use the bark."""
    from pathlib import Path

    from pxr import Sdf, UsdShade

    key = (id(stage), model)
    if key in _TREE_MATS:
        return _TREE_MATS[key]
    tex = Path(root) / "models" / model / "textures"
    out = {}
    for d in sorted(tex.glob("*_diff_*")):
        part = d.name.split("_diff_")[0]
        if part == model:
            continue
        path = f"/World/Materials/MtnTree_{part}"
        m = UsdShade.Material.Define(stage, path)
        sf = UsdShade.Shader.Define(stage, f"{path}/Surface")
        sf.CreateIdAttr("UsdPreviewSurface")
        sf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
        st = UsdShade.Shader.Define(stage, f"{path}/TexCoord")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        t = UsdShade.Shader.Define(stage, f"{path}/Diffuse")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(d))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
        al = sorted(tex.glob(f"{part}_alpha_*"))
        from PIL import Image
        if not al and Image.open(d).mode == "RGBA":  # cut-out mask in the diffuse PNG's own alpha (pine twigs, birch leaves)
            t.CreateOutput("a", Sdf.ValueTypeNames.Float)
            sf.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(t.ConnectableAPI(), "a")
            sf.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.3)
        if al:
            a = UsdShade.Shader.Define(stage, f"{path}/Alpha")
            a.CreateIdAttr("UsdUVTexture")
            a.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(al[0]))
            a.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
            a.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            a.CreateOutput("r", Sdf.ValueTypeNames.Float)
            sf.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(a.ConnectableAPI(), "r")
            sf.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.3)
        sf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        m.CreateSurfaceOutput().ConnectToSource(sf.ConnectableAPI(), "surface")
        out[part] = m
    bark = next((v for k, v in out.items() if "bark" in k or "branches" in k), next(iter(out.values())))
    out["_bark"] = bark
    for k in list(out):  # parts with no texture of their own (dead branches, trunks) -> bark
        pass
    _TREE_MATS[key] = out
    return out


def _instancer(stage, nv, path, asset, sub, off, mat, items):
    """nv.point_instancer with a variant sub-prim reference and a recentring offset (spatially partitioned)."""
    from pxr import Gf, Sdf, UsdGeom

    if False and len(items) > 120:  # NO spatial partitioning: every cell got its own copy of the prototype (a 4-17 M-triangle
        # Poly Haven tree) and the preview jobs ran out of their 140 GB (Mountain 2026-10-20); one instancer per variant
        cells = {}
        for it in items:
            cells.setdefault((math.floor(it[1][0] / 160.0), math.floor(it[1][1] / 160.0)), []).append(it)
        if len(cells) > 1:
            for (cx, cy), its in sorted(cells.items()):
                sfx = f"Cell_{cx:+05d}_{cy:+05d}".replace("+", "p").replace("-", "n")
                _instancer(stage, nv, f"{path}/{sfx}", asset, sub, off, mat, its)
            return
    inst = UsdGeom.PointInstancer.Define(stage, path)
    proto = UsdGeom.Xform.Define(stage, f"{path}/Prototypes/P0")
    inner = UsdGeom.Xform.Define(stage, f"{path}/Prototypes/P0/m")
    inner.AddTranslateOp().Set(Gf.Vec3d(-off[0], -off[1], 0.0))
    # the WHOLE file is referenced (a sub-prim reference cut the variant off from its materials under /root/_materials:
    # untextured trees, Mountain preview 2026-10-20) and the two other variants are deactivated
    inner.GetPrim().GetReferences().AddReference(asset)
    if sub:
        me = sub.split("/")[-1]
        for letter in "abc":
            other = me[:-len("_a_LOD0")] + f"_{letter}_LOD0"
            if other != me:
                stage.OverridePrim(f"{path}/Prototypes/P0/m/{other}").SetActive(False)
    if isinstance(mat, dict):
        # the file's own materials (MaterialX node graphs under /root/_materials) render RED in RTX (Mountain preview
        # 2026-10-20): rebind every mesh / subset to our UsdPreviewSurface twin of the material it names
        from pxr import Usd, UsdShade
        for prim in Usd.PrimRange(inner.GetPrim()):
            rel = prim.GetRelationship("material:binding")
            if not rel:
                continue
            tg = rel.GetTargets()
            name = tg[0].name if tg else ""
            m_ = mat.get(name) or mat.get("_bark")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(m_)  # each subset keeps its own (twig vs bark)
    elif mat is not None:
        nv.bind(proto.GetPrim(), mat, strong=True)
    inst.CreatePrototypesRel().SetTargets([Sdf.Path(f"{path}/Prototypes/P0")])
    inst.CreateProtoIndicesAttr([0] * len(items))
    inst.CreatePositionsAttr([Gf.Vec3f(*it[1]) for it in items])
    inst.CreateOrientationsAttr([Gf.Quath(float(math.cos(it[2] / 2)), Gf.Vec3h(0.0, 0.0, float(math.sin(it[2] / 2)))) for it in items])
    inst.CreateScalesAttr([Gf.Vec3f(*it[3]) for it in items])
    pos = np.asarray([it[1] for it in items], float)
    mx = max(max(it[3]) for it in items)
    lo, hi = pos.min(axis=0) - 25.0 * mx, pos.max(axis=0) + 25.0 * mx
    inst.CreateExtentAttr([Gf.Vec3f(*lo.astype(float)), Gf.Vec3f(*hi.astype(float))])
