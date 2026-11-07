"""
Mountain v2 USD build (runs inside Isaac Sim): lighting, the banded terrain mesh, distant ranges, water, roads, sites,
pads, vegetation and the movable props. Everything comes from the plan (mountain_v2.plan) -- the flight job and every
render job build the identical scene.
"""
import json
import math
from pathlib import Path

import numpy as np

import coastal_props as cp
import mountain_nature as mn
import mountain_props as mp
import mountain_v2 as mv
import props

# Poly Haven textures whose roughness map is broken (all zero -> mirror-like ground, Coastal v2.1 Simulation_9)
BROKEN_ROUGHNESS_MAPS = {"aerial_grass_rock"}


# the gravel_road texture is orange-brown (mean linear albedo 0.19 / 0.10 / 0.05): roads and car parks read as rust-coloured
# strips; tinted to a grey-beige mountain gravel (~0.20 / 0.18 / 0.15)
GRAVEL_TINT = (1.05, 1.8, 2.9)


def _pbr(stage, nv, root, name, tex, roughness=0.95, tint=None, uv=1.0):
    import coastal_scene as cs
    return cs.create_pbr_material_fixed(stage, nv, f"/World/Materials/Mtn_{name}", root, tex, uv_scale=uv, tint=tint, use_normal=False,
                                        roughness=roughness, ior=1.05, roughness_map=tex not in BROKEN_ROUGHNESS_MAPS)


# ---- textured building skins ---------------------------------------------------------------------------------------
SKIN_CLASSES = ("building", "church", "hut", "bridge")
_ALB = {}


def _albedo(nv, root, tex):
    """Mean linear albedo of a texture's diffuse map (to tint it to an exact described colour)."""
    if tex not in _ALB:
        from PIL import Image
        a = np.asarray(Image.open(nv.texture_path(root, tex, "diffuse")).convert("RGB").resize((64, 64)), float) / 255.0
        a = np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)
        _ALB[tex] = a.reshape(-1, 3).mean(0)
    return _ALB[tex]


def _near(c, ref, tol=0.03):
    return all(abs(float(a) - float(b)) < tol for a, b in zip(c, ref))


def _skin_of(p):
    """(texture, tile metres, target colour) for a wall / roof / timber / stone part, or None (trim, glass, small detail)."""
    if p["shape"] not in ("box", "prism") or p.get("hidden"):
        return None
    if p["shape"] == "box" and (min(p["s"]) < 0.05 or max(p["s"]) < 1.2):
        return None
    m = p["mat"]
    if isinstance(m, str):
        if m in ("wood", "dark_wood", "door_wood"):
            return ("brown_planks_05", 2.0, props.COLORS[m][0])
        if m == "brick":
            return ("brick_4", 2.5, props.COLORS[m][0])
        if m == "barn_red":
            return ("dark_planks", 2.0, props.COLORS[m][0])
        if m in ("concrete", "gravel"):
            return ("concrete_wall_008", 3.0, props.COLORS[m][0])
        return None
    c = tuple(float(v) for v in m)
    for name, rc in props.ROOFS.items():
        if _near(c, rc, 0.005):
            return ("clay_roof_tiles", 2.0, c) if name == "red tile" else ("ceramic_roof_01", 2.0, c)
    if _near(c, props.WALLS["log"], 0.005) or _near(c, mp.WOOD, 0.005) or _near(c, mp.DARK_WOOD, 0.005) or _near(c, props.COLORS["dark_wood"][0], 0.005):
        return ("brown_planks_05", 2.0, c)
    if _near(c, props.WALLS["red brick"], 0.005):
        return ("brick_4", 2.5, c)
    if _near(c, mp.STONE, 0.005):
        return ("stacked_stone_wall", 3.0, c)
    if max(c) < 0.45 and max(c) - min(c) > 0.15:  # dark saturated colours (shutters, painted boards) -> painted planks
        return ("brown_planks_05", 2.0, c)
    if max(c) > 0.2:  # painted wall colours -> plaster
        return ("white_stucco", 3.0, c)
    return None


def _skin_material(stage, nv, root, tex, target, k):
    import coastal_scene as cs
    alb = _albedo(nv, root, tex)
    tint = tuple(float(np.clip(t / max(a, 1e-3), 0.0, 4.0)) for t, a in zip(target, alb))
    return cs.create_pbr_material_fixed(stage, nv, f"/World/Materials/Mtn_Skin_{k:03d}_{tex}", root, tex, uv_scale=1.0, tint=tint,
                                        use_normal=True, roughness=0.85 if "roof" not in tex else 0.6, ior=1.3, roughness_map=False)


def _skin_mesh(stage, nv, path, p, mat, tile, M):
    """UV mesh of a box / prism part (faces unshared -> flat shading; uv in metres / tile)."""
    from pxr import Gf, UsdGeom, Vt
    if p["shape"] == "box":
        hx, hy, hz = (v / 2.0 for v in p["s"])
        F = [[(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)], [(-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz)],
             [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)], [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)],
             [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)], [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)]]
        off = p["c"]
    else:
        x0, x1, hw, z0, rise = p["x0"], p["x1"], p["hw"], p["z0"], p["rise"]
        q = [(x0, -hw, z0), (x0, hw, z0), (x0, 0, z0 + rise), (x1, -hw, z0), (x1, hw, z0), (x1, 0, z0 + rise)]
        F = [[q[i] for i in f] for f in ([0, 2, 1], [3, 4, 5], [0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5])]
        off = (0.0, 0.0, 0.0)
    pts, uvs, idx, cnt = [], [], [], []
    for f in F:
        a = np.array(f, float)
        n = np.cross(a[1] - a[0], a[2] - a[0])
        n = n / (np.linalg.norm(n) or 1.0)
        if abs(n[2]) > 0.7:          # top / bottom (and roof slabs lying flat in their own frame)
            u_, v_ = a[:, 0], a[:, 1]
        elif abs(n[0]) > abs(n[1]):  # faces looking along x
            u_, v_ = a[:, 1], a[:, 2]
        else:
            u_, v_ = a[:, 0], a[:, 2]
        for k in range(len(f)):
            idx.append(len(pts))
            pts.append(Gf.Vec3f(*map(float, a[k])))
            uvs.append(Gf.Vec2f(float(u_[k] / tile), float(v_[k] / tile)))
        cnt.append(len(f))
    g = UsdGeom.Mesh.Define(stage, path)
    g.CreatePointsAttr(pts)
    g.CreateFaceVertexIndicesAttr(idx)
    g.CreateFaceVertexCountsAttr(cnt)
    g.CreateSubdivisionSchemeAttr().Set("none")
    pv = UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", nv.Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
    pv.Set(Vt.Vec2fArray(uvs))
    g.AddTranslateOp().Set(Gf.Vec3d(*map(float, off)))
    if any(p.get("rot") or (0, 0, 0)):
        g.AddRotateXYZOp().Set(Gf.Vec3f(*map(float, p["rot"])))
    nv.bind(g.GetPrim(), mat)
    g.GetPrim().CreateAttribute("x500:noCollision", nv.Sdf.ValueTypeNames.Bool).Set(True)
    return g


def build(stage, nv, pl, root):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    info = {}
    info.update(build_lighting(stage, nv, root, pl["sky"]))
    info.update(build_terrain(stage, nv, pl, root))
    info.update(build_far(stage, nv, pl, root))
    info.update(build_water(stage, nv, pl))
    info.update(build_roads(stage, nv, pl, root))
    UsdGeom.Xform.Define(stage, "/World/Mtn/Sites")
    import coastal_scene as cs

    tex_mats = {}
    skinned = 0
    for st in pl["statics"]:
        plain = [p for p in st["parts"] if p["shape"] != "tbox"]
        # realistic surfaces (user 2026-10-21 "maximum realistic"): walls, roofs, timber and stone of the buildings get Poly Haven
        # PBR textures (plaster, planks, clay / slate tiles, stone, concrete) tinted to the described colour, on UV meshes;
        # the original primitive stays as an invisible exact collider
        if st["cls"] in SKIN_CLASSES:
            out = []
            for i, p in enumerate(plain):
                sk = _skin_of(p)
                if sk is None:
                    out.append(p)
                    continue
                tex, tile, tint = sk
                key = (tex, tint)
                if key not in tex_mats:
                    tex_mats[key] = _skin_material(stage, nv, root, tex, tint, len(tex_mats))
                _skin_mesh(stage, nv, f"{st['path']}/s{i:03d}", p, tex_mats[key], tile, st["M"])
                out.append(dict(p, hidden=True))
                skinned += 1
            plain = out
        props.author(stage, nv, st["path"], dict(parts=plain), root_matrix=st["M"])
        for i, p in enumerate(p for p in st["parts"] if p["shape"] == "tbox"):
            if p["tex"] not in tex_mats:
                tex_mats[p["tex"]] = _pbr(stage, nv, root, f"T_{p['tex']}", p["tex"], 0.9, tint=GRAVEL_TINT if p["tex"] == "gravel_road" else None)
            cs.textured_box(nv, stage, f"{st['path']}/t{i:02d}", p["c"], p["s"], tex_mats[p["tex"]], tile_m=p["tile"])
            tp = stage.GetPrimAtPath(f"{st['path']}/t{i:02d}")
            tp.CreateAttribute("x500:noCollision", Sdf.ValueTypeNames.Bool).Set(True)
            cb = UsdGeom.Cube.Define(stage, f"{st['path']}/t{i:02d}_col")  # exact native box collider (Coastal lesson)
            cb.CreateSizeAttr(1.0)
            xf = UsdGeom.XformCommonAPI(cb)
            xf.SetTranslate(Gf.Vec3d(*map(float, p["c"])))
            xf.SetScale(Gf.Vec3f(*map(float, p["s"])))
            cb.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            UsdPhysics.CollisionAPI.Apply(cb.GetPrim())
    info.update(build_pads(stage, nv, pl))
    info.update(mn.build_vegetation(stage, nv, pl, root))
    rigid = []
    UsdGeom.Xform.Define(stage, "/World/Props")
    for m in mv.movables(pl):
        path = f"/World/Props/{m['group']}/{m['name']}"
        UsdGeom.Xform.Define(stage, f"/World/Props/{m['group']}")
        props.author(stage, nv, path, m["prop"], root_matrix=m["matrix"])
        rigid.append({"path": path, "name": f"{m['group']}/{m['name']}", "kind": m["kind"], "mass_kg": m["prop"]["mass"],
                      "pos": [m["matrix"][3][0], m["matrix"][3][1], m["matrix"][3][2]]})
    info["rigid_bodies"] = rigid
    info["textured_parts"] = skinned
    info["movable_props"] = len(rigid)
    info.update(dict(sites=len(pl["sites"]), statics=len(pl["statics"]), pads=len(pl["pads"]), spots=len(pl.get("spots", {}))))
    return info


# ================================================================================================== lighting
def build_lighting(stage, nv, root, sky):
    UsdLux, Gf = nv.UsdLux, nv.Gf
    hdri_id = sky["id"]
    hdr = Path(root) / "hdri" / hdri_id / f"{hdri_id}_2k.hdr"
    dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    dome.CreateTextureFileAttr().Set(str(hdr))
    dome.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateAngleAttr(0.55)
    cfg = mv.SKIES[hdri_id]
    info = json.loads((Path(root) / "hdri" / "sky_info.json").read_text())[hdri_id]
    elev = max(float(info["sun_elev_deg"]), cfg.get("min_elev", 12.0))
    az = float(sky["sun_az_deg"])
    r, g, b = info["sun_rgb"]
    col = (1.0, 0.97, 0.92) if cfg.get("diffuse") else (1.0, 0.55 + 0.45 * g, 0.35 + 0.65 * b)
    dome.CreateIntensityAttr(float(cfg["dome"]))
    dome.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float((az - info["sun_u_deg"]) % 360.0)))
    sun.CreateIntensityAttr(float(cfg["sun"]))
    sun.CreateColorAttr(Gf.Vec3f(*map(float, col)))
    sun.AddRotateXYZOp().Set(Gf.Vec3f(float(90.0 - elev), 0.0, float((az + 90.0) % 360.0)))
    if cfg.get("diffuse"):
        sun.CreateAngleAttr(8.0)
    return {"hdri": hdri_id, "sky": cfg["label"], "sun_elevation_deg": round(elev, 1), "sun_azimuth_deg": round(az, 1),
            "sun_color": [round(float(v), 3) for v in col]}


# ================================================================================================== terrain
def terrain_materials(stage, nv, pl, root, prefix="", haze=None):
    s = mv.SEASONS[pl["season"]]
    land = pl["landscape"]
    rock_tex = {"dolomite towers": "cliff_side", "rolling highlands": "lichen_rock", "gorge": "rock_face_03"}.get(land, "rock_face")
    h = (lambda t: t) if haze is None else (lambda t: tuple(min(1.0, c * haze[0] + haze[1][i]) for i, c in enumerate(t)))
    mats = {
        "meadow": _pbr(stage, nv, root, prefix + "Meadow", s["grass"], 0.97, tint=h(s["tint"]), uv=1.3),
        "alpine": _pbr(stage, nv, root, prefix + "Alpine", "sparse_grass", 0.97, tint=h((0.85, 0.85, 0.75)), uv=1.0),
        "forest": _pbr(stage, nv, root, prefix + "Forest", "forest_ground_04", 0.97, tint=h((0.9, 0.9, 0.9)), uv=1.4),
        "dirt": _pbr(stage, nv, root, prefix + "Dirt", "brown_mud_rocks_01", 0.97, tint=h((1.0, 1.0, 1.0)), uv=1.0),
        "scree": _pbr(stage, nv, root, prefix + "Scree", "aerial_rocks_02" if land != "dolomite towers" else "aerial_rocks_04", 0.95, tint=h((1, 1, 1)), uv=0.7),
        "rock": _pbr(stage, nv, root, prefix + "Rock", rock_tex, 0.93, tint=h((1, 1, 1)), uv=0.45),
        "snow": _pbr(stage, nv, root, prefix + "Snow", "snow_02", 0.75, tint=h((1.0, 1.0, 1.02)), uv=0.5),
        "riverbed": _pbr(stage, nv, root, prefix + "RiverBed", "river_small_rocks", 0.9, tint=h((0.8, 0.8, 0.8)), uv=3.0),
        "gravel": _pbr(stage, nv, root, prefix + "Gravel", "aerial_ground_rock", 0.95, tint=h((1, 1, 1)), uv=1.5),
    }
    return mats


def band_of(rel, sl, n, forest, rd, pl, under_water, zone):
    """Material band of one terrain quad from height above the valley floor, slope, noise and the forest mask."""
    v = pl["valley"]
    snow_l = pl["snowline"] + 18.0 * n
    tl = pl["treeline"] + 25.0 * n
    if under_water or (rd < v["rw"] + 1.0 and rel < 4.0):
        return "riverbed"
    if zone:
        return "gravel" if zone == "cut" else ("meadow" if sl < 0.3 else "dirt")
    if sl > 0.95 + 0.12 * n:
        return "rock" if not (rel > snow_l and sl < 1.25 and n > 0.2) else "snow"
    if rel > snow_l:
        return "snow" if sl < 1.0 else "rock"
    if pl["season"] == "early winter" and rel > 35 + 15 * n and sl < 0.8:
        return "snow"
    if sl > 0.68 + 0.1 * n:
        return "scree" if rel > tl - 40 else "dirt"
    if rel > tl + 30:
        return "scree" if (sl > 0.4 or n > 0.35) else "alpine"
    if rel > tl - 20:
        return "alpine" if n > -0.25 else "dirt"
    if forest > -0.12 and sl > 0.08 and rel > 8:
        return "forest"
    if sl > 0.55:
        return "dirt"
    return "meadow"


def _baked_material(stage, nv, path, diffuse, rough):
    Sdf, Gf, UsdShade = nv.Sdf, nv.Gf, nv.UsdShade
    mat = UsdShade.Material.Define(stage, path)
    sf = UsdShade.Shader.Define(stage, f"{path}/Surface")
    sf.CreateIdAttr("UsdPreviewSurface")
    sf.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    sf.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.05)  # terrain: no grazing-angle sheen (Coastal lesson)
    st = UsdShade.Shader.Define(stage, f"{path}/TexCoord")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    for name, f, out, inp, typ in (("Diffuse", diffuse, "rgb", "diffuseColor", Sdf.ValueTypeNames.Color3f),
                                   ("Rough", rough, "r", "roughness", Sdf.ValueTypeNames.Float)):
        t = UsdShade.Shader.Define(stage, f"{path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(f))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        if name == "Rough":
            t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        t.CreateOutput(out, Sdf.ValueTypeNames.Float3 if out == "rgb" else Sdf.ValueTypeNames.Float)
        sf.CreateInput(inp, typ).ConnectToSource(t.ConnectableAPI(), out)
    sf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(sf.ConnectableAPI(), "surface")
    return mat


def build_terrain(stage, nv, pl, root):
    """The terrain mesh in NB x NB tiles, each with its own baked ground texture (mountain_bake): soft natural blends of
    meadow, forest floor, moraine, scree, rock and snow instead of one material per terrain cell."""
    import os

    import mountain_bake as mb

    Gf, UsdGeom = nv.Gf, nv.UsdGeom
    flight_only = os.environ.get("X500_NO_BAKE") == "1"  # flight jobs render nothing: no 40-min texture bake there
    bdir = None if flight_only else mb.bake(pl)
    plain = nv.create_color_material(stage, "/World/Materials/Mtn_GroundPlain", (0.3, 0.33, 0.22), 0.95) if flight_only else None
    H, xs = pl["_H"], pl["_xs"]
    n = mv.GRID
    gy, gx = np.gradient(H, mv.STEP)
    NB = mb.NT
    k = (n - 1) // NB
    size = 2 * mv.EXTENT / NB
    UsdGeom.Xform.Define(stage, "/World/Terrain")
    for i in range(NB):
        for j in range(NB):
            c0, r0 = i * k, j * k
            sub = H[r0:r0 + k + 1, c0:c0 + k + 1]
            X, Y = np.meshgrid(xs[c0:c0 + k + 1], xs[r0:r0 + k + 1])
            pts = [Gf.Vec3f(float(a), float(b), float(c)) for a, b, c in zip(X.ravel(), Y.ravel(), sub.ravel())]
            x0, y0 = -mv.EXTENT + i * size, -mv.EXTENT + j * size
            uvs = [Gf.Vec2f(float((a - x0) / size), float((b - y0) / size)) for a, b in zip(X.ravel(), Y.ravel())]
            nr = np.stack([-gx[r0:r0 + k + 1, c0:c0 + k + 1].ravel(), -gy[r0:r0 + k + 1, c0:c0 + k + 1].ravel(), np.ones((k + 1) ** 2)], 1)
            nr /= np.linalg.norm(nr, axis=1, keepdims=True)
            m = k + 1
            rr, cc = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
            a = (rr * m + cc).ravel()
            tri = np.stack([a, a + 1, a + m + 1, a, a + m + 1, a + m], axis=1).ravel().tolist()
            mat = plain if flight_only else _baked_material(stage, nv, f"/World/Materials/Mtn_Ground_{i}_{j}", bdir / f"tile_{i}_{j}.jpg",
                                                              bdir / f"rough_{i}_{j}.png")
            mesh = nv.create_mesh(stage, f"/World/Terrain/T_{i}_{j}", pts, tri, [3] * (len(tri) // 3), uvs, mat, collision=False)
            mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, q)) for q in nr])
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return {"terrain": f"{NB}x{NB} tiles, baked ground texture {mb.RES} px per {size:.0f} m ({size / mb.RES * 100:.1f} cm/px)",
            "terrain_extent_m": mv.EXTENT, "bake": str(bdir) if bdir else "not baked in the flight job"}


FAR_COL = {"meadow": (0.27, 0.34, 0.15), "forest": (0.07, 0.10, 0.055), "alpine": (0.36, 0.37, 0.24), "scree": (0.48, 0.47, 0.45),
           "rock": (0.40, 0.39, 0.37), "snow": (0.86, 0.88, 0.92)}


def _far_colours(pl, X, Y, Hf, step):
    """Per-vertex colours of the distant ranges (same rules as the bake, smooth), hazed with distance."""
    S = mv.sstep
    v = pl["valley"]
    gy, gx = np.gradient(Hf, step)
    sl = np.hypot(gx, gy)
    rel = Hf - mv.floor_z(np.clip(X, -mv.EXTENT, mv.EXTENT), v)
    nz = mv.fbm(X / 400.0, Y / 400.0, v["seed"] + 409, 3)
    hl = pl["landscape"] == "rolling highlands"
    # snow caps only the highest part of the distant ranges (top 20 %, top 50 % in early winter)
    q = 50 if pl["season"] == "early winter" else 80
    snow_l = max(pl["snowline"] + 260.0, float(np.percentile(rel[np.hypot(X, Y) > 1500], q))) + 80 * nz
    # distant ranges: forest on their lower 40 % (they stand on deeper valleys than ours), alpine grass, rock, snow on top
    tl = max(pl["treeline"] + 120.0, float(np.percentile(rel[np.hypot(X, Y) > 1500], 40))) + 60 * nz
    w = {"snow": S(snow_l - 30, snow_l + 30, rel) * (1 - S(0.9, 1.3, sl)),
         "rock": S(0.8, 1.1, sl) + (0.0 if hl else 0.6 * S(tl + 120, tl + 260, rel)),
         "scree": (0.0 if hl else 0.4) * S(tl + 40, tl + 140, rel),
         "alpine": S(tl - 30, tl + 30, rel) * (1.0 if hl else 0.6),
         "forest": (1 - S(tl - 30, tl + 10, rel)) * S(0.06, 0.18, sl) * 1.2 * (0.5 + 0.5 * S(-0.3, 0.3, nz)),
         "meadow": (1 - S(tl - 30, tl + 10, rel)) * 0.6}
    if pl["landscape"] == "dolomite towers":
        FAR = dict(FAR_COL, rock=(0.58, 0.56, 0.52))
    else:
        FAR = FAR_COL
    w["rock"] *= 1 - w["snow"]
    tot = sum(w.values()) + 1e-6
    col = sum(w[k_][..., None] * np.asarray(FAR[k_]) for k_ in w) / tot[..., None]
    if pl["season"] == "autumn":
        col = col * np.asarray([1.12, 1.0, 0.85])
    d = np.hypot(X, Y)
    # light aerial perspective only (the first version washed the distant ranges into a pale wall, preview 2026-10-20)
    haze = (1.0 - np.exp(-np.maximum(d - 900.0, 0.0) / 9000.0)) * 0.35
    sky = np.asarray([0.55, 0.62, 0.74])
    return col * (1 - haze[..., None]) + sky * haze[..., None]


def far_texture(pl, N=1536):
    """Colour image of the whole far terrain (+-FAR, ~12 m/px), cached next to the ground tiles. (Per-vertex displayColor
    through a primvar reader rendered WHITE in RTX, preview 2026-10-20.)"""
    import mountain_bake as mb
    from PIL import Image
    f = mb.tile_dir(pl) / "far_colour.png"
    if f.exists():
        return f
    cs = np.linspace(-mv.FAR, mv.FAR, N)
    X, Y = np.meshgrid(cs, cs)
    Hf = mv.far_height(X, Y, pl)
    col = _far_colours(pl, X, Y, Hf, cs[1] - cs[0])
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp.png")
    Image.fromarray((np.clip(col[::-1], 0, 1) * 255 + 0.5).astype(np.uint8)).save(tmp)
    tmp.rename(f)
    return f


def build_far(stage, nv, pl, root):
    """Distant ranges to the horizon: a fine ring (12.5 m) out to 2.6 km and a coarse one (75 m) out to 9 km, coloured by one
    texture (snow, rock, scree, alpine grass, conifer canopy, meadow) with light aerial perspective."""
    import os
    Gf, UsdGeom, Sdf, UsdShade = nv.Gf, nv.UsdGeom, nv.Sdf, nv.UsdShade
    if os.environ.get("X500_NO_BAKE") == "1":  # flight job: the far ranges are never seen, no collider either
        return {"far_ranges": "skipped in the flight job"}
    tex = far_texture(pl)
    mat = UsdShade.Material.Define(stage, "/World/Materials/Mtn_FarColour")
    sf = UsdShade.Shader.Define(stage, "/World/Materials/Mtn_FarColour/Surface")
    sf.CreateIdAttr("UsdPreviewSurface")
    sf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.95)
    sf.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.05)
    st = UsdShade.Shader.Define(stage, "/World/Materials/Mtn_FarColour/TexCoord")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    t = UsdShade.Shader.Define(stage, "/World/Materials/Mtn_FarColour/Diffuse")
    t.CreateIdAttr("UsdUVTexture")
    t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(tex))
    t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
    t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
    t.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(t.ConnectableAPI(), "rgb")
    sf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(sf.ConnectableAPI(), "surface")
    out = {}
    for name, half, step, hole in (("FarRing/Mid", 2600.0, 12.5, mv.EXTENT), ("FarRing/Outer", mv.FAR, 75.0, 2600.0 - 75.0)):
        N = int(round(2 * half / step)) + 1
        cs = np.linspace(-half, half, N)
        X, Y = np.meshgrid(cs, cs)
        Hf = mv.far_height(X, Y, pl)
        inner = (np.abs(X) < mv.EXTENT) & (np.abs(Y) < mv.EXTENT)
        Hf = np.where(inner, Hf - 4.0, Hf)
        gy, gx = np.gradient(Hf, step)
        nr = np.stack([-gx.ravel(), -gy.ravel(), np.ones(N * N)], 1)
        nr /= np.linalg.norm(nr, axis=1, keepdims=True)
        r, c = np.meshgrid(np.arange(N - 1), np.arange(N - 1), indexing="ij")
        keep = ~((np.maximum(np.abs(cs[c]), np.abs(cs[c + 1])) < hole - step) & (np.maximum(np.abs(cs[r]), np.abs(cs[r + 1])) < hole - step))
        a = (r * N + c)[keep]
        tri = np.stack([a, a + 1, a + N + 1, a, a + N + 1, a + N], axis=1).ravel().tolist()
        uvs = [Gf.Vec2f(float((p + mv.FAR) / (2 * mv.FAR)), float((q + mv.FAR) / (2 * mv.FAR))) for p, q in zip(X.ravel(), Y.ravel())]
        mesh = nv.create_mesh(stage, f"/World/{name}", [Gf.Vec3f(float(p), float(q), float(z)) for p, q, z in zip(X.ravel(), Y.ravel(), Hf.ravel())],
                              tri, [3] * (len(tri) // 3), uvs, mat, collision=False)
        mesh.CreateNormalsAttr([Gf.Vec3f(*map(float, q)) for q in nr])
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        mesh.GetPrim().CreateAttribute("x500:noCollision", Sdf.ValueTypeNames.Bool).Set(True)
        out[name] = len(tri) // 3
    return {"far_ranges_km": mv.FAR / 1000.0, "far_triangles": out}


# ================================================================================================== water
def build_water(stage, nv, pl):
    Gf, UsdGeom = nv.Gf, nv.UsdGeom
    v = pl["valley"]
    # alpine water: dark green-blue, a soft (not mirror) reflection -- the first preview's lake read as a teal swimming pool
    mat = nv.create_color_material(stage, "/World/Materials/Mtn_Water", (0.03, 0.06, 0.06), 0.38, metallic=0.0)  # moving water: rough
    lake_mat = nv.create_color_material(stage, "/World/Materials/Mtn_LakeWater", (0.012, 0.035, 0.042), 0.16, metallic=0.0)
    lk = v.get("lake")
    # river ribbon (outside the lake)
    xs = np.arange(-mv.EXTENT - 400, mv.EXTENT + 400.1, 3.0)
    hw = v["rw"] + 4.5
    pts, idx, cnt = [], [], []
    rows = 0
    for x in xs:
        yr = float(mv.river_y(x, v))
        if lk:
            e = ((x - lk["x"]) / lk["L"]) ** 2
            if e < 1.0:
                if rows:
                    pass
        lvl = float(mv.river_level(x, v))
        dyx = float(mv.river_y(x + 1.5, v) - mv.river_y(x - 1.5, v)) / 3.0
        nx, ny = -dyx / math.hypot(1, dyx), 1 / math.hypot(1, dyx)
        for s in (-1, 1):
            pts.append(Gf.Vec3f(float(x + s * hw * nx), float(yr + s * hw * ny), lvl))
        rows += 1
    for k in range(rows - 1):
        x = xs[k]
        if lk and abs(x - lk["x"]) < lk["L"] * 0.97:
            continue  # the lake surface covers it (no z-fighting)
        a = 2 * k
        idx += [a, a + 2, a + 3, a + 1]
        cnt.append(4)
    m = nv.create_mesh(stage, "/World/River/Stream", pts, idx, cnt, None, mat, collision=False)
    m.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(pts))
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    bodies = []
    if lk:
        yv = float(mv.valley_y(lk["x"], v))
        bodies.append(("Lake", lk["x"], yv, lk["L"] * 1.3 + 20, float(mv.valley_w(lk["x"], v)) * 1.5 + 60, lk["z"]))
    for k, tn in enumerate(v.get("tarns", [])):
        bodies.append((f"Tarn{k + 1}", tn["x"], tn["y"], tn["r"] * 1.6, tn["r"] * 1.6, tn["z"]))
    tarn_of = {f"Tarn{k + 1}": tn for k, tn in enumerate(v.get("tarns", []))}
    for name, cx, cy, hx, hy, z in bodies:
        g = 3.0
        nxg, nyg = int(2 * hx / g) + 1, int(2 * hy / g) + 1
        gx = np.linspace(cx - hx, cx + hx, nxg)
        gy = np.linspace(cy - hy, cy + hy, nyg)
        P, I, C = [], [], []
        for y in gy:
            for x in gx:
                P.append(Gf.Vec3f(float(x), float(y), float(z)))
        for j in range(nyg - 1):
            for i in range(nxg - 1):
                xs4 = (gx[i], gx[i + 1])
                ys4 = (gy[j], gy[j + 1])
                if min(mv.terrain_height(xx, yy, pl) for xx in xs4 for yy in ys4) > z + 0.05:
                    continue
                mx, my = (xs4[0] + xs4[1]) / 2, (ys4[0] + ys4[1]) / 2
                inside = mv.in_lake_basin(mx, my, pl) if name == "Lake" else mv.in_tarn_basin(mx, my, tarn_of[name], pl)
                if not inside:  # only the basin holds water (low valley floor next to it stays dry land)
                    continue
                a = j * nxg + i
                I += [a, a + 1, a + nxg + 1, a + nxg]
                C.append(4)
        mm = nv.create_mesh(stage, f"/World/River/{name}", P, I, C, None, lake_mat, collision=False)
        mm.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * len(P))
        mm.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return {"water_bodies": [b[0] for b in bodies] + ["river"]}


# ================================================================================================== roads, pads
def build_roads(stage, nv, pl, root):
    Gf = nv.Gf
    if not pl["roads"]:
        return {}
    mat = _pbr(stage, nv, root, "Road", "gravel_road", 0.95, tint=GRAVEL_TINT, uv=1.0)
    nv.UsdGeom.Xform.Define(stage, "/World/Mtn/Roads")
    for k, r in enumerate(pl["roads"]):
        P = np.asarray(r["pts"], float)
        seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
        s = np.concatenate([[0.0], np.cumsum(seg)])
        if s[-1] < 2:
            continue
        ss = np.arange(0.0, s[-1] + 0.01, 2.0)
        xs, ys = np.interp(ss, s, P[:, 0]), np.interp(ss, s, P[:, 1])
        pts, uvs, idx, cnt = [], [], [], []
        hw = r["w"] / 2
        for i in range(len(ss)):
            i0, i1 = max(0, i - 1), min(len(ss) - 1, i + 1)
            dx, dy = xs[i1] - xs[i0], ys[i1] - ys[i0]
            L = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / L, dx / L
            for j, o in enumerate((-hw, -hw / 2, 0.0, hw / 2, hw)):
                x, y = xs[i] + nx * o, ys[i] + ny * o
                if abs(x) > mv.EXTENT - 1 or abs(y) > mv.EXTENT - 1:
                    z = mv.terrain_height(x, y, pl) - 5.0
                else:
                    z = mv.surface_z(x, y, pl) + 0.07
                pts.append(Gf.Vec3f(float(x), float(y), float(z)))
                uvs.append(Gf.Vec2f(float(j / 4 * r["w"] / 4.0), float(ss[i] / 4.0)))
        for i in range(len(ss) - 1):
            for j in range(4):
                a = i * 5 + j
                idx += [a, a + 1, a + 6, a + 5]
                cnt.append(4)
        m = nv.create_mesh(stage, f"/World/Mtn/Roads/road_{k}", pts, idx, cnt, uvs, mat, collision=False)
        m.GetPrim().CreateAttribute("x500:noCollision", nv.Sdf.ValueTypeNames.Bool).Set(True)
    return {"roads": len(pl["roads"])}


def build_pads(stage, nv, pl):
    slab = nv.create_color_material(stage, "/World/Materials/Mtn_PadSlab", (0.36, 0.36, 0.35), 0.85)
    paint = nv.create_color_material(stage, "/World/Materials/Mtn_PadPaint", (0.95, 0.80, 0.10), 0.5)
    white = nv.create_color_material(stage, "/World/Materials/Mtn_PadWhite", (0.92, 0.92, 0.90), 0.5)
    for k, p in pl["pads"].items():
        (px, py), z = p["xy"], p["z"]
        root = f"/World/Mtn/Pads/{k}"
        nv.cube(stage, f"{root}/slab", (px, py, z - 0.2), (15, 15, 0.42), slab)
        for j in range(24):
            a = 2 * math.pi * j / 24
            nv.cube(stage, f"{root}/ring{j:02d}", (px + 5.5 * math.cos(a), py + 5.5 * math.sin(a), z + 0.025), (1.3, 0.3, 0.02), paint, rotation_z=math.degrees(a) + 90)
        for (dx, sx, sy) in ((-1.4, 0.55, 4.4), (1.4, 0.55, 4.4), (0.0, 2.8, 0.55)):
            nv.cube(stage, f"{root}/H{int(dx * 10) + 20}{int(sx * 10)}", (px + dx, py, z + 0.026), (sx, sy, 0.02), white)
    return {"landing_pads": len(pl["pads"])}
