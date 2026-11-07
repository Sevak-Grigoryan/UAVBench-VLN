"""
Mountain v2 terrain texture BAKE (user review 2026-10-20: "the mountains is simple texture, i need more realistic").

One material per terrain cell (the first version) gave blocky snow / rock edges and flat, repeated colour. Instead every
simulation gets its own ground texture: the terrain is split into NT x NT tiles, each with a RES x RES diffuse map
(1400 m / 4 / 4096 = 8.5 cm per pixel) that BLENDS high-resolution Poly Haven materials per pixel:

  weights   meadow / alpine grass / forest floor / moraine / scree / rock / snow / river bed / levelled ground, from the
            height above the valley floor, slope, curvature (gullies hold snow and scree, ridges are bare), the forest
            mask and multi-scale noise -- soft, irregular boundaries, no straight band edges
  sampling  every material at two scales with a per-region random offset (no visible tiling), colour variation noise
  shading   ambient occlusion from curvature (darker gullies, lighter ridges), rock strata on cliffs, darker ground
            under every tree (the forest reads as forest from above), wet dark bank along the river

Pure numpy + PIL (CPU). bake(pl, out_dir) writes tile_<i>_<j>.jpg (+ rough_<i>_<j>.png); the scene build loads them.
Deterministic for a plan, cached on disk by plan index + attempt.
"""
import math
from pathlib import Path

import numpy as np
from PIL import Image

import mountain_v2 as mv

NT = 4
RES = 4096
LOW = 1024                      # weights / shading are computed at LOW and upsampled (smooth fields)
SRC = Path("/mnt/weka/sgrigoryan/assets/polyhaven-mountain-v2/bake_src")
CACHE = Path("/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_bake")
VERSION = "bake-v7"

# material -> (texture ids (two variants blended by noise), real tile size in metres, roughness)
ROCKS = {"dolomite towers": (["marble_cliff_05", "rock_boulder_dry"], 14.0),
         "rolling highlands": (["mossy_rock", "aerial_rocks_02"], 10.0),
         "gorge": (["rock_face_04", "mossy_rock"], 12.0)}


def materials(pl):
    land, season = pl["landscape"], pl["season"]
    meadow = {"summer": ["leafy_grass", "grass_ground"], "early summer": ["leafy_grass", "grass_ground"],
              "autumn": ["withered_grass", "grass_ground"], "early winter": ["withered_grass", "sparse_grass"]}[season]
    rock, rock_m = ROCKS.get(land, (["rock_face_04", "mossy_rock"], 12.0))
    return {
        "meadow": (meadow, 7.0, 0.97), "alpine": (["sparse_grass", "aerial_grass_rock"], 9.0, 0.97),
        "forest": (["forest_leaves_02", "forest_ground_04"], 7.0, 0.97), "moraine": (["aerial_rocks_01", "dry_ground_rocks"], 11.0, 0.95),
        "scree": (["gray_rocks", "rocks_ground_01"], 9.0, 0.93), "rock": (rock, rock_m * 1.5, 0.9), "snow": (["snow_01", "snow_01"], 18.0, 0.6),
        "riverbed": (["river_small_rocks", "rocks_ground_01"], 6.0, 0.7), "levelled": (["rock_ground_02", "grass_ground"], 8.0, 0.95),
    }


# target mean colour of each material (linear-ish sRGB 0-1): the textures keep their detail, their average colour is moved
# to what alpine ground looks like from a drone (Poly Haven scans differ a lot in white balance)
TARGET = {"meadow": {"summer": (0.28, 0.35, 0.15), "early summer": (0.26, 0.37, 0.13), "autumn": (0.43, 0.38, 0.19),
                     "early winter": (0.40, 0.36, 0.25)},
          "alpine": {"summer": (0.34, 0.37, 0.21), "early summer": (0.31, 0.38, 0.18), "autumn": (0.45, 0.38, 0.22),
                     "early winter": (0.42, 0.38, 0.29)},
          "forest": (0.16, 0.15, 0.10), "moraine": (0.43, 0.41, 0.37), "scree": (0.50, 0.49, 0.47), "snow": (0.87, 0.89, 0.93),
          "riverbed": (0.36, 0.34, 0.31), "levelled": (0.46, 0.43, 0.37),
          "rock": {"dolomite towers": (0.60, 0.58, 0.53), "rolling highlands": (0.37, 0.38, 0.34), "gorge": (0.42, 0.40, 0.37),
                   "default": (0.41, 0.40, 0.38)}}


def target(k, pl):
    t = TARGET[k]
    if isinstance(t, dict):
        t = t.get(pl["season"]) or t.get(pl["landscape"]) or t["default"]
    return np.asarray(t, np.float32)


SEASON_GRADE = {"summer": (1.0, 1.0, 1.0), "early summer": (1.0, 1.0, 1.0), "autumn": (1.0, 1.0, 1.0), "early winter": (1.0, 1.0, 1.0)}
_TEX = {}


def _tex(name, px):
    """Source texture resized so that px texels span one tile (prefiltered: no aliasing at the bake resolution)."""
    key = (name, px)
    if key not in _TEX:
        im = Image.open(SRC / f"{name}_2k.jpg").convert("RGB")
        _TEX[key] = np.asarray(im.resize((px, px), Image.LANCZOS), dtype=np.float32) / 255.0
    return _TEX[key]


def _mean(texs, tile_m, ppm):
    """Mean colour of a material's blend (both textures equally)."""
    m = [_tex(t, max(16, int(round(tile_m * ppm)))).reshape(-1, 3).mean(axis=0) for t in texs]
    return np.mean(m, axis=0).astype(np.float32)


def _sample(name, tile_m, X, Y, px_per_m, off, ang=0.0):
    """Texture at world (X, Y), tile_m metres per repeat, rotated by ang (two differently rotated samples blended by noise
    hide the repetition grid)."""
    px = max(16, int(round(tile_m * px_per_m)))
    t = _tex(name, px)
    c, s_ = math.cos(ang), math.sin(ang)
    Xr, Yr = c * X - s_ * Y, s_ * X + c * Y
    u = (np.floor((Xr + off[0]) / tile_m * px).astype(np.int64)) % px
    v = (np.floor((Yr + off[1]) / tile_m * px).astype(np.int64)) % px
    return t[v, u]


def _up(a, n):
    """Bilinear upsample of a LOW x LOW float field to n x n."""
    return np.asarray(Image.fromarray(a.astype(np.float32), mode="F").resize((n, n), Image.BILINEAR), dtype=np.float32)


def _bilinear(H, xs, X, Y):
    step = xs[1] - xs[0]
    fx = np.clip((X - xs[0]) / step, 0, len(xs) - 1.001)
    fy = np.clip((Y - xs[0]) / step, 0, len(xs) - 1.001)
    c, r = fx.astype(int), fy.astype(int)
    u, w = fx - c, fy - r
    return (H[r, c] * (1 - u) * (1 - w) + H[r, c + 1] * u * (1 - w) + H[r + 1, c] * (1 - u) * w + H[r + 1, c + 1] * u * w)


def tile_dir(pl):
    return CACHE / f"{pl['index']:03d}_a{pl['attempt']}_{VERSION}"


def weights(pl, X, Y, tree_shade):
    """Per-material blend weights (dict of LOW x LOW arrays, summing to 1) + AO / strata shading."""
    v = pl["valley"]
    H, xs = pl["_H"], pl["_xs"]
    h = _bilinear(H, xs, X, Y)
    gy, gx = np.gradient(H, mv.STEP)
    sl = np.hypot(_bilinear(gx, xs, X, Y), _bilinear(gy, xs, X, Y))
    # curvature (Laplacian of a smoothed height): > 0 concave gully, < 0 convex ridge
    lap = (np.roll(H, 3, 0) + np.roll(H, -3, 0) + np.roll(H, 3, 1) + np.roll(H, -3, 1) - 4 * H) / (3 * mv.STEP) ** 2
    curv = _bilinear(lap, xs, X, Y)
    rel = h - mv.floor_z(X, v)
    s = v["seed"]
    n1 = mv.fbm(X / 60.0, Y / 60.0, s + 501, 4)
    n2 = mv.fbm(X / 15.0, Y / 15.0, s + 503, 3)
    n3 = mv.fbm(X / 220.0, Y / 220.0, s + 505, 3)
    forest = mv.fbm(X / 140.0, Y / 140.0, s + 301, 4)
    snow_l = pl["snowline"] + 25 * n1 + 10 * n2 + np.clip(curv, -0.05, 0.05) * 300.0  # snow lower in gullies
    tl = pl["treeline"] + 25.0 * mv.fbm(X / 90.0, Y / 90.0, s + 307, 2)
    S = mv.sstep
    # bare rock: only real cliffs below the tree line, already moderately steep ground above it
    rock_s = 0.74 + 0.24 * (1 - S(tl - 40, tl + 30, rel)) + 0.10 * n1 + 0.05 * n2 - np.clip(curv, -0.05, 0.05) * 1.5
    w = {}
    w["rock"] = S(rock_s - 0.12, rock_s + 0.12, sl) * (1 - 0.6 * S(0.0, 0.2, n2 - 0.25) * (sl < 1.2))
    w["snow"] = S(snow_l - 15, snow_l + 15, rel) * (1 - S(0.95, 1.25, sl + 0.1 * n2))
    if pl["season"] == "early winter":
        w["snow"] = np.maximum(w["snow"], S(25, 70, rel + 20 * n1) * (1 - S(0.75, 1.05, sl)) * S(-0.3, 0.1, n2 + 0.3))
    w["scree"] = S(0.45, 0.7, sl + 0.15 * n2) * (1 - w["rock"]) * S(tl - 60, tl + 10, rel) + 0.25 * S(0.005, 0.03, curv) * S(tl, tl + 60, rel) * (1 - S(0.2, 0.6, sl))
    w["alpine"] = S(tl - 25, tl + 15, rel) * (1 - S(0.4, 0.65, sl))
    w["forest"] = 1.3 * S(-0.3, -0.05, forest) * (1 - S(tl - 30, tl, rel)) * S(4, 14, rel) * (1 - S(0.95, 1.15, sl))
    w["moraine"] = 0.12 * S(0.5, 0.8, sl) * (1 - S(tl, tl + 40, rel)) + 0.45 * S(0.25, 0.55, n3) * S(tl - 10, tl + 40, rel)
    w["meadow"] = (1 - S(tl - 30, tl, rel)) * (1 - S(0.85, 1.05, sl)) * 0.9
    # river bed and lake floor
    rd = np.abs(Y - mv.river_y(X, v))
    w["riverbed"] = (1 - S(v["rw"] + 0.5, v["rw"] + 3.5 + 2 * n2, rd)) * (1 - S(3, 8, rel))
    lk = v.get("lake")
    # lake / tarn floor ONLY inside the basin the water is drawn in (mv.in_lake_basin / in_tarn_basin, vectorised here):
    # "below the lake level" alone painted the whole lower valley floor downstream of the lake as grey lakebed gravel
    # (Simulation 39 preview, 2026-10-21)
    if lk:
        Wv = mv.valley_w(X, v)
        e = (((X - lk["x"]) / lk["L"]) ** 2 + ((Y - mv.valley_y(X, v)) / (Wv * lk["wf"])) ** 2) * \
            (1.0 + 0.22 * mv.fbm(X / 70.0, Y / 70.0, v["seed"] + 801, 3))
        w["riverbed"] = np.maximum(w["riverbed"], ((h < lk["z"] + 0.3) & (e < 1.1)) * 1.0)
    for tn in v.get("tarns", []):
        e = np.hypot(X - tn["x"], Y - tn["y"]) / tn["r"] * (1.0 + 0.2 * mv.fbm(X / 30.0, Y / 30.0, v["seed"] + 803, 3))
        w["riverbed"] = np.maximum(w["riverbed"], ((h < tn["z"] + 0.3) & (e < 1.1)) * 1.0)
    # levelled site ground
    lev = np.zeros_like(h)
    for (cx, cy, yaw, hx, hy, z, f, kind) in pl["zones"]:
        c, sn = math.cos(-yaw), math.sin(-yaw)
        uu, ww = c * (X - cx) - sn * (Y - cy), sn * (X - cx) + c * (Y - cy)
        dd = np.hypot(np.maximum(np.abs(uu) - hx, 0), np.maximum(np.abs(ww) - hy, 0))
        lev = np.maximum(lev, 1 - S(0.0, 4.0 + 0.4 * f, dd + 3 * n2))
    w["levelled"] = lev * 0.8
    # snow and water win over everything below them
    ws = w["snow"]
    for k in ("meadow", "alpine", "forest", "moraine", "scree"):
        w[k] = w[k] * (1 - ws)
    w["rock"] = w["rock"] * (1 - 0.85 * ws)
    for k in ("meadow", "alpine", "forest", "moraine", "scree", "rock", "snow", "levelled"):
        w[k] = w[k] * (1 - w["riverbed"])
    tot = sum(w.values()) + 1e-4
    for k in w:
        w[k] = (w[k] + (1e-4 if k == "meadow" else 0.0)) / tot
    # shading: ambient occlusion from curvature, strata on rock, darker under trees, wet river banks
    ao = 1.0 - np.clip(curv * 2.5, -0.12, 0.25)
    strata = 1.0 + 0.05 * np.sin(h / 2.7 + 2.0 * n1) * w["rock"] * S(0.9, 1.2, sl)
    wet = 1.0 - 0.25 * (1 - S(v["rw"] + 2, v["rw"] + 7, rd)) * (1 - w["riverbed"])
    ao = 1.0 - (1.0 - ao) * (1.0 - 0.6 * w["snow"])  # snow fills gullies: much less contrast there
    shade = ao * strata * wet * (1.0 - 0.45 * tree_shade * (1 - w["snow"]))
    return w, shade, n1, n3


def _tree_shade(pl, x0, y0, size):
    """LOW x LOW map of tree crowns' shade (soft discs under every tree, sized by the tree)."""
    import mountain_nature as mn
    m = np.zeros((LOW, LOW), np.float32)
    px = LOW / size
    for k, x, y, z, yaw, sc in pl["vegetation"]:
        kind = mn.PROTOS[k][4]
        if kind not in ("tree", "young", "broadleaf"):
            continue
        r = {"tree": 3.2, "young": 2.2, "broadleaf": 2.4}[kind] * sc
        cx, cy = (x - x0) * px, (y - y0) * px
        rp = r * px
        if cx < -rp or cy < -rp or cx > LOW + rp or cy > LOW + rp:
            continue
        i0, i1 = max(0, int(cx - rp)), min(LOW, int(cx + rp) + 1)
        j0, j1 = max(0, int(cy - rp)), min(LOW, int(cy + rp) + 1)
        if i1 <= i0 or j1 <= j0:
            continue
        jj, ii = np.mgrid[j0:j1, i0:i1]
        d = np.hypot(ii + 0.5 - cx, jj + 0.5 - cy) / max(rp, 1e-3)
        m[j0:j1, i0:i1] = np.maximum(m[j0:j1, i0:i1], np.clip(1.0 - d, 0, 1) ** 0.7)
    return m


def bake_tile(pl, i, j, out):
    size = 2 * mv.EXTENT / NT
    x0, y0 = -mv.EXTENT + i * size, -mv.EXTENT + j * size
    lo = (np.arange(LOW) + 0.5) / LOW * size
    XL, YL = np.meshgrid(x0 + lo, y0 + lo)
    w, shade, n1, n3 = weights(pl, XL, YL, _tree_shade(pl, x0, y0, size))
    hi = (np.arange(RES) + 0.5) / RES * size
    XH, YH = np.meshgrid(x0 + hi, y0 + hi)
    ppm = RES / size
    mats = materials(pl)
    rng = np.random.default_rng(pl["valley"]["seed"] + 97 * i + 13 * j)
    col = np.zeros((RES, RES, 3), np.float32)
    rough = np.zeros((RES, RES), np.float32)
    mix = _up(mv.sstep(-0.3, 0.3, mv.fbm(XL / 35.0, YL / 35.0, pl["valley"]["seed"] + 601, 3)), RES)[..., None]
    mix2 = _up(mv.sstep(-0.25, 0.25, mv.fbm(XL / 22.0, YL / 22.0, pl["valley"]["seed"] + 607, 3)), RES)[..., None]
    for k, (texs, tile_m, rgh) in mats.items():
        wk = _up(w[k], RES)
        if float(wk.max()) < 1e-3:
            continue
        if k == "snow":  # procedural wind-packed snow: the scanned snow textures carry boot prints / dark debris
            ang = float(rng.uniform(0, 6.28))
            rip = np.sin((math.cos(ang) * XH + math.sin(ang) * YH) * 2.2 + 3.0 * _up(mv.fbm(XL / 9.0, YL / 9.0, pl["valley"]["seed"] + 611, 2), RES))
            fine = _up(mv.fbm(XL / 2.5, YL / 2.5, pl["valley"]["seed"] + 613, 2), RES)
            tex = target(k, pl)[None, None, :] * (0.95 + 0.035 * fine[..., None] + 0.015 * rip[..., None]) * np.asarray([1.0, 1.0, 1.0], np.float32)
            col += wk[..., None] * tex
            rough += wk * rgh
            continue
        a = _sample(texs[0], tile_m, XH, YH, ppm, rng.uniform(0, 100, 2), float(rng.uniform(0, 6.28)))
        a2 = _sample(texs[0], tile_m * 1.31, XH, YH, ppm, rng.uniform(0, 100, 2), float(rng.uniform(0, 6.28)))
        b = _sample(texs[1], tile_m * 1.7, XH, YH, ppm, rng.uniform(0, 100, 2), float(rng.uniform(0, 6.28)))
        a = a * (1 - mix2) + a2 * mix2
        tex = a * (1 - mix) + b * mix
        tex = tex * (target(k, pl) / np.maximum(_mean(texs, tile_m, ppm), 1e-3))
        col += wk[..., None] * tex
        rough += wk * rgh
    grade = np.asarray(SEASON_GRADE[pl["season"]], np.float32)
    var = 1.0 + 0.10 * _up(n1, RES)[..., None] + 0.06 * _up(n3, RES)[..., None] * np.asarray([1.0, 0.6, -0.3], np.float32)
    col = col * _up(shade, RES)[..., None] * var * grade
    col = np.clip(col, 0, 1)
    Image.fromarray((col[::-1] * 255 + 0.5).astype(np.uint8)).save(out / f"tile_{i}_{j}.jpg", quality=93)
    Image.fromarray((np.clip(rough[::-1], 0, 1) * 255).astype(np.uint8)).resize((RES // 4, RES // 4)).save(out / f"rough_{i}_{j}.png")


def bake(pl, force=False):
    out = tile_dir(pl)
    if not force and (out / "DONE").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    for i in range(NT):
        for j in range(NT):
            bake_tile(pl, i, j, out)
    (out / "DONE").write_text(VERSION)
    return out


if __name__ == "__main__":
    import sys
    import time
    sys.path[:0] = [str(Path(__file__).resolve().parents[2] / "common/pipeline"), str(Path(__file__).resolve().parents[1])]
    import missions as mm
    import pilot
    for a in sys.argv[1:]:
        t = time.time()
        pl, m, L = mm.mission_and_plan(int(a), mv, pilot)
        print(a, bake(pl, force="--force" in sys.argv), round(time.time() - t, 1), "s", flush=True)
