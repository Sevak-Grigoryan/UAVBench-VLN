"""
Mountain v2 -- the Easy Mountain dataset (2026-10-20): a different mountain landscape for every simulation.

User brief: "generate simulations for Mountains but i need diversity in environment, different tasks, different
trajectories and missions" / "the environment is simple, do more realistic and diversity" (the old Simulation_0_0_6
mountain was one authored peak with two pads).

plan(idx, attempt) is a pure function (numpy RNG, no Isaac Sim) that decides everything:
  * a LANDSCAPE TYPE: alpine valley with a river, glacial-lake valley, narrow gorge, dolomite towers, rolling
    highlands, high cirque with a tarn -- each with its own valley width, ridge heights, peak shapes, ruggedness;
  * a SEASON (summer, early summer, autumn, early winter) that moves the snow line, tree line and meadow colours;
  * a sky / time of day (Poly Haven puresky HDRIs, sun light aligned with the sky's sun);
  * the terrain: a meandering valley floor (river channel, optional lake) between two ridges with 3-7 named peaks,
    passes (saddles), outer high valleys beyond the ridges and distant ranges to the horizon;
  * 6-10 SITES chosen per landscape: alpine village with a church, farm with animals, campsite, trailhead car park,
    mountain-rescue station with a helicopter, chapel, mountain hut, cable car (stations, pylons, cables, cabins),
    summit cross, radio mast, observatory, wind turbines, quarry, paraglider launch, stone bridge, lakeside boathouse,
    shepherd's hut with a flock;
  * landing pads (painted) and unmarked take-off / landing SPOTS (meadows, hut terraces, summit plateau, car park...);
  * every movable object (people, animals, vehicles) as a rigid body with mass, set down by the static-equilibrium test.

terrain_height(x, y, pl) is the EXACT surface of the rendered terrain mesh (same grid, same triangulation), so props,
pads and the pilot's height above ground agree with what is rendered.
"""
import math

import numpy as np

import coastal_props as cp
import mountain_props as mp
import props

EXTENT = 700.0         # high-detail terrain half-extent [m]
GRID = 561             # vertices per side (2.5 m spacing)
FAR = 9000.0           # distant ranges half-extent [m]
STEP = 2 * EXTENT / (GRID - 1)
N_SIMS = 100

# ---------------------------------------------------------------------------------------------------- tasks
# the mission's task is decided first: the landscape must contain what the task needs (a lake to cross, a cable car...)
TASK_PLAN = (["circle"] * 18 + ["inspect"] * 15 + ["pass_over"] * 12 + ["follow_valley"] * 10 + ["follow_ridge"] * 10
             + ["cross_pass"] * 10 + ["land_at"] * 13 + ["follow_cable_car"] * 6 + ["cross_lake"] * 6)


def task_of(idx):
    order = np.random.default_rng(727272).permutation(len(TASK_PLAN))
    return TASK_PLAN[int(order[idx - 1])]


LAND_PLAN = {"alpine valley": 22, "glacial lake valley": 17, "gorge": 14, "dolomite towers": 16, "rolling highlands": 15, "high cirque": 16}
SEASON_PLAN = {"summer": 38, "early summer": 20, "autumn": 24, "early winter": 18}
_LAND_CACHE = {}


def landscape_of(idx):
    """Balanced landscape types over the 100 simulations; the tasks that need a lake / a cable car get a landscape that has one."""
    if not _LAND_CACHE:
        rng = np.random.default_rng(818181)
        left = dict(LAND_PLAN)
        need = {"cross_lake": ["glacial lake valley", "high cirque"],
                "follow_cable_car": ["alpine valley", "glacial lake valley", "dolomite towers", "high cirque"]}
        order = sorted(range(1, N_SIMS + 1), key=lambda i: (task_of(i) not in need, float(rng.random())))
        for i in order:
            ok = [k for k in sorted(left) if left[k] > 0 and k in need.get(task_of(i), list(LAND_PLAN))]
            w = np.array([left[k] for k in ok], float)
            k = str(rng.choice(ok, p=w / w.sum()))
            left[k] -= 1
            _LAND_CACHE[i] = k
    return _LAND_CACHE[idx]


def season_of(idx):
    seq = [k for k in sorted(SEASON_PLAN) for _ in range(SEASON_PLAN[k])]
    order = np.random.default_rng(919191).permutation(len(seq))
    return seq[int(order[idx - 1])]


LANDSCAPES = {
    # valley floor half-width, ridge heights, wall length, wall shape exponent, ruggedness, snow/tree lines above the floor
    "alpine valley": dict(W=(55, 95), R=(240, 340), Dw=(330, 400), q=(1.4, 1.75), prof="pow", rug=(0.12, 0.17), snow=(240, 300), tree=(140, 190),
                          peaks=(2, 4), sharp=(1.5, 2.1), weight=2.4),
    "glacial lake valley": dict(W=(85, 125), R=(230, 320), Dw=(320, 390), q=(1.4, 1.75), prof="pow", rug=(0.11, 0.16), snow=(230, 290), tree=(130, 180),
                                peaks=(2, 3), sharp=(1.5, 2.0), weight=1.5),
    "gorge": dict(W=(24, 36), R=(170, 260), Dw=(120, 170), q=(0.45, 0.7), rug=(0.15, 0.2), snow=(260, 330), tree=(150, 220),
                  peaks=(1, 3), sharp=(1.4, 1.9), weight=1.5),
    "dolomite towers": dict(W=(60, 100), R=(150, 230), Dw=(240, 300), q=(1.3, 1.6), prof="pow", rug=(0.10, 0.14), snow=(300, 360), tree=(120, 170),
                            peaks=(3, 4), sharp=(0.5, 0.8), weight=1.7),
    "rolling highlands": dict(W=(80, 130), R=(90, 170), Dw=(250, 330), q=(1.3, 1.7), rug=(0.04, 0.07), snow=(900, 900), tree=(60, 120),
                              peaks=(2, 3), sharp=(2.2, 2.8), weight=1.6),
    "high cirque": dict(W=(70, 110), R=(260, 380), Dw=(200, 260), q=(0.75, 1.0), rug=(0.14, 0.19), snow=(110, 170), tree=(20, 50),
                        peaks=(3, 4), sharp=(1.4, 1.9), weight=1.3),
}
SEASONS = {"summer": dict(snow=0, grass="leafy_grass", tint=(0.62, 0.78, 0.42), weight=3.0),
           "early summer": dict(snow=-60, grass="leafy_grass", tint=(0.55, 0.80, 0.38), weight=1.5),
           "autumn": dict(snow=-25, grass="withered_grass", tint=(0.95, 0.80, 0.55), weight=1.5),
           "early winter": dict(snow=-200, grass="withered_grass", tint=(0.80, 0.72, 0.55), weight=1.0)}
SKIES = {
    "kloofendal_48d_partly_cloudy_puresky": dict(label="midday, partly cloudy", dome=900, sun=1050, weight=3),
    "kloofendal_43d_clear_puresky": dict(label="midday, clear sky", dome=850, sun=1100, weight=2),
    "kloofendal_38d_partly_cloudy_puresky": dict(label="afternoon, partly cloudy", dome=880, sun=1050, weight=2),
    "qwantani_noon_puresky": dict(label="noon, clear sky", dome=850, sun=1150, weight=1.5),
    "qwantani_mid_morning_puresky": dict(label="mid-morning", dome=850, sun=1050, weight=1.5),
    "qwantani_morning_puresky": dict(label="early morning, low warm sun", dome=800, sun=950, weight=1.2),
    "qwantani_afternoon_puresky": dict(label="afternoon", dome=850, sun=1050, weight=1.5),
    "qwantani_late_afternoon_puresky": dict(label="late afternoon, low warm sun", dome=800, sun=950, weight=1.2),
    "syferfontein_18d_clear_puresky": dict(label="late afternoon, clear sky", dome=800, sun=950, weight=1.0),
    "qwantani_sunset_puresky": dict(label="sunset", dome=750, sun=700, min_elev=9.0, weight=0.8),
    "kloofendal_overcast_puresky": dict(label="overcast", dome=1250, sun=180, diffuse=True, weight=1.2),
    "kloofendal_misty_morning_puresky": dict(label="misty morning", dome=1150, sun=150, diffuse=True, weight=0.8),
    "kloofendal_28d_misty_puresky": dict(label="hazy, thin cloud", dome=1100, sun=500, weight=0.8),
    "pizzo_pernice_puresky": dict(label="alpine afternoon, scattered clouds", dome=900, sun=1000, weight=1.5),
    "rocky_ridge_puresky": dict(label="clear mountain sky", dome=850, sun=1050, weight=1.0),
    "table_mountain_1_puresky": dict(label="morning, broken clouds", dome=900, sun=1000, weight=1.0),
    "snow_field_puresky": dict(label="cold clear winter sky", dome=900, sun=1000, weight=0.8),
}
FROZEN = set()  # no published Mountain v2 simulation yet: every plan may still change


# ================================================================================================== noise (numpy)
def _lat(ix, iy, seed):
    with np.errstate(over="ignore"):
        return _lat_(np.atleast_1d(ix), np.atleast_1d(iy), seed).reshape(np.shape(ix))


def _lat_(ix, iy, seed):
    h = (ix.astype(np.uint64) * np.uint64(374761393) + iy.astype(np.uint64) * np.uint64(668265263)
         + np.uint64(seed * 1442695041 % (1 << 32)))
    h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
    h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(0xFFFF)).astype(np.float64) / 32767.5 - 1.0


def vnoise(x, y, seed):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ix, iy = np.floor(x), np.floor(y)
    fx, fy = x - ix, y - iy
    ix, iy = ix.astype(np.int64), iy.astype(np.int64)
    u = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
    v = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
    a, b = _lat(ix, iy, seed), _lat(ix + 1, iy, seed)
    c, d = _lat(ix, iy + 1, seed), _lat(ix + 1, iy + 1, seed)
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def fbm(x, y, seed, octaves=4, gain=0.5):
    out, amp, f, norm = 0.0, 1.0, 1.0, 0.0
    for k in range(octaves):
        out = out + amp * vnoise(x * f + 17.3 * k, y * f - 9.1 * k, seed + 101 * k)
        norm += amp
        amp *= gain
        f *= 2.03
    return out / norm


def ridged(x, y, seed, octaves=5):
    out, amp, f, w, norm = 0.0, 1.0, 1.0, 1.0, 0.0
    for k in range(octaves):
        n = 1.0 - np.abs(vnoise(x * f + 5.7 * k, y * f + 3.3 * k, seed + 37 * k))
        n = n * n * w
        w = np.clip(n * 1.6, 0.0, 1.0)
        out = out + amp * n
        norm += amp
        amp *= 0.5
        f *= 2.1
    return out / norm


def sstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def smax(a, b, frac=0.18):
    k = frac * np.minimum(np.abs(a), np.abs(b)) + 1e-6
    h = np.clip(0.5 + 0.5 * (a - b) / k, 0.0, 1.0)
    return b + (a - b) * h + k * h * (1 - h)


def rot(yaw, a, b):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * a - s * b, s * a + c * b


# ================================================================================================== valley geometry
def valley_y(x, v):
    x = np.asarray(x, float)
    return v["A1"] * np.sin(x / v["L1"] + v["P1"]) + v["A2"] * np.sin(x / v["L2"] + v["P2"])


def valley_w(x, v):
    return v["W"] * (1.0 + 0.22 * np.sin(np.asarray(x, float) / v["L3"] + v["P3"]))


def floor_z(x, v):
    """Valley floor height along the valley (the river falls with it); flat around a lake."""
    x = np.asarray(x, float)
    z = v["slope"] * x
    lk = v.get("lake")
    if lk:
        w = sstep(lk["L"] + 120, lk["L"] + 10, np.abs(x - lk["x"]))
        z = z * (1 - w) + (lk["z"] + 1.3) * w
    return z


def river_y(x, v):
    x = np.asarray(x, float)
    return valley_y(x, v) + 0.32 * valley_w(x, v) * np.sin(x / v["Lr"] + v["Pr"])


def river_level(x, v):
    lvl = floor_z(x, v) - 0.55
    lk = v.get("lake")
    if lk:
        w = sstep(lk["L"] + 40, lk["L"] - 10, np.abs(np.asarray(x, float) - lk["x"]))
        lvl = lvl * (1 - w) + lk["z"] * w
    return lvl


def base_height(X, Y, pl):
    """Terrain without the site zones (vectorised over arrays X, Y)."""
    v = pl["valley"]
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    yv = valley_y(X, v)
    W = valley_w(X, v)
    zf = floor_z(X, v)
    dy = Y - yv
    north = dy > 0
    d = np.abs(dy) - W
    far_f = sstep(EXTENT + 300.0, EXTENT + 1500.0, np.maximum(np.abs(X), np.abs(Y)))  # 0 on the flown terrain
    R0 = np.where(north, v["Rn"], v["Rs"])
    R = R0 * (0.80 + 0.20 * vnoise(X / 260.0, np.where(north, 40.0, -40.0), v["seed"] + 5))
    for p in v["passes"]:
        side = (dy > 0) == (p["side"] > 0)
        R = R * np.where(side, 1.0 - p["depth"] * np.exp(-((X - p["x"]) / p["w"]) ** 2), 1.0)
    # far away the ridge height no longer follows these x-only profiles (they drew north-south stripes to the horizon)
    R = R * (1 - far_f) + R0 * (0.8 + 0.25 * fbm(X / 1800.0, Y / 1800.0, v["seed"] + 9, 3)) * far_f
    t = np.maximum(d, 0.0) / v["Dw"]
    if v.get("prof") == "pow":  # glaciated valley wall: gentle forested foot, steepening to the crest
        tt = np.minimum(t, 1.0)
        prof_in = tt ** v["q"] * (1.0 - 0.15 * (1 - tt) * tt)
    else:
        prof_in = (0.5 - 0.5 * np.cos(math.pi * np.minimum(t, 1.0))) ** v["q"]
    beyond = np.maximum(t - 1.0, 0.0)
    prof = prof_in - v["drop"] * sstep(0.0, v["out"], beyond) + v["far_rise"] * sstep(v["out"] + 0.3, v["out"] + 2.2, beyond)
    # warped coordinates for the erosion pattern: spurs and gullies run DOWN the valley sides (elongated across the valley),
    # so the walls read as ridges and couloirs, not as a smooth trough
    wx = X + 75.0 * fbm(X / 260.0, Y / 260.0, v["seed"] + 1, 3)
    wy = Y + 75.0 * fbm(X / 260.0 + 7.3, Y / 260.0 - 3.1, v["seed"] + 2, 3)
    dd = np.abs(wy - valley_y(wx, v)) - valley_w(wx, v)
    spur = ridged(wx / v["spur_l"], dd / 420.0 + np.where(north, 31.7, -17.3), v["seed"] + 41, 4)
    iso = ridged(wx / 280.0, wy / 280.0, v["seed"] + 43, 4)
    spur = 0.65 * spur + 0.35 * iso
    gul = np.clip(1.0 + v["spur_a"] * (spur - 0.42), 0.35, 1.4)
    # beyond the detailed terrain the across-valley gully pattern fades out (it drew stripes to the horizon); r < EXTENT + 300
    # is untouched, so the flown terrain does not change
    gul = gul * (1 - far_f) + far_f
    Hm = R * prof * gul
    for pk in v["peaks"]:
        dist = np.hypot(X - pk["x"], Y - pk["y"]) / pk["r"]
        if pk["shape"] == "tower":  # dolomite tower group: irregular, cracked, several summits, scree apron
            r = pk["r"]
            tx = X + 0.35 * r * fbm(X / (0.7 * r), Y / (0.7 * r), v["seed"] + pk["seed"], 3)
            ty = Y + 0.35 * r * fbm(X / (0.7 * r) + 3.1, Y / (0.7 * r) - 1.7, v["seed"] + pk["seed"] + 1, 3)
            core = 0.0
            for (ox, oy, sr, sh) in pk["subs"]:
                dd_ = np.hypot(tx - pk["x"] - ox * r, ty - pk["y"] - oy * r) / (sr * r)
                core = np.maximum(core, sh * pk["h"] * sstep(1.05, 0.3, dd_) ** 0.8)  # steep, not sheer (sheer walls smeared the texture)
            crack = 0.94 + 0.06 * ridged(tx / 40.0, ty / 40.0, v["seed"] + pk["seed"] + 5, 2)  # soft fissures (0.18 made crystal spikes)
            core = core * np.where(core > 0.5 * pk["h"], crack, 1.0)
            dist = np.hypot(X - pk["x"], Y - pk["y"]) / r
            apron = pk["h"] * 0.38 * np.clip(1.0 - dist / 1.7, 0.0, 1.0) ** 1.4
            bump = np.maximum(core, apron)
        else:  # pyramid-like peak: elongated along its own axis, ridged faces carved by the spur pattern (round cones looked fake)
            ca, sa = math.cos(pk.get("ang", 0.0)), math.sin(pk.get("ang", 0.0))
            u_, w_ = (ca * (X - pk["x"]) + sa * (Y - pk["y"])) / pk.get("asp", 1.0), (-sa * (X - pk["x"]) + ca * (Y - pk["y"])) * pk.get("asp", 1.0)
            dist = np.hypot(u_, w_) / pk["r"] * (1.0 + 0.25 * fbm(X / 140.0, Y / 140.0, v["seed"] + 91, 3))
            bump = pk["h"] * np.clip(1.0 - dist, 0.0, 1.0) ** pk["sharp"] * np.clip(0.8 + 0.45 * (spur - 0.42), 0.55, 1.15)
        Hm = smax(Hm, bump)
    rug = v["rug"] * Hm * (ridged(wx / 120.0, wy / 120.0, v["seed"] + 11) - 0.35)
    rug = rug + 0.35 * v["rug"] * Hm * (fbm(wx / 35.0, wy / 35.0, v["seed"] + 19, 3))
    Hm = Hm + rug * sstep(2.0, 30.0, Hm)
    rd = np.abs(Y - river_y(X, v))
    micro = 0.7 * fbm(X / 55.0, Y / 55.0, v["seed"] + 23, 3) * sstep(v["rw"] + 4, v["rw"] + 16, rd) * (1 - sstep(0, 12, Hm))
    H = zf + Hm + micro
    H = H - v["chan"] * (1.0 - sstep(v["rw"], v["rw"] + 5.0, rd)) * (1 - sstep(10, 30, Hm))
    lk = v.get("lake")
    if lk:
        e = (((X - lk["x"]) / lk["L"]) ** 2 + (dy / (W * lk["wf"])) ** 2) * (1.0 + 0.22 * fbm(X / 70.0, Y / 70.0, v["seed"] + 801, 3))
        cut = (2.0 + lk["depth"] * np.sqrt(np.clip(1.0 - e, 0.0, 1.0))) * sstep(1.06, 0.9, e)
        H = np.where(e < 1.1, np.minimum(H, lk["z"] + 1.3 - cut), H)
    for tn in v.get("tarns", []):
        e = np.hypot(X - tn["x"], Y - tn["y"]) / tn["r"] * (1.0 + 0.2 * fbm(X / 30.0, Y / 30.0, v["seed"] + 803, 3))
        flat = sstep(2.2, 1.2, e)
        H = H * (1 - flat) + (tn["z"] + 1.0) * flat
        cut = (1.8 + 5.0 * np.sqrt(np.clip(1.0 - e, 0.0, 1.0))) * sstep(1.06, 0.9, e)
        H = np.where(e < 1.1, np.minimum(H, tn["z"] + 1.0 - cut), H)
    # the river is never buried: a peak, tower apron or rough bump over its course is cut back to a gorge around the water
    # (a stone bridge sat inside a 17 m rock hump over the river, physics audit 2026-10-21)
    # the cap rises without bound away from the river (1.2 m per m: a gorge wall), so it only cuts what stands over the
    # water; a saturating cap (+40 m) flattened every summit (caught by the cable-car placement failing, 2026-10-21);
    # applied after the lake / tarn shaping, which levelled the ground around a tarn over the river channel
    cap = river_level(X, v) - 1.4 + 2.0 * sstep(v["rw"] - 1.0, v["rw"] + 4.0, rd) + 1.2 * np.maximum(rd - v["rw"] - 4.0, 0.0)
    H = np.minimum(H, cap)
    return H


def far_height(X, Y, pl):
    """Distant ranges beyond the detailed terrain: the same landscape, plus higher snowy ranges with distance."""
    v = pl["valley"]
    H = base_height(X, Y, pl)
    r = np.maximum(np.abs(X), np.abs(Y))
    grow = sstep(EXTENT + 900.0, EXTENT + 4500.0, r)  # distant ranges rise further out (they walled in the valley ends)
    dy = np.abs(Y - valley_y(X, v)) - valley_w(X, v) * 1.6
    keep_valley = sstep(-20.0, 250.0, dy)  # the valley stays open to the horizon at both ends
    # broad massifs (warped, low-frequency) with ridged crests: the 900 m ridged noise alone drew a maze of walls
    wx = X + 600.0 * fbm(X / 3000.0, Y / 3000.0, v["seed"] + 73, 3)
    wy = Y + 600.0 * fbm(X / 3000.0 + 5.1, Y / 3000.0 - 2.3, v["seed"] + 75, 3)
    massif = 0.5 + 0.5 * fbm(wx / 2600.0, wy / 2600.0, v["seed"] + 77, 4)
    ranges = v["far_amp"] * (0.25 + 0.75 * massif * (0.55 + 0.45 * ridged(wx / 1400.0, wy / 1400.0, v["seed"] + 71, 5))) * keep_valley
    return H + grow * ranges


def apply_zones(H, xs, ys, zones):
    for (cx, cy, yaw, hx, hy, z, f, kind) in zones:
        R = math.hypot(hx, hy) + f
        i0, i1 = np.searchsorted(xs, cx - R), np.searchsorted(xs, cx + R)
        j0, j1 = np.searchsorted(ys, cy - R), np.searchsorted(ys, cy + R)
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(xs[i0:i1], ys[j0:j1])
        c, s = math.cos(-yaw), math.sin(-yaw)
        u, w_ = c * (X - cx) - s * (Y - cy), s * (X - cx) + c * (Y - cy)
        d = np.hypot(np.maximum(np.abs(u) - hx, 0.0), np.maximum(np.abs(w_) - hy, 0.0))
        w = 1.0 - sstep(0.0, f, d)
        sub = H[j0:j1, i0:i1]
        if kind == "cut":      # only lowers (quarry benches)
            H[j0:j1, i0:i1] = np.minimum(sub, sub * (1 - w) + z * w)
        else:
            H[j0:j1, i0:i1] = sub * (1 - w) + z * w
    return H


def compute_heightmap(pl, zones=True):
    xs = np.linspace(-EXTENT, EXTENT, GRID)
    X, Y = np.meshgrid(xs, xs)
    H = base_height(X, Y, pl)
    if zones:
        H = apply_zones(H, xs, xs, pl["zones"])
        # the river channel itself stays open after the site zones too (a bridge's abutment fall-off raised the bed under
        # the bridge by 0.4 m, audit 2026-10-21); only the channel (rd < rw + 3), so abutments and banks keep their shape
        v = pl["valley"]
        rd = np.abs(Y - river_y(X, v))
        ch = river_level(X, v) - 1.4 + 2.0 * sstep(v["rw"] - 1.0, v["rw"] + 4.0, rd)
        H = np.where(rd < v["rw"] + 3.0, np.minimum(H, ch), H)
    pl["_H"] = H
    pl["_xs"] = xs
    return H


def terrain_height(x, y, pl):
    """Exact height of the terrain MESH at (x, y): the grid is triangulated (a, b, d) + (a, d, c) like build_terrain."""
    H = pl["_H"]
    fx, fy = (x + EXTENT) / STEP, (y + EXTENT) / STEP
    if not (0 <= fx < GRID - 1 and 0 <= fy < GRID - 1):
        return float(far_height(np.array([x]), np.array([y]), pl)[0])
    c, r = int(fx), int(fy)
    u, w = fx - c, fy - r
    a, b, cc, d = H[r, c], H[r, c + 1], H[r + 1, c], H[r + 1, c + 1]
    if u >= w:
        return float(a + u * (b - a) + w * (d - b))
    return float(a + w * (cc - a) + u * (d - cc))


def terrain_heights(x, y, pl):
    x, y = np.asarray(x, float), np.asarray(y, float)
    H = pl["_H"]
    fx = np.clip((x + EXTENT) / STEP, 0, GRID - 1.0001)
    fy = np.clip((y + EXTENT) / STEP, 0, GRID - 1.0001)
    c, r = fx.astype(int), fy.astype(int)
    u, w = fx - c, fy - r
    a, b, cc, d = H[r, c], H[r, c + 1], H[r + 1, c], H[r + 1, c + 1]
    return np.where(u >= w, a + u * (b - a) + w * (d - b), a + w * (cc - a) + u * (d - cc))


def slope_at(x, y, pl, e=3.0):
    gx = (terrain_height(x + e, y, pl) - terrain_height(x - e, y, pl)) / (2 * e)
    gy = (terrain_height(x, y + e, pl) - terrain_height(x, y - e, pl)) / (2 * e)
    return math.hypot(gx, gy)


def in_lake_basin(x, y, pl):
    """Inside the carved lake basin (same noisy shoreline as base_height). Water is drawn ONLY there: drawing it over all
    ground below the lake level flooded the lower valley floor beyond the lake (Mountain Simulation 39: the rescue station
    and its helicopter stood under the water sheet, user report 2026-10-21)."""
    v = pl["valley"]
    lk = v.get("lake")
    if not lk:
        return False
    W = float(valley_w(x, v))
    e = (((x - lk["x"]) / lk["L"]) ** 2 + ((y - float(valley_y(x, v))) / (W * lk["wf"])) ** 2) * \
        (1.0 + 0.22 * float(fbm(np.array([x / 70.0]), np.array([y / 70.0]), v["seed"] + 801, 3)[0]))
    return e < 1.1


def in_tarn_basin(x, y, tn, pl):
    v = pl["valley"]
    e = math.hypot(x - tn["x"], y - tn["y"]) / tn["r"] * (1.0 + 0.2 * float(fbm(np.array([x / 30.0]), np.array([y / 30.0]), v["seed"] + 803, 3)[0]))
    return e < 1.1


def water_level(x, y, pl):
    """Water surface at (x, y) if there is water there (river, lake, tarn), else None."""
    v = pl["valley"]
    lk = v.get("lake")
    if lk and terrain_height(x, y, pl) < lk["z"] and in_lake_basin(x, y, pl):
        return lk["z"]
    for tn in v.get("tarns", []):
        if terrain_height(x, y, pl) < tn["z"] and in_tarn_basin(x, y, tn, pl):
            return tn["z"]
    if abs(y - float(river_y(x, v))) < v["rw"] + 4.5:
        lvl = float(river_level(x, v))
        if terrain_height(x, y, pl) < lvl:
            return lvl
    return None


def _in_rect(x, y, r):
    cx, cy, yaw, hx, hy = r[:5]
    u, v = rot(-yaw, x - cx, y - cy)
    return abs(u) <= hx and abs(v) <= hy


def surface_z(x, y, pl, water=False):
    """Top of the solid under (x, y): terrain, or a deck (terrace, helipad, bridge, jetty) above it. water=True also
    counts a water surface -- the 'ground' a drone measures its height above."""
    h = terrain_height(x, y, pl)
    for r in pl["decks"]:
        if r[5] > h and _in_rect(x, y, r):
            h = r[5]
    if water:
        wl = water_level(x, y, pl)
        if wl is not None:
            h = max(h, wl)
    return h


def canopy_z(x, y, pl):
    """Highest tree top near (x, y) (10 m cells), or the terrain where there are no trees."""
    cg = pl.get("_canopy")
    g = terrain_height(x, y, pl)
    if cg is None:
        return g
    i, j = int((x + EXTENT) // 10), int((y + EXTENT) // 10)
    if 0 <= i < cg.shape[1] and 0 <= j < cg.shape[0]:
        return max(g, float(cg[j, i]))
    return g


def side_words(y_rel):
    return "north" if y_rel > 0 else "south"


def end_words(x):
    return "western" if x < -230 else "eastern" if x > 230 else "middle"


# ================================================================================================== plan
def plan(idx, attempt=0):
    rng = np.random.default_rng(8_000_009 * idx + 23 + 7_919 * attempt)
    task = task_of(idx)
    land, season = landscape_of(idx), season_of(idx)  # fixed per simulation (a re-plan keeps them: balanced dataset)
    L = LANDSCAPES[land]
    U = lambda k: float(rng.uniform(*L[k]))  # noqa: E731
    v = dict(seed=int(rng.integers(1, 1 << 20)), A1=float(rng.uniform(30, 95)), L1=float(rng.uniform(260, 420)), P1=float(rng.uniform(0, 6.28)),
             A2=float(rng.uniform(8, 25)), L2=float(rng.uniform(95, 160)), P2=float(rng.uniform(0, 6.28)), W=U("W"), L3=float(rng.uniform(150, 260)),
             P3=float(rng.uniform(0, 6.28)), slope=float(rng.uniform(-0.03, 0.03)), Rn=U("R"), Rs=U("R"), Dw=U("Dw"), q=U("q"), rug=U("rug"),
             drop=float(rng.uniform(0.2, 0.45)) if land != "gorge" else float(rng.uniform(0.0, 0.08)), out=float(rng.uniform(0.7, 1.2)),
             far_rise=float(rng.uniform(0.4, 1.0)), far_amp=float(rng.uniform(450, 1100)) if land != "rolling highlands" else float(rng.uniform(250, 600)),
             Lr=float(rng.uniform(60, 110)), Pr=float(rng.uniform(0, 6.28)), rw=float(rng.uniform(5, 9)) if land != "gorge" else float(rng.uniform(6, 10)),
             chan=float(rng.uniform(1.8, 2.4)), passes=[], peaks=[], tarns=[],
             spur_l=float(rng.uniform(170, 320)), spur_a=float(rng.uniform(0.6, 1.0)) if land in ("gorge", "high cirque") else
             float(rng.uniform(0.35, 0.6)) if land != "rolling highlands" else float(rng.uniform(0.25, 0.45)), prof=L.get("prof", "cos"))
    if land == "gorge":
        v["Lr"] = 1e9  # the river runs down the middle of the narrow floor
    if land == "rolling highlands":
        v["rw"] = float(rng.uniform(3.5, 5.5))
    pl = dict(index=idx, attempt=attempt, task=task, landscape=land, season=season, valley=v, zones=[], decks=[], solids=[], sites=[],
              pads={}, statics=[], movables=[], obstacles=[], refs=[], landmarks={}, roads=[], cables=[])
    s = SEASONS[season]
    pl["snowline"] = U("snow") + s["snow"]
    pl["treeline"] = U("tree") + (-20 if season == "early winter" else 0)
    # lake
    if land == "glacial lake valley" or (land == "high cirque" and task == "cross_lake"):
        xc = float(rng.uniform(-260, 260))
        v["lake"] = dict(x=xc, L=float(rng.uniform(170, 260)), z=0.0, wf=float(rng.uniform(0.85, 0.98)), depth=float(rng.uniform(5, 10)))
        v["lake"]["z"] = float(v["slope"] * xc) - 0.2
    # passes (a saddle in the ridge on one or both sides)
    for side in (1, -1):
        if rng.random() < (0.85 if side == 1 else 0.6) or task == "cross_pass":
            v["passes"].append(dict(side=side, x=float(rng.uniform(-330, 330)), w=float(rng.uniform(70, 120)), depth=float(rng.uniform(0.35, 0.55))))
    # peaks: on the ridges (and a few on the shoulders), away from the passes
    n_pk = int(rng.integers(L["peaks"][0], L["peaks"][1] + 1))
    for side in (1, -1):
        xs_taken = [p["x"] for p in v["passes"] if p["side"] == side]
        for k in range(n_pk if side == 1 else max(1, n_pk - int(rng.integers(0, 2)))):
            for _ in range(80):
                x = float(rng.uniform(-560, 560))
                if all(abs(x - t) > 170 for t in xs_taken):
                    xs_taken.append(x)
                    break
            else:
                continue
            frac = float(rng.uniform(0.85, 1.12))
            dist = float(valley_w(x, v)) + frac * v["Dw"]
            y = float(valley_y(x, v)) + side * dist
            R = v["Rn"] if side == 1 else v["Rs"]
            if land == "dolomite towers":
                h = float(rng.uniform(1.3, 1.7)) * R
                subs = [(0.0, 0.0, 1.0, 1.0)] + [(float(rng.uniform(-0.8, 0.8)), float(rng.uniform(-0.8, 0.8)), float(rng.uniform(0.35, 0.6)),
                                                  float(rng.uniform(0.6, 0.92))) for _ in range(int(rng.integers(1, 4)))]
                pk = dict(x=x, y=y, h=h, r=float(rng.uniform(80, 130)), shape="tower", sharp=U("sharp"), side=side, subs=subs,
                          seed=int(rng.integers(1, 9999)))
            else:
                h = float(rng.uniform(1.12, 1.5)) * R
                pk = dict(x=x, y=y, h=h, r=float(rng.uniform(170, 260)) * (1.3 if land == "rolling highlands" else 1.0),
                          shape="cone", sharp=U("sharp") * 0.8, side=side, ang=float(rng.uniform(0, math.pi)), asp=float(rng.uniform(1.0, 1.6)))
            v["peaks"].append(pk)
    # tarn in an outer high valley / the cirque floor
    if land == "high cirque" and "lake" not in v:
        for _ in range(60):
            x = float(rng.uniform(-350, 350))
            side = int(rng.choice([1, -1]))
            y = float(valley_y(x, v)) + side * float(rng.uniform(0.0, 0.5)) * float(valley_w(x, v))
            if all(math.hypot(x - p["x"], y - p["y"]) > p["r"] * 0.9 for p in v["peaks"]):
                v["tarns"].append(dict(x=x, y=y, r=float(rng.uniform(45, 75)), z=0.0))
                break
    elif rng.random() < 0.45:
        for _ in range(60):
            x = float(rng.uniform(-450, 450))
            side = int(rng.choice([1, -1]))
            y = float(valley_y(x, v)) + side * (float(valley_w(x, v)) + v["Dw"] * (1.0 + v["out"] * float(rng.uniform(0.35, 0.7))))
            if abs(y) < EXTENT - 90 and all(math.hypot(x - p["x"], y - p["y"]) > p["r"] * 0.8 for p in v["peaks"]):
                v["tarns"].append(dict(x=x, y=y, r=float(rng.uniform(30, 55)), z=0.0))
                break
    for tn in v["tarns"]:
        h0 = base_height(np.array([tn["x"]]), np.array([tn["y"]]), dict(valley={**v, "tarns": []}))[0]
        tn["z"] = float(h0) - 0.5
    compute_heightmap(pl, zones=False)
    _find_peaks(pl)
    # ---- sites
    import mountain_sites as ms

    ms.plan_sites(pl, rng)
    compute_heightmap(pl, zones=True)
    ms.populate_sites(pl, rng)
    sm = sorted(pl["summits"], key=lambda p: -p["z"])
    for rank, p in enumerate(sm):
        p["rank"] = rank
    _summit_words(pl, sm)  # sites on summits take their words from the summit, before pads / spots refer to them
    ms.plan_pads_and_spots(pl, rng)
    ms.dry_roads(pl)
    _prune_overlaps(pl)
    keep, dropped = [], []
    for m, mv in zip(pl["movables"], movables(pl)):
        (keep if mv["static_check"]["stable"] else dropped).append(m)
    pl["movables"] = keep
    pl["dropped_unstable"] = [(m["group"], m["name"], m["kind"]) for m in dropped]
    _landmarks(pl)
    import mountain_nature as mn

    mn.plan_vegetation(pl)
    srng = np.random.default_rng(66_000_017 * idx + 5)
    ids = sorted(SKIES)
    if season == "early winter":
        wsk = np.array([SKIES[k]["weight"] * (2.0 if k in ("snow_field_puresky", "kloofendal_overcast_puresky") else 1.0) for k in ids], float)
    else:
        wsk = np.array([SKIES[k]["weight"] for k in ids], float)
    pl["sky"] = dict(id=str(srng.choice(ids, p=wsk / wsk.sum())), sun_az_deg=float(round(srng.uniform(0, 360), 1)))
    return pl


def _find_peaks(pl):
    """Summits = the highest grid point near every planned peak; plus the pass (saddle) points."""
    v = pl["valley"]
    H, xs = pl["_H"], pl["_xs"]
    out = []
    for k, pk in enumerate(v["peaks"]):
        r = pk["r"] * 0.45
        i0, i1 = np.searchsorted(xs, pk["x"] - r), np.searchsorted(xs, pk["x"] + r)
        j0, j1 = np.searchsorted(xs, pk["y"] - r), np.searchsorted(xs, pk["y"] + r)
        if i1 - i0 < 3 or j1 - j0 < 3:
            continue
        sub = H[j0:j1, i0:i1]
        jj, ii = np.unravel_index(int(np.argmax(sub)), sub.shape)
        x, y, z = float(xs[i0 + ii]), float(xs[j0 + jj]), float(sub[jj, ii])
        if abs(x) > EXTENT - 40 or abs(y) > EXTENT - 40:
            continue
        out.append(dict(name=f"peak_{len(out) + 1}", x=x, y=y, z=z, side=pk["side"], shape=pk["shape"], r=pk["r"]))
    pl["summits"] = out
    cols = []
    for p in v["passes"]:
        x = p["x"]
        best = None
        for d in np.linspace(0.6, 1.4, 41):
            y = float(valley_y(x, v)) + p["side"] * (float(valley_w(x, v)) + d * v["Dw"])
            if abs(y) > EXTENT - 30:
                continue
            z = terrain_height(x, y, pl)
            if best is None or z > best[2]:
                best = (x, y, z)
        if best:
            # walk along the crest a little to its lowest point
            cols.append(dict(name=f"pass_{len(cols) + 1}", x=best[0], y=best[1], z=best[2], side=p["side"],
                             words=f"the pass in the {side_words(p['side'])} ridge"))
    pl["cols"] = cols


def _landmarks(pl):
    """Named things an instruction can refer to: every site, every summit, the passes, the lake / tarns, the river."""
    v = pl["valley"]
    L = pl["landmarks"]
    sm = sorted(pl["summits"], key=lambda p: -p["z"])
    for rank, p in enumerate(sm):  # summits first: a site on a summit takes its words from the summit
        p["rank"] = rank
    _summit_words(pl, sm)
    for s in pl["sites"]:
        L[s["name"]] = dict(xy=s["xy"], top_z=s["top"], base_z=s["base_z"], words=s["words"], kind=s["kind"], size_r=s["size_r"], box=s.get("box"))
    for p in sm:
        rank = p["rank"]
        hz = p["z"]
        L[p["name"]] = dict(xy=(p["x"], p["y"]), top_z=hz, base_z=hz - 60.0, words=p["words"], kind="summit", size_r=55.0,
                            box=(p["x"], p["y"], 0.0, 45.0, 45.0, hz), summit=True, rank=rank, height_above_floor=round(hz - float(floor_z(p["x"], v)), 0))
    for c in pl["cols"]:
        side = side_words(c["y"] - float(valley_y(c["x"], v)))
        L[c["name"]] = dict(xy=(c["x"], c["y"]), top_z=c["z"], base_z=c["z"] - 20, words=f"the pass in the {side} ridge", kind="pass", size_r=40.0,
                            box=(c["x"], c["y"], 0.0, 30.0, 30.0, c["z"]))
        c["words"] = L[c["name"]]["words"]
    _cable_and_water(pl, L, v)
    _unique_words(pl)


def _summit_words(pl, sm):
    """Unique, natural names: the highest summit is 'the highest ...'; the others are named by their side of the valley and
    their order along it ('the westernmost peak on the north side', 'the second peak from the west on the south side')."""
    v = pl["valley"]
    ords = ["first", "second", "third", "fourth", "fifth"]
    for p in sm:
        p["_side"] = side_words(p["y"] - float(valley_y(p["x"], v)))
    for p in sm:
        rank = p["rank"]
        snowy = p["z"] - float(floor_z(p["x"], v)) > pl["snowline"] + 15
        shape = "tower" if p["shape"] == "tower" else ("rounded" if pl["landscape"] == "rolling highlands" else "rocky")
        what = {"tower": "rock tower", "rounded": "hill", "rocky": "peak"}[shape]
        adj = "snow-capped " if snowy else ("pale " if shape == "tower" else "rounded " if shape == "rounded" else "")
        side = p["_side"]
        same = sorted([q for q in sm if q["_side"] == side], key=lambda q: q["x"])
        k = same.index(p)
        if rank == 0:
            pos = f"the highest {adj}{what}"
        elif len(same) == 1:
            pos = f"the {adj}{what}"
        elif k == 0:
            pos = f"the westernmost {adj}{what}"
        elif k == len(same) - 1:
            pos = f"the easternmost {adj}{what}"
        else:
            pos = f"the {ords[k]} {adj}{what} from the west"
        where = f"on the {side} side of the valley"
        on_top = next((s for s in pl["sites"] if s.get("summit") == p["name"]), None)
        if on_top is not None:
            thing = {"summit cross": "the summit cross", "radio mast": "the radio mast", "observatory": "the observatory"}[on_top["kind"]]
            words = f"{pos} with {thing} {where}"
            on_top["words"] = f"{thing} on top of {pos} {where}"
        else:
            words = f"{pos} {where}"
        p["words"] = words


def _cable_and_water(pl, L, v):
    for s in pl["sites"]:
        for sub, (x, y, z0, top) in s.get("sub", {}).items():
            L[f"{s['name']}/{sub.replace(' ', '_')}"] = dict(xy=(x, y), top_z=top, base_z=z0, words=f"the {sub} of the cable car", kind=sub,
                                                             size_r=12.0, box=(x, y, 0.0, 10.0, 10.0, top))
    lk = v.get("lake")
    if lk:
        L["lake"] = dict(xy=(lk["x"], float(valley_y(lk["x"], v))), top_z=lk["z"] + 0.5, base_z=lk["z"], words="the lake", kind="lake",
                         size_r=lk["L"], box=(lk["x"], float(valley_y(lk["x"], v)), 0.0, lk["L"], float(valley_w(lk["x"], v)) * lk["wf"], lk["z"] + 0.5))
    for k, tn in enumerate(v.get("tarns", [])):
        in_valley = abs(tn["y"] - float(valley_y(tn["x"], v))) < float(valley_w(tn["x"], v)) + 20
        words = "the small mountain lake on the valley floor" if in_valley else f"the small mountain lake beyond the {side_words(tn['y'] - float(valley_y(tn['x'], v)))} ridge"
        L[f"tarn_{k + 1}"] = dict(xy=(tn["x"], tn["y"]), top_z=tn["z"] + 0.5, base_z=tn["z"], words=words, kind="tarn", size_r=tn["r"],
                                  box=(tn["x"], tn["y"], 0.0, tn["r"], tn["r"], tn["z"] + 0.5))


def _unique_words(pl):
    seen = {}
    for n, lm in pl["landmarks"].items():
        seen.setdefault(lm["words"], []).append(n)
    for words, names in seen.items():
        if len(names) < 2:
            continue
        names.sort(key=lambda n: pl["landmarks"][n]["xy"][0])
        for i, n in enumerate(names):
            pl["landmarks"][n]["words"] = words + (" (the western one)" if i == 0 else " (the eastern one)" if i == len(names) - 1 else f" (number {i + 1} from the west)")


# ================================================================================================== objects
def _foot_r(prop):
    sup = prop.get("support") or [(0.3, 0.3)]
    r = max(math.hypot(a, b) for a, b in sup) + 0.12
    if prop.get("length", 0) > 2.0:
        r = max(r, math.hypot(prop["length"] / 2, prop.get("width", prop["length"] * 0.4) / 2) + 0.1)
    return r


def _solid_hit(sd, x, y, r):
    kind, sx, sy, a, hx, hy = sd[:6]
    if kind == "circle":
        return math.hypot(x - sx, y - sy) < r + a
    u, v = rot(-a, x - sx, y - sy)
    return math.hypot(max(abs(u) - hx, 0.0), max(abs(v) - hy, 0.0)) < r


def _prune_overlaps(pl):
    keep, occ, dropped = [], [], []
    for m in pl["movables"]:
        r = _foot_r(m["prop"])
        zg = surface_z(m["x"], m["y"], pl)
        top = zg + max(0.3, m["prop"].get("height", 1.0))
        bad = any(_solid_hit(sd, m["x"], m["y"], r) and sd[7] > zg + 0.04 and sd[6] < top for sd in pl["solids"])
        bad = bad or any(math.hypot(m["x"] - ox, m["y"] - oy) < r + orr for ox, oy, orr in occ)
        bad = bad or water_level(m["x"], m["y"], pl) is not None
        if bad:
            dropped.append((m["group"], m["name"], m["kind"]))
            continue
        keep.append(m)
        occ.append((m["x"], m["y"], r))
    pl["movables"] = keep
    pl["dropped_overlap"] = dropped


def movables(pl):
    out = []
    for m in pl["movables"]:
        gl = lambda x, y: surface_z(x, y, pl)  # noqa: E731
        M, resid, tilt = props.rest_pose(m["prop"], m["x"], m["y"], m["yaw"], gl)
        out.append(dict(group=m["group"], name=m["name"], prop=m["prop"], matrix=M, static_check=props.check_static(m["prop"], tilt),
                        support_residual_m=round(resid, 4), xy=(m["x"], m["y"]), yaw=m["yaw"], kind=m["kind"]))
    return out


def obstacles(pl):
    return list(pl["obstacles"])


SEG_CLASSES = ["background/sky", "terrain", "water", "vegetation", "landing pad", "road", "building", "church", "hut", "cable car",
               "mast", "summit cross", "wind turbine", "tent", "helicopter", "bridge", "boat", "person", "animal", "vehicle", "other"]


def object_registry(pl):
    """Every object of the scene in a fixed order -> segmentation id (1-based; 0 = sky).
    [(name, [prim path prefixes], class, description)] -- every entry has a description (review item C8)."""
    snow = "snow-capped " if pl["snowline"] < 400 else ""
    out = [("terrain", ["/World/Terrain", "/World/FarRing"], "terrain",
            f"the mountain terrain: {SEASONS[pl['season']]['grass'].replace('_', ' ')} meadows on the valley floor, forested lower slopes, scree, "
            f"rock faces and {snow}summits"),
           ("water", ["/World/River"], "water", "the river" + (" and the lake" if pl["valley"].get("lake") else "") + (" and mountain lakes" if pl["valley"].get("tarns") else "")),
           ("vegetation", ["/World/Mtn/Nature"], "vegetation", "conifer forest, broadleaf trees, shrubs, boulders and grass"),
           ("roads", ["/World/Mtn/Roads"], "road", "the gravel valley road and tracks")]
    for k, p in pl["pads"].items():
        w_ = p["words"][4:] if p["words"].startswith("the ") else p["words"]
        out.append((f"pad_{k}", [f"/World/Mtn/Pads/{k}"], "landing pad", f"painted {w_}" if w_.startswith(("landing pad", "helipad")) else f"painted landing pad: {w_}"))
    for st in pl["statics"]:
        out.append((st["name"], [st["path"]], st["cls"], st["words"]))
    for r in pl["refs"]:
        pass
    for m in pl["movables"]:
        out.append((f"{m['group']}/{m['name']}", [f"/World/Props/{m['group']}/{m['name']}"], m["cls"], m["words"]))
    return out
