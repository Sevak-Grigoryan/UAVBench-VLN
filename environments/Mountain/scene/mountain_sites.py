"""
Mountain v2 sites: where they go (plan_sites, on the zone-free terrain), the levelled ground they need (zones), and
everything they contain (populate_sites, on the final terrain): buildings, cable car, masts, people, animals, vehicles.
Then the landing pads and the unmarked take-off / landing spots (plan_pads_and_spots).

Every static object and every movable object gets a description with its colours and where it is (review item C8).
"""
import math

import numpy as np

import coastal_props as cp
import mountain_props as mp
import mountain_v2 as mv
import props

# kind -> (zone half-length along the valley, zone half-width, falloff, placement)
SITE_SPEC = {
    "village": (46, 26, 14, "floor"), "farm": (34, 24, 12, "floor"), "campsite": (28, 20, 10, "floor"),
    "trailhead": (20, 13, 8, "floor"), "rescue station": (26, 18, 10, "floor"), "stone bridge": (0, 0, 0, "river"),
    "chapel": (10, 8, 9, "low"), "quarry": (24, 15, 6, "low"), "mountain hut": (17, 14, 10, "shoulder"),
    "cable car": (11, 9, 8, "cable"), "paraglider launch": (16, 10, 10, "shoulder"), "summit cross": (7, 7, 6, "summit"),
    "radio mast": (9, 7, 6, "summit"), "observatory": (11, 9, 8, "summit"), "wind farm": (5, 5, 5, "crest"),
    "shepherd hut": (12, 10, 8, "outer"), "boathouse": (8, 6, 6, "lakeshore"),
}
POOLS = {
    "alpine valley": {"village": 1.0, "farm": 0.8, "campsite": 0.7, "trailhead": 0.8, "rescue station": 0.6, "chapel": 0.6,
                      "mountain hut": 0.9, "cable car": 0.6, "summit cross": 0.8, "radio mast": 0.5, "paraglider launch": 0.5,
                      "stone bridge": 0.6, "quarry": 0.35, "shepherd hut": 0.4},
    "glacial lake valley": {"boathouse": 1.0, "village": 0.9, "campsite": 0.8, "chapel": 0.6, "mountain hut": 0.8, "cable car": 0.5,
                            "summit cross": 0.7, "radio mast": 0.4, "trailhead": 0.6, "rescue station": 0.5, "farm": 0.5, "observatory": 0.3},
    "gorge": {"stone bridge": 1.0, "village": 0.6, "mountain hut": 0.7, "quarry": 0.5, "summit cross": 0.7, "radio mast": 0.5,
              "trailhead": 0.7, "campsite": 0.5, "farm": 0.5, "chapel": 0.5, "shepherd hut": 0.4},
    "dolomite towers": {"mountain hut": 1.0, "summit cross": 0.9, "chapel": 0.6, "village": 0.7, "farm": 0.6, "cable car": 0.6,
                        "trailhead": 0.7, "paraglider launch": 0.6, "observatory": 0.3, "campsite": 0.5, "rescue station": 0.4},
    "rolling highlands": {"wind farm": 1.0, "farm": 0.9, "shepherd hut": 0.9, "observatory": 0.5, "radio mast": 0.6, "village": 0.7,
                          "chapel": 0.6, "campsite": 0.5, "trailhead": 0.5, "stone bridge": 0.4, "paraglider launch": 0.4},
    "high cirque": {"mountain hut": 1.0, "summit cross": 0.9, "observatory": 0.5, "radio mast": 0.5, "shepherd hut": 0.7,
                    "trailhead": 0.6, "cable car": 0.5, "rescue station": 0.4, "paraglider launch": 0.3, "campsite": 0.4},
}
NEEDS = {"follow_cable_car": ["cable car"], "cross_lake": ["boathouse"], "land_at": ["mountain hut", "rescue station"]}


def _compass(dx, dy):
    names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    return names[int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8]


def U(rng, a, b):
    return float(rng.uniform(a, b))


# ================================================================================================== helpers
def _rot_xyz(v, r):
    """USD RotateXYZ (degrees): rotate about X, then Y, then Z."""
    x, y, z = v
    a, b, c = (math.radians(t) for t in r)
    y, z = y * math.cos(a) - z * math.sin(a), y * math.sin(a) + z * math.cos(a)
    x, z = x * math.cos(b) + z * math.sin(b), -x * math.sin(b) + z * math.cos(b)
    x, y = x * math.cos(c) - y * math.sin(c), x * math.sin(c) + y * math.cos(c)
    return x, y, z


def _part_bounds(p):
    """Local axis-aligned bounds (x0, x1, y0, y1, z0, z1) of one part, rotations included exactly."""
    sh = p["shape"]
    rot_ = p.get("rot") or (0, 0, 0)
    if sh in ("box", "tbox"):
        (cx, cy, cz), (sx, sy, sz) = p["c"], p["s"]
        pts = [_rot_xyz((a * sx / 2, b * sy / 2, c * sz / 2), rot_) for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
        return (cx + min(q[0] for q in pts), cx + max(q[0] for q in pts), cy + min(q[1] for q in pts), cy + max(q[1] for q in pts),
                cz + min(q[2] for q in pts), cz + max(q[2] for q in pts))
    if sh in ("cyl", "cap"):
        (cx, cy, cz), r, h = p["c"], p["r"], p["h"]
        ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[p.get("axis", "Z")]
        d = _rot_xyz(ax, rot_)
        half = h / 2 + (r if sh == "cap" else 0.0)
        ex = [abs(d[k]) * half + r for k in range(3)]
        return cx - ex[0], cx + ex[0], cy - ex[1], cy + ex[1], cz - ex[2], cz + ex[2]
    if sh == "sph":
        (cx, cy, cz), r, s = p["c"], p["r"], p["scale"]
        return cx - r * s[0], cx + r * s[0], cy - r * s[1], cy + r * s[1], cz - r * s[2], cz + r * s[2]
    if sh == "cone":
        (cx, cy, cz), r, h = p["c"], p["r"], p["h"]
        return cx - r, cx + r, cy - r, cy + r, cz - h / 2, cz + h / 2
    if sh == "prism":
        return p["x0"], p["x1"], -p["hw"], p["hw"], p["z0"], p["z0"] + p["rise"]
    if sh == "mesh":
        a = np.asarray(p["pts"], float)
        return a[:, 0].min(), a[:, 0].max(), a[:, 1].min(), a[:, 1].max(), a[:, 2].min(), a[:, 2].max()
    return 0, 0, 0, 0, 0, 0


def _solids(pl, parts, x, y, z, yaw):
    for p in parts:
        if not p.get("col") or p["shape"] not in ("box", "tbox", "cyl", "cap", "sph", "cone", "prism"):
            continue
        b = _part_bounds(p)
        cx, cy = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
        dx, dy = mv.rot(yaw, cx, cy)
        pl["solids"].append(("rect", x + dx, y + dy, yaw, (b[1] - b[0]) / 2, (b[3] - b[2]) / 2, z + b[4], z + b[5]))


def add_static(pl, site, part, parts, x, y, z, yaw, cls, words, obstacle=True, owner=None):
    name = f"{site['name']}/{part}"
    path = f"/World/Mtn/Sites/{site['name']}/{part}"
    pl["statics"].append(dict(name=name, path=path, parts=parts, M=cp.M(x, y, z, yaw), cls=cls, words=words, xy=(x, y)))
    _solids(pl, parts, x, y, z, yaw)
    if obstacle and parts:
        bs = [_part_bounds(p) for p in parts if not p.get("hidden")]
        x0, x1 = min(b[0] for b in bs), max(b[1] for b in bs)
        y0, y1 = min(b[2] for b in bs), max(b[3] for b in bs)
        top = max(b[5] for b in bs)
        pts = [mv.rot(yaw, a, b) for a in (x0, x1) for b in (y0, y1)]
        pl["obstacles"].append((x + min(p[0] for p in pts), x + max(p[0] for p in pts), y + min(p[1] for p in pts),
                                y + max(p[1] for p in pts), z + top, owner or site["name"]))
        return z + top
    return None


def add_movable(pl, site, name, prop, x, y, yaw, cls):
    if not _flat_ok(pl, x, y, max(0.5, min(3.0, prop.get("length", 1.0) / 2 + 0.2))):
        return False
    words = f"{prop.get('words', prop['kind'])} {site['at']}"
    pl["movables"].append(dict(group=site["name"], name=name, prop=prop, x=x, y=y, yaw=yaw, kind=prop["kind"], cls=cls, words=words))
    return True


def _flat_ok(pl, x, y, r):
    zs = [mv.surface_z(x + dx, y + dy, pl) for dx in (-r, 0.0, r) for dy in (-r, 0.0, r)]
    return max(zs) - min(zs) < 0.10 * r / 0.7 + 0.05


def _foundation(parts, L, W, depth=2.5, mat="concrete"):
    parts.insert(0, props.box((0, 0, -depth / 2 + 0.05), (L, W, depth), mat))
    return parts


def _site(pl, kind, x, y, yaw, z, words, at, size_r):
    n = sum(1 for s in pl["sites"] if s["kind"] == kind)
    s = dict(kind=kind, name=f"{kind.replace(' ', '_')}_{n + 1}", xy=(x, y), yaw=yaw, base_z=z, words=words, at=at, size_r=size_r, top=z + 8.0)
    pl["sites"].append(s)
    return s


def _H0(pl, x, y):
    return mv.terrain_height(x, y, pl)


def _foot_range(pl, x, y, yaw, hx, hy):
    zs = []
    for a in np.linspace(-hx, hx, 5):
        for b in np.linspace(-hy, hy, 5):
            dx, dy = mv.rot(yaw, a, b)
            zs.append(_H0(pl, x + dx, y + dy))
    return float(np.median(zs)), max(zs) - min(zs)


def _clear_of_sites(pl, x, y, r):
    if any(math.hypot(x - s["xy"][0], y - s["xy"][1]) < s["size_r"] + r + 25 for s in pl["sites"]):
        return False
    for c in pl.get("_cable_lines", []):
        if _seg_d(x, y, c) < r + 30:
            return False
    return abs(x) < mv.EXTENT - r - 50 and abs(y) < mv.EXTENT - r - 50


def _seg_d(x, y, seg):
    (ax, ay), (bx, by) = seg
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 < 1e-9 else max(0.0, min(1.0, ((x - ax) * dx + (y - ay) * dy) / L2))
    return math.hypot(x - ax - t * dx, y - ay - t * dy)


def _in_water_zone(pl, x, y, margin=30.0):
    v = pl["valley"]
    lk = v.get("lake")
    if lk:
        e = ((x - lk["x"]) / (lk["L"] + margin)) ** 2 + ((y - float(mv.valley_y(x, v))) / (float(mv.valley_w(x, v)) * lk["wf"] + margin)) ** 2
        if e < 1.0:
            return True
    return any(math.hypot(x - t["x"], y - t["y"]) < t["r"] * 2.2 + margin for t in v.get("tarns", []))


def valley_yaw(x, v):
    return math.atan2(float(mv.valley_y(x + 3, v) - mv.valley_y(x - 3, v)), 6.0)


def d_of(pl, x, y):
    v = pl["valley"]
    return abs(y - float(mv.valley_y(x, v))) - float(mv.valley_w(x, v))


# ================================================================================================== placement
def plan_sites(pl, rng):
    land, v = pl["landscape"], pl["valley"]
    pool = dict(POOLS[land])
    need = list(NEEDS.get(pl["task"], []))
    if pl["task"] == "land_at":
        need = [str(rng.choice(["mountain hut", "rescue station", "summit cross", "campsite"]))]
    if pl["task"] == "cross_lake" and not v.get("lake"):
        need = []
    if pl["task"] == "follow_cable_car" and "cable car" not in pool:
        pool["cable car"] = 1.0
    n = int(rng.integers(6, 10))
    kinds = list(need)
    rest = [k for k in pool if k not in kinds]
    w = np.array([pool[k] for k in rest], float)
    order = list(rng.choice(rest, size=len(rest), replace=False, p=w / w.sum()))
    kinds += [str(k) for k in order]
    pl["road_side"] = int(rng.choice([1, -1]))
    pl["_cable_lines"] = []
    placed = 0
    for k in kinds:
        if placed >= n and k not in need:
            break
        ok = _place(pl, k, rng)
        if ok:
            placed += 1
        elif k in need:
            raise ValueError(f"could not place required site {k}")
    _plan_roads(pl, rng)


def _place(pl, kind, rng):
    hx, hy, f, where = SITE_SPEC[kind]
    v = pl["valley"]
    land = pl["landscape"]
    if where == "floor" and land == "gorge":
        where = "rim"
    for _ in range(400):
        if where in ("floor",):
            x = U(rng, -mv.EXTENT + hx + 70, mv.EXTENT - hx - 70)
            W = float(mv.valley_w(x, v))
            side = pl["road_side"] if rng.random() < 0.7 else -pl["road_side"]
            off = U(rng, 0.5 * W + hy + 6, W + hy * 0.6) if side == pl["road_side"] else U(rng, 0.35 * W, W + hy * 0.4)
            y = float(mv.valley_y(x, v)) + side * off
            if abs(y - float(mv.river_y(x, v))) < v["rw"] + hy + f + 4 or _in_water_zone(pl, x, y, 25.0):
                continue
            yaw = valley_yaw(x, v)
        elif where == "rim":
            x = U(rng, -mv.EXTENT + hx + 70, mv.EXTENT - hx - 70)
            side = int(rng.choice([1, -1]))
            y = float(mv.valley_y(x, v)) + side * (float(mv.valley_w(x, v)) + v["Dw"] * U(rng, 1.05, 1.5))
            yaw = valley_yaw(x, v)
        elif where in ("low", "shoulder", "outer"):
            x = U(rng, -mv.EXTENT + 90, mv.EXTENT - 90)
            side = int(rng.choice([1, -1]))
            lo, hi = {"low": (0.06, 0.3), "shoulder": (0.35, 0.72), "outer": (1.15, 1.0 + 0.9 * v["out"])}[where]
            y = float(mv.valley_y(x, v)) + side * (float(mv.valley_w(x, v)) + v["Dw"] * U(rng, lo, hi))
            yaw = valley_yaw(x, v) + (0 if side > 0 else math.pi)  # front faces the valley
            if where == "low" and kind == "quarry":
                yaw = valley_yaw(x, v)
        elif where == "summit":
            free = [p for p in pl["summits"] if not p.get("used") and abs(p["x"]) < mv.EXTENT - 80 and abs(p["y"]) < mv.EXTENT - 80]
            if not free:
                return False
            p = free[int(rng.integers(len(free)))] if kind != "summit cross" else max(free, key=lambda q: q["z"] + float(rng.uniform(0, 60)))
            x, y, yaw = p["x"], p["y"], U(rng, 0, 2 * math.pi)
        elif where == "crest":
            return _place_wind(pl, rng)
        elif where == "cable":
            return _place_cable(pl, rng)
        elif where == "river":
            return _place_bridge(pl, rng)
        elif where == "lakeshore":
            return _place_boathouse(pl, rng)
        else:
            return False
        if not _clear_of_sites(pl, x, y, math.hypot(hx, hy)):
            continue
        if _in_water_zone(pl, x, y, 20.0):
            continue
        z, rng_z = _foot_range(pl, x, y, yaw, hx, hy)
        sl = mv.slope_at(x, y, pl, 6.0)
        lim = {"floor": (5.0, 0.25), "rim": (10.0, 0.35), "low": (12.0, 0.5), "shoulder": (16.0, 0.6), "outer": (12.0, 0.5),
               "summit": (45.0, 9.0)}[where]
        if kind in ("paraglider launch",):
            lim = (14.0, 0.6)
            if sl < 0.18:
                continue
        if rng_z > lim[0] or sl > lim[1]:
            continue
        if where == "summit":
            z = mv.terrain_height(x, y, pl) - 1.2
            p["used"] = True
        if kind == "quarry":
            dx, dy = mv.rot(yaw, 0.0, -side * hy)
            z = _H0(pl, x + dx, y + dy) + 0.5  # level of the low (valley-side) edge: the bench cuts into the slope
            pl["zones"].append((x, y, yaw, hx, hy, z, f, "cut"))
        else:
            # the levelled platform blends into the slope over a distance that grows with the height it cuts / fills (a short
            # fall-off made vertical dirt walls around huts on steep ground, Mountain preview 2026-10-20)
            ff = f if where == "summit" else min(45.0, max(f, 1.4 * rng_z + 6.0))  # a summit keeps its shape (wide fall-off = a dome)
            pl["zones"].append((x, y, yaw, hx, hy, z, ff, "flat"))
        s = _site(pl, kind, x, y, yaw, z, "", "", max(hx, hy) + 6)
        s["side"] = 1 if y > float(mv.valley_y(x, v)) else -1
        if where == "summit":
            s["summit"] = p["name"]
        return True
    return False


def _place_wind(pl, rng):
    v = pl["valley"]
    side = int(rng.choice([1, -1]))
    xs = []
    x = U(rng, -480, -200)
    while x < 480 and len(xs) < int(rng.integers(3, 6)):
        best = None
        for d in np.linspace(0.75, 1.25, 26):
            y = float(mv.valley_y(x, v)) + side * (float(mv.valley_w(x, v)) + d * v["Dw"])
            if abs(y) > mv.EXTENT - 60:
                continue
            z = _H0(pl, x, y)
            if best is None or z > best[2]:
                best = (x, y, z)
        if best and mv.slope_at(best[0], best[1], pl, 6.0) < 0.45 and _clear_of_sites(pl, best[0], best[1], 25):
            xs.append(best)
        x += U(rng, 95, 140)
    if len(xs) < 3:
        return False
    s = _site(pl, "wind farm", float(np.mean([p[0] for p in xs])), float(np.mean([p[1] for p in xs])), 0.0, float(np.mean([p[2] for p in xs])),
              "", "", 0.5 * math.hypot(xs[-1][0] - xs[0][0], xs[-1][1] - xs[0][1]) + 25)
    s["turbines"] = []
    for (x, y, z) in xs:
        pl["zones"].append((x, y, 0.0, 5.0, 5.0, z - 0.3, 6.0, "flat"))
        s["turbines"].append((x, y, z - 0.3))
    s["side"] = side
    return True


def _place_cable(pl, rng):
    v = pl["valley"]
    for _ in range(300):
        x = U(rng, -mv.EXTENT + 160, mv.EXTENT - 160)
        side = int(rng.choice([1, -1]))
        W = float(mv.valley_w(x, v))
        y = float(mv.valley_y(x, v)) + side * (W - U(rng, 8, 25))
        if abs(y - float(mv.river_y(x, v))) < v["rw"] + 25 or _in_water_zone(pl, x, y, 20):
            continue
        ang = valley_yaw(x, v) + side * math.pi / 2 + U(rng, -0.45, 0.45)
        span = U(rng, 330, 560)
        tx, ty = x + span * math.cos(ang), y + span * math.sin(ang)
        if abs(tx) > mv.EXTENT - 70 or abs(ty) > mv.EXTENT - 70:
            continue
        if not (_clear_of_sites(pl, x, y, 16) and _clear_of_sites(pl, tx, ty, 16)):
            continue
        if any(_seg_d(s["xy"][0], s["xy"][1], ((x, y), (tx, ty))) < s["size_r"] + 30 for s in pl["sites"]):
            continue
        z0, r0 = _foot_range(pl, x, y, ang, 11, 9)
        z1, r1 = _foot_range(pl, tx, ty, ang, 11, 9)
        if r0 > 6 or r1 > 18 or z1 - z0 < 110 or mv.slope_at(tx, ty, pl, 6.0) > 0.75:
            continue
        # the line must climb steadily: no spur or tower in between higher than the top station
        prof = [_H0(pl, x + (tx - x) * t, y + (ty - y) * t) for t in np.linspace(0.0, 1.0, 40)]
        if max(prof) > z1 + 8.0 or any(prof[k + 8] < prof[k] - 25.0 for k in range(len(prof) - 8)):
            continue
        pl["zones"].append((x, y, ang, 11, 9, z0, min(40.0, max(8.0, 1.4 * r0 + 6.0)), "flat"))
        pl["zones"].append((tx, ty, ang, 11, 9, z1, min(40.0, max(10.0, 1.4 * r1 + 6.0)), "flat"))
        s = _site(pl, "cable car", (x + tx) / 2, (y + ty) / 2, ang, z0, "", "", 0.5 * span)
        s.update(valley=(x, y, z0), top_st=(tx, ty, z1), span=span, side=side)
        pl["_cable_lines"].append(((x, y), (tx, ty)))
        return True
    return False


def _place_bridge(pl, rng):
    v = pl["valley"]
    for _ in range(200):
        x = U(rng, -mv.EXTENT + 120, mv.EXTENT - 120)
        y = float(mv.river_y(x, v))
        if _in_water_zone(pl, x, y, 60) or not _clear_of_sites(pl, x, y, 20):
            continue
        ry = math.atan2(float(mv.river_y(x + 3, v) - mv.river_y(x - 3, v)), 6.0)
        yaw = ry + math.pi / 2  # deck across the river
        lvl = float(mv.river_level(x, v))
        span = 2 * (v["rw"] + 7.0)
        deck = lvl + float(rng.uniform(3.8, 5.0))
        for sgn in (-1, 1):
            cx, cy = x + sgn * math.cos(yaw) * (span / 2 + 9), y + sgn * math.sin(yaw) * (span / 2 + 9)
            pl["zones"].append((cx, cy, yaw, 9.0, 4.0, deck, 14.0, "flat"))
        s = _site(pl, "stone bridge", x, y, yaw, deck, "", "", span / 2 + 10)
        s.update(span=span, deck=deck, level=lvl)
        pl["decks"].append((x, y, yaw, span / 2 + 1.0, 2.6, deck + 0.0, "bridge"))
        return True
    return False


def _place_boathouse(pl, rng):
    v = pl["valley"]
    lk = v.get("lake")
    if not lk:
        return False
    for _ in range(200):
        a = U(rng, 0, 2 * math.pi)
        side = 1 if math.sin(a) > 0 else -1
        x = lk["x"] + lk["L"] * 0.92 * math.cos(a)
        if abs(x - lk["x"]) > lk["L"] * 0.8:
            continue
        W = float(mv.valley_w(x, v)) * lk["wf"]
        yv = float(mv.valley_y(x, v))
        # walk outward from the water to the first dry ground
        for k in range(60):
            y = yv + side * (W * 0.7 + 2.0 * k)
            if _H0(pl, x, y) > lk["z"] + 1.0:
                break
        bx, by = x, y + side * 9.0
        if not _clear_of_sites(pl, bx, by, 12):
            continue
        z, rz = _foot_range(pl, bx, by, 0.0, 8, 6)
        if rz > 8:
            continue
        zb = max(lk["z"] + 1.2, min(z, lk["z"] + 3.0))
        yaw = -side * math.pi / 2  # jetty points into the lake
        pl["zones"].append((bx, by, 0.0, 8, 6, zb, 6.0, "flat"))
        s = _site(pl, "boathouse", bx, by, yaw, zb, "", "", 22.0)
        s.update(shore=(x, y), side=side)
        return True
    return False


def _plan_roads(pl, rng):
    """Gravel valley road on one side of the river + short spurs to the floor sites."""
    v = pl["valley"]
    if pl["landscape"] == "gorge":
        return
    side = pl["road_side"]
    lk = v.get("lake")
    pts = []
    for x in np.arange(-mv.EXTENT - 60, mv.EXTENT + 60.1, 6.0):
        W = float(mv.valley_w(x, v))
        off = 0.5 * W
        if lk:
            w = float(mv.sstep(lk["L"] + 60, lk["L"] - 20, abs(x - lk["x"])))
            off = off * (1 - w) + (W * lk["wf"] + 10.0) * w
        yv = float(mv.valley_y(x, v))
        for _ in range(40):  # push the road off any water (its edge + 4 m stays dry; roads ran under the lake, audit 2026-10-21)
            y = yv + side * off
            if not any(mv.water_level(float(x), y + side * e, pl) is not None for e in (-6.3, -3.0, 0.0, 3.0, 6.3)):
                break
            off += 2.0
        pts.append((float(x), yv + side * off))
    pl["roads"].append(dict(name="valley road", pts=pts, w=4.6))
    for s in pl["sites"]:
        if s["kind"] in ("village", "farm", "campsite", "trailhead", "rescue station") and s.get("side") == side:
            x, y = s["xy"]
            k = min(range(len(pts)), key=lambda i: math.hypot(pts[i][0] - x, pts[i][1] - y))
            pl["roads"].append(dict(name=f"track to {s['name']}", pts=[pts[k], (x, y)], w=3.6))


def dry_roads(pl):
    """Final pass on the finished terrain: a road point with water under the road or within 3 m of its edge moves sideways
    (perpendicular to the road, 1 m steps, up to 40 m, whichever side is dry first). The planner's own push ran on the
    terrain before the site zones and checked only across the valley; a few points still clipped a tarn (audit 2026-10-21)."""
    def wet(x, y, hw):
        return any(mv.water_level(x + nx * e, y + ny * e, pl) is not None for e in (-hw - 3.0, -hw / 2, 0.0, hw / 2, hw + 3.0))
    moved = 0
    for r in pl["roads"]:
        pts = [tuple(p) for p in r["pts"]]
        hw = r["w"] / 2
        for i, (x, y) in enumerate(pts):
            a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
            L = math.hypot(b[0] - a[0], b[1] - a[1]) or 1.0
            nx, ny = -(b[1] - a[1]) / L, (b[0] - a[0]) / L
            if not wet(x, y, hw):
                continue
            for d in range(1, 41):
                cand = [(x + nx * s_ * d, y + ny * s_ * d) for s_ in (1, -1)]
                ok = [c for c in cand if not wet(c[0], c[1], hw)]
                if ok:
                    pts[i] = ok[0]
                    moved += 1
                    break
        r["pts"] = pts
    pl["roads_moved_off_water"] = moved
    return moved


# ================================================================================================== populate
def populate_sites(pl, rng):
    for s in list(pl["sites"]):
        r = np.random.default_rng(int(rng.integers(1 << 31)))
        globals()["_pop_" + s["kind"].replace(" ", "_")](pl, s, r)
        s["box"] = (s["xy"][0], s["xy"][1], 0.0, s["size_r"], s["size_r"], s["top"])


def _P(s, u, w):
    dx, dy = mv.rot(s["yaw"], u, w)
    return s["xy"][0] + dx, s["xy"][1] + dy


def _gl(pl, x, y, yaw):
    return lambda a, b: mv.terrain_height(x + mv.rot(yaw, a, b)[0], y + mv.rot(yaw, a, b)[1], pl)


def _people(pl, s, rng, n, region, role="hiker", prefix="p", child_frac=0.0, taken=None):
    u0, u1, w0, w1 = region
    taken = taken if taken is not None else []
    k = 0
    for i in range(n * 3):
        if k >= n:
            break
        u, w = U(rng, u0, u1), U(rng, w0, w1)
        x, y = _P(s, u, w)
        if any(math.hypot(x - a, y - b) < 1.4 for a, b in taken):
            continue
        child = rng.random() < child_frac
        pr = mp.hiker(rng, role=role if not child else "villager", child=child)
        if add_movable(pl, s, f"{prefix}{k}", pr, x, y, U(rng, 0, 2 * math.pi), "person"):
            taken.append((x, y))
            k += 1
    return taken


def _cars(pl, s, rng, slots, prefix="car"):
    for k, (u, w, yaw_rel) in enumerate(slots):
        x, y = _P(s, u, w)
        add_movable(pl, s, f"{prefix}{k}", mp.car(rng), x, y, s["yaw"] + yaw_rel, "vehicle")


def _pop_village(pl, s, rng):
    n_ch = int(rng.integers(7, 13))
    s["words"] = str(rng.choice(["the alpine village with the white church", "the mountain village with the church tower",
                                 "the village of wooden chalets around a white church"]))
    s["at"] = "in the village"
    cx, cy = _P(s, U(rng, -10, 10), U(rng, -4, 4))
    ch = mp.church(rng, _gl(pl, cx, cy, s["yaw"]))
    top = add_static(pl, s, "church", ch["parts"], cx, cy, 0.0, s["yaw"], "church", f"{ch['words']} in the centre of the village")
    s["top"] = top
    taken = [(cx, cy, 16.0)]
    k = 0
    for _ in range(300):
        if k >= n_ch:
            break
        u, w = U(rng, -42, 42), U(rng, -22, 22)
        x, y = _P(s, u, w)
        if any(math.hypot(x - a, y - b) < r + 9 for a, b, r in taken):
            continue
        yaw = s["yaw"] + float(rng.choice([0, math.pi / 2, math.pi, -math.pi / 2])) + U(rng, -0.15, 0.15)
        pr = mp.chalet(rng, _gl(pl, x, y, yaw))
        add_static(pl, s, f"chalet_{k}", pr["parts"], x, y, 0.0, yaw, "building", f"{pr['words']} in the village")
        taken.append((x, y, 7.0))
        k += 1
        if rng.random() < 0.6:
            dx, dy = mv.rot(yaw, 0, -9.5)
            add_movable(pl, s, f"car{k}", mp.car(rng), x + dx, y + dy, yaw + U(rng, -0.2, 0.2), "vehicle")
    _people(pl, s, rng, int(rng.integers(6, 13)), (-40, 40, -20, 20), role="villager", child_frac=0.15)
    for d in range(int(rng.integers(0, 3))):
        x, y = _P(s, U(rng, -30, 30), U(rng, -18, 18))
        add_movable(pl, s, f"dog{d}", mp.animal(rng, "dog"), x, y, U(rng, 0, 6.28), "animal")
    s["size_r"] = 55.0


def _pop_farm(pl, s, rng):
    s["words"] = str(rng.choice(["the farm with the red barn", "the mountain farm with the red barn and a fenced paddock"]))
    s["at"] = "on the farm"
    bx, by = _P(s, -16, 10)
    barn = props.barn(_gl(pl, bx, by, s["yaw"]), rng)
    s["top"] = add_static(pl, s, "barn", barn["parts"], bx, by, 0.0, s["yaw"], "building", "large red barn with white trim on the farm")
    hx, hy = _P(s, 8, 14)
    house = mp.chalet(rng, _gl(pl, hx, hy, s["yaw"]))
    add_static(pl, s, "farmhouse", house["parts"], hx, hy, 0.0, s["yaw"], "building", f"farmhouse: {house['words']}")
    corners = [_P(s, u, w) for u, w in ((-30, -20), (28, -20), (28, -3), (-30, -3))]
    fen = props.fence(lambda x, y: mv.terrain_height(x, y, pl), corners, gate_side=2)
    add_static(pl, s, "fence", fen["parts"], 0.0, 0.0, 0.0, 0.0, "other", "wooden paddock fence on the farm", obstacle=False)
    kind = str(rng.choice(["cow", "cow", "sheep", "horse"]))
    n = {"cow": int(rng.integers(4, 9)), "sheep": int(rng.integers(8, 15)), "horse": int(rng.integers(2, 5))}[kind]
    for k in range(n):
        x, y = _P(s, U(rng, -26, 24), U(rng, -17, -6))
        add_movable(pl, s, f"{kind}{k}", mp.animal(rng, kind), x, y, U(rng, 0, 6.28), "animal")
    s["animals"] = kind
    for k in range(int(rng.integers(2, 6))):
        x, y = _P(s, U(rng, 5, 30), U(rng, 2, 20))
        pr = props.round_bale(rng)
        pr["words"] = "round hay bale"
        add_movable(pl, s, f"bale{k}", pr, x, y, U(rng, 0, 6.28), "other")
    x, y = _P(s, 22, 4)
    pr = props.tractor(str(rng.choice(["green", "red", "blue"])), rng)
    pr["words"] = "farm tractor"
    add_movable(pl, s, "tractor", pr, x, y, s["yaw"] + U(rng, -0.5, 0.5), "vehicle")
    _people(pl, s, rng, int(rng.integers(1, 3)), (-5, 20, 0, 8), role="farmer")
    s["size_r"] = 42.0


def _pop_campsite(pl, s, rng):
    s["words"] = str(rng.choice(["the campsite with colourful tents", "the campground with tents and parked cars"]))
    s["at"] = "at the campsite"
    k = 0
    for i in range(3):
        for j in range(4):
            if rng.random() < 0.25:
                continue
            u, w = -18 + 11 * j + U(rng, -1.5, 1.5), -12 + 9 * i + U(rng, -1.2, 1.2)
            x, y = _P(s, u, w)
            t = mp.tent(rng)
            add_static(pl, s, f"tent_{k}", t["parts"], x, y, mv.terrain_height(x, y, pl) - 0.02, s["yaw"] + U(rng, -0.6, 0.6), "tent",
                       f"{t['words']} at the campsite")
            k += 1
    s["top"] = s["base_z"] + 3.5
    x, y = _P(s, 22, 12)
    b = mp.simple_building(rng, 7.0, 4.0, 3.0, "wood", mp.WOOD, "dark", mp.DARK, "small wooden toilet and shower block at the campsite")
    add_static(pl, s, "washhouse", _foundation(b["parts"], 7, 4), x, y, s["base_z"], s["yaw"], "building", b["words"])
    x, y = _P(s, 0, 15)
    cf = mp.campfire()
    add_static(pl, s, "campfire", cf["parts"], x, y, mv.terrain_height(x, y, pl), 0.0, "other", "stone campfire ring at the campsite", obstacle=False)
    x, y = _P(s, -22, 14)
    add_static(pl, s, "table", mp.picnic_table_parts(0, 0, 0.0), x, y, mv.terrain_height(x, y, pl), s["yaw"], "other", "wooden picnic table at the campsite")
    _cars(pl, s, rng, [(-24 + 6 * k, 18.5, math.pi / 2) for k in range(int(rng.integers(2, 6)))])
    _people(pl, s, rng, int(rng.integers(5, 11)), (-24, 24, -16, 16), role="camper", child_frac=0.2)
    if rng.random() < 0.5:
        x, y = _P(s, U(rng, -15, 15), U(rng, -10, 10))
        add_movable(pl, s, "dog", mp.animal(rng, "dog"), x, y, U(rng, 0, 6.28), "animal")
    s["size_r"] = 34.0


def _pop_trailhead(pl, s, rng):
    s["words"] = str(rng.choice(["the trailhead car park", "the hikers' car park at the trailhead"]))
    s["at"] = "at the trailhead car park"
    z = s["base_z"]
    lot = [dict(shape="tbox", c=(0, 0, -0.15), s=(36.0, 22.0, 0.4), tex="gravel_road", tile=6.0, rot=(0, 0, 0), mat=None, col=True)]
    add_static(pl, s, "lot", lot, *s["xy"], z, s["yaw"], "road", "gravel car park at the trailhead", obstacle=False)
    pl["decks"].append((s["xy"][0], s["xy"][1], s["yaw"], 18.0, 11.0, z + 0.05, "car park"))
    slots = [(-15 + 3.2 * k, -7.5, math.pi / 2) for k in range(10)] + [(-15 + 3.2 * k, 7.5, -math.pi / 2) for k in range(10)]
    pick = [slots[i] for i in sorted(rng.choice(len(slots), int(rng.integers(5, 14)), replace=False))]
    _cars(pl, s, rng, pick)
    x, y = _P(s, 15, -11.8)
    ib = mp.info_board()
    add_static(pl, s, "board", ib["parts"], x, y, z + 0.05, s["yaw"], "other", "wooden trail information board at the trailhead")
    x, y = _P(s, 16.5, 11.5)
    sp = mp.signpost(rng)
    add_static(pl, s, "signpost", sp["parts"], x, y, z + 0.05, s["yaw"], "other", "yellow trail signpost at the trailhead")
    _people(pl, s, rng, int(rng.integers(3, 9)), (-14, 15, -3, 3), role="hiker")
    s["top"] = z + 3.0
    s["size_r"] = 24.0
    s["lot"] = True


def _pop_rescue_station(pl, s, rng):
    s["words"] = "the mountain rescue station with the helicopter"
    s["at"] = "at the mountain rescue station"
    z = s["base_z"]
    x, y = _P(s, -12, 8)
    b = mp.simple_building(rng, 16.0, 10.0, 6.5, "red and white", (0.75, 0.10, 0.07), "dark grey", (0.12, 0.12, 0.13),
                           "{wall} mountain-rescue station building with a {roof} roof")
    add_static(pl, s, "building", _foundation(b["parts"], 16, 10), x, y, z, s["yaw"], "building", b["words"])
    x2, y2 = _P(s, 9, 9)
    g = mp.simple_building(rng, 9.0, 8.0, 4.5, "white", (0.85, 0.85, 0.83), "dark grey", (0.12, 0.12, 0.13), "white vehicle garage of the rescue station")
    add_static(pl, s, "garage", _foundation(g["parts"], 9, 8), x2, y2, z, s["yaw"], "building", g["words"])
    hx_, hy_ = _P(s, 12, -9)  # helicopter apron (a separate slab: the painted pad stays free)
    heli = mp.helicopter(rng)
    apron = [props.box((0, 0, -0.1), (14.0, 9.0, 0.3), (0.45, 0.45, 0.44))] + [dict(q, c=(q["c"][0], q["c"][1], q["c"][2] + 0.05)) if "c" in q else q for q in heli["parts"]]
    s["top"] = add_static(pl, s, "helicopter", apron, hx_, hy_, z, s["yaw"] + U(rng, -0.4, 0.4), "helicopter", heli["words"])
    s["heli_pad_uv"] = (-14.0, -9.0)
    for k in range(int(rng.integers(1, 3))):
        x, y = _P(s, 2 + 5 * k, 17.5)
        add_movable(pl, s, f"vehicle{k}", mp.rescue_vehicle(rng), x, y, s["yaw"] - math.pi / 2, "vehicle")
    _people(pl, s, rng, int(rng.integers(2, 5)), (-2, 6, -2, 2), role="rescuer")
    s["size_r"] = 32.0


def _pop_stone_bridge(pl, s, rng):
    s["words"] = str(rng.choice(["the old stone bridge over the river", "the stone arch bridge across the river"]))
    s["at"] = "on the stone bridge"
    parts = mp.stone_bridge(s["span"] + 2.0, 5.2, 0.0, s["level"] - s["deck"])
    s["top"] = add_static(pl, s, "bridge", parts, s["xy"][0], s["xy"][1], s["deck"], s["yaw"], "bridge", "old grey stone bridge with parapets over the river")
    pl["roads"].append(dict(name="bridge track", pts=[_P(s, -s["span"] / 2 - 30, 0), _P(s, -s["span"] / 2, 0)], w=3.6))
    pl["roads"].append(dict(name="bridge track 2", pts=[_P(s, s["span"] / 2, 0), _P(s, s["span"] / 2 + 30, 0)], w=3.6))
    _people(pl, s, rng, int(rng.integers(0, 3)), (-s["span"] / 2 + 2, s["span"] / 2 - 2, -1.2, 1.2), role="hiker")


def _pop_chapel(pl, s, rng):
    ch = mp.church(rng, _gl(pl, *s["xy"], s["yaw"]), small=True)
    s["words"] = "the small white chapel on the hillside"
    s["at"] = "by the chapel"
    s["top"] = add_static(pl, s, "chapel", ch["parts"], *s["xy"], 0.0, s["yaw"], "church", f"{ch['words']} on the hillside")
    _people(pl, s, rng, int(rng.integers(0, 4)), (-8, 8, -7, -4), role="hiker")
    s["size_r"] = 14.0


def _pop_quarry(pl, s, rng):
    s["words"] = "the stone quarry on the hillside"
    s["at"] = "in the quarry"
    z = s["base_z"]
    for k in range(int(rng.integers(2, 4))):
        x, y = _P(s, -16 + 11 * k, U(rng, -3, 6))
        gp = mp.gravel_pile(rng, (0.55, 0.53, 0.50))
        add_static(pl, s, f"heap_{k}", gp["parts"], x, y, z - 0.3, 0.0, "other", "heap of grey crushed stone in the quarry")
    x, y = _P(s, 10, 2)
    add_movable(pl, s, "excavator", mp.machine(rng, "excavator"), x, y, s["yaw"] + U(rng, -0.6, 0.6), "vehicle")
    x, y = _P(s, 16, -9)
    add_movable(pl, s, "dumptruck", mp.machine(rng, "dump truck"), x, y, s["yaw"] + U(rng, -0.3, 0.3), "vehicle")
    x, y = _P(s, -20, -10)
    b = mp.simple_building(rng, 6.0, 2.5, 2.6, "white", (0.85, 0.85, 0.83), "grey", (0.4, 0.4, 0.4), "white site office container in the quarry")
    add_static(pl, s, "office", _foundation(b["parts"], 6, 2.5), x, y, z, s["yaw"], "building", b["words"])
    _people(pl, s, rng, int(rng.integers(1, 4)), (-12, 12, -10, -4), role="worker")
    s["top"] = z + 6.0
    s["size_r"] = 30.0


def _pop_mountain_hut(pl, s, rng):
    h = mp.mountain_hut(rng)
    s["words"] = str(rng.choice(["the mountain hut", "the alpine refuge", "the mountain hut with the flag"]))
    s["at"] = "at the mountain hut"
    z = s["base_z"]
    s["top"] = add_static(pl, s, "hut", _foundation(h["parts"], 20, 12, depth=4.0, mat=mp.STONE), *s["xy"], z, s["yaw"], "hut", h["words"])
    tu, tw_, tL, tW, tz = h["terrace"]
    cx, cy = _P(s, tu, tw_)
    pl["decks"].append((cx, cy, s["yaw"], tL / 2, tW / 2, z + tz, "terrace"))
    s["terrace"] = (tw_, tW)
    sx, sy = _P(s, 12, -14)
    sp = mp.signpost(rng)
    add_static(pl, s, "signpost", sp["parts"], sx, sy, mv.terrain_height(sx, sy, pl), s["yaw"], "other", "yellow trail signpost at the mountain hut")
    _people(pl, s, rng, int(rng.integers(4, 10)), (-9, 9, tw_ - 2.5, tw_ + 2.5), role="hiker")
    _people(pl, s, rng, int(rng.integers(0, 3)), (-15, 15, 9, 12), role="hiker", prefix="q")
    s["size_r"] = 20.0


def _pop_cable_car(pl, s, rng):
    (x0, y0, z0), (x1, y1, z1) = s["valley"], s["top_st"]
    ang = s["yaw"]
    cn, col = str(rng.choice(["red", "yellow", "blue"])), None
    col = {"red": (0.70, 0.07, 0.05), "yellow": (0.92, 0.72, 0.08), "blue": (0.10, 0.25, 0.62)}[cn]
    s["words"] = "the cable car"
    s["at"] = "at the cable car"
    vs = mp.station(rng, False, col, cn)
    ts = mp.station(rng, True, col, cn)
    add_static(pl, s, "valley_station", _foundation(vs["parts"], 18, 12), x0, y0, z0, ang, "cable car", vs["words"])
    s["top"] = add_static(pl, s, "top_station", _foundation(ts["parts"], 18, 12, depth=5.0), x1, y1, z1, ang + math.pi, "cable car", ts["words"])
    s["stations"] = dict(valley=(x0, y0, z0 + vs["top_z"]), top=(x1, y1, z1 + ts["top_z"]))
    # cable line: from the valley station's cable wheel to the top station's, over pylons that keep it >= 12 m up
    cv = (x0 + math.cos(ang) * vs["cable_x"], y0 + math.sin(ang) * vs["cable_x"], z0 + vs["cable_z"])
    ct = (x1 - math.cos(ang) * ts["cable_x"], y1 - math.sin(ang) * ts["cable_x"], z1 + ts["cable_z"])
    L = math.hypot(ct[0] - cv[0], ct[1] - cv[1])
    for n_py in range(1, 7):
        sup = [cv]
        okay = True
        for k in range(1, n_py + 1):
            t = k / (n_py + 1)
            px, py = cv[0] + (ct[0] - cv[0]) * t, cv[1] + (ct[1] - cv[1]) * t
            sup.append((px, py, None))
        sup.append(ct)
        # pylon heights: lowest of 12..42 m that keeps every span >= 12 m above the ground
        for k in range(1, n_py + 1):
            px, py, _ = sup[k]
            g = mv.terrain_height(px, py, pl)
            sup[k] = (px, py, g + 14.0)
        for it in range(40):
            bad = False
            for k in range(len(sup) - 1):
                a, b = sup[k], sup[k + 1]
                span_ = math.hypot(b[0] - a[0], b[1] - a[1])
                for t in np.linspace(0.05, 0.95, 19):
                    x, y = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
                    zc = a[2] + (b[2] - a[2]) * t - 0.03 * span_ * 4 * t * (1 - t)
                    # next to a station the cable is low by design (it enters the station at 7 m)
                    ds = min(math.hypot(x - cv[0], y - cv[1]), math.hypot(x - ct[0], y - ct[1]))
                    need = min(12.0, 4.0 + 0.25 * ds)
                    if zc - mv.terrain_height(x, y, pl) < need:
                        bad = True
                        for j in (k, k + 1):
                            if 0 < j < len(sup) - 1:
                                sup[j] = (sup[j][0], sup[j][1], sup[j][2] + 2.0)
                        break
            if not bad:
                break
        hs = [sup[k][2] - mv.terrain_height(sup[k][0], sup[k][1], pl) for k in range(1, n_py + 1)]
        if not bad and max(hs) <= 42.0:
            break
    else:
        okay = False
    s["pylons"] = []
    for k in range(1, len(sup) - 1):
        px, py, top = sup[k]
        g = mv.terrain_height(px, py, pl) - 0.5
        h = top - g - 0.75
        add_static(pl, s, f"pylon_{k}", _foundation(mp.pylon(h), 6, 6, depth=2.0), px, py, g, ang, "cable car", f"steel pylon {k} of the cable car, about {h:.0f} m tall")
        s["pylons"].append((px, py, top))
    # two cables (y = +-3.2 m) with sag, drawn as thin segments; gondolas hang on them
    nx, ny = -math.sin(ang), math.cos(ang)
    segs, cab_pts = [], []
    for k in range(len(sup) - 1):
        a, b = sup[k], sup[k + 1]
        span = math.hypot(b[0] - a[0], b[1] - a[1])
        for t0, t1 in zip(np.linspace(0, 1, 9)[:-1], np.linspace(0, 1, 9)[1:]):
            pa = (a[0] + (b[0] - a[0]) * t0, a[1] + (b[1] - a[1]) * t0, a[2] + (b[2] - a[2]) * t0 - 0.03 * span * 4 * t0 * (1 - t0))
            pb = (a[0] + (b[0] - a[0]) * t1, a[1] + (b[1] - a[1]) * t1, a[2] + (b[2] - a[2]) * t1 - 0.03 * span * 4 * t1 * (1 - t1))
            segs.append((pa, pb))
            cab_pts.append(pa)
    cab_pts.append(sup[-1])
    parts = []
    for side in (-3.2, 3.2):
        for pa, pb in segs:
            parts.append(mp.segment((pa[0] + nx * side, pa[1] + ny * side, pa[2]), (pb[0] + nx * side, pb[1] + ny * side, pb[2]), 0.045, mp.DARK))
    add_static(pl, s, "cables", parts, 0.0, 0.0, 0.0, 0.0, "cable car", "steel haul cables of the cable car", obstacle=False)
    for pa, pb in segs:  # short boxes hugging the line (an AABB of a long diagonal span would cover a whole slope)
        L_ = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        for t in np.linspace(0.0, 1.0, max(2, int(L_ / 6.0) + 1)):
            cx_, cy_, cz_ = pa[0] + (pb[0] - pa[0]) * t, pa[1] + (pb[1] - pa[1]) * t, pa[2] + (pb[2] - pa[2]) * t
            pl["obstacles"].append((cx_ - 4.0, cx_ + 4.0, cy_ - 4.0, cy_ + 4.0, cz_ + 1.0, s["name"]))
    for k, (frac, side) in enumerate(((float(rng.uniform(0.2, 0.4)), -3.2), (float(rng.uniform(0.6, 0.8)), 3.2))):
        i = int(frac * (len(cab_pts) - 1))
        px, py, pz = cab_pts[i]
        g = mp.gondola(col)
        add_static(pl, s, f"cabin_{k}", g, px + nx * side, py + ny * side, pz, ang, "cable car", f"{cn} cable-car cabin hanging on the cable")
    s["line"] = [(p[0], p[1], p[2]) for p in sup]
    pl["cables"].append(dict(segs=[((sup[k][0], sup[k][1]), (sup[k + 1][0], sup[k + 1][1])) for k in range(len(sup) - 1)]))
    s["xy"] = ((x0 + x1) / 2, (y0 + y1) / 2)
    s["base_z"] = z0
    s["size_r"] = 30.0
    s["cable_ok"] = okay
    # the stations are landmarks of their own (an instruction can name either end)
    s["sub"] = {"valley station": (x0, y0, z0, z0 + vs["top_z"]), "top station": (x1, y1, z1, z1 + ts["top_z"])}
    _people(pl, s, rng, int(rng.integers(1, 4)), (-14, -9, -8, 8), role="hiker", prefix="v")


def _pop_paraglider_launch(pl, s, rng):
    s["words"] = "the paraglider launch site with the windsock"
    s["at"] = "at the paraglider launch site"
    z = s["base_z"]
    for k in range(int(rng.integers(1, 4))):
        x, y = _P(s, -9 + 8 * k, U(rng, -3, 3))
        wg = mp.wing(rng)
        add_static(pl, s, f"wing_{k}", wg["parts"], x, y, mv.terrain_height(x, y, pl), s["yaw"], "other", f"{wg['words']} at the launch site", obstacle=False)
    x, y = _P(s, 13, 7)
    ws = mp.windsock()
    s["top"] = add_static(pl, s, "windsock", ws["parts"], x, y, z, s["yaw"], "other", "orange windsock on a pole at the paraglider launch site")
    _people(pl, s, rng, int(rng.integers(2, 6)), (-12, 12, 4, 8), role="paraglider pilot")
    s["size_r"] = 20.0


def _pop_summit_cross(pl, s, rng):
    c = mp.summit_cross(rng)
    pk = next(p for p in pl["summits"] if p["name"] == s["summit"])
    s["words"] = "the summit cross"
    s["at"] = "on the summit next to the cross"
    s["top"] = add_static(pl, s, "cross", c["parts"], *s["xy"], s["base_z"], s["yaw"], "summit cross", c["words"])
    _people(pl, s, rng, int(rng.integers(1, 5)), (-5.5, 5.5, -5.5, 5.5), role="hiker")
    s["size_r"] = 9.0


def _pop_radio_mast(pl, s, rng):
    m = mp.mast(rng, float(rng.uniform(28, 44)))
    s["words"] = "the red-and-white radio mast"
    s["at"] = "by the radio mast"
    s["top"] = add_static(pl, s, "mast", _foundation(m["parts"], 4, 4), *s["xy"], s["base_z"], s["yaw"], "mast", m["words"])
    s["size_r"] = 12.0


def _pop_observatory(pl, s, rng):
    o = mp.observatory(rng)
    s["words"] = "the observatory with the white " + ("domes" if "two" in o["words"] else "dome")
    s["at"] = "at the observatory"
    s["top"] = add_static(pl, s, "observatory", _foundation(o["parts"], 14, 9), *s["xy"], s["base_z"], s["yaw"], "building", o["words"])
    _people(pl, s, rng, int(rng.integers(0, 3)), (-6, 6, -8, -6), role="villager")
    s["size_r"] = 14.0


def _pop_wind_farm(pl, s, rng):
    s["words"] = f"the wind turbines on the {mv.side_words(s['side'])} ridge"
    s["at"] = "by the wind turbines"
    yaw = U(rng, 0, 2 * math.pi)
    tops = []
    for k, (x, y, z) in enumerate(s["turbines"]):
        t = mp.wind_turbine(rng, h=float(rng.uniform(44, 52)))
        tops.append(add_static(pl, s, f"turbine_{k}", _foundation(t["parts"], 5, 5), x, y, z, yaw, "wind turbine", f"{t['words']}, number {k + 1} from the west"))
    s["top"] = max(tops)


def _pop_shepherd_hut(pl, s, rng):
    s["words"] = "the shepherd's stone hut with a flock of sheep"
    s["at"] = "on the high pasture by the shepherd's hut"
    b = mp.simple_building(rng, 7.0, 5.0, 3.0, "grey stone", mp.STONE, "dark", mp.DARK, "small {wall} shepherd's hut with a dark roof")
    s["top"] = add_static(pl, s, "hut", _foundation(b["parts"], 7, 5), *s["xy"], s["base_z"], s["yaw"], "hut", b["words"])
    for k in range(int(rng.integers(10, 19))):
        a, r = U(rng, 0, 2 * math.pi), U(rng, 9, 30)
        x, y = s["xy"][0] + r * math.cos(a), s["xy"][1] + r * math.sin(a)
        add_movable(pl, s, f"sheep{k}", mp.animal(rng, "sheep"), x, y, U(rng, 0, 6.28), "animal")
    x, y = _P(s, 6, -6)
    add_movable(pl, s, "dog", mp.animal(rng, "dog"), x, y, U(rng, 0, 6.28), "animal")
    _people(pl, s, rng, 1, (-3, 3, -7, -5), role="farmer")
    s["size_r"] = 32.0


def _pop_boathouse(pl, s, rng):
    lk = pl["valley"]["lake"]
    bh = mp.boathouse(rng)
    s["words"] = f"the {bh['words'][:-10].strip()} boathouse with the wooden jetty"
    s["at"] = "at the boathouse"
    s["top"] = add_static(pl, s, "boathouse", _foundation(bh["parts"], 10, 7), *s["xy"], s["base_z"], s["yaw"] + math.pi / 2, "building", f"{bh['words']} on the lake shore")
    sx, sy = s["shore"]
    L = float(rng.uniform(18, 26))
    yaw = s["yaw"]
    jx, jy = sx - math.cos(yaw) * 4.0, sy - math.sin(yaw) * 4.0
    jz = lk["z"] + 0.6
    beds = lambda u, w: mv.terrain_height(jx + math.cos(yaw) * u - math.sin(yaw) * w, jy + math.sin(yaw) * u + math.cos(yaw) * w, pl) - jz  # noqa: E731
    add_static(pl, s, "jetty", mp.jetty(L, beds), jx, jy, jz, yaw, "other", "wooden jetty on wooden piles reaching into the lake", obstacle=True)
    cx, cy = jx + math.cos(yaw) * L / 2, jy + math.sin(yaw) * L / 2
    pl["decks"].append((cx, cy, yaw, L / 2, 1.2, jz + 0.1, "jetty"))
    nx, ny = -math.sin(yaw), math.cos(yaw)
    u = 4.0
    for k in range(int(rng.integers(2, 5))):
        side = 1 if k % 2 == 0 else -1
        while u < L - 1.0:  # a moored boat floats: its whole hull over open water (boat_0 sat on the beach, audit 2026-10-21)
            bx, by = jx + math.cos(yaw) * u + nx * side * 2.6, jy + math.sin(yaw) * u + ny * side * 2.6
            if all(mv.water_level(bx + math.cos(yaw) * a + nx * c, by + math.sin(yaw) * a + ny * c, pl) is not None
                   for a in (-2.5, 0.0, 2.5) for c in (-0.9, 0.9)):
                break
            u += 1.0
        else:
            break
        u += 4.5
        b = mp.rowing_boat(rng)
        add_static(pl, s, f"boat_{k}", b["parts"], bx, by, lk["z"] - 0.05, yaw + U(rng, -0.15, 0.15), "boat", f"{b['words']} moored at the jetty")
    _people(pl, s, rng, int(rng.integers(1, 4)), (-8, 8, -6, 6), role="villager")
    s["size_r"] = 24.0
    s["jetty_end"] = (jx + math.cos(yaw) * L, jy + math.sin(yaw) * L)


# ================================================================================================== pads + spots
def _spot_clear(pl, x, y, z, r_obs=7.0, r_solid=2.5):
    for o in pl["obstacles"]:
        if o[4] > z + 1.0 and math.hypot(max(o[0] - x, 0, x - o[1]), max(o[2] - y, 0, y - o[3])) < r_obs:
            return False
    for sd in pl["solids"]:
        if sd[7] > z + 0.25 and mv._solid_hit(sd, x, y, r_solid):
            return False
    return True


def _dry_ring(pl, x, y, r=10.0):
    """No water at the place or within r m of it (a drone takes off / lands on dry ground, not at the water's edge)."""
    if mv.water_level(x, y, pl) is not None:
        return False
    for rr in (r / 2, r):
        for k in range(12):
            a = k * math.pi / 6
            if mv.water_level(x + rr * math.cos(a), y + rr * math.sin(a), pl) is not None:
                return False
    return True


def _spot_flat(pl, x, y, r=3.0, tol=0.35):
    zs = [mv.surface_z(x + dx, y + dy, pl) for dx in (-r, 0.0, r) for dy in (-r, 0.0, r)]
    return max(zs) - min(zs) < tol


def plan_pads_and_spots(pl, rng):
    v = pl["valley"]
    pads, spots = {}, []

    def ok_xy(x, y, r=3.0, tol=0.35, sep=35.0):
        if abs(x) > mv.EXTENT - 40 or abs(y) > mv.EXTENT - 40 or not _dry_ring(pl, x, y) or _in_water_zone(pl, x, y, 6.0):
            return False
        z = mv.surface_z(x, y, pl)
        if not _spot_flat(pl, x, y, r, tol) or not _spot_clear(pl, x, y, z):
            return False
        if any(math.hypot(x - a["xy"][0], y - a["xy"][1]) < sep for a in spots) or any(math.hypot(x - p["xy"][0], y - p["xy"][1]) < sep for p in pads.values()):
            return False
        return True

    def add_spot(kind, x, y, words, site=None, r=3.0, tol=0.35):
        if not ok_xy(x, y, r, tol):
            return False
        spots.append(dict(xy=(x, y), z=mv.surface_z(x, y, pl), words=words, site=site, spot=kind))
        return True

    # painted pads: the rescue helipad + 1-3 pads on meadows
    for s in pl["sites"]:
        if s["kind"] == "rescue station":
            x, y = _P(s, *s["heli_pad_uv"])
            z = s["base_z"]
            pads[f"P{len(pads) + 1}"] = dict(xy=(x, y), z=z + 0.03, words="the helipad of the mountain rescue station", site=s["name"])
    n_pads = int(rng.integers(1, 4))
    for _ in range(500):
        if sum(1 for p in pads.values() if p.get("site") is None) >= n_pads:
            break
        x = U(rng, -mv.EXTENT + 80, mv.EXTENT - 80)
        where = rng.random()
        if where < 0.65 and pl["landscape"] != "gorge":
            W = float(mv.valley_w(x, v))
            y = float(mv.valley_y(x, v)) + float(rng.choice([1, -1])) * U(rng, 0.2, 0.9) * W
            words_where = "on the valley floor"
        else:
            side = int(rng.choice([1, -1]))
            y = float(mv.valley_y(x, v)) + side * (float(mv.valley_w(x, v)) + v["Dw"] * U(rng, 1.1, 1.6))
            words_where = f"on the high ground beyond the {mv.side_words(side)} ridge"
        if abs(y - float(mv.river_y(x, v))) < v["rw"] + 14 or not _clear_of_sites(pl, x, y, 10):
            continue
        z0 = mv.terrain_height(x, y, pl)
        zs = [mv.terrain_height(x + a, y + b, pl) for a in (-8, 0, 8) for b in (-8, 0, 8)]
        if max(zs) - min(zs) > 2.5 or any(math.hypot(x - p["xy"][0], y - p["xy"][1]) < 120 for p in pads.values()):
            continue
        near = min(pl["sites"], key=lambda s: math.hypot(s["xy"][0] - x, s["xy"][1] - y))
        dn = math.hypot(near["xy"][0] - x, near["xy"][1] - y)
        cmp_ = _compass(x - near["xy"][0], y - near["xy"][1])
        words = (f"the landing pad {words_where}" if "beyond" in words_where else f"the landing pad {words_where} {cmp_} of {near['words']}") if dn < 260 else f"the landing pad {words_where} at the {mv.end_words(x)} end of the valley" if mv.end_words(x) != "middle" else f"the landing pad {words_where} in the middle of the valley"
        if any(p["words"] == words for p in pads.values()):
            continue
        z = float(np.median(zs))
        pl["zones"].append((x, y, 0.0, 8.0, 8.0, z, max(10.0, 1.4 * (max(zs) - min(zs)) + 6.0), "flat"))
        pads[f"P{len(pads) + 1}"] = dict(xy=(x, y), z=z + 0.03, words=words, site=None)
    if pl["task"] == "cross_pass":  # a landing pad on the far side of a pass (the mission lands beyond the ridge)
        for c in pl.get("cols", []):
            done = False
            for _ in range(200):
                x = c["x"] + U(rng, -220, 220)
                y = float(mv.valley_y(x, v)) + c["side"] * (float(mv.valley_w(x, v)) + v["Dw"] * U(rng, 1.15, 1.75))
                if abs(x) > mv.EXTENT - 70 or abs(y) > mv.EXTENT - 70 or _in_water_zone(pl, x, y, 10) or not _clear_of_sites(pl, x, y, 10):
                    continue
                zs = [mv.terrain_height(x + a, y + b, pl) for a in (-8, 0, 8) for b in (-8, 0, 8)]
                if max(zs) - min(zs) > 6.0 or not (90 < math.hypot(x - c["x"], y - c["y"]) < 480):
                    continue
                z = float(np.median(zs))
                pl["zones"].append((x, y, 0.0, 8.0, 8.0, z, max(12.0, 1.4 * (max(zs) - min(zs)) + 6.0), "flat"))
                pads[f"P{len(pads) + 1}"] = dict(xy=(x, y), z=z + 0.03, site=None,
                                                 words=f"the landing pad on the high ground beyond {c.get('words', 'the pass')}")
                done = True
                break
            if done:
                break
    if any(p.get("site") is None for p in pads.values()):
        mv.compute_heightmap(pl, zones=True)  # the pads levelled the ground under them
        for k, p in pads.items():
            p["z"] = mv.terrain_height(*p["xy"], pl) + 0.03 if p.get("site") is None else p["z"]
    if pl["index"] not in (1, 7, 28):
        # no painted pads (user 2026-10-21): the levelled pad places become unmarked flat clearings; only the rescue station
        # keeps its helipad marking as scenery (missions never use it)
        for k in [k for k, p in pads.items() if p.get("site") is None]:
            p = pads.pop(k)
            w = p["words"].replace("the landing pad on the high ground", "the flat clearing on the high ground") \
                .replace("the landing pad on the valley floor", "the flat open meadow on the valley floor").replace("the landing pad", "the flat clearing")
            if _dry_ring(pl, *p["xy"]) and _spot_clear(pl, *p["xy"], p["z"]):
                spots.append(dict(xy=p["xy"], z=p["z"] - 0.03, words=w, site=None, spot="clearing"))
    pl["pads"] = pads
    # unmarked spots near the sites
    for s in pl["sites"]:
        k, w = s["kind"], s["words"]
        for _ in range(40):
            if k == "trailhead":
                x, y = _P(s, U(rng, -12, 12), U(rng, -1.5, 1.5))
                if add_spot("car park", x, y, "the middle of the trailhead car park", s["name"], r=2.0):
                    break
            elif k == "campsite":
                x, y = _P(s, U(rng, -20, 20), float(rng.choice([-24.0, 24.0])))
                if add_spot("meadow", x, y, "the open grass at the edge of the campsite", s["name"]):
                    break
            elif k == "mountain hut":
                x, y = _P(s, U(rng, -10, 10), U(rng, 9.0, 12.5))
                if add_spot("hut", x, y, "the flat ground behind the mountain hut", s["name"], r=2.5):
                    break
            elif k == "summit cross":
                a = U(rng, 0, 2 * math.pi)
                x, y = s["xy"][0] + 3.6 * math.cos(a), s["xy"][1] + 3.6 * math.sin(a)
                if add_spot("summit", x, y, f"the summit plateau next to the summit cross", s["name"], r=2.0, tol=0.3):
                    break
            elif k == "village":
                x, y = _P(s, float(rng.choice([-58.0, 58.0])), U(rng, -12, 12))
                if add_spot("meadow", x, y, "the meadow at the edge of the village", s["name"]):
                    break
            elif k == "farm":
                x, y = _P(s, U(rng, -20, 25), U(rng, 26, 32))
                if add_spot("meadow", x, y, "the field beside the farm", s["name"]):
                    break
            elif k == "paraglider launch":
                x, y = _P(s, U(rng, -10, 10), U(rng, -8, -5))
                if add_spot("meadow", x, y, "the paraglider launch meadow", s["name"], r=2.5):
                    break
            elif k == "shepherd hut":
                a = U(rng, 0, 2 * math.pi)
                x, y = s["xy"][0] + 12 * math.cos(a), s["xy"][1] + 12 * math.sin(a)
                if add_spot("pasture", x, y, "the high pasture next to the shepherd's hut", s["name"], r=2.5, tol=0.5):
                    break
            elif k == "cable car":
                x1, y1, z1 = s["top_st"]
                a = s["yaw"] + math.pi / 2 * float(rng.choice([1, -1]))
                x, y = x1 + 13.5 * math.cos(a), y1 + 13.5 * math.sin(a)
                if add_spot("station", x, y, "the flat ground beside the cable car top station", s["name"], r=2.0):
                    break
            elif k == "boathouse":
                x, y = _P(s, U(rng, -12, 12), U(rng, -12, 12))
                if add_spot("shore", x, y, "the lakeshore meadow by the boathouse", s["name"], r=2.5, tol=0.5):
                    break
            elif k == "quarry":
                x, y = _P(s, U(rng, -15, 0), U(rng, -6, 6))
                if add_spot("quarry", x, y, "the floor of the stone quarry", s["name"], r=2.5):
                    break
            else:
                break
    # 1-2 open meadows on the valley floor
    for _ in range(300):
        if sum(1 for a in spots if a["spot"] == "valley meadow") >= int(rng.integers(1, 3)) or pl["landscape"] == "gorge":
            break
        x = U(rng, -mv.EXTENT + 80, mv.EXTENT - 80)
        y = float(mv.valley_y(x, v)) + float(rng.choice([1, -1])) * U(rng, 0.15, 0.8) * float(mv.valley_w(x, v))
        if abs(y - float(mv.river_y(x, v))) < v["rw"] + 12 or not _clear_of_sites(pl, x, y, 8):
            continue
        near = min(pl["sites"], key=lambda s: math.hypot(s["xy"][0] - x, s["xy"][1] - y))
        dn = math.hypot(near["xy"][0] - x, near["xy"][1] - y)
        ew = mv.end_words(x)
        words = f"the open meadow on the valley floor near {near['words']}" if dn < 230 else (
            f"the open meadow by the river at the {ew} end of the valley" if ew != "middle" else "the open meadow by the river in the middle of the valley")
        add_spot("valley meadow", x, y, words, tol=0.45)
    seen = {}
    for a in spots:
        seen.setdefault(a["words"], []).append(a)
    for wds, lst in seen.items():
        if len(lst) > 1:
            lst.sort(key=lambda a: a["xy"][0])
            for i, a in enumerate(lst):
                a["words"] = f"{wds} ({'the western one' if i == 0 else 'the eastern one' if i == len(lst) - 1 else 'the middle one'})"
    removed = [m for m in pl["movables"] if any(math.hypot(m["x"] - a["xy"][0], m["y"] - a["xy"][1]) < 4.0 for a in spots)]
    pl["movables"] = [m for m in pl["movables"] if m not in removed]
    pl["spots"] = {f"S{k + 1}": a for k, a in enumerate(spots)}
    pl["spot_cleared_props"] = [(m["group"], m["name"]) for m in removed]
