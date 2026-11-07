"""
City v2 (Easy City dataset, 2026-10-12): the authored city of city_scene.py, but a different city in every
simulation. User brief: "generate 100 simulations for Easy City but in environment create more objects for
difference in simulations".

What changes per simulation (plan(idx) is pure Python and deterministic, so the flight job, every render
slice and the offline checks build exactly the same city):
  * the ordinary buildings: new footprints, heights, setbacks and facades on every block;
  * 10-13 special blocks chosen among the 27 inner blocks, each one of: park, parking lot, construction
    site (tower crane), playground, football pitch, sports courts, market, fountain square, gas station,
    fire station (city_props.FEATURES), at a random orientation;
  * street life: cars of many types and colours in the lanes (queued before intersections) and parked at
    the kerb, taxis, buses, police cars, ambulances, trucks; pedestrians and dogs on the sidewalks and on
    zebra crossings; bus shelters with people waiting; traffic lights; roadworks with cones.
Fixed in every simulation: the street grid, the 58 m glass tower at (31, 31), the far-city surroundings,
the street furniture (lamps, benches, hydrants, bins...) and 8 landing pads.

Movable props become rigid bodies in the flight job exactly as in the Forest set (city ground is flat:
roads at +0.01 m, sidewalks +0.16 m, block ground -0.05 m, special-block and pad surfaces +0.03 m).
"""
import math

import numpy as np

import city_props as cp
import city_scene as cs
import props

SEED = 7
CENTRES = [-217.0, -155.0, -93.0, -31.0, 31.0, 93.0, 155.0, 217.0]  # block centres (odd multiples of 31 m)
INNER = [c for c in CENTRES if abs(c) < 200]
ROADS = cs.road_centre_offsets()  # -248 .. 248 every 62 m
TOWER = cs.TOWER_POS
TOWER_TOP = cs.TOWER_HEIGHT_M + 14.0  # mast tip
PADS = {
    "NW": ((-93.0, 93.0), "the north-west landing pad"),
    "SE": ((93.0, -93.0), "the south-east landing pad"),
    "W": ((-155.0, -31.0), "the west landing pad"),
    "E": ((155.0, 31.0), "the east landing pad"),
    "S": ((-31.0, -155.0), "the south landing pad"),
    "N": ((31.0, 155.0), "the north landing pad"),
    "FSW": ((-155.0, -155.0), "the far south-west landing pad"),
    "FNE": ((155.0, 155.0), "the far north-east landing pad"),
}
PAD_XY = [v[0] for v in PADS.values()]
BLOCK_INNER_HALF = 31.0 - cs.ROAD_HALF_WIDTH_M - cs.SIDEWALK_M  # 20.8 m
WALK_IN, WALK_OUT = cs.ROAD_HALF_WIDTH_M + 0.12, cs.ROAD_HALF_WIDTH_M + cs.SIDEWALK_M  # sidewalk band


# ---- ground ---------------------------------------------------------------------------------------------
def surface_z(x, y, features=None):
    """Height of the walkable/drivable surface at (x, y) (roads, sidewalks, block ground, special blocks)."""
    z = -0.05
    for c in ROADS:
        dx, dy = abs(x - c), abs(y - c)
        if dx <= cs.ROAD_HALF_WIDTH_M or dy <= cs.ROAD_HALF_WIDTH_M:
            z = max(z, 0.01)
        if WALK_IN - 0.12 < dx < WALK_OUT or WALK_IN - 0.12 < dy < WALK_OUT:
            z = max(z, cs.SIDEWALK_H)
    for (px, py) in PAD_XY:
        if abs(x - px) < 12 and abs(y - py) < 12:
            z = max(z, cp.SLAB_TOP)
    if features:
        for (bx, by) in features:
            if abs(x - bx) < BLOCK_INNER_HALF - 0.3 and abs(y - by) < BLOCK_INNER_HALF - 0.3:
                z = max(z, cp.SLAB_TOP)
    if abs(x - TOWER[0]) < cs.TOWER_PLAZA_R and abs(y - TOWER[1]) < cs.TOWER_PLAZA_R:
        z = max(z, 0.18)
    return z


def compass_words(x, y):
    if math.hypot(x, y) < 50:
        return "near the city centre"
    names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    return f"in the {names[int(round(math.atan2(y, x) / (math.pi / 4))) % 8]} part of the city"


# ---- plan -----------------------------------------------------------------------------------------------
def _rot(yaw, a, b):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * a - s * b, s * a + c * b


def plan_buildings(rng, skip):
    """Ordinary buildings for every block not in skip. Each gets its own detail seed, so the footprint/
    height plan (used for obstacle checks) does not depend on how many facade details were drawn."""
    out = []
    for i in range(len(ROADS) - 1):
        for j in range(len(ROADS) - 1):
            cx, cy = (ROADS[i] + ROADS[i + 1]) / 2, (ROADS[j] + ROADS[j + 1]) / 2
            if (cx, cy) in skip:
                continue
            x0, x1 = ROADS[i] + cs.ROAD_HALF_WIDTH_M + cs.SIDEWALK_M + cs.BLOCK_MARGIN_M, ROADS[i + 1] - cs.ROAD_HALF_WIDTH_M - cs.SIDEWALK_M - cs.BLOCK_MARGIN_M
            y0, y1 = ROADS[j] + cs.ROAD_HALF_WIDTH_M + cs.SIDEWALK_M + cs.BLOCK_MARGIN_M, ROADS[j + 1] - cs.ROAD_HALF_WIDTH_M - cs.SIDEWALK_M - cs.BLOCK_MARGIN_M
            n = int(rng.choice([1, 1, 2, 2, 3]))
            placed = []
            for k in range(n * 6):
                if len(placed) >= n:
                    break
                w = float(rng.uniform(10.0, min(24.0 if n == 1 else 16.0, x1 - x0)))
                d = float(rng.uniform(10.0, min(24.0 if n == 1 else 16.0, y1 - y0)))
                bx = float(rng.uniform(x0 + w / 2, x1 - w / 2))
                by = float(rng.uniform(y0 + d / 2, y1 - d / 2))
                if any(abs(bx - q["bx"]) < (w + q["w"]) / 2 + 2 and abs(by - q["by"]) < (d + q["d"]) / 2 + 2 for q in placed):
                    continue
                h = float(rng.uniform(9.0, 34.0)) if n < 3 else float(rng.uniform(8.0, 22.0))
                b = dict(bx=bx, by=by, w=w, d=d, h=h, detail_seed=int(rng.integers(1 << 30)),
                         setback=bool(h > 22.0 and rng.random() < 0.6))
                if b["setback"]:
                    b["h_base"] = h * float(rng.uniform(0.55, 0.72))
                    b["w2"], b["d2"] = w * float(rng.uniform(0.62, 0.82)), d * float(rng.uniform(0.62, 0.82))
                    b["ox"] = float(rng.uniform(-(w - b["w2"]) / 2, (w - b["w2"]) / 2)) * 0.7
                    b["oy"] = float(rng.uniform(-(d - b["d2"]) / 2, (d - b["d2"]) / 2)) * 0.7
                placed.append(b)
            out += placed
    return out


def _lane_slots(rng, n, kinds_fn, prefix, occupied, parked):
    """Vehicles along the streets inside +-190 m. parked=True: kerb lane (|offset| 5.4); else driving lane
    (|offset| 1.9), queued towards the next intersection. Right-hand traffic."""
    out = []
    tries = 0
    while len(out) < n and tries < n * 40:
        tries += 1
        axis = "NS" if rng.random() < 0.5 else "EW"
        c = float(rng.choice([r for r in ROADS if abs(r) < 200]))
        side = 1 if rng.random() < 0.5 else -1
        prop = kinds_fn(rng)
        L = prop["length"]
        s = float(rng.uniform(-190, 190))
        if any(abs(s - r) < cs.ROAD_HALF_WIDTH_M + 3.5 + L / 2 for r in ROADS):  # keep intersections clear
            continue
        off = side * (5.35 if parked else 1.9)
        # heading: on NS roads the +x lane drives north, on EW roads the -y lane drives east
        if axis == "NS":
            x, y = c + off, s
            yaw = math.pi / 2 if side > 0 else -math.pi / 2
        else:
            x, y = s, c + off
            yaw = 0.0 if side < 0 else math.pi
        if parked and rng.random() < 0.15:
            yaw += math.pi  # some park facing the other way
        key = (axis, c, side, parked)
        iv = occupied.setdefault(key, [])
        if any(abs(s - a) < (L + b) / 2 + (1.2 if parked else 2.0) for a, b in iv):
            continue
        iv.append((s, L))
        out.append((f"{prefix}_{len(out)}", prop, x, y, yaw + float(rng.uniform(-0.02, 0.02))))
    return out


def plan(idx, attempt=0):
    rng = np.random.default_rng(9_000_011 * idx + 31 + 7_919 * attempt)
    inner_blocks = [(x, y) for x in INNER for y in INNER if (x, y) not in PAD_XY and (x, y) != TOWER]
    n_feat = int(rng.integers(10, 14))
    kinds = list(cp.FEATURES)
    order = list(rng.permutation(kinds)) + list(rng.choice(kinds, n_feat))
    blocks = [inner_blocks[i] for i in rng.permutation(len(inner_blocks))[:n_feat]]
    features = {}
    for k, b in enumerate(blocks):
        kind = str(order[k])
        yaw = float(rng.integers(4)) * math.pi / 2
        f = cp.FEATURES[kind](np.random.default_rng(int(rng.integers(1 << 30))))
        f.update(xy=b, yaw=yaw, name=f"{kind.replace(' ', '_')}_{k + 1}")
        features[b] = f
    skip = set(features) | set(PAD_XY) | {TOWER}
    buildings = plan_buildings(rng, skip)

    # ---- street life ----
    occupied = {}
    ped_rng = np.random.default_rng(int(rng.integers(1 << 30)))
    # roadworks first (their lane stretch is then occupied), traffic lights and bus shelters next, so
    # vehicles and pedestrians are placed around them
    cones, roadworks = [], []
    for k in range(int(rng.integers(1, 4))):
        axis = "NS" if rng.random() < 0.5 else "EW"
        c = float(rng.choice([r for r in ROADS if abs(r) < 200]))
        s0 = float(rng.choice([r for r in ROADS if abs(r) < 180])) + float(rng.choice([-1, 1])) * 22
        side = int(rng.choice([-1, 1]))
        if any(r[0] == axis and r[1] == c for r in roadworks):
            continue
        for m in range(8):
            off, s_ = side * 1.6, s0 + m * 1.6
            x, y = (c + off, s_) if axis == "NS" else (s_, c + off)
            cones.append((f"cone_{k}_{m}", cp.traffic_cone(rng), x, y, 0.0))
        occupied.setdefault((axis, c, side, False), []).append((s0 + 5.0, 18.0))
        roadworks.append((axis, c, s0, side))
    lights = []
    for ix in [r for r in ROADS if abs(r) <= 186]:
        for iy in [r for r in ROADS if abs(r) <= 186]:
            if rng.random() < 0.55:
                lights.append((ix + 9.0, iy - 9.0, math.pi))
                lights.append((ix - 9.0, iy + 9.0, 0.0))
    furniture = street_furniture_points() + [(x, y) for x, y, _ in lights]
    shelters = []
    for k in range(int(rng.integers(5, 10))):
        for _ in range(30):
            c = float(rng.choice([r for r in ROADS if abs(r) < 200]))
            side = float(rng.choice([-1, 1]))
            s_ = float(rng.uniform(-180, 180))
            if any(abs(s_ - r) < 20 for r in ROADS):
                continue
            ns = rng.random() < 0.5
            x, y = (c + side * 9.3, s_) if ns else (s_, c + side * 9.3)
            if any(math.hypot(x - a, y - b) < 3.0 for a, b in furniture) or any(math.hypot(x - a[0], y - a[1]) < 6 for a in shelters):
                continue
            # local +y of the shelter (its back wall) faces away from the road
            yaw = (-math.pi / 2 if side > 0 else math.pi / 2) if ns else (0.0 if side > 0 else math.pi)
            shelters.append((x, y, yaw))
            break
    shelter_pts = [(x + _rot(yaw, a, 0)[0], y + _rot(yaw, a, 0)[1]) for x, y, yaw in shelters for a in (-2.2, -1.2, 0, 1.2, 1.9)]
    vehicles = []
    specials = (["city bus"] * int(rng.integers(3, 8)) + ["taxi"] * int(rng.integers(8, 16)) + ["police car"] * int(rng.integers(1, 4))
                + ["ambulance"] * int(rng.integers(0, 3)) + ["box truck"] * int(rng.integers(2, 6)) + ["school bus"] * int(rng.integers(0, 2)))
    for k, kind in enumerate(specials):
        parked = kind in ("box truck",) and rng.random() < 0.5
        vehicles += _lane_slots(rng, 1, lambda r, kind=kind: cp.special_vehicle(kind, r), f"{kind.replace(' ', '_')}_{k}", occupied, parked)
    vehicles += _lane_slots(rng, int(rng.integers(45, 75)), lambda r: cp.random_car(r), "car", occupied, False)
    vehicles += _lane_slots(rng, int(rng.integers(70, 110)), lambda r: cp.random_car(r), "parkedcar", occupied, True)

    # pedestrians on sidewalks (kept 1 m from street furniture, lights, shelters), some with dogs
    people = []
    for k, (x, y, yaw) in enumerate(shelters):  # people waiting in front of each shelter
        for m in range(int(rng.integers(1, 4))):
            lx, ly = _rot(yaw, -1.3 + m * 1.25, -1.5)  # in front of the shelter, clear of its posts
            people.append((f"waiting_{k}_{m}", cp.person(ped_rng, False, "stand"), x + lx, y + ly, yaw - math.pi / 2))
    avoid = furniture + shelter_pts
    n_ped = int(rng.integers(110, 180))
    tries = 0
    while len(people) < n_ped and tries < n_ped * 50:
        tries += 1
        c = float(ped_rng.choice([r for r in ROADS if abs(r) < 200]))
        off = float(ped_rng.choice([-1, 1])) * float(ped_rng.uniform(WALK_IN + 0.5, WALK_OUT - 0.4))
        s_ = float(ped_rng.uniform(-195, 195))
        x, y = (c + off, s_) if ped_rng.random() < 0.5 else (s_, c + off)
        if any(abs(s_ - r) < 12.0 for r in ROADS):  # corners: two sidewalks and kerbs meet
            continue
        if any(math.hypot(x - a, y - b) < 1.1 for a, b in avoid):
            continue
        if any(math.hypot(x - q[2], y - q[3]) < 1.2 for q in people):
            continue
        along = math.pi / 2 if abs(x - c) < abs(y - c) else 0.0
        yaw = along + (math.pi if ped_rng.random() < 0.5 else 0.0) + float(ped_rng.normal(0, 0.2))
        people.append((f"pedestrian_{len(people)}", cp.person(ped_rng, ped_rng.random() < 0.08), x, y, yaw))
        if ped_rng.random() < 0.08:  # walking a dog
            dx, dy = 0.9 * math.cos(yaw) - 0.5 * math.sin(yaw), 0.9 * math.sin(yaw) + 0.5 * math.cos(yaw)
            dd = abs((x + dx) - c) if abs(x - c) < abs(y - c) else abs((y + dy) - c)
            if (WALK_IN + 0.5 < dd < WALK_OUT - 0.5 and not any(math.hypot(x + dx - a, y + dy - b) < 1.0 for a, b in avoid)
                    and not any(math.hypot(x + dx - q[2], y + dy - q[3]) < 1.0 for q in people[:-1])):
                people.append((f"dog_{len(people)}", cp.dog(ped_rng), x + dx, y + dy, yaw))
    # people on the zebra crossings of 2-5 inner intersections
    for k in range(int(rng.integers(2, 6))):
        ix = float(rng.choice([r for r in ROADS if abs(r) <= 124]))
        iy = float(rng.choice([r for r in ROADS if abs(r) <= 124]))
        sgn = float(rng.choice([-1, 1]))
        for m in range(int(rng.integers(1, 4))):
            x = ix + float(rng.uniform(-5, 5))
            y = iy + sgn * (cs.ROAD_HALF_WIDTH_M - 1.5) + float(rng.uniform(-0.8, 0.8))
            if any(math.hypot(x - q[2], y - q[3]) < 1.2 for q in people):
                continue
            people.append((f"crossing_{k}_{m}", cp.person(ped_rng, False, "walk"), x, y, math.pi / 2 if rng.random() < 0.5 else -math.pi / 2))
    return dict(index=idx, attempt=attempt, features=features, buildings=buildings, vehicles=vehicles,
                people=people, shelters=shelters, cones=cones, roadworks=roadworks, lights=lights)


# ---- street furniture (fixed, same seeds as city_scene.build_props / build_trees) ----------------------------
_FURN = None
FURNITURE_SPEC = [  # city_scene.build_props without the covered cars (replaced by the v2 vehicles)
    ("street_lamp_01", 460, (0.9, 1.1), True, 4.0), ("street_lamp_02", 320, (0.9, 1.1), True, 4.0),
    ("fire_hydrant", 170, (0.9, 1.15), False, 0.0), ("bench", 260, (0.95, 1.1), True, 6.0),
    ("seating", 150, (0.9, 1.1), True, 6.0), ("trash_can", 240, (0.9, 1.1), False, 0.0),
    ("planter", 230, (0.9, 1.2), True, 5.0), ("poles", 130, (0.9, 1.1), True, 3.0),
]


def _clear_of_pads(x, y):
    return all(abs(x - px) > 16 or abs(y - py) > 16 for px, py in PAD_XY)


def _on_sidewalk(x, y):
    d = cs.distance_to_road(x, y)
    return _clear_of_pads(x, y) and cs.ROAD_HALF_WIDTH_M + 0.4 < d < cs.ROAD_HALF_WIDTH_M + cs.SIDEWALK_M


def furniture_instances():
    global _FURN
    if _FURN is None:
        rng = np.random.default_rng(SEED + 4)
        out = {}
        for key, n, srange, align, jit in FURNITURE_SPEC:
            out[key] = cs._prop_instances(rng, n, _on_sidewalk, srange, z_fn=lambda: cs.SIDEWALK_H, align=align, jitter_deg=jit)
        trng = np.random.default_rng(SEED + 6)

        def tree_ok(x, y):
            if not _clear_of_pads(x, y) or math.hypot(x - TOWER[0], y - TOWER[1]) < cs.TOWER_PLAZA_R:
                return False
            d = cs.distance_to_road(x, y)
            return cs.ROAD_HALF_WIDTH_M + 0.6 < d < cs.ROAD_HALF_WIDTH_M + cs.SIDEWALK_M + 1.0
        out["trees"] = cs._prop_instances(trng, 170, tree_ok, (0.22, 0.34), z_fn=lambda: cs.SIDEWALK_H)
        _FURN = out
    return _FURN


def street_furniture_points():
    return [(i[1][0], i[1][1]) for v in furniture_instances().values() for i in v]


# ---- realise movables -----------------------------------------------------------------------------------------
def movables(pl):
    """All movable props with their rest pose: [(group, name, prop, matrix, static_check, resid)]."""
    feats = list(pl["features"])
    gl = lambda x, y: surface_z(x, y, feats)
    out = []

    def add(group, name, prop, x, y, yaw):
        M, resid, tilt = props.rest_pose(prop, x, y, yaw, gl)
        out.append(dict(group=group, name=name, prop=prop, matrix=M, static_check=props.check_static(prop, tilt),
                        support_residual_m=round(resid, 4), xy=(x, y), yaw=yaw, kind=prop["kind"]))
    for (bx, by), f in pl["features"].items():
        for name, maker, lx, ly, lyaw, _z in f["movables"]:
            dx, dy = _rot(f["yaw"], lx, ly)
            add(f["name"], name, maker(), bx + dx, by + dy, lyaw + f["yaw"])
    for name, prop, x, y, yaw in pl["vehicles"] + pl["people"] + pl["cones"]:
        add("street", name, prop, x, y, yaw)
    return out


def obstacles(pl):
    """Axis-aligned boxes (xmin, xmax, ymin, ymax, top) of everything tall enough to matter to a drone."""
    obs = []
    for b in pl["buildings"]:
        top = b["h"] + 7.5  # rooftop plant / antenna masts up to 7 m
        obs.append((b["bx"] - b["w"] / 2 - 0.5, b["bx"] + b["w"] / 2 + 0.5, b["by"] - b["d"] / 2 - 0.5, b["by"] + b["d"] / 2 + 0.5, top))
    tw = (cs.TOWER_W + 3.0) / 2
    obs.append((TOWER[0] - tw, TOWER[0] + tw, TOWER[1] - tw, TOWER[1] + tw, TOWER_TOP))
    for (bx, by), f in pl["features"].items():
        for lx, ly, hx, hy, top in f["obstacles"]:
            dx, dy = _rot(f["yaw"], lx, ly)
            r = max(hx, hy)
            obs.append((bx + dx - r, bx + dx + r, by + dy - r, by + dy + r, top))
    return obs


# ---- USD -----------------------------------------------------------------------------------------------------
def _local_parts_to_world(parts, bx, by, yaw):
    return parts  # authored under a root xform translate(bx, by, 0) * rotZ(yaw)


def build(stage, nv, pl, assets_root, forest_root):
    from pxr import UsdGeom

    info = {}
    cs.build_lighting(stage, nv, assets_root)
    info.update(cs.build_surroundings(stage, nv, assets_root, seed=SEED + 14))
    info.update(cs.build_ground_and_roads(stage, nv, assets_root))
    info.update(cs.build_tower(stage, nv, assets_root))
    info.update(_build_buildings(stage, nv, pl, assets_root))
    info.update(_build_furniture(stage, nv, assets_root, forest_root))
    info.update(_build_pads(stage, nv, assets_root))
    info.update(_build_features(stage, nv, pl, assets_root, forest_root))
    info.update(_build_street(stage, nv, pl))
    rigid = []
    UsdGeom.Xform.Define(stage, "/World/Props")
    for m in movables(pl):
        path = f"/World/Props/{m['group']}/{m['name']}"
        UsdGeom.Xform.Define(stage, f"/World/Props/{m['group']}")
        props.author(stage, nv, path, m["prop"], root_matrix=m["matrix"])
        rigid.append({"path": path, "name": f"{m['group']}/{m['name']}", "kind": m["kind"], "mass_kg": m["prop"]["mass"],
                      "pos": [m["matrix"][3][0], m["matrix"][3][1], m["matrix"][3][2]]})
    info["rigid_bodies"] = rigid
    info["movable_props"] = len(rigid)
    return info


def _build_buildings(stage, nv, pl, assets_root):
    FACADE_TINTS = [(1.00, 0.98, 0.95), (0.86, 0.80, 0.72), (0.72, 0.74, 0.78), (0.95, 0.86, 0.70), (0.62, 0.58, 0.56),
                    (0.88, 0.72, 0.62), (0.78, 0.82, 0.80), (1.05, 1.02, 0.96)]
    facades = [cs._pbr(stage, nv, assets_root, f"Facade{i}_{k}", tex, tint=tint)
               for i, tex in enumerate(cs.FACADE_TEXTURES) for k, tint in enumerate(FACADE_TINTS)]
    roofs = [cs._pbr(stage, nv, assets_root, f"Roof{i}_{k}", tex, tint=tint)
             for i, tex in enumerate(cs.ROOF_TEXTURES) for k, tint in enumerate(((1.0, 1.0, 1.0), (0.80, 0.78, 0.76), (0.66, 0.60, 0.58)))]
    trim = nv.create_color_material(stage, "/World/Materials/CityTrim", (0.78, 0.77, 0.74), 0.45)
    metal = nv.create_color_material(stage, "/World/Materials/RoofPlant", (0.55, 0.56, 0.58), 0.55, metallic=0.6)
    win_glass = nv.create_color_material(stage, "/World/Materials/CityGlass", (0.06, 0.09, 0.13), 0.09)
    shop_glass = nv.create_color_material(stage, "/World/Materials/ShopGlass", (0.10, 0.13, 0.16), 0.12)
    awning_cols = [nv.create_color_material(stage, f"/World/Materials/Awning{i}", c, 0.65)
                   for i, c in enumerate(((0.62, 0.18, 0.16), (0.16, 0.34, 0.52), (0.20, 0.42, 0.26), (0.70, 0.60, 0.26)))]
    shops = 0
    for n, b in enumerate(pl["buildings"]):
        rng = np.random.default_rng(b["detail_seed"])
        root = f"/World/City/Blocks/B_{n:03d}"
        bx, by, w, d, h = b["bx"], b["by"], b["w"], b["d"], b["h"]
        fac = facades[int(rng.integers(len(facades)))]
        floor_h = float(rng.uniform(3.1, 3.8))
        style = "ribbon" if rng.random() < 0.45 else "punched"
        if b["setback"]:
            hb = b["h_base"]
            cs._building_tier(nv, stage, root, "base", bx, by, 0.0, w, d, hb, fac, style, floor_h, win_glass, trim, rng)
            cs.textured_box(nv, stage, f"{root}/terrace", (bx, by, hb + 0.16), (w + 0.3, d + 0.3, 0.32), roofs[int(rng.integers(len(roofs)))], tile_m=2.2)
            tx, ty = bx + b["ox"], by + b["oy"]
            cs._building_tier(nv, stage, root, "top", tx, ty, hb, b["w2"], b["d2"], h - hb, fac, style, floor_h, win_glass, trim, rng)
            cs._rooftop_clutter(nv, stage, root, tx, ty, b["w2"], b["d2"], h, trim, metal, rng)
            cs.textured_box(nv, stage, f"{root}/roof", (tx, ty, h + 0.16), (b["w2"] + 0.3, b["d2"] + 0.3, 0.32), roofs[int(rng.integers(len(roofs)))], tile_m=2.2)
            cs._parapet(nv, stage, root, "top", tx, ty, b["w2"], b["d2"], h + 0.32, trim)
        else:
            cs._building_tier(nv, stage, root, "base", bx, by, 0.0, w, d, h, fac, style, floor_h, win_glass, trim, rng)
            cs._rooftop_clutter(nv, stage, root, bx, by, w, d, h, trim, metal, rng)
            cs.textured_box(nv, stage, f"{root}/roof", (bx, by, h + 0.16), (w + 0.3, d + 0.3, 0.32), roofs[int(rng.integers(len(roofs)))], tile_m=2.2)
            cs._parapet(nv, stage, root, "main", bx, by, w, d, h + 0.32, trim)
        if rng.random() < 0.75:
            for axis, sx, sy in (("N", 0.0, 1.0), ("S", 0.0, -1.0), ("E", 1.0, 0.0), ("W", -1.0, 0.0)):
                span = (w - 2.0) if axis in ("N", "S") else (d - 2.0)
                if span <= 2.0:
                    continue
                gx, gy = bx + sx * (w / 2.0 - 0.12), by + sy * (d / 2.0 - 0.12)
                ww, dd = (span, 0.2) if axis in ("N", "S") else (0.2, span)
                nv.cube(stage, f"{root}/shop{axis}", (gx, gy, 1.9), (ww, dd, 3.0), shop_glass)
                aw = awning_cols[int(rng.integers(len(awning_cols)))]
                ax, ay = bx + sx * (w / 2.0 + 0.55), by + sy * (d / 2.0 + 0.55)
                aww, add = (span * 0.8, 1.3) if axis in ("N", "S") else (1.3, span * 0.8)
                nv.cube(stage, f"{root}/awn{axis}", (ax, ay, 3.6), (aww, add, 0.22), aw)
                shops += 1
    return {"buildings": len(pl["buildings"]), "shopfronts": shops}


# Not built (but their positions stay in furniture_instances(), so every plan / flight is unchanged):
# Poly Haven "modular_electricity_poles" and "modular_street_seating" are KITS of loose parts laid out for
# assembly -- 90 of the poles' 103 pieces and 11 of the seating's 16 hang in the air -- and "street_lamp_02" is
# a 1.7 m lamp head sunk 0.39 m into the ground. User: "there are some physical issues" (2026-10-12).
NOT_BUILT = {"poles", "seating", "street_lamp_02"}


def _build_furniture(stage, nv, assets_root, forest_root):
    inst = furniture_instances()
    counts = {}
    for key, *_ in FURNITURE_SPEC:
        if key in NOT_BUILT:
            continue
        mat = nv.create_asset_material(stage, f"/World/Materials/Prop_{key}", assets_root, cs.PROP_MODELS[key])
        nv.point_instancer(stage, f"/World/City/Props/{key}", [(cs.model_path(assets_root, cs.PROP_MODELS[key]), 1.0, mat)], inst[key])
        counts[f"prop_{key}"] = len(inst[key])
    mat = nv.create_asset_material(stage, "/World/Materials/CityTree", forest_root, nv.ASSET_IDS["mature_tree"])
    nv.point_instancer(stage, "/World/City/Trees", [(nv.asset_path(forest_root, "mature_tree"), 1.0, mat)], inst["trees"])
    counts["street_trees"] = len(inst["trees"])
    return counts


def _build_pads(stage, nv, assets_root):
    slab = cs._pbr(stage, nv, assets_root, "PadConcrete", "concrete_panels")
    paint = nv.create_color_material(stage, "/World/Materials/PadPaint", (0.95, 0.80, 0.10), 0.5)
    white = nv.create_color_material(stage, "/World/Materials/PadWhite", (0.92, 0.92, 0.90), 0.5)
    for k, (key, ((px, py), _)) in enumerate(PADS.items()):
        root = f"/World/City/Pads/{key}"
        cs.textured_box(nv, stage, f"{root}/slab", (px, py, -0.015), (24, 24, 0.09), slab, tile_m=3.0)
        for j in range(32):
            a = 2 * math.pi * j / 32
            nv.cube(stage, f"{root}/ring{j:02d}", (px + 7.5 * math.cos(a), py + 7.5 * math.sin(a), 0.035), (1.6, 0.35, 0.02), paint,
                    rotation_z=math.degrees(a) + 90)
        for (dx, sx, sy) in ((-1.6, 0.6, 5.0), (1.6, 0.6, 5.0), (0.0, 3.2, 0.6)):
            nv.cube(stage, f"{root}/H{int(dx * 10) + 20}{int(sx * 10)}", (px + dx, py, 0.036), (sx, sy, 0.02), white)
    return {"landing_pads": len(PADS)}


def _build_features(stage, nv, pl, assets_root, forest_root):
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, "/World/City/Features")
    slabs = {}
    counts = {}
    tree_inst = []
    for (bx, by), f in pl["features"].items():
        tex, src, tile = f["slab"]
        key = (tex, src)
        if key not in slabs:
            root = forest_root if src == "forest" else assets_root
            slabs[key] = cs.create_pbr_material_fixed(stage, nv, f"/World/Materials/FeatureSlab_{tex}", root, tex)
        cs.textured_box(nv, stage, f"/World/City/Features/{f['name']}_slab", (bx, by, -0.015),
                        (2 * BLOCK_INNER_HALF - 0.4, 2 * BLOCK_INNER_HALF - 0.4, 0.09), slabs[key], tile_m=tile)
        props.author(stage, nv, f"/World/City/Features/{f['name']}", dict(parts=f["parts"]), root_xy_yaw=(bx, by, f["yaw"]))
        for (lx, ly, s) in f.get("trees", []):
            dx, dy = _rot(f["yaw"], lx, ly)
            tree_inst.append((0, (bx + dx, by + dy, cp.SLAB_TOP), float(np.random.default_rng(int(abs(lx * 100 + ly))).uniform(-3, 3)), (s, s, s)))
        counts[f["kind"]] = counts.get(f["kind"], 0) + 1
    if tree_inst:
        mat = nv.create_asset_material(stage, "/World/Materials/ParkTree", forest_root, nv.ASSET_IDS["mature_tree"])
        nv.point_instancer(stage, "/World/City/ParkTrees", [(nv.asset_path(forest_root, "mature_tree"), 1.0, mat)], tree_inst, partition=False)
    return {"special_blocks": counts}


def _build_street(stage, nv, pl):
    from pxr import UsdGeom

    UsdGeom.Xform.Define(stage, "/World/City/StreetExtras")
    for k, (x, y, yaw) in enumerate(pl["shelters"]):
        props.author(stage, nv, f"/World/City/StreetExtras/Shelter_{k}", dict(parts=cp.bus_shelter(cs.SIDEWALK_H)), root_xy_yaw=(x, y, yaw))
    for k, (x, y, yaw) in enumerate(pl["lights"]):
        props.author(stage, nv, f"/World/City/StreetExtras/Light_{k}", dict(parts=cp.traffic_light(cs.SIDEWALK_H, 1)), root_xy_yaw=(x, y, yaw))
    for k, (axis, c, s0, side) in enumerate(pl["roadworks"]):  # a barrier at the head of each closure
        x, y = (c + side * 1.6, s0 - 1.5) if axis == "NS" else (s0 - 1.5, c + side * 1.6)
        props.author(stage, nv, f"/World/City/StreetExtras/Works_{k}",
                     dict(parts=[props.box((0, 0, 0.6), (0.2, 3.0, 0.25), (0.92, 0.92, 0.9)), props.box((0, 0, 0.3), (0.15, 0.15, 0.6), "dark_metal")]),
                     root_xy_yaw=(x, y, math.pi / 2 if axis == "NS" else 0.0))
    return {"bus_shelters": len(pl["shelters"]), "traffic_lights": len(pl["lights"]), "roadworks": len(pl["roadworks"])}
