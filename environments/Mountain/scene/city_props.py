"""
City props for the Easy City dataset (2026-10-12): people, many kinds of vehicles, traffic cones, and the
special blocks that make every simulation's city different (park, parking lot, construction site with a
tower crane, playground, football pitch, sports courts, market, fountain square, gas station, fire
station), plus street furniture (bus shelters, traffic lights).

Same rules as Forest/scene/props.py (whose helpers this imports): simple primitives at real dimensions and
masses. Movable things (people, dogs, vehicles, cones, balls) have their support points at local z = 0 and
become PhysX rigid bodies in the flight job; static things (structures) stand on the ground.

Static feature builders work in the block's local frame (block centre = origin, x east, y north, z world
height) and return:
    parts      static primitives (props.author with root translate(x, y, 0) * rotZ(yaw))
    movables   [(name, maker, lx, ly, lyaw, surface_z)] placed later on the ground as rigid bodies
    obstacles  [(lx, ly, half_x, half_y, top_z)] for the flight-path clearance check
    slab       (texture_id, assets, tile_m) ground surface of the block, or None
    top_z      height of the feature for camera framing
"""
import math

import numpy as np

import props
from props import box, cap, cone, cyl, sph

P = props.COLORS
SLAB_TOP = 0.03  # feature ground surfaces: slab from -0.06 to +0.03

SHIRTS = [(0.70, 0.12, 0.10), (0.12, 0.25, 0.55), (0.92, 0.92, 0.90), (0.10, 0.10, 0.11), (0.20, 0.45, 0.25),
          (0.85, 0.68, 0.15), (0.55, 0.30, 0.60), (0.95, 0.50, 0.20), (0.45, 0.55, 0.65), (0.75, 0.40, 0.45)]
PANTS = [(0.10, 0.14, 0.28), (0.12, 0.12, 0.13), (0.35, 0.30, 0.22), (0.45, 0.45, 0.47), (0.20, 0.22, 0.30)]
SKINS = [(0.87, 0.70, 0.58), (0.76, 0.57, 0.43), (0.55, 0.38, 0.27), (0.36, 0.24, 0.17), (0.93, 0.78, 0.66)]
HAIRS = [(0.08, 0.06, 0.05), (0.30, 0.18, 0.08), (0.62, 0.45, 0.22), (0.55, 0.55, 0.55), (0.15, 0.10, 0.07)]
CAR_PAINT = {"white": (0.86, 0.86, 0.85), "black": (0.04, 0.04, 0.045), "silver": (0.60, 0.61, 0.63),
             "red": (0.58, 0.05, 0.04), "blue": (0.07, 0.17, 0.45), "dark green": (0.08, 0.22, 0.13),
             "grey": (0.33, 0.34, 0.35), "orange": (0.85, 0.38, 0.08), "beige": (0.72, 0.64, 0.50)}
TAXI_YELLOW = (0.95, 0.72, 0.05)

MASS = {"person": 72.0, "child": 30.0, "dog": 30.0, "sedan": 1400.0, "hatchback": 1150.0, "suv": 2000.0,
        "van": 2500.0, "pickup": 2100.0, "taxi": 1450.0, "police car": 1600.0, "ambulance": 3500.0,
        "city bus": 12000.0, "school bus": 11000.0, "box truck": 8000.0, "fire truck": 15000.0,
        "food truck": 4500.0, "dump truck": 14000.0, "excavator": 20000.0, "cone": 4.5, "ball": 0.43}


# ---- people ---------------------------------------------------------------------------------------------
def person(rng, child=False, pose=None, shirt=None):
    """Standing or walking person, ~1.55-1.90 m (children 1.05-1.35 m). Support = the four corners of the
    two feet; the body is centred over them (a statue of a person stands, so does this)."""
    h = float(rng.uniform(1.05, 1.35)) if child else float(rng.uniform(1.58, 1.90))
    s = h / 1.75
    _shirt = SHIRTS[int(rng.integers(len(SHIRTS)))]
    shirt = shirt or _shirt
    pants = PANTS[int(rng.integers(len(PANTS)))]
    skin = SKINS[int(rng.integers(len(SKINS)))]
    hair = HAIRS[int(rng.integers(len(HAIRS)))]
    pose = pose or ("walk" if rng.random() < 0.5 else "stand")
    fx = (0.17, -0.14) if pose == "walk" else (0.02, 0.02)
    p, sup = [], []
    for k, side in enumerate((-1, 1)):
        x = fx[k] * s
        p.append(box((x + 0.04 * s, side * 0.1 * s, 0.035), (0.26 * s, 0.10 * s, 0.07), (0.10, 0.08, 0.07)))
        sup += [(x + 0.04 * s + dx, side * 0.1 * s + dy) for dx in (-0.12 * s, 0.12 * s) for dy in (-0.045 * s, 0.045 * s)]
        lean = math.degrees(math.atan2(-x, 0.85 * s)) if pose == "walk" else 0.0
        p.append(cyl((x / 2, side * 0.1 * s, 0.07 + 0.42 * s), 0.07 * s, 0.84 * s, pants, rot=(0, lean, 0)))
    p.append(box((0, 0, 0.95 * s), (0.22 * s, 0.34 * s, 0.18 * s), pants))
    p.append(cap((0, 0, 1.24 * s), 0.17 * s, 0.30 * s, shirt, axis="Z"))
    for k, side in enumerate((-1, 1)):
        swing = (15 if k == 0 else -15) if pose == "walk" else 0
        p.append(cyl((0, side * 0.235 * s, 1.17 * s), 0.045 * s, 0.6 * s, shirt, rot=(0, swing, 0), col=False))
        p.append(sph((0.0, side * 0.235 * s, 0.86 * s), 0.05 * s, skin))
    p.append(cyl((0, 0, 1.53 * s), 0.05 * s, 0.1 * s, skin, col=False))
    p.append(sph((0, 0, 1.65 * s), 0.11 * s, skin, col=True))
    p.append(sph((-0.015 * s, 0, 1.69 * s), 0.112 * s, hair, scale=(1, 1, 0.75)))
    return dict(kind="child" if child else "person", parts=p, support=sup, mass=MASS["child" if child else "person"],
                height=h, length=0.5, dynamic=True)


def dog(rng):
    return props.dog(str(rng.choice(list(props.DOG_COATS))), rng)


# ---- vehicles ---------------------------------------------------------------------------------------------
VEHICLES = {  # length, width, body height, cabin(len frac, height), wheel r, wheelbase frac
    "sedan": (4.6, 1.80, 0.62, (0.52, 0.46), 0.32, 0.60),
    "hatchback": (4.0, 1.75, 0.66, (0.55, 0.52), 0.31, 0.62),
    "suv": (4.8, 1.95, 0.85, (0.62, 0.55), 0.38, 0.60),
    "taxi": (4.7, 1.82, 0.62, (0.52, 0.46), 0.32, 0.60),
    "police car": (4.9, 1.85, 0.64, (0.52, 0.46), 0.33, 0.60),
    "pickup": (5.3, 1.95, 0.85, (0.34, 0.60), 0.38, 0.62),
    "van": (5.2, 2.00, 1.05, (0.80, 0.95), 0.36, 0.64),
    "ambulance": (5.8, 2.10, 1.25, (0.85, 1.25), 0.38, 0.62),
    "food truck": (6.2, 2.25, 1.30, (0.85, 1.30), 0.40, 0.60),
    "box truck": (7.6, 2.45, 0.9, None, 0.48, 0.60),
    "city bus": (12.0, 2.55, 1.0, (0.96, 1.85), 0.50, 0.62),
    "school bus": (10.5, 2.45, 1.0, (0.94, 1.70), 0.48, 0.64),
    "fire truck": (9.0, 2.50, 1.1, None, 0.52, 0.58),
}


def vehicle(kind, color, rng):
    L, W, bh, cab_spec, r, wb = VEHICLES[kind]
    paint = color
    p = []
    wx, wy = L * wb / 2.0, W / 2.0 - 0.18
    axles = [-wx, wx] + ([-wx + 1.4] if kind in ("city bus", "box truck", "fire truck") and L > 7 else [])
    for x in axles:
        for y in (-wy, wy):
            p.append(cyl((x, y, r), r, 0.26, "tyre", axis="Y", col=False))
            # the tyre's contact patch: a flat box collider (a rigid cylinder on a triangle mesh rocks and rolls)
            p.append(box((x, y, 0.05), (min(0.35, r), 0.24, 0.1), "tyre"))
            p.append(cyl((x, y + math.copysign(0.135, y), r), r * 0.55, 0.01, "rim", axis="Y", col=False))
    z0 = r * 0.85
    if kind == "box truck":
        p.append(box((L / 2 - 1.1, 0, z0 + 0.9), (2.2, W - 0.1, 1.8), paint))
        p.append(box((L / 2 - 0.15, 0, z0 + 1.3), (0.06, W - 0.4, 0.7), "glass", col=False))
        p.append(box((-1.0, 0, z0 + 1.55), (L - 2.4, W, 2.8), (0.90, 0.90, 0.88)))
        top = z0 + 2.95
    elif kind == "fire truck":
        p.append(box((0, 0, z0 + 0.75), (L, W, 1.5), paint))
        p.append(box((L / 2 - 1.3, 0, z0 + 1.95), (2.4, W - 0.1, 1.0), paint))
        p.append(box((L / 2 - 0.25, 0, z0 + 1.95), (0.06, W - 0.4, 0.7), "glass", col=False))
        for s_ in (-1, 1):  # ladder on the roof
            p.append(box((-0.8, s_ * 0.45, z0 + 1.75), (L - 3.2, 0.08, 0.12), "metal", col=False))
        for k in range(12):
            p.append(box((-0.8 - (L - 3.4) / 2 + k * (L - 3.4) / 11, 0, z0 + 1.75), (0.06, 0.9, 0.06), "metal", col=False))
        p.append(box((L / 2 - 1.3, 0, z0 + 2.5), (0.3, 1.2, 0.12), (0.9, 0.1, 0.1), col=False))
        p.append(box((0, 0, z0 + 0.25), (L + 0.02, W + 0.02, 0.12), "trim", col=False))
        top = z0 + 2.6
    else:
        p.append(box((0, 0, z0 + bh / 2), (L, W, bh), paint))
        cl, ch = cab_spec
        cab_len = L * cl
        cx = 0.0 if cl > 0.75 else (-0.15 if kind != "pickup" else L / 2 - cab_len / 2 - 0.9)
        p.append(box((cx, 0, z0 + bh + ch / 2), (cab_len, W - 0.12, ch), "glass"))
        p.append(box((cx, 0, z0 + bh + ch - 0.04), (cab_len - 0.12, W - 0.2, 0.09), paint, col=False))
        if cl > 0.75:  # tall vehicles: painted pillars between the windows
            n = int(cab_len / 1.3)
            for k in range(n + 1):
                xx = cx - cab_len / 2 + k * cab_len / n
                p.append(box((xx, 0, z0 + bh + ch / 2), (0.12, W - 0.1, ch), paint, col=False))
        top = z0 + bh + ch
        if kind == "pickup":
            for s_ in (-1, 1):
                p.append(box((-L / 4, s_ * (W / 2 - 0.05), z0 + bh + 0.25), (L / 2, 0.08, 0.5), paint, col=False))
        if kind == "taxi":
            p.append(box((cx, 0, top + 0.12), (0.5, 0.25, 0.2), (0.95, 0.95, 0.9), col=False))
        if kind in ("police car", "ambulance"):
            p.append(box((cx, -0.3, top + 0.08), (0.3, 0.55, 0.12), (0.85, 0.05, 0.05), col=False))
            p.append(box((cx, 0.3, top + 0.08), (0.3, 0.55, 0.12), (0.05, 0.15, 0.85), col=False))
        if kind == "police car":
            for s_ in (-1, 1):
                p.append(box((0, s_ * (W / 2 + 0.005), z0 + bh / 2), (L * 0.55, 0.01, bh * 0.9), "black", col=False))
        if kind == "ambulance":
            for s_ in (-1, 1):
                p.append(box((0, s_ * (W / 2 + 0.005), z0 + bh * 0.6), (L * 0.95, 0.01, 0.25), (0.85, 0.08, 0.06), col=False))
        if kind == "food truck":
            p.append(box((0.3, -(W / 2 + 0.005), z0 + bh + 0.7), (2.6, 0.01, 0.9), "black", col=False))  # serving hatch
            p.append(box((0.3, -(W / 2 + 0.45), z0 + bh + 1.2), (2.8, 0.9, 0.06), (0.9, 0.2, 0.15), col=False))  # awning
    for s_ in (-1, 1):
        p.append(box((L / 2 + 0.005, s_ * (W / 2 - 0.3), z0 + bh * 0.6), (0.01, 0.3, 0.14), "light", col=False))
        p.append(box((-L / 2 - 0.005, s_ * (W / 2 - 0.3), z0 + bh * 0.6), (0.01, 0.28, 0.12), "tail_light", col=False))
    sup = [(x, y) for x in (axles[0], axles[1]) for y in (-wy, wy)]
    return dict(kind=kind, parts=p, support=sup, mass=MASS[kind], height=top, length=L, width=W, dynamic=True)


def random_car(rng, kinds=("sedan", "sedan", "hatchback", "suv", "suv", "van", "pickup")):
    kind = str(rng.choice(list(kinds)))
    color = CAR_PAINT[str(rng.choice(list(CAR_PAINT)))]
    return vehicle(kind, color, rng)


def special_vehicle(kind, rng):
    color = {"taxi": TAXI_YELLOW, "police car": (0.92, 0.92, 0.92), "ambulance": (0.93, 0.93, 0.92),
             "city bus": [(0.10, 0.35, 0.65), (0.75, 0.10, 0.08), (0.15, 0.50, 0.30)][int(rng.integers(3))],
             "school bus": (0.95, 0.68, 0.05), "box truck": (0.85, 0.85, 0.85), "fire truck": (0.72, 0.06, 0.04),
             "food truck": [(0.95, 0.85, 0.30), (0.30, 0.60, 0.75), (0.90, 0.45, 0.55)][int(rng.integers(3))]}[kind]
    return vehicle(kind, color, rng)


def excavator(rng):
    yellow = (0.92, 0.66, 0.06)
    p = []
    for s_ in (-1, 1):
        p.append(box((0, s_ * 1.2, 0.45), (4.4, 0.7, 0.9), "dark_metal"))
    p.append(box((0, 0, 1.25), (3.2, 2.6, 0.7), yellow))
    p.append(box((-1.0, 0, 1.95), (1.4, 2.4, 0.9), yellow))  # engine / counterweight
    p.append(box((0.7, 0.75, 2.2), (1.2, 0.95, 1.2), "glass"))  # cab
    p.append(box((1.9, -0.3, 2.6), (2.8, 0.35, 0.45), yellow, rot=(0, -35, 0)))  # boom
    p.append(box((3.6, -0.3, 2.4), (0.35, 0.3, 2.0), yellow, rot=(0, 20, 0)))  # stick
    p.append(box((4.0, -0.3, 1.2), (0.7, 0.9, 0.6), "dark_metal"))  # bucket
    sup = [(x, y) for x in (-2.2, 2.2) for y in (-1.55, 1.55)]
    return dict(kind="excavator", parts=p, support=sup, mass=MASS["excavator"], height=3.3, length=6.0, width=3.1, dynamic=True)


def dump_truck(rng):
    yellow = (0.92, 0.66, 0.06)
    p = []
    for x in (-2.2, -0.8, 2.6):
        for y in (-1.05, 1.05):
            p.append(cyl((x, y, 0.62), 0.62, 0.45, "tyre", axis="Y", col=False))
            p.append(box((x, y, 0.05), (0.4, 0.42, 0.1), "tyre"))
    p.append(box((0, 0, 1.25), (7.0, 2.3, 0.4), "dark_metal"))
    p.append(box((2.6, 0, 2.2), (1.8, 2.3, 1.6), yellow))
    p.append(box((3.45, 0, 2.5), (0.06, 1.9, 0.8), "glass", col=False))
    p.append(box((-1.0, 0, 2.3), (4.4, 2.5, 1.7), yellow))
    p.append(box((-1.0, 0, 3.05), (4.2, 2.3, 0.2), (0.45, 0.35, 0.25), col=False))  # load of soil
    sup = [(x, y) for x in (-2.2, 2.6) for y in (-1.05, 1.05)]
    return dict(kind="dump truck", parts=p, support=sup, mass=MASS["dump truck"], height=3.2, length=7.0, width=2.5, dynamic=True)


def traffic_cone(rng):
    p = [box((0, 0, 0.015), (0.38, 0.38, 0.03), (0.95, 0.35, 0.05)),
         cyl((0, 0, 0.25), 0.07, 0.4, (0.95, 0.35, 0.05)),
         cone((0, 0, 0.36), 0.15, 0.66, (0.95, 0.35, 0.05)),
         cyl((0, 0, 0.45), 0.105, 0.08, "white_coat", col=False)]
    return dict(kind="cone", parts=p, support=[(x, y) for x in (-0.18, 0.18) for y in (-0.18, 0.18)], mass=MASS["cone"],
                height=0.7, length=0.4, width=0.4, dynamic=True)


def ball(rng):
    # a resting football: support = small contact patch under it
    return dict(kind="ball", parts=[sph((0, 0, 0.11), 0.11, "white_coat", col=True)],
                support=[(x, y) for x in (-0.05, 0.05) for y in (-0.05, 0.05)], mass=MASS["ball"], height=0.22,
                length=0.22, width=0.22, dynamic=True)


# ---- static street furniture ---------------------------------------------------------------------------------
def bus_shelter(z0):
    p = [box((0, 0, z0 + 2.45), (3.6, 1.6, 0.1), "dark_metal"),
         box((0, 0.75, z0 + 1.25), (3.5, 0.05, 2.3), "glass"),
         box((0, 0.45, z0 + 0.45), (2.6, 0.4, 0.06), "wood"),
         box((1.75, 0, z0 + 1.25), (0.05, 1.5, 2.3), "glass")]
    for x in (-1.75, 1.75):
        for y in (-0.75, 0.75):
            p.append(box((x, y, z0 + 1.2), (0.08, 0.08, 2.4), "dark_metal"))
    p.append(box((-2.2, -0.6, z0 + 1.4), (0.08, 0.08, 2.8), "dark_metal"))
    p.append(box((-2.2, -0.6, z0 + 2.65), (0.5, 0.05, 0.5), (0.1, 0.35, 0.7), col=False))
    return p


def traffic_light(z0, arm_dir):
    """Pole on the corner, arm over the road (arm_dir: +1/-1 along local x), three-lamp head."""
    p = [cyl((0, 0, z0 + 3.0), 0.09, 6.0, "dark_metal"),
         box((arm_dir * 2.6, 0, z0 + 5.8), (5.2, 0.12, 0.12), "dark_metal", col=False),
         box((arm_dir * 4.8, 0, z0 + 5.2), (0.35, 0.35, 1.0), "black", col=False)]
    for k, c in enumerate(((0.85, 0.05, 0.03), (0.9, 0.65, 0.05), (0.05, 0.75, 0.2))):
        p.append(sph((arm_dir * 4.8 + 0.19, 0, z0 + 5.5 - k * 0.3), 0.09, c))
    return p


# ---- special blocks ------------------------------------------------------------------------------------------
B = 19.5  # usable half-size of a block interior (the interior between sidewalks is 41.6 m wide)


def _bench(x, y, yaw_deg, z=SLAB_TOP):
    c, s_ = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    r = lambda a, b: (x + c * a - s_ * b, y + s_ * a + c * b)
    p = [box((*r(0, 0), z + 0.45), (1.8, 0.45, 0.06), "wood", rot=(0, 0, yaw_deg)),
         box((*r(0, -0.22), z + 0.75), (1.8, 0.06, 0.4), "wood", rot=(0, 0, yaw_deg))]
    for a in (-0.8, 0.8):
        p.append(box((*r(a, 0), z + 0.22), (0.08, 0.45, 0.44), "dark_metal", rot=(0, 0, yaw_deg)))
    return p


def _people(rng, n, region, prefix, child_frac=0.1, avoid=(), taken=None, maker=None):
    """region(rng) -> (x, y). Returns movables 1.2 m apart (also from everything already in `taken`, which is
    extended) and not inside the avoid boxes. maker(seed) overrides the person model (dogs, team players)."""
    out = []
    pts = taken if taken is not None else []
    tries = 0
    while len(out) < n and tries < n * 60:
        tries += 1
        x, y = region(rng)
        if any(math.hypot(x - a, y - b) < 1.2 for a, b in pts):
            continue
        if any(abs(x - ax) < hx + 0.5 and abs(y - ay) < hy + 0.5 for ax, ay, hx, hy in avoid):
            continue
        child = rng.random() < child_frac
        seed = int(rng.integers(1 << 30))
        mk = (lambda s=seed: maker(s)) if maker else (lambda child=child, s=seed: person(np.random.default_rng(s), child))
        out.append((f"{prefix}_{len(out)}", mk, x, y, float(rng.uniform(-math.pi, math.pi)), SLAB_TOP))
        pts.append((x, y))
    return out


def feature_park(rng):
    p, mov, obs = [], [], []
    # cross-shaped gravel paths + pond
    for (cx, cy, sx, sy) in ((0, 0, 2 * B, 2.6), (0, 0, 2.6, 2 * B)):
        p.append(box((cx, cy, SLAB_TOP + 0.01), (sx, sy, 0.02), "gravel", col=False))
    px, py = float(rng.choice([-1, 1])) * 9.5, float(rng.choice([-1, 1])) * 9.5
    rr = float(rng.uniform(4.0, 5.2))
    p.append(cyl((px, py, SLAB_TOP + 0.005), rr, 0.02, (0.03, 0.06, 0.06), col=False))  # water
    for k in range(20):
        a = 2 * math.pi * k / 20
        p.append(box((px + (rr + 0.25) * math.cos(a), py + (rr + 0.25) * math.sin(a), SLAB_TOP + 0.15), (1.7, 0.5, 0.3),
                     "concrete", rot=(0, 0, math.degrees(a) + 90)))
    for (bx, by, yaw) in ((5, 2.6, 0), (-5, -2.6, 180), (2.6, -5, 90), (-2.6, 5, -90)):
        p += _bench(bx, by, yaw)
    trees = []
    for _ in range(int(rng.integers(8, 15))):
        for _ in range(40):
            x, y = float(rng.uniform(-B + 2, B - 2)), float(rng.uniform(-B + 2, B - 2))
            if abs(x) < 3.5 or abs(y) < 3.5 or math.hypot(x - px, y - py) < rr + 3:
                continue
            if any(math.hypot(x - a, y - b) < 5 for a, b, _ in trees):
                continue
            trees.append((x, y, float(rng.uniform(0.22, 0.34))))
            break
    reg = lambda r: (float(r.uniform(-B + 1, B - 1)), float(r.choice([-1, 1])) * float(r.uniform(0, 1.0))) if r.random() < 0.5 else \
        (float(r.choice([-1, 1])) * float(r.uniform(0, 1.0)), float(r.uniform(-B + 1, B - 1)))
    taken = []
    benches = [(5, 2.6, 1.1, 0.5), (-5, -2.6, 1.1, 0.5), (2.6, -5, 0.5, 1.1), (-2.6, 5, 0.5, 1.1)]
    mov += _people(rng, int(rng.integers(6, 14)), reg, "walker", 0.15, avoid=benches, taken=taken)
    mov += _people(rng, int(rng.integers(1, 4)), lambda r: (float(r.uniform(-B + 2, B - 2)), float(r.choice([-1, 1])) * 1.6), "parkdog",
                   avoid=benches, taken=taken, maker=lambda s: dog(np.random.default_rng(s)))
    obs += [(x, y, 4.0, 4.0, 13.0) for x, y, _ in trees]
    return dict(kind="park", parts=p, movables=mov, obstacles=obs, slab=("aerial_grass_rock", "forest", 6.0),
                trees=trees, top_z=8.0, words="the park with the pond and the trees")


def feature_parking(rng):
    p, mov = [], []
    rows = (-11.0, 11.0)
    nb = 14
    bay_w = 2.6
    for ry in rows:
        for k in range(nb + 1):
            x = -nb * bay_w / 2 + k * bay_w
            p.append(box((x, ry, SLAB_TOP + 0.005), (0.1, 5.0, 0.01), "white_coat", col=False))
        p.append(box((0, ry - math.copysign(2.5, ry), SLAB_TOP + 0.005), (nb * bay_w, 0.1, 0.01), "white_coat", col=False))
    # middle aisle arrows
    for x in (-8, 8):
        p.append(box((x, 0, SLAB_TOP + 0.005), (3.0, 0.25, 0.01), "white_coat", col=False))
    for x in (-B + 0.6, B - 0.6):  # light poles
        for y in (-B + 0.6, B - 0.6, 0):
            p.append(cyl((x, y, SLAB_TOP + 4.0), 0.08, 8.0, "dark_metal"))
            p.append(box((x - math.copysign(0.4, x), y, SLAB_TOP + 8.0), (0.9, 0.3, 0.15), "dark_metal", col=False))
    fill = float(rng.uniform(0.55, 0.9))
    n = 0
    for ry in rows:
        for k in range(nb):
            if rng.random() > fill:
                continue
            x = -nb * bay_w / 2 + (k + 0.5) * bay_w
            yaw = math.pi / 2 if ry > 0 else -math.pi / 2
            if rng.random() < 0.3:
                yaw += math.pi
            mov.append((f"parked_{n}", (lambda s=int(rng.integers(1 << 30)): random_car(np.random.default_rng(s))),
                        x, ry + float(rng.uniform(-0.2, 0.2)), yaw + float(rng.uniform(-0.04, 0.04)), SLAB_TOP))
            n += 1
    mov += _people(rng, int(rng.integers(2, 5)), lambda r: (float(r.uniform(-15, 15)), float(r.uniform(-2.5, 2.5))), "shopper", 0.1)
    return dict(kind="parking lot", parts=p, movables=mov, obstacles=[], slab=("asphalt_02", "city", 8.0),
                top_z=2.0, words=f"the parking lot with about {n} parked cars", n_cars=n)


def feature_construction(rng):
    p, mov, obs = [], [], []
    conc = (0.40, 0.40, 0.38)
    yellow = (0.95, 0.70, 0.08)
    # perimeter fence panels with a gate gap on the south side
    for k in range(-9, 10):
        for (x, y, rot) in ((k * 2.05, -B, 0), (k * 2.05, B, 0), (-B, k * 2.05, 90), (B, k * 2.05, 90)):
            if y == -B and abs(x) < 4:
                continue
            p.append(box((x, y, SLAB_TOP + 1.0), (2.0, 0.05, 2.0), (0.55, 0.60, 0.55), rot=(0, 0, rot), col=False))
            p.append(box((x, y, SLAB_TOP + 0.08), (0.5, 0.3, 0.16), conc, rot=(0, 0, rot)))
    # concrete frame under construction: 4 x 3 columns, floors up to n_fl
    fx0, fy0 = float(rng.uniform(-4, 1)), float(rng.uniform(1, 4))
    n_fl = int(rng.integers(2, 5))
    fh = 3.4
    for i in range(4):
        for j in range(3):
            x, y = fx0 - 6 + i * 4, fy0 - 4 + j * 4
            p.append(box((x, y, SLAB_TOP + n_fl * fh / 2), (0.4, 0.4, n_fl * fh), conc))
            for k in range(1, 3):  # rebar sticking out of the top
                p.append(cyl((x + 0.12 * (k - 1.5), y, SLAB_TOP + n_fl * fh + 0.5), 0.015, 1.0, "dark_metal", col=False))
    for f in range(1, n_fl):
        p.append(box((fx0, fy0, SLAB_TOP + f * fh), (12.8, 8.8, 0.25), conc))
    p.append(box((fx0, fy0, SLAB_TOP + 0.15), (13.0, 9.0, 0.3), conc))
    obs.append((fx0, fy0, 6.6, 4.6, SLAB_TOP + n_fl * fh + 1.0))
    # scaffolding along one side
    for k in range(7):
        p.append(cyl((fx0 - 6 + k * 2, fy0 - 5.2, SLAB_TOP + n_fl * fh / 2), 0.03, n_fl * fh, "metal", col=False))
    for f in range(1, n_fl + 1):
        p.append(box((fx0, fy0 - 5.2, SLAB_TOP + f * fh - 0.1), (12.4, 0.6, 0.05), "wood", col=False))
    # tower crane
    cx, cy = float(rng.uniform(-14, -9)), float(rng.uniform(-14, -8))
    ch = float(rng.uniform(26, 32))
    p.append(box((cx, cy, SLAB_TOP + 0.5), (4, 4, 1.0), conc))
    p.append(box((cx, cy, SLAB_TOP + ch / 2), (1.6, 1.6, ch), yellow))
    for z in np.arange(2.0, ch, 3.0):  # lattice impression: dark cross bands
        p.append(box((cx, cy, SLAB_TOP + z), (1.65, 1.65, 0.15), "dark_metal", col=False))
    jyaw = float(rng.uniform(0, 360))
    ja = math.radians(jyaw)
    jl, cl = 24.0, 8.0
    jx, jy = cx + math.cos(ja) * (jl / 2 - 1), cy + math.sin(ja) * (jl / 2 - 1)
    p.append(box((jx, jy, SLAB_TOP + ch + 0.6), (jl, 1.2, 1.2), yellow, rot=(0, 0, jyaw)))
    kx, ky = cx - math.cos(ja) * (cl / 2 + 0.5), cy - math.sin(ja) * (cl / 2 + 0.5)
    p.append(box((kx, ky, SLAB_TOP + ch + 0.6), (cl, 1.2, 1.0), yellow, rot=(0, 0, jyaw)))
    p.append(box((cx - math.cos(ja) * (cl - 0.5), cy - math.sin(ja) * (cl - 0.5), SLAB_TOP + ch - 0.6), (2.2, 2.0, 2.4), conc, rot=(0, 0, jyaw)))
    p.append(box((cx + math.cos(ja) * 1.4, cy + math.sin(ja) * 1.4, SLAB_TOP + ch - 0.5), (1.6, 1.4, 1.6), "glass", rot=(0, 0, jyaw)))
    p.append(box((cx, cy, SLAB_TOP + ch + 3.0), (1.0, 1.0, 4.0), yellow))  # top mast
    hx, hy = cx + math.cos(ja) * 15, cy + math.sin(ja) * 15
    hook_len = float(rng.uniform(8, 18))
    p.append(cyl((hx, hy, SLAB_TOP + ch - hook_len / 2), 0.03, hook_len, "dark_metal", col=False))
    p.append(box((hx, hy, SLAB_TOP + ch - hook_len - 0.3), (0.5, 0.5, 0.6), (0.9, 0.2, 0.1), col=False))
    obs.append((cx, cy, 2.5, 2.5, SLAB_TOP + ch + 5.5))
    obs.append((jx, jy, jl / 2 + 1, jl / 2 + 1, SLAB_TOP + ch + 1.5))
    # site office containers, material piles
    for k in range(int(rng.integers(1, 3))):
        p.append(box((13.5, 14.5 - k * 2.8, SLAB_TOP + 1.3), (6.0, 2.4, 2.6), [(0.2, 0.45, 0.7), (0.85, 0.85, 0.82)][k % 2]))
    for k in range(int(rng.integers(3, 7))):
        x, y = -15.0 + (k % 3) * 3.2, 13.0 + (k // 3) * 2.2
        p.append(box((x, y, SLAB_TOP + 0.25), (2.4, 1.2, 0.5), "wood"))
    for k in range(6):
        p.append(cyl((14 + 0.0, -4 + k * 0.45, SLAB_TOP + 0.22), 0.2, 4.0, (0.3, 0.3, 0.32), axis="X"))
    mov.append(("excavator", (lambda s=int(rng.integers(1 << 30)): excavator(np.random.default_rng(s))),
                6.5, -10.5, float(rng.uniform(-math.pi, math.pi)), SLAB_TOP))
    mov.append(("dump_truck", (lambda s=int(rng.integers(1 << 30)): dump_truck(np.random.default_rng(s))),
                -1.0, -16.5, 0.0, SLAB_TOP))
    for k in range(int(rng.integers(4, 9))):
        mov.append((f"cone_{k}", (lambda s=int(rng.integers(1 << 30)): traffic_cone(np.random.default_rng(s))),
                    -4 + k * 1.1, -B + 1.2, 0.0, SLAB_TOP))
    mov += _people(rng, int(rng.integers(3, 7)), lambda r: (float(r.uniform(-B + 2, B - 2)), float(r.uniform(-12, -4))), "worker", 0.0,
                   avoid=[(fx0, fy0 - 0.6, 6.6, 5.2), (6.5, -10.5, 3.6, 3.6), (-1.0, -16.5, 3.8, 1.6), (cx, cy, 2.3, 2.3), (14, -3, 2.4, 1.6)])
    return dict(kind="construction site", parts=p, movables=mov, obstacles=obs, slab=("forrest_ground_03", "forest", 5.0),
                top_z=ch, words=f"the construction site with the yellow tower crane and the {n_fl}-storey concrete frame",
                crane=(cx, cy, ch))


def feature_playground(rng):
    p, mov = [], []
    surf = [(0.40, 0.10, 0.08), (0.08, 0.20, 0.40), (0.12, 0.32, 0.14)][int(rng.integers(3))]
    p.append(box((0, 0, SLAB_TOP + 0.01), (26, 26, 0.03), surf, col=False))
    p.append(box((-8, -8, SLAB_TOP + 0.12), (7, 7, 0.24), (0.85, 0.75, 0.50)))  # sandbox
    # swing set
    for x in (4, 12):
        for y in (-0.9, 0.9):
            p.append(cyl((x, -9 + y, SLAB_TOP + 1.25), 0.06, 2.6, (0.8, 0.2, 0.1), rot=(math.copysign(12, y), 0, 0)))
    p.append(cyl((8, -9, SLAB_TOP + 2.5), 0.06, 8.2, (0.8, 0.2, 0.1), axis="X"))
    for x in (6, 10):
        p.append(box((x, -9, SLAB_TOP + 0.5), (0.5, 0.2, 0.04), "black", col=False))
        for dx in (-0.22, 0.22):
            p.append(cyl((x + dx, -9, SLAB_TOP + 1.5), 0.01, 2.0, "metal", col=False))
    # slide + platform
    p.append(box((6, 7, SLAB_TOP + 1.5), (2.0, 2.0, 0.12), (0.2, 0.5, 0.8)))
    for dx in (-0.9, 0.9):
        for dy in (-0.9, 0.9):
            p.append(cyl((6 + dx, 7 + dy, SLAB_TOP + 0.75), 0.06, 1.5, (0.2, 0.5, 0.8)))
    p.append(box((9.0, 7, SLAB_TOP + 0.85), (4.2, 0.8, 0.06), (0.95, 0.75, 0.1), rot=(0, 21, 0)))
    p.append(box((4.6, 7, SLAB_TOP + 0.85), (0.1, 0.6, 1.8), (0.8, 0.2, 0.1), rot=(0, -15, 0)))
    # climbing frame
    for i in range(4):
        for j in range(4):
            p.append(cyl((-9 + i * 1.2, 7 + j * 1.2, SLAB_TOP + 1.0), 0.04, 2.0, (0.25, 0.6, 0.3), col=False))
    p.append(box((-7.2, 8.8, SLAB_TOP + 2.0), (3.8, 3.8, 0.06), (0.25, 0.6, 0.3)))
    # seesaw, benches
    p.append(box((0, 0, SLAB_TOP + 0.3), (0.3, 0.3, 0.6), (0.4, 0.4, 0.4)))
    p.append(box((0, 0, SLAB_TOP + 0.62), (3.6, 0.3, 0.08), (0.9, 0.5, 0.1)))
    for (bx, by, yaw) in ((-14, 0, 90), (14, 0, -90), (0, 14, 180)):
        p += _bench(bx, by, yaw)
    mov += _people(rng, int(rng.integers(6, 12)), lambda r: (float(r.uniform(-12, 12)), float(r.uniform(-12, 12))), "kid", 0.7,
                   avoid=[(-8, -8, 3.5, 3.5), (8, -9, 4.5, 1.9), (7.5, 7, 5.6, 1.3), (-7.2, 8.8, 2.2, 2.2), (0, 0, 2, 0.4),
                          (-14, 0, 0.5, 1.1), (14, 0, 0.5, 1.1), (0, 14, 1.1, 0.5)])
    return dict(kind="playground", parts=p, movables=mov, obstacles=[], slab=("cobblestone_floor_01", "city", 2.5),
                top_z=2.5, words="the children's playground with the swings and the slide")


def _court_lines(p, L, W, cx, cy, col):
    for (x, y, sx, sy) in ((cx, cy - W / 2, L, 0.1), (cx, cy + W / 2, L, 0.1), (cx - L / 2, cy, 0.1, W), (cx + L / 2, cy, 0.1, W),
                           (cx, cy, 0.1, W)):
        p.append(box((x, y, SLAB_TOP + 0.022), (sx, sy, 0.01), col, col=False))


def feature_football(rng):
    p, mov = [], []
    L, W = 34.0, 22.0
    p.append(box((0, 0, SLAB_TOP + 0.01), (L + 3, W + 3, 0.02), (0.06, 0.22, 0.06), col=False))
    for k in range(-3, 4):  # mowing stripes
        p.append(box((k * L / 7, 0, SLAB_TOP + 0.015), (L / 14, W + 3, 0.01), (0.08, 0.27, 0.08), col=False))
    _court_lines(p, L, W, 0, 0, "white_coat")
    for k in range(24):
        a = 2 * math.pi * k / 24
        p.append(box((3.0 * math.cos(a), 3.0 * math.sin(a), SLAB_TOP + 0.022), (0.85, 0.1, 0.01), "white_coat",
                     rot=(0, 0, math.degrees(a) + 90), col=False))
    net = (0.82, 0.82, 0.80)
    for s_ in (-1, 1):
        x = s_ * L / 2
        _court_lines(p, 5.0, 10.0, x - s_ * 2.5, 0, "white_coat")
        # a real 5 x 2 m goal: front posts + crossbar, back posts + ground bars + top bars, net on every side,
        # all joined (the first version had a net panel standing 0.8 m behind the goal on its own)
        d = 1.2  # goal depth
        xb = x + s_ * d
        for y in (-2.5, 2.5):
            p.append(cyl((x, y, SLAB_TOP + 1.0), 0.06, 2.0, "white_coat"))                        # front post
            p.append(cyl((xb, y, SLAB_TOP + 0.75), 0.03, 1.5, "metal", col=False))                 # back post
            p.append(cyl((x + s_ * d / 2, y, SLAB_TOP + 0.03), 0.03, d, "metal", axis="X", col=False))  # ground bar
            ang = math.degrees(math.atan2(0.5, d))
            p.append(cyl((x + s_ * d / 2, y, SLAB_TOP + 1.75), 0.03, math.hypot(d, 0.5), "metal", axis="X",
                         rot=(0, s_ * ang, 0), col=False))                                       # top side bar
            p.append(box((x + s_ * d / 2, y, SLAB_TOP + 0.9), (d, 0.01, 1.7), net, col=False))    # side net
        p.append(cyl((x, 0, SLAB_TOP + 2.0), 0.06, 5.12, "white_coat", axis="Y"))                 # crossbar
        p.append(cyl((xb, 0, SLAB_TOP + 1.5), 0.03, 5.0, "metal", axis="Y", col=False))           # back top bar
        p.append(cyl((xb, 0, SLAB_TOP + 0.03), 0.03, 5.0, "metal", axis="Y", col=False))          # back ground bar
        p.append(box((xb, 0, SLAB_TOP + 0.78), (0.01, 5.0, 1.5), net, col=False))                # back net
        p.append(box((x + s_ * d / 2, 0, SLAB_TOP + 1.78), (math.hypot(d, 0.5), 5.0, 0.01), net,
                     rot=(0, s_ * math.degrees(math.atan2(0.5, d)), 0), col=False))               # roof net
    teams = [SHIRTS[int(rng.integers(len(SHIRTS)))], SHIRTS[int(rng.integers(len(SHIRTS)))]]
    n = int(rng.integers(6, 11))
    taken = []
    for side, shirt in ((1, teams[0]), (-1, teams[1])):
        mov += _people(rng, (n + (1 if side > 0 else 0)) // 2,
                       lambda r, side=side: (side * float(r.uniform(2, L / 2 - 2)), float(r.uniform(-W / 2 + 1, W / 2 - 1))),
                       f"player_{'a' if side > 0 else 'b'}", taken=taken,
                       maker=lambda s, shirt=shirt: person(np.random.default_rng(s), False, "walk", shirt=shirt))
    mov.append(("ball", (lambda s=int(rng.integers(1 << 30)): ball(np.random.default_rng(s))),
                float(rng.uniform(-5, 5)), float(rng.uniform(-4, 4)), 0.0, SLAB_TOP + 0.02))
    return dict(kind="football pitch", parts=p, movables=mov, obstacles=[], slab=("concrete_wall_008", "city", 6.0),
                top_z=2.5, words=f"the small football pitch where {n} players are on the field", n_players=n)


def feature_courts(rng):
    p, mov = [], []
    col = [(0.07, 0.17, 0.36), (0.09, 0.25, 0.13), (0.36, 0.13, 0.08)][int(rng.integers(3))]
    for cy in (-9.5, 9.5):
        p.append(box((0, cy, SLAB_TOP + 0.01), (30, 16, 0.02), col, col=False))
        _court_lines(p, 26, 14, 0, cy, "white_coat")
    # basketball (south): hoops at both ends
    for s_ in (-1, 1):
        x = s_ * 13.6
        p.append(cyl((x + s_ * 0.9, -9.5, SLAB_TOP + 1.75), 0.08, 3.5, "dark_metal"))
        p.append(box((x + s_ * 0.45, -9.5, SLAB_TOP + 3.45), (0.9, 0.1, 0.1), "dark_metal", col=False))  # arm to the board
        p.append(box((x, -9.5, SLAB_TOP + 3.3), (0.06, 1.8, 1.05), "white_coat"))
        for k in range(8):
            a = 2 * math.pi * k / 8
            p.append(box((x - s_ * 0.45 + 0.23 * math.cos(a), -9.5 + 0.23 * math.sin(a), SLAB_TOP + 3.05), (0.18, 0.03, 0.03),
                         (0.9, 0.35, 0.05), rot=(0, 0, math.degrees(a) + 90), col=False))
    # tennis (north): net
    p.append(box((0, 9.5, SLAB_TOP + 0.5), (0.04, 13.0, 0.9), (0.9, 0.9, 0.9), col=False))
    for y in (3.0, 16.0):
        p.append(cyl((0, y, SLAB_TOP + 0.55), 0.05, 1.1, "dark_metal"))
    # fence around both courts
    for k in range(-7, 8):
        for (x, y, rot) in ((k * 2.0, -B + 1, 0), (k * 2.0, B - 1, 0), (-15.5, k * 2.6, 90), (15.5, k * 2.6, 90)):
            p.append(box((x, y, SLAB_TOP + 1.5), (2.0, 0.03, 3.0), (0.25, 0.3, 0.28), rot=(0, 0, rot), col=False))
    mov += _people(rng, int(rng.integers(4, 9)), lambda r: (float(r.uniform(-12, 12)), float(r.choice([-9.5, 9.5]) + r.uniform(-6, 6))),
                   "athlete", 0.0)
    return dict(kind="sports courts", parts=p, movables=mov, obstacles=[], slab=("concrete_wall_008", "city", 6.0),
                top_z=3.5, words="the fenced basketball and tennis courts")


def feature_market(rng):
    p, mov = [], []
    canopies = [(0.50, 0.07, 0.05), (0.07, 0.18, 0.42), (0.70, 0.52, 0.06), (0.08, 0.32, 0.14), (0.80, 0.80, 0.76)]
    cols = [canopies[int(rng.integers(len(canopies)))] for _ in range(2)]
    stalls = []
    for row, y in enumerate((-8.0, 0.0, 8.0)):
        for k in range(-3, 4):
            if rng.random() < 0.15:
                continue
            x = k * 4.6
            c = cols[(k + row) % 2]
            p.append(box((x, y, SLAB_TOP + 2.45), (3.4, 2.6, 0.08), c))
            for dx in (-1.6, 1.6):
                for dy in (-1.2, 1.2):
                    p.append(cyl((x + dx, y + dy, SLAB_TOP + 1.2), 0.04, 2.4, "metal", col=False))
            p.append(box((x, y - 0.6, SLAB_TOP + 0.45), (3.0, 1.0, 0.9), "wood"))
            for j in range(6):  # produce crates
                p.append(box((x - 1.2 + j * 0.48, y - 0.6, SLAB_TOP + 0.98), (0.4, 0.55, 0.16),
                             [(0.85, 0.2, 0.1), (0.95, 0.6, 0.1), (0.3, 0.6, 0.2), (0.7, 0.5, 0.3)][int(rng.integers(4))], col=False))
            stalls.append((x, y))
    avoid = [(x, y, 1.9, 1.5) for x, y in stalls]
    mov += _people(rng, int(rng.integers(14, 26)), lambda r: (float(r.uniform(-16, 16)), float(r.choice([-4.0, 4.0, 12.5]) + r.uniform(-1.2, 1.2))),
                   "shopper", 0.12, avoid=avoid)
    mov.append(("food_truck", (lambda s=int(rng.integers(1 << 30)): special_vehicle("food truck", np.random.default_rng(s))),
                float(rng.uniform(-8, 8)), -15.0, 0.0, SLAB_TOP))
    return dict(kind="market", parts=p, movables=mov, obstacles=[], slab=("cobblestone_floor_01", "city", 2.5),
                top_z=3.0, words=f"the open-air market with {len(stalls)} stalls", n_stalls=len(stalls))


def feature_fountain(rng):
    p, mov = [], []
    stone = (0.52, 0.50, 0.46)
    water = (0.05, 0.10, 0.12)
    R = float(rng.uniform(5.0, 6.5))
    for k in range(28):
        a = 2 * math.pi * k / 28
        p.append(box((R * math.cos(a), R * math.sin(a), SLAB_TOP + 0.35), (2 * math.pi * R / 28 + 0.05, 0.5, 0.7), stone,
                     rot=(0, 0, math.degrees(a) + 90)))
    p.append(cyl((0, 0, SLAB_TOP + 0.3), R - 0.2, 0.5, water, col=False))
    p.append(cyl((0, 0, SLAB_TOP + 1.2), 0.6, 2.4, stone))
    p.append(cyl((0, 0, SLAB_TOP + 2.45), 2.0, 0.25, stone))
    p.append(cyl((0, 0, SLAB_TOP + 2.6), 1.8, 0.08, water, col=False))
    p.append(cyl((0, 0, SLAB_TOP + 3.3), 0.3, 1.4, stone))
    p.append(cyl((0, 0, SLAB_TOP + 4.05), 0.9, 0.18, stone))
    statue = rng.random() < 0.6
    if statue:  # bronze statue on a pedestal at one end of the square
        sx = float(rng.choice([-1, 1])) * 13.0
        p.append(box((sx, 0, SLAB_TOP + 1.0), (2.0, 2.0, 2.0), stone))
        p.append(box((sx, 0, SLAB_TOP + 2.95), (0.8, 0.5, 1.9), (0.30, 0.22, 0.12)))
        p.append(sph((sx, 0, SLAB_TOP + 4.1), 0.32, (0.30, 0.22, 0.12)))
    for k in range(8):
        a = 2 * math.pi * k / 8 + math.pi / 8
        p += _bench(10 * math.cos(a), 10 * math.sin(a), math.degrees(a) - 90)
    for (x, y) in ((-B + 2, -B + 2), (B - 2, -B + 2), (-B + 2, B - 2), (B - 2, B - 2)):
        p.append(box((x, y, SLAB_TOP + 0.4), (2.0, 2.0, 0.8), "concrete"))
        p.append(sph((x, y, SLAB_TOP + 1.3), 0.9, (0.18, 0.40, 0.15)))
    taken = []
    avoid_f = ([(10 * math.cos(2 * math.pi * k / 8 + math.pi / 8), 10 * math.sin(2 * math.pi * k / 8 + math.pi / 8), 1.2, 1.2) for k in range(8)]
               + ([(sx, 0, 1.4, 1.4)] if statue else []) + [(x, y, 1.2, 1.2) for x in (-B + 2, B - 2) for y in (-B + 2, B - 2)])
    ring = lambda r: (lambda a, d: (d * math.cos(a), d * math.sin(a)))(r.uniform(0, 2 * math.pi), r.uniform(R + 1.4, 17))
    mov += _people(rng, int(rng.integers(10, 20)), ring, "visitor", 0.15, avoid=avoid_f, taken=taken)
    mov += _people(rng, int(rng.integers(0, 3)), ring, "plazadog", avoid=avoid_f, taken=taken, maker=lambda s: dog(np.random.default_rng(s)))
    return dict(kind="fountain square", parts=p, movables=mov, obstacles=[], slab=("cobblestone_floor_01", "city", 2.5),
                top_z=4.5, words="the square with the round fountain" + (" and the bronze statue" if statue else ""))


def feature_gas_station(rng):
    p, mov = [], []
    brand = [(0.85, 0.10, 0.08), (0.10, 0.45, 0.20), (0.10, 0.30, 0.70), (0.95, 0.65, 0.05)][int(rng.integers(4))]
    p.append(box((-2, -2, SLAB_TOP + 5.2), (18, 10, 0.6), (0.92, 0.92, 0.90)))  # canopy
    p.append(box((-2, -2, SLAB_TOP + 5.2), (18.1, 10.1, 0.3), brand, col=False))
    for x in (-8, 4):
        for y in (-5, 1):
            p.append(box((x, y, SLAB_TOP + 2.5), (0.5, 0.5, 5.0), (0.85, 0.85, 0.85)))
    pumps = []
    for x in (-6, -2, 2):
        for y in (-2,):
            p.append(box((x, y, SLAB_TOP + 0.2), (1.2, 4.0, 0.4), "concrete"))
            p.append(box((x, y, SLAB_TOP + 1.2), (0.6, 0.9, 1.7), brand))
            pumps.append((x, y))
    p.append(box((10, 10, SLAB_TOP + 2.0), (12, 8, 4.0), (0.88, 0.86, 0.80)))  # shop
    p.append(box((10, 5.95, SLAB_TOP + 1.5), (10, 0.1, 2.4), "glass", col=False))
    p.append(box((10, 5.9, SLAB_TOP + 3.5), (12.1, 0.1, 0.7), brand, col=False))
    p.append(box((10, 10, SLAB_TOP + 4.1), (12.3, 8.3, 0.25), "dark_metal"))
    p.append(cyl((-15, 15, SLAB_TOP + 4.5), 0.18, 9.0, "dark_metal"))  # price sign
    p.append(box((-15, 15, SLAB_TOP + 8.0), (0.3, 2.4, 3.0), brand))
    for k, (x, y) in enumerate(pumps):
        if rng.random() < 0.75:
            mov.append((f"fueling_{k}", (lambda s=int(rng.integers(1 << 30)): random_car(np.random.default_rng(s))),
                        x + 1.9, y, math.pi / 2 if rng.random() < 0.5 else -math.pi / 2, SLAB_TOP))
    for k in range(int(rng.integers(1, 4))):
        mov.append((f"shopcar_{k}", (lambda s=int(rng.integers(1 << 30)): random_car(np.random.default_rng(s))),
                    1.0 + k * 6.0, 16.6, 0.0, SLAB_TOP))
    mov += _people(rng, int(rng.integers(1, 4)), lambda r: (float(r.uniform(5, 15)), float(r.uniform(4.0, 5.0))), "customer", 0.0)
    return dict(kind="gas station", parts=p, movables=mov, obstacles=[], slab=("concrete_wall_008", "city", 6.0),
                top_z=6.0, words="the gas station with the coloured canopy over the pumps")


def feature_fire_station(rng):
    p, mov = [], []
    brick = (0.32, 0.09, 0.06)
    p.append(box((0, 8, SLAB_TOP + 4.5), (24, 14, 9.0), brick))
    p.append(box((0, 8, SLAB_TOP + 9.1), (24.4, 14.4, 0.3), "dark_metal"))
    for k, x in enumerate((-7.5, 0, 7.5)):
        p.append(box((x, 0.95, SLAB_TOP + 2.3), (5.0, 0.1, 4.6), (0.75, 0.75, 0.75), col=False))
        for z in np.arange(0.5, 4.6, 0.55):
            p.append(box((x, 0.9, SLAB_TOP + z), (5.0, 0.02, 0.04), (0.55, 0.55, 0.55), col=False))
    p.append(box((0, 0.9, SLAB_TOP + 6.5), (12, 0.1, 1.2), (0.92, 0.92, 0.9), col=False))  # sign band
    p.append(box((10, 14, SLAB_TOP + 7.0), (3.0, 3.0, 14.0), brick))  # hose tower
    mov.append(("fire_truck_1", (lambda s=int(rng.integers(1 << 30)): special_vehicle("fire truck", np.random.default_rng(s))),
                -7.5, -6.5, -math.pi / 2, SLAB_TOP))
    if rng.random() < 0.7:
        mov.append(("fire_truck_2", (lambda s=int(rng.integers(1 << 30)): special_vehicle("fire truck", np.random.default_rng(s))),
                    0.0, -6.5, -math.pi / 2, SLAB_TOP))
    if rng.random() < 0.6:
        mov.append(("ambulance", (lambda s=int(rng.integers(1 << 30)): special_vehicle("ambulance", np.random.default_rng(s))),
                    7.5, -5.0, -math.pi / 2, SLAB_TOP))
    mov += _people(rng, int(rng.integers(2, 6)), lambda r: (float(r.uniform(-14, 14)), float(r.uniform(-15, -12))), "firefighter", 0.0)
    return dict(kind="fire station", parts=p, movables=mov, obstacles=[(10, 14, 1.6, 1.6, SLAB_TOP + 14.0), (0, 8, 12.2, 7.2, 9.5)],
                slab=("concrete_wall_008", "city", 6.0), top_z=14.0, words="the red-brick fire station with the fire engines in front")


FEATURES = {"park": feature_park, "parking lot": feature_parking, "construction site": feature_construction,
            "playground": feature_playground, "football pitch": feature_football, "sports courts": feature_courts,
            "market": feature_market, "fountain square": feature_fountain, "gas station": feature_gas_station,
            "fire station": feature_fire_station}
