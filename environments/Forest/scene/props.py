"""
Simple, physically sound props for the Forest Easy dataset (2026-10-11): farmhouses, garages, barns,
silos, sheds, fenced paddocks, hay bales, water troughs, cars, tractors, cows, horses, sheep, dogs.

User brief: "add realistic new objects ... the objects can not be so hard use simple objects but with
real physic laws". So every prop is built from a handful of primitives (boxes, cylinders, capsules,
spheres, one prism) at real-world dimensions and masses, and is placed the way physics would leave it:

  * Movable things (animals, vehicles, bales, troughs) are rigid bodies. Their model has its support
    points (hooves, tyre contact points, base corners) at local z = 0; they are set down on the plane
    fitted through the ground under those points, so every support touches the ground and the body
    tilts with the slope exactly as it would at rest. check_static() verifies support contact and that
    the centre of mass lies inside the support polygon (it would tip over otherwise). In the flight job
    they are simulated as PhysX rigid bodies with gravity for the whole flight and must not move
    (see make_dynamic / settle report in run_x500_mission.py).
  * Buildings and fences are static and stand on foundations / posts sunk into the ground below the
    lowest point of their footprint, with the floor above the highest -- nothing hovers, nothing is
    buried.

Geometry convention for a prop: local x = forward/length, y = left, z = up. place() turns a prop into
USD prims under /World/Props.
"""
import math

import numpy as np

# ---- palette -------------------------------------------------------------------------------------------
COLORS = {
    "concrete": ((0.52, 0.51, 0.49), 0.9), "glass": ((0.04, 0.05, 0.06), 0.08), "trim": ((0.88, 0.87, 0.84), 0.6),
    "door_wood": ((0.30, 0.17, 0.08), 0.7), "dark_wood": ((0.20, 0.12, 0.06), 0.8), "wood": ((0.42, 0.27, 0.14), 0.8),
    "gravel": ((0.47, 0.44, 0.40), 0.95), "metal": ((0.55, 0.57, 0.58), 0.45), "dark_metal": ((0.15, 0.15, 0.16), 0.5),
    "tyre": ((0.03, 0.03, 0.03), 0.9), "rim": ((0.62, 0.62, 0.63), 0.35), "hay": ((0.74, 0.62, 0.32), 0.95),
    "trough_water": ((0.04, 0.06, 0.05), 0.15), "hoof": ((0.08, 0.07, 0.06), 0.8), "nose_pink": ((0.72, 0.50, 0.48), 0.7),
    "black": ((0.035, 0.033, 0.03), 0.75), "white_coat": ((0.86, 0.85, 0.82), 0.85), "wool": ((0.84, 0.81, 0.74), 0.98),
    "brick": ((0.47, 0.21, 0.14), 0.9), "barn_red": ((0.30, 0.045, 0.03), 0.85), "light": ((0.95, 0.93, 0.80), 0.3),
    "tail_light": ((0.6, 0.04, 0.03), 0.4),
}
WALLS = {"white": (0.86, 0.85, 0.80), "cream": (0.85, 0.77, 0.58), "pale blue": (0.56, 0.66, 0.74),
         "sage green": (0.55, 0.62, 0.50), "light grey": (0.70, 0.70, 0.68), "yellow": (0.86, 0.73, 0.36),
         "red brick": COLORS["brick"][0], "log": (0.40, 0.25, 0.13)}
ROOFS = {"dark grey": (0.07, 0.07, 0.075), "red tile": (0.30, 0.08, 0.05), "green metal": (0.08, 0.17, 0.10),
         "brown": (0.16, 0.09, 0.05), "blue-grey": (0.12, 0.15, 0.19)}
CAR_COLORS = {"white": (0.85, 0.85, 0.84), "black": (0.04, 0.04, 0.045), "silver": (0.6, 0.61, 0.63), "red": (0.55, 0.05, 0.04),
              "blue": (0.07, 0.17, 0.42), "dark green": (0.08, 0.2, 0.12), "grey": (0.33, 0.34, 0.35)}
COW_COATS = {"black-and-white": ("white_coat", "black"), "brown": ((0.36, 0.20, 0.10), None), "black": ("black", None),
             "light brown": ((0.58, 0.40, 0.23), None), "red-and-white": ("white_coat", (0.45, 0.14, 0.06))}
HORSE_COATS = {"bay": (0.33, 0.17, 0.08), "chestnut": (0.50, 0.24, 0.10), "black": (0.04, 0.035, 0.03),
               "grey": (0.72, 0.71, 0.68), "palomino": (0.80, 0.63, 0.36), "dark brown": (0.20, 0.11, 0.05)}
DOG_COATS = {"golden": (0.76, 0.55, 0.30), "black": (0.04, 0.04, 0.04), "brown": (0.36, 0.21, 0.10), "white": (0.85, 0.84, 0.80)}

MASS_KG = {"cow": 620.0, "horse": 500.0, "sheep": 70.0, "dog": 30.0, "car": 1350.0, "tractor": 3600.0,
           "round_bale": 300.0, "square_bale": 25.0, "trough": 140.0}


# ---- part constructors (all in the prop's local frame) -------------------------------------------------
def box(c, s, mat, rot=(0, 0, 0), col=True):
    return dict(shape="box", c=tuple(c), s=tuple(s), rot=tuple(rot), mat=mat, col=col)


def cyl(c, r, h, mat, axis="Z", rot=(0, 0, 0), col=True):
    return dict(shape="cyl", c=tuple(c), r=r, h=h, axis=axis, rot=tuple(rot), mat=mat, col=col)


def cap(c, r, h, mat, axis="X", rot=(0, 0, 0), col=True):
    return dict(shape="cap", c=tuple(c), r=r, h=h, axis=axis, rot=tuple(rot), mat=mat, col=col)


def sph(c, r, mat, scale=(1, 1, 1), rot=(0, 0, 0), col=False):
    return dict(shape="sph", c=tuple(c), r=r, scale=tuple(scale), rot=tuple(rot), mat=mat, col=col)


def prism(x0, x1, half_w, z0, rise, mat):
    """Gable infill: triangular prism along x, base 2*half_w wide at z0, apex rise above."""
    return dict(shape="prism", x0=x0, x1=x1, hw=half_w, z0=z0, rise=rise, mat=mat, col=True)


def _volume(p):
    if p["shape"] == "box":
        return p["s"][0] * p["s"][1] * p["s"][2]
    if p["shape"] == "cyl":
        return math.pi * p["r"] ** 2 * p["h"]
    if p["shape"] == "cap":
        return math.pi * p["r"] ** 2 * p["h"] + 4 / 3 * math.pi * p["r"] ** 3
    if p["shape"] == "sph":
        return 4 / 3 * math.pi * p["r"] ** 3 * p["scale"][0] * p["scale"][1] * p["scale"][2]
    return 0.0


# ---- movable props: support points at z = 0 ------------------------------------------------------------
def legs(xs, ys, top, r, mat, hoof="hoof"):
    out = []
    for x in xs:
        for y in ys:
            out.append(cyl((x, y, top / 2.0 + 0.03), r, top - 0.06, mat))
            out.append(cyl((x, y, 0.035), r * 1.12, 0.07, hoof))
    return out


def cow(coat, grazing, rng):
    base, patch = COW_COATS[coat]
    body_z, lx, ly = 1.02, 0.62, 0.22
    p = legs((-lx, lx), (-ly, ly), body_z - 0.1, 0.075, base)
    p.append(cap((0, 0, body_z), 0.38, 1.25, base))  # barrel
    if patch is not None:  # coat patches: flattened ellipsoids just proud of the barrel
        for k in range(int(rng.integers(3, 6))):
            a = rng.uniform(0.3, 2.8) * (1 if k % 2 else -1)
            px = rng.uniform(-0.6, 0.6)
            p.append(sph((px, 0.33 * math.sin(a), body_z + 0.33 * math.cos(a)), 0.2, patch, scale=(1.6, 0.9, 0.9)))
    if grazing:  # neck down to the grass
        p.append(cap((0.95, 0, 0.75), 0.17, 0.55, base, axis="X", rot=(0, 55, 0)))
        hx, hz, hp = 1.18, 0.30, 70
    else:
        p.append(cap((0.95, 0, 1.15), 0.18, 0.45, base, axis="X", rot=(0, -25, 0)))
        hx, hz, hp = 1.30, 1.18, 30
    p.append(box((hx, 0, hz), (0.48, 0.26, 0.28), base, rot=(0, hp, 0)))  # head
    nx, nz = hx + 0.26 * math.cos(math.radians(hp)), hz - 0.26 * math.sin(math.radians(hp))
    p.append(box((nx, 0, nz), (0.12, 0.22, 0.2), "nose_pink", rot=(0, hp, 0), col=False))
    for s in (-1, 1):
        p.append(box((hx - 0.12, s * 0.2, hz + 0.1), (0.08, 0.16, 0.05), base, col=False))  # ears
    p.append(cyl((-0.92, 0, 0.78), 0.03, 0.7, base, col=False))  # tail
    p.append(sph((-0.45, 0, 0.62), 0.14, "nose_pink"))  # udder
    return dict(kind="cow", parts=p, support=[(x, y) for x in (-lx, lx) for y in (-ly, ly)], mass=MASS_KG["cow"],
                height=1.45 if not grazing else 1.40, length=2.6, dynamic=True)


def horse(coat, grazing, rng):
    c = HORSE_COATS[coat]
    mane = "black" if coat not in ("grey", "palomino") else ((0.9, 0.88, 0.8) if coat == "palomino" else "white_coat")
    body_z, lx, ly = 1.25, 0.60, 0.20
    p = legs((-lx, lx), (-ly, ly), body_z - 0.12, 0.072, c)
    p.append(cap((0, 0, body_z), 0.37, 1.15, c))
    if grazing:
        p.append(cap((0.95, 0, 0.85), 0.14, 0.75, c, rot=(0, 55, 0)))
        p.append(box((1.25, 0, 0.32), (0.55, 0.2, 0.22), c, rot=(0, 75, 0)))
        p.append(box((0.92, 0, 0.95), (0.7, 0.05, 0.12), mane, rot=(0, 55, 0), col=False))
    else:
        p.append(cap((0.98, 0, 1.55), 0.17, 0.6, c, rot=(0, -45, 0)))
        p.append(box((1.30, 0, 1.78), (0.6, 0.22, 0.26), c, rot=(0, 40, 0)))
        p.append(box((0.88, 0, 1.70), (0.6, 0.05, 0.14), mane, rot=(0, -45, 0), col=False))
    p.append(cyl((-0.86, 0, 0.98), 0.06, 0.75, mane, rot=(0, -15, 0), col=False))  # tail
    return dict(kind="horse", parts=p, support=[(x, y) for x in (-lx, lx) for y in (-ly, ly)], mass=MASS_KG["horse"],
                height=1.95 if not grazing else 1.6, length=2.5, dynamic=True)


def sheep(rng):
    body_z, lx, ly = 0.62, 0.30, 0.13
    p = legs((-lx, lx), (-ly, ly), body_z - 0.15, 0.035, "black")
    p.append(cap((0, 0, body_z), 0.30, 0.55, "wool"))
    p.append(box((0.58, 0, 0.72), (0.28, 0.16, 0.18), "black", rot=(0, 30, 0)))
    return dict(kind="sheep", parts=p, support=[(x, y) for x in (-lx, lx) for y in (-ly, ly)], mass=MASS_KG["sheep"],
                height=0.9, length=1.3, dynamic=True)


def dog(coat, rng):
    c = DOG_COATS[coat]
    body_z, lx, ly = 0.46, 0.24, 0.09
    p = legs((-lx, lx), (-ly, ly), body_z - 0.06, 0.035, c, hoof=c)
    p.append(cap((0, 0, body_z), 0.13, 0.42, c))
    p.append(cap((0.36, 0, 0.6), 0.08, 0.12, c, rot=(0, -45, 0)))
    p.append(sph((0.45, 0, 0.7), 0.11, c, col=True))
    p.append(box((0.58, 0, 0.67), (0.14, 0.09, 0.08), c, col=False))
    for s in (-1, 1):
        p.append(box((0.42, s * 0.08, 0.76), (0.06, 0.04, 0.1), c, rot=(15 * s, 0, 0), col=False))
    p.append(cyl((-0.34, 0, 0.56), 0.025, 0.3, c, rot=(0, -40, 0), col=False))
    return dict(kind="dog", parts=p, support=[(x, y) for x in (-lx, lx) for y in (-ly, ly)], mass=MASS_KG["dog"],
                height=0.8, length=0.9, dynamic=True)


def car(color, rng):
    col_key = CAR_COLORS[color]
    wx, wy, r = 1.35, 0.78, 0.32
    p = []
    for x in (-wx, wx):
        for y in (-wy, wy):
            p.append(cyl((x, y, r), r, 0.22, "tyre", axis="Y"))
            p.append(cyl((x, y + math.copysign(0.115, y), r), 0.18, 0.01, "rim", axis="Y", col=False))
    p.append(box((0, 0, 0.62), (4.4, 1.78, 0.6), col_key))
    p.append(box((-0.25, 0, 1.15), (2.3, 1.6, 0.46), "glass"))
    p.append(box((-0.25, 0, 1.40), (2.1, 1.5, 0.06), col_key, col=False))
    for s in (-1, 1):
        p.append(box((2.2, s * 0.62, 0.72), (0.03, 0.3, 0.12), "light", col=False))
        p.append(box((-2.2, s * 0.65, 0.75), (0.03, 0.28, 0.1), "tail_light", col=False))
    return dict(kind="car", parts=p, support=[(x, y) for x in (-wx, wx) for y in (-wy, wy)], mass=MASS_KG["car"],
                height=1.43, length=4.4, dynamic=True)


def tractor(color, rng):
    body = {"green": (0.10, 0.36, 0.12), "red": (0.62, 0.07, 0.04), "blue": (0.08, 0.22, 0.5)}[color]
    p = []
    for y in (-0.88, 0.88):
        p.append(cyl((-0.9, y, 0.80), 0.80, 0.45, "tyre", axis="Y"))
        p.append(cyl((-0.9, y + math.copysign(0.23, y), 0.80), 0.42, 0.01, "rim", axis="Y", col=False))
        p.append(cyl((1.35, y * 0.85, 0.46), 0.46, 0.30, "tyre", axis="Y"))
    p.append(box((0.65, 0, 1.05), (2.2, 0.85, 0.75), body))  # hood
    p.append(box((-0.85, 0, 1.0), (1.3, 1.1, 0.5), body))  # chassis under cab
    p.append(box((-0.9, 0, 1.95), (1.2, 1.15, 1.4), "glass"))  # cab glazing
    p.append(box((-0.9, 0, 2.70), (1.4, 1.35, 0.1), body, col=False))  # cab roof
    p.append(cyl((1.2, 0.25, 1.75), 0.05, 0.7, "dark_metal", col=False))  # exhaust
    return dict(kind="tractor", parts=p, support=[(-0.9, -0.88), (-0.9, 0.88), (1.35, -0.75), (1.35, 0.75)],
                mass=MASS_KG["tractor"], height=2.75, length=3.9, dynamic=True)


def round_bale(rng):
    return dict(kind="round_bale", parts=[cyl((0, 0, 0.6), 0.75, 1.2, "hay")],
                support=[(0.7 * math.cos(a), 0.7 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 6, endpoint=False)],
                mass=MASS_KG["round_bale"], height=1.2, length=1.5, dynamic=True)


def trough(rng):
    return dict(kind="trough", parts=[box((0, 0, 0.3), (2.4, 0.7, 0.6), "metal"),
                                      box((0, 0, 0.58), (2.25, 0.55, 0.03), "trough_water", col=False)],
                support=[(x, y) for x in (-1.15, 1.15) for y in (-0.3, 0.3)], mass=MASS_KG["trough"],
                height=0.6, length=2.4, dynamic=True)


# ---- static props: built against the ground under their footprint ------------------------------------
def _footprint_ground(gl, L, W, n=5):
    """gl(lx, ly) -> ground z. Returns (min, max) ground over an L x W footprint centred on the origin."""
    zs = [gl(-L / 2 + L * i / (n - 1), -W / 2 + W * j / (n - 1)) for i in range(n) for j in range(n)]
    return min(zs), max(zs)


def _gable_roof(parts, L, W, top, pitch_deg, roof_mat, wall_mat, overhang=0.45):
    p = math.radians(pitch_deg)
    rise = (W / 2) * math.tan(p)
    for s in (-1, 1):
        y0, z0 = 0.0, top + rise
        y1, z1 = s * (W / 2 + overhang), top - overhang * math.tan(p)
        width = math.hypot(y1 - y0, z1 - z0)
        parts.append(box((0, (y0 + y1) / 2, (z0 + z1) / 2 + 0.09), (L + 0.6, width, 0.16), roof_mat, rot=(-s * pitch_deg, 0, 0)))
    parts.append(prism(-L / 2, L / 2, W / 2, top, rise, wall_mat))
    return rise


def _windows(parts, L, W, z0, h, n_long, gable=True, width=1.1, skip_center_front=False):
    for side in (-1, 1):
        for k in range(n_long):
            x = -L / 2 + L * (k + 0.5) / n_long
            if skip_center_front and side < 0 and abs(x) < 1.0:
                continue
            parts.append(box((x, side * (W / 2 + 0.02), z0 + h / 2), (width + 0.16, 0.04, h + 0.16), "trim", col=False))
            parts.append(box((x, side * (W / 2 + 0.045), z0 + h / 2), (width, 0.03, h), "glass", col=False))
    if gable:
        for side in (-1, 1):
            parts.append(box((side * (L / 2 + 0.02), 0, z0 + h / 2), (0.04, width + 0.16, h + 0.16), "trim", col=False))
            parts.append(box((side * (L / 2 + 0.045), 0, z0 + h / 2), (0.03, width, h), "glass", col=False))


def house(gl, wall, roof, storeys, L, W, pitch, chimney, rng):
    gmin, gmax = _footprint_ground(gl, L + 0.4, W + 0.4)
    floor = gmax + 0.3
    H = 2.8 * storeys
    wm = WALLS[wall]
    p = [box((0, 0, (gmin - 0.4 + floor) / 2), (L + 0.3, W + 0.3, floor - gmin + 0.4), "concrete"),
         box((0, 0, floor + H / 2), (L, W, H), wm)]
    rise = _gable_roof(p, L, W, floor + H, pitch, ROOFS[roof], wm)
    for s in range(storeys):
        _windows(p, L, W, floor + 0.9 + 2.8 * s, 1.25, max(2, int(L / 2.8)), skip_center_front=(s == 0))
    p.append(box((0, -(W / 2 + 0.04), floor + 1.05), (1.0, 0.06, 2.1), "door_wood", col=False))
    step_h = floor - gl(0, -(W / 2 + 0.6)) + 0.05
    p.append(box((0, -(W / 2 + 0.55), floor - step_h / 2), (1.6, 0.9, step_h + 0.25), "concrete"))
    if chimney:
        top = floor + H + rise + 0.9
        p.append(box((L / 4, W / 6, (floor + H + top) / 2), (0.6, 0.6, top - floor - H), "brick"))
    return dict(kind="house", parts=p, height=floor + H + rise - gmin, top_z=floor + H + rise + (0.9 if chimney else 0),
                footprint=(L + 1.5, W + 2.5), dynamic=False, floor_z=floor, ground_minmax=(gmin, gmax))


def garage(gl, wall, roof, double, rng):
    L, W, H = 6.2, (6.4 if double else 3.6), 2.7
    gmin, gmax = _footprint_ground(gl, L + 0.4, W + 0.4)
    floor = gmax + 0.12
    wm = WALLS[wall]
    p = [box((0, 0, (gmin - 0.4 + floor) / 2), (L + 0.2, W + 0.2, floor - gmin + 0.4), "concrete"),
         box((0, 0, floor + H / 2), (L, W, H), wm)]
    rise = _gable_roof(p, L, W, floor + H, 18.0, ROOFS[roof], wm, overhang=0.3)
    doors = (-W / 4, W / 4) if double else (0.0,)
    dw = (W / 2 - 0.6) if double else W - 0.9
    for y in doors:
        p.append(box((L / 2 + 0.03, y, floor + 1.1), (0.05, dw, 2.2), "trim", col=False))
        for k in range(1, 4):
            p.append(box((L / 2 + 0.06, y, floor + 0.55 * k), (0.02, dw - 0.05, 0.04), "metal", col=False))
    return dict(kind="garage", parts=p, height=floor + H + rise - gmin, top_z=floor + H + rise, footprint=(L + 1, W + 1),
                dynamic=False, floor_z=floor, door_x=L / 2, ground_minmax=(gmin, gmax))


def barn(gl, rng):
    L, W, H = float(rng.uniform(14, 18)), float(rng.uniform(9, 11)), 4.6
    gmin, gmax = _footprint_ground(gl, L + 0.4, W + 0.4)
    floor = gmax + 0.1
    p = [box((0, 0, (gmin - 0.4 + floor) / 2), (L + 0.3, W + 0.3, floor - gmin + 0.4), "concrete"),
         box((0, 0, floor + H / 2), (L, W, H), "barn_red")]
    roof = ROOFS[str(rng.choice(["dark grey", "green metal", "blue-grey"]))]
    rise = _gable_roof(p, L, W, floor + H, 38.0, roof, "barn_red", overhang=0.5)
    for sx in (-1, 1):  # white corner trim
        for sy in (-1, 1):
            p.append(box((sx * (L / 2 + 0.02), sy * (W / 2 + 0.02), floor + H / 2), (0.18, 0.18, H), "trim", col=False))
    dw, dh = 4.2, 3.8
    x = L / 2 + 0.04
    p.append(box((x, 0, floor + dh / 2), (0.05, dw, dh), "barn_red", col=False))
    for y0, y1 in ((-dw / 2, dw / 2),):
        for zz in (floor + 0.05, floor + dh - 0.05):
            p.append(box((x + 0.03, 0, zz), (0.04, dw, 0.16), "trim", col=False))
    for yy in (-dw / 2, 0, dw / 2):
        p.append(box((x + 0.03, yy, floor + dh / 2), (0.04, 0.16, dh), "trim", col=False))
    ang = math.degrees(math.atan2(dh, dw / 2))
    for yc in (-dw / 4, dw / 4):
        for s in (-1, 1):
            p.append(box((x + 0.035, yc, floor + dh / 2), (0.03, math.hypot(dw / 2, dh), 0.14), "trim", rot=(s * ang, 0, 0), col=False))
    p.append(box((x, 0, floor + H + rise * 0.45), (0.05, 1.6, 1.4), "dark_wood", col=False))  # hay loft door
    _windows(p, L, W, floor + 2.4, 0.9, max(2, int(L / 4.5)), gable=False, width=0.9)
    return dict(kind="barn", parts=p, height=floor + H + rise - gmin, top_z=floor + H + rise, footprint=(L + 1, W + 1),
                dynamic=False, floor_z=floor, L=L, W=W, ground_minmax=(gmin, gmax))


def silo(gl, rng):
    r, h = float(rng.uniform(2.0, 2.6)), float(rng.uniform(8.5, 11.0))
    gmin, gmax = _footprint_ground(gl, 2 * r + 0.6, 2 * r + 0.6)
    base = gmax + 0.25
    mat = str(rng.choice(["metal", "concrete"]))
    p = [cyl((0, 0, (gmin - 0.4 + base) / 2), r + 0.3, base - gmin + 0.4, "concrete"),
         cyl((0, 0, base + h / 2), r, h, mat),
         sph((0, 0, base + h), r, mat, scale=(1, 1, 0.5))]
    for k in range(int(h / 2.2)):  # hoops
        p.append(cyl((0, 0, base + 1.0 + 2.2 * k), r + 0.03, 0.08, "dark_metal", col=False))
    p.append(box((r + 0.12, 0, base + h / 2), (0.06, 0.5, h), "dark_metal", col=False))  # ladder
    return dict(kind="silo", parts=p, height=base + h + 0.5 * r - gmin, top_z=base + h + 0.5 * r, footprint=(2 * r + 1, 2 * r + 1),
                dynamic=False, radius=r, ground_minmax=(gmin, gmax))


def shed(gl, rng):
    L, W, H = 3.2, 2.5, 2.3
    gmin, gmax = _footprint_ground(gl, L + 0.3, W + 0.3)
    floor = gmax + 0.1
    p = [box((0, 0, (gmin - 0.3 + floor) / 2), (L + 0.1, W + 0.1, floor - gmin + 0.3), "concrete"),
         box((0, 0, floor + H / 2), (L, W, H), "wood"),
         box((0, 0, floor + H + 0.25), (L + 0.5, W + 0.5, 0.12), ROOFS["dark grey"], rot=(8, 0, 0)),
         box((L / 2 + 0.03, 0, floor + 1.0), (0.04, 0.9, 1.9), "dark_wood", col=False)]
    return dict(kind="shed", parts=p, height=floor + H + 0.5 - gmin, top_z=floor + H + 0.5, footprint=(L + 0.8, W + 0.8),
                dynamic=False, ground_minmax=(gmin, gmax))


def woodpile(gl, rng):
    L = 2.4
    gmin, gmax = _footprint_ground(gl, L, 1.2, n=3)
    base = gmin - 0.05
    p = []
    for row in range(3):
        n = 5 - row
        for k in range(n):
            y = -0.5 + (k + 0.5 + row * 0.5) * 0.2 * (5 / 5)
            p.append(cyl((0, y - 0.0, base + 0.11 + row * 0.19 + (gmax - gmin)), 0.11, L, "wood", axis="X"))
    p.insert(0, box((0, 0, (base - 0.1 + gmax + 0.05) / 2), (L, 1.2, gmax + 0.05 - base + 0.1), "dark_wood"))
    return dict(kind="woodpile", parts=p, height=1.0, top_z=gmax + 0.7, footprint=(L + 0.4, 1.6), dynamic=False,
                ground_minmax=(gmin, gmax))


def fence(world_gl, corners, gate_side=0, gate_w=3.5, post_step=2.6):
    """Closed fence through `corners` (world xy, in order); one gap on side gate_side. Static, world frame."""
    p, posts = [], []
    n = len(corners)
    for i in range(n):
        (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % n]
        Ls = math.hypot(x1 - x0, y1 - y0)
        k = max(1, int(round(Ls / post_step)))
        pts = [(x0 + (x1 - x0) * j / k, y0 + (y1 - y0) * j / k) for j in range(k + 1)]
        gate = None
        if i == gate_side:
            mid = k // 2
            gate = (mid, mid + 1)
        for j, (x, y) in enumerate(pts[:-1] if True else pts):
            g = world_gl(x, y)
            posts.append((x, y, g))
            p.append(box((x, y, g + (1.3 - 0.45) / 2), (0.13, 0.13, 1.3 + 0.45), "dark_wood"))
        for j in range(k):
            if gate and j == gate[0]:
                continue
            (xa, ya), (xb, yb) = pts[j], pts[j + 1]
            ga, gb = world_gl(xa, ya), world_gl(xb, yb)
            yaw = math.degrees(math.atan2(yb - ya, xb - xa))
            dxy = math.hypot(xb - xa, yb - ya)
            pitch = -math.degrees(math.atan2(gb - ga, dxy))
            for hh in (0.55, 1.1):
                p.append(box(((xa + xb) / 2, (ya + yb) / 2, (ga + gb) / 2 + hh), (math.hypot(dxy, gb - ga), 0.05, 0.11),
                             "wood", rot=(0, pitch, yaw)))
    return dict(kind="fence", parts=p, posts=len(posts), dynamic=False)


def driveway(world_gl, x0, y0, x1, y1, half_w):
    """Gravel strip draped on the ground from (x0,y0) to (x1,y1) (visual only)."""
    L = math.hypot(x1 - x0, y1 - y0)
    ux, uy = (x1 - x0) / L, (y1 - y0) / L
    pts, idx, cnt = [], [], []
    n_l, n_w = max(2, int(L / 1.0) + 1), 5
    for i in range(n_l):
        for j in range(n_w):
            a, b = L * i / (n_l - 1), -half_w + 2 * half_w * j / (n_w - 1)
            x, y = x0 + ux * a - uy * b, y0 + uy * a + ux * b
            pts.append((x, y, world_gl(x, y) + 0.025))
    for i in range(n_l - 1):
        for j in range(n_w - 1):
            a = i * n_w + j
            idx += [a, a + n_w + 1, a + 1, a, a + n_w, a + n_w + 1]  # counter-clockwise from above (normals up)
            cnt += [3, 3]
    return dict(kind="driveway", parts=[dict(shape="mesh", pts=pts, idx=idx, cnt=cnt, mat="gravel", col=False)], dynamic=False)


# ---- placing movable props on the ground ---------------------------------------------------------------
def rest_pose(prop, x, y, yaw, world_gl):
    """Pose of a movable prop resting on the ground: support points on the least-squares plane through the
    ground under them. Returns (4x4 row-major world matrix as nested lists, max support residual m, tilt deg)."""
    c, s = math.cos(yaw), math.sin(yaw)
    sup = [(x + c * a - s * b, y + s * a + c * b) for a, b in prop["support"]]
    g = np.array([world_gl(px, py) for px, py in sup])
    A = np.array([[1.0, px - x, py - y] for px, py in sup])
    coef, *_ = np.linalg.lstsq(A, g, rcond=None)
    z0, gx, gy = coef
    resid = float(np.max(np.abs(A @ coef - g)))
    n = np.array([-gx, -gy, 1.0])
    n /= np.linalg.norm(n)
    fwd = np.array([c, s, gx * c + gy * s])
    fwd -= n * fwd.dot(n)
    fwd /= np.linalg.norm(fwd)
    left = np.cross(n, fwd)
    M = [[*fwd, 0.0], [*left, 0.0], [*n, 0.0], [x, y, z0, 1.0]]  # USD row-vector convention
    tilt = math.degrees(math.acos(n[2]))
    return M, resid, tilt


def com_and_support(prop):
    """Volume-weighted centre of mass of the colliding parts (uniform density) and the support polygon."""
    vs, cs = [], []
    for p in prop["parts"]:
        if p.get("col") and p["shape"] in ("box", "cyl", "cap", "sph"):
            vs.append(_volume(p))
            cs.append(p["c"])
    vs, cs = np.array(vs), np.array(cs)
    com = (vs[:, None] * cs).sum(0) / vs.sum()
    return com, np.array(prop["support"])


def _inside_margin(pt, poly):
    """Signed distance of pt inside the convex hull of poly (positive = inside)."""
    from math import atan2
    P = sorted(poly.tolist(), key=lambda q: atan2(q[1] - poly[:, 1].mean(), q[0] - poly[:, 0].mean()))
    m = 1e9
    for i in range(len(P)):
        (x0, y0), (x1, y1) = P[i], P[(i + 1) % len(P)]
        ex, ey = x1 - x0, y1 - y0
        L = math.hypot(ex, ey)
        d = (ex * (pt[1] - y0) - ey * (pt[0] - x0)) / L
        m = min(m, d)
    return m


def check_static(prop, tilt_deg):
    """Static equilibrium on the slope: the COM projected along gravity must fall inside the support
    polygon (tilt shifts it by com_z * tan(tilt)); friction must hold the slope (mu 0.5 -> 26.6 deg)."""
    com, sup = com_and_support(prop)
    margin = _inside_margin(com[:2], sup) - com[2] * math.tan(math.radians(tilt_deg))
    return {"com_local": [round(float(v), 3) for v in com], "stability_margin_m": round(float(margin), 3),
            "slope_deg": round(tilt_deg, 2), "stable": bool(margin > 0.03 and tilt_deg < 20.0)}


# ---- USD authoring ---------------------------------------------------------------------------------------
_MATS = {}


def _mat(stage, nv, key):
    if isinstance(key, str) and key in COLORS:
        color, rough = COLORS[key]
        name = key
    else:
        color, rough = tuple(key), 0.8
        name = "c_" + "_".join(f"{int(v * 1000):03d}" for v in color)
    path = f"/World/Materials/Props/{name.replace(' ', '_').replace('-', '_')}"
    m = _MATS.get((id(stage), path))
    if m is None:
        m = nv.create_color_material(stage, path, color, roughness=rough)
        _MATS[(id(stage), path)] = m
    return m


def author(stage, nv, root_path, prop, root_matrix=None, root_xy_yaw=None):
    """Write a prop under root_path. Static props: root = translate(x,y,0) * rotZ(yaw), parts at world z.
    Movable props: root = the rest-pose matrix, parts in the local frame."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    root = UsdGeom.Xform.Define(stage, root_path)
    if root_matrix is not None:
        root.AddTransformOp().Set(Gf.Matrix4d(*[v for row in root_matrix for v in row]))
    elif root_xy_yaw is not None:
        x, y, yaw = root_xy_yaw
        root.AddTranslateOp().Set(Gf.Vec3d(x, y, 0.0))
        root.AddRotateZOp().Set(math.degrees(yaw))
    for i, p in enumerate(prop["parts"]):
        path = f"{root_path}/p{i:03d}"
        sh = p["shape"]
        if sh == "box":
            g = UsdGeom.Cube.Define(stage, path)
            g.CreateSizeAttr(1.0)
            g.AddTranslateOp().Set(Gf.Vec3d(*p["c"]))
            if any(p["rot"]):
                g.AddRotateXYZOp().Set(Gf.Vec3f(*p["rot"]))
            g.AddScaleOp().Set(Gf.Vec3f(*p["s"]))
        elif sh in ("cyl", "cap"):
            g = (UsdGeom.Cylinder if sh == "cyl" else UsdGeom.Capsule).Define(stage, path)
            g.CreateRadiusAttr(p["r"])
            g.CreateHeightAttr(p["h"])
            g.CreateAxisAttr(p["axis"])
            g.AddTranslateOp().Set(Gf.Vec3d(*p["c"]))
            if any(p["rot"]):
                g.AddRotateXYZOp().Set(Gf.Vec3f(*p["rot"]))
        elif sh == "sph":
            g = UsdGeom.Sphere.Define(stage, path)
            g.CreateRadiusAttr(p["r"])
            g.AddTranslateOp().Set(Gf.Vec3d(*p["c"]))
            if any(p["rot"]):
                g.AddRotateXYZOp().Set(Gf.Vec3f(*p["rot"]))
            if p["scale"] != (1, 1, 1):
                g.AddScaleOp().Set(Gf.Vec3f(*p["scale"]))
        elif sh == "prism":
            x0, x1, hw, z0, rise = p["x0"], p["x1"], p["hw"], p["z0"], p["rise"]
            pts = [(x0, -hw, z0), (x0, hw, z0), (x0, 0, z0 + rise), (x1, -hw, z0), (x1, hw, z0), (x1, 0, z0 + rise)]
            idx = [0, 2, 1, 3, 4, 5, 0, 1, 4, 3, 1, 2, 5, 4, 2, 0, 3, 5]
            cnt = [3, 3, 4, 4, 4]
            g = UsdGeom.Mesh.Define(stage, path)
            g.CreatePointsAttr([Gf.Vec3f(*q) for q in pts])
            g.CreateFaceVertexIndicesAttr(idx)
            g.CreateFaceVertexCountsAttr(cnt)
            g.CreateSubdivisionSchemeAttr().Set("none")
        elif sh == "mesh":
            g = UsdGeom.Mesh.Define(stage, path)
            g.CreatePointsAttr([Gf.Vec3f(*q) for q in p["pts"]])
            g.CreateFaceVertexIndicesAttr(p["idx"])
            g.CreateFaceVertexCountsAttr(p["cnt"])
            g.CreateSubdivisionSchemeAttr().Set("none")
        else:
            raise ValueError(sh)
        nv.bind(g.GetPrim(), _mat(stage, nv, p["mat"]))
        if p.get("col"):
            UsdPhysics.CollisionAPI.Apply(g.GetPrim())
            if sh == "prism":
                UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("convexHull")
        else:
            # decorative detail: mark it so the flight job's "add static colliders" pass leaves it alone
            g.GetPrim().CreateAttribute("x500:noCollision", Sdf.ValueTypeNames.Bool).Set(True)
    return root
