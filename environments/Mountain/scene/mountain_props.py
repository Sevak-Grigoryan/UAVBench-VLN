"""
Mountain v2 props (Easy Mountain dataset, 2026-10-20): the objects of the alpine sites, built like every other prop of
the project (props.py / coastal_props.py) from a handful of primitives at real-world size and mass.

Every builder returns (parts, info) or a prop dict, and every object carries a short human DESCRIPTION with its colours
(review item C8: scene_objects must tell similar objects apart -- "the hiker in the red jacket with a blue backpack",
not just "person").

Static props: parts in the local frame, z = 0 at the base, placed with coastal_props.M(x, y, z_base, yaw).
Movable props (people, animals, vehicles): support points at z = 0, rest pose fitted to the ground (props.rest_pose).
"""
import math

import numpy as np

import coastal_props as cp
import props

box, cyl, cap, sph, cone = props.box, props.cyl, props.cap, props.sph, props.cone

# named colours (the names go into the descriptions)
JACKETS = {"red": (0.72, 0.08, 0.06), "orange": (0.92, 0.42, 0.06), "yellow": (0.90, 0.75, 0.10), "blue": (0.10, 0.25, 0.62),
           "light blue": (0.30, 0.58, 0.80), "green": (0.12, 0.42, 0.20), "purple": (0.42, 0.18, 0.52), "black": (0.06, 0.06, 0.07),
           "grey": (0.45, 0.46, 0.48), "turquoise": (0.05, 0.55, 0.55), "pink": (0.85, 0.35, 0.55), "white": (0.88, 0.88, 0.86)}
PACKS = {"red": (0.60, 0.07, 0.05), "blue": (0.08, 0.18, 0.45), "green": (0.15, 0.32, 0.15), "black": (0.05, 0.05, 0.05),
         "orange": (0.85, 0.40, 0.05), "grey": (0.35, 0.36, 0.38), "yellow": (0.85, 0.70, 0.10)}
TENTS = {"orange": (0.90, 0.40, 0.06), "red": (0.70, 0.08, 0.06), "green": (0.18, 0.45, 0.18), "blue": (0.10, 0.28, 0.65),
         "yellow": (0.92, 0.78, 0.12), "grey": (0.55, 0.56, 0.58), "olive": (0.35, 0.38, 0.18), "light blue": (0.35, 0.62, 0.82)}
WINGS = {"red and white": [(0.75, 0.07, 0.06), (0.92, 0.92, 0.90)], "blue and yellow": [(0.10, 0.25, 0.65), (0.92, 0.78, 0.10)],
         "green and white": [(0.10, 0.50, 0.20), (0.92, 0.92, 0.90)], "orange and black": [(0.92, 0.42, 0.05), (0.06, 0.06, 0.07)],
         "purple and pink": [(0.45, 0.15, 0.55), (0.90, 0.40, 0.60)]}
SKIN, HAIR = cp.SKINS, cp.HAIRS
PANTS = {"dark grey": (0.15, 0.15, 0.16), "black": (0.05, 0.05, 0.06), "navy": (0.08, 0.10, 0.25), "khaki": (0.45, 0.40, 0.28),
         "brown": (0.30, 0.20, 0.12), "grey": (0.40, 0.40, 0.42)}
WOOD = (0.36, 0.22, 0.11)
DARK_WOOD = (0.20, 0.12, 0.06)
STONE = (0.47, 0.46, 0.43)
WHITE = (0.86, 0.86, 0.83)
STEEL = (0.50, 0.52, 0.54)
DARK = (0.08, 0.08, 0.09)
GLASS = (0.04, 0.05, 0.06)


def _pick(rng, d):
    k = sorted(d)[int(rng.integers(len(d)))]
    return k, d[k]


# ============================================================================================================ people
def hiker(rng, role="hiker", child=False, jacket=None):
    """Standing person in mountain clothing. role: hiker (backpack, sometimes poles), climber (helmet, rope coil),
    camper, villager, worker (hi-vis + hard hat), rescuer (red jacket + helmet), farmer, paraglider pilot (helmet)."""
    h = float(rng.uniform(1.08, 1.35)) if child else float(rng.uniform(1.58, 1.90))
    s = h / 1.75
    skin, hair = cp._pick(rng, SKIN), cp._pick(rng, HAIR)
    jn, jc = (jacket, JACKETS[jacket]) if jacket else _pick(rng, JACKETS)
    pn, pc = _pick(rng, PANTS)
    hat, pack, poles = None, None, False
    if role == "worker":
        jn, jc = "hi-vis orange", (0.98, 0.45, 0.05)
        hat = ("hard hat", (0.95, 0.85, 0.10))
    elif role == "rescuer":
        jn, jc = "red", (0.80, 0.07, 0.05)
        hat = ("helmet", (0.95, 0.95, 0.93))
    elif role in ("climber", "paraglider pilot"):
        hat = ("helmet", _pick(rng, JACKETS)[1])
    elif role == "hiker":
        pack = _pick(rng, PACKS)
        poles = rng.random() < 0.5
        if rng.random() < 0.35:
            hat = ("cap", _pick(rng, PACKS)[1])
    elif role == "farmer":
        jn, jc = str(rng.choice(["green", "blue", "grey"])), None
        jc = JACKETS[jn]
        hat = ("cap", (0.15, 0.25, 0.15))
    pose = "walk" if (role in ("hiker", "villager", "camper") and rng.random() < 0.4) else "stand"
    fx = (0.17, -0.14) if pose == "walk" else (0.02, 0.02)
    p, sup = [], []
    boots = (0.22, 0.14, 0.08)
    for k, side in enumerate((-1, 1)):
        x = fx[k] * s
        p.append(box((x + 0.04 * s, side * 0.1 * s, 0.04), (0.27 * s, 0.11 * s, 0.08), boots, col=False))
        sup += [(x + 0.04 * s + dx, side * 0.1 * s + dy) for dx in (-0.12 * s, 0.12 * s) for dy in (-0.045 * s, 0.045 * s)]
        lean = math.degrees(math.atan2(-x, 0.85 * s)) if pose == "walk" else 0.0
        p.append(cyl((x / 2, side * 0.1 * s, 0.07 + 0.42 * s), 0.07 * s, 0.84 * s, pc, rot=(0, lean, 0)))
    x0, x1 = min(fx) * s + 0.04 * s - 0.13 * s, max(fx) * s + 0.04 * s + 0.13 * s
    base = box(((x0 + x1) / 2, 0, 0.035), (x1 - x0, 0.30 * s, 0.07), boots)  # one flat collider under both feet
    base["hidden"] = True
    p.append(base)
    p.append(box((0, 0, 0.95 * s), (0.22 * s, 0.34 * s, 0.18 * s), pc))
    p.append(cap((0, 0, 1.24 * s), 0.18 * s, 0.30 * s, jc, axis="Z"))
    for k, side in enumerate((-1, 1)):
        swing = (15 if k == 0 else -15) if pose == "walk" else 0
        p.append(cyl((0, side * 0.24 * s, 1.17 * s), 0.05 * s, 0.6 * s, jc, rot=(0, swing, 0), col=False))
        p.append(sph((0.0, side * 0.24 * s, 0.86 * s), 0.05 * s, skin))
        if poles:
            p.append(cyl((0.12 * s, side * 0.30 * s, 0.45 * s), 0.012, 1.0 * s, STEEL, rot=(0, -12, 0), col=False))
    if pack is not None:
        p.append(box((-0.24 * s, 0, 1.20 * s), (0.22 * s, 0.32 * s, 0.50 * s), pack[1], col=False))
    if role == "climber":
        p.append(cyl((-0.05 * s, 0.0, 1.30 * s), 0.16 * s, 0.05, (0.85, 0.25, 0.10), axis="X", col=False))  # rope coil
    p.append(cyl((0, 0, 1.53 * s), 0.05 * s, 0.1 * s, skin, col=False))
    p.append(sph((0, 0, 1.65 * s), 0.11 * s, skin, col=True))
    if hat is None or hat[0] == "cap":
        p.append(sph((-0.015 * s, 0, 1.69 * s), 0.112 * s, hair, scale=(1, 1, 0.75)))
    if hat is not None:
        if hat[0] in ("hard hat", "helmet"):
            p.append(sph((0, 0, 1.70 * s), 0.13 * s, hat[1], scale=(1.05, 1, 0.75)))
        else:
            p.append(sph((0, 0, 1.71 * s), 0.115 * s, hat[1], scale=(1, 1, 0.55)))
            p.append(box((0.12 * s, 0, 1.70 * s), (0.12 * s, 0.16 * s, 0.015), hat[1], col=False))
    who = {"hiker": "hiker", "climber": "climber", "camper": "camper", "villager": "person", "worker": "worker",
           "rescuer": "mountain rescuer", "farmer": "farmer", "paraglider pilot": "paraglider pilot"}[role]
    if child:
        who = "child"
    words = f"{who} in a {jn} jacket and {pn} trousers"
    if pack is not None:
        words += f" with a {pack[0]} backpack"
    if hat is not None and hat[0] in ("helmet", "hard hat"):
        words += f" and a {hat[0]}"
    if poles:
        words += " and trekking poles"
    kind = "child" if child else "person"
    return dict(kind=kind, role=role, parts=p, support=sup, mass=cp.MASS[kind], height=h, length=0.5, dynamic=True,
                words=words, colour=jn)


def animal(rng, kind):
    if kind == "cow":
        coat = str(rng.choice(list(props.COW_COATS)))
        pr = props.cow(coat, rng.random() < 0.5, rng)
        pr["words"] = f"{coat} cow" + (" grazing" if pr.get("height", 2) < 1.5 else "")
        return pr
    if kind == "horse":
        coat = str(rng.choice(list(props.HORSE_COATS)))
        pr = props.horse(coat, rng.random() < 0.5, rng)
        pr["words"] = f"{coat} horse"
        return pr
    if kind == "sheep":
        pr = props.sheep(rng)
        pr["words"] = "white sheep with a black face"
        return pr
    coat = str(rng.choice(list(props.DOG_COATS)))
    pr = props.dog(coat, rng)
    pr["words"] = f"{coat} dog"
    return pr


def car(rng, color=None, kinds=("sedan", "hatchback", "suv", "suv", "van", "pickup")):
    import city_props
    color = color or str(rng.choice(list(city_props.CAR_PAINT)))
    kind = str(rng.choice(list(kinds)))
    pr = city_props.vehicle(kind, city_props.CAR_PAINT[color], rng)
    pr["words"] = f"{color} {kind}"
    return pr


def rescue_vehicle(rng):
    import city_props
    kind = str(rng.choice(["ambulance", "suv", "pickup"]))
    pr = city_props.vehicle(kind, (0.80, 0.07, 0.05) if kind != "ambulance" else (0.93, 0.93, 0.92), rng)
    pr["words"] = "white mountain-rescue ambulance" if kind == "ambulance" else f"red mountain-rescue {kind}"
    return pr


def machine(rng, kind):
    import city_props
    pr = city_props.excavator(rng) if kind == "excavator" else city_props.dump_truck(rng)
    pr["words"] = f"yellow {kind}"
    return pr


# ============================================================================================================ statics
def chalet(rng, gl):
    wall = str(rng.choice(["log", "log", "white", "cream", "light grey"]))
    roof = str(rng.choice(["dark grey", "brown", "red tile", "dark grey"]))
    L, W = float(rng.uniform(9, 14)), float(rng.uniform(7.5, 10))
    pr = props.house(gl, wall, roof, int(rng.choice([1, 2, 2, 3])), L, W, float(rng.uniform(22, 30)), rng.random() < 0.7, rng)
    # wooden balcony along the long side (alpine chalet)
    if pr["parts"]:
        fz = pr["floor_z"] + 2.8
        pr["parts"].append(box((0, -(W / 2 + 0.7), fz), (L * 0.8, 1.3, 0.14), DARK_WOOD))
        pr["parts"].append(box((0, -(W / 2 + 1.3), fz + 0.5), (L * 0.8, 0.06, 0.9), DARK_WOOD, col=False))
    walls = {"log": "dark timber", "white": "white", "cream": "cream", "light grey": "light grey"}[wall]
    pr["words"] = f"chalet with {walls} walls, a {roof} roof and a wooden balcony"
    return pr


def church(rng, gl, small=False):
    """Village church / chapel: nave with a gable roof and a bell tower with a pointed spire (or an onion dome)."""
    L, W = (9.0, 6.0) if small else (float(rng.uniform(18, 24)), float(rng.uniform(9, 11)))
    H = 4.2 if small else 7.5
    gmin, gmax = props._footprint_ground(gl, L + 6, W + 1)
    fl = gmax + 0.2
    roof = str(rng.choice(["dark grey", "red tile", "brown"]))
    p = [box((0, 0, (gmin - 0.5 + fl) / 2), (L + 0.4, W + 0.4, fl - gmin + 0.5), "concrete"), box((0, 0, fl + H / 2), (L, W, H), WHITE)]
    rise = props._gable_roof(p, L, W, fl + H, 40.0, props.ROOFS[roof], WHITE)
    th = (7.5 if small else float(rng.uniform(20, 28)))
    tw = 2.6 if small else 4.4
    tx = L / 2 + tw / 2 - 0.2
    p.append(box((tx, 0, fl + th / 2), (tw, tw, th), WHITE))
    for side in (-1, 1):  # belfry openings
        p.append(box((tx, side * (tw / 2 + 0.02), fl + th - 1.6), (tw * 0.45, 0.04, 1.6), DARK, col=False))
        p.append(box((tx + side * (tw / 2 + 0.02), 0, fl + th - 1.6), (0.04, tw * 0.45, 1.6), DARK, col=False))
    p.append(box((tx, 0, fl + th - 3.2), (tw + 0.2, tw + 0.2, 0.25), (0.55, 0.45, 0.30), col=False))  # clock band
    onion = (not small) and rng.random() < 0.4
    if onion:
        p.append(sph((tx, 0, fl + th + 1.6), tw * 0.55, (0.20, 0.35, 0.25), scale=(1, 1, 1.25), col=True))
        p.append(cone((tx, 0, fl + th + 4.2), 0.35, 2.4, (0.20, 0.35, 0.25)))
        top = fl + th + 5.6
    else:
        sh = tw * (2.6 if small else 2.4)
        p.append(cone((tx, 0, fl + th + sh / 2), tw * 0.72, sh, props.ROOFS[roof] if roof != "red tile" else (0.25, 0.25, 0.27)))
        top = fl + th + sh
    p.append(cyl((tx, 0, top + 0.8), 0.05, 1.6, (0.75, 0.60, 0.20), col=False))  # cross
    p.append(box((tx, 0, top + 1.2), (0.05, 0.7, 0.06), (0.75, 0.60, 0.20), col=False))
    p.append(box((-L / 2 - 0.04, 0, fl + 1.5), (0.05, 1.6, 3.0), "door_wood", col=False))
    for k in range(3 if small else 5):
        x = -L / 2 + L * (k + 0.5) / (3 if small else 5)
        for side in (-1, 1):
            p.append(box((x, side * (W / 2 + 0.03), fl + H * 0.55), (0.9, 0.04, 2.2 if not small else 1.3), GLASS, col=False))
    tower_kind = "an onion-domed" if onion else "a pointed"
    words = ("small white chapel with " if small else "white church with ") + f"{tower_kind} bell tower"
    return dict(parts=p, top_z=top + 1.6, foot=(L + tw, max(W, tw)), words=words, kind="chapel" if small else "church")


def mountain_hut(rng):
    """Alpine refuge: stone ground floor, timber upper floor, steep roof, terrace with picnic tables, flag pole, solar
    panels. Base at z = 0 on a levelled platform."""
    L, W = float(rng.uniform(16, 22)), float(rng.uniform(9, 12))
    roof_n, roof = _pick(rng, {"dark grey": (0.10, 0.10, 0.11), "red": (0.45, 0.08, 0.05), "green": (0.10, 0.25, 0.14)})
    shutters_n, shutters = _pick(rng, {"red": (0.60, 0.07, 0.05), "green": (0.10, 0.35, 0.15), "dark brown": DARK_WOOD})
    p = [box((0, 0, 1.6 - 0.3), (L, W, 3.8), STONE), box((0, 0, 3.5 + 1.5), (L, W, 3.0), WOOD)]
    props._gable_roof(p, L, W, 6.5, 35.0, roof, WOOD, overhang=0.8)
    for z0 in (1.0, 4.0):
        for k in range(int(L // 3)):
            x = -L / 2 + 1.8 + 3 * k
            for side in (-1, 1):
                p.append(box((x, side * (W / 2 + 0.03), z0 + 0.7), (1.0, 0.04, 1.2), GLASS, col=False))
                p.append(box((x - 0.75, side * (W / 2 + 0.05), z0 + 0.7), (0.45, 0.04, 1.25), shutters, col=False))
                p.append(box((x + 0.75, side * (W / 2 + 0.05), z0 + 0.7), (0.45, 0.04, 1.25), shutters, col=False))
    p.append(box((0, -(W / 2 + 3.5), 0.12), (L, 6.5, 0.3), (0.40, 0.30, 0.18)))  # terrace boards
    for k in range(int(L // 5)):
        x = -L / 2 + 3 + 5 * k
        p += picnic_table_parts(x, -(W / 2 + 3.5), 0.27)
    p.append(cyl((L / 2 + 2.0, -(W / 2 + 1.0), 4.5), 0.06, 9.0, WHITE, col=False))  # flag pole
    flag_n, flag = _pick(rng, {"red and white": (0.80, 0.08, 0.06), "blue": (0.10, 0.25, 0.65), "green": (0.10, 0.45, 0.20)})
    p.append(box((L / 2 + 2.6, -(W / 2 + 1.0), 8.3), (1.2, 0.03, 0.8), flag, col=False))
    for k in range(3):  # solar panels on the roof's south face
        p.append(box((-L / 3 + k * L / 3, W * 0.22, 8.35), (2.6, 1.6, 0.06), (0.05, 0.08, 0.20), rot=(-35, 0, 0), col=False))
    words = f"mountain hut with stone walls, a timber upper floor, {shutters_n} shutters and a {roof_n} roof"
    return dict(parts=p, top_z=6.5 + W / 2 * math.tan(math.radians(35)) + 0.5, foot=(L + 2, W + 8), words=words, flag=flag_n,
                terrace=(0, -(W / 2 + 3.5), L, 6.5, 0.27))


def picnic_table_parts(x, y, z):
    return [box((x, y, z + 0.74), (1.9, 0.75, 0.06), WOOD), box((x, y - 0.65, z + 0.44), (1.9, 0.28, 0.05), WOOD),
            box((x, y + 0.65, z + 0.44), (1.9, 0.28, 0.05), WOOD),
            box((x - 0.75, y, z + 0.37), (0.08, 1.5, 0.74), DARK_WOOD), box((x + 0.75, y, z + 0.37), (0.08, 1.5, 0.74), DARK_WOOD)]


def station(rng, top=False, color=(0.65, 0.08, 0.06), cname="red"):
    """Cable-car station: concrete building with a large open bay on the cable side (+x), flat roof, sign band."""
    L, W, H = 18.0, 12.0, 9.0
    p = [box((-2, 0, H / 2 - 0.3), (L - 4, W, H + 0.6), (0.72, 0.72, 0.70)),
         box((L / 2 - 2, W / 2 - 0.6, H / 2), (4, 1.2, H), (0.72, 0.72, 0.70)), box((L / 2 - 2, -W / 2 + 0.6, H / 2), (4, 1.2, H), (0.72, 0.72, 0.70)),
         box((0, 0, H + 0.25), (L + 1, W + 1, 0.5), DARK), box((0, -(W / 2 + 0.05), H - 1.0), (L * 0.8, 0.06, 0.9), color, col=False),
         box((L / 2 - 2, 0, 0.3), (4, W - 2.4, 0.6), (0.6, 0.6, 0.6)), cyl((L / 2 - 2, 0, H - 2.0), 2.6, 0.5, STEEL, col=False)]
    for k in range(4):
        p.append(box((-L / 2 + 2 + 3.5 * k, -(W / 2 + 0.04), 3.0), (2.4, 0.04, 2.6), GLASS, col=False))
    words = ("top station" if top else "valley station") + f" of the cable car, a grey concrete building with a {cname} sign band"
    return dict(parts=p, top_z=H + 0.5, foot=(L, W), words=words, cable_z=H - 2.0, cable_x=L / 2 - 2)


def pylon(h):
    """Steel cable-car pylon: tapered 4-leg tower with a cross arm carrying both cables (y = +-3.2 m)."""
    p = []
    b, t = 2.6, 0.9
    for sx in (-1, 1):
        for sy in (-1, 1):
            dx, dy = (b - t) * sx, (b - t) * sy
            ang_x = math.degrees(math.atan2(dy, h))
            ang_y = -math.degrees(math.atan2(dx, h))
            p.append(cyl(((b + t) / 2 * sx, (b + t) / 2 * sy, h / 2), 0.16, math.hypot(h, (b - t) * 1.41), STEEL, rot=(ang_x, ang_y, 0)))
    for k in range(1, int(h // 4)):
        z = k * 4.0
        w = b - (b - t) * z / h
        p.append(box((0, 0, z), (2 * w, 2 * w, 0.18), STEEL, col=False))
    p.append(box((0, 0, h + 0.3), (1.0, 7.6, 0.6), STEEL))
    for sy in (-3.2, 3.2):
        p.append(cyl((0, sy, h + 0.75), 0.35, 1.6, DARK, axis="X", col=False))
    return p


def segment(a, b, r, mat, col=False):
    """Cylinder from point a to point b (world-local coordinates of the static it belongs to)."""
    dx, dy, dz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    L = math.sqrt(dx * dx + dy * dy + dz * dz)
    yaw = math.degrees(math.atan2(dy, dx))
    elev = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    return cyl(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2), r, L, mat, axis="X", rot=(0, -elev, yaw), col=col)


def gondola(color):
    return [box((0, 0, -3.4), (2.6, 2.0, 2.2), color, col=False), box((0, 0, -3.1), (2.65, 2.05, 0.9), GLASS, col=False),
            cyl((0, 0, -1.2), 0.06, 2.4, STEEL, col=False), box((0, 0, -0.1), (1.0, 0.3, 0.3), DARK, col=False)]


def mast(rng, h):
    """Lattice radio / telecom mast with red-and-white bands, dish and panel antennas, equipment cabin."""
    p = []
    w = 1.3
    n = int(h // 5)
    for k in range(n):
        z0 = k * h / n
        c = (0.80, 0.08, 0.06) if k % 2 == 0 else (0.92, 0.92, 0.90)
        for sx in (-1, 1):
            for sy in (-1, 1):
                p.append(cyl((sx * w, sy * w, z0 + h / n / 2), 0.09, h / n + 0.02, c))
        p.append(box((0, 0, z0 + 0.1), (2 * w + 0.2, 2 * w + 0.2, 0.12), c, col=False))
        for side in (-1, 1):
            p.append(box((0, side * w, z0 + h / n / 2), (2 * w * 1.41, 0.06, 0.06), c, rot=(0, 45 if k % 2 else -45, 0), col=False))
    for k, a in enumerate((0, 120, 240)):
        r = math.radians(a)
        p.append(cyl((1.9 * math.cos(r), 1.9 * math.sin(r), h * 0.72), 0.55, 0.25, WHITE, axis="X", rot=(0, 0, a), col=False))
        p.append(box((1.6 * math.cos(r + 0.6), 1.6 * math.sin(r + 0.6), h - 2.0), (0.25, 0.5, 2.2), WHITE, rot=(0, 0, a + 35), col=False))
    p.append(cyl((0, 0, h + 2.0), 0.08, 4.0, STEEL, col=False))
    p.append(sph((0, 0, h + 4.1), 0.18, (0.9, 0.1, 0.05), col=False))
    p.append(box((5.5, 0, 1.3), (4.0, 2.6, 2.6), (0.75, 0.75, 0.72)))
    return dict(parts=p, top_z=h + 4.3, foot=(15, 8), words=f"red-and-white radio mast, about {h:.0f} m tall, with a small equipment cabin")


def summit_cross(rng):
    metal = rng.random() < 0.4
    h = float(rng.uniform(3.5, 6.0))
    c = (0.55, 0.56, 0.58) if metal else (0.30, 0.18, 0.08)
    p = [cyl((0, 0, h / 2), 0.13, h, c), box((0, 0, h * 0.72), (0.18, h * 0.55, 0.22), c)]
    p.append(cone((0, 0, 0.45), 1.3, 0.9, STONE, col=True))  # cairn base
    for k in range(5):
        a = k * 1.25
        p.append(sph((1.1 * math.cos(a), 1.1 * math.sin(a), 0.25), 0.35, STONE, scale=(1.2, 1, 0.7), col=False))
    return dict(parts=p, top_z=h + 0.2, foot=(3, 3), words=("metal" if metal else "wooden") + f" summit cross about {h:.0f} m tall on a stone cairn")


def observatory(rng):
    n = int(rng.integers(1, 3))
    p = [box((0, 0, 2.0), (14.0, 9.0, 4.4), WHITE)]
    for k in range(n):
        x = -4 + 8 * k if n == 2 else 0.0
        p.append(cyl((x, 0, 5.5), 3.2, 3.2, WHITE))
        p.append(sph((x, 0, 7.1), 3.25, (0.92, 0.92, 0.92), scale=(1, 1, 0.95), col=True))
        p.append(box((x + 1.4, 0, 8.8), (2.4, 0.9, 1.6), (0.15, 0.15, 0.16), rot=(0, -40, 0), col=False))
    words = "observatory with " + ("two white domes" if n == 2 else "a white dome")
    return dict(parts=p, top_z=10.4, foot=(15, 10), words=words)


def wind_turbine(rng, h=48.0, yaw_rotor=0.0):
    p = []
    for k in range(6):
        z0 = h * k / 6
        r = 1.9 - 0.9 * (k + 0.5) / 6
        p.append(cyl((0, 0, z0 + h / 12), r, h / 6 + 0.02, (0.90, 0.90, 0.88)))
    p.append(box((-0.5, 0, h + 1.2), (5.5, 2.3, 2.4), (0.90, 0.90, 0.88)))
    p.append(sph((2.6, 0, h + 1.2), 1.1, (0.90, 0.90, 0.88), col=False))
    ang0 = float(rng.uniform(0, 120))
    R = 22.0
    for k in range(3):
        a = math.radians(ang0 + 120 * k)
        cy_, cz_ = (R / 2 + 0.8) * math.cos(a), (R / 2 + 0.8) * math.sin(a)
        p.append(box((3.0, cy_, h + 1.2 + cz_), (0.3, R, 1.6), (0.92, 0.92, 0.90), rot=(math.degrees(a) - 90, 0, 0), col=False))
    return dict(parts=p, top_z=h + 1.2 + R + 1.0, foot=(R * 2, R * 2), words=f"white three-bladed wind turbine, about {h:.0f} m tall")


def tent(rng):
    cn, c = _pick(rng, TENTS)
    if rng.random() < 0.55:  # dome tent
        r = float(rng.uniform(1.1, 1.6))
        p = [sph((0, 0, 0.0), r, c, scale=(1.15, 1.0, 0.85), col=True), box((r * 1.1, 0, 0.45), (0.05, 0.7, 0.8), DARK, col=False)]
        return dict(parts=p, top_z=0.85 * r, foot=(2.6 * r, 2.2 * r), words=f"{cn} dome tent")
    L, hw, H = float(rng.uniform(2.2, 3.2)), float(rng.uniform(0.9, 1.3)), float(rng.uniform(1.1, 1.5))
    p = [props.prism(-L / 2, L / 2, hw, 0.0, H, c), box((L / 2 + 0.02, 0, H * 0.35), (0.04, hw * 0.8, H * 0.6), DARK, col=False)]
    return dict(parts=p, top_z=H, foot=(L + 0.6, 2 * hw + 0.6), words=f"{cn} ridge tent")


def helicopter(rng):
    cn, (c1, c2) = _pick(rng, {"red and white": ((0.78, 0.06, 0.05), (0.92, 0.92, 0.90)), "yellow": ((0.92, 0.75, 0.08), (0.10, 0.10, 0.11)),
                               "blue and white": ((0.10, 0.25, 0.62), (0.92, 0.92, 0.90))})
    p = [cap((0.4, 0, 1.55), 1.0, 2.6, c1, axis="X"), sph((2.0, 0, 1.6), 0.95, GLASS, scale=(1.0, 0.95, 0.85), col=False),
         cyl((-3.4, 0, 1.75), 0.28, 4.6, c2, axis="X"), box((-5.6, 0, 2.5), (0.9, 0.1, 1.5), c1),
         cyl((0.2, 0, 2.85), 0.25, 0.5, DARK, col=False)]
    for side in (-1, 1):
        p.append(cyl((0.4, side * 1.0, 0.1), 0.06, 3.6, DARK, axis="X"))
        for x in (-0.6, 1.4):
            p.append(cyl((x, side * 0.85, 0.5), 0.05, 0.8, DARK, rot=(side * 15, 0, 0), col=False))
    for a in (20, 110, 200, 290):
        r = math.radians(a)
        p.append(box((0.2 + 2.6 * math.cos(r), 2.6 * math.sin(r), 3.12), (5.2, 0.3, 0.05), DARK, rot=(0, 0, a), col=False))
    return dict(parts=p, top_z=3.3, foot=(11, 3), words=f"{cn} rescue helicopter parked on the helipad")


def windsock():
    return dict(parts=[cyl((0, 0, 3.0), 0.06, 6.0, STEEL), cone((0.8, 0, 5.8), 0.3, 1.5, (0.95, 0.45, 0.05))], top_z=6.2, foot=(2, 2),
                words="orange windsock on a pole")


def wing(rng):
    """A paraglider wing laid out on the grass: an arc of cells in two colours."""
    cn, (a, b) = _pick(rng, WINGS)
    p = []
    span, chord = float(rng.uniform(10, 12)), 2.6
    n = 11
    for k in range(n):
        t = (k + 0.5) / n - 0.5
        y = t * span
        x = 1.6 * (2 * t) ** 2
        p.append(box((x, y, 0.06), (chord, span / n + 0.05, 0.10), a if k % 2 == 0 else b, rot=(0, 0, math.degrees(math.atan(6.4 * t / span * 10)) * 0.2), col=False))
    return dict(parts=p, top_z=0.15, foot=(4, span + 1), words=f"{cn} paraglider wing laid out on the grass")


def campfire():
    p = [cyl((0, 0, 0.05), 0.3, 0.1, (0.05, 0.04, 0.03), col=False)]
    for k in range(8):
        a = k * math.pi / 4
        p.append(sph((0.8 * math.cos(a), 0.8 * math.sin(a), 0.1), 0.18, STONE, scale=(1.2, 1, 0.7), col=False))
    p.append(cyl((0.0, 0.0, 0.15), 0.07, 0.9, DARK_WOOD, axis="X", rot=(0, 0, 30), col=False))
    p.append(cyl((0.0, 0.0, 0.2), 0.07, 0.9, DARK_WOOD, axis="X", rot=(0, 0, -40), col=False))
    return dict(parts=p, top_z=0.3, foot=(2, 2), words="stone campfire ring")


def gravel_pile(rng, tone):
    r, h = float(rng.uniform(4, 7)), float(rng.uniform(2.5, 4.5))
    return dict(parts=[cone((0, 0, h / 2), r, h, tone, col=True)], top_z=h, foot=(2 * r, 2 * r), words="heap of crushed stone")


def boathouse(rng):
    wn, wc = _pick(rng, {"dark red": (0.38, 0.07, 0.05), "dark green": (0.10, 0.22, 0.14), "brown": (0.30, 0.18, 0.09)})
    p = cp.shed_building(10.0, 7.0, 4.0, wc, (0.10, 0.10, 0.11))
    return dict(parts=p, top_z=4.4, foot=(11, 8), words=f"{wn} wooden boathouse")


def jetty(L, bed=None):
    """Plank deck on piles; bed(u, w) = lake-bed height under (u, w) relative to the deck: every pile stands on the bed
    (0.4 m driven into it) instead of hanging in the water."""
    p = [box((L / 2, 0, 0.0), (L, 2.4, 0.2), (0.45, 0.33, 0.20))]
    for k in range(int(L // 3) + 1):
        for side in (-1, 1):
            u, w = min(L - 0.2, 3.0 * k + 0.2), side * 1.1
            lo = min(-1.2, bed(u, w) - 0.4) if bed else -2.1
            p.append(cyl((u, w, (lo - 0.1) / 2), 0.12, -0.1 - lo, DARK_WOOD))
    return p


def rowing_boat(rng):
    cn, c = _pick(rng, {"red": (0.60, 0.08, 0.06), "white": (0.85, 0.85, 0.83), "green": (0.12, 0.35, 0.18), "blue": (0.10, 0.25, 0.55),
                        "yellow": (0.90, 0.72, 0.10)})
    L, B = float(rng.uniform(3.6, 4.6)), 1.4
    mesh = cp.hull_mesh(L, B, 0.45, 0.35, c, bow=0.4, stern_w=0.6)
    p = list(mesh) if isinstance(mesh, list) else [mesh]
    p.append(box((0, 0, 0.18), (0.3, B * 0.85, 0.05), WOOD, col=False))
    return dict(parts=p, top_z=0.5, foot=(L, B), words=f"small {cn} rowing boat")


def signpost(rng):
    p = [cyl((0, 0, 1.3), 0.06, 2.6, DARK_WOOD)]
    for k, a in enumerate((20, -60, 150)):
        p.append(box((0.4 * math.cos(math.radians(a)), 0.4 * math.sin(math.radians(a)), 2.3 - 0.25 * k), (0.9, 0.04, 0.18),
                     (0.90, 0.75, 0.10), rot=(0, 0, a), col=False))
    return dict(parts=p, top_z=2.6, foot=(1.5, 1.5), words="yellow trail signpost")


def info_board():
    return dict(parts=[cyl((-0.9, 0, 1.0), 0.06, 2.0, DARK_WOOD), cyl((0.9, 0, 1.0), 0.06, 2.0, DARK_WOOD),
                       box((0, 0, 1.6), (1.9, 0.08, 1.1), (0.25, 0.32, 0.20)), box((0, 0, 2.25), (2.1, 0.5, 0.06), DARK_WOOD, rot=(20, 0, 0), col=False)],
                top_z=2.4, foot=(2.4, 1.0), words="wooden trail information board")


def simple_building(rng, L, W, H, wall_name, wall, roof_name, roof, words):
    p = cp.shed_building(L, W, H, wall, roof)
    return dict(parts=p, top_z=H + 0.4, foot=(L + 1, W + 1), words=words.format(wall=wall_name, roof=roof_name))


def stone_bridge(span, width, deck_z, water_z):
    """Old stone arch bridge: deck with parapets and two arches over the river, abutments down to the bank."""
    p = [box((0, 0, deck_z - 0.4), (span, width, 0.8), STONE)]
    for side in (-1, 1):
        p.append(box((0, side * (width / 2 - 0.25), deck_z + 0.55), (span, 0.5, 1.1), STONE))
    pier_h = max(1.0, deck_z - water_z + 1.5)
    for x in (-span / 2 + 1.5, 0.0, span / 2 - 1.5):
        p.append(box((x, 0, deck_z - 0.8 - pier_h / 2), (3.0 if abs(x) < 0.1 else 3.6, width, pier_h), STONE))
    return p
