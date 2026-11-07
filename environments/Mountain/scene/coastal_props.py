"""
Coastal v2 props (Easy Coastal, 2026-10-13): people of the coast (fishermen, dock workers, crew, swimmers,
sunbathers, lifeguards, tourists), beach and harbour objects (umbrellas, towels, loungers, kayaks, fish crates,
barrels, nets, bollards, forklifts), and ships built from real hull shapes (container ship, tanker, ferry, fishing
trawler, research vessel with a helideck, coast-guard cutter, tug, motor yacht, sailboat, motorboat, rowing boat,
kayak).

Same rules as City/Forest props (props.py): simple primitives at real dimensions and masses.
  * Movable things (people, crates, barrels, kayaks on the sand, vehicles, loungers) are rigid bodies: support
    points at local z = 0, set down on the surface under them (sand, quay, pier deck, ship deck), checked with
    props.check_static and simulated with gravity during the flight.
  * Ships float at their waterline (local z = 0 = sea level). Calm water + a moored / anchored ship = static
    equilibrium, so ships are static colliders (the drone can land on a helideck). Crew on a ship's deck are
    rigid bodies standing on that deck.
  * Hulls are closed meshes wound counter-clockwise seen from outside (RTX shades back faces as a bright band).

Local frame of every prop: x = forward (bow), y = left (port), z = up.
"""
import math

import numpy as np

import props

box, cyl, cap, sph, cone = props.box, props.cyl, props.cap, props.sph, props.cone

MASS = {"person": 72.0, "child": 30.0, "sunbather": 68.0, "dog": 30.0, "fish crate": 18.0, "crate stack": 54.0,
        "barrel": 140.0, "cool box": 9.0, "bucket": 6.0, "lounger": 12.0, "kayak on sand": 24.0, "net pile": 120.0,
        "forklift": 3200.0, "pallet of crates": 160.0, "beach ball": 0.3, "surfboard rack": 40.0}

SKINS = [(0.87, 0.70, 0.58), (0.76, 0.57, 0.43), (0.55, 0.38, 0.27), (0.36, 0.24, 0.17), (0.93, 0.78, 0.66)]
HAIRS = [(0.08, 0.06, 0.05), (0.30, 0.18, 0.08), (0.62, 0.45, 0.22), (0.55, 0.55, 0.55), (0.15, 0.10, 0.07)]
SHIRTS = [(0.70, 0.12, 0.10), (0.12, 0.25, 0.55), (0.92, 0.92, 0.90), (0.10, 0.10, 0.11), (0.20, 0.45, 0.25),
          (0.85, 0.65, 0.15), (0.45, 0.20, 0.50), (0.25, 0.55, 0.70)]
PANTS = [(0.10, 0.14, 0.28), (0.12, 0.12, 0.13), (0.35, 0.30, 0.22), (0.45, 0.45, 0.47), (0.20, 0.22, 0.30)]
SWIM = [(0.85, 0.10, 0.12), (0.10, 0.20, 0.65), (0.05, 0.05, 0.06), (0.95, 0.55, 0.10), (0.15, 0.55, 0.45), (0.80, 0.30, 0.60)]
HIVIS = [(0.98, 0.45, 0.05), (0.85, 0.95, 0.10)]
OILSKIN = [(0.95, 0.75, 0.05), (0.90, 0.35, 0.05), (0.15, 0.30, 0.20)]

HULL_COLORS = {"red": (0.55, 0.06, 0.05), "dark blue": (0.05, 0.10, 0.25), "black": (0.03, 0.03, 0.035),
               "green": (0.06, 0.25, 0.13), "white": (0.88, 0.88, 0.86), "grey": (0.40, 0.42, 0.44),
               "orange": (0.90, 0.38, 0.05), "light blue": (0.30, 0.55, 0.75), "yellow": (0.90, 0.72, 0.10),
               "teal": (0.05, 0.40, 0.42)}
WHITE = (0.80, 0.80, 0.78)
DECK_GREEN = (0.20, 0.33, 0.22)
DECK_RED = (0.40, 0.12, 0.08)
TEAK = (0.55, 0.38, 0.20)
CONTAINER_COLORS = [(0.62, 0.10, 0.08), (0.08, 0.25, 0.50), (0.10, 0.40, 0.22), (0.80, 0.55, 0.10), (0.55, 0.55, 0.55),
                    (0.85, 0.85, 0.82), (0.45, 0.15, 0.35), (0.90, 0.40, 0.05), (0.20, 0.55, 0.65)]


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def M(x, y, z, yaw):
    """Root matrix (USD row-vector convention): translate(x, y, z) * rotZ(yaw)."""
    c, s = math.cos(yaw), math.sin(yaw)
    return [[c, s, 0.0, 0.0], [-s, c, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [x, y, z, 1.0]]


# =================================================================================================== people
def person(rng, role="tourist", child=False, pose=None):
    """Standing person (City's person with coastal clothing): role in tourist, worker (hi-vis + hard hat),
    fisherman (oilskin jacket, cap, rod), crew (overalls), swimmer (swimwear), lifeguard (red), market seller
    (apron). Support = the four corners of both feet."""
    h = float(rng.uniform(1.05, 1.35)) if child else float(rng.uniform(1.58, 1.90))
    s = h / 1.75
    skin, hair = _pick(rng, SKINS), _pick(rng, HAIRS)
    shirt, pants, sleeve = _pick(rng, SHIRTS), _pick(rng, PANTS), True
    hat = None
    if role == "worker":
        shirt, hat = _pick(rng, HIVIS), ("hard hat", (0.95, 0.85, 0.10) if rng.random() < 0.6 else (0.95, 0.95, 0.93))
    elif role == "fisherman":
        shirt, hat = _pick(rng, OILSKIN), ("cap", (0.10, 0.12, 0.20))
    elif role == "crew":
        shirt = pants = _pick(rng, [(0.10, 0.20, 0.45), (0.90, 0.40, 0.05), (0.20, 0.20, 0.22)])
        hat = ("cap", (0.10, 0.10, 0.12)) if rng.random() < 0.4 else None
    elif role == "swimmer":
        shirt, pants, sleeve = skin, _pick(rng, SWIM), False
    elif role == "lifeguard":
        shirt, pants, hat = (0.85, 0.08, 0.06), (0.85, 0.80, 0.25), ("cap", (0.85, 0.08, 0.06))
    elif role == "seller":
        shirt, hat = (0.92, 0.92, 0.90), ("cap", (0.92, 0.92, 0.90))
    pose = pose or ("walk" if rng.random() < 0.45 else "stand")
    fx = (0.17, -0.14) if pose == "walk" else (0.02, 0.02)
    p, sup = [], []
    for k, side in enumerate((-1, 1)):
        x = fx[k] * s
        p.append(box((x + 0.04 * s, side * 0.1 * s, 0.035), (0.26 * s, 0.10 * s, 0.07),
                     (0.10, 0.08, 0.07) if role != "swimmer" else skin, col=False))
        sup += [(x + 0.04 * s + dx, side * 0.1 * s + dy) for dx in (-0.12 * s, 0.12 * s) for dy in (-0.045 * s, 0.045 * s)]
        lean = math.degrees(math.atan2(-x, 0.85 * s)) if pose == "walk" else 0.0
        leg = skin if role == "swimmer" else pants
        p.append(cyl((x / 2, side * 0.1 * s, 0.07 + 0.42 * s), 0.07 * s, 0.84 * s, leg, rot=(0, lean, 0)))
    # ONE flat invisible base under both feet: two separate small foot boxes made PhysX creep (people pivoted about
    # one foot ~3 cm and 2.5 deg per minute on the harbour quays, Coastal probes 2026-10-14)
    x0, x1 = min(fx) * s + 0.04 * s - 0.13 * s, max(fx) * s + 0.04 * s + 0.13 * s
    base = box(((x0 + x1) / 2, 0, 0.035), (x1 - x0, 0.30 * s, 0.07), (0.10, 0.08, 0.07))
    base["hidden"] = True
    p.append(base)
    p.append(box((0, 0, 0.95 * s), (0.22 * s, 0.34 * s, 0.18 * s), pants))  # hips / trunks
    p.append(cap((0, 0, 1.24 * s), 0.17 * s, 0.30 * s, shirt, axis="Z"))
    if role == "seller":  # apron
        p.append(box((0.13 * s, 0, 1.05 * s), (0.02, 0.30 * s, 0.55 * s), (0.85, 0.85, 0.80), col=False))
    if role == "fisherman" and rng.random() < 0.5:  # rubber boots
        for side in (-1, 1):
            p.append(cyl((0.02 * s, side * 0.1 * s, 0.25 * s), 0.08 * s, 0.38 * s, (0.15, 0.30, 0.15), col=False))
    arm_mat = shirt if sleeve else skin
    holding = role == "fisherman" and pose == "stand"
    for k, side in enumerate((-1, 1)):
        swing = (15 if k == 0 else -15) if pose == "walk" else (-40 if holding else 0)
        p.append(cyl((0.12 * s if holding else 0, side * 0.235 * s, 1.17 * s), 0.045 * s, 0.6 * s, arm_mat, rot=(0, swing, 0), col=False))
        p.append(sph((0.25 * s if holding else 0.0, side * 0.235 * s, 0.95 * s if holding else 0.86 * s), 0.05 * s, skin))
    if holding:  # fishing rod held in both hands, angled up over the water ahead
        L = float(rng.uniform(2.4, 3.6))
        ang = math.radians(float(rng.uniform(30, 45)))
        cx, cz = 0.3 * s + L / 2 * math.cos(ang), 0.95 * s + L / 2 * math.sin(ang)
        p.append(cyl((cx, 0, cz), 0.012, L, (0.12, 0.12, 0.12), rot=(0, 90 - math.degrees(ang), 0), col=False))
        p.append(cyl((0.3 * s, -0.06, 0.95 * s), 0.04, 0.07, (0.6, 0.6, 0.62), axis="Y", col=False))  # reel
    p.append(cyl((0, 0, 1.53 * s), 0.05 * s, 0.1 * s, skin, col=False))
    p.append(sph((0, 0, 1.65 * s), 0.11 * s, skin, col=True))
    if hat is None or hat[0] == "cap":
        p.append(sph((-0.015 * s, 0, 1.69 * s), 0.112 * s, hair, scale=(1, 1, 0.75)))
    if hat is not None:
        if hat[0] == "hard hat":
            p.append(sph((0, 0, 1.70 * s), 0.13 * s, hat[1], scale=(1.05, 1, 0.7)))
            p.append(cyl((0, 0, 1.68 * s), 0.15 * s, 0.015, hat[1], col=False))
        else:
            p.append(sph((0, 0, 1.71 * s), 0.115 * s, hat[1], scale=(1, 1, 0.55)))
            p.append(box((0.12 * s, 0, 1.70 * s), (0.12 * s, 0.16 * s, 0.015), hat[1], col=False))
    kind = "child" if child else "person"
    return dict(kind=kind, role=role, parts=p, support=sup, mass=MASS[kind], height=h, length=0.5, dynamic=True)


def sunbather(rng, on_back=None):
    """Person lying on the sand (on a towel laid separately). Support = the body's footprint (very stable)."""
    s = float(rng.uniform(1.58, 1.88)) / 1.75
    skin, hair, swim = _pick(rng, SKINS), _pick(rng, HAIRS), _pick(rng, SWIM)
    p = [cap((0.0, 0, 0.12 * s), 0.13 * s, 0.55 * s, skin, axis="X"),               # torso
         box((-0.35 * s, 0, 0.12 * s), (0.22 * s, 0.32 * s, 0.2 * s), swim),        # trunks
         sph((0.55 * s, 0, 0.12 * s), 0.11 * s, skin, col=True),                     # head
         sph((0.57 * s, 0, 0.16 * s), 0.112 * s, hair, scale=(1, 1, 0.7))]
    for side in (-1, 1):
        p.append(cyl((-0.82 * s, side * 0.09 * s, 0.08 * s), 0.07 * s, 0.84 * s, skin, axis="X"))
        p.append(cyl((0.05 * s, side * 0.28 * s, 0.07 * s), 0.045 * s, 0.6 * s, skin, axis="X", col=False))
    sup = [(x, y) for x in (-1.2 * s, 0.6 * s) for y in (-0.16 * s, 0.16 * s)]
    return dict(kind="sunbather", role="sunbather", parts=p, support=sup, mass=MASS["sunbather"], height=0.3, length=1.9, dynamic=True)


# =================================================================================================== beach
def towel(rng):
    col = _pick(rng, SWIM + [(0.95, 0.95, 0.92), (0.25, 0.65, 0.85)])
    return [box((0, 0, 0.006), (2.0, 1.0, 0.012), col, col=False), box((0.0, 0, 0.007), (2.0, 0.12, 0.012), (0.95, 0.95, 0.92), col=False)]


def umbrella(rng):
    col = _pick(rng, [(0.85, 0.15, 0.12), (0.10, 0.35, 0.70), (0.95, 0.80, 0.15), (0.15, 0.55, 0.35), (0.95, 0.95, 0.92), (0.90, 0.45, 0.10)])
    r = float(rng.uniform(1.0, 1.4))
    return [cyl((0, 0, 0.85), 0.025, 2.3, (0.85, 0.85, 0.85)),       # pole, 0.3 m pushed into the sand
            cone((0, 0, 2.05), r, 0.45, col),
            cyl((0, 0, 1.84), r, 0.03, col, col=False)]


def lounger(rng):
    col = _pick(rng, [(0.95, 0.95, 0.92), (0.10, 0.35, 0.70), (0.85, 0.40, 0.10), (0.30, 0.30, 0.32)])
    p = [box((0.1, 0, 0.32), (1.4, 0.62, 0.05), col), box((-0.75, 0, 0.55), (0.55, 0.62, 0.05), col, rot=(0, -40, 0))]
    for x in (-0.55, 0.7):
        for y in (-0.27, 0.27):
            p.append(box((x, y, 0.15), (0.06, 0.06, 0.30), (0.75, 0.75, 0.75)))
    sup = [(x, y) for x in (-0.58, 0.73) for y in (-0.30, 0.30)]
    return dict(kind="lounger", parts=p, support=sup, mass=MASS["lounger"], height=0.6, length=2.0, dynamic=True)


def kayak_on_sand(rng):
    col = _pick(rng, [(0.95, 0.75, 0.05), (0.90, 0.20, 0.05), (0.10, 0.45, 0.75), (0.15, 0.60, 0.30)])
    L = float(rng.uniform(3.4, 4.6))
    p = [cap((0, 0, 0.17), 0.17, L - 0.4, col, axis="X", col=False),
         box((0, 0, 0.33), (0.9, 0.42, 0.05), (0.08, 0.08, 0.09), col=False),       # cockpit
         box((-L * 0.25, 0, 0.04), (L * 0.35, 0.30, 0.08), col),                     # flat contact pads under the hull
         box((L * 0.25, 0, 0.04), (L * 0.35, 0.30, 0.08), col)]
    sup = [(x, y) for x in (-L * 0.42, L * 0.42) for y in (-0.15, 0.15)]
    return dict(kind="kayak on sand", parts=p, support=sup, mass=MASS["kayak on sand"], height=0.35, length=L, dynamic=True)


def cool_box(rng):
    col = _pick(rng, [(0.10, 0.35, 0.70), (0.85, 0.15, 0.12), (0.95, 0.95, 0.92)])
    p = [box((0, 0, 0.2), (0.6, 0.4, 0.4), col), box((0, 0, 0.42), (0.62, 0.42, 0.06), (0.95, 0.95, 0.92))]
    return dict(kind="cool box", parts=p, support=[(x, y) for x in (-0.3, 0.3) for y in (-0.2, 0.2)], mass=MASS["cool box"],
                height=0.45, length=0.6, dynamic=True)


def lifeguard_tower():
    red, white = (0.85, 0.10, 0.08), (0.95, 0.95, 0.92)
    p = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            p.append(box((x, y, 1.0), (0.15, 0.15, 2.6), white))         # legs set 0.3 m into the sand
    p += [box((0, 0, 2.35), (2.6, 2.6, 0.12), white), box((0, 0, 3.4), (2.2, 2.2, 2.0), red),
          box((0.0, 0, 3.6), (2.25, 2.25, 0.7), (0.08, 0.10, 0.12), col=False), box((0, 0, 4.5), (2.9, 2.9, 0.15), white),
          box((1.8, 0, 1.2), (1.6, 0.6, 0.06), white, rot=(0, 52, 0))]   # ramp
    return p


def beach_hut(rng):
    col = _pick(rng, [(0.30, 0.60, 0.85), (0.95, 0.80, 0.20), (0.90, 0.40, 0.45), (0.40, 0.75, 0.50), (0.95, 0.95, 0.92), (0.85, 0.35, 0.15)])
    p = [box((0, 0, 1.15), (2.4, 2.2, 2.5), col), box((1.21, 0, 1.0), (0.02, 1.0, 1.9), (0.95, 0.95, 0.92), col=False),
         props.prism(-1.35, 1.35, 1.25, 2.4, 0.8, (0.95, 0.95, 0.92))]
    return p


def volleyball_net():
    p = [cyl((0, -4.8, 1.1), 0.05, 2.9, (0.9, 0.9, 0.9)), cyl((0, 4.8, 1.1), 0.05, 2.9, (0.9, 0.9, 0.9)),
         box((0, 0, 2.2), (0.01, 9.6, 0.9), (0.08, 0.08, 0.08), col=False), box((0, 0, 2.66), (0.03, 9.6, 0.05), (0.95, 0.95, 0.95), col=False)]
    return p


# =================================================================================================== harbour
def fish_crate(rng, n=1):
    """A blue/red plastic fish crate (or a stack of n) full of fish: silver bodies with dark backs."""
    col = _pick(rng, [(0.10, 0.30, 0.65), (0.75, 0.12, 0.10), (0.85, 0.85, 0.82), (0.95, 0.55, 0.10)])
    p = []
    for k in range(n):
        z = 0.15 + 0.3 * k
        p.append(box((0, 0, z), (0.80, 0.50, 0.30), col))
    top = 0.3 * n
    for j in range(6):  # fish on the top crate
        fx = -0.25 + 0.1 * j
        p.append(sph((fx, 0.0, top - 0.02), 0.06, (0.75, 0.78, 0.80), scale=(1.0, 3.2, 0.6), rot=(0, 0, float(rng.uniform(-20, 20)))))
    kind = "fish crate" if n == 1 else "crate stack"
    return dict(kind=kind, parts=p, support=[(x, y) for x in (-0.4, 0.4) for y in (-0.25, 0.25)],
                mass=MASS["fish crate"] * n, height=top, length=0.8, dynamic=True)


def barrel(rng):
    col = _pick(rng, [(0.10, 0.30, 0.60), (0.60, 0.10, 0.08), (0.20, 0.35, 0.20), (0.35, 0.22, 0.10)])
    p = [cyl((0, 0, 0.45), 0.29, 0.9, col, col=False)]
    # collide as the inscribed square prism: the PhysX cylinder standing on its end tipped over and rolled (barrels
    # rolled up to 24 m, Coastal flight 020); the box is stable and 6 cm inside the visible barrel
    core = box((0, 0, 0.45), (0.41, 0.41, 0.9), col)
    core["hidden"] = True
    p.append(core)
    sup = [(0.29 * math.cos(a), 0.29 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]
    return dict(kind="barrel", parts=p, support=sup, mass=MASS["barrel"], height=0.9, length=0.6, dynamic=True)


def net_pile(rng):
    col = _pick(rng, [(0.10, 0.35, 0.20), (0.75, 0.25, 0.10), (0.15, 0.20, 0.45)])
    p = [box((0, 0, 0.25), (1.8, 1.3, 0.5), col), sph((0.1, 0.1, 0.5), 0.6, col, scale=(1.3, 0.9, 0.45))]
    for k in range(5):
        p.append(sph((-0.7 + 0.35 * k, 0.6, 0.5), 0.07, (0.95, 0.75, 0.10), col=False))  # floats
    return dict(kind="net pile", parts=p, support=[(x, y) for x in (-0.9, 0.9) for y in (-0.65, 0.65)], mass=MASS["net pile"],
                height=0.8, length=1.8, dynamic=True)


def pallet_of_crates(rng):
    p = [box((0, 0, 0.07), (1.2, 1.0, 0.14), (0.55, 0.40, 0.22))]
    col = _pick(rng, [(0.10, 0.30, 0.65), (0.75, 0.12, 0.10), (0.85, 0.85, 0.82)])
    for i in (-1, 1):
        for k in range(int(rng.integers(2, 5))):
            p.append(box((i * 0.3, 0, 0.29 + 0.3 * k), (0.58, 0.96, 0.29), col))
    return dict(kind="pallet of crates", parts=p, support=[(x, y) for x in (-0.6, 0.6) for y in (-0.5, 0.5)],
                mass=MASS["pallet of crates"], height=1.3, length=1.2, dynamic=True)


def forklift(rng):
    col = _pick(rng, [(0.95, 0.70, 0.05), (0.85, 0.30, 0.05), (0.15, 0.40, 0.20)])
    p = []
    for x in (-0.65, 0.55):
        for y in (-0.5, 0.5):
            p.append(cyl((x, y, 0.3), 0.3, 0.22, "tyre", axis="Y", col=False))
            p.append(box((x, y, 0.05), (0.3, 0.22, 0.1), "tyre"))
    p += [box((-0.15, 0, 0.75), (2.0, 1.1, 0.9), col), box((-0.75, 0, 1.25), (0.7, 1.1, 0.6), (0.2, 0.2, 0.2)),
          box((0.0, 0, 1.6), (0.9, 1.0, 0.05), (0.15, 0.15, 0.15), col=False)]
    for y in (-0.45, 0.45):
        p.append(box((0.0, y, 1.1), (0.05, 0.05, 1.1), (0.2, 0.2, 0.2), col=False))
        p.append(box((1.0, y * 0.9, 1.15), (0.08, 0.1, 2.2), (0.25, 0.25, 0.27)))
        p.append(box((1.55, y * 0.6, 0.12), (1.0, 0.12, 0.05), (0.25, 0.25, 0.27)))
    sup = [(x, y) for x in (-0.65, 0.55) for y in (-0.5, 0.5)]
    return dict(kind="forklift", parts=p, support=sup, mass=MASS["forklift"], height=2.3, length=3.2, dynamic=True)


def bollard():
    return [cyl((0, 0, 0.2), 0.18, 0.6, (0.12, 0.12, 0.13)), cyl((0, 0, 0.52), 0.24, 0.06, (0.12, 0.12, 0.13), col=False)]


def market_stall(rng):
    """Fish-market table under a striped awning: 4 legs set into the ground, top at 0.9 m, canopy on 4 posts."""
    col = _pick(rng, [(0.85, 0.15, 0.12), (0.10, 0.35, 0.70), (0.15, 0.55, 0.35), (0.95, 0.95, 0.92)])
    p = [box((0, 0, 0.88), (2.4, 1.1, 0.06), (0.85, 0.85, 0.85))]
    for x in (-1.1, 1.1):
        for y in (-0.5, 0.5):
            p.append(box((x, y, 0.42), (0.06, 0.06, 0.9), (0.6, 0.6, 0.62)))
            p.append(cyl((x * 1.05, y * 1.6, 1.2), 0.03, 2.5, (0.6, 0.6, 0.62)))
    p.append(box((0, 0, 2.45), (2.7, 2.0, 0.05), col, col=False))
    p.append(box((0, 0, 0.92), (2.3, 1.0, 0.02), (0.92, 0.95, 0.96), col=False))   # crushed ice
    for j in range(10):  # fish laid on the ice
        p.append(sph((-0.95 + 0.21 * j, float(rng.uniform(-0.3, 0.3)), 0.96), 0.06, (0.72, 0.75, 0.78), scale=(1.0, 3.5, 0.6),
                     rot=(0, 0, float(rng.uniform(70, 110)))))
    if rng.random() < 0.5:  # a big tuna across the end of the table
        p.append(sph((0.7, 0.0, 1.05), 0.17, (0.30, 0.35, 0.45), scale=(5.0, 1.0, 1.0)))
    return p


def fish_rack():
    """Fish-drying rack: two A-frames and poles with fish hanging."""
    p = []
    for x in (-2.0, 2.0):
        for y in (-0.6, 0.6):
            p.append(cyl((x, y * 0.5, 0.9), 0.04, 2.0, (0.45, 0.30, 0.15), rot=(math.copysign(15, y), 0, 0)))
    for z in (1.4, 1.8):
        p.append(cyl((0, 0, z), 0.03, 4.4, (0.45, 0.30, 0.15), axis="X", col=False))
        for k in range(12):
            p.append(sph((-1.9 + 0.35 * k, 0, z - 0.22), 0.05, (0.70, 0.62, 0.48), scale=(0.6, 0.6, 3.5), col=False))
    return p


def jib_crane(height=14.0):
    yel = (0.92, 0.70, 0.08)
    return [cyl((0, 0, height / 2), 0.5, height, yel), box((4.0, 0, height + 0.5), (10.0, 0.6, 0.8), yel),
            box((-2.0, 0, height + 0.3), (2.0, 1.6, 1.4), (0.3, 0.3, 0.3)),
            cyl((8.5, 0, height - 3.0), 0.02, 6.0, (0.1, 0.1, 0.1), col=False), box((8.5, 0, height - 6.1), (0.5, 0.3, 0.3), (0.2, 0.2, 0.2), col=False)]


def gantry_crane(reach=38.0, height=46.0):
    """Ship-to-shore container crane: 4 legs on rails along the quay (local x), boom across over the water
    (local -y, seaward), machinery house and A-frame on top. Returns parts and its obstacle boxes."""
    blue, grey = (0.10, 0.35, 0.65), (0.35, 0.36, 0.38)
    span, gauge = 16.0, 30.0
    p = []
    for x in (-gauge / 2 / 2.2, gauge / 2 / 2.2):
        for y in (-span / 2, span / 2):
            p.append(box((x, y, height * 0.38), (1.4, 1.4, height * 0.76), blue))
    p += [box((0, 0, height * 0.76), (gauge / 2.2 + 1.4, span + 1.4, 1.6), blue),
          box((0, -reach / 2 + span / 2, height * 0.8), (3.0, reach + span, 2.0), blue),                 # boom over the water
          box((0, span / 2 + 6, height * 0.8), (3.0, 12.0, 2.0), blue),                                   # back reach
          box((0, span / 2 + 4, height * 0.86), (6.0, 7.0, 4.0), grey),                                   # machinery house
          box((0, -2.0, height * 0.81 - 2.2), (2.6, 3.0, 2.4), (0.85, 0.85, 0.82)),                       # trolley cab
          box((0, 0, height * 0.97), (1.2, 1.2, height * 0.22), blue)]                                   # apex post
    obs = [(-gauge / 4.4 - 1, gauge / 4.4 + 1, -span / 2 - reach, span / 2 + 12, height * 1.08)]
    return p, obs


def container_stack(rng, rows, tiers, bays):
    """Block of 40-ft containers (12.2 x 2.44 x 2.6 m) on the terminal: one collider box, coloured boxes for looks."""
    L, W, H = 12.2, 2.44, 2.6
    p = [box((0, 0, H * tiers / 2), (L * 0 + bays * L, rows * W, H * tiers), (0.3, 0.3, 0.3), col=True)]
    p[0]["mat"] = _pick(rng, CONTAINER_COLORS)
    for b in range(bays):
        for r in range(rows):
            n = int(rng.integers(max(1, tiers - 2), tiers + 1))
            for t in range(n):
                p.append(box(((b - (bays - 1) / 2) * L, (r - (rows - 1) / 2) * W, H * t + H / 2), (L - 0.1, W - 0.06, H - 0.04),
                             _pick(rng, CONTAINER_COLORS), col=False))
    return p, H * tiers


# =================================================================================================== hull
def hull_mesh(L, B, F, D, mat, bow=0.30, stern_w=0.80, sheer=0.0, n=16):
    """Closed ship hull, waterline at z = 0: deck edge at z = F (+ sheer rising toward the bow), keel at -D.
    Stations from the transom (x = -L/2) to the stem (x = +L/2); half-breadth full amidships, tapering over the
    fore `bow` fraction to a sharp stem, transom at stern_w of the beam. Counter-clockwise winding from outside."""
    xs = np.linspace(-L / 2, L / 2, n)
    rings = []
    for x in xs:
        u = (x + L / 2) / L  # 0 stern .. 1 bow
        if u > 1 - bow:
            t = (u - (1 - bow)) / bow
            w = B / 2 * max(0.015, math.sqrt(max(0.0, 1 - t ** 1.8)))
        elif u < 0.18:
            w = B / 2 * (stern_w + (1 - stern_w) * math.sin(u / 0.18 * math.pi / 2))
        else:
            w = B / 2
        f = F + sheer * max(0.0, u - 0.5) * 2
        rings.append([(x, w, f), (x, 0.92 * w, -0.45 * D), (x, 0.35 * w, -D), (x, -0.35 * w, -D), (x, -0.92 * w, -0.45 * D), (x, -w, f)])
    pts, idx, cnt = [], [], []
    for r in rings:
        pts += r
    k = 6
    for i in range(n - 1):
        for j in range(k):
            a, b = i * k + j, i * k + (j + 1) % k
            c, d = (i + 1) * k + (j + 1) % k, (i + 1) * k + j
            idx += [a, d, c, b]
            cnt.append(4)
    idx += list(range(k))                                       # transom, normal -x
    cnt.append(k)
    idx += [(n - 1) * k + j for j in reversed(range(k))]        # stem, normal +x
    cnt.append(k)
    return dict(shape="mesh", c=(0, 0, 0), pts=pts, idx=idx, cnt=cnt, rot=(0, 0, 0), mat=mat, col=True)


def _half_breadth(L, B, x, bow=0.30, stern_w=0.80):
    u = (x + L / 2) / L
    if u > 1 - bow:
        t = (u - (1 - bow)) / bow
        return B / 2 * max(0.015, math.sqrt(max(0.0, 1 - t ** 1.8)))
    if u < 0.18:
        return B / 2 * (stern_w + (1 - stern_w) * math.sin(u / 0.18 * math.pi / 2))
    return B / 2


def deck_rect(L, B, x0, x1, margin, bow=0.30, stern_w=0.80):
    """Flat part of a hull's deck between x0 and x1: aft of amidships only (forward of it the sheer lifts the deck),
    as wide as the narrowest hull section in that range minus a margin. -> (x0, x1, half_width)."""
    x1 = min(x1, -0.02 * L)
    w = min(_half_breadth(L, B, float(x), bow, stern_w) for x in np.linspace(x0, x1, 9)) - margin
    return x0, x1, w


def _windows(p, x0, x1, y_half, z, h, step=1.6, glass=(0.05, 0.07, 0.09)):
    """A dark window band on both sides and the front of a deckhouse."""
    for side in (-1, 1):
        p.append(box(((x0 + x1) / 2, side * (y_half + 0.01), z), (x1 - x0 - 0.6, 0.02, h), glass, col=False))


def _rail(p, L, B, F, bow=0.30, stern_w=0.80, xs=None, mat=(0.85, 0.85, 0.85)):
    """Deck-edge railing posts and top rail (no collision -- decoration)."""
    xs = xs if xs is not None else np.linspace(-L / 2 + 0.6, L / 2 - 2.0, max(4, int(L / 2.0)))
    for x in xs:
        w = _half_breadth(L, B, x, bow, stern_w) - 0.15
        for side in (-1, 1):
            p.append(cyl((x, side * w, F + 0.5), 0.025, 1.0, mat, col=False))
    for side in (-1, 1):
        p.append(cyl((float(np.mean(xs)), side * (B / 2 - 0.15), F + 1.0), 0.02, float(xs[-1] - xs[0]), mat, axis="X", col=False))


def _pad_marking(p, x, y, z, r=5.0, h_col=(0.95, 0.95, 0.92), ring=(0.95, 0.80, 0.10)):
    """Helideck marking: yellow circle and white H, painted (no collision)."""
    for j in range(28):
        a = 2 * math.pi * j / 28
        p.append(box((x + r * math.cos(a), y + r * math.sin(a), z + 0.01), (1.1, 0.3, 0.02), ring, rot=(0, 0, math.degrees(a) + 90), col=False))
    for (dx, sx, sy) in ((-1.3, 0.5, 4.0), (1.3, 0.5, 4.0), (0.0, 2.6, 0.5)):
        p.append(box((x + dx, y, z + 0.012), (sx, sy, 0.02), h_col, col=False))


def ship(kind, rng):
    """A vessel at its waterline. Returns dict(kind, words, L, B, parts, decks=[(x0,x1,y0,y1,z)], boxes=[(x0,x1,y0,y1,top)],
    crew=[(x, y, z)] free standing spots on deck, helideck=(x, y, z) or None, top=z max, mass_t (displacement))."""
    hull_name = None
    p, decks, boxes, crew, heli = [], [], [], [], None
    if kind == "container ship":
        L = float(rng.uniform(110, 150)); B = round(L * 0.15, 1); F, D = 6.5, 7.0
        hull_name = _pick(rng, ["dark blue", "red", "black", "green", "grey"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.22, stern_w=0.85, sheer=1.5))
        sx = -0.38 * L
        p += [box((sx, 0, F + 7.5), (12, B - 2, 15), WHITE), box((sx, 0, F + 15.6), (9, B + 3, 1.4), WHITE),
              box((sx + 6.05, 0, F + 13), (0.1, B - 3, 3), (0.06, 0.08, 0.10), col=False),
              box((sx - 7.5, 0, F + 12), (3.5, 4.5, 8), HULL_COLORS["black"] if hull_name != "black" else (0.6, 0.1, 0.1)),
              cyl((sx, 0, F + 19.5), 0.25, 6.0, (0.9, 0.9, 0.9), col=False)]
        for z in np.arange(F + 2.0, F + 14, 2.8):
            _windows(p, sx - 5.5, sx + 5.5, (B - 2) / 2, z, 0.7)
        n_bays = int((0.68 * L) // 6.4)
        rows = int((B - 2) // 2.44)
        tiers = int(rng.integers(2, 5))
        for b in range(n_bays):
            x = -0.30 * L + 3.2 + b * 6.4
            w = min(rows, int((_half_breadth(L, B, x + 3, 0.22, 0.85) * 2 - 1.5) // 2.44))
            if w < 2:
                continue
            for r_ in range(w):
                nt = int(rng.integers(max(1, tiers - 1), tiers + 1))
                for t in range(nt):
                    p.append(box((x, (r_ - (w - 1) / 2) * 2.44, F + 1.3 + 2.6 * t), (6.0, 2.38, 2.56), _pick(rng, CONTAINER_COLORS), col=t == 0))
        p.append(box((0.45 * L, 0, F + 2.5), (6, 3, 3), WHITE))  # forecastle deckhouse
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 1.3 + 2.6 * tiers + 1), (sx - 9.5, sx + 6.5, -B / 2 - 1.5, B / 2 + 1.5, F + 22.5)]
        top = F + 22.5
        x0, x1, w = deck_rect(L, B, sx + 6.6, sx + 10.4, 0.8, 0.22, 0.85)
        decks = [(x0, x1, -w, w, F)]
        crew = [(sx + 8.5, side * (w - 0.8), F) for side in (-1, 1)]
        words = f"the large {hull_name}-hulled container ship loaded with coloured containers"
    elif kind == "tanker":
        L = float(rng.uniform(110, 145)); B = round(L * 0.17, 1); F, D = 5.0, 8.0
        hull_name = _pick(rng, ["black", "red", "dark blue", "grey"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.22, stern_w=0.88, sheer=1.2))
        p.append(box((0, 0, F + 0.02), (L * 0.75, B - 1.2, 0.04), DECK_RED, col=False))
        for y in (-2.5, -1.0, 1.0, 2.5):
            p.append(cyl((0.05 * L, y, F + 1.0), 0.35, 0.72 * L, (0.80, 0.80, 0.78), axis="X", col=False))
        p.append(box((0.05 * L, 0, F + 2.0), (0.72 * L, 1.2, 0.15), (0.8, 0.8, 0.78), col=False))
        sx = -0.40 * L
        p += [box((sx, 0, F + 6.5), (12, B - 2, 13), WHITE), box((sx, 0, F + 13.6), (8, B + 2, 1.2), WHITE),
              box((sx - 6.8, 0, F + 11), (3, 4, 7), (0.85, 0.15, 0.08))]
        for z in np.arange(F + 2.0, F + 12, 2.8):
            _windows(p, sx - 5.5, sx + 5.5, (B - 2) / 2, z, 0.7)
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 3), (sx - 8.5, sx + 6.5, -B / 2 - 1, B / 2 + 1, F + 16)]
        top = F + 16
        x0, x1, w = deck_rect(L, B, -0.30 * L, -0.02 * L, 0.9, 0.22, 0.88)
        decks = [(x0, x1, -w, w, F)]
        crew = [(x, side * (w - 0.6), F) for x in (-0.1 * L, -0.2 * L) for side in (-1, 1)]
        words = f"the long {hull_name} oil tanker with pipes along its deck"
    elif kind == "ferry":
        L = float(rng.uniform(65, 95)); B = round(L * 0.19, 1); F, D = 4.0, 3.5
        hull_name = _pick(rng, ["white", "dark blue", "red", "teal"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.25, stern_w=0.9, sheer=0.8))
        band = _pick(rng, [(0.85, 0.10, 0.08), (0.05, 0.25, 0.55), (0.10, 0.45, 0.25)])
        x0, x1 = -0.40 * L, 0.30 * L
        for k, (h, inset) in enumerate(((2.8, 0.6), (2.8, 1.2), (2.6, 2.4))):
            z = F + 2.8 * k
            p.append(box(((x0 + x1) / 2 + 0.03 * k * L, 0, z + h / 2), (x1 - x0 - 0.08 * k * L, B - 2 * inset, h), WHITE))
            _windows(p, x0 + 0.03 * k * L, x1 - 0.05 * k * L, B / 2 - inset, z + 1.5, 1.1)
        p.append(box((0, -(B / 2 + 0.01), F - 0.6), (L * 0.85, 0.02, 0.6), band, col=False))
        p.append(box((0, (B / 2 + 0.01), F - 0.6), (L * 0.85, 0.02, 0.6), band, col=False))
        p.append(box((-0.15 * L, 0, F + 8.4 + 3.0), (6, 4, 6), band))  # funnel
        p.append(box((0.28 * L, 0, F + 8.6), (6, B - 1, 1.0), (0.08, 0.10, 0.12), col=False))  # bridge windows
        top_deck = F + 8.2
        tc, th, tw = (x0 + x1) / 2 + 0.06 * L, (x1 - x0 - 0.16 * L) / 2, (B - 4.8) / 2   # top tier centre, half length, half width
        decks = [(tc - th + 0.6, tc + th - 0.6, -(tw - 0.6), tw - 0.6, top_deck)]
        for x in np.linspace(tc - th + 1.5, tc + th - 1.5, 8):
            if abs(x + 0.15 * L) < 4.5:  # funnel
                continue
            for side in (-1, 1):
                crew.append((float(x), side * (tw - 1.2), top_deck))
        _rail(p, x1 - x0 - 0.16 * L, B - 5, top_deck, xs=np.linspace(x0 + 0.08 * L, x1 - 0.08 * L, 18))
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 1), (x0, x1, -B / 2, B / 2, top_deck + 1.5), (-0.15 * L - 3, -0.15 * L + 3, -2, 2, F + 14.5)]
        top = F + 14.5
        words = f"the {'white' if hull_name == 'white' else hull_name + ' and white'} passenger ferry"
    elif kind == "trawler":
        L = float(rng.uniform(20, 32)); B = round(L * 0.30, 1); F, D = 2.0, 2.6
        hull_name = _pick(rng, ["red", "dark blue", "green", "black", "light blue", "orange", "teal"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.32, stern_w=0.82, sheer=1.2))
        wx = 0.12 * L
        p += [box((wx, 0, F + 1.6), (5.0, B * 0.55, 3.2), WHITE), box((wx + 2.51, 0, F + 2.4), (0.02, B * 0.5, 0.9), (0.06, 0.08, 0.1), col=False),
              cyl((wx, 0, F + 5.5), 0.12, 5.0, (0.9, 0.9, 0.9), col=False), box((wx, 0, F + 7.4), (1.6, 0.25, 0.2), (0.2, 0.2, 0.2), col=False)]
        for side in (-1, 1):  # A-frame gantry at the stern
            p.append(box((-L / 2 + 1.2, side * (B / 2 - 0.6), F + 3.0), (0.4, 0.4, 6.0), (0.95, 0.65, 0.05)))
        p.append(box((-L / 2 + 1.2, 0, F + 6.0), (0.5, B - 0.8, 0.5), (0.95, 0.65, 0.05)))
        p.append(cyl((-0.25 * L, 0, F + 0.9), 0.8, B * 0.5, (0.10, 0.30, 0.20), axis="Y"))   # net drum
        for side in (-1, 1):  # outrigger booms raised in a V
            p.append(cyl((wx - 1.0, side * 2.2, F + 6.5), 0.1, 9.0, (0.6, 0.6, 0.6), rot=(side * 32, 0, 0), col=False))
        x0, x1, w = deck_rect(L, B, -0.25 * L + 1.3, wx - 3.0, 0.6, 0.32, 0.82)
        decks = [(x0, x1, -w, w, F)]
        crew = [(float(rng.uniform(x0 + 0.4, x1 - 0.4)), float(rng.uniform(-w + 0.4, w - 0.4)), F) for _ in range(3)] if x1 - x0 > 1.5 else []
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 3.5), (wx - 2.5, wx + 2.5, -B / 4, B / 4, F + 8.2), (-L / 2, -L / 2 + 2, -B / 2, B / 2, F + 6.3)]
        top = F + 8.2
        words = f"the {hull_name} fishing trawler with the orange gantry"
    elif kind in ("research vessel", "supply vessel"):
        L = float(rng.uniform(50, 72)); B = round(L * 0.22, 1); F, D = 4.0, 4.0
        hull_name = _pick(rng, ["orange", "red", "dark blue", "white"] if kind == "research vessel" else ["red", "orange", "dark blue", "yellow"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.28, stern_w=0.92, sheer=2.0))
        sx0, sx1 = 0.12 * L, 0.36 * L
        for k in range(3):
            p.append(box(((sx0 + sx1) / 2 + 0.6 * k, 0, F + 1.4 + 2.8 * k), (sx1 - sx0 - 1.5 * k, B - 2 - 0.8 * k, 2.8), WHITE))
            _windows(p, sx0 + 0.6 * k, sx1 - 0.9 * k, (B - 2 - 0.8 * k) / 2, F + 1.6 + 2.8 * k, 0.8)
        p.append(box((sx1 - 0.2, 0, F + 7.0), (0.04, B - 4, 1.0), (0.06, 0.08, 0.1), col=False))
        p.append(cyl(((sx0 + sx1) / 2, 0, F + 11.5), 0.2, 6.0, (0.9, 0.9, 0.9), col=False))
        p.append(box(((sx0 + sx1) / 2, 0, F + 13.0), (2.4, 0.3, 0.3), (0.2, 0.2, 0.2), col=False))
        p.append(box((-0.05 * L, 0, F + 0.02), (0.75 * L, B - 1.4, 0.04), (0.30, 0.32, 0.34), col=False))  # steel deck
        hx = -0.30 * L
        _pad_marking(p, hx, 0.0, F + 0.04, r=min(5.0, B / 2 - 2.0))
        p.append(box((sx0 - 1.0, B / 2 - 1.8, F + 3.0), (0.5, 0.5, 6.0), (0.95, 0.65, 0.05)))          # side crane post
        p.append(box((sx0 - 4.0, B / 2 - 1.8, F + 6.0), (6.0, 0.4, 0.4), (0.95, 0.65, 0.05), rot=(0, -15, 0)))
        heli = (hx, 0.0, F)
        x0, x1, w = deck_rect(L, B, -L / 2 + 1.5, sx0 - 2.0, 0.8, 0.28, 0.92)
        decks = [(x0, x1, -w, w, F)]
        crew = [(x1 - 1.0, -(w - 0.5), F), (x1 - 3.5, -(w - 0.5), F)]
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 1.2), (sx0 - 7.5, sx1, -B / 2, B / 2, F + 9.0), ((sx0 + sx1) / 2 - 1.5, (sx0 + sx1) / 2 + 1.5, -1.5, 1.5, F + 14.5)]
        top = F + 14.5
        words = f"the {hull_name} {kind}"
    elif kind == "coast guard cutter":
        L = float(rng.uniform(42, 56)); B = round(L * 0.17, 1); F, D = 3.2, 2.8
        hull_name = "white"
        p.append(hull_mesh(L, B, F, D, HULL_COLORS["white"], bow=0.30, stern_w=0.85, sheer=1.6))
        for side in (-1, 1):  # diagonal racing stripe
            p.append(box((0.22 * L, side * (_half_breadth(L, B, 0.22 * L) + 0.01), F - 0.9), (4.0, 0.02, 2.2), (0.85, 0.15, 0.08), rot=(0, -25, 0), col=False))
            p.append(box((0.22 * L - 2.4, side * (_half_breadth(L, B, 0.22 * L - 2.4) + 0.01), F - 0.9), (1.0, 0.02, 2.2), (0.10, 0.25, 0.55), rot=(0, -25, 0), col=False))
        sx0, sx1 = -0.02 * L, 0.28 * L
        p += [box(((sx0 + sx1) / 2, 0, F + 1.6), (sx1 - sx0, B - 2.0, 3.2), WHITE), box(((sx0 + sx1) / 2 + 1, 0, F + 4.4), (sx1 - sx0 - 3, B - 3.2, 2.4), WHITE),
              cyl(((sx0 + sx1) / 2, 0, F + 8.5), 0.2, 5.0, (0.9, 0.9, 0.9), col=False)]
        _windows(p, sx0 + 2, sx1 - 2, (B - 3.2) / 2, F + 4.6, 0.8)
        hx = -0.32 * L
        p.append(box((hx, 0, F + 0.35), (0.30 * L, B - 1.6, 0.7), (0.30, 0.32, 0.34)))                    # raised flight deck
        _pad_marking(p, hx, 0.0, F + 0.7, r=min(4.6, B / 2 - 1.6))
        heli = (hx, 0.0, F + 0.7)
        decks = [(hx - 0.15 * L + 0.3, hx + 0.15 * L - 0.3, -B / 2 + 1.1, B / 2 - 1.1, F + 0.7)]
        crew = [(hx + 0.15 * L - 1.0, B / 2 - 1.6, F + 0.7)]
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 1.0), (sx0, sx1, -B / 2, B / 2, F + 6.0), ((sx0 + sx1) / 2 - 1.2, (sx0 + sx1) / 2 + 1.2, -1.2, 1.2, F + 11.0)]
        top = F + 11.0
        words = "the white coast-guard cutter with the red stripe"
    elif kind == "tug":
        L = float(rng.uniform(24, 32)); B = round(L * 0.36, 1); F, D = 2.2, 3.0
        hull_name = _pick(rng, ["black", "red", "dark blue"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.30, stern_w=0.85, sheer=1.0))
        top_col = _pick(rng, [(0.85, 0.15, 0.08), WHITE, (0.95, 0.70, 0.05)])
        p += [box((0.05 * L, 0, F + 1.5), (8, B - 2.5, 3.0), top_col), box((0.08 * L, 0, F + 4.2), (4.5, B - 4, 2.4), top_col),
              box((0.08 * L + 2.26, 0, F + 4.4), (0.02, B - 4.2, 0.8), (0.06, 0.08, 0.1), col=False),
              box((-0.1 * L, 0, F + 5.5), (1.6, 1.4, 5), (0.15, 0.15, 0.15))]
        for x in np.linspace(-L / 2 + 2, L / 2 - 4, 7):  # tyre fenders
            for side in (-1, 1):
                p.append(cyl((float(x), side * (_half_breadth(L, B, float(x)) + 0.2), F - 0.6), 0.45, 0.35, "tyre", axis="Y", col=False))
        x0, x1, w = deck_rect(L, B, -L / 2 + 1.5, -0.12 * L, 0.8, 0.30, 0.85)
        decks = [(x0, x1, -w, w, F)]
        crew = [((x0 + x1) / 2, 0.5, F)]
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 1), (0.05 * L - 4, 0.05 * L + 4, -B / 2, B / 2, F + 8.0)]
        top = F + 8.0
        words = f"the {hull_name} harbour tug"
    elif kind == "motor yacht":
        L = float(rng.uniform(18, 30)); B = round(L * 0.24, 1); F, D = 2.4, 1.6
        hull_name = "white"
        p.append(hull_mesh(L, B, F, D, WHITE, bow=0.38, stern_w=0.9, sheer=0.6))
        trim = _pick(rng, [(0.05, 0.10, 0.25), (0.08, 0.08, 0.09), (0.55, 0.55, 0.57)])
        p += [box((0.0, 0, F + 1.1), (0.45 * L, B - 1.6, 2.2), WHITE), box((-0.02 * L, 0, F + 3.0), (0.30 * L, B - 2.6, 1.6), WHITE),
              box((0.0, 0, F + 1.4), (0.44 * L, B - 1.55, 0.8), trim, col=False), box((-0.02 * L, 0, F + 3.1), (0.29 * L, B - 2.55, 0.6), trim, col=False),
              box((0.0, -(B / 2 + 0.01), F - 0.7), (0.6 * L, 0.02, 0.35), trim, col=False), box((0.0, B / 2 + 0.01, F - 0.7), (0.6 * L, 0.02, 0.35), trim, col=False),
              box((0.34 * L, 0, F + 0.02), (0.2 * L, B * 0.5, 0.04), TEAK, col=False), box((-0.40 * L, 0, F + 0.02), (0.15 * L, B - 1.5, 0.04), TEAK, col=False)]
        x0, x1, w = deck_rect(L, B, -0.47 * L, -0.30 * L, 0.6, 0.38, 0.9)
        decks = [(x0, x1, -w, w, F)]
        crew = []
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 4.2)]
        top = F + 4.2
        words = "the sleek white motor yacht"
    elif kind == "sailboat":
        L = float(rng.uniform(9, 14)); B = round(L * 0.33, 1); F, D = 1.2, 1.8
        hull_name = _pick(rng, ["white", "white", "dark blue", "red", "green"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.40, stern_w=0.75, sheer=0.3))
        mh = 1.35 * L
        p += [box((0.0, 0, F + 0.35), (0.35 * L, B * 0.45, 0.7), WHITE), cyl((0.05 * L, 0, F + mh / 2), 0.08, mh, (0.85, 0.85, 0.85)),
              cyl((-0.15 * L, 0, F + 1.6), 0.06, 0.45 * L, (0.85, 0.85, 0.85), axis="X", col=False)]
        sail = _pick(rng, ["furled", "furled", "up"])
        if sail == "furled":
            p.append(cap((-0.15 * L, 0, F + 1.75), 0.14, 0.42 * L, _pick(rng, [(0.10, 0.20, 0.50), (0.85, 0.85, 0.82)]), axis="X", col=False))
        else:  # mainsail: a thin vertical triangle from boom to masthead
            h = mh - 2.0
            pts = [(0.05 * L, 0.03, F + 1.7), (0.05 * L, 0.03, F + 1.7 + h), (-0.37 * L, 0.03, F + 1.7),
                   (0.05 * L, -0.03, F + 1.7), (-0.37 * L, -0.03, F + 1.7), (0.05 * L, -0.03, F + 1.7 + h)]
            p.append(dict(shape="mesh", c=(0, 0, 0), pts=pts, idx=[0, 1, 2, 3, 4, 5], cnt=[3, 3], rot=(0, 0, 0), mat=(0.95, 0.95, 0.92), col=False))
        decks = []
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 1.5), (0.05 * L - 1, 0.05 * L + 1, -1, 1, F + mh)]
        top = F + mh
        words = f"the {'white ' if hull_name == 'white' else hull_name + ' '}sailing yacht with its {'sail up' if sail == 'up' else 'sails furled'}"
    elif kind == "motorboat":
        L = float(rng.uniform(5.5, 8.5)); B = round(L * 0.36, 1); F, D = 0.9, 0.5
        hull_name = _pick(rng, ["white", "white", "light blue", "red", "grey"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.40, stern_w=0.85, sheer=0.3))
        p += [box((0.1 * L, 0, F + 0.5), (0.7, 1.0, 1.0), WHITE), box((0.1 * L + 0.4, 0, F + 1.1), (0.05, 1.0, 0.35), (0.3, 0.4, 0.45), col=False),
              box((-L / 2 - 0.25, 0, 0.3), (0.5, 0.45, 1.0), (0.12, 0.12, 0.13), col=False)]
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 1.4)]
        top = F + 1.4
        words = f"the small {hull_name} motorboat"
    elif kind == "rowing boat":
        L = float(rng.uniform(3.2, 4.5)); B = round(L * 0.36, 2); F, D = 0.55, 0.25
        hull_name = _pick(rng, ["white", "green", "light blue", "red", "yellow"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.40, stern_w=0.7, sheer=0.1, n=10))
        for x in (-0.25 * L, 0.15 * L):
            p.append(box((x, 0, F - 0.15), (0.25, B - 0.2, 0.04), (0.55, 0.38, 0.2), col=False))
        for side in (-1, 1):
            p.append(cyl((0.0, side * (B / 2 + 0.6), F), 0.03, 2.4, (0.55, 0.38, 0.2), axis="Y", rot=(0, 0, side * 70), col=False))
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, F + 1.5)]
        top = F + 1.5
        words = f"the small {hull_name} rowing boat"
    elif kind == "kayak":
        L = float(rng.uniform(3.6, 5.0)); B = 0.62; F, D = 0.25, 0.15
        hull_name = _pick(rng, ["yellow", "orange", "red", "light blue", "green"])
        p.append(hull_mesh(L, B, F, D, HULL_COLORS[hull_name], bow=0.45, stern_w=0.2, sheer=0.05, n=10))
        boxes = [(-L / 2, L / 2, -B / 2, B / 2, 1.3)]
        top = 1.3
        words = f"the {hull_name} kayak"
    else:
        raise ValueError(kind)
    return dict(kind=kind, hull=hull_name, words=words, L=L, B=B, F=F, parts=p, decks=decks, boxes=boxes, crew=crew,
                helideck=heli, top=top)


def seated_people(rng, kind, L, B, F):
    """People sitting in small open boats (part of the boat's static model, like a figure fixed in place):
    rowing boats get 1-2 anglers with rods, kayaks a paddler, motorboats a driver."""
    p = []
    n = {"rowing boat": int(rng.integers(1, 3)), "kayak": 1, "motorboat": int(rng.integers(1, 3)), "sailboat": int(rng.integers(1, 4))}.get(kind, 0)
    for k in range(n):
        x = (0.15 - 0.4 * k) * L * (0.6 if kind != "kayak" else 0.0)
        skin, shirt = _pick(rng, SKINS), _pick(rng, OILSKIN if kind == "rowing boat" else SHIRTS)
        zb = F - 0.1 if kind != "kayak" else 0.18
        p += [cap((x, 0, zb + 0.35), 0.16, 0.3, shirt, axis="Z", col=False), sph((x, 0, zb + 0.78), 0.11, skin),
              sph((x - 0.01, 0, zb + 0.82), 0.112, _pick(rng, HAIRS), scale=(1, 1, 0.75))]
        if kind == "rowing boat":
            ang = math.radians(float(rng.uniform(25, 40)))
            side = 1 if k % 2 == 0 else -1
            Lr = 2.6
            p.append(cyl((x + 0.1, side * (0.3 + Lr / 2 * math.cos(ang)), zb + 0.5 + Lr / 2 * math.sin(ang)), 0.012, Lr,
                         (0.1, 0.1, 0.1), axis="Y", rot=(side * -math.degrees(ang), 0, 0), col=False))
        if kind == "kayak":
            p.append(cyl((x, 0, zb + 0.55), 0.02, 2.2, (0.2, 0.2, 0.2), axis="Y", rot=(20, 0, 0), col=False))
            for side in (-1, 1):
                p.append(box((x, side * 1.1, zb + 0.55 + side * 0.38), (0.2, 0.05, 0.4), (0.9, 0.9, 0.2), rot=(20, 0, 0), col=False))
    return p, n


# =================================================================================================== buildings
def shed_building(L, W, H, wall, roof, door=True, rng=None):
    """Simple rectangular building (boat shed / warehouse / market hall / office) on a 0.5 m foundation."""
    p = [box((0, 0, H / 2 - 0.25), (L, W, H + 0.5), wall), box((0, 0, H + 0.12), (L + 0.6, W + 0.6, 0.25), roof)]
    if door:
        p.append(box((L / 2 + 0.01, 0, min(3.0, H * 0.45)), (0.02, min(W * 0.6, 5.0), min(6.0, H * 0.9)), (0.25, 0.25, 0.27), col=False))
    for k in range(int(L // 4)):
        x = -L / 2 + 2 + 4 * k
        for side in (-1, 1):
            p.append(box((x, side * (W / 2 + 0.01), H * 0.62), (1.4, 0.02, 1.0), (0.06, 0.08, 0.10), col=False))
    return p


def open_hall(L, W, H, roof):
    """Fish-market hall: posts and a pitched roof, open sides."""
    p = []
    for x in np.linspace(-L / 2, L / 2, max(2, int(L / 5) + 1)):
        for side in (-1, 1):
            p.append(cyl((float(x), side * W / 2, H / 2 - 0.25), 0.15, H + 0.5, (0.55, 0.55, 0.57)))
    p.append(box((0, -W / 4, H + 0.6), (L + 1.0, W / 2 + 1.0, 0.18), roof, rot=(-12, 0, 0)))
    p.append(box((0, W / 4, H + 0.6), (L + 1.0, W / 2 + 1.0, 0.18), roof, rot=(12, 0, 0)))
    return p


def lighthouse(rng):
    """Tapered lighthouse tower with gallery, lamp room and dome; colour scheme varies."""
    scheme = _pick(rng, [("red and white", (0.80, 0.08, 0.06), WHITE), ("black and white", (0.06, 0.06, 0.07), WHITE),
                         ("white", WHITE, WHITE), ("yellow and black", (0.90, 0.72, 0.10), (0.06, 0.06, 0.07))])
    name, c1, c2 = scheme
    H = float(rng.uniform(18, 28))
    p = []
    n = 6
    for k in range(n):
        z0 = -0.5 + (H + 0.5) * k / n
        r = 3.0 - 1.1 * (k + 0.5) / n
        p.append(cyl((0, 0, z0 + (H + 0.5) / n / 2), r, (H + 0.5) / n + 0.01, c1 if k % 2 == 0 else c2))
    p += [cyl((0, 0, H + 0.15), 2.6, 0.3, (0.15, 0.15, 0.16)), cyl((0, 0, H + 1.4), 1.3, 2.2, (0.75, 0.85, 0.80)),
          cyl((0, 0, H + 2.6), 1.5, 0.25, (0.10, 0.10, 0.11)), sph((0, 0, H + 2.7), 1.25, (0.10, 0.10, 0.11), scale=(1, 1, 0.7), col=True),
          cyl((0, 0, H + 3.9), 0.06, 1.2, (0.2, 0.2, 0.2), col=False)]
    for a in np.linspace(0, 2 * math.pi, 18, endpoint=False):
        p.append(cyl((2.45 * math.cos(a), 2.45 * math.sin(a), H + 0.75), 0.03, 0.9, (0.15, 0.15, 0.16), col=False))
    return p, H + 4.5, f"the {name} lighthouse"
