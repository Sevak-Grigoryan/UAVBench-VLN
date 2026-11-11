"""
Forest Easy v2 (2026-10-11): 100 single-task missions, each in its own version of the valley.

User brief: "maintain only easy part ... generate for example 100 simulations for easy forest, add
realistic new objects ... house, some animals cow, horse, dog, farm, garage etc. all in different places,
each simulation must be different from another, fix the river part".

For simulation k (1..100) a seeded generator lays out
  * 2-3 homesteads: house (+ garage, gravel driveway, car, often a dog, sometimes a shed and a woodpile)
  * 1-2 farms: barn, silo, fenced paddock with cows / horses / sheep, a water trough, round hay bales,
    sometimes a tractor
  * 1-2 herds grazing in open meadows
at different places every time (dry ground >= 30 m from the river, clear of the pads and landmarks, on
gentle slopes), clears the trees around each of them, and then picks ONE task about one of those objects
(or the river): circle it, inspect it (hover facing it), fly over it with the camera down, or follow the
river -- Easy = 1 task (missions.DIFFICULTY_RULE). Start pad, landing pad, heading, orbit radius / height /
direction all vary; the generator rejects any mission that repeats an earlier one.

Everything here is pure Python: the layout and mission for a given id are rebuilt identically by the
flight job, every render slice and the offline checks.
"""
import math

import numpy as np

import props
from missions import PADS, difficulty_from_tasks, tasks

N_SIMS = 100
PREFIX = "easy_v2_"
SEED = 5200
REGION_X, REGION_Y = (-185.0, 245.0), (-165.0, 165.0)
RIVER_MIN_M = 30.0  # buildings / animals keep this far from the river centreline (dry valley floor)
PAD_MIN_M = 16.0  # from a pad centre to the nearest object (pads have an 11 m vegetation-free clearing)
LANDMARK_MIN_M = 22.0
LANDMARKS_XY = {"footbridge": (72.0, None), "cairn": (180.0, -55.0), "lookout": (130.0, 85.0), "marker": (-55.0, -95.0),
                "rockslide": (218.0, 75.0)}
PAD_WORDS = {k: v[1] for k, v in PADS.items()}
NUM = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
       11: "eleven", 12: "twelve"}


# Regenerated layouts (attempt k instead of 0) for the missions whose first layout / flight path ran into the
# 26-45 m valley-wall cliffs (flight easy_v2_036 orbited into one, 2026-10-11). Found by tools/easy_v2_obstacles.py;
# every other mission keeps attempt 0, i.e. exactly the layout it was flown and rendered with.
ATTEMPT = {6: 1, 8: 3, 9: 1, 14: 2, 20: 2, 22: 1, 23: 1, 25: 2, 27: 1, 28: 1, 31: 1, 32: 2, 34: 1, 36: 1, 37: 1, 40: 4, 41: 3,
           45: 1, 46: 3, 50: 1, 53: 2, 54: 1, 56: 1, 57: 1, 58: 1, 63: 4, 64: 1, 65: 1, 76: 1, 79: 1, 82: 1, 86: 2, 92: 1,
           93: 1, 94: 1, 100: 2}


def mission_ids():
    return [f"{PREFIX}{k:03d}" for k in range(1, N_SIMS + 1)]


def index_of(mission_id):
    return int(mission_id[len(PREFIX):])


# ---- geometry helpers --------------------------------------------------------------------------------
def _rot(yaw, a, b):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * a - s * b, s * a + c * b


class Ctx:
    def __init__(self, nv, fs, seed=SEED):
        self.nv, self.fs, self.seed = nv, fs, seed
        self.vg = lambda x, y: fs.visible_ground(nv, x, y, seed)
        self.lm = {k: (v[0], nv.river_y(v[0]) if v[1] is None else v[1]) for k, v in LANDMARKS_XY.items()}

    def river_d(self, x, y):
        return abs(y - self.nv.river_y(x))

    def side(self, x, y):
        return "north" if y > self.nv.river_y(x) else "south"


def _circles_ok(circles, others):
    for (x, y, r) in circles:
        for (ox, oy, orad) in others:
            if math.hypot(x - ox, y - oy) < r + orad:
                return False
    return True


def _site_ok(ctx, circles, taken):
    for (x, y, r) in circles:
        if not (REGION_X[0] <= x <= REGION_X[1] and REGION_Y[0] <= y <= REGION_Y[1]):
            return False
        if ctx.river_d(x, y) < RIVER_MIN_M + r:
            return False
        for (px, py), _ in PADS.values():
            if math.hypot(x - px, y - py) < PAD_MIN_M + r:
                return False
        for lx, ly in ctx.lm.values():
            if math.hypot(x - lx, y - ly) < LANDMARK_MIN_M + r:
                return False
    return _circles_ok(circles, taken)


# ---- groups -------------------------------------------------------------------------------------------
def _local_gl(ctx, ox, oy, yaw):
    return lambda a, b: ctx.vg(*(lambda p: (ox + p[0], oy + p[1]))(_rot(yaw, a, b)))


def _movable(ctx, rng, kind_fn, x, y, yaw, name, max_tilt=12.0):
    prop = kind_fn()
    M, resid, tilt = props.rest_pose(prop, x, y, yaw, ctx.vg)
    st = props.check_static(prop, tilt)
    if resid > 0.03 or tilt > max_tilt or not st["stable"]:
        return None
    return dict(name=name, kind=prop["kind"], xy=(x, y), yaw=yaw, matrix=M, prop=prop, static_check=st,
                support_residual_m=round(resid, 4), r=prop["length"] / 2 + 0.3, top_z=M[3][2] + prop["height"])


def _static(ctx, prop_fn, x, y, yaw, name, max_range=1.6):
    gl = _local_gl(ctx, x, y, yaw)
    prop = prop_fn(gl)
    gmin, gmax = prop["ground_minmax"]
    if gmax - gmin > max_range:
        return None
    fx, fy = prop["footprint"]
    return dict(name=name, kind=prop["kind"], xy=(x, y), yaw=yaw, prop=prop, r=0.5 * math.hypot(fx, fy),
                top_z=prop["top_z"], ground_range_m=round(gmax - gmin, 3))


def homestead(ctx, rng, gid):
    L, W = float(rng.uniform(8.0, 12.0)), float(rng.uniform(6.5, 8.5))
    storeys = int(rng.choice([1, 1, 2]))
    wall = str(rng.choice(list(props.WALLS)))
    roof = str(rng.choice(list(props.ROOFS)))
    pitch = float(rng.uniform(28, 40))
    yaw = float(rng.uniform(-math.pi, math.pi))
    double = bool(rng.random() < 0.4)
    gW = 6.4 if double else 3.6
    # garage beside the house (same frontage, door facing the front), driveway out to the front
    side = 1 if rng.random() < 0.5 else -1
    gx_l = side * (L / 2 + 3.5 + gW / 2)
    gy_l = -W / 2 + 3.1
    pieces = [("house", 0.0, 0.0, 0.0), ("garage", gx_l, gy_l, -math.pi / 2)]
    has_dog = rng.random() < 0.7
    has_shed = rng.random() < 0.5
    car_color = str(rng.choice(list(props.CAR_COLORS)))
    wall_g = wall if rng.random() < 0.6 else str(rng.choice(list(props.WALLS)))
    return dict(type="homestead", gid=gid, L=L, W=W, storeys=storeys, wall=wall, roof=roof, pitch=pitch, yaw=yaw,
                double=double, gW=gW, gx=gx_l, gy=gy_l, has_dog=has_dog, has_shed=has_shed, car_color=car_color,
                wall_g=wall_g, chimney=bool(rng.random() < 0.6), dog_coat=str(rng.choice(list(props.DOG_COATS))),
                radius=max(L, W) / 2 + gW + 12.0)


def _realise_homestead(ctx, rng, g, x, y):
    yaw = g["yaw"]
    objs = []
    w = lambda a, b: (lambda p: (x + p[0], y + p[1]))(_rot(yaw, a, b))
    h = _static(ctx, lambda gl: props.house(gl, g["wall"], g["roof"], g["storeys"], g["L"], g["W"], g["pitch"], g["chimney"], rng),
                x, y, yaw, f"house_{g['gid']}")
    if h is None:
        return None
    objs.append(h)
    gx, gy = w(g["gx"], g["gy"])
    gyaw = yaw - math.pi / 2
    gar = _static(ctx, lambda gl: props.garage(gl, g["wall_g"], g["roof"], g["double"], rng), gx, gy, gyaw, f"garage_{g['gid']}")
    if gar is None:
        return None
    objs.append(gar)
    # driveway from the garage door 10 m out to the front, car parked on it
    d0 = w(g["gx"], g["gy"] - 3.1 - 0.1)
    d1 = w(g["gx"], g["gy"] - 3.1 - 11.0)
    dw = props.driveway(ctx.vg, d0[0], d0[1], d1[0], d1[1], g["gW"] / 2 - 0.3)
    objs.append(dict(name=f"driveway_{g['gid']}", kind="driveway", xy=d0, yaw=0.0, prop=dw, r=0.0, top_z=0.0, world=True))
    cx, cy = w(g["gx"] + (g["gW"] / 4 if g["double"] else 0.0), g["gy"] - 3.1 - 4.2)
    c = _movable(ctx, rng, lambda: props.car(g["car_color"], rng), cx, cy, yaw - math.pi / 2, f"car_{g['gid']}")
    if c is None:
        return None
    objs.append(c)
    if g["has_dog"]:
        dx, dy = w(float(rng.uniform(-2.5, 2.5)), -g["W"] / 2 - float(rng.uniform(3.0, 5.0)))
        d = _movable(ctx, rng, lambda: props.dog(g["dog_coat"], rng), dx, dy, float(rng.uniform(-math.pi, math.pi)), f"dog_{g['gid']}")
        if d is not None:
            objs.append(d)
    if g["has_shed"]:
        sx, sy = w(-g["gx"] * 0.4, g["W"] / 2 + 6.0)
        s = _static(ctx, lambda gl: props.shed(gl, rng), sx, sy, yaw + math.pi / 2, f"shed_{g['gid']}")
        if s is not None:
            objs.append(s)
        wx, wy = w(-g["gx"] * 0.4 + 3.5, g["W"] / 2 + 6.0)
        wp = _static(ctx, lambda gl: props.woodpile(gl, rng), wx, wy, yaw, f"woodpile_{g['gid']}")
        if wp is not None:
            objs.append(wp)
    return objs


def farm(ctx, rng, gid):
    animal = str(rng.choice(["cow", "cow", "horse", "sheep"]))
    n = int(rng.integers(3, 8)) if animal != "sheep" else int(rng.integers(6, 12))
    return dict(type="farm", gid=gid, animal=animal, n=n, yaw=float(rng.uniform(-math.pi, math.pi)),
                PL=float(rng.uniform(24, 32)), PW=float(rng.uniform(18, 24)), bales=int(rng.integers(3, 8)),
                tractor=bool(rng.random() < 0.6), tractor_color=str(rng.choice(["green", "red", "blue"])),
                coat=str(rng.choice(list(props.COW_COATS if animal == "cow" else props.HORSE_COATS if animal == "horse" else ["white"]))),
                mixed=bool(rng.random() < 0.35), radius=34.0)


def _animal_fn(kind, coat, rng):
    grazing = bool(rng.random() < 0.55)
    if kind == "cow":
        return lambda: props.cow(coat, grazing, rng)
    if kind == "horse":
        return lambda: props.horse(coat, grazing, rng)
    return lambda: props.sheep(rng)


def _realise_farm(ctx, rng, g, x, y):
    yaw = g["yaw"]
    w = lambda a, b: (lambda p: (x + p[0], y + p[1]))(_rot(yaw, a, b))
    objs = []
    b = _static(ctx, lambda gl: props.barn(gl, rng), *w(-12.0, 0.0), yaw, f"barn_{g['gid']}")
    if b is None:
        return None
    objs.append(b)
    BL, BW = b["prop"]["L"], b["prop"]["W"]
    s = _static(ctx, lambda gl: props.silo(gl, rng), *w(-12.0 - BL / 2 + 3.0, BW / 2 + 4.0), 0.0, f"silo_{g['gid']}")
    if s is None:
        return None
    objs.append(s)
    # paddock in front of the barn doors
    px0 = -12.0 + BL / 2 + 5.0
    PL, PW = g["PL"], g["PW"]
    corners_l = [(px0, -PW / 2), (px0 + PL, -PW / 2), (px0 + PL, PW / 2), (px0, PW / 2)]
    corners = [w(a, bb) for a, bb in corners_l]
    f = props.fence(ctx.vg, corners, gate_side=3)
    objs.append(dict(name=f"fence_{g['gid']}", kind="fence", xy=w(px0 + PL / 2, 0), yaw=yaw, prop=f, r=0.0, top_z=0.0, world=True))
    placed = []
    kinds = [g["animal"]] * g["n"]
    coats = [g["coat"]] * g["n"]
    if g["mixed"] and g["animal"] == "cow":
        coats = [str(rng.choice(list(props.COW_COATS))) for _ in kinds]
    if g["mixed"] and g["animal"] == "horse":
        coats = [str(rng.choice(list(props.HORSE_COATS))) for _ in kinds]
    for i, (k, c) in enumerate(zip(kinds, coats)):
        for _ in range(60):
            a, bb = float(rng.uniform(px0 + 3.0, px0 + PL - 3.0)), float(rng.uniform(-PW / 2 + 2.6, PW / 2 - 2.6))
            ax, ay = w(a, bb)
            if any(math.hypot(ax - q[0], ay - q[1]) < (3.4 if k != "sheep" else 1.8) for q in placed):
                continue
            m = _movable(ctx, rng, _animal_fn(k, c, rng), ax, ay, float(rng.uniform(-math.pi, math.pi)), f"{k}_{g['gid']}_{i}")
            if m is not None:
                objs.append(m)
                placed.append((ax, ay))
                break
    tx, ty = w(px0 + PL - 4.0, PW / 2 - 1.6)
    t = _movable(ctx, rng, lambda: props.trough(rng), tx, ty, yaw, f"trough_{g['gid']}")
    if t is not None and all(math.hypot(tx - q[0], ty - q[1]) > 3.0 for q in placed):
        objs.append(t)
    for i in range(g["bales"]):
        bx, by = w(-12.0 - BL / 2 + 2.0 + (i % 4) * 1.75, -BW / 2 - 3.0 - (i // 4) * 1.75)
        m = _movable(ctx, rng, lambda: props.round_bale(rng), bx, by, 0.0, f"bale_{g['gid']}_{i}")
        if m is not None:
            objs.append(m)
    if g["tractor"]:
        tx, ty = w(-12.0 + BL / 2 + 1.0, -PW / 2 - 6.0)
        m = _movable(ctx, rng, lambda: props.tractor(g["tractor_color"], rng), tx, ty, yaw + math.pi / 2 * float(rng.choice([1, -1])),
                     f"tractor_{g['gid']}")
        if m is not None:
            objs.append(m)
    n_animals = sum(1 for o in objs if o["kind"] == g["animal"])
    if n_animals < 2:
        return None
    g["n_placed"] = n_animals
    g["coats_placed"] = sorted({c for c, o in zip(coats, objs) if True})
    return objs


def herd(ctx, rng, gid):
    animal = str(rng.choice(["cow", "horse", "horse", "sheep"]))
    n = int(rng.integers(3, 7)) if animal != "sheep" else int(rng.integers(5, 11))
    coat = str(rng.choice(list(props.COW_COATS if animal == "cow" else props.HORSE_COATS if animal == "horse" else ["white"])))
    return dict(type="herd", gid=gid, animal=animal, n=n, coat=coat, mixed=bool(rng.random() < 0.4), radius=16.0,
                yaw=float(rng.uniform(-math.pi, math.pi)))


def _realise_herd(ctx, rng, g, x, y):
    objs, placed = [], []
    for i in range(g["n"]):
        coat = g["coat"] if not g["mixed"] or g["animal"] == "sheep" else str(
            rng.choice(list(props.COW_COATS if g["animal"] == "cow" else props.HORSE_COATS)))
        for _ in range(60):
            r, a = 9.0 * math.sqrt(rng.random()), rng.uniform(-math.pi, math.pi)
            ax, ay = x + r * math.cos(a), y + r * math.sin(a)
            if any(math.hypot(ax - q[0], ay - q[1]) < (3.4 if g["animal"] != "sheep" else 1.8) for q in placed):
                continue
            # herds face roughly the same way, as grazing animals do
            m = _movable(ctx, rng, _animal_fn(g["animal"], coat, rng), ax, ay, g["yaw"] + float(rng.normal(0, 0.6)),
                         f"{g['animal']}_{g['gid']}_{i}")
            if m is not None:
                objs.append(m)
                placed.append((ax, ay))
                break
    if len(objs) < 3:
        return None
    g["n_placed"] = len(objs)
    return objs


def _group_circle(objs, x, y):
    r = 0.0
    for o in objs:
        if o.get("world") and o["kind"] in ("fence", "driveway"):
            pts = [p["c"] for p in o["prop"]["parts"] if "c" in p] or [q for p in o["prop"]["parts"] for q in p.get("pts", [])]
            for q in pts:
                r = max(r, math.hypot(q[0] - x, q[1] - y) + 0.5)
        else:
            r = max(r, math.hypot(o["xy"][0] - x, o["xy"][1] - y) + o["r"])
    return r


def layout(idx, nv, fs, seed=SEED, attempt=None):
    """Deterministic scene layout for simulation idx (1..100)."""
    ctx = Ctx(nv, fs, seed)
    attempt = ATTEMPT.get(idx, 0) if attempt is None else attempt
    rng = np.random.default_rng(1_000_003 * idx + 17 + 7_919 * attempt)
    n_home = int(rng.integers(2, 4))
    n_farm = int(rng.integers(1, 3))
    n_herd = int(rng.integers(1, 3))
    plan = [("homestead", homestead, _realise_homestead)] * n_home + [("farm", farm, _realise_farm)] * n_farm + \
           [("herd", herd, _realise_herd)] * n_herd
    groups, taken = [], []
    for gi, (typ, make, realise) in enumerate(plan):
        for attempt in range(400):
            g = make(ctx, rng, gi + 1)
            x, y = float(rng.uniform(*REGION_X)), float(rng.uniform(*REGION_Y))
            if not _site_ok(ctx, [(x, y, g["radius"])], taken):
                continue
            objs = realise(ctx, rng, g, x, y)
            if objs is None:
                continue
            # exact checks on the realised objects: every object on dry land, clear of pads/landmarks
            circles = [(o["xy"][0], o["xy"][1], max(o["r"], 1.0)) for o in objs if not o.get("world")]
            if not _site_ok(ctx, circles, taken):
                continue
            R = _group_circle(objs, x, y)
            cx, cy = x, y
            if g["type"] == "farm":  # centre of barn + paddock
                pts = [o["xy"] for o in objs if not o.get("world")]
                cx, cy = float(np.mean([p[0] for p in pts])), float(np.mean([p[1] for p in pts]))
                R = _group_circle(objs, cx, cy)
            g.update(xy=(cx, cy), clear_r=R + 5.0, objects=objs)
            groups.append(g)
            taken.append((cx, cy, R + 8.0))
            break
    return dict(index=idx, seed=seed, groups=groups)


# ---- descriptions -------------------------------------------------------------------------------------
def _animal_words(kind, n, coat=None, mixed=False):
    plural = {"cow": "cows", "horse": "horses", "sheep": "sheep"}[kind]
    if kind == "sheep":
        return f"{NUM.get(n, n)} sheep"
    if mixed or coat is None:
        return f"{NUM.get(n, n)} {plural}"
    return f"{NUM.get(n, n)} {coat} {plural}"


def describe(ctx, g, focus):
    """(long description, short name) of the object a mission is about."""
    x, y = g["xy"]
    where = f"{ctx.side(x, y)} of the river"
    if g["type"] == "homestead":
        h = g["objects"][0]
        storey = "two-storey " if g["storeys"] == 2 else ""
        wall = g["wall"]
        house = "log cabin" if wall == "log" else f"{wall} {storey}house".replace("red brick house", "red-brick house")
        if wall == "log":
            house = f"{storey}log cabin"
        long = f"the {house} with the {g['roof']} roof (a garage and a parked car next to it), {where}"
        return long, f"the {house}"
    if g["type"] == "farm":
        a = _animal_words(g["animal"], g["n_placed"], g["coat"], g["mixed"])
        if focus == "barn":
            return f"the red barn with the grain silo beside it, {where}", "the red barn"
        return f"the fenced paddock with {a} next to the red barn, {where}", "the paddock"
    a = _animal_words(g["animal"], g["n_placed"], g["coat"], g["mixed"])
    verb = "grazing"
    return f"the {a} {verb} in an open meadow {where}", f"the {'flock' if g['animal'] == 'sheep' else 'herd'}"


def _target_point(g, focus):
    objs = g["objects"]
    if g["type"] == "homestead":
        h = objs[0]
        return h["xy"], h["top_z"], max(h["r"], 9.0)
    if g["type"] == "farm":
        if focus == "barn":
            b = objs[0]
            return b["xy"], b["top_z"], b["r"] + 4.0
        an = [o for o in objs if o["kind"] == g["animal"]]
        xs, ys = [o["xy"][0] for o in an], [o["xy"][1] for o in an]
        return (float(np.mean(xs)), float(np.mean(ys))), max(o["top_z"] for o in an), 12.0
    xs, ys = [o["xy"][0] for o in objs], [o["xy"][1] for o in objs]
    return (float(np.mean(xs)), float(np.mean(ys))), max(o["top_z"] for o in objs), 10.0


def _to_pad(xy, pad):
    """Transit to the landing pad at cruise speed, then land (pilot's land beat only crabs at <= 2 m/s)."""
    return [dict(type="goto", xy=xy, agl=22.0, speed=0.85, radius=4.0, label=f"fly to {pad} pad"),
            dict(type="land", xy=xy, label=f"land {pad}", pad=pad)]


# ---- missions -----------------------------------------------------------------------------------------
TASK_PLAN = ["circle"] * 45 + ["inspect"] * 25 + ["pass_over"] * 20 + ["follow_river"] * 10


def _task_type(idx):
    order = np.random.default_rng(424242).permutation(len(TASK_PLAN))
    return TASK_PLAN[int(order[idx - 1])]


def mission(idx, lay, nv, fs, ground, seed=SEED, attempt=None):
    """One Easy mission for layout `lay`. Returns (mission dict, landmark dict for its target)."""
    ctx = Ctx(nv, fs, seed)
    attempt = ATTEMPT.get(idx, 0) if attempt is None else attempt
    rng = np.random.default_rng(7_000_001 * idx + 3 + 104_729 * attempt)
    task = _task_type(idx)
    P = {k: v[0] for k, v in PADS.items()}
    beats = [dict(type="takeoff", agl=float(rng.choice([20.0, 22.0, 25.0])), label="take off")]
    L = {}
    if task == "follow_river":
        for _ in range(500):
            x_a = float(rng.uniform(-160, 200))
            length = float(rng.uniform(110, 190))
            down = bool(rng.random() < 0.5)
            x_b = x_a - length if down else x_a + length
            if not (-175 <= x_b <= 230):
                continue
            a, b = (x_a, nv.river_y(x_a)), (x_b, nv.river_y(x_b))
            sp = [k for k in P if 35 < math.hypot(P[k][0] - a[0], P[k][1] - a[1]) < 120]
            lp = [k for k in P if 35 < math.hypot(P[k][0] - b[0], P[k][1] - b[1]) < 120]
            if not sp or not lp:
                continue
            start = str(rng.choice(sp))
            land = str(rng.choice([k for k in lp if k != start] or lp))
            if land == start:
                continue
            break
        agl = float(rng.choice([15.0, 16.0, 18.0]))
        n = max(1, int(abs(x_b - x_a) // 35))
        pts = [(x_a + (x_b - x_a) * k / n, nv.river_y(x_a + (x_b - x_a) * k / n)) for k in range(n + 1)]
        beats += [dict(type="goto", xy=p, agl=agl, speed=0.7, radius=6.0, label="follow river", river=True) for p in pts]
        beats += _to_pad(P[land], land)
        way = "downstream (west)" if down else "upstream (east)"
        text = (f"Take off from {PAD_WORDS[start]}, fly to the river and follow it {way} at about {agl:.0f} m above the "
                f"water for roughly {abs(x_b - x_a):.0f} m, then land at {PAD_WORDS[land]}.")
        focus_desc = None
    else:
        cand = []
        for g in lay["groups"]:
            if g["type"] == "homestead":
                cand.append((g, "house"))
            elif g["type"] == "farm":
                cand += [(g, "barn"), (g, "paddock")]
            else:
                cand.append((g, "herd"))
        g, focus = cand[int(rng.integers(len(cand)))]
        (tx, ty), top_z, size_r = _target_point(g, focus)
        long, short = describe(ctx, g, focus)
        name = f"{focus}_{g['gid']}"
        L[name] = dict(xy=(tx, ty), top_z=top_z, desc=long, short=short)
        pads_by = sorted(P, key=lambda k: math.hypot(P[k][0] - tx, P[k][1] - ty))
        near = [k for k in pads_by if 45 < math.hypot(P[k][0] - tx, P[k][1] - ty) < 190] or pads_by[:3]
        start = str(rng.choice(near[:4]))
        land = str(rng.choice([k for k in near[:5] if k != start] or [k for k in pads_by if k != start][:2]))
        g_t = ground(tx, ty)
        if task == "circle":
            radius = float(round(size_r + rng.uniform(8.0, 13.0)))
            ring = max(ground(tx + radius * math.cos(a), ty + radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 24))
            z = max(top_z + float(rng.uniform(7.0, 12.0)), ring + 16.0, g_t + 16.0)
            d = int(rng.choice([1, -1]))
            a = math.atan2(P[start][1] - ty, P[start][0] - tx)
            beats.append(dict(type="goto", xy=(tx + radius * math.cos(a), ty + radius * math.sin(a)), agl=25.0, speed=0.8,
                              radius=3.0, label=f"fly to {name}", landmark=name, look_at=(tx, ty, top_z)))
            beats.append(dict(type="orbit", center=(tx, ty, top_z), radius=radius, z=z, laps=1.0, dir=d, speed=4.0,
                              label=f"orbit {name}", landmark=name))
            sense = "counter-clockwise" if d > 0 else "clockwise"
            text = (f"Take off from {PAD_WORDS[start]}, fly to {long} and circle it once {sense} at about {radius:.0f} m, "
                    f"keeping the camera on it, then land at {PAD_WORDS[land]}.")
        elif task == "inspect":
            stand = float(round(size_r + rng.uniform(10.0, 16.0)))
            a = math.atan2(P[start][1] - ty, P[start][0] - tx) + float(rng.uniform(-0.6, 0.6))
            hx, hy = tx + stand * math.cos(a), ty + stand * math.sin(a)
            agl = max(16.0, top_z - ground(hx, hy) + float(rng.uniform(4.0, 8.0)))
            secs = float(rng.choice([5.0, 6.0, 8.0]))
            beats.append(dict(type="goto", xy=(hx, hy), agl=agl, speed=0.75, radius=3.0, label=f"fly to {name}", landmark=name,
                              look_at=(tx, ty, top_z)))
            beats.append(dict(type="hover", s=secs, look_at=(tx, ty, top_z - 1.0), label=f"inspect {name}", landmark=name))
            text = (f"Take off from {PAD_WORDS[start]}, fly to {long}, stop about {stand:.0f} m from it and hover for about "
                    f"{secs:.0f} seconds looking at it, then land at {PAD_WORDS[land]}.")
        else:  # pass_over
            agl = float(rng.choice([22.0, 25.0, 28.0]))
            gim = float(rng.choice([60.0, 65.0, 70.0]))
            beats.append(dict(type="goto", xy=(tx, ty), agl=agl, speed=0.6, radius=4.0, label=f"fly over {name}",
                              landmark=name, gimbal=gim))
            text = (f"Take off from {PAD_WORDS[start]}, fly directly over {long}, with the camera looking down at it, "
                    f"then land at {PAD_WORDS[land]}.")
        beats += _to_pad(P[land], land)
        focus_desc = long
    m = dict(id=f"{PREFIX}{idx:03d}", difficulty="Easy", start=start, start_xy=P[start],
             yaw_deg=float(round(rng.uniform(-180, 180))), beats=beats, instruction=text, task_type=task,
             target_description=focus_desc, layout_index=idx)
    m["tasks"] = tasks(beats)
    if difficulty_from_tasks(len(m["tasks"])) != "Easy":
        raise ValueError(f"{m['id']}: {len(m['tasks'])} tasks")
    return m, L


def signature(m):
    """What makes two missions 'the same' -- used to reject repeats."""
    tgt = next((b.get("landmark") for b in m["beats"] if b.get("landmark")), None)
    return (m["start"], next(b["pad"] for b in m["beats"] if b["type"] == "land"), m["task_type"], tgt,
            round(m["beats"][1]["xy"][0] / 10), round(m["beats"][1]["xy"][1] / 10))


# ---- layout -> JSON-able summary (meta.json) ----------------------------------------------------------
def summary(lay):
    out = []
    for g in lay["groups"]:
        objs = []
        for o in g["objects"]:
            e = {"name": o["name"], "kind": o["kind"]}
            if not o.get("world"):
                e.update(xy=[round(o["xy"][0], 2), round(o["xy"][1], 2)], yaw_deg=round(math.degrees(o["yaw"]), 1),
                         top_z=round(o["top_z"], 2))
            if "static_check" in o:
                e.update(mass_kg=o["prop"]["mass"], rigid_body=True, rest_tilt_deg=o["static_check"]["slope_deg"],
                         stability_margin_m=o["static_check"]["stability_margin_m"],
                         support_residual_m=o["support_residual_m"])
            elif o["kind"] not in ("driveway",):
                e.update(rigid_body=False, static="foundation / posts set into the ground")
            objs.append(e)
        gd = {k: g[k] for k in ("type", "gid") if k in g}
        gd.update(center_xy=[round(g["xy"][0], 2), round(g["xy"][1], 2)], cleared_radius_m=round(g["clear_r"], 1), objects=objs)
        for k in ("animal", "n_placed", "wall", "roof", "storeys", "coat"):
            if k in g:
                gd[k] = g[k]
        out.append(gd)
    return out


# ---- USD: author the layout and clear the vegetation around it ---------------------------------------
def build(stage, nv, lay):
    """Author every prop of the layout under /World/Props and remove vegetation where it would intersect:
    trees / saplings / rocks / logs inside each group's cleared radius (homesteads and farms sit in
    clearings), grass / ferns / shrubs only under buildings and right under animals and vehicles."""
    from pxr import Gf, Usd, UsdGeom

    UsdGeom.Xform.Define(stage, "/World/Props")
    n_parts = 0
    rigid = []
    for g in lay["groups"]:
        gpath = f"/World/Props/G{g['gid']}_{g['type']}"
        UsdGeom.Xform.Define(stage, gpath)
        for o in g["objects"]:
            path = f"{gpath}/{o['name']}"
            if o.get("world"):
                props.author(stage, nv, path, o["prop"])
            elif "matrix" in o:
                props.author(stage, nv, path, o["prop"], root_matrix=o["matrix"])
                rigid.append({"path": path, "name": o["name"], "kind": o["kind"], "mass_kg": o["prop"]["mass"],
                              "pos": [o["matrix"][3][0], o["matrix"][3][1], o["matrix"][3][2]]})
            else:
                props.author(stage, nv, path, o["prop"], root_xy_yaw=(o["xy"][0], o["xy"][1], o["yaw"]))
            n_parts += len(o["prop"]["parts"])
    clear = [(g["xy"][0], g["xy"][1], g["clear_r"]) for g in lay["groups"]]
    small = []
    for g in lay["groups"]:
        for o in g["objects"]:
            if o.get("world"):
                continue
            small.append((o["xy"][0], o["xy"][1], o["r"] + (1.0 if "matrix" not in o else 0.3)))
    removed = {}
    for root in ("/World/Nature", "/World/DenseNature"):
        rp = stage.GetPrimAtPath(root)
        if not rp:
            continue
        for prim in Usd.PrimRange(rp):
            if not prim.IsA(UsdGeom.PointInstancer) or "DistantTrees" in str(prim.GetPath()):
                continue
            pi = UsdGeom.PointInstancer(prim)
            pos = pi.GetPositionsAttr().Get()
            if not pos:
                continue
            P = np.array([[p[0], p[1]] for p in pos])
            zones = small if "Groundcover" in str(prim.GetPath()) else clear
            keep = np.ones(len(P), bool)
            for (x, y, r) in zones:
                keep &= (P[:, 0] - x) ** 2 + (P[:, 1] - y) ** 2 > r * r
            if keep.all():
                continue
            k = np.nonzero(keep)[0].tolist()
            for attr in (pi.GetPositionsAttr(), pi.GetOrientationsAttr(), pi.GetScalesAttr(), pi.GetProtoIndicesAttr()):
                v = attr.Get()
                if v is not None and len(v) == len(P):
                    attr.Set(type(v)([v[i] for i in k]))
            key = str(prim.GetPath()).split("/")[3] if str(prim.GetPath()).count("/") >= 3 else str(prim.GetPath())
            removed[key] = removed.get(key, 0) + int((~keep).sum())
    return {"props_groups": len(lay["groups"]), "props_objects": sum(len(g["objects"]) for g in lay["groups"]),
            "props_primitives": n_parts, "rigid_bodies": rigid, "vegetation_removed": removed}


def make_dynamic(stage, rigid):
    """Flight job only: turn every movable prop into a PhysX rigid body with its real mass, under gravity."""
    from pxr import UsdPhysics

    for r in rigid:
        prim = stage.GetPrimAtPath(r["path"])
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(r["mass_kg"]))
    return len(rigid)
