"""
Coastal v2 -- the Easy Coastal dataset (2026-10-13): a different coast for every simulation.

User brief: "the coastal environment is simple with few objects. Create environment more diverse ... ships in sea
where drone can go turn around or ... sit in the ship, objects like people, fish, hunters [fishermen] or workers
in ship stations; the environments must be maximum realistic and maximum diverse for different simulations".

plan(idx, attempt) is a pure function (numpy RNG, no Isaac Sim) that decides everything:
  * the coastline: meandering shore along x (land at +y), beach / plateau profile, 1-3 cliff headlands;
  * 4-6 shore SITES from: fishing harbour (quay, fish market, moored trawlers, dock workers), container terminal
    (gantry crane, container stacks, berthed container ship), marina (pontoons, yachts, sailboats), sandy beach
    (umbrellas, towels, sunbathers, swimmers, lifeguard tower, kayaks, beach huts, volleyball), fishing pier
    (anglers with rods), boatyard (slipway, boats on stands), lighthouse point (on a headland), coast-guard station,
    seaside village with a promenade;
  * 4-9 vessels at anchor offshore (container ship, tanker, ferry, trawler, research / supply vessel or coast-guard
    cutter with a helicopter deck, tug, motor yacht, sailboats, motorboats, rowing boats with anglers, kayaks,
    historic Dutch sailing ships from Poly Haven), rocky islets, buoys;
  * landing pads on shore + the helideck(s) of ships (the drone can land on a ship);
  * every movable object (people, crates, barrels, vehicles, loungers, kayaks on sand) as a rigid body with mass.
surface_z(x, y, pl) = the top of whatever solid is under (x, y): terrain, quay, pier / pontoon deck, ship deck --
used to set props down, by the pilot (height above ground) and to check the flight against the scene.
"""
import math
import os

import numpy as np

import coastal_props as cp
import props

EXTENT = 650.0          # high-detail terrain half-extent [m]
FAR = 3500.0            # distant terrain / ocean half-extent [m]
SLAB_COLLIDER = os.environ.get("COASTAL_SLAB_COLLIDER", "cube")  # cube | convex | mesh (textured slabs)
GRID = 401              # terrain vertices per side (3.25 m spacing)
SEA_FLOOR = -6.0
QUAY_Z = 2.2
SITE_KINDS = ["fishing harbour", "container terminal", "marina", "beach", "fishing pier", "boatyard",
              "lighthouse point", "coast guard station", "village"]
SITE_WIDTH = {"fishing harbour": 170, "container terminal": 210, "marina": 160, "beach": 190, "fishing pier": 70,
              "boatyard": 120, "lighthouse point": 60, "coast guard station": 110, "village": 150}
OFFSHORE_KINDS = ["container ship", "tanker", "ferry", "trawler", "trawler", "research vessel", "supply vessel",
                  "coast guard cutter", "tug", "motor yacht", "motor yacht", "sailboat", "sailboat", "sailboat",
                  "motorboat", "rowing boat", "rowing boat", "kayak", "dutch ship"]
HELI_KINDS = ("research vessel", "supply vessel", "coast guard cutter")
DUTCH = {"dutch_ship_large_01": (34.3, 8.4, 32.1), "dutch_ship_large_02": (34.3, 8.5, 32.1),
         "dutch_ship_medium": (24.1, 6.8, 24.7), "ship_pinnace": (42.0, 14.2, 34.2)}  # length (model y), beam, height


def tbox(c, s, tex, tile):
    """A box with a tiled Poly Haven PBR texture (UVs in metres), in the prop's local frame."""
    return dict(shape="tbox", c=tuple(c), s=tuple(s), tex=tex, tile=tile, rot=(0, 0, 0), mat=None, col=True)


def smooth(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def rot(yaw, a, b):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * a - s * b, s * a + c * b


# ================================================================================================ coastline
def shore_y(x, c):
    return c["A1"] * math.sin(x / c["L1"] + c["P1"]) + c["A2"] * math.sin(x / c["L2"] + c["P2"])


def shore_yaw(x, c):
    d = (shore_y(x + 2, c) - shore_y(x - 2, c)) / 4
    return math.atan(d)


def natural_height(x, y, c):
    lat = y - shore_y(x, c)
    W = c["W"]
    if lat < -W - 45:
        h = SEA_FLOOR + 0.5 * math.sin(x / 57.0) * math.cos(y / 43.0)
    elif lat < -W:     # shelf edge: -0.9 m -> sea floor
        h = -0.9 + (SEA_FLOOR + 0.9) * smooth(-W, -W - 45, lat)
    elif lat < 0:      # gentle wading shelf (~1.3 deg): people can stand in the shallows
        h = -0.9 * (-lat / W)
    elif lat < 6:      # wet sand at the waterline (~1.4 deg)
        h = 0.15 * (lat / 6)
    elif lat < W:      # dry beach, ~1.2-1.9 deg
        h = 0.15 + 0.65 * (lat - 6) / (W - 6)
    else:
        h = 0.8 + (c["P"] - 0.8) * smooth(W, W + c["D"], lat)
        h += smooth(W, W + c["D"], lat) * (1.6 * math.sin(x / 83.0 + c["P1"]) * math.cos(y / 71.0) + 0.5 * math.sin((x + y) / 23.0))
    for hx, hh, hr in c["headlands"]:
        d = math.hypot(x - hx, y - (shore_y(hx, c) + 25.0))
        shape = max(0.0, 1.0 - (d / hr) ** 1.2) ** 1.5
        cut = smooth(-W * 0.5, -W * 0.05, lat)  # the sea cuts the headland: a cliff face at the waterline
        h = max(h, hh * shape * cut + h * (1 - cut))
    return h


def terrain_height(x, y, pl):
    c = pl["coast"]
    h = natural_height(x, y, c)
    for z in pl["zones"]:
        cx, cy, yaw, hx, hy, zz, f, kind = z
        u, v = rot(-yaw, x - cx, y - cy)
        du, dv = max(0.0, abs(u) - hx), max(0.0, abs(v) - hy)
        if kind == "quay" and v < -hy:
            continue  # sharp quay wall on the seaward side
        d = math.hypot(du, dv)
        if d >= f:
            continue
        w = 1.0 - smooth(0.0, f, d) if d > 0 else 1.0
        if kind == "basin":
            h = min(h, h * (1 - w) + zz * w)
        else:
            h = h * (1 - w) + zz * w
    return h


def _in_rect(x, y, r):
    cx, cy, yaw, hx, hy = r[:5]
    u, v = rot(-yaw, x - cx, y - cy)
    return abs(u) <= hx and abs(v) <= hy


def surface_z(x, y, pl, water=False):
    """Top of the solid under (x, y): terrain, or a deck (pier, pontoon, ship, slab) above it. water=True also
    counts the sea surface (z = 0) -- the 'ground' a drone measures its height above."""
    h = terrain_height(x, y, pl)
    for r in pl["decks"]:
        if r[5] > h and _in_rect(x, y, r):
            h = r[5]
    if water:
        h = max(h, 0.0)
    return h


def compass_words(x, y, pl):
    ew = "west" if x < -170 else "east" if x > 170 else "central"
    return f"in the {ew} part of the bay" if ew != "central" else "in the middle of the bay"


# ================================================================================================ plan
def _site_frame(x, c, inland=0.0):
    yaw = shore_yaw(x, c)
    y = shore_y(x, c)
    nx, ny = -math.sin(yaw), math.cos(yaw)  # inland normal
    return x + nx * inland, y + ny * inland, yaw


def _world(frame, u, v):
    x0, y0, yaw = frame
    dx, dy = rot(yaw, u, v)
    return x0 + dx, y0 + dy


def plan(idx, attempt=0):
    rng = np.random.default_rng(7_000_003 * idx + 17 + 6_007 * attempt)
    c = dict(A1=float(rng.uniform(25, 65)), L1=float(rng.uniform(230, 380)), P1=float(rng.uniform(0, 6.28)),
             A2=float(rng.uniform(8, 24)), L2=float(rng.uniform(90, 150)), P2=float(rng.uniform(0, 6.28)),
             W=float(rng.uniform(26, 42)), D=float(rng.uniform(55, 110)), P=float(rng.uniform(6, 15)))
    pl = dict(index=idx, attempt=attempt, coast=c, zones=[], decks=[], solids=[], sites=[], ships=[], pads={}, statics=[],
              movables=[], obstacles=[], refs=[], landmarks={}, islets=[], wind=float(rng.uniform(0, 2 * math.pi)))
    # ---- sites along the shore
    n_sites = int(rng.integers(4, 7))
    first = [str(rng.choice(["fishing harbour", "fishing harbour", "container terminal", "marina"])), "beach"]
    rest = [str(k) for k in rng.permutation(SITE_KINDS) if k not in first]
    kinds = (first + rest)[:n_sites]
    rng.shuffle(kinds)
    xs, placed = [], []
    for k in kinds:
        w = SITE_WIDTH[k]
        for _ in range(200):
            x = float(rng.uniform(-560 + w / 2, 560 - w / 2))
            if all(abs(x - px) > (w + pw) / 2 + 25 for px, pw in placed):
                placed.append((x, w))
                xs.append((k, x))
                break
    # headlands: one under each lighthouse point + 0-2 free ones away from the other sites
    heads = []
    for k, x in xs:
        if k == "lighthouse point":
            heads.append((x, float(rng.uniform(26, 45)), float(rng.uniform(110, 150))))
    for _ in range(int(rng.integers(0, 3))):
        for _try in range(100):
            x = float(rng.uniform(-600, 600))
            if all(abs(x - px) > pw / 2 + 120 for px, pw in placed) and all(abs(x - hx) > 220 for hx, _, _ in heads):
                heads.append((x, float(rng.uniform(22, 48)), float(rng.uniform(100, 160))))
                break
    c["headlands"] = heads
    pl["site_slots"] = xs
    sites = []
    for n, (k, x) in enumerate(sorted(xs, key=lambda t: t[1])):
        n0 = len(pl["obstacles"])
        site = _build_site(k, n, x, pl, np.random.default_rng(int(rng.integers(1 << 30))))
        pl["obstacles"][n0:] = [o if o[5] is not None else o[:5] + (site["name"],) for o in pl["obstacles"][n0:]]
        sites.append(site)
    pl["sites"] = sites
    # ---- shore pads (on flat land, away from sites and headlands)
    _plan_pads(pl, rng)
    # ---- offshore vessels, islets, buoys
    _plan_offshore(pl, rng)
    # landmarks: every site and every vessel >= 9 m long, and islets
    for s in sites:
        pl["landmarks"][s["name"]] = dict(xy=s["xy"], top_z=s["top"], words=s["words"], kind=s["kind"], size_r=s["size_r"], box=s.get("box"))
    for sh in pl["ships"]:
        if (sh["L"] >= 9 and not sh.get("moored")) or sh["kind"] in ("container ship", "coast guard cutter"):
            pl["landmarks"][sh["name"]] = dict(xy=sh["xy"], top_z=sh["top"], words=sh["words"], kind=sh["kind"],
                                               size_r=max(sh["L"], sh["B"]) / 2, box=sh["box"], moored=sh.get("moored"))
    for isl in pl["islets"]:
        pl["landmarks"][isl["name"]] = dict(xy=isl["xy"], top_z=isl["top"], words=isl["words"], kind="islet", size_r=isl["r"], box=isl["box"])
    _unique_words(pl)
    _prune_overlaps(pl)
    # keep only objects that pass the static-equilibrium test where they stand (every published prop is stable)
    keep, dropped = [], []
    for m, mv in zip(pl["movables"], movables(pl)):
        (keep if mv["static_check"]["stable"] else dropped).append(m)
    pl["movables"] = keep
    pl["dropped_unstable"] = [(m["group"], m["name"], m["kind"]) for m in dropped]
    if idx not in FROZEN:
        plan_spots(pl)
        srng = np.random.default_rng(55_000_001 * idx + 3)
        ids = sorted(SKIES)
        w = np.array([SKIES[k]["weight"] for k in ids], float)
        pl["sky"] = dict(id=str(srng.choice(ids, p=w / w.sum())), sun_az_deg=float(round(srng.uniform(0, 360), 1)))
    return pl


# Sky / time of day per simulation (Poly Haven CC0 "puresky" HDRIs; sun elevation + colour measured from each image,
# assets/polyhaven-coastal-v2/hdri/sky_info.json). dome = dome-light intensity, sun = distant-light intensity.
SKIES = {
    "kloofendal_48d_partly_cloudy_puresky": dict(label="midday, partly cloudy", dome=900, sun=1050, weight=3),
    "kloofendal_43d_clear_puresky": dict(label="midday, clear sky", dome=850, sun=1100, weight=2),
    "qwantani_noon_puresky": dict(label="noon, clear sky", dome=850, sun=1150, weight=2),
    "qwantani_mid_morning_puresky": dict(label="mid-morning", dome=850, sun=1050, weight=2),
    "qwantani_morning_puresky": dict(label="early morning, low warm sun", dome=800, sun=950, weight=1.5),
    "qwantani_late_afternoon_puresky": dict(label="late afternoon, low warm sun", dome=800, sun=950, weight=1.5),
    "syferfontein_18d_clear_puresky": dict(label="late afternoon, clear sky", dome=800, sun=950, weight=1.5),
    "qwantani_sunset_puresky": dict(label="sunset", dome=750, sun=700, min_elev=9.0, weight=1.0),
    "kloofendal_overcast_puresky": dict(label="overcast", dome=1250, sun=180, diffuse=True, weight=1.2),
    "kloofendal_misty_morning_puresky": dict(label="misty morning", dome=1150, sun=150, diffuse=True, weight=0.8),
}
SKY_U_OFFSET = 0.0  # dome-rotation offset between the image longitude and world yaw (checked on the preview)


# Simulations already published with the pad-only missions (Easy/Coastal Simulation_2, 4, 6, 10, 11): their plans and
# missions stay exactly as published. Every other simulation also gets unmarked take-off / landing SPOTS.
FROZEN = {2, 4, 6, 10, 11}


def _spot_clear(pl, x, y, z, r_obs=7.0, r_solid=2.5):
    for o in pl["obstacles"]:
        if o[4] > z + 1.0 and math.hypot(max(o[0] - x, 0, x - o[1]), max(o[2] - y, 0, y - o[3])) < r_obs:
            return False
    for sd in pl["solids"]:
        if sd[7] > z + 0.25 and _solid_hit(sd, x, y, r_solid):
            return False
    return True


def _spot_flat(pl, x, y, r=3.0, tol=0.3):
    zs = [surface_z(x + dx, y + dy, pl) for dx in (-r, 0.0, r) for dy in (-r, 0.0, r)]
    return max(zs) - min(zs) < tol and min(zs) > 0.3


def plan_spots(pl):
    """Unmarked places a mission can take off from or land on (no painted pad): open beach sand, a clear part of a
    quay, the pier, the village promenade, the grass by the lighthouse, an open meadow inland. Own rng (the rest of
    the plan is unchanged); props standing within 4 m of a chosen spot are removed so the drone has room."""
    rng = np.random.default_rng(91_000_013 * pl["index"] + 7 + 131 * pl["attempt"])
    c = pl["coast"]
    spots = []

    def add(kind, x, y, words, site=None, tries_ok=True):
        z = surface_z(x, y, pl)
        r = 1.5 if kind in ("pier", "promenade") else 3.0  # the pier deck is only 4 m wide
        if not tries_ok or not _spot_flat(pl, x, y, r=r) or not _spot_clear(pl, x, y, z):
            return False
        if any(math.hypot(x - a["xy"][0], y - a["xy"][1]) < 30 for a in spots):
            return False
        if any(math.hypot(x - p["xy"][0], y - p["xy"][1]) < 30 for p in pl["pads"].values()):
            return False
        spots.append(dict(xy=(x, y), z=z, words=words, site=site, ship=None, spot=kind))
        return True

    for s in pl["sites"]:
        k, fr, w = s["kind"], s.get("frame"), s["words"]
        hw = SITE_WIDTH[k] / 2
        for _ in range(60):
            if k == "beach":
                v = float(rng.uniform(7, c["W"] * 0.8))
                x, y = _world(fr, float(rng.uniform(-hw + 15, hw - 15)), v)
                if add("beach", x, y, f"the open sand near the water on {w}" if v < 13 else f"an open patch of sand on {w}", s["name"]):
                    break
            elif k in ("fishing harbour", "container terminal", "marina", "coast guard station", "boatyard"):
                d = {"fishing harbour": 45, "container terminal": 95, "marina": 35, "coast guard station": 40, "boatyard": 45}[k]
                v = float(rng.uniform(5, d - 5))
                x, y = _world(fr, float(rng.uniform(-hw + 8, hw - 8)), v)
                if add("quay", x, y, f"the edge of the quay at {w}, next to the water" if v < 10 else f"a clear part of the quay at {w}", s["name"]):
                    break
            elif k == "fishing pier":
                dk = min((d_ for d_ in pl["decks"] if d_[6] == "pier"), key=lambda d_: math.hypot(d_[0] - s["xy"][0], d_[1] - s["xy"][1]))
                end = rng.random() < 0.6
                v = -dk[4] + float(rng.uniform(10, 16)) if end else dk[4] - float(rng.uniform(6, 12))
                x, y = _world((dk[0], dk[1], dk[2]), 0.0, v)
                if add("pier", x, y, f"the far end of {w}" if end else f"the landward end of {w}", s["name"]):
                    break
            elif k == "village":
                dk = min((d_ for d_ in pl["decks"] if d_[6] == "promenade"), key=lambda d_: math.hypot(d_[0] - s["xy"][0], d_[1] - s["xy"][1]))
                x, y = _world((dk[0], dk[1], dk[2]), float(rng.uniform(-dk[3] + 10, dk[3] - 10)), float(rng.uniform(-2, 2)))
                if add("promenade", x, y, "the seafront promenade of the seaside village", s["name"]):
                    break
            elif k == "lighthouse point":
                a, r = float(rng.uniform(0, 2 * math.pi)), float(rng.uniform(11, 15))
                x, y = s["xy"][0] + r * math.cos(a), s["xy"][1] + r * math.sin(a)
                if add("grass", x, y, f"the grass beside {w}", s["name"]):
                    break
    for _ in range(400):  # 1-2 open meadows inland
        if sum(1 for a in spots if a["spot"] == "meadow") >= int(rng.integers(1, 3)):
            break
        x = float(rng.uniform(-520, 520))
        y = shore_y(x, c) + c["W"] * 1.5 + float(rng.uniform(25, 120))
        if any(math.hypot(x - s["xy"][0], y - s["xy"][1]) < s.get("size_r", 30) + 40 for s in pl["sites"]):
            continue
        near = min(pl["sites"], key=lambda s: math.hypot(s["xy"][0] - x, s["xy"][1] - y))
        side = "west" if x < -200 else "east" if x > 200 else "middle"
        add("meadow", x, y, f"the open meadow inland of {near['words']}" if math.hypot(near["xy"][0] - x, near["xy"][1] - y) < 220
            else f"the open meadow inland at the {side} of the bay" if side == "middle" else f"the open meadow at the {side} end of the coast")
    # unique descriptions
    seen = {}
    for a in spots:
        seen.setdefault(a["words"], []).append(a)
    for w, lst in seen.items():
        if len(lst) > 1:
            for a in sorted(lst, key=lambda a: a["xy"][0]):
                a["words"] = f"{w} ({'the western one' if a is min(lst, key=lambda b: b['xy'][0]) else 'the eastern one'})"
    # room to land: drop props within 4 m of a spot
    removed = [m for m in pl["movables"] if any(math.hypot(m["x"] - a["xy"][0], m["y"] - a["xy"][1]) < 4.0 for a in spots)]
    pl["movables"] = [m for m in pl["movables"] if m not in removed]
    pl["spots"] = {f"S{k + 1}": a for k, a in enumerate(spots)}
    pl["spot_cleared_props"] = [(m["group"], m["name"]) for m in removed]


def _unique_words(pl):
    """Make every landmark description unique within the scene (an instruction must name exactly one thing)."""
    seen = {}
    for name, lm in pl["landmarks"].items():
        seen.setdefault(lm["words"], []).append(name)
    for words, names in seen.items():
        if len(names) < 2:
            continue
        for name in names:
            lm = pl["landmarks"][name]
            x, y = lm["xy"]
            where = compass_words(x, y, pl)
            dist = y - shore_y(x, pl["coast"])
            near = "close to the shore" if dist > -160 else "far out to sea"
            lm["words"] = f"{words} {where}" if sum(1 for n in names if compass_words(*pl['landmarks'][n]['xy'], pl) == where) == 1 else f"{words} {near} {where}"
    for name, lm in pl["landmarks"].items():  # last resort: number them west to east
        same = [n for n, l in pl["landmarks"].items() if l["words"] == lm["words"]]
        if len(same) > 1:
            same.sort(key=lambda n: pl["landmarks"][n]["xy"][0])
            for i, n in enumerate(same):
                pl["landmarks"][n]["words"] += ["", " (the western one)", " (the eastern one)", " (the third one from the west)"][min(i + 1 if i else 1, 3)] if i else " (the western one)"


def _foot_r(prop, body=False):
    """Footprint radius. body=True (v2.1) also covers the whole body of long objects: from the wheel positions alone
    a 7.6 m box truck got 2.63 m, so two trucks 5 m apart nose to tail overlapped and shoved each other 0.3 m."""
    sup = prop.get("support") or [(0.3, 0.3)]
    r = max(math.hypot(a, b) for a, b in sup) + 0.12
    if body and prop.get("length", 0) > 2.0:
        r = max(r, math.hypot(prop["length"] / 2, prop.get("width", prop["length"] * 0.4) / 2) + 0.1)
    return r


def _solid_hit(sd, x, y, r):
    kind, sx, sy, a, hx, hy = sd[:6]
    if kind == "circle":
        return math.hypot(x - sx, y - sy) < r + a
    u, v = rot(-a, x - sx, y - sy)
    return math.hypot(max(abs(u) - hx, 0.0), max(abs(v) - hy, 0.0)) < r


def _prune_overlaps(pl):
    """Drop movable objects that would start inside a solid part (other than the surface they stand on) or inside
    another movable object -- PhysX would push them out and they would jump (City/Coastal flight 001 lesson)."""
    keep, occ, dropped = [], [], []
    for m in pl["movables"]:
        r = _foot_r(m["prop"], body=pl["index"] not in FROZEN)
        zg = terrain_height(m["x"], m["y"], pl) if m["seabed"] else surface_z(m["x"], m["y"], pl)
        top = zg + max(0.3, m["prop"].get("height", 1.0))
        bad = any(_solid_hit(sd, m["x"], m["y"], r) and sd[7] > zg + 0.04 and sd[6] < top for sd in pl["solids"])
        bad = bad or any(math.hypot(m["x"] - ox, m["y"] - oy) < r + orr for ox, oy, orr in occ)
        if bad:
            dropped.append((m["group"], m["name"], m["kind"]))
            continue
        keep.append(m)
        occ.append((m["x"], m["y"], r))
    pl["movables"] = keep
    pl["dropped_overlap"] = dropped


def _flat_ok(pl, x, y, r, water):
    f = (lambda a, b: terrain_height(a, b, pl)) if water else (lambda a, b: surface_z(a, b, pl))
    zs = [f(x + dx, y + dy) for dx in (-r, 0.0, r) for dy in (-r, 0.0, r)]
    return max(zs) - min(zs) < 0.06 * r / 0.7 + 0.02


def _add_movable(pl, group, name, prop, x, y, yaw, water=False):
    r = max(0.5, min(3.0, prop.get("length", 1.0) / 2 + 0.2))
    if not _flat_ok(pl, x, y, r, water):
        pl.setdefault("rejected", []).append((group, name, prop["kind"]))
        return False
    pl["movables"].append(dict(group=group, name=name, prop=prop, x=x, y=y, yaw=yaw, kind=prop["kind"], seabed=water))


def _solids(pl, parts, x, y, z, yaw):
    """Occupancy of collider parts: (x, y, radius, z0, z1) in world -- used to keep movable objects out of solid parts."""
    for p in parts:
        if not p.get("col") or p["shape"] not in ("box", "tbox", "cyl", "cap", "sph"):
            continue
        cx, cy, cz = p["c"]
        dx, dy = rot(yaw, cx, cy)
        sh = p["shape"]
        if sh in ("box", "tbox"):  # oriented rectangle (long walls must not become huge circles)
            ry = yaw + math.radians(p["rot"][2]) if p.get("rot") else yaw
            pl["solids"].append(("rect", x + dx, y + dy, ry, p["s"][0] / 2, p["s"][1] / 2, z + cz - p["s"][2] / 2, z + cz + p["s"][2] / 2))
            continue
        elif sh == "cyl":
            r = p["r"] if p.get("axis", "Z") == "Z" else max(p["r"], p["h"] / 2)
            hz = p["h"] / 2 if p.get("axis", "Z") == "Z" else p["r"]
            h0, h1 = cz - hz, cz + hz
        elif sh == "cap":
            r = p["r"] + (0 if p.get("axis") == "Z" else p["h"] / 2)
            hz = p["r"] + (p["h"] / 2 if p.get("axis") == "Z" else 0)
            h0, h1 = cz - hz, cz + hz
        else:
            r, h0, h1 = p["r"] * max(p["scale"][:2]), cz - p["r"] * p["scale"][2], cz + p["r"] * p["scale"][2]
        pl["solids"].append(("circle", x + dx, y + dy, r, 0.0, 0.0, z + h0, z + h1))


def _add_static(pl, path, parts, x, y, z, yaw, label, words=None):
    pl["statics"].append(dict(path=path, parts=parts, M=cp.M(x, y, z, yaw), label=label, words=words, xy=(x, y)))
    _solids(pl, parts, x, y, z, yaw)


def _add_obstacle_local(pl, frame_xyz_yaw, lbox, owner=None):
    """Local box (x0, x1, y0, y1, top) under a frame (x, y, z, yaw) -> world AABB obstacle."""
    x, y, z, yaw = frame_xyz_yaw
    x0, x1, y0, y1, top = lbox
    pts = [rot(yaw, a, b) for a in (x0, x1) for b in (y0, y1)]
    pl["obstacles"].append((x + min(p[0] for p in pts), x + max(p[0] for p in pts), y + min(p[1] for p in pts),
                            y + max(p[1] for p in pts), z + top, owner))


def _deck(pl, x, y, yaw, hx, hy, z, label):
    pl["decks"].append((x, y, yaw, hx, hy, z, label))


# ------------------------------------------------------------------------------------------------ sites
def _build_site(kind, n, x, pl, rng):
    c = pl["coast"]
    fr = _site_frame(x, c)
    yaw = fr[2]
    name = f"{kind.replace(' ', '_')}_{n + 1}"
    s = dict(kind=kind, name=name, x=x, xy=(fr[0], fr[1]), yaw=yaw, top=8.0, size_r=40.0)
    grp = name
    P = lambda u, v: _world(fr, u, v)  # noqa: E731  site frame: u along the shore, v inland (+) / seaward (-)

    def people(nn, u0, u1, v0, v1, z_fn=None, roles=("tourist",), child_frac=0.1, taken=None, prefix="p"):
        taken = taken if taken is not None else []
        out = 0
        for k in range(nn):
            for _ in range(40):
                u, v = float(rng.uniform(u0, u1)), float(rng.uniform(v0, v1))
                if all(math.hypot(u - a, v - b) > 1.3 for a, b in taken):
                    taken.append((u, v))
                    role = str(rng.choice(list(roles)))
                    pr = cp.person(rng, role=role, child=rng.random() < child_frac and role == "tourist")
                    wx, wy = P(u, v)
                    _add_movable(pl, grp, f"{prefix}{k}", pr, wx, wy, yaw + float(rng.uniform(-math.pi, math.pi)))
                    out += 1
                    break
        return out

    if kind in ("fishing harbour", "container terminal", "marina", "coast guard station", "boatyard"):
        hw = SITE_WIDTH[kind] / 2
        depth = {"fishing harbour": 45, "container terminal": 95, "marina": 35, "coast guard station": 40, "boatyard": 45}[kind]
        qz = {"fishing harbour": QUAY_Z, "container terminal": 2.6, "marina": 1.6, "coast guard station": 2.0, "boatyard": 2.0}[kind]
        cx, cy = P(0, depth / 2)
        pl["zones"].append((cx, cy, yaw, hw, depth / 2, qz, 25.0, "quay"))
        bx, by = P(0, -60)
        pl["zones"].append((bx, by, yaw, hw + 30, 60, SEA_FLOOR, 35.0, "basin"))
        # quay wall + paved top (a slab over the flattened ground)
        _add_static(pl, f"/World/Coast/Sites/{name}/quaywall", [tbox((0, -1.0, (qz + SEA_FLOOR) / 2), (2 * hw, 2.2, qz - SEA_FLOOR + 0.1), "large_sandstone_blocks", 3.0)],
                    *P(0, 0), 0.0, yaw, "quay")
        _add_static(pl, f"/World/Coast/Sites/{name}/paving", [tbox((0, depth / 2, qz - 0.12), (2 * hw, depth + 0.5, 0.3), "gravel_concrete", 5.0)],
                    *P(0, 0), 0.0, yaw, "quay")
        _deck(pl, cx, cy, yaw, hw, depth / 2 + 0.25, qz + 0.03, "quay")
        for k, u in enumerate(np.arange(-hw + 6, hw - 4, 14)):
            _add_static(pl, f"/World/Coast/Sites/{name}/bollard_{k}", cp.bollard(), *P(float(u), 0.8), qz + 0.03, yaw, "bollard")
        s["quay_z"] = qz
        s["top"] = qz + 6
    if kind == "fishing harbour":
        hw = SITE_WIDTH[kind] / 2
        s["words"] = "the fishing harbour with the fish market"
        # fish market hall on the quay
        hx, hy = P(-hw * 0.35, 22)
        _add_static(pl, f"/World/Coast/Sites/{name}/market_hall", cp.open_hall(24, 12, 4.5, (0.50, 0.18, 0.12)), hx, hy, QUAY_Z, yaw, "building")
        _add_obstacle_local(pl, (hx, hy, QUAY_Z, yaw), (-13, 13, -7, 7, 6.5))
        for k in range(6):
            u, v = -hw * 0.35 - 8 + 8 * (k % 3), 18.5 + 7 * (k // 3)
            sx_, sy_ = P(u, v)
            _add_static(pl, f"/World/Coast/Sites/{name}/stall_{k}", cp.market_stall(rng), sx_, sy_, QUAY_Z, yaw, "market stall")
            _deck(pl, sx_, sy_, yaw, 1.2, 0.55, QUAY_Z + 0.91, "table")
            wx, wy = P(u, v - 1.4)
            _add_movable(pl, grp, f"seller{k}", cp.person(rng, role="seller", pose="stand"), wx, wy, yaw + math.pi / 2)
        people(14, -hw * 0.35 - 12, -hw * 0.35 + 12, 12, 28, roles=("tourist", "tourist", "fisherman"), prefix="buyer")
        # office + boat sheds
        ox, oy = P(hw * 0.55, 34)
        _add_static(pl, f"/World/Coast/Sites/{name}/office", cp.shed_building(14, 9, 7, (0.85, 0.82, 0.74), (0.30, 0.10, 0.07)), ox, oy, QUAY_Z, yaw + math.pi / 2, "building")
        _add_obstacle_local(pl, (ox, oy, QUAY_Z, yaw), (-5, 5, -8, 8, 8))
        # crates, barrels, nets, workers on the quay edge; a jib crane
        for k in range(10):
            u, v = float(rng.uniform(-hw + 8, hw - 8)), float(rng.uniform(3.5, 9))
            if abs(u - hw * 0.55) < 9:
                continue
            maker = [cp.fish_crate, cp.barrel, cp.net_pile, cp.pallet_of_crates][k % 4]
            prop = maker(rng) if maker is not cp.fish_crate else cp.fish_crate(rng, int(rng.integers(1, 4)))
            wx, wy = P(u, v)
            _add_movable(pl, grp, f"cargo{k}", prop, wx, wy, yaw + float(rng.uniform(-0.3, 0.3)))
        people(8, -hw + 6, hw - 6, 2.5, 12, roles=("worker", "fisherman", "fisherman"), prefix="crew")
        jx, jy = P(hw * 0.1, 4.0)
        _add_static(pl, f"/World/Coast/Sites/{name}/jib_crane", cp.jib_crane(), jx, jy, QUAY_Z, yaw - math.pi / 2, "crane")
        _add_obstacle_local(pl, (jx, jy, QUAY_Z, yaw - math.pi / 2), (-3, 10, -1, 1, 16))
        rx, ry = P(-hw * 0.75, 30)
        _add_static(pl, f"/World/Coast/Sites/{name}/fish_rack", cp.fish_rack(), rx, ry, QUAY_Z, yaw, "fish rack")
        wx, wy = P(hw * 0.3, 14)
        _add_movable(pl, grp, "forklift", cp.forklift(rng), wx, wy, yaw + 0.2)
        # moored fishing boats along the quay
        u = -hw + 14
        k = 0
        while u < hw - 12:
            kind_b = str(rng.choice(["trawler", "trawler", "motorboat", "rowing boat"]))
            sh = cp.ship(kind_b, rng)
            uu = u + sh["L"] / 2
            if uu + sh["L"] / 2 > hw - 6:
                break
            _place_ship(pl, sh, *P(uu, -2.4 - sh["B"] / 2), yaw + (0 if rng.random() < 0.5 else math.pi), f"{name}_boat{k}", rng, moored=s["name"])
            u = uu + sh["L"] / 2 + float(rng.uniform(3, 9))
            k += 1
        s.update(top=QUAY_Z + 16, size_r=hw * 0.8, pad_spot=(hw * 0.75, 30.0))
    elif kind == "container terminal":
        hw = SITE_WIDTH[kind] / 2
        s["words"] = "the container port with the big blue cranes"
        for k, u in enumerate((-hw * 0.45, hw * 0.05)):
            parts, lobs = cp.gantry_crane(reach=34.0, height=float(rng.uniform(42, 52)))
            gx, gy = P(u, 9.0)
            _add_static(pl, f"/World/Coast/Sites/{name}/gantry_{k}", parts, gx, gy, 2.6, yaw, "crane")
            _add_obstacle_local(pl, (gx, gy, 2.6, yaw), lobs[0])
        for k in range(int(rng.integers(6, 11))):
            u, v = -hw + 18 + (k % 4) * (2 * hw - 36) / 3, 40 + 18 * (k // 4)
            parts, h = cp.container_stack(rng, rows=int(rng.integers(3, 6)), tiers=int(rng.integers(2, 5)), bays=int(rng.integers(1, 3)))
            wx, wy = P(u, v)
            _add_static(pl, f"/World/Coast/Sites/{name}/stack_{k}", parts, wx, wy, 2.6, yaw, "containers")
            _add_obstacle_local(pl, (wx, wy, 2.6, yaw), (-13, 13, -7, 7, h + 0.5))
        for k in range(5):
            wx, wy = P(float(rng.uniform(-hw + 10, hw - 10)), float(rng.uniform(22, 30)))
            import city_props
            _add_movable(pl, grp, f"truck{k}", city_props.special_vehicle("box truck", rng), wx, wy, yaw)
        people(10, -hw + 10, hw - 10, 4, 34, roles=("worker",), prefix="docker")
        sh = cp.ship("container ship", rng)
        if sh["L"] > 2 * hw - 20:
            sh = cp.ship("container ship", np.random.default_rng(int(rng.integers(1 << 30))))
        _place_ship(pl, sh, *P(-hw * 0.2, -2.6 - sh["B"] / 2), yaw + (0 if rng.random() < 0.5 else math.pi), f"{name}_ship", rng, moored=s["name"])
        s.update(top=60.0, size_r=hw * 0.85, pad_spot=(hw * 0.78, 70.0))
    elif kind == "marina":
        hw = SITE_WIDTH[kind] / 2
        s["words"] = "the marina full of yachts and sailing boats"
        # main pontoon parallel to the shore, finger pontoons seaward
        mz = 0.55
        mx, my = P(0, -14)
        _add_static(pl, f"/World/Coast/Sites/{name}/pontoon_main", [tbox((0, 0, mz - 0.25), (2 * hw - 10, 3.0, 0.5), "brown_planks_05", 2.0)], mx, my, 0.0, yaw, "pier")
        _deck(pl, mx, my, yaw, hw - 5, 1.5, mz, "pontoon")
        for k in range(2):
            gx, gy = P(-hw * 0.5 + k * hw, -7)
            _add_static(pl, f"/World/Coast/Sites/{name}/gangway_{k}", [props.box((0, 0, 1.0), (2.0, 13.0, 0.25), (0.6, 0.6, 0.6), rot=(-5, 0, 0))], gx, gy, 0.0, yaw, "pier")
        u = -hw + 12
        k = 0
        while u < hw - 10:
            fx, fy = P(u, -26)
            _add_static(pl, f"/World/Coast/Sites/{name}/finger_{k}", [tbox((0, 0, mz - 0.25), (1.8, 21.0, 0.5), "brown_planks_05", 2.0)], fx, fy, 0.0, yaw, "pier")
            _deck(pl, fx, fy, yaw, 0.9, 10.5, mz, "pontoon")
            for side in (-1, 1):
                if rng.random() < 0.8:
                    kind_b = str(rng.choice(["sailboat", "sailboat", "motor yacht", "motorboat"]))
                    sh = cp.ship(kind_b, rng)
                    if sh["B"] > 5.5:
                        continue
                    _place_ship(pl, sh, *P(u + side * (0.9 + 0.6 + sh["B"] / 2), -26 - 1.0), yaw - math.pi / 2, f"{name}_boat{k}{'a' if side < 0 else 'b'}", rng, moored=s["name"])
            u += 14.0
            k += 1
        people(8, -hw + 8, hw - 8, -15.0, -13.0, roles=("tourist",), prefix="walker")
        hx, hy = P(0, 25)
        _add_static(pl, f"/World/Coast/Sites/{name}/clubhouse", cp.shed_building(20, 11, 6, (0.95, 0.95, 0.92), (0.10, 0.25, 0.45)), hx, hy, 1.6, yaw, "building")
        _add_obstacle_local(pl, (hx, hy, 1.6, yaw), (-11, 11, -6.5, 6.5, 7))
        people(8, -18, 18, 6, 16, roles=("tourist",), prefix="visitor")
        s.update(top=22.0, size_r=hw * 0.8, pad_spot=(-hw * 0.7, 28.0))
    elif kind == "coast guard station":
        hw = SITE_WIDTH[kind] / 2
        s["words"] = "the coast-guard station with the red and white boathouse"
        hx, hy = P(-15, 22)
        _add_static(pl, f"/World/Coast/Sites/{name}/boathouse", cp.shed_building(22, 14, 9, (0.92, 0.92, 0.90), (0.80, 0.10, 0.08)), hx, hy, 2.0, yaw + math.pi / 2, "building")
        _add_obstacle_local(pl, (hx, hy, 2.0, yaw), (-7.5, 7.5, -11.5, 11.5, 10))
        _add_static(pl, f"/World/Coast/Sites/{name}/tower", [props.box((0, 0, 7.0), (3.5, 3.5, 14.0), (0.92, 0.92, 0.90)),
                                                             props.box((0, 0, 14.6), (4.5, 4.5, 1.2), (0.85, 0.10, 0.08)),
                                                             props.cyl((0, 0, 17.0), 0.08, 3.6, (0.9, 0.9, 0.9), col=False)], *P(5, 30), 2.0, yaw, "building")
        _add_obstacle_local(pl, (*P(5, 30), 2.0, yaw), (-2.5, 2.5, -2.5, 2.5, 19))
        sx_, sy_ = P(-15, -9)
        _add_static(pl, f"/World/Coast/Sites/{name}/slipway", [props.box((0, 0, -1.6), (6.0, 18.0, 0.4), "concrete", rot=(-13, 0, 0))], sx_, sy_, 0.0, yaw, "slipway")
        sh = cp.ship("coast guard cutter", rng)
        _place_ship(pl, sh, *P(25, -2.6 - sh["B"] / 2), yaw + math.pi, f"{name}_cutter", rng, moored=s["name"])
        people(6, -30, 30, 4, 20, roles=("crew", "worker"), prefix="guard")
        s.update(top=20.0, size_r=hw * 0.75, pad_spot=(hw * 0.6, 28.0))
    elif kind == "boatyard":
        hw = SITE_WIDTH[kind] / 2
        s["words"] = "the boatyard with boats up on stands"
        sx_, sy_ = P(-hw * 0.4, -9)
        _add_static(pl, f"/World/Coast/Sites/{name}/slipway", [props.box((0, 0, -1.6), (8.0, 18.0, 0.4), "concrete", rot=(-13, 0, 0))], sx_, sy_, 0.0, yaw, "slipway")
        hx, hy = P(hw * 0.4, 30)
        _add_static(pl, f"/World/Coast/Sites/{name}/shed", cp.shed_building(26, 16, 10, (0.55, 0.57, 0.58), (0.25, 0.27, 0.28)), hx, hy, 2.0, yaw + math.pi / 2, "building")
        _add_obstacle_local(pl, (hx, hy, 2.0, yaw), (-8.5, 8.5, -13.5, 13.5, 11))
        for k in range(int(rng.integers(3, 6))):  # boats out of the water on cradles
            kind_b = str(rng.choice(["trawler", "sailboat", "motor yacht", "motorboat"]))
            sh = cp.ship(kind_b, rng)
            if sh["L"] > 24:
                continue
            u, v = -hw + 12 + k * 17, 16 + float(rng.uniform(-2, 6))
            wx, wy = P(u, v)
            lift = sh["L"] * 0 + 2.0 + 1.6  # keel clear of the yard on stands
            parts = [dict(pp) for pp in sh["parts"]]
            stands = []
            for xx in (-sh["L"] * 0.25, sh["L"] * 0.25):
                for side in (-1, 1):
                    stands.append(props.box((xx, side * sh["B"] * 0.3, -1.5), (0.25, 0.25, 3.2), (0.25, 0.25, 0.27)))
            _add_static(pl, f"/World/Coast/Sites/{name}/cradle_{k}", parts + stands, wx, wy, 2.0 + 3.05, yaw + math.pi / 2, "boat")
            _add_obstacle_local(pl, (wx, wy, 5.05, yaw + math.pi / 2), (-sh["L"] / 2, sh["L"] / 2, -sh["B"] / 2, sh["B"] / 2, sh["top"] + 0.5))
        people(7, -hw + 6, hw - 6, 5, 40, roles=("worker", "worker", "tourist"), prefix="yard")
        s.update(top=20.0, size_r=hw * 0.8, pad_spot=(-hw * 0.75, 34.0))
    elif kind == "beach":
        hw = SITE_WIDTH[kind] / 2
        s["words"] = "the sandy beach with colourful umbrellas"
        W = c["W"]
        taken = []
        for k in range(int(rng.integers(10, 18))):  # umbrella groups: umbrella + towels + sunbathers
            for _ in range(30):
                u, v = float(rng.uniform(-hw + 10, hw - 10)), float(rng.uniform(5, W - 6))
                if all(math.hypot(u - a, v - b) > 7 for a, b in taken):
                    taken.append((u, v))
                    break
            else:
                continue
            wx, wy = P(u, v)
            gz = terrain_height(wx, wy, pl)
            _add_static(pl, f"/World/Coast/Sites/{name}/umbrella_{k}", cp.umbrella(rng), wx, wy, gz - 0.3, 0.0, "umbrella")
            for j in range(int(rng.integers(1, 3))):
                tu, tv = u + float(rng.uniform(1.4, 2.6)) * (1 if j == 0 else -1), v + float(rng.uniform(-0.8, 0.8))
                tx, ty = P(tu, tv)
                tyaw = yaw + math.pi / 2 + float(rng.uniform(-0.3, 0.3))
                tz = terrain_height(tx, ty, pl)
                _add_static(pl, f"/World/Coast/Sites/{name}/towel_{k}_{j}", cp.towel(rng), tx, ty, tz + 0.01, tyaw, "towel")
                if rng.random() < 0.75:
                    _add_movable(pl, grp, f"sunbather{k}_{j}", cp.sunbather(rng), tx, ty, tyaw)
            if rng.random() < 0.4:
                lx, ly = P(u - 2.5, v + 2.5)
                _add_movable(pl, grp, f"lounger{k}", cp.lounger(rng), lx, ly, yaw + math.pi / 2)
            if rng.random() < 0.35:
                bx_, by_ = P(u + 1.5, v - 2.0)
                _add_movable(pl, grp, f"coolbox{k}", cp.cool_box(rng), bx_, by_, yaw + float(rng.uniform(0, 3)))
        people(16, -hw + 8, hw - 8, 2.0, W - 4, roles=("tourist", "tourist", "swimmer", "swimmer"), child_frac=0.25, taken=taken, prefix="beachgoer")
        # swimmers standing in shallow water (feet on the seabed, 0.6-1.1 m deep)
        n_sw = 0
        for k in range(30):
            if n_sw >= int(rng.integers(5, 12)):
                break
            u = float(rng.uniform(-hw + 10, hw - 10))
            for v in np.arange(-2.0, -W, -0.7):
                wx, wy = P(u, float(v))
                z = terrain_height(wx, wy, pl)
                if -0.85 < z < -0.5:
                    _add_movable(pl, grp, f"swimmer{k}", cp.person(rng, role="swimmer", pose="stand", child=rng.random() < 0.2), wx, wy,
                                 float(rng.uniform(0, 6.28)), water=True)
                    n_sw += 1
                    break
        lx, ly = P(float(rng.uniform(-hw * 0.3, hw * 0.3)), W * 0.6)
        lz = terrain_height(lx, ly, pl)
        _add_static(pl, f"/World/Coast/Sites/{name}/lifeguard_tower", cp.lifeguard_tower(), lx, ly, lz - 0.3, yaw + math.pi / 2, "lifeguard tower")
        _add_obstacle_local(pl, (lx, ly, lz, yaw), (-2, 2, -2, 2, 5))
        lgx, lgy = P(float(rng.uniform(-hw * 0.3, hw * 0.3)) + 4, W * 0.6 - 3)
        _add_movable(pl, grp, "lifeguard", cp.person(rng, role="lifeguard", pose="stand"), lgx, lgy, yaw - math.pi / 2)
        for k in range(int(rng.integers(3, 7))):  # kayaks on the sand
            wx, wy = P(-hw + 6 + 3.0 * k, float(rng.uniform(4, 9)))
            _add_movable(pl, grp, f"kayak{k}", cp.kayak_on_sand(rng), wx, wy, yaw + math.pi / 2 + float(rng.uniform(-0.2, 0.2)))
        vx, vy = P(hw * 0.6, W * 0.5)
        _add_static(pl, f"/World/Coast/Sites/{name}/volleyball", cp.volleyball_net(), vx, vy, terrain_height(vx, vy, pl) - 0.3, yaw, "volleyball net")
        for k in range(int(rng.integers(4, 9))):  # beach huts at the back of the beach
            hx, hy = P(-hw + 12 + k * 4.2, W + 6)
            hz = terrain_height(hx, hy, pl)
            _add_static(pl, f"/World/Coast/Sites/{name}/hut_{k}", cp.beach_hut(rng), hx, hy, hz - 0.2, yaw - math.pi / 2, "beach hut")
        s.update(top=6.0, size_r=hw * 0.8, pad_spot=(hw * 0.75, W + 30))
    elif kind == "fishing pier":
        Lp = float(rng.uniform(70, 130))
        s["words"] = "the long wooden fishing pier"
        pz = 2.6
        ang = yaw  # the pier runs straight out to sea (local -v)
        cx, cy = P(0, -Lp / 2 + 8)
        deck = [tbox((0, 0, pz - 0.15), (4.0, Lp, 0.3), "dark_planks", 2.0)]
        for v in np.arange(-Lp / 2 + 2, Lp / 2, 6.0):
            for side in (-1, 1):
                deck.append(props.cyl((side * 1.8, float(v), (pz + SEA_FLOOR) / 2 - 0.2), 0.18, pz - SEA_FLOOR, (0.35, 0.25, 0.15)))
                deck.append(props.cyl((side * 1.95, float(v), pz + 0.55), 0.04, 1.1, (0.40, 0.28, 0.16), col=False))
        for side in (-1, 1):
            deck.append(props.box((side * 1.95, 0, pz + 1.1), (0.08, Lp, 0.08), (0.40, 0.28, 0.16), col=False))
        deck.append(props.box((0, -Lp / 2 + 4, pz + 1.5), (5.0, 5.0, 3.0), (0.25, 0.45, 0.65)))  # kiosk at the end
        _add_static(pl, f"/World/Coast/Sites/{name}/pier", deck, cx, cy, 0.0, yaw, "pier")
        _deck(pl, cx, cy, yaw, 2.0, Lp / 2, pz, "pier")
        _add_obstacle_local(pl, (cx, cy, 0.0, yaw), (-2.5, 2.5, -Lp / 2, -Lp / 2 + 7, pz + 3.2))
        k = 0
        for v in np.arange(-Lp / 2 + 10, Lp / 2 - 12, float(rng.uniform(4.0, 7.0))):  # anglers along both rails
            side = 1 if k % 2 == 0 else -1
            if rng.random() < 0.8:
                ax, ay = _world((cx, cy, yaw), side * 1.3, float(v))
                _add_movable(pl, grp, f"angler{k}", cp.person(rng, role="fisherman", pose="stand"), ax, ay, yaw + side * math.pi / 2)
                if rng.random() < 0.5:
                    bx_, by_ = _world((cx, cy, yaw), side * 0.5, float(v) + 0.9)
                    _add_movable(pl, grp, f"bucket{k}", cp.cool_box(rng), bx_, by_, yaw)
            k += 1
        for k in range(int(rng.integers(3, 8))):
            wx, wy = _world((cx, cy, yaw), float(rng.uniform(-0.4, 0.4)), float(rng.uniform(-Lp / 2 + 10, Lp / 2 - 6)))
            _add_movable(pl, grp, f"walker{k}", cp.person(rng, role="tourist"), wx, wy, yaw + float(rng.choice([0, math.pi])))
        s.update(top=pz + 3.5, size_r=Lp / 2, xy=(cx, cy), pad_spot=(30.0, 32.0), pier_len=Lp)
        s["box"] = (cx, cy, yaw, 3.0, Lp / 2, pz + 2.0)
    elif kind == "lighthouse point":
        hd = next(h for h in c["headlands"] if abs(h[0] - x) < 1)
        lx, ly = hd[0], shore_y(hd[0], c) + 25.0
        gz = natural_height(lx, ly, c)
        pl["zones"].append((lx, ly, yaw, 16, 16, gz, 20.0, "flat"))
        parts, H, words = cp.lighthouse(rng)
        _add_static(pl, f"/World/Coast/Sites/{name}/lighthouse", parts, lx, ly, gz, 0.0, "lighthouse")
        _add_obstacle_local(pl, (lx, ly, gz, 0.0), (-3.2, 3.2, -3.2, 3.2, H))
        kx, ky = lx + 10 * math.cos(yaw + 0.5), ly + 10 * math.sin(yaw + 0.5)
        _add_static(pl, f"/World/Coast/Sites/{name}/keeper_house", cp.shed_building(9, 7, 4.5, (0.92, 0.92, 0.90), (0.25, 0.25, 0.28)), kx, ky, gz, yaw, "building")
        _add_obstacle_local(pl, (kx, ky, gz, yaw), (-5, 5, -4, 4, 6))
        for k in range(int(rng.integers(3, 8))):
            a = float(rng.uniform(0, 6.28))
            r = float(rng.uniform(5.5, 13))
            px, py = lx + r * math.cos(a), ly + r * math.sin(a)
            if math.hypot(px - kx, py - ky) < 6:
                continue
            _add_movable(pl, grp, f"visitor{k}", cp.person(rng, role="tourist"), px, py, float(rng.uniform(0, 6.28)))
        s.update(xy=(lx, ly), top=gz + H, size_r=12.0, words=f"{words} on the cliff headland", pad_spot=None)
    elif kind == "village":
        hw = SITE_WIDTH[kind] / 2
        s["words"] = "the seaside village with the promenade"
        cx, cy = P(0, c["W"] + 6)
        pz = terrain_height(cx, cy, pl) + 0.6
        pl["zones"].append((cx, cy, yaw, hw, 7, pz, 18.0, "flat"))
        _add_static(pl, f"/World/Coast/Sites/{name}/promenade", [tbox((0, 0, pz - 0.2), (2 * hw, 12, 0.4), "gravel_concrete", 4.0),
                                                                 tbox((0, -6.1, pz - 1.5), (2 * hw, 0.4, 3.2), "large_sandstone_blocks", 3.0)], cx, cy, 0.0, yaw, "promenade")
        _deck(pl, cx, cy, yaw, hw, 6, pz, "promenade")
        for k in range(int(rng.integers(5, 9))):  # cafe tables with umbrellas
            wx, wy = _world((cx, cy, yaw), -hw + 10 + k * 13, 2.5)
            _add_static(pl, f"/World/Coast/Sites/{name}/cafe_umbrella_{k}", cp.umbrella(rng), wx, wy, pz - 0.3, 0.0, "umbrella")
        n_p = 0
        taken = []
        for k in range(18):
            u, v = float(rng.uniform(-hw + 4, hw - 4)), float(rng.uniform(-4.5, 4.5))
            if all(math.hypot(u - a, v - b) > 1.4 for a, b in taken):
                taken.append((u, v))
                wx, wy = _world((cx, cy, yaw), u, v)
                _add_movable(pl, grp, f"stroller{k}", cp.person(rng, role="tourist", child=rng.random() < 0.15), wx, wy, yaw + float(rng.choice([0.0, math.pi])))
                n_p += 1
        for k in range(int(rng.integers(5, 9))):  # houses behind the promenade, on their own foundations
            u = -hw + 12 + k * (2 * hw - 24) / 7
            hx, hy = P(u, c["W"] + 30 + float(rng.uniform(0, 12)))
            hyaw = yaw + math.pi + float(rng.uniform(-0.15, 0.15))  # front door facing the sea
            gl = lambda lx, ly, hx=hx, hy=hy, hyaw=hyaw: terrain_height(hx + rot(hyaw, lx, ly)[0], hy + rot(hyaw, lx, ly)[1], pl)  # noqa: E731
            house = props.house(gl, str(rng.choice(list(props.WALLS))), str(rng.choice(list(props.ROOFS))), int(rng.integers(1, 3)),
                                float(rng.uniform(9, 13)), float(rng.uniform(7, 9)), float(rng.uniform(25, 38)), bool(rng.random() < 0.5), rng)
            _add_static(pl, f"/World/Coast/Sites/{name}/house_{k}", house["parts"], hx, hy, 0.0, hyaw, "building")
            _add_obstacle_local(pl, (hx, hy, 0.0, hyaw), (-7.5, 7.5, -6.5, 6.5, house["top_z"] + 0.5))
        for k in range(int(rng.integers(3, 7))):
            car = props_city_car(rng)
            wx, wy = P(-hw + 15 + k * 12, c["W"] + 16)
            _add_movable(pl, grp, f"car{k}", car, wx, wy, yaw)
        s.update(xy=(cx, cy), top=14.0, size_r=hw * 0.8, pad_spot=(hw * 0.85, c["W"] + 50))
    s["frame"] = fr
    s.setdefault("box", (s["xy"][0], s["xy"][1], yaw, max(15.0, s["size_r"]), 30.0, s["top"]))
    return s


def props_city_car(rng):
    import city_props
    return city_props.random_car(rng)


# ------------------------------------------------------------------------------------------------ ships
def _place_ship(pl, sh, x, y, yaw, name, rng, moored=None):
    sh = dict(sh)
    name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)  # valid USD prim name
    sh.update(name=name, xy=(x, y), yaw=yaw, moored=moored)
    _solids(pl, sh["parts"], x, y, 0.0, yaw)
    for (x0, x1, y0, y1, z) in sh["decks"]:
        cx_, cy_ = rot(yaw, (x0 + x1) / 2, (y0 + y1) / 2)
        _deck(pl, x + cx_, y + cy_, yaw, (x1 - x0) / 2, (y1 - y0) / 2, z, f"deck:{name}")
    if sh["helideck"] is not None:
        hx, hy, hz = sh["helideck"]
        dx, dy = rot(yaw, hx, hy)
        sh["heli_world"] = (x + dx, y + dy, hz)
        r = min(5.5, sh["B"] / 2 - 0.6)
        _deck(pl, x + dx, y + dy, yaw, r, r, hz, f"helideck:{name}")
    for lb in sh["boxes"]:
        _add_obstacle_local(pl, (x, y, 0.0, yaw), lb, owner=name)
    ob = pl["obstacles"][-len(sh["boxes"]):]
    sh["box"] = (x, y, yaw, sh["L"] / 2, sh["B"] / 2, sh["top"])
    for k, (cx_, cy_, cz) in enumerate(sh["crew"]):
        wx, wy = x + rot(yaw, cx_, cy_)[0], y + rot(yaw, cx_, cy_)[1]
        role = "crew" if sh["kind"] not in ("ferry", "motor yacht") else str(rng.choice(["tourist", "tourist", "crew"]))
        _add_movable(pl, name, f"person{k}", cp.person(rng, role=role, pose="stand"), wx, wy, yaw + float(rng.uniform(-3, 3)))
    if sh["kind"] == "trawler":
        for k in range(int(rng.integers(1, 4))):
            if not sh["decks"]:
                break
            x0, x1, y0, y1, _ = sh["decks"][0]
            fx_, fy_ = float(rng.uniform(x0 + 0.5, x1 - 0.5)), float(rng.uniform(y0 + 0.4, y1 - 0.4))
            wx, wy = x + rot(yaw, fx_, fy_)[0], y + rot(yaw, fx_, fy_)[1]
            _add_movable(pl, name, f"catch{k}", cp.fish_crate(rng, int(rng.integers(1, 3))), wx, wy, yaw + float(rng.uniform(-0.4, 0.4)))
    if sh["kind"] in ("motor yacht",) and sh["decks"]:
        x0, x1, y0, y1, z = sh["decks"][0]
        wx, wy = x + rot(yaw, (x0 + x1) / 2, 0)[0], y + rot(yaw, (x0 + x1) / 2, 0)[1]
        _add_movable(pl, name, "sunbather", cp.sunbather(rng), wx, wy, yaw + math.pi / 2)
    seats, n_seated = cp.seated_people(rng, sh["kind"], sh["L"], sh["B"], sh["F"])
    sh["parts"] = sh["parts"] + seats
    sh["seated_people"] = n_seated
    pl["ships"].append(sh)
    return sh


def _plan_offshore(pl, rng):
    c = pl["coast"]
    need_heli = pl.get("_need_heli", True)
    kinds = [str(k) for k in rng.choice(OFFSHORE_KINDS, int(rng.integers(5, 10)))]
    if need_heli and not any(k in HELI_KINDS for k in kinds):
        kinds[0] = str(rng.choice(list(HELI_KINDS)))
    big = [k for k in kinds if k in ("container ship", "tanker")]
    if len(big) > 1:
        kinds = [k for k in kinds if k not in ("container ship", "tanker")] + [big[0]]
    taken = [(s["xy"][0], s["xy"][1], s["size_r"] + 30) for s in pl["sites"]]
    for sh in pl["ships"]:
        taken.append((sh["xy"][0], sh["xy"][1], sh["L"] / 2 + 20))
    for n, k in enumerate(kinds):
        if k == "dutch ship":
            model = str(rng.choice(list(DUTCH)))
            L, B, H = DUTCH[model]
            sh = dict(kind="historic sailing ship", hull="wooden", words="the old wooden three-masted sailing ship" if "large" in model or "pinnace" in model
                      else "the old wooden two-masted sailing ship", L=L, B=B, F=3.0, parts=[], decks=[], crew=[], helideck=None, top=H - 2.0,
                      boxes=[(-L / 2, L / 2, -B / 2, B / 2, 5.0), (-L * 0.35, L * 0.35, -1.5, 1.5, H - 2.0)], model=model)
        else:
            sh = cp.ship(k, rng)
        for _ in range(300):
            x = float(rng.uniform(-520, 520))
            off = float(rng.uniform(70, 380)) if sh["L"] < 40 else float(rng.uniform(170, 420))
            y = shore_y(x, c) - c["W"] - off
            r = max(sh["L"], sh["B"]) / 2 + 25
            if all(math.hypot(x - a, y - b) > r + rr for a, b, rr in taken) and terrain_height(x, y, pl) < -4.5:
                taken.append((x, y, r))
                break
        else:
            continue
        yaw = pl["wind"] + float(rng.uniform(-0.35, 0.35)) if sh["L"] > 20 else float(rng.uniform(0, 2 * math.pi))
        placed = _place_ship(pl, sh, x, y, yaw, f"{k.replace(' ', '_')}_{n + 1}", rng)
        if "model" in sh:
            pl["refs"].append(dict(model=sh["model"], x=x, y=y, z=0.0, yaw=yaw - math.pi / 2, label="ship", name=placed["name"]))
    # islets: flat rock platforms with gulls; buoys
    for k in range(int(rng.integers(0, 3))):
        for _ in range(100):
            x = float(rng.uniform(-480, 480))
            y = shore_y(x, c) - c["W"] - float(rng.uniform(140, 380))
            if all(math.hypot(x - a, y - b) > 70 + rr for a, b, rr in taken):
                taken.append((x, y, 40))
                break
        else:
            continue
        rocks = []
        for j in range(int(rng.integers(2, 5))):
            m = str(rng.choice(["coast_rocks_01", "coast_rocks_02", "coast_rocks_03"]))
            s_ = float(rng.uniform(0.35, 0.6))
            rocks.append(dict(model=m, x=x + float(rng.uniform(-12, 12)), y=y + float(rng.uniform(-12, 12)), z=-0.6, yaw=float(rng.uniform(0, 6.28)),
                              scale=s_, label="rock", name=f"islet_{k + 1}"))
        pl["refs"] += rocks
        beacon = rng.random() < 0.5
        if beacon:
            pl["refs"].append(dict(model="lateral_sea_marker", x=x, y=y, z=1.0, yaw=0.0, label="buoy", name=f"islet_{k + 1}"))
        words = "the small rocky islet with a navigation beacon" if beacon else "the small rocky islet"
        pl["islets"].append(dict(name=f"islet_{k + 1}", xy=(x, y), top=8.0 if beacon else 3.0, r=22.0, words=words, box=(x, y, 0.0, 22.0, 22.0, 8.0)))
        pl["obstacles"].append((x - 22, x + 22, y - 22, y + 22, 8.0 if beacon else 3.5, f"islet_{k + 1}"))
    for k in range(int(rng.integers(3, 9))):
        x = float(rng.uniform(-520, 520))
        y = shore_y(x, c) - c["W"] - float(rng.uniform(40, 300))
        if terrain_height(x, y, pl) > -3 or any(math.hypot(x - a, y - b) < rr + 8 for a, b, rr in taken):
            continue
        m = str(rng.choice(["ocean_buoy", "lateral_sea_marker"]))
        pl["refs"].append(dict(model=m, x=x, y=y, z=0.0, yaw=float(rng.uniform(0, 6.28)), label="buoy", name=f"buoy_{k}"))
        pl["obstacles"].append((x - 2, x + 2, y - 2, y + 2, 6.0 if m == "lateral_sea_marker" else 2.5, f"buoy_{k}"))


# ------------------------------------------------------------------------------------------------ pads
def _plan_pads(pl, rng):
    c = pl["coast"]
    pads = {}
    for s in pl["sites"]:
        if s.get("pad_spot") is None:
            continue
        u, v = s["pad_spot"]
        x, y = _world(s["frame"], u, v)
        if s["kind"] == "fishing pier":
            x, y = _world(_site_frame(s["x"], c), u, v)
        pads[f"P{len(pads) + 1}"] = (x, y, f"the landing pad beside {s['words']}", s["name"])
    for _ in range(400):  # extra pads on open land
        if len(pads) >= 6:
            break
        x = float(rng.uniform(-540, 540))
        y = shore_y(x, c) + c["W"] + float(rng.uniform(40, 150))
        if any(math.hypot(x - px, y - py) < 120 for px, py, _, _ in pads.values()):
            continue
        if any(math.hypot(x - s["xy"][0], y - s["xy"][1]) < s["size_r"] + 35 for s in pl["sites"]):
            continue
        if any(math.hypot(x - hx, y - (shore_y(hx, c) + 25)) < hr * 0.8 for hx, _, hr in c["headlands"]):
            continue
        pads[f"P{len(pads) + 1}"] = (x, y, None, None)
    out = {}
    for k, (x, y, words, site) in pads.items():
        z = terrain_height(x, y, pl) if site is None or not any(s["name"] == site and "quay_z" in s and _in_quay(s, x, y) for s in pl["sites"]) else None
        if z is None:
            s = next(s for s in pl["sites"] if s["name"] == site)
            z = s["quay_z"] + 0.03
        else:
            pl["zones"].append((x, y, 0.0, 12.0, 12.0, z, 18.0, "flat"))
        if words is None:
            words = f"the landing pad on the hill {_side_words(x)}"
        out[k] = dict(xy=(x, y), z=z, words=words, site=site, ship=None)
    pl["pads"] = out


def _side_words(x):
    return "at the west end of the coast" if x < -200 else "at the east end of the coast" if x > 200 else "above the middle of the bay"


def _in_quay(s, x, y):
    u, v = rot(-s["yaw"], x - s["frame"][0], y - s["frame"][1])
    d = {"fishing harbour": 45, "container terminal": 95, "marina": 35, "coast guard station": 40, "boatyard": 45}.get(s["kind"])
    return d is not None and abs(u) <= SITE_WIDTH[s["kind"]] / 2 and 0 <= v <= d


def heli_pads(pl):
    """Helidecks of the ships: name -> dict(xy, z, words, ship)."""
    out = {}
    for sh in pl["ships"]:
        if sh.get("heli_world"):
            x, y, z = sh["heli_world"]
            out[f"H_{sh['name']}"] = dict(xy=(x, y), z=z, words=f"the helicopter deck of {pl['landmarks'][sh['name']]['words']}",
                                          ship_words=pl['landmarks'][sh['name']]['words'], site=None, ship=sh["name"])
    return out


# ================================================================================================ checks
def obstacles(pl):
    return list(pl["obstacles"])


def movables(pl):
    out = []
    for m in pl["movables"]:
        gl = (lambda x, y: terrain_height(x, y, pl)) if m["seabed"] else (lambda x, y: surface_z(x, y, pl))
        M, resid, tilt = props.rest_pose(m["prop"], m["x"], m["y"], m["yaw"], gl)
        out.append(dict(group=m["group"], name=m["name"], prop=m["prop"], matrix=M, static_check=props.check_static(m["prop"], tilt),
                        support_residual_m=round(resid, 4), xy=(m["x"], m["y"]), yaw=m["yaw"], kind=m["kind"]))
    return out


# ================================================================================================ objects
SEG_CLASSES = ["background/sky", "sea", "terrain", "vegetation", "landing pad", "quay", "pier", "building", "crane", "containers",
               "lighthouse", "ship", "boat", "person", "vehicle", "cargo", "beach item", "rock", "buoy", "other"]


def seg_class(label, kind=None):
    lab = (label or "").replace("site:", "")
    table = {"quay": "quay", "bollard": "quay", "slipway": "quay", "promenade": "quay", "pier": "pier", "building": "building",
             "market stall": "building", "fish rack": "building", "beach hut": "building", "lifeguard tower": "building",
             "crane": "crane", "containers": "containers", "lighthouse": "lighthouse", "boat": "boat", "ship": "ship",
             "umbrella": "beach item", "towel": "beach item", "volleyball net": "beach item", "rock": "rock", "buoy": "buoy",
             "person": "person", "child": "person", "sunbather": "person", "sedan": "vehicle", "hatchback": "vehicle", "suv": "vehicle",
             "van": "vehicle", "pickup": "vehicle", "box truck": "vehicle", "forklift": "vehicle", "fish crate": "cargo",
             "crate stack": "cargo", "barrel": "cargo", "net pile": "cargo", "pallet of crates": "cargo", "cool box": "beach item",
             "lounger": "beach item", "kayak on sand": "boat", "landing pad": "landing pad", "terrain": "terrain", "sea": "sea",
             "vegetation": "vegetation"}
    return table.get(lab, "other")


def object_registry(pl):
    """Every object of the scene in a fixed order -> segmentation id (1-based; 0 = sky / nothing).
    [(name, [prim path prefixes], class, words)] -- built from the plan alone, identical in every job."""
    out = [("terrain", ["/World/Terrain", "/World/DistantTerrain"], "terrain", "the ground"),
           ("sea", ["/World/Ocean"], "sea", "the sea"), ("vegetation", ["/World/Coast/Nature"], "vegetation", "grass, shrubs and trees")]
    for k, p in pl["pads"].items():
        if not p.get("ship"):
            out.append((f"pad_{k}", [f"/World/Coast/Pads/{k}"], "landing pad", p["words"]))
    sites = {}
    for st in pl["statics"]:
        owner = st["path"].split("/")[4]
        sites.setdefault(owner, []).append(st)
    for s in pl["sites"]:
        for st in sites.get(s["name"], []):
            part = st["path"].split("/")[-1]
            out.append((f"{s['name']}/{part}", [st["path"]], seg_class(st["label"]), s["words"] if part in ("lighthouse",) else None))
    for sh in pl["ships"]:
        if sh.get("model"):
            continue
        out.append((sh["name"], [f"/World/Coast/Ships/{sh['name']}"], "boat" if sh["L"] < 20 else "ship",
                    pl["landmarks"].get(sh["name"], {}).get("words") or sh["words"]))
    names = {}
    for k, r in enumerate(pl["refs"]):
        names.setdefault(r["name"], []).append(f"/World/Coast/Refs/{r['name']}_{k}")
    for n_, paths in names.items():
        lab = next(r["label"] for r in pl["refs"] if r["name"] == n_)
        out.append((n_, paths, seg_class(lab), pl["landmarks"].get(n_, {}).get("words")))
    for m in pl["movables"]:
        out.append((f"{m['group']}/{m['name']}", [f"/World/Props/{m['group']}/{m['name']}"], seg_class(m["kind"]), None))
    return out


# ================================================================================================ build
def build(stage, nv, pl, assets_root, forest_root):
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    info = {}
    info.update(build_lighting(stage, nv, assets_root, sky=pl.get("sky")))
    info.update(build_terrain(stage, nv, pl, assets_root))
    info.update(build_ocean(stage, nv, pl))
    UsdGeom.Xform.Define(stage, "/World/Coast")
    registry = {}

    def reg(name, path, label, words=None):
        registry.setdefault(name, {"paths": [], "label": label, "words": words})
        registry[name]["paths"].append(path)

    import coastal_scene as old

    tex_mats = {}
    for st in pl["statics"]:
        plain = [p for p in st["parts"] if p["shape"] != "tbox"]
        props.author(stage, nv, st["path"], dict(parts=plain), root_matrix=st["M"])
        for i, p in enumerate(p for p in st["parts"] if p["shape"] == "tbox"):
            if p["tex"] not in tex_mats:
                tex_mats[p["tex"]] = _pbr(stage, nv, assets_root, f"T_{p['tex']}", p["tex"], 0.9)
            old.textured_box(nv, stage, f"{st['path']}/t{i:02d}", p["c"], p["s"], tex_mats[p["tex"]], tile_m=p["tile"])
            # exact box collider: as a triangle mesh a 170 m quay slab is 12 huge triangles and PhysX contacts on it
            # jitter -- people walked 0.3-1.7 m during flight and one fell through (Coastal flights 001-078)
            tp = stage.GetPrimAtPath(f"{st['path']}/t{i:02d}")
            if SLAB_COLLIDER == "cube" or (SLAB_COLLIDER == "grid" and p["s"][2] > 1.0):  # thick walls: native box
                # an invisible native box collider (a 170 m convex hull "failed to cook GPU-compatible mesh" and
                # props on it still drifted, Coastal flight 020); the textured mesh is visual only
                tp.CreateAttribute("x500:noCollision", Sdf.ValueTypeNames.Bool).Set(True)
                cb = UsdGeom.Cube.Define(stage, f"{st['path']}/t{i:02d}_col")
                cb.CreateSizeAttr(1.0)
                xf = UsdGeom.XformCommonAPI(cb)
                xf.SetTranslate(Gf.Vec3d(*map(float, p["c"])))
                xf.SetScale(Gf.Vec3f(*map(float, p["s"])))
                cb.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                UsdPhysics.CollisionAPI.Apply(cb.GetPrim())
            elif SLAB_COLLIDER == "grid":
                # invisible collider of the slab's TOP FACE tessellated like the terrain (~3 m triangles): props stand on
                # terrain-like triangles, the configuration that stays at rest everywhere else (beach, City, Forest)
                tp.CreateAttribute("x500:noCollision", Sdf.ValueTypeNames.Bool).Set(True)
                (cx_, cy_, cz_), (sx_, sy_, sz_) = p["c"], p["s"]
                nu, nw = max(1, int(math.ceil(sx_ / 3.0))), max(1, int(math.ceil(sy_ / 3.0)))
                gp = [Gf.Vec3f(cx_ - sx_ / 2 + sx_ * i / nu, cy_ - sy_ / 2 + sy_ * j / nw, cz_ + sz_ / 2) for j in range(nw + 1) for i in range(nu + 1)]
                gi = []
                for jj in range(nw):
                    for ii in range(nu):
                        a = jj * (nu + 1) + ii
                        gi += [a, a + 1, a + nu + 2, a, a + nu + 2, a + nu + 1]
                gm = UsdGeom.Mesh.Define(stage, f"{st['path']}/t{i:02d}_col")
                gm.CreatePointsAttr(gp)
                gm.CreateFaceVertexIndicesAttr(gi)
                gm.CreateFaceVertexCountsAttr([3] * (len(gi) // 3))
                gm.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
                UsdPhysics.CollisionAPI.Apply(gm.GetPrim())
                UsdPhysics.MeshCollisionAPI.Apply(gm.GetPrim()).CreateApproximationAttr("none")
            elif SLAB_COLLIDER == "convex":
                UsdPhysics.CollisionAPI.Apply(tp)
                UsdPhysics.MeshCollisionAPI.Apply(tp).CreateApproximationAttr("convexHull")
        owner = st["path"].split("/")[4] if st["path"].startswith("/World/Coast/Sites/") else st["path"]
        reg(owner, st["path"], "site:" + st["label"])
    UsdGeom.Xform.Define(stage, "/World/Coast/Ships")
    for sh in pl["ships"]:
        if sh.get("model"):
            continue
        path = f"/World/Coast/Ships/{sh['name']}"
        props.author(stage, nv, path, dict(parts=sh["parts"]), root_matrix=cp.M(sh["xy"][0], sh["xy"][1], 0.0, sh["yaw"]))
        reg(sh["name"], path, "ship")
    info.update(build_refs(stage, nv, pl, assets_root, reg))
    info.update(build_pads(stage, nv, pl, assets_root))
    info.update(build_vegetation(stage, nv, pl, forest_root))
    rigid = []
    UsdGeom.Xform.Define(stage, "/World/Props")
    for m in movables(pl):
        path = f"/World/Props/{m['group']}/{m['name']}"
        UsdGeom.Xform.Define(stage, f"/World/Props/{m['group']}")
        props.author(stage, nv, path, m["prop"], root_matrix=m["matrix"])
        rigid.append({"path": path, "name": f"{m['group']}/{m['name']}", "kind": m["kind"], "mass_kg": m["prop"]["mass"],
                      "pos": [m["matrix"][3][0], m["matrix"][3][1], m["matrix"][3][2]]})
        reg(f"{m['group']}/{m['name']}", path, m["kind"])
    info["rigid_bodies"] = rigid
    info["movable_props"] = len(rigid)
    info["registry"] = registry
    info.update(dict(sites=len(pl["sites"]), ships=len(pl["ships"]), islets=len(pl["islets"]), pads=len(pl["pads"])))
    return info


def build_lighting(stage, nv, assets_root, hdri_id="kloofendal_48d_partly_cloudy_puresky", sky=None):
    """Dome (HDRI sky) + distant sun. Frozen (already published) simulations keep the original fixed setup; every other
    simulation gets its own sky / time of day (pl["sky"]) with the sun light pointing where the sky image has its sun."""
    import json as _json
    from pathlib import Path

    UsdLux, Gf = nv.UsdLux, nv.Gf
    if sky:
        hdri_id = sky["id"]
    hdr = Path(assets_root) / "hdri" / hdri_id / f"{hdri_id}_2k.hdr"
    sky_dome = UsdLux.DomeLight.Define(stage, "/World/Sky")
    if hdr.exists():
        sky_dome.CreateTextureFileAttr().Set(str(hdr))
        sky_dome.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateAngleAttr(0.55)
    if not sky:
        sky_dome.CreateIntensityAttr(900.0)
        sky_dome.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 15.0))
        sun.CreateIntensityAttr(1050.0)
        sun.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
        sun.AddRotateXYZOp().Set(Gf.Vec3f(48.0, -14.0, 32.0))
        return {"hdri": hdri_id if hdr.exists() else None}
    cfg = SKIES[hdri_id]
    info = _json.loads((Path(assets_root) / "hdri" / "sky_info.json").read_text())[hdri_id]
    elev = max(float(info["sun_elev_deg"]), cfg.get("min_elev", 12.0))
    az = float(sky["sun_az_deg"])  # world yaw of the sun (0 = east, counter-clockwise)
    r, g, b = info["sun_rgb"]
    col = (1.0, 0.97, 0.92) if cfg.get("diffuse") else (1.0, 0.55 + 0.45 * g, 0.35 + 0.65 * b)
    sky_dome.CreateIntensityAttr(float(cfg["dome"]))
    sky_dome.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, float((az - info["sun_u_deg"] + SKY_U_OFFSET) % 360.0)))
    sun.CreateIntensityAttr(float(cfg["sun"]))
    sun.CreateColorAttr(Gf.Vec3f(*map(float, col)))
    # a distant light shines along its local -Z: tilt it (90 - elev) about X (light now comes from local -Y), then
    # turn about Z so it comes FROM yaw `az`: local -Y has yaw -90, so rotate by az + 90
    sun.AddRotateXYZOp().Set(Gf.Vec3f(float(90.0 - elev), 0.0, float((az + 90.0) % 360.0)))
    if cfg.get("diffuse"):
        sun.CreateAngleAttr(8.0)  # soft, hazy shadows
    return {"hdri": hdri_id if hdr.exists() else None, "sky": cfg["label"], "sun_elevation_deg": round(elev, 1),
            "sun_azimuth_deg": round(az, 1), "sun_color": [round(float(v), 3) for v in col]}


# Poly Haven's aerial_grass_rock ships an all-zero roughness map: with it the grass rendered as a MIRROR (houses, trees
# and people reflected in the meadow at low camera height, Coastal v2.1 Simulation_9). Use the constant roughness.
BROKEN_ROUGHNESS_MAPS = {"aerial_grass_rock"}


def _pbr(stage, nv, root, name, tex, roughness=0.9, tint=None):
    import coastal_scene as old  # the validated PBR helper (normal map off, ior lowered: no grazing glare)
    return old.create_pbr_material_fixed(stage, nv, f"/World/Materials/CoastV2_{name}", root, tex, uv_scale=1.0, tint=tint,
                                         use_normal=False, roughness=roughness, ior=1.05,
                                         roughness_map=tex not in BROKEN_ROUGHNESS_MAPS)


def build_terrain(stage, nv, pl, assets_root):
    """One terrain mesh banded by elevation and slope (GeomSubsets): wet sand at the waterline, dry sand, rock on
    steep faces (headland cliffs), grass inland. Plus a coarse haze-tinted distant ring."""
    from pxr import Vt

    Gf, UsdGeom, UsdShade = nv.Gf, nv.UsdGeom, nv.UsdShade
    mats = {"wet": _pbr(stage, nv, assets_root, "WetSand", "damp_beach_sand", 0.6),
            "sand": _pbr(stage, nv, assets_root, "Sand", "coast_sand_01", 0.97),
            "rock": _pbr(stage, nv, assets_root, "Rock", "cliff_side", 0.95),
            "grass": _pbr(stage, nv, assets_root, "Grass", "aerial_grass_rock", 0.98, tint=(0.55, 0.70, 0.40)),
            "seabed": nv.create_color_material(stage, "/World/Materials/CoastV2_Seabed", (0.16, 0.15, 0.12), 0.95)}
    c = pl["coast"]
    xs = np.linspace(-EXTENT, EXTENT, GRID)
    H = np.array([[terrain_height(float(x), float(y), pl) for x in xs] for y in xs])
    step = xs[1] - xs[0]
    gy, gx = np.gradient(H, step)
    pts, uvs, nrm = [], [], []
    for iy, y in enumerate(xs):
        for ix, x in enumerate(xs):
            pts.append(Gf.Vec3f(float(x), float(y), float(H[iy, ix])))
            uvs.append(Gf.Vec2f(float((x + EXTENT) / 18.0), float((y + EXTENT) / 18.0)))
            n = np.array([-gx[iy, ix], -gy[iy, ix], 1.0])
            n /= np.linalg.norm(n)
            nrm.append(Gf.Vec3f(*map(float, n)))
    idx, cnt, bands = [], [], {k: [] for k in mats}
    f = 0
    for r in range(GRID - 1):
        for col in range(GRID - 1):
            a = r * GRID + col
            b, cc, d = a + 1, a + GRID, a + GRID + 1
            idx += [a, b, d, a, d, cc]
            cnt += [3, 3]
            hq = (H[r, col] + H[r, col + 1] + H[r + 1, col] + H[r + 1, col + 1]) / 4
            sl = math.hypot((gx[r, col] + gx[r + 1, col + 1]) / 2, (gy[r, col] + gy[r + 1, col + 1]) / 2)
            qx, qy = (xs[col] + xs[col + 1]) / 2, (xs[r] + xs[r + 1]) / 2
            lat = qy - shore_y(qx, c)
            if hq < -1.0:
                band = "seabed"
            elif hq < 0.45 and lat < c["W"]:
                band = "wet"
            elif sl > 0.55:
                band = "rock"
            elif hq < 1.4 and lat < c["W"] * 1.3:
                band = "sand"
            else:
                band = "grass"
            bands[band] += [f, f + 1]
            f += 2
    mesh = nv.create_mesh(stage, "/World/Terrain", pts, idx, cnt, uvs, None, collision=False)
    mesh.CreateNormalsAttr(nrm)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    for k, faces in bands.items():
        if faces:
            sub = UsdGeom.Subset.CreateGeomSubset(mesh, f"Band_{k}", UsdGeom.Tokens.face, Vt.IntArray(faces))
            UsdShade.MaterialBindingAPI(sub.GetPrim()).Bind(mats[k])
    # distant ring: coarse, flat haze-graded colour (no tiled texture to alias near the horizon)
    far_mat = nv.create_color_material(stage, "/World/Materials/CoastV2_FarLand", (0.52, 0.50, 0.40), 0.9)
    cf = np.linspace(-FAR, FAR, 141)
    pf, idf, cf_ = [], [], []
    for y in cf:
        for x in cf:
            r_ = max(abs(x), abs(y))
            fade = min(1.0, max(0.0, (r_ - EXTENT) / (FAR - EXTENT)))
            h = (natural_height(float(x), float(y), c) if r_ < 1200 else (c["P"] if y > shore_y(float(x), c) else SEA_FLOOR))
            pf.append(Gf.Vec3f(float(x), float(y), float(h * (1 - 0.5 * fade) - 0.5 * fade)))
    for r in range(140):
        for col in range(140):
            if max(abs(cf[col]), abs(cf[col + 1]), abs(cf[r]), abs(cf[r + 1])) < EXTENT:
                continue
            a = r * 141 + col
            idf += [a, a + 1, a + 142, a, a + 142, a + 141]
            cf_ += [3, 3]
    nv.create_mesh(stage, "/World/DistantTerrain", pf, idf, cf_, None, far_mat, collision=False)
    return {"terrain_faces": {k: len(v) // 2 for k, v in bands.items()}, "terrain_extent_m": EXTENT}


def _wave_h(x, y):
    return 0.16 * math.sin(x / 42.0 + y / 61.0) + 0.10 * math.sin(x / 27.0 - y / 33.0) + 0.06 * math.cos(x / 71.0)


def build_ocean(stage, nv, pl):
    """Sea surface: gentle swell geometry (+-0.3 m) with authored normals, slightly reflective tinted water at
    z ~ 0 -- below every dry-land vertex (the beach reaches 0 only at the waterline)."""
    Gf, UsdGeom = nv.Gf, nv.UsdGeom
    mat = nv.create_color_material(stage, "/World/Materials/CoastV2_Ocean", (0.010, 0.048, 0.062), 0.16, metallic=0.04)
    n = 241
    xs = np.linspace(-FAR, FAR, n)
    pts, nrm = [], []
    for y in xs:
        for x in xs:
            pts.append(Gf.Vec3f(float(x), float(y), float(-0.12 + _wave_h(float(x), float(y)))))
            e = 4.0
            dx = (_wave_h(x + e, y) - _wave_h(x - e, y)) / (2 * e)
            dy = (_wave_h(x, y + e) - _wave_h(x, y - e)) / (2 * e)
            v = np.array([-dx, -dy, 1.0]) / math.sqrt(dx * dx + dy * dy + 1)
            nrm.append(Gf.Vec3f(*map(float, v)))
    idx, cnt = [], []
    for r in range(n - 1):
        for col in range(n - 1):
            a = r * n + col
            idx += [a, a + 1, a + n + 1, a, a + n + 1, a + n]
            cnt += [3, 3]
    m = nv.create_mesh(stage, "/World/Ocean", pts, idx, cnt, None, mat, collision=False)
    m.CreateNormalsAttr(nrm)
    m.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    return {"ocean": "swell +-0.32 m around z=-0.12"}


def build_refs(stage, nv, pl, root, reg):
    """Poly Haven models (historic ships, islet rocks, buoys): one referenced Xform each, so every model has its own
    prim path (per-object segmentation). No colliders (the flight check keeps the drone clear with obstacle boxes)."""
    from pathlib import Path

    from pxr import Gf, UsdGeom

    mats, counts = {}, {}
    for k, r in enumerate(pl["refs"]):
        model = r["model"]
        n_diff = len(list((Path(root) / "models" / model / "textures").glob("*_diff_*")))
        if model not in mats:  # multi-material models (the sailing ships: hull / rigging / sails) keep their own materials
            mats[model] = nv.create_asset_material(stage, f"/World/Materials/PH_{model}", root, model) if n_diff == 1 else None
        path = f"/World/Coast/Refs/{r['name']}_{k}"
        xf = UsdGeom.Xform.Define(stage, path)
        xf.AddTranslateOp().Set(Gf.Vec3d(r["x"], r["y"], r["z"]))
        xf.AddRotateZOp().Set(math.degrees(r["yaw"]))
        s_ = r.get("scale", 1.0)
        xf.AddScaleOp().Set(Gf.Vec3f(s_, s_, s_))
        xf.GetPrim().GetReferences().AddReference(str(Path(root) / "models" / model / f"{model}_1k.usdc"))
        if mats[model] is not None:
            nv.bind(xf.GetPrim(), mats[model], strong=True)
        xf.GetPrim().CreateAttribute("x500:noCollision", nv.Sdf.ValueTypeNames.Bool).Set(True)
        reg(r["name"], path, r["label"])
        counts[model] = counts.get(model, 0) + 1
    return {"poly_haven_models": counts}


def build_pads(stage, nv, pl, assets_root):
    slab = nv.create_color_material(stage, "/World/Materials/CoastV2_PadSlab", (0.55, 0.54, 0.52), 0.8)
    paint = nv.create_color_material(stage, "/World/Materials/CoastV2_PadPaint", (0.95, 0.80, 0.10), 0.5)
    white = nv.create_color_material(stage, "/World/Materials/CoastV2_PadWhite", (0.92, 0.92, 0.90), 0.5)
    for k, p in pl["pads"].items():
        if p.get("ship"):
            continue
        (px, py), z = p["xy"], p["z"]
        root = f"/World/Coast/Pads/{k}"
        nv.cube(stage, f"{root}/slab", (px, py, z - 0.04), (20, 20, 0.12), slab)
        for j in range(28):
            a = 2 * math.pi * j / 28
            nv.cube(stage, f"{root}/ring{j:02d}", (px + 6.5 * math.cos(a), py + 6.5 * math.sin(a), z + 0.025), (1.4, 0.32, 0.02), paint, rotation_z=math.degrees(a) + 90)
        for (dx, sx, sy) in ((-1.4, 0.55, 4.4), (1.4, 0.55, 4.4), (0.0, 2.8, 0.55)):
            nv.cube(stage, f"{root}/H{int(dx * 10) + 20}{int(sx * 10)}", (px + dx, py, z + 0.026), (sx, sy, 0.02), white)
    return {"landing_pads": sum(1 for p in pl["pads"].values() if not p.get("ship"))}


def build_vegetation(stage, nv, pl, forest_root):
    """Grass tufts, shrubs and a few trees on the land above the beach, clear of sites, pads and roads (instanced,
    visual only -- under /World/Coast/Nature, which the flight job's collider pass skips)."""
    rng = np.random.default_rng(pl["index"] * 31 + 5)
    c = pl["coast"]
    keys = ["grass", "shrub", "pine", "mature_tree"]
    inst = {k: [] for k in keys}
    blocked = [(s["xy"][0], s["xy"][1], s["size_r"] + 25) for s in pl["sites"]] + [(p["xy"][0], p["xy"][1], 22) for p in pl["pads"].values()] \
        + [(a["xy"][0], a["xy"][1], 16) for a in pl.get("spots", {}).values()]
    for _ in range(9000):
        x, y = float(rng.uniform(-EXTENT + 10, EXTENT - 10)), float(rng.uniform(-EXTENT + 10, EXTENT - 10))
        lat = y - shore_y(x, c)
        if lat < c["W"] * 1.35 or any(math.hypot(x - a, y - b) < r for a, b, r in blocked):
            continue
        z = terrain_height(x, y, pl)
        u = rng.random()
        k = "grass" if u < 0.62 else "shrub" if u < 0.88 else "pine" if u < 0.96 else "mature_tree"
        s = {"grass": 1.0, "shrub": 1.0, "pine": 1.4, "mature_tree": 0.7}[k] * float(rng.uniform(0.75, 1.3))
        inst[k].append((0, (x, y, z - 0.05), float(rng.uniform(0, 6.28)), (s, s, s)))
    out = {}
    for k in keys:
        if not inst[k]:
            continue
        mat = nv.create_asset_material(stage, f"/World/Materials/CoastV2_Veg_{k}", forest_root, nv.ASSET_IDS[k])
        nv.point_instancer(stage, f"/World/Coast/Nature/{k}", [(nv.asset_path(forest_root, k), 1.0, mat)], inst[k])
        out[f"veg_{k}"] = len(inst[k])
    return out
