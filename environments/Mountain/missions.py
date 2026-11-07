"""
Mountain missions for the Holybro X500 V2 -- Easy Mountain dataset (2026-10-20), 100 single-task missions.

Each simulation idx (1..100) has its own mountain landscape (scene/mountain_v2.plan) and ONE task:
  circle            orbit a site, a summit or a cable-car station once
  inspect           stop at a stand-off distance and hover facing it (hover clock starts once the drone faces it)
  pass_over         fly directly over it, camera looking down
  follow_valley     fly along the river, up or down the valley, low over the water
  follow_ridge      fly along a ridge crest from one summit / pass to another
  cross_pass        climb through a pass (saddle) in a ridge and land on the far side
  land_at           precision landing at a named site (hut, helipad, summit plateau, campsite...)
  follow_cable_car  fly up beside the cable-car line from the valley station to the top station
  cross_lake        fly the length of the lake low over the water
Take-off and landing places vary: painted pads, the rescue helipad, and unmarked spots (meadows, car park, hut, summit).

Every mission is flown offline against the PX4-like model (pilot.simulate_offline) and checked against every obstacle
box (8 m horizontal / 6 m vertical) and against the terrain AND the tree canopy (>= 12 m outside take-off / landing).
A failing mission regenerates the whole landscape (attempt + 1): landscape and mission are a pure function of idx.

The instruction is built from CLAUSES, each tied to its own mission beats (no two clauses share a beat: review item C3);
enrich.py turns them into instruction_alignment after the flight.
"""
import math

import numpy as np

DIFFICULTY_RULE = "1 task = Easy, 2 tasks = Medium, 3+ tasks = Hard (take-off, transit and landing are not tasks)"
N_SIMS = 100
PREFIX = "easy_mountain_"
H_CLEAR, V_CLEAR = 8.0, 6.0
TERRAIN_CLEAR = 12.0
# User 2026-10-21: no painted landing pads / helipads for take-off or landing. Simulations 1, 7, 28 were already published with
# the earlier mix (pads allowed) and must stay reproducible; every other simulation uses unmarked places only.
PADS_ALLOWED = {1, 7, 28}
NO_PADS = set(range(1, N_SIMS + 1)) - PADS_ALLOWED
MAX_DURATION_S = 420.0  # longer flights are re-planned (render cost; a 10-minute Easy flight is mostly transit)


def mission_ids():
    return [f"{PREFIX}{k:03d}" for k in range(1, N_SIMS + 1)]


# Benchmark simulations (2026-10-30): bench_mountain_<k> rebuilds the scene of published Easy simulation k (same landscape, season,
# sky, objects, take-off place and landing place) with ONE goal only: fly to the landing place and land there. The instruction names
# the landing place by its distance and direction from a visible landmark, so success = landing within the goal radius can be
# judged from the text alone. idx = BENCH_BASE + k; published missions 1..100 are unchanged.
BENCH_PREFIX = "bench_mountain_"
BENCH_BASE = 1000
COMPASS = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]


def index_of(mid):
    if mid.startswith(BENCH_PREFIX):
        return BENCH_BASE + int(mid[len(BENCH_PREFIX):])
    return int(mid[len(PREFIX):])


def tasks(beats):
    out = []
    for b in beats:
        t = b.get("task")
        if t and not any(o["task"] == t for o in out):
            d = {"task": t}
            if b.get("landmark"):
                d["landmark"] = b["landmark"]
            if t == "circle":
                d.update(laps=b.get("laps", 1.0), direction="counter-clockwise" if b.get("dir", 1) > 0 else "clockwise")
            if t == "inspect":
                d["seconds"] = b["s"]
            out.append(d)
    return out


def difficulty_from_tasks(n):
    return "Easy" if n <= 1 else "Medium" if n == 2 else "Hard"


def instruction(m, L, ground):
    return m["instruction"], [c["text"] for c in m["clauses"]]


# ---- clearance ------------------------------------------------------------------------------------------------
def _hdist(o, x, y):
    return math.hypot(max(o[0] - x, 0, x - o[1]), max(o[2] - y, 0, y - o[3]))


def max_top_near(obs, x, y, r, skip=()):
    return max([o[4] for o in obs if _hdist(o, x, y) < r and o[5] not in skip] + [-1e9])


def clearance_violation(frames, obs, beats, mv, pl):
    """First frame that enters an obstacle's safety box or flies too low over the terrain / tree canopy. The site the
    drone takes off from or lands at is ignored while it climbs out / descends onto it."""
    lab = {b.get("label") or b["type"]: b for b in beats}
    for f in frames[::3]:
        b = lab.get(f["beat"], {})
        near_pad = b.get("type") in ("takeoff", "descend", "land")
        own = b.get("own_site")
        for o in obs:
            if near_pad and own and o[5] == own:
                continue
            if f["z"] < o[4] + V_CLEAR and _hdist(o, f["x"], f["y"]) < H_CLEAR:
                if near_pad and o[4] < f["z"] + 0.5 and _hdist(o, f["x"], f["y"]) > 3.0:
                    continue
                return (round(f["t"], 1), f["beat"], o)
        if not near_pad and f["t"] > 3.0:
            g = mv.canopy_z(f["x"], f["y"], pl)
            wl = mv.water_level(f["x"], f["y"], pl)
            if wl is not None:
                g = max(g, wl)
            lim = b.get("clear", TERRAIN_CLEAR)
            if f["z"] - g < lim:
                return (round(f["t"], 1), f["beat"], ("terrain", round(g, 1)))
    return None


# ---- helpers ------------------------------------------------------------------------------------------------------
def _ground_max(mv, pl, pts, r=0.0):
    zs = []
    for x, y in pts:
        if r <= 0:
            zs.append(mv.canopy_z(x, y, pl))
        else:
            for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
                zs.append(mv.canopy_z(x + r * math.cos(a), y + r * math.sin(a), pl))
            zs.append(mv.canopy_z(x, y, pl))
        wl = mv.water_level(x, y, pl)
        if wl is not None:
            zs.append(wl)
    return max(zs)


def _to_pad(key, p, agl, spd):
    xy = p["xy"]
    return [dict(type="goto", xy=xy, agl=agl, speed=round(min(1.0, 0.85 * spd), 3), radius=4.0, label=f"fly to {key}", own_site=p.get("site"),
                 follow_terrain=True),
            dict(type="descend", agl=8.0, label="descend over the landing place", gimbal=5.0, own_site=p.get("site")),
            dict(type="land", xy=xy, label=f"land {key}", pad=key, own_site=p.get("site"))]


def _near(places, xy, lo, hi):
    return sorted([k for k, p in places.items() if lo < math.hypot(p["xy"][0] - xy[0], p["xy"][1] - xy[1]) < hi],
                  key=lambda k: math.hypot(places[k]["xy"][0] - xy[0], places[k]["xy"][1] - xy[1]))


def _r5(h):
    return int(max(5, 5 * round(float(h) / 5.0)))


def _pick(rng, options):
    return options[int(rng.integers(len(options)))]


START = ["Take off from {p}", "Lift off from {p}", "From {p}, take off", "Take off at {p}", "Rise from {p}"]
LAND_TO = ["then land on {p}", "and finish by landing on {p}", "then come down on {p}", "and set down on {p}", "then touch down on {p}"]
LAND_BACK = ["then fly back and land where you took off", "then return to your take-off point and land",
             "and come back to land where you started"]


def place_words(p):
    return p["words"]


# ---- mission ----------------------------------------------------------------------------------------------------
def make_mission(idx, pl, mv, pilot, attempt):
    rng = np.random.default_rng(5_000_011 * idx + 13 + 77_777 * attempt)
    task = pl["task"]
    v = pl["valley"]
    places = {**pl["pads"], **pl.get("spots", {})}
    if idx in NO_PADS:  # user 2026-10-21 (Simulation 39 redo): take off and land on unmarked places only, no painted pad / helipad
        places = dict(pl.get("spots", {}))
    vr = np.random.default_rng(911_003 * idx + 23 * attempt)
    agl = float(vr.choice([40.0, 45.0, 50.0, 55.0, 60.0, 70.0, 80.0]))
    spd = float(vr.uniform(0.75, 1.15))
    obs = mv.obstacles(pl)
    ground = lambda x, y: mv.surface_z(x, y, pl, water=True)  # noqa: E731
    LM = pl["landmarks"]
    L = {}
    target_site = [None]

    def pool(xy, lo, hi, exclude=(), side=None):
        cands = [k for k in _near(places, xy, lo, hi) if k not in exclude and places[k].get("site") != target_site[0]]
        if side is not None:
            cands = [k for k in cands if (places[k]["xy"][1] - float(mv.valley_y(places[k]["xy"][0], v))) * side > 0 or abs(
                places[k]["xy"][1] - float(mv.valley_y(places[k]["xy"][0], v))) < float(mv.valley_w(places[k]["xy"][0], v))]
        if not cands:
            cands = [k for k in _near(places, xy, 30, 900) if k not in exclude] or [k for k in sorted(places) if k not in exclude]
        return str(rng.choice(cands[:4]))

    def pick_land(xy, start, z_end=None):
        """Landing place near where the task ends; with the task's end height known, prefer places reached without first
        climbing over a ridge (a long climb + a long vertical descent is minutes of video of nothing)."""
        sp = places[start]
        if rng.random() < 0.15 and 40 < math.hypot(sp["xy"][0] - xy[0], sp["xy"][1] - xy[1]) < 450:
            return start
        cands = [k for k in _near(places, xy, 40, 500) if k != start and places[k].get("site") != target_site[0]] or \
            [k for k in _near(places, xy, 30, 900) if k != start]
        if z_end is not None and cands:
            def cost(k):
                p = places[k]["xy"]
                hump = max(0.0, max(mv.terrain_height(xy[0] + (p[0] - xy[0]) * t, xy[1] + (p[1] - xy[1]) * t, pl) for t in np.linspace(0, 1, 15)) + agl - z_end)
                drop = max(0.0, z_end - places[k]["z"])
                return hump * 2.0 + 0.3 * drop + 0.15 * math.hypot(p[0] - xy[0], p[1] - xy[1])
            cands = sorted(cands, key=cost)
        return str(rng.choice(cands[:3]))

    def start_clause(s):
        return _pick(rng, START).format(p=place_words(places[s]))

    def land_clause(land, start):
        if land == start:
            return _pick(rng, LAND_BACK)
        return _pick(rng, LAND_TO).format(p=place_words(places[land]))

    beats, clauses = [], []

    def takeoff(start):
        beats.append(dict(type="takeoff", agl=min(agl, 30.0), label="take off", own_site=places[start].get("site")))
        clauses.append(dict(text=start_clause(start), beats=[0]))

    def finish(land, start):
        i0 = len(beats)
        beats.extend(_to_pad(land, places[land], agl, spd))
        clauses.append(dict(text=land_clause(land, start), beats=[i0, i0 + 1, i0 + 2]))

    def transit(xy, label):
        """Cross the landscape at cruise height above the ground first: a straight line at the task's own (low) height could
        run into a ridge between the take-off place and the task."""
        beats.append(dict(type="goto", xy=tuple(xy), agl=agl, speed=round(min(1.0, 0.85 * spd), 3), radius=6.0, label=label, follow_terrain=True))
        return len(beats) - 1

    def look_of(t):
        if t.get("summit"):
            return (t["xy"][0], t["xy"][1], t["top_z"] - 8.0)
        return (t["xy"][0], t["xy"][1], t["base_z"] + 0.4 * max(2.0, t["top_z"] - t["base_z"]))

    focus = None
    if task in ("circle", "inspect", "pass_over", "land_at"):
        names = sorted(n for n in LM if LM[n]["kind"] not in ("pass", "lake", "tarn"))
        if task == "pass_over":
            names = [n for n in names if LM[n]["kind"] not in ("summit",) or rng.random() < 0.5]
        if task == "land_at":
            names = sorted({p["site"] for p in places.values() if p.get("site")})
            # the landing place must be at that site
        name = str(rng.choice(names))
        t = LM[name]
        L[name] = t
        site_of = name.split("/")[0]
        target_site[0] = site_of
        tx, ty = t["xy"]
        focus = t["words"]
        if task == "land_at":
            lands = [k for k, p in places.items() if p.get("site") == name]
            land = str(rng.choice(lands))
            target_site[0] = name
            start = pool((tx, ty), 150, 700)
            takeoff(start)
            lp = places[land]
            i0 = len(beats)
            beats.append(dict(type="goto", xy=lp["xy"], agl=agl, speed=round(min(1.0, 0.85 * spd), 3), radius=4.0, label=f"fly to {name}",
                              landmark=name, look_at=look_of(t), follow_terrain=True))
            beats.append(dict(type="descend", agl=8.0, label="descend over the landing place", gimbal=5.0, own_site=lp.get("site")))
            beats.append(dict(type="land", xy=lp["xy"], label=f"land {land}", pad=land, own_site=lp.get("site"), task="land_at", landmark=name))
            clauses.append(dict(text=_pick(rng, ["fly to {t}", "head over to {t}", "make your way to {t}"]).format(t=t["words"]), beats=[i0]))
            clauses.append(dict(text=_pick(rng, ["and land on {p}", "and touch down on {p}", "and set down precisely on {p}"]).format(p=lp["words"]),
                                beats=[i0 + 1, i0 + 2]))
            text = f"{clauses[0]['text']}, {clauses[1]['text']} {clauses[2]['text']}."
            m = _finish_mission(idx, pl, mv, pilot, attempt, rng, task, start, land, beats, clauses, text, focus)
            return _check(m, L, beats, obs, mv, pl, pilot, ground)
        start = pool((tx, ty), 90, 450)
        land = pick_land((tx, ty), start)
        takeoff(start)
        look = look_of(t)
        size = t.get("size_r", 20.0)
        sx, sy = places[start]["xy"]
        a0 = math.atan2(sy - ty, sx - tx)
        if task == "circle":
            radius = float(_r5(max(30.0, min(140.0, size + float(rng.uniform(22, 45)) if not t.get("summit") else float(rng.uniform(70, 120))))))
            ring = [(tx + radius * math.cos(a), ty + radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 48)]
            g = _ground_max(mv, pl, ring, r=6.0)
            z = max(g + float(rng.uniform(22, 32)), max(max_top_near(obs, x, y, H_CLEAR + 2, skip=()) for x, y in ring) + V_CLEAR + 2,
                    t["top_z"] + (8.0 if not t.get("summit") else 4.0))
            d = int(rng.choice([1, -1]))
            i0 = transit((tx + radius * math.cos(a0), ty + radius * math.sin(a0)), f"cruise toward {name}")
            beats.append(dict(type="goto", xy=(tx + radius * math.cos(a0), ty + radius * math.sin(a0)), z=z, speed=round(min(1.0, 0.8 * spd), 3),
                              radius=3.0, label=f"fly to {name}", landmark=name, look_at=look))
            beats.append(dict(type="orbit", center=look, radius=radius, z=z, laps=1.0, dir=d, speed=float(rng.uniform(3.5, 5.0)),
                              label=f"orbit {name}", landmark=name, task="circle"))
            way = "counter-clockwise" if d > 0 else "clockwise"
            clauses.append(dict(text=_pick(rng, ["fly to {t}", "head for {t}", "go to {t}"]).format(t=t["words"]), beats=[i0, i0 + 1]))
            clauses.append(dict(text=_pick(rng, ["circle it once {w} at a radius of about {r} m, keeping the camera on it",
                                                 "make one full {w} circle around it about {r} m out, camera pointed at it",
                                                 "orbit it once {w}, about {r} m away, filming it the whole time"]).format(w=way, r=_r5(radius)),
                                beats=[i0 + 2]))
            text = f"{clauses[0]['text']}, {clauses[1]['text']} and {clauses[2]['text']}, "
        elif task == "inspect":
            stand = float(_r5(max(35.0, min(110.0, size + float(rng.uniform(18, 40)) if not t.get("summit") else float(rng.uniform(70, 110))))))
            a = a0 + float(rng.uniform(-0.6, 0.6))
            hx, hy = tx + stand * math.cos(a), ty + stand * math.sin(a)
            g = _ground_max(mv, pl, [(hx, hy)], r=10.0)
            z = max(g + float(rng.uniform(18, 30)), max_top_near(obs, hx, hy, H_CLEAR + 4) + V_CLEAR + 2,
                    # a summit hover stays high enough that the top is >= 6 deg inside the frame's upper edge (the gimbal cannot
                    # look up; vertical FOV 27.6 deg): 25 m below a summit 95 m away put it out of view (Mountain S020, 2026-10-22)
                    # an object standing on a summit (cross, observatory, mast) needs the same rule: g + 70 left the drone 40 m
                    # below a summit cross 34 m away, the cross 50 deg above the frame (Mountain S085, 2026-10-29)
                    max(min(t["top_z"] + 6.0, g + 70.0), t["top_z"] - 0.14 * stand) if not t.get("summit") else t["top_z"] - min(float(rng.uniform(0, 25)), 0.14 * stand))
            z = max(z, g + 15.0)
            secs = float(rng.choice([5.0, 6.0, 7.0, 8.0, 10.0]))
            i0 = transit((hx, hy), f"cruise toward {name}")
            beats.append(dict(type="goto", xy=(hx, hy), z=z, speed=0.75, radius=3.0, label=f"fly to {name}", landmark=name, look_at=look))
            beats.append(dict(type="hover", s=secs, look_at=look, label=f"inspect {name}", landmark=name, align_first=True, task="inspect"))
            clauses.append(dict(text=_pick(rng, ["fly to {t}", "approach {t}", "head toward {t}"]).format(t=t["words"]), beats=[i0, i0 + 1]))
            clauses.append(dict(text=_pick(rng, ["stop about {d} m from it and hover for {s} seconds looking at it",
                                                 "hold position roughly {d} m away and watch it for {s} seconds",
                                                 "hover about {d} m short of it, facing it, for {s} seconds"]).format(d=_r5(stand), s=int(secs)),
                                beats=[i0 + 2]))
            text = f"{clauses[0]['text']}, {clauses[1]['text']} and {clauses[2]['text']}, "
        else:  # pass_over
            over = float(rng.uniform(25, 40))
            g = _ground_max(mv, pl, [(tx, ty)], r=15.0)
            z = max(g + over, max_top_near(obs, tx, ty, 30.0) + 12.0)
            # a straight line over the target, 70 m before to 60 m past it, along the heading that keeps the line lowest
            best = None
            for hd in np.linspace(0, 2 * math.pi, 12, endpoint=False):
                ax, ay = tx - math.cos(hd) * 70.0, ty - math.sin(hd) * 70.0
                bx, by = tx + math.cos(hd) * 60.0, ty + math.sin(hd) * 60.0
                zl_ = max(z, _ground_max(mv, pl, [(ax, ay), (bx, by), ((ax + tx) / 2, (ay + ty) / 2), ((bx + tx) / 2, (by + ty) / 2)], r=8.0) + 20.0)
                cost = zl_ + 0.02 * math.hypot(ax - sx, ay - sy)
                if best is None or cost < best[0]:
                    best = (cost, ax, ay, bx, by, zl_)
            _, ax, ay, bx, by, zl = best
            gim = float(rng.choice([55.0, 60.0, 65.0, 70.0]))
            i0 = transit((ax, ay), f"cruise toward {name}")
            beats.append(dict(type="goto", xy=(ax, ay), z=zl, speed=round(min(1.0, 0.8 * spd), 3), radius=4.0, label=f"approach {name}", landmark=name,
                              look_at=look))
            beats.append(dict(type="goto", xy=(bx, by), z=zl, speed=0.55, radius=5.0, label=f"fly over {name}", landmark=name, gimbal=gim,
                              task="pass_over", over_xy=(tx, ty)))
            clauses.append(dict(text=_pick(rng, ["line up on {t}", "make for {t}", "head for {t}"]).format(t=t["words"]), beats=[i0, i0 + 1]))
            clauses.append(dict(text=_pick(rng, ["pass directly over it about {h} m above, camera looking down",
                                                 "fly straight across the top of it at about {h} m with the camera tilted down",
                                                 "cross right over it roughly {h} m up, looking down at it"]).format(h=_r5(zl - g)),
                                beats=[i0 + 2]))
            text = f"{clauses[0]['text']}, {clauses[1]['text']} and {clauses[2]['text']}, "
        finish(land, start)
        text = text + clauses[-1]["text"] + "."
    elif task == "follow_valley":
        lk = v.get("lake")
        for _ in range(80):
            xa = float(rng.uniform(-560, 450))
            span = float(rng.uniform(260, 560)) * (1 if rng.random() < 0.5 else -1)
            xb = max(-600.0, min(600.0, xa + span))
            if abs(xb - xa) < 220:
                continue
            if lk and min(xa, xb) < lk["x"] + lk["L"] and max(xa, xb) > lk["x"] - lk["L"]:
                continue
            break
        else:
            raise ValueError("no river stretch")
        h_over = float(rng.choice([15.0, 18.0, 20.0, 25.0]))
        n = max(5, int(abs(xb - xa) // 28))
        pts = [(xa + (xb - xa) * k / n, float(mv.river_y(xa + (xb - xa) * k / n, v))) for k in range(n + 1)]
        start = pool(pts[0], 60, 500)
        land = pick_land(pts[-1], start)
        takeoff(start)
        z0 = _ground_max(mv, pl, [pts[0]], r=8.0) + max(h_over, 15.0)
        i_tr = transit(pts[0], "cruise toward the river")
        beats.append(dict(type="goto", xy=pts[0], z=z0, speed=round(min(1.0, 0.8 * spd), 3), radius=4.0, label="fly to the river"))
        fb, hh = [], []
        for p in pts[1:]:
            gz = max(float(mv.river_level(p[0], v)) + h_over, _ground_max(mv, pl, [p], r=9.0) + 12.0, max_top_near(obs, p[0], p[1], H_CLEAR + 3) + V_CLEAR + 2)
            hh.append(gz - float(mv.river_level(p[0], v)))
            fb.append(len(beats))
            beats.append(dict(type="goto", xy=p, z=gz, speed=0.65, radius=5.0, label="follow the river", gimbal=float(rng.choice([12.0, 18.0, 25.0])),
                              task="follow_valley", clear=9.0))
        down = (xb - xa) * (-v["slope"]) > 0 or (abs(v["slope"]) < 0.004 and rng.random() < 0.5)
        way = "east" if xb > xa else "west"
        low = [s for s in pl["sites"] if s.get("summit") is None and s["kind"] not in ("wind farm", "radio mast", "observatory", "summit cross")
               and abs(s["xy"][1] - float(mv.valley_y(s["xy"][0], v))) < float(mv.valley_w(s["xy"][0], v)) + 80.0] or pl["sites"]
        near_a = min(low, key=lambda s: math.hypot(s["xy"][0] - pts[0][0], s["xy"][1] - pts[0][1]))
        clauses.append(dict(text=_pick(rng, ["fly down to the river near {a}", "drop down to the river by {a}", "go to the riverbank near {a}"]).format(a=near_a["words"]),
                            beats=[i_tr, i_tr + 1]))
        clauses.append(dict(text=_pick(rng, ["follow the river {d} ({w}) for about {L} m, about {h} m above the water",
                                             "fly {d} along the river toward the {w} for roughly {L} m, staying about {h} m over the water",
                                             "trace the river {d}, heading {w}, for about {L} m at about {h} m above it"]).format(
            d="downstream" if down else "upstream", w=way, L=int(round(abs(xb - xa), -1)), h=_r5(np.median(hh))), beats=fb))
        finish(land, start)
        text = f"{clauses[0]['text']}, {clauses[1]['text']} and {clauses[2]['text']}, {clauses[3]['text']}."
        focus = "the river"
        L["river"] = dict(xy=pts[len(pts) // 2], words="the river")
    elif task == "follow_ridge":
        cands = []
        for side in (1, -1):
            sm = [p for p in pl["summits"] if p["side"] == side] + [c for c in pl["cols"] if c["side"] == side]
            for a in sm:
                for b in sm:
                    if a is not b and 200 < abs(b["x"] - a["x"]) < 480 and (a.get("name", "").startswith("peak") or b.get("name", "").startswith("peak")):
                        cands.append((side, a, b))
        if not cands:
            raise ValueError("no ridge")
        side, a, b = cands[int(rng.integers(len(cands)))]
        n = max(6, int(abs(b["x"] - a["x"]) // 30))
        crest = []
        for k in range(n + 1):
            x = a["x"] + (b["x"] - a["x"]) * k / n
            ylo = a["y"] + (b["y"] - a["y"]) * k / n
            best = None
            for dy in np.linspace(-90, 90, 37):
                y = ylo + dy
                if abs(y) > mv.EXTENT - 20:
                    continue
                z = mv.terrain_height(x, y, pl)
                if best is None or z > best[2]:
                    best = (x, y, z)
            crest.append(best)
        ys = np.convolve([c[1] for c in crest], np.ones(3) / 3, mode="same")
        ys[0], ys[-1] = crest[0][1], crest[-1][1]
        crest = [(c[0], float(y_)) for c, y_ in zip(crest, ys)]
        above = float(rng.choice([25.0, 30.0, 35.0, 40.0]))
        start = pool(crest[0], 120, 700, side=side)
        land = pick_land(crest[-1], start, z_end=mv.terrain_height(*crest[-1], pl) + above)
        takeoff(start)
        z0 = _ground_max(mv, pl, [crest[0]], r=12.0) + above
        i_tr = transit(crest[0], "cruise toward the ridge")
        beats.append(dict(type="goto", xy=crest[0], z=z0, speed=round(min(1.0, 0.8 * spd), 3), radius=5.0, label="climb to the ridge",
                          landmark=a["name"], look_at=(crest[-1][0], crest[-1][1], mv.terrain_height(*crest[-1], pl))))
        fb, hh = [], []
        for k, p in enumerate(crest[1:], start=1):
            gz = max(_ground_max(mv, pl, [p], r=14.0) + above, max_top_near(obs, p[0], p[1], H_CLEAR + 4) + V_CLEAR + 3)
            hh.append(gz - mv.terrain_height(p[0], p[1], pl))
            fb.append(len(beats))
            gim = float(rng.choice([15.0, 20.0, 28.0]))  # drawn as before (keeps the plan's random sequence)
            # the camera is aimed at the crest one waypoint further on (60 m past the end on the last leg): a fixed tilt left a
            # descending crest 40-70 deg below a camera tilted 15-28 deg, out of the frame (Mountain S026: 20 %, 2026-10-24)
            if k + 1 < len(crest):
                ax, ay = crest[k + 1]
            else:
                dx, dy = crest[-1][0] - crest[-2][0], crest[-1][1] - crest[-2][1]
                dl = math.hypot(dx, dy) or 1.0
                ax, ay = crest[-1][0] + dx / dl * 60.0, crest[-1][1] + dy / dl * 60.0
            beats.append(dict(type="goto", xy=p, z=gz, speed=0.6, radius=6.0, label="follow the ridge", gimbal=gim, steady_climb=True,
                              look_at=(ax, ay, mv.terrain_height(ax, ay, pl) + 2.0),
                              landmark=b["name"],  # the summit / pass the crest flight heads for (ahead of the camera)
                              task="follow_ridge"))
        ridge = f"the {mv.side_words(side)} ridge"
        A, B = pl["landmarks"][a["name"]]["words"], pl["landmarks"][b["name"]]["words"]
        L[a["name"]], L[b["name"]] = pl["landmarks"][a["name"]], pl["landmarks"][b["name"]]
        at_a = A if a["name"].startswith("pass") else f"{ridge} at {A}"
        clauses.append(dict(text=_pick(rng, ["climb to {r}", "fly up to {r}", "gain height and reach {r}"]).format(r=at_a), beats=[i_tr, i_tr + 1]))
        clauses.append(dict(text=_pick(rng, ["follow the crest, about {h} m above it, for about {L} m to {b}",
                                             "fly along the ridgeline about {h} m above the rocks for roughly {L} m until {b}",
                                             "trace the crest about {h} m up, for about {L} m, as far as {b}"]).format(
            L=int(round(sum(math.hypot(crest[k + 1][0] - crest[k][0], crest[k + 1][1] - crest[k][1]) for k in range(len(crest) - 1)), -1)), h=_r5(np.median(hh)), b=B),
                            beats=fb))
        finish(land, start)
        text = f"{clauses[0]['text']}, {clauses[1]['text']}, {clauses[2]['text']}, {clauses[3]['text']}."
        focus = ridge
    elif task == "cross_pass":
        cols = [c for c in pl["cols"] if abs(c["x"]) < mv.EXTENT - 120]
        if not cols:
            raise ValueError("no pass")
        c = cols[int(rng.integers(len(cols)))]
        side = c["side"]
        far = [k for k, p in places.items() if (p["xy"][1] - float(mv.valley_y(p["xy"][0], v))) * side > float(mv.valley_w(p["xy"][0], v)) + v["Dw"] * 0.95
               and 80 < math.hypot(p["xy"][0] - c["x"], p["xy"][1] - c["y"]) < 520]
        near = [k for k, p in places.items() if (p["xy"][1] - float(mv.valley_y(p["xy"][0], v))) * side < float(mv.valley_w(p["xy"][0], v)) + v["Dw"] * 0.6
                and 150 < math.hypot(p["xy"][0] - c["x"], p["xy"][1] - c["y"]) < 750]
        if not far or not near:
            raise ValueError("no places on both sides of the pass")
        start = str(rng.choice(near))
        land = str(rng.choice(far))
        takeoff(start)
        above = float(rng.choice([25.0, 30.0, 35.0, 40.0]))
        # pass line: perpendicular to the ridge, through the col
        ny = side
        p_in = (c["x"], c["y"] - ny * 70.0)
        p_out = (c["x"], c["y"] + ny * 80.0)
        zc = _ground_max(mv, pl, [(c["x"], c["y"]), p_in, p_out], r=20.0) + above
        L[c["name"]] = pl["landmarks"][c["name"]]
        i0 = transit(p_in, "cruise toward the pass")
        beats.append(dict(type="goto", xy=p_in, z=zc, speed=round(min(1.0, 0.8 * spd), 3), radius=5.0, label="climb toward the pass", landmark=c["name"],
                          look_at=(c["x"], c["y"], c["z"])))
        beats.append(dict(type="goto", xy=(c["x"], c["y"]), z=zc, speed=0.6, radius=5.0, label="fly through the pass", landmark=c["name"], task="cross_pass",
                          gimbal=12.0))
        beats.append(dict(type="goto", xy=p_out, z=zc, speed=0.6, radius=5.0, label="cross to the far side", landmark=c["name"], gimbal=20.0))
        clauses.append(dict(text=_pick(rng, ["climb toward {c}", "fly up the slope toward {c}", "head up to {c}"]).format(c=c["words"]), beats=[i0, i0 + 1]))
        clauses.append(dict(text=_pick(rng, ["cross through it about {h} m above the saddle into the next valley",
                                             "fly over the saddle roughly {h} m above the ground and drop into the valley beyond",
                                             "go through the gap about {h} m above the lowest point and over to the other side"]).format(h=int(above)),
                            beats=[i0 + 2, i0 + 3]))
        i1 = len(beats)
        lp = places[land]
        beats.extend(_to_pad(land, lp, agl, spd))
        beats[i1]["agl"] = 30.0
        clauses.append(dict(text=_pick(rng, ["then land on {p}", "and land on {p}", "then set down on {p}"]).format(p=lp["words"]), beats=[i1, i1 + 1, i1 + 2]))
        text = f"{clauses[0]['text']}, {clauses[1]['text']}, {clauses[2]['text']}, {clauses[3]['text']}."
        focus = c["words"]
    elif task == "follow_cable_car":
        s = next(s for s in pl["sites"] if s["kind"] == "cable car")
        line = s["line"]
        lat = float(rng.choice([22.0, 26.0, 30.0])) * (1 if rng.random() < 0.5 else -1)
        above = float(rng.choice([10.0, 14.0, 18.0]))
        ang = s["yaw"]
        nx, ny = -math.sin(ang), math.cos(ang)
        x0, y0, z0 = s["valley"]
        x1, y1, z1 = s["top_st"]
        n = max(6, int(s["span"] // 35))
        pts = []
        for k in range(n + 1):
            t = k / n
            # cable height at t (piecewise over the supports)
            dist = t * math.hypot(line[-1][0] - line[0][0], line[-1][1] - line[0][1])
            acc, zc = 0.0, line[0][2]
            for j in range(len(line) - 1):
                seg = math.hypot(line[j + 1][0] - line[j][0], line[j + 1][1] - line[j][1])
                if acc + seg >= dist or j == len(line) - 2:
                    u = min(1.0, (dist - acc) / max(seg, 1e-6))
                    zc = line[j][2] + (line[j + 1][2] - line[j][2]) * u
                    break
                acc += seg
            px, py = x0 + (x1 - x0) * t + nx * lat, y0 + (y1 - y0) * t + ny * lat
            pts.append((px, py, max(zc + above, _ground_max(mv, pl, [(px, py)], r=8.0) + 16.0), zc))
        L[s["name"]] = pl["landmarks"][s["name"]]
        start = pool((x0, y0), 80, 500)
        land_opts = [k for k, p in places.items() if p.get("site") == s["name"]]
        land = str(rng.choice(land_opts)) if land_opts and rng.random() < 0.6 else pick_land((x1, y1), start, z_end=pts[-1][2])
        takeoff(start)
        i0 = transit(pts[0][:2], "cruise toward the cable car")
        beats.append(dict(type="goto", xy=pts[0][:2], z=pts[0][2], speed=round(min(1.0, 0.8 * spd), 3), radius=4.0, label="fly to the valley station",
                          landmark=f"{s['name']}/valley_station", look_at=(x1, y1, z1)))
        fb = []
        for p in pts[1:]:
            fb.append(len(beats))
            beats.append(dict(type="goto", xy=p[:2], z=p[2], speed=0.6, radius=5.0, label="follow the cable car", task="follow_cable_car", steady_climb=True,
                              gimbal=float(rng.choice([10.0, 15.0, 20.0])), landmark=s["name"]))
        side = "left" if lat > 0 else "right"
        clauses.append(dict(text=_pick(rng, ["fly to the valley station of the cable car", "go to the cable car's valley station",
                                             "head for the bottom station of the cable car"]), beats=[i0, i0 + 1]))
        clauses.append(dict(text=_pick(rng, ["follow the cable line up to the top station, about {a} m to its {s} and {h} m above the cables",
                                             "fly up along the cables to the top station, keeping them about {a} m to your {o} and staying {h} m higher",
                                             "climb beside the cable car line all the way to the top station, about {a} m off to its {s}, {h} m above the cables"]).format(
            a=int(abs(lat)), s=side, o="right" if side == "left" else "left", h=_r5(np.median([p[2] - p[3] for p in pts]))), beats=fb))
        finish(land, start)
        text = f"{clauses[0]['text']}, {clauses[1]['text']} and {clauses[2]['text']}, {clauses[3]['text']}."
        focus = "the cable car"
    elif task == "cross_lake":
        lk = v.get("lake")
        if lk:
            xa, xb = lk["x"] - lk["L"] * 0.97, lk["x"] + lk["L"] * 0.97
            if rng.random() < 0.5:
                xa, xb = xb, xa
            n = max(5, int(abs(xb - xa) // 30))
            pts = [(xa + (xb - xa) * k / n, float(mv.valley_y(xa + (xb - xa) * k / n, v))) for k in range(n + 1)]
            lvl = lk["z"]
            what = "the lake"
            L["lake"] = pl["landmarks"]["lake"]
        else:
            tn = v["tarns"][0]
            a = float(rng.uniform(0, 2 * math.pi))
            pts = [(tn["x"] + tn["r"] * 1.05 * math.cos(a) * (1 - 2 * k / 6), tn["y"] + tn["r"] * 1.05 * math.sin(a) * (1 - 2 * k / 6)) for k in range(7)]
            lvl = tn["z"]
            what = pl["landmarks"]["tarn_1"]["words"]
            L["tarn_1"] = pl["landmarks"]["tarn_1"]
        h_over = float(rng.choice([10.0, 12.0, 15.0, 20.0]))
        start = pool(pts[0], 60, 500)
        land = pick_land(pts[-1], start)
        takeoff(start)
        i0 = transit(pts[0], "cruise toward the lake")
        beats.append(dict(type="goto", xy=pts[0], z=_ground_max(mv, pl, [pts[0]], r=10.0) + 15.0, speed=round(min(1.0, 0.8 * spd), 3), radius=4.0,
                          label="fly to the end of the lake"))
        fb, hh = [], []
        for p in pts[1:]:
            fb.append(len(beats))
            hh.append(max(lvl + h_over, _ground_max(mv, pl, [p], r=10.0) + 10.0) - lvl)
            beats.append(dict(type="goto", xy=p, z=max(lvl + h_over, _ground_max(mv, pl, [p], r=10.0) + 10.0), speed=0.6, radius=5.0, label="cross the lake",
                              task="cross_lake", gimbal=float(rng.choice([10.0, 15.0, 20.0])), clear=8.0))
        way = "east" if pts[-1][0] > pts[0][0] else "west"
        clauses.append(dict(text=_pick(rng, ["fly to the {e} end of {w}", "go to the {e} shore of {w}", "head to the {e} tip of {w}"]).format(
            e="west" if way == "east" else "east", w=what), beats=[i0, i0 + 1]))
        length = int(round(sum(math.hypot(pts[k + 1][0] - pts[k][0], pts[k + 1][1] - pts[k][1]) for k in range(len(pts) - 1)), -1))
        clauses.append(dict(text=_pick(rng, ["fly its whole length, about {L} m, {h} m above the water",
                                             "skim across it toward the {d} for about {L} m, staying about {h} m over the water",
                                             "cross the full {L} m of water heading {d}, about {h} m above the surface"]).format(L=length, h=_r5(np.median(hh)), d=way),
                            beats=fb))
        finish(land, start)
        text = f"{clauses[0]['text']}, {clauses[1]['text']} and {clauses[2]['text']}, {clauses[3]['text']}."
        focus = what
    else:
        raise ValueError(task)
    m = _finish_mission(idx, pl, mv, pilot, attempt, rng, task, start, land, beats, clauses, text, focus)
    m["cruise"] = {"transit_agl_m": agl, "speed_scale": round(spd, 3)}
    return _check(m, L, beats, obs, mv, pl, pilot, ground)


def _finish_mission(idx, pl, mv, pilot, attempt, rng, task, start, land, beats, clauses, text, focus):
    for b in beats:  # mountain beats climb / descend a lot inside one beat: time budget (stuck detection) grows with it
        if b["type"] in ("goto", "descend"):
            b["budget_extra"] = 150.0
    places = {**pl["pads"], **pl.get("spots", {})}
    sp = places[start]
    text = text[0].upper() + text[1:]
    m = dict(id=f"{PREFIX}{idx:03d}", difficulty="Easy", start=start, start_xy=tuple(sp["xy"]), start_z=sp["z"], yaw_deg=float(round(rng.uniform(-180, 180))),
             beats=beats, instruction=text, clauses=clauses, task_type=task, target_description=focus, layout_index=idx, attempt=attempt, land_pad=land)
    m["tasks"] = tasks(beats)
    return m


def _check(m, L, beats, obs, mv, pl, pilot, ground):
    if difficulty_from_tasks(len(m["tasks"])) != "Easy" or len(m["tasks"]) != 1:
        raise ValueError(f"{m['id']}: {len(m['tasks'])} tasks")
    used = [b for c in m["clauses"] for b in c["beats"]]
    if len(used) != len(set(used)):
        raise ValueError(f"{m['id']}: two clauses share a beat")
    frames, p = pilot.simulate_offline(beats, ground, m["start_xy"], math.radians(m["yaw_deg"]), max_s=900.0)
    bad = clearance_violation(frames, obs, beats, mv, pl)
    ok = p.done and not p.failed and bad is None and frames[-1]["t"] <= MAX_DURATION_S
    m["offline_check"] = {"done": p.done, "failed": p.failed, "clearance_violation": bad and [bad[0], bad[1], str(bad[2][-1])],
                          "duration_s": round(frames[-1]["t"], 1)}
    return m, L, ok


def bench_mission(k, pl, m0, mv, pilot):
    """Goal-only mission in the scene of published simulation k: take-off place, heading and landing place of the original."""
    places = pl["pads_all"]
    start, land = m0["start"], m0["land_pad"]
    sp, lp = places[start], places[land]
    gx, gy = lp["xy"]
    # reference landmark: the nearest site (not a summit / pass / lake) to the goal; the goal is stated relative to it
    refs = [(math.hypot(t["xy"][0] - gx, t["xy"][1] - gy), n) for n, t in pl["landmarks"].items()
            if t["kind"] not in ("summit", "pass", "lake", "tarn")]
    d, ref = min(refs)
    t = pl["landmarks"][ref]
    ang = math.degrees(math.atan2(gy - t["xy"][1], gx - t["xy"][0]))
    way = COMPASS[int(round(ang / 45.0)) % 8]
    agl = (m0.get("cruise") or {}).get("transit_agl_m", 45.0)
    spd = (m0.get("cruise") or {}).get("speed_scale", 1.0)
    beats = [dict(m0["beats"][0])]
    beats.extend(_to_pad(land, lp, agl, spd))
    # no look_at on the goal itself (a look-at point under the drone gives no heading: the pilot holds its heading over the goal), and
    # no landmark on the task: the goal is an unmarked place, the reference landmark is behind or beside the drone on the approach
    beats[-1].update(task="land_at")
    for b in beats:
        if b["type"] in ("goto", "descend"):
            b["budget_extra"] = 150.0
    first = m0["clauses"][0]["text"]
    spot = lp.get("spot") or "landing place"
    goal = f"the {spot} about {_r5(d)} m {way} of {t['words']}" if d >= 25 else f"the {spot} next to {t['words']}"
    clauses = [dict(text=first, beats=[0]),
               dict(text=f"fly to {goal}", beats=[1]),
               dict(text="and land precisely in the middle of it", beats=[2, 3])]
    text = f"{clauses[0]['text']}, {clauses[1]['text']} {clauses[2]['text']}."
    m = dict(id=f"{BENCH_PREFIX}{k:03d}", difficulty="Easy", start=start, start_xy=tuple(sp["xy"]), start_z=sp["z"], yaw_deg=m0["yaw_deg"],
             beats=beats, instruction=text, clauses=clauses, task_type="land_at", target_description=goal, layout_index=k,
             attempt=m0["attempt"], land_pad=land, source_simulation=m0["id"],
             goal={"xyz": [gx, gy, lp["z"]], "place": land, "words": goal, "reference_landmark": ref, "reference_distance_m": round(d, 1),
                   "reference_direction": way})
    m["tasks"] = tasks(beats)
    m["cruise"] = m0.get("cruise")
    L = {ref: t}
    m, L, ok = _check(m, L, beats, mv.obstacles(pl), mv, pl, pilot, lambda x, y: mv.surface_z(x, y, pl, water=True))
    if not ok:
        raise RuntimeError(f"{m['id']}: goal-only flight fails the offline check: {m['offline_check']}")
    return pl, m, L


def mission_and_plan(idx, mv, pilot, max_attempts=40):
    if idx > BENCH_BASE:
        k = idx - BENCH_BASE
        pl, m0, _ = mission_and_plan(k, mv, pilot, max_attempts)
        return bench_mission(k, pl, m0, mv, pilot)
    last = None
    for attempt in range(max_attempts):
        try:
            pl = mv.plan(idx, attempt)
            m, L, ok = make_mission(idx, pl, mv, pilot, attempt)
        except ValueError as e:
            last = str(e)
            continue
        if ok:
            pl["pads_all"] = {**pl["pads"], **pl.get("spots", {})}
            return pl, m, L
        last = m["offline_check"]
    raise RuntimeError(f"no clear mission for {idx}: {last}")
