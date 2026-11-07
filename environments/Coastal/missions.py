"""
Coastal missions for the Holybro X500 V2 -- Easy Coastal dataset (2026-10-13), 100 single-task missions.

Each simulation idx (1..100) has its own coast (scene/coastal_v2.plan) and ONE task:
  circle        orbit a ship, a shore site or an islet once
  inspect       stop at a stand-off distance and hover facing it (the hover clock starts only once the drone faces
                the target -- review item #3: a "looking at it" label must be true)
  pass_over     fly directly over it, camera looking down
  follow_shore  fly along the coastline from one place to another at low height over the water
  land_on_ship  fly out to a ship and land on its helicopter deck (new spatial task, review item #8)
Some missions take off from a ship's helicopter deck instead of a shore pad.

Every mission is flown offline against the PX4-like model (pilot.simulate_offline) and checked against every
obstacle box (buildings, cranes, ships, masts, islets: 8 m horizontal / 6 m vertical clearance) and against the
terrain (>= 10 m above land or sea outside take-off / landing). A failing mission regenerates the whole coast
(attempt + 1), so coast and mission are a pure function of idx.

The instruction is built from CLAUSES, each tied to the mission beats it describes; finalize turns them into
`instruction_alignment` (text -> event -> frames) after the flight (review item #2).
"""
import math

import numpy as np

DIFFICULTY_RULE = "1 task = Easy, 2 tasks = Medium, 3+ tasks = Hard (take-off, transit and landing are not tasks)"
N_SIMS = 100
PREFIX = "easy_coastal_"
TRANSIT_AGL = 50.0
H_CLEAR, V_CLEAR = 8.0, 6.0
TERRAIN_CLEAR = 10.0
TASK_PLAN = ["circle"] * 35 + ["inspect"] * 20 + ["pass_over"] * 15 + ["follow_shore"] * 10 + ["land_on_ship"] * 20


def mission_ids():
    return [f"{PREFIX}{k:03d}" for k in range(1, N_SIMS + 1)]


def index_of(mid):
    return int(mid[len(PREFIX):])


def _task_type(idx):
    order = np.random.default_rng(626262).permutation(len(TASK_PLAN))
    return TASK_PLAN[int(order[idx - 1])]


def tasks(beats):
    out, prev_shore = [], False
    for b in beats:
        t = b["type"]
        if t == "orbit":
            out.append({"task": "circle", "landmark": b["landmark"], "laps": b.get("laps", 1.0),
                        "direction": "counter-clockwise" if b.get("dir", 1) > 0 else "clockwise"})
        elif t == "hover":
            out.append({"task": "inspect", "landmark": b["landmark"], "seconds": b["s"]})
        elif t == "goto" and b.get("shore"):
            if not prev_shore:
                out.append({"task": "follow_shore"})
        elif t == "goto" and "over" in b.get("label", ""):
            out.append({"task": "pass_over", "landmark": b["landmark"]})
        elif t == "land" and b.get("ship"):
            out.append({"task": "land_on_ship", "landmark": b["ship"]})
        prev_shore = t == "goto" and bool(b.get("shore"))
    return out


def difficulty_from_tasks(n):
    return "Easy" if n <= 1 else "Medium" if n == 2 else "Hard"


def instruction(m, L, ground):
    return m["instruction"], [c["text"] for c in m["clauses"]]


# ---- clearance ------------------------------------------------------------------------------------------------
def _hdist(o, x, y):
    return math.hypot(max(o[0] - x, 0, x - o[1]), max(o[2] - y, 0, y - o[3]))


def max_top_near(obs, x, y, r, skip=()):
    return max([o[4] for o in obs if _hdist(o, x, y) < r and o[5] not in skip] + [0.0])


def clearance_violation(frames, obs, beats, cv, pl):
    """First frame that enters an obstacle's safety box or flies too low over land / sea. The ship the drone takes off
    from or lands on is ignored during the take-off / descent / landing beats (its own deck is the target); every
    other obstacle, and that ship's superstructure above deck + 2 m, still count."""
    lab = {b.get("label") or b["type"]: b for b in beats}
    for f in frames[::3]:
        b = lab.get(f["beat"], {})
        own = b.get("own_ship")
        near_pad = b.get("type") in ("takeoff", "descend", "land")
        for o in obs:
            if own and o[5] == own and (near_pad or b.get("type") == "takeoff") and o[4] <= b.get("deck_z", 0) + 2.5:
                continue
            if f["z"] < o[4] + V_CLEAR and _hdist(o, f["x"], f["y"]) < H_CLEAR:
                if near_pad and own is None and o[4] < 3.0:   # low clutter beside a shore pad (bollards, beach huts)
                    continue
                return (round(f["t"], 1), f["beat"], o)
        if not near_pad:
            g = max(cv.terrain_height(f["x"], f["y"], pl), 0.0)
            if f["z"] - g < TERRAIN_CLEAR and f["t"] > 3.0:
                return (round(f["t"], 1), f["beat"], ("terrain", round(g, 1)))
    return None


# ---- mission ----------------------------------------------------------------------------------------------------
def _to_pad(pad, key, words, on_ship=False, agl=TRANSIT_AGL, spd=1.0):
    xy = pad["xy"]
    beats = [dict(type="goto", xy=xy, agl=agl, speed=round(min(1.0, 0.85 * spd), 3), radius=4.0, label=f"fly to {key}"),
             dict(type="descend", agl=8.0, label="descend over the pad", gimbal=5.0, own_ship=pad.get("ship"), deck_z=pad["z"]),
             dict(type="land", xy=xy, label=f"land {key}", pad=key, ship=pad.get("ship") if on_ship else None,
                  own_ship=pad.get("ship"), deck_z=pad["z"])]
    return beats


def _near(pads, xy, lo=60, hi=420):
    return sorted([k for k, p in pads.items() if lo < math.hypot(p["xy"][0] - xy[0], p["xy"][1] - xy[1]) < hi],
                  key=lambda k: math.hypot(pads[k]["xy"][0] - xy[0], pads[k]["xy"][1] - xy[1]))


def make_mission(idx, pl, cv, pilot, attempt):
    rng = np.random.default_rng(4_000_037 * idx + 11 + 99_991 * attempt)
    task = _task_type(idx)
    shore = dict(pl["pads"])
    heli = cv.heli_pads(pl)
    spots = dict(pl.get("spots", {}))  # unmarked places (not for the frozen, already published simulations)
    allpads = {**shore, **heli, **spots}
    places = {**shore, **spots}
    # cruise height and speed vary between the new simulations (frozen ones: 50 m, nominal speed); own rng
    vr = np.random.default_rng(733_001 * idx + 19 * attempt)
    agl, spd = (TRANSIT_AGL, 1.0) if not spots else (float(vr.choice([35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 70.0])), float(vr.uniform(0.8, 1.15)))
    obs = cv.obstacles(pl)
    ground = lambda x, y: cv.surface_z(x, y, pl, water=True)  # noqa: E731
    LM = pl["landmarks"]
    L = {}
    clauses = []

    # start: a shore pad, or (15 %, never for land_on_ship) a ship's helicopter deck
    target_site = [None]

    def _pool(target_xy, lo, hi, exclude=()):
        """Places to start / land: a painted landing pad (about a third of the time) or an unmarked spot."""
        use_spot = bool(spots) and rng.random() < 0.68
        src = spots if use_spot else shore
        cands = [k for k in (_near(src, target_xy, lo, hi) or []) if k not in exclude and src[k].get("site") != target_site[0]]
        if not cands:
            cands = [k for k in (_near(places, target_xy, lo, hi) or sorted(places)) if k not in exclude] or [k for k in sorted(shore) if k not in exclude]
        return str(rng.choice(cands[:4]))

    def pick_start(target_xy):
        if task != "land_on_ship" and heli and rng.random() < 0.15:
            hk = str(rng.choice(sorted(heli)))
            if 60 < math.hypot(heli[hk]["xy"][0] - target_xy[0], heli[hk]["xy"][1] - target_xy[1]) < 500:
                return hk
        return _pool(target_xy, 60, 420)

    def pick_land(target_xy, start):
        sp_ = allpads[start]
        if spots and not sp_.get("ship") and rng.random() < 0.15 and 40 < math.hypot(sp_["xy"][0] - target_xy[0], sp_["xy"][1] - target_xy[1]) < 450:
            return start  # come back to where it took off
        return _pool(target_xy, 40, 450, exclude=(start,))

    def start_clause(s):
        p = allpads[s]
        if p.get("ship"):
            return f"Take off from the helicopter deck of {p['ship_words']}"
        return f"Take off from {p['words']}"

    def land_clause(land, start):
        if land == start:
            return "then fly back and land where you took off"
        p = allpads[land]
        return f"then land on {p['words']}" if p.get("spot") else f"then land at {p['words']}"

    if task == "land_on_ship":
        hk = str(rng.choice(sorted(heli)))
        hp = heli[hk]
        target = hp["ship"]
        L[target] = LM[target]
        start = _pool(hp["xy"], 120, 600)
        beats = [dict(type="takeoff", agl=agl, label="take off", own_ship=allpads[start].get("ship"))]
        clauses.append(dict(text=start_clause(start), beats=[0]))
        beats += _to_pad(hp, hk, hp["words"], on_ship=True, agl=agl, spd=spd)
        beats[1]["landmark"] = target
        beats[1]["look_at"] = (hp["xy"][0], hp["xy"][1], hp["z"])
        clauses.append(dict(text=f"fly out to {hp['ship_words']}", beats=[1]))
        clauses.append(dict(text="and land on its helicopter deck", beats=[2, 3]))
        text = f"{clauses[0]['text']}, fly out to {hp['ship_words']} and land on its helicopter deck."
        land = hk
        focus = hp["ship_words"]
    elif task == "follow_shore":
        c = pl["coast"]
        for _ in range(60):
            xa = float(rng.uniform(-470, 300))
            span = float(rng.uniform(170, 330)) * (1 if rng.random() < 0.5 else -1)
            xb = max(-480.0, min(480.0, xa + span))
            if abs(xb - xa) > 150:
                break
        off = float(rng.uniform(28, 40))
        alt = float(rng.choice([22.0, 25.0, 28.0]))
        n = max(4, int(abs(xb - xa) // 30))
        pts = []
        for k in range(n + 1):
            x = xa + (xb - xa) * k / n
            yaw_s = cv.shore_yaw(x, c)
            nx, ny = -math.sin(yaw_s), math.cos(yaw_s)
            y0 = cv.shore_y(x, c)
            pts.append((x - nx * (off + c["W"] * 0.2), y0 - ny * (off + c["W"] * 0.2)))
        a, b = pts[0], pts[-1]
        start = pick_start(a)
        land = pick_land(b, start)
        beats = [dict(type="takeoff", agl=agl, label="take off", own_ship=allpads[start].get("ship"))]
        beats.append(dict(type="goto", xy=a, agl=agl, speed=round(min(1.0, 0.8 * spd), 3), radius=4.0, label="fly to the shoreline"))
        beats += [dict(type="goto", xy=p, z=alt, speed=0.65, radius=5.0, label="follow the shoreline", shore=True,
                       gimbal=float(rng.choice([15.0, 20.0, 25.0]))) for p in pts]
        n_follow = len(pts)
        near_a = min(pl["sites"], key=lambda s: math.hypot(s["xy"][0] - a[0], s["xy"][1] - a[1]))
        near_b = min(pl["sites"], key=lambda s: math.hypot(s["xy"][0] - b[0], s["xy"][1] - b[1]))
        way = "east" if xb > xa else "west"
        side = "left" if way == "east" else "right"  # land is at +y: heading east the shore is on the left
        beats += _to_pad(allpads[land], land, allpads[land]["words"], agl=agl, spd=spd)
        i_land = len(beats) - 3
        clauses = [dict(text=start_clause(start), beats=[0]),
                   dict(text=f"fly to the water's edge near {near_a['words']}", beats=[1]),
                   dict(text=f"follow the coastline {way} for about {abs(xb - xa):.0f} m at about {alt:.0f} m above the water, keeping the shore on your {side}",
                        beats=list(range(2, 2 + n_follow))),
                   dict(text=f"until you are near {near_b['words']}" if near_b is not near_a else "", beats=[1 + n_follow]),
                   dict(text=land_clause(land, start), beats=[i_land, i_land + 1, i_land + 2])]
        clauses = [cl for cl in clauses if cl["text"]]
        text = (f"{clauses[0]['text']}, {clauses[1]['text']} and {clauses[2]['text']}"
                + (f" {clauses[3]['text']}" if len(clauses) == 5 else "") + f", {clauses[-1]['text']}.")
        focus = None
    else:
        names = sorted(LM)
        if task == "pass_over":
            names = [n_ for n_ in names if LM[n_]["kind"] not in ("lighthouse point",)]
        name = str(rng.choice(names))
        t = LM[name]
        L[name] = t
        target_site[0] = name
        tx, ty = t["xy"]
        start = pick_start((tx, ty))
        land = pick_land((tx, ty), start)
        beats = [dict(type="takeoff", agl=agl, label="take off", own_ship=allpads[start].get("ship"))]
        look = (tx, ty, max(1.0, t["top_z"] * 0.4))
        size = min(t.get("size_r", 20.0), 45.0 if t["kind"] in cv.SITE_KINDS else 80.0)
        if task == "circle":
            radius = float(round(max(28.0, min(95.0, size + float(rng.uniform(16, 26))))))
            ring = [(tx + radius * math.cos(a), ty + radius * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 36)]
            g_ring = max(max(cv.terrain_height(x, y, pl), 0.0) for x, y in ring)
            z = max(max(max_top_near(obs, x, y, H_CLEAR + 2) for x, y in ring) + V_CLEAR + 2, g_ring + 22.0, 30.0)
            d = int(rng.choice([1, -1]))
            a0 = math.atan2(allpads[start]["xy"][1] - ty, allpads[start]["xy"][0] - tx)
            beats.append(dict(type="goto", xy=(tx + radius * math.cos(a0), ty + radius * math.sin(a0)), agl=agl, speed=round(min(1.0, 0.8 * spd), 3),
                              radius=3.0, label=f"fly to {name}", landmark=name, look_at=look))
            beats.append(dict(type="orbit", center=look, radius=radius, z=z, laps=1.0, dir=d, speed=4.0, label=f"orbit {name}", landmark=name))
            way = "counter-clockwise" if d > 0 else "clockwise"
            clauses = [dict(text=start_clause(start), beats=[0]), dict(text=f"fly to {t['words']}", beats=[1]),
                       dict(text=f"circle it once {way} at about {radius:.0f} m, keeping the camera on it", beats=[2])]
        elif task == "inspect":
            stand = float(round(max(30.0, min(80.0, size + float(rng.uniform(12, 22))))))
            a0 = math.atan2(allpads[start]["xy"][1] - ty, allpads[start]["xy"][0] - tx) + float(rng.uniform(-0.5, 0.5))
            hx, hy = tx + stand * math.cos(a0), ty + stand * math.sin(a0)
            g = max(cv.terrain_height(hx, hy, pl), 0.0)
            z = max(max_top_near(obs, hx, hy, H_CLEAR + 4) + V_CLEAR + 2, g + 22.0, min(t["top_z"] + 8.0, 60.0), 25.0)
            secs = float(rng.choice([5.0, 6.0, 8.0]))
            beats.append(dict(type="goto", xy=(hx, hy), z=z, speed=0.75, radius=3.0, label=f"fly to {name}", landmark=name, look_at=look))
            beats.append(dict(type="hover", s=secs, look_at=look, label=f"inspect {name}", landmark=name, align_first=True))
            clauses = [dict(text=start_clause(start), beats=[0]), dict(text=f"fly to {t['words']}", beats=[1]),
                       dict(text=f"stop about {stand:.0f} m from it and hover for about {secs:.0f} seconds looking at it", beats=[2])]
        else:
            z = max(max_top_near(obs, tx, ty, 26.0) + 10.0, max(cv.terrain_height(tx, ty, pl), 0.0) + 30.0, 40.0)
            gim = float(rng.choice([60.0, 65.0, 70.0]))
            beats.append(dict(type="goto", xy=(tx, ty), z=z, speed=0.6, radius=4.0, label=f"fly over {name}", landmark=name, gimbal=gim))
            clauses = [dict(text=start_clause(start), beats=[0]),
                       dict(text=f"fly directly over {t['words']}, with the camera looking down at it", beats=[1])]
        beats += _to_pad(allpads[land], land, allpads[land]["words"], agl=agl, spd=spd)
        i_land = len(beats) - 3
        clauses.append(dict(text=land_clause(land, start), beats=[i_land, i_land + 1, i_land + 2]))
        if task == "pass_over":
            text = f"{clauses[0]['text']}, {clauses[1]['text']}, {clauses[2]['text']}."
        else:
            text = f"{clauses[0]['text']}, {clauses[1]['text']} and {clauses[2]['text']}, {clauses[3]['text']}."
        focus = t["words"]
    if allpads[start].get("ship"):  # taking off from a deck: the take-off beat climbs from the deck
        beats[0]["deck_z"] = allpads[start]["z"]
    sp = allpads[start]
    m = dict(id=f"{PREFIX}{idx:03d}", difficulty="Easy", start=start, start_xy=tuple(sp["xy"]), start_z=sp["z"],
             start_on_ship=sp.get("ship"), yaw_deg=float(round(rng.uniform(-180, 180))), beats=beats, instruction=text,
             clauses=clauses, task_type=task, target_description=focus, layout_index=idx, attempt=attempt, land_pad=land)
    if spots:
        m["cruise"] = {"transit_agl_m": agl, "speed_scale": round(spd, 3)}
    if sp.get("ship"):  # on a ship the drone starts pointing along the ship (its deck is narrow)
        shp = next(s for s in pl["ships"] if s["name"] == sp["ship"])
        m["yaw_deg"] = float(round(math.degrees(shp["yaw"]) + (180 if rng.random() < 0.5 else 0))) % 360 - 180
    m["tasks"] = tasks(beats)
    if difficulty_from_tasks(len(m["tasks"])) != "Easy":
        raise ValueError(f"{m['id']}: {len(m['tasks'])} tasks")
    frames, p = pilot.simulate_offline(beats, ground, m["start_xy"], math.radians(m["yaw_deg"]))
    bad = clearance_violation(frames, obs, beats, cv, pl)
    ok = p.done and not p.failed and bad is None
    m["offline_check"] = {"done": p.done, "failed": p.failed, "clearance_violation": bad and [bad[0], bad[1], str(bad[2][-1])],
                          "duration_s": round(frames[-1]["t"], 1)}
    return m, L, ok


def mission_and_plan(idx, cv, pilot, max_attempts=40):
    for attempt in range(max_attempts):
        pl = cv.plan(idx, attempt)
        m, L, ok = make_mission(idx, pl, cv, pilot, attempt)
        if ok:
            pl["pads_all"] = {**pl["pads"], **cv.heli_pads(pl), **pl.get("spots", {})}
            return pl, m, L
    raise RuntimeError(f"no clear mission for {idx}")
