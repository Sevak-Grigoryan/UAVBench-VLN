"""
City missions for the Holybro X500 V2 -- Easy City dataset (2026-10-12), 100 single-task missions.

Each simulation idx (1..100) has its own city (scene/city_v2.plan) and ONE task about one of that city's
special blocks or the glass tower: circle it, inspect it (hover facing it), fly over it with the camera down,
or follow a street. Start / landing pad (8 pads), heading, radius, height and direction vary.

Every mission is flown offline against a PX4-like model (pilot.simulate_offline) and checked against every
building, the tower, cranes and tall structures of its own city before it is accepted; a mission that comes
within 8 m horizontally of an obstacle below obstacle top + 6 m is regenerated (attempt + 1). So the city
and mission for idx are a pure function of idx, rebuilt identically by the flight job and every render slice.
"""
import math

import numpy as np

DIFFICULTY_RULE = "1 task = Easy, 2 tasks = Medium, 3+ tasks = Hard (take-off, transit and landing are not tasks)"
N_SIMS = 100
PREFIX = "easy_city_"
TRANSIT_AGL = 50.0  # above every ordinary building (<= 34 m + 7.5 m rooftop plant)
H_CLEAR, V_CLEAR = 8.0, 6.0

TASK_PLAN = ["circle"] * 45 + ["inspect"] * 25 + ["pass_over"] * 20 + ["follow_road"] * 10


def mission_ids():
    return [f"{PREFIX}{k:03d}" for k in range(1, N_SIMS + 1)]


def index_of(mid):
    return int(mid[len(PREFIX):])


def _task_type(idx):
    order = np.random.default_rng(515151).permutation(len(TASK_PLAN))
    return TASK_PLAN[int(order[idx - 1])]


def tasks(beats):
    out, prev_road = [], False
    for b in beats:
        t = b["type"]
        if t == "orbit":
            out.append({"task": "circle", "landmark": b["landmark"], "laps": b.get("laps", 1.0),
                        "direction": "counter-clockwise" if b.get("dir", 1) > 0 else "clockwise"})
        elif t == "hover":
            out.append({"task": "inspect", "landmark": b["landmark"], "seconds": b["s"]})
        elif t == "goto" and b.get("road"):
            if not prev_road:
                out.append({"task": "follow_road"})
        elif t == "goto" and "over" in b.get("label", ""):
            out.append({"task": "pass_over", "landmark": b["landmark"]})
        prev_road = t == "goto" and bool(b.get("road"))
    return out


def difficulty_from_tasks(n):
    return "Easy" if n <= 1 else "Medium" if n == 2 else "Hard"


def instruction(m, L, ground):
    return m["instruction"], [m["instruction"]]


# ---- obstacle clearance -----------------------------------------------------------------------------------
def _hdist(o, x, y):
    return math.hypot(max(o[0] - x, 0, x - o[1]), max(o[2] - y, 0, y - o[3]))


def max_top_near(obs, x, y, r):
    return max([o[4] for o in obs if _hdist(o, x, y) < r] + [0.0])


def clearance_violation(frames, obs):
    for f in frames[::3]:
        for o in obs:
            if f["z"] < o[4] + V_CLEAR and _hdist(o, f["x"], f["y"]) < H_CLEAR:
                return (round(f["t"], 1), f["beat"], o)
    return None


# ---- mission generation -------------------------------------------------------------------------------------
def _to_pad(xy, pad):
    # descend quickly to 8 m over the (clear) pad block, then the normal landing; 50 m at the landing beat's
    # gentle rate alone made every video ~15 s longer
    return [dict(type="goto", xy=xy, agl=TRANSIT_AGL, speed=0.85, radius=4.0, label=f"fly to {pad} pad"),
            dict(type="descend", agl=8.0, label="descend over the pad", gimbal=5.0),
            dict(type="land", xy=xy, label=f"land {pad}", pad=pad)]


def targets(cv, pl):
    """name -> dict(xy, top_z, desc, short, size_r) for every special block and the tower."""
    T = {}
    for (bx, by), f in pl["features"].items():
        T[f["name"]] = dict(xy=(bx, by), top_z=float(f["top_z"]), desc=f"{f['words']} {cv.compass_words(bx, by)}",
                            short=f"the {f['kind']}", size_r=20.0, kind=f["kind"])
    tx, ty = cv.TOWER
    T["tower"] = dict(xy=(tx, ty), top_z=cv.cs.TOWER_HEIGHT_M, desc="the tall blue glass office tower near the city centre",
                      short="the glass tower", size_r=10.0, kind="tower")
    return T


def make_mission(idx, pl, cv, pilot, attempt):
    rng = np.random.default_rng(3_000_017 * idx + 5 + 104_729 * attempt)
    task = _task_type(idx)
    P = {k: v[0] for k, v in cv.PADS.items()}
    W = {k: v[1] for k, v in cv.PADS.items()}
    obs = cv.obstacles(pl)
    ground = lambda x, y: cv.surface_z(x, y, list(pl["features"]))
    beats = [dict(type="takeoff", agl=TRANSIT_AGL, label="take off")]
    T = targets(cv, pl)
    L = {}
    if task == "follow_road":
        roads = [r for r in cv.ROADS if abs(r) < 200]
        axis = "NS" if rng.random() < 0.5 else "EW"
        c = float(rng.choice(roads))
        nblk = int(rng.integers(2, 5))
        s_a = float(rng.choice([r for r in cv.ROADS if abs(r) <= 186]))
        sgn = float(rng.choice([-1, 1]))
        s_b = s_a + sgn * nblk * cv.cs.BLOCK_PITCH_M
        if abs(s_b) > 190:
            s_b = s_a - sgn * nblk * cv.cs.BLOCK_PITCH_M
            sgn = -sgn
        pt = (lambda s_: (c, s_)) if axis == "NS" else (lambda s_: (s_, c))
        a, b = pt(s_a), pt(s_b)
        start = min(P, key=lambda k: math.hypot(P[k][0] - a[0], P[k][1] - a[1]) + 1000 * (rng.random() < 0.0))
        cand = sorted(P, key=lambda k: math.hypot(P[k][0] - a[0], P[k][1] - a[1]))
        start = str(rng.choice(cand[:3]))
        land = str(rng.choice([k for k in sorted(P, key=lambda k: math.hypot(P[k][0] - b[0], P[k][1] - b[1]))[:3] if k != start]))
        agl = float(rng.choice([26.0, 28.0, 30.0]))
        beats.append(dict(type="goto", xy=a, agl=TRANSIT_AGL, speed=0.8, radius=4.0, label="fly to the street"))
        n = int(abs(s_b - s_a) // 31)
        beats += [dict(type="goto", xy=pt(s_a + (s_b - s_a) * k / n), agl=agl, speed=0.7, radius=5.0, label="follow street", road=True)
                  for k in range(n + 1)]
        beats += _to_pad(P[land], land)
        if axis == "NS":
            way = "north" if s_b > s_a else "south"
        else:
            way = "east" if s_b > s_a else "west"
        text = (f"Take off from {W[start]}, fly to the street {cv.compass_words(*a)} and follow it {way} for {nblk} blocks "
                f"(about {abs(s_b - s_a):.0f} m) at about {agl:.0f} m above the road, between the buildings, then land at {W[land]}.")
        focus = None
    else:
        names = sorted(T)
        name = str(rng.choice(names))
        t = T[name]
        L[name] = t
        tx, ty = t["xy"]
        by_d = sorted(P, key=lambda k: math.hypot(P[k][0] - tx, P[k][1] - ty))
        near = [k for k in by_d if 50 < math.hypot(P[k][0] - tx, P[k][1] - ty) < 210] or by_d[:4]
        start = str(rng.choice(near[:4]))
        land = str(rng.choice([k for k in near[:5] if k != start] or [k for k in by_d if k != start][:2]))
        look = (tx, ty, t["top_z"] * 0.4)
        if task == "circle":
            radius = float(round(rng.uniform(26, 33) if t["kind"] != "tower" else rng.uniform(30, 36)))
            z = max(max_top_near(obs, tx, ty, radius + H_CLEAR + 2) + V_CLEAR + 2, 32.0)
            if t["kind"] == "tower":
                # the tower itself is inside the circle; only what stands on the circle matters
                z = max([o[4] for o in obs if o[4] < cv.TOWER_TOP - 1 and any(
                    _hdist(o, tx + radius * math.cos(a), ty + radius * math.sin(a)) < H_CLEAR + 2 for a in np.linspace(0, 2 * math.pi, 24))] + [0]) + V_CLEAR + 2
                z = max(z, 42.0)
            d = int(rng.choice([1, -1]))
            a0 = math.atan2(P[start][1] - ty, P[start][0] - tx)
            beats.append(dict(type="goto", xy=(tx + radius * math.cos(a0), ty + radius * math.sin(a0)), agl=TRANSIT_AGL, speed=0.8,
                              radius=3.0, label=f"fly to {name}", landmark=name, look_at=look))
            beats.append(dict(type="orbit", center=look, radius=radius, z=z, laps=1.0, dir=d, speed=4.0, label=f"orbit {name}", landmark=name))
            text = (f"Take off from {W[start]}, fly to {t['desc']} and circle it once {'counter-clockwise' if d > 0 else 'clockwise'} "
                    f"at about {radius:.0f} m, keeping the camera on it, then land at {W[land]}.")
        elif task == "inspect":
            stand = float(round(rng.uniform(30, 38)))
            a0 = math.atan2(P[start][1] - ty, P[start][0] - tx) + float(rng.uniform(-0.5, 0.5))
            hx, hy = tx + stand * math.cos(a0), ty + stand * math.sin(a0)
            z = max(max_top_near(obs, hx, hy, H_CLEAR + 4) + V_CLEAR + 2, 30.0 if t["kind"] != "tower" else 40.0)
            secs = float(rng.choice([5.0, 6.0, 8.0]))
            beats.append(dict(type="goto", xy=(hx, hy), z=z, speed=0.75, radius=3.0, label=f"fly to {name}", landmark=name, look_at=look))
            beats.append(dict(type="hover", s=secs, look_at=look, label=f"inspect {name}", landmark=name))
            text = (f"Take off from {W[start]}, fly to {t['desc']}, stop about {stand:.0f} m from it and hover for about "
                    f"{secs:.0f} seconds looking at it, then land at {W[land]}.")
        else:
            if t["kind"] == "tower":
                name = str(rng.choice([n for n in names if n != "tower"]))
                t = T[name]
                L = {name: t}
                tx, ty = t["xy"]
            z = max(max_top_near(obs, tx, ty, 24.0) + 10.0, 46.0)
            gim = float(rng.choice([60.0, 65.0, 70.0]))
            beats.append(dict(type="goto", xy=(tx, ty), z=z, speed=0.6, radius=4.0, label=f"fly over {name}", landmark=name, gimbal=gim))
            text = f"Take off from {W[start]}, fly directly over {t['desc']}, with the camera looking down at it, then land at {W[land]}."
        beats += _to_pad(P[land], land)
        focus = t["desc"]
    m = dict(id=f"{PREFIX}{idx:03d}", difficulty="Easy", start=start, start_xy=P[start], yaw_deg=float(round(rng.uniform(-180, 180))),
             beats=beats, instruction=text, task_type=task, target_description=focus, layout_index=idx, attempt=attempt)
    m["tasks"] = tasks(beats)
    if difficulty_from_tasks(len(m["tasks"])) != "Easy":
        raise ValueError(f"{m['id']}: {len(m['tasks'])} tasks")
    frames, p = pilot.simulate_offline(beats, ground, m["start_xy"], math.radians(m["yaw_deg"]))
    bad = clearance_violation(frames, obs)
    ok = p.done and not p.failed and bad is None
    m["offline_check"] = {"done": p.done, "failed": p.failed, "clearance_violation": bad and [bad[0], bad[1]],
                          "duration_s": round(frames[-1]["t"], 1)}
    return m, L, ok


def mission_and_plan(idx, cv, pilot, max_attempts=40):
    """The accepted (plan, mission, landmarks) for idx: first attempt whose flight is clear of every obstacle."""
    for attempt in range(max_attempts):
        pl = cv.plan(idx, attempt)
        m, L, ok = make_mission(idx, pl, cv, pilot, attempt)
        if ok:
            return pl, m, L
    raise RuntimeError(f"no clear mission for {idx}")
