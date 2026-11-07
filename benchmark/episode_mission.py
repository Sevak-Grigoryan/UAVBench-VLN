"""
Benchmark episode = ONE task, nothing else, with a success rule that can score ANY flight.

An episode is built in the scene of a published Easy Mountain "inspect" simulation K (same landscape, objects, sky and
target as All_Simulations/Easy/Mountain/Simulation_K -- that data is only read, never changed):

  pre-roll   (flown, NOT published)  take off from the original take-off place, climb, turn to the start heading, settle
  EPISODE    (published)             start: hovering above the take-off place, target usually NOT in view
                                     instruction: "fly to <target> and stop about D m from it, facing it"
                                     expert: transit -> stand-off point -> face the target -> STOP
  post-roll  (flown, NOT published)  fly to the landing place and land (PX4 needs a clean landing; not part of the task)

The flight itself runs through the dataset pipeline (dataset_codebase/common/pipeline, unchanged): install() patches
the Mountain `missions` module in memory so that the mission id  vlnbench_mountain_<K>  resolves to the mission below.
"""
import math
import sys
from pathlib import Path

import numpy as np

CB = Path("/mnt/weka/sgrigoryan/dataset_codebase")
PREFIX = "vlnbench_mountain_"   # vlnbench_mountain_<K>: K = the published Easy Mountain Simulation_K whose scene is used
INDEX_BASE = 50_000             # private index range (dataset missions use 1..100, the dataset's own bench ids 1000+)

START_AGL_M = 30.0               # the episode starts hovering this high above the take-off place
SETTLE_S = 3.0                   # pre-roll: hold the start heading this long before the episode starts
STOP_HOLD_S = 2.0                # expert: hold still facing the target this long, then STOP
START_TURN_DEG = (35.0, 100.0)   # start heading = bearing to the target +- U(35, 100) deg: the target starts outside the
                                 # 47 deg wide image, the agent has to find it (the "landmark blindness" case of CityNav)

PRE, EPISODE, POST = "pre-roll", "episode", "post-roll"


def setup_paths():
    for p in (CB / "common/vln_simulation", CB / "Mountain/scene", CB / "Mountain", CB / "common/pipeline"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def is_bench(mid):
    return str(mid).startswith(PREFIX)


def source_sim(mid):
    return int(str(mid)[len(PREFIX):])


def build(k, mv, mm, pilot):
    """-> (pl, mission, landmarks). pl / target / stand-off point come from published simulation k's own plan."""
    pl, m0, _ = mm.mission_and_plan(k, mv, pilot)
    if m0["task_type"] != "inspect":
        raise ValueError(f"Simulation_{k} is a {m0['task_type']} mission; this episode builder needs an inspect mission")
    places = pl["pads_all"]
    b0 = m0["beats"]
    ih = next(i for i, b in enumerate(b0) if b.get("task") == "inspect")
    hover, stand = b0[ih], b0[ih - 1]
    name = hover["landmark"]
    t = pl["landmarks"][name]
    look = tuple(hover["look_at"])
    sx, sy = m0["start_xy"]
    D = math.hypot(stand["xy"][0] - t["xy"][0], stand["xy"][1] - t["xy"][1])   # stand-off distance (a multiple of 5 m)

    rng = np.random.default_rng(77_001 * k + 5)
    bearing = math.atan2(t["xy"][1] - sy, t["xy"][0] - sx)
    turn = math.radians(float(rng.uniform(*START_TURN_DEG))) * (1 if rng.random() < 0.5 else -1)
    head = math.atan2(math.sin(bearing + turn), math.cos(bearing + turn))
    g0 = mv.surface_z(sx, sy, pl, water=True)
    # look-at point 400 m ahead, 10 deg below the horizon: the settle hover turns to `head`, camera at a transit tilt
    ahead = (sx + 400.0 * math.cos(head), sy + 400.0 * math.sin(head), g0 + START_AGL_M - 400.0 * math.tan(math.radians(10.0)))

    beats = [dict(b0[0], agl=START_AGL_M, phase=PRE),
             dict(type="hover", s=SETTLE_S, look_at=ahead, align_first=True, label="settle at the start heading", phase=PRE)]
    i_ep = len(beats)
    for b in b0[1:ih]:                                    # the original transit + approach to the stand-off point
        beats.append(dict(b, phase=EPISODE))
    beats.append(dict(hover, s=STOP_HOLD_S, label=f"face {name} and stop", phase=EPISODE))
    i_stop = len(beats) - 1
    land = m0["land_pad"]
    for b in mm._to_pad(land, places[land], (m0.get("cruise") or {}).get("transit_agl_m", 45.0),
                        (m0.get("cruise") or {}).get("speed_scale", 1.0)):
        beats.append(dict(b, phase=POST))
    for b in beats:
        if b["type"] in ("goto", "descend"):
            b["budget_extra"] = 150.0

    words = t["words"]
    text = f"Fly to {words} and stop about {int(round(D))} m from it, facing it."
    mid = f"{PREFIX}{k:03d}"
    m = dict(id=mid, difficulty="Easy", start=m0["start"], start_xy=tuple(m0["start_xy"]), start_z=m0["start_z"],
             yaw_deg=m0["yaw_deg"], beats=beats, instruction=text,
             clauses=[dict(text=text, beats=list(range(i_ep, i_stop + 1)))],
             task_type="inspect", target_description=words, layout_index=k, attempt=m0["attempt"], land_pad=land,
             cruise=m0.get("cruise"), source_simulation=m0["id"])
    m["tasks"] = mm.tasks(beats)
    m["bench"] = {
        "episode_beats": [i_ep, i_stop],
        "start_heading_deg": round(math.degrees(head), 2),
        "start_turn_from_target_deg": round(math.degrees(turn), 1),
        "target": {"name": name, "description": words, "kind": t.get("kind"), "xy": [float(v) for v in t["xy"]],
                   "look_at_xyz": [float(v) for v in look], "top_z": float(t.get("top_z", look[2])),
                   "base_z": float(t.get("base_z", look[2]))},
        "standoff_m": round(D, 2),
        "expert_standoff_xyz": [float(stand["xy"][0]), float(stand["xy"][1]), float(stand["z"])],
    }
    L = {name: t}
    obs = mv.obstacles(pl)
    m, L, ok = mm._check(m, L, beats, obs, mv, pl, pilot, lambda x, y: mv.surface_z(x, y, pl, water=True))
    if not ok:
        raise RuntimeError(f"{mid}: offline check failed: {m['offline_check']}")
    return pl, m, L


def install():
    """Make the dataset pipeline resolve vlnbench_mountain_<K> (in memory only; dataset files are not touched)."""
    setup_paths()
    import missions as mm
    if getattr(mm, "_vlnbench_installed", False):
        return mm
    orig_index_of, orig_plan = mm.index_of, mm.mission_and_plan

    def index_of(mid):
        return INDEX_BASE + source_sim(mid) if is_bench(mid) else orig_index_of(mid)

    def mission_and_plan(idx, mv, pilot, max_attempts=40):
        if idx > INDEX_BASE:
            return build(idx - INDEX_BASE, mv, mm, pilot)
        return orig_plan(idx, mv, pilot, max_attempts)

    mm.index_of, mm.mission_and_plan = index_of, mission_and_plan
    mm._vlnbench_installed = True
    return mm
