"""
Benchmark episode in the scene of a published Easy Forest "inspect" simulation (collision-aware benchmark, Benchmark_collision).

Same design as the Mountain v0.1 episode (Benchmark_codebase/episode_mission.py), built from Forest Easy v2 mission easy_v2_<idx>
(All_Simulations/Easy/Forest/Simulation_<idx + 4>: same valley, props layout, target; that data is only read):

  pre-roll   (flown, NOT published)  take off from the original take-off clearing, climb, turn to the start heading, settle
  EPISODE    (published, evaluated)  start: hovering above the take-off clearing, target usually NOT in view
                                     instruction: "fly to <target> and stop about D m from it, facing it"
                                     expert: transit -> stand-off point -> face the target -> STOP
  post-roll  (flown, NOT published)  fly to the landing clearing and land
"""
import math

import numpy as np

PREFIX = "vlncol_forest_"        # vlncol_forest_<idx>: idx = Forest Easy v2 mission index (scene easy_v2_<idx>)
START_AGL_M = 25.0
SETTLE_S = 3.0
STOP_HOLD_S = 2.0
START_TURN_DEG = (35.0, 100.0)
PRE, EPISODE, POST = "pre-roll", "episode", "post-roll"


def scene_of(mid):
    return f"easy_v2_{int(str(mid)[len(PREFIX):]):03d}"


def build(idx, env):
    """-> mission dict (beats with phases + bench block) from the scene's own Easy v2 inspect mission."""
    import missions as fm

    m0 = env.missions[f"easy_v2_{idx:03d}"]
    if m0["task_type"] != "inspect":
        raise ValueError(f"easy_v2_{idx:03d} is a {m0['task_type']} mission; this builder needs an inspect mission")
    ground = env.ground
    b0 = m0["beats"]
    ih = next(i for i, b in enumerate(b0) if b["type"] == "hover")
    hover, stand = b0[ih], b0[ih - 1]
    name = hover["landmark"]
    t = env.landmarks[name]
    tx, ty = t["xy"]
    look = tuple(hover["look_at"])
    sx, sy = m0["start_xy"]
    D = math.hypot(stand["xy"][0] - tx, stand["xy"][1] - ty)

    rng = np.random.default_rng(91_003 * idx + 7)
    bearing = math.atan2(ty - sy, tx - sx)
    turn = math.radians(float(rng.uniform(*START_TURN_DEG))) * (1 if rng.random() < 0.5 else -1)
    head = math.atan2(math.sin(bearing + turn), math.cos(bearing + turn))
    g0 = ground(sx, sy)
    ahead = (sx + 400.0 * math.cos(head), sy + 400.0 * math.sin(head), g0 + START_AGL_M - 400.0 * math.tan(math.radians(10.0)))
    beats = [dict(b0[0], agl=START_AGL_M, phase=PRE),
             dict(type="hover", s=SETTLE_S, look_at=ahead, align_first=True, label="settle at the start heading", phase=PRE)]
    i_ep = len(beats)
    for b in b0[1:ih]:
        beats.append(dict(b, phase=EPISODE))
    beats.append(dict(hover, s=STOP_HOLD_S, label=f"face {name} and stop", phase=EPISODE))
    i_stop = len(beats) - 1
    land = next(b["pad"] for b in b0 if b["type"] == "land")
    for b in b0[ih + 1:]:
        beats.append(dict(b, phase=POST))
    sep = ", and" if "," in t["desc"] else " and"
    text = f"Fly to {t['desc']}{sep} stop about {int(round(D))} m from it, facing it."
    mid = f"{PREFIX}{idx:03d}"
    m = dict(m0, id=mid, beats=beats, instruction=text, task_type="inspect", target_description=t["desc"],
             source_simulation=m0["id"], land_pad=land)
    m["tasks"] = fm.tasks(beats[i_ep:i_stop + 1])
    m["bench"] = {
        "episode_beats": [i_ep, i_stop],
        "start_heading_deg": round(math.degrees(head), 2),
        "start_turn_from_target_deg": round(math.degrees(turn), 1),
        "target": {"name": name, "description": t["desc"], "kind": name.split("_")[0], "xy": [float(tx), float(ty)],
                   "look_at_xyz": [float(v) for v in look], "top_z": float(t["top_z"]), "base_z": float(ground(tx, ty))},
        "standoff_m": round(D, 2),
        "expert_standoff_xyz": [float(stand["xy"][0]), float(stand["xy"][1]), None],
        "scene": {"environment": "Forest", "scene_mission": m0["id"], "source_simulation": f"All_Simulations/Easy/Forest/Simulation_{idx + 4}",
                  "start_pad": m0["start"], "landing_pad_post_roll": land},
    }
    m["landing"] = None     # the inspect episode ends in the air: no contact is allowed while it is evaluated
    return m
