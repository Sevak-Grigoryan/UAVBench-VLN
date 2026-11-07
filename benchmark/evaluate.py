"""
SCORE one flight of a benchmark episode (the expert's or any agent's), using only goal.json + the flight.

    python evaluate.py <episode dir> --telemetry flown_telemetry.json --actions flown_action.json [-o result.json]

The flight is in the reference format: telemetry = one record per frame (t, pose, agl_m, camera, px4), actions = one entry
per change; the episode ended by the agent's STOP if the last action is {"stop": true}, otherwise by the time limit.
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geometry as geo  # noqa: E402


def goal_state(rec, goal, K, size):
    """(in distance band, target in frame, horizontal distance) for one frame."""
    s, tg = goal["success"], goal["target"]
    dh = math.hypot(rec["pose"]["x"] - tg["xy"][0], rec["pose"]["y"] - tg["xy"][1])
    band = abs(dh - s["standoff_m"]) <= s["standoff_tolerance_m"]
    view = geo.in_frame(geo.project(rec["camera"], tg["point_xyz"], K), size, s["frame_margin"])
    return band, view, dh


def evaluate(task, goal, tele, actions):
    s, tg = goal["success"], goal["target"]
    K, size = task["observation"]["rgb"]["camera_intrinsics"], task["observation"]["rgb"]["size"]
    t_limit = s["time_limit_s"]
    stopped = bool(actions) and bool(actions[-1].get("stop"))
    stop_t = actions[-1]["t"] if stopped else None
    tele = [r for r in tele if r["t"] <= t_limit + 1e-6] or tele[:1]
    if stopped and stop_t > t_limit + 1e-6:
        stopped = False                                   # said STOP only after the time limit
    last = tele[-1]

    crash = next((r["t"] for r in tele if abs(r["pose"]["roll_deg"]) > s["max_tilt_deg"] or abs(r["pose"]["pitch_deg"]) > s["max_tilt_deg"]), None)
    low = next((r["t"] for r in tele if r["agl_m"] < s["min_agl_m"]), None)
    states = [goal_state(r, goal, K, size) for r in tele]
    band, view, dh = states[-1]
    safe = crash is None and low is None
    success = stopped and safe and band and view
    L, L_ref = geo.path_length(tele), goal["reference"]["path_m"]
    if not stopped:
        end = "time limit reached without STOP" if last["t"] >= t_limit - 1.0 else "flight ended without STOP"
    elif crash is not None:
        end = f"STOP, but tilted beyond {s['max_tilt_deg']:.0f} deg at t = {crash:.1f} s (crash)"
    elif low is not None:
        end = f"STOP, but below {s['min_agl_m']:.0f} m above the ground at t = {low:.1f} s"
    elif not band:
        end = f"STOP at {dh:.1f} m from the target (needs {s['standoff_m']:.0f} +- {s['standoff_tolerance_m']:.0f} m)"
    elif not view:
        end = "STOP at the right distance, but the target is not in the frame"
    else:
        end = "STOP in the goal state"
    return {
        "episode_id": goal["episode_id"],
        "success": bool(success),
        "oracle_success": any(b and v for b, v, _ in states) and safe,
        "navigation_error_m": round(abs(dh - s["standoff_m"]), 2),
        "spl": round(float(success) * L_ref / max(L, L_ref, 1e-6), 4),
        "episode_end": end,
        "stopped": stopped,
        "distance_to_target_at_end_m": round(dh, 2),
        "in_distance_band_at_end": band,
        "target_in_frame_at_end": view,
        "heading_error_at_end_deg": round(geo.bearing_error_deg(last["pose"], tg["point_xyz"]), 1),
        "target_in_frame_fraction": round(sum(v for _, v, _ in states) / len(states), 3),
        "closest_to_ring_m": round(min(abs(d - s["standoff_m"]) for _, _, d in states), 2),
        "crashed": crash is not None,
        "below_min_agl": low is not None,
        "path_length_m": round(L, 1),
        "reference_path_m": L_ref,
        "duration_s": round(last["t"], 2),
        "time_limit_s": t_limit,
        "end_position_xyz": [round(v, 2) for v in geo.xyz(last)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("episode", type=Path)
    ap.add_argument("--telemetry", required=True, type=Path)
    ap.add_argument("--actions", required=True, type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    task = json.loads((a.episode / "task.json").read_text())
    goal = json.loads((a.episode / "goal.json").read_text())
    res = evaluate(task, goal, json.loads(a.telemetry.read_text()), json.loads(a.actions.read_text())["actions"])
    txt = json.dumps(res, indent=1)
    if a.out:
        a.out.write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
