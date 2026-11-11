"""
SELF-TEST of the success checker on one published episode: the expert flight must succeed, and edited copies of it (made in
memory only -- nothing on disk is changed) must give the expected verdict. These are checker tests, NOT agent flights.

    python selftest.py <episode dir> [-o results/checker_selftest.json]
"""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import evaluate  # noqa: E402


def shift_end(tele, goal, metres):
    """Move the last frame `metres` horizontally toward (+) or away from (-) the target."""
    t = copy.deepcopy(tele)
    r = t[-1]
    tx, ty = goal["target"]["xy"]
    dx, dy = tx - r["pose"]["x"], ty - r["pose"]["y"]
    d = math.hypot(dx, dy)
    for key in ("pose", "camera"):
        r[key]["x"] += metres * dx / d
        r[key]["y"] += metres * dy / d
    return t


def turn_end(tele, deg):
    t = copy.deepcopy(tele)
    for key in ("pose", "camera"):
        t[-1][key]["yaw_deg"] += deg
    return t


def stop_at(tele, acts, frac):
    n = max(2, int(len(tele) * frac))
    a = [x for x in copy.deepcopy(acts) if not x.get("stop") and x["frame"] < n]
    return tele[:n], a + [{"index": len(a), "frame": n, "t": tele[n - 1]["t"], "stop": True}]


def edit_frame(tele, i, **kw):
    t = copy.deepcopy(tele)
    for path, v in kw.items():
        a, b = path.split("__")
        t[i][a][b] = v
    return t


def with_contact(col, tele, frac, verdict="prohibited"):
    """A copy of the collision record with one contact of `verdict` at fraction `frac` of the episode (in memory)."""
    c = copy.deepcopy(col)
    t = tele[-1]["t"] * frac
    rec = {"t": round(t, 6), "episode_step": int(round(t * c["physics"]["hz"])), "step": None, "phase": "episode",
           "verdict": verdict, "drone_part": "rotor", "drone_collider": "/World/x500/body/collision_proxies/rotor_disc_0",
           "other_collider": "/World/CollisionProxies/(edited copy)", "total_impulse_ns": 0.01, "points": []}
    if verdict == "prohibited":
        c["first_prohibited"] = rec
    c["contacts"] = c.get("contacts", []) + [rec]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("-o", "--out", type=Path)
    a = ap.parse_args()
    task = json.loads((a.episode / "task.json").read_text())
    goal = json.loads((a.episode / "goal.json").read_text())
    tele = json.loads((a.episode / "reference/telemetry.json").read_text())
    acts = json.loads((a.episode / "reference/action.json").read_text())["actions"]
    col = json.loads((a.episode / "reference/collision.json").read_text())
    tol = goal["success"]["standoff_tolerance_m"]
    mid = len(tele) // 2
    no_stop = [x for x in acts if not x.get("stop")]
    low = copy.deepcopy(tele)
    low[mid]["agl_m"] = 0.5
    unverified = dict(copy.deepcopy(col), monitor_verified=False)
    cases = [
        ("expert flight, unchanged", tele, acts, goal, True),
        ("rotor touches a tree half-way (prohibited contact)", tele, acts, goal, False, with_contact(col, tele, 0.5)),
        ("brief prohibited contact 0.2 s before STOP", tele, acts, goal, False,
         with_contact(col, tele, 1.0 - 0.2 / tele[-1]["t"])),
        ("contact only inside the contact offset (proximity, no touch)", tele, acts, goal, True,
         with_contact(col, tele, 0.5, "proximity")),
        ("no collision-monitoring data", tele, acts, goal, False, None),
        ("collision monitor not verified", tele, acts, goal, False, unverified),
        (f"stops {tol * 0.6:.0f} m further out (inside the +-{tol:.0f} m band)", shift_end(tele, goal, -0.6 * tol), acts, goal, True),
        ("stops half-way to the target", *stop_at(tele, acts, 0.5), goal, False),
        (f"stops {tol + 10:.0f} m too close to the target", shift_end(tele, goal, tol + 10), acts, goal, False),
        ("stops at the right distance but turned 120 deg away", turn_end(tele, 120.0), acts, goal, False),
        ("never says STOP (flies until the time limit)", tele, no_stop, goal, False),
        ("too slow: time limit cut to 20 s", tele, acts,
         {**goal, "success": {**goal["success"], "time_limit_s": 20.0}}, False),
        ("crash: 75 deg tilt in the middle of the flight", edit_frame(tele, mid, pose__roll_deg=75.0), acts, goal, False),
        ("touches the ground in the middle of the flight", low, acts, goal, False),
    ]
    rows = []
    for name, t, ac, g, expect, *c in cases:
        r = evaluate(task, g, t, ac, c[0] if c else col)
        ok = r["success"] == expect
        rows.append({"case": name, "expected_success": expect, "success": r["success"], "oracle_success": r["oracle_success"],
                     "valid": r["valid"], "navigation_error_m": r["navigation_error_m"], "spl": r["spl"],
                     "episode_end": r["episode_end"], "pass": ok})
        ne = "  n/a" if r["navigation_error_m"] is None else f'{r["navigation_error_m"]:5.1f}'
        print(f'{"PASS" if ok else "FAIL"}  {name:62s} SR={r["success"]!s:5} OSR={r["oracle_success"]!s:5} '
              f'NE={ne} m SPL={r["spl"]:.2f}  ({r["episode_end"]})')
    if a.out:
        a.out.write_text(json.dumps({"note": "checker self-test: edited copies of the expert flight, in memory; not agent flights",
                                     "all_pass": all(x["pass"] for x in rows), "cases": rows}, indent=1) + "\n")
    sys.exit(0 if all(x["pass"] for x in rows) else 1)


if __name__ == "__main__":
    main()
