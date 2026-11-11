"""
Checks the REAL simulated collision tests (flown by job.sbatch through run_flight.py, PX4 SITL + Isaac Sim PhysX) and writes
tests/results.json + tests/RESULTS.md.

    python tests/check_tests.py --root /mnt/weka/sgrigoryan/Benchmark_collision_work [--episode <published episode dir>]

Per test: expected outcome / first prohibited part / other collider; the latched contact step == the termination step (the run
ended right after that physics step); the terminal telemetry record is at that step's time; where the contact fell relative to
the 30 fps frames and the 0.5 s decisions; PX4 disarmed in the shutdown; and -- failure propagation -- the evaluator, fed the
test flight as if it were an agent flight of the published Forest episode, reports the collision (or INVALID without data).
"""
import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import collision_record  # noqa: E402
import scenarios  # noqa: E402
from evaluate import evaluate  # noqa: E402

HZ, FPS, DECIDE = 240, 30, 0.5


def check(name, d, task, goal):
    f = d / "flight"
    if not (f / "FLIGHT_DONE").exists():
        return {"test": name, "ran": False, "pass": False, "why": "not flown"}
    col = json.loads((f / "collision.json").read_text())
    tele = json.loads((f / "telemetry.json").read_text())
    acts = json.loads((f / "action.json").read_text())["actions"]
    exp = scenarios.EXPECTED[name]
    fp, term = col.get("first_prohibited"), col.get("termination")
    checks = {"monitor_verified": col["monitor_verified"], "outcome": col["outcome"] == exp["outcome"]}
    if "first_prohibited" in exp:
        checks["no_prohibited_contact"] = fp is None
    if "first_prohibited_part" in exp:
        checks["first_part"] = bool(fp) and fp["drone_part"] == exp["first_prohibited_part"]
    if "first_prohibited_part_in" in exp:
        checks["first_part"] = bool(fp) and fp["drone_part"] in exp["first_prohibited_part_in"]
    if "first_prohibited_other_prefix" in exp:
        checks["other_collider"] = bool(fp) and fp["other_collider"].startswith(exp["first_prohibited_other_prefix"])
    if exp.get("allowed_landing"):
        checks["allowed_landing_seen"] = col.get("first_allowed_landing") is not None
    row = {"test": name, "ran": True, "outcome": col["outcome"], "expected": exp}
    if fp:
        term_rec = next((r for r in tele if r.get("terminal")), None)
        t_step = (fp["step"] - col["recording_start_step"]) / HZ
        checks["terminated_at_contact_step"] = bool(term) and term["step"] == fp["step"] == term["contact_step"]
        checks["terminal_record_at_step_time"] = bool(term_rec) and abs(term_rec["t"] - t_step) < 1e-6
        checks["no_frame_after_contact"] = all(r["t"] <= t_step + 1e-9 for r in tele)
        checks["px4_disarmed_after"] = bool((col.get("px4_shutdown") or {}).get("px4_disarmed"))
        last_frame = max((r for r in tele if not r.get("terminal")), key=lambda r: r["t"])
        row["contact"] = {"t": fp["t"], "physics_step": fp["step"], "drone_part": fp["drone_part"],
                          "drone_collider": fp["drone_collider"], "other_collider": fp["other_collider"],
                          "total_impulse_ns": fp["total_impulse_ns"], "points": len(fp["points"]),
                          "first_point": fp["points"][0] if fp["points"] else None,
                          "s_after_last_frame": round(t_step - last_frame["t"], 6),
                          "s_after_last_decision": round(t_step - DECIDE * math.floor(t_step / DECIDE + 1e-9), 6),
                          "between_frames": abs(t_step * FPS - round(t_step * FPS)) > 1e-6,
                          "between_decisions": abs(t_step / DECIDE - round(t_step / DECIDE)) > 1e-6,
                          "speed_at_last_frame_mps": round(math.hypot(last_frame["velocity"]["vx"], last_frame["velocity"]["vy"]), 2)}
        row["termination"] = {k: v for k, v in term.items() if k != "record"} if term else None
        row["px4_shutdown"] = col.get("px4_shutdown")
    if col.get("first_allowed_landing"):
        al = col["first_allowed_landing"]
        row["allowed_landing"] = {"t": al["t"], "step": al["step"], "drone_collider": al["drone_collider"],
                                  "other_collider": al["other_collider"], "points": [p["position"] for p in al["points"]]}
    row["pairs"] = [{k: p[k] for k in ("verdict", "phase", "drone_part", "drone_collider", "other_collider", "first_t", "steps")}
                    for p in col["pairs"] if p["verdict"] != "proximity"]
    row["test_notes"] = col.get("test_notes")
    # failure propagation through the evaluator (the test flight scored as an agent flight of the published episode)
    if task is not None:
        ep = [r for r in tele if r.get("phase") == "episode" and not r.get("terminal")]
        if ep:
            t0, step0 = ep[0]["t"], ep[0]["physics_step"]
            t1 = tele[-1]["t"]
            ecol = collision_record.episode_record(col, t0, t1, step0)
            et = [dict(r, t=round(r["t"] - t0, 6)) for r in tele if r["t"] >= t0 - 1e-9]
            ea = [dict(a, t=round(a["t"] - t0, 6)) for a in acts if a["t"] >= t0 - 1e-9]
            r = evaluate(task, goal, et, ea, ecol)
            r_none = evaluate(task, goal, et, ea, None)
            row["evaluator"] = {"success": r["success"], "valid": r["valid"], "collided": r.get("collided"),
                                "episode_end": r["episode_end"], "without_collision_file": r_none["episode_end"]}
            checks["evaluator_reports_collision"] = (r.get("collided") is True) == (fp is not None) and not r["success"]
            checks["evaluator_invalid_without_data"] = r_none["valid"] is False and not r_none["success"]
    row["checks"] = checks
    row["pass"] = all(checks.values())
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--episode", type=Path)
    a = ap.parse_args()
    task = goal = None
    if a.episode and (a.episode / "task.json").exists():
        task = json.loads((a.episode / "task.json").read_text())
        goal = json.loads((a.episode / "goal.json").read_text())
    rows = [check(n, a.root / f"colltest_{n}", task, goal) for n in scenarios.NAMES]
    out = {"note": "real simulated flights (PX4 v1.14.3 SITL + Isaac Sim 6.2 PhysX, 240 Hz), Forest scene; not synthetic telemetry",
           "all_pass": all(r["pass"] for r in rows), "tests": rows}
    (HERE / "tests/results.json").write_text(json.dumps(out, indent=1) + "\n")
    md = ["# Collision tests (real simulated flights)", "",
          "| Test | Pass | Outcome | First prohibited contact | t (s) | Physics step | Between decisions | Evaluator |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        c = r.get("contact") or {}
        ev = (r.get("evaluator") or {}).get("episode_end", "")
        md.append(f"| {r['test']} | {'PASS' if r['pass'] else 'FAIL'} | {r.get('outcome', 'not flown')} | "
                  f"{(c.get('drone_part', '') + ' ' + c.get('drone_collider', '').rsplit('/', 1)[-1] + ' -> ' + c.get('other_collider', '')) if c else '-'} | "
                  f"{c.get('t', '-')} | {c.get('physics_step', '-')} | {c.get('between_decisions', '-')} | {ev[:80]} |")
    vids = [(n, a.root / f"colltest_{n}" / ("debug/debug_external_onboard.mp4" if n == "debug_sideways" else "video/collision_test.mp4"))
            for n in scenarios.NAMES]
    md += ["", "Videos (external camera + onboard, debug only):", ""] + [f"- {n}: `{v}`" + ("" if v.exists() else " (missing)")
                                                                          for n, v in vids]
    md += ["", "Failed checks:", ""] + [f"- {r['test']}: {[k for k, v in r.get('checks', {}).items() if not v] or r.get('why')}"
                                        for r in rows if not r["pass"]]
    (HERE / "tests/RESULTS.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    sys.exit(0 if out["all_pass"] else 1)


if __name__ == "__main__":
    main()
