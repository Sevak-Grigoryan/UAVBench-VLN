"""
Package the flown collision TEST flights like benchmark episodes, so one can see how a contact shows up in the data
(login node, no GPU). Tests are not benchmark episodes: there is no instruction or target, so task.json / goal.json describe
the test and its expected verdict instead.

    python package_collision_tests.py --root /mnt/weka/sgrigoryan/Benchmark_collision_work \
        --dest /mnt/weka/sgrigoryan/Benchmark_collision/Easy/Forest/Collision_situations wall rotor_ceiling wrong_ground pad_ok

dest/<N>_<test>/
  task.json          the test: scenario, scene, start pose (first frame of the evaluated part), test obstacles
  goal.json          the expected verdict (scenarios.EXPECTED) and the rule
  reference/
    trajectory.mp4   collision video: outside camera following the drone | onboard camera (last <= 12 s up to the end)
    action.json      sticks + gimbal command, one entry per change; ends with the end of the flight (no STOP: tests never say it)
    telemetry.json   every frame of the evaluated part, the episode format + "contacts_touching" (touching contact pairs in
                     that frame), "contact_events" (the monitor's judged contacts inside the frame's physics steps) and, on the
                     last record of a stopped run, "terminal" / "termination"
    collision.json   the PhysX contact monitor's record of the evaluated part (collision_record.episode_record, collision-v1)
  results/
    collision_verdict.json   how the evaluator reads that record: valid / collided / episode end, + matches_expected
Refuses to write into an existing test folder.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collision_record  # noqa: E402
import geometry as geo  # noqa: E402
from publish_episode import BENCH_VERSION, FPS, episode_actions, episode_telemetry  # noqa: E402
from scenarios import EXPECTED  # noqa: E402

DESCRIPTION = {
    "wall": "fly forward at 3 m/s into an upright 0.3 m thick wall spawned 6 m ahead",
    "rotor_ceiling": "climb slowly (0.6 m/s) into a flat plate spawned 0.8 m above the drone",
    "tree": "approach the nearest suitable Forest tree at mid-height",
    "thin_fast": "fly at 8 m/s (full stick) into a 2 cm pole spawned 30 m ahead",
    "wrong_ground": "land 8 m away from the designated landing clearing",
    "pad_ok": "land on the designated landing clearing (allowed contact)",
    "pad_tip": "land on the designated clearing, then a 45 N side push tips the drone over",
}
EVENT_KEYS = ("step", "verdict", "drone_part", "drone_collider", "other_collider", "event", "total_impulse_ns")


def verdict(rec, exp):
    """The evaluator's collision branch (evaluate.py), without the target-dependent parts."""
    if not rec.get("monitor_verified"):
        return {"valid": False, "episode_end": "INVALID: collision monitor not verified"}
    if rec.get("invalid"):
        return {"valid": False, "episode_end": f"INVALID: {rec['invalid'].get('verdict')}"}
    hit = rec.get("first_prohibited")
    out = {"valid": True, "collided": hit is not None, "success_possible": hit is None}
    if hit:
        out["episode_end"] = (f"COLLISION at t = {hit['t']:.4f} s (physics step {hit['episode_step']}): {hit['drone_part']} "
                              f"{hit['drone_collider'].rsplit('/', 1)[-1]} touched {hit['other_collider']}")
        out["collision"] = {k: hit[k] for k in ("t", "episode_step", "drone_part", "drone_collider", "other_collider",
                                                "total_impulse_ns")}
    else:
        out["episode_end"] = "no prohibited contact"
        out["collision"] = None
    al = rec.get("first_allowed_landing")
    out["allowed_landing_contact"] = al and {k: al[k] for k in ("t", "episode_step", "drone_part", "drone_collider", "other_collider")}
    ok = (exp["outcome"] == "collision") == (hit is not None)
    if hit and "first_prohibited_other_prefix" in exp:
        ok &= hit["other_collider"].startswith(exp["first_prohibited_other_prefix"])
    if hit and "first_prohibited_part" in exp:
        ok &= hit["drone_part"] == exp["first_prohibited_part"]
    if hit and "first_prohibited_part_in" in exp:
        ok &= hit["drone_part"] in exp["first_prohibited_part_in"]
    if exp.get("allowed_landing") and exp["outcome"] == "completed":
        ok &= al is not None
    out["matches_expected"] = bool(ok)
    return out


def package(work, dest, name):
    fl = work / "flight"
    tele = json.loads((fl / "telemetry.json").read_text())
    fact = json.loads((fl / "action.json").read_text())["actions"]
    col = json.loads((fl / "collision.json").read_text())
    ep = [i for i, r in enumerate(tele) if r["phase"] == "episode"]
    f0 = ep[0]
    f1 = len(tele) - 1 if tele[-1].get("terminal") else len(tele)       # the terminal record (contact step) is appended below
    t0, t1, step0 = tele[f0]["t"], tele[-1]["t"], tele[f0]["physics_step"]
    rec = collision_record.episode_record(col, t0, t1, step0)
    ended = col.get("termination") is not None

    acts = episode_actions(tele, fact, f0, f1)
    acts[-1] = {"index": acts[-1]["index"], "frame": f1 - f0, "t": round((f1 - f0) / FPS, 4),
                "end": col["termination"]["reason"] if ended else "end of the evaluated part (no STOP in a test)"}
    et = episode_telemetry(tele, acts, f0, f1)
    # contact events (judged by the monitor) -> the frame whose physics steps contain them
    ev = [e for e in col["events"] if step0 < e["step"] <= tele[-1]["physics_step"]]
    for i, r in enumerate(et):
        src, prev = tele[f0 + i], tele[f0 + i - 1]["physics_step"]
        r["contacts_touching"] = src.get("contacts_touching")
        r["contact_events"] = [dict({k: e.get(k) for k in EVENT_KEYS}, episode_step=e["step"] - step0)
                               for e in ev if prev < e["step"] <= src["physics_step"]]
    if f1 < len(tele):
        src, prev = tele[-1], tele[-2]["physics_step"]
        et.append({"frame": f1 - f0, "t": round(src["t"] - t0, 6), "action_index": len(acts) - 2,
                   "physics_step": src["physics_step"] - step0, "pose": src["pose"],
                   "attitude_quat_xyzw": src["attitude_quat_xyzw"], "velocity": src["velocity"], "agl_m": src["agl_m"],
                   "camera": src["camera"], "camera_extrinsics": geo.extrinsics(src["camera"]), "px4": src["px4"],
                   "contacts_touching": src.get("contacts_touching"),
                   "contact_events": [dict({k: e.get(k) for k in EVENT_KEYS}, episode_step=e["step"] - step0)
                                      for e in ev if prev < e["step"] <= src["physics_step"]],
                   "terminal": True, "termination": src.get("termination"),
                   "note": "terminal record: written at the physics step of the first prohibited contact (between video frames)"})

    vid = work / "video" / "collision_test.mp4"
    vmeta = json.loads((work / "video" / "collision_test.json").read_text())
    exp = EXPECTED[name]
    task = {
        "episode_id": f"Easy/Forest/Collision_situations/{dest.name}",
        "benchmark_version": BENCH_VERSION,
        "kind": "collision TEST flight (not a benchmark episode: no instruction, no target)",
        "test": name, "scenario": DESCRIPTION[name],
        "scene": {"environment": "Forest", "scene_id": col.get("scene"), "same_scene_as": "Easy/Forest/Simulation_1"},
        "start": {"xyz": [tele[f0]["pose"][k] for k in ("x", "y", "z")], "yaw_deg": tele[f0]["pose"]["yaw_deg"],
                  "flight_t_s": t0, "note": "first frame of the evaluated part; take-off and settling are flown before it"},
        "test_obstacles": (col.get("test_notes") or {}).get("fixtures") or [],
        "test_notes": {k: v for k, v in (col.get("test_notes") or {}).items() if k != "fixtures"},
        "video": {"file": "reference/trajectory.mp4", "layout": "left: outside camera following the drone; right: onboard camera",
                  "frames": [vmeta["frames"][0] - f0, vmeta["frames"][1] - f0], "stride": vmeta.get("stride"),
                  "note": "frames are episode-relative (telemetry frame index); red banner + circle = first prohibited contact"},
    }
    goal = {"episode_id": task["episode_id"], "expected": exp,
            "rule": rec["rule"] + "; a flight without verified collision monitoring is invalid",
            "source_flight": str(work)}
    dest.mkdir(parents=True)
    (dest / "reference").mkdir()
    (dest / "results").mkdir()
    shutil.copy2(vid, dest / "reference" / "trajectory.mp4")
    (dest / "reference" / "action.json").write_text(json.dumps({"actions": acts}, indent=1))
    (dest / "reference" / "telemetry.json").write_text(json.dumps(et))
    (dest / "reference" / "collision.json").write_text(json.dumps(rec, indent=1))
    (dest / "task.json").write_text(json.dumps(task, indent=1))
    (dest / "goal.json").write_text(json.dumps(goal, indent=1))
    v = verdict(rec, exp)
    (dest / "results" / "collision_verdict.json").write_text(json.dumps(v, indent=1))
    hit_frames = [r["frame"] for r in et if any(e["verdict"] == "prohibited" for e in r["contact_events"])]
    print(f"{dest.name}: {len(et)} frames, {len(acts) - 1} actions, {len(ev)} contact events; {v['episode_end']}; "
          f"first prohibited frame {hit_frames[:1]}; matches expected: {v.get('matches_expected')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("tests", nargs="+")
    a = ap.parse_args()
    for i, n in enumerate(a.tests, 1):
        d = a.dest / f"{i}_{n}"
        if d.exists():
            sys.exit(f"{d} exists; refusing to overwrite")
        package(a.root / f"colltest_{n}", d, n)


if __name__ == "__main__":
    main()
