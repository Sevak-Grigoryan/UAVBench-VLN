"""
PUBLISH one collision-aware benchmark episode from its flown + rendered work dir (login node, ffmpeg only).

    python publish_episode.py --work /mnt/weka/sgrigoryan/Benchmark_collision_work/vlncol_forest_037 \
                              --dest /mnt/weka/sgrigoryan/Benchmark_collision/Easy/Forest/Simulation_1

dest/
  task.json          WHAT THE AGENT GETS: scene, start pose, instruction (+ paraphrases), observation / action interface
  goal.json          HIDDEN, evaluator only: target, stand-off distance + tolerance, success rule, thresholds, reference stats
  reference/         the expert solution of the episode (one valid way to succeed, NOT the definition of success)
    trajectory.mp4   onboard camera, episode frames only (no take-off / landing)
    action.json      sticks + gimbal command, one entry per change, ends with STOP
    telemetry.json   every frame: pose, attitude quaternion, physics step, velocity, camera pose + extrinsics, PX4 state
    collision.json   the PhysX contact monitor's record of the episode (collision-v1): monitor verified, contacts, coverage
  results/           filled by evaluate.py / selftest.py

Refuses to publish if: dest is not empty, the video frame count differs from the telemetry, a paraphrase is too close to the
canonical instruction (similarity >= 0.8), a paraphrase / the short instruction drops the stand-off distance, the flight has no
verified collision monitoring, or the expert touched anything during the episode.
"""
import argparse
import difflib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import collision_record  # noqa: E402
import geometry as geo  # noqa: E402
from window import episode_window  # noqa: E402

FFMPEG = "/mnt/weka/shared-cache/miniforge3/bin/ffmpeg"
FFPROBE = "/mnt/weka/shared-cache/miniforge3/bin/ffprobe"
BENCH_VERSION = "vln-x500-bench-collision v0.2 (2026-11-07; v0.1 + PhysX collision termination, collision-v1)"
FPS = 30
GIMBAL_STEP_DEG = 5.0
TIME_LIMIT_FACTOR = 2.0
STANDOFF_TOL = (10.0, 0.30)     # |horizontal distance - D| <= max(10 m, 30 % of D)
FRAME_MARGIN = 0.10              # target point inside the image with 10 % of width / height clear at every edge
MIN_AGL_M, MAX_TILT_DEG = 2.0, 60.0
SIMILARITY_MAX = 0.8


def nframes(p):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(p)], capture_output=True, text=True).stdout.strip()
    return int(out) if out.isdigit() else 0


def stitch(work, f0, f1, out):
    slices = []
    for p in sorted((work / "slices").glob("slice_*.json")):
        info = json.loads(p.read_text())
        if f0 <= info["start"] < f1:
            slices.append((info, p.with_suffix(".mp4")))
    pos = f0
    for info, mp4 in slices:
        if info["start"] != pos or not mp4.exists() or nframes(mp4) != info["frames"]:
            sys.exit(f"slice coverage broken at frame {pos} ({mp4.name})")
        pos = info["end"]
    if pos != f1:
        sys.exit(f"slices cover [{f0},{pos}) but the episode is [{f0},{f1})")
    lst = work / "slices" / "concat_episode.txt"
    lst.write_text("".join(f"file '{mp4.resolve()}'\n" for _, mp4 in slices))
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
    got = nframes(out)
    if got != f1 - f0:
        sys.exit(f"stitched video has {got} frames, the episode has {f1 - f0}")


def episode_actions(tele, flight_actions, f0, f1):
    """Sticks the expert held + the gimbal pitch command (5 deg steps), one entry per change, then STOP."""
    out = []
    for f in tele[f0:f1]:
        a = flight_actions[f["action_index"]]
        g = GIMBAL_STEP_DEG * round(f["camera"]["pitch_deg"] / GIMBAL_STEP_DEG)
        cmd = (a["x1"], a["y1"], a["x2"], a["y2"], g)
        if not out or tuple(out[-1][k] for k in ("x1", "y1", "x2", "y2", "gimbal_pitch_deg")) != cmd:
            if out:
                out[-1]["hold_frames"] = f["frame"] - f0 - out[-1]["frame"]
                out[-1]["hold_s"] = round(out[-1]["hold_frames"] / FPS, 4)
            out.append({"index": len(out), "frame": f["frame"] - f0, "t": round((f["frame"] - f0) / FPS, 4),
                        "x1": cmd[0], "y1": cmd[1], "x2": cmd[2], "y2": cmd[3], "gimbal_pitch_deg": g})
    out[-1]["hold_frames"] = f1 - f0 - out[-1]["frame"]
    out[-1]["hold_s"] = round(out[-1]["hold_frames"] / FPS, 4)
    out.append({"index": len(out), "frame": f1 - f0, "t": round((f1 - f0) / FPS, 4), "stop": True})
    return out


def episode_telemetry(tele, acts, f0, f1):
    out, ai = [], 0
    for f in tele[f0:f1]:
        i = f["frame"] - f0
        while ai + 1 < len(acts) and not acts[ai + 1].get("stop") and acts[ai + 1]["frame"] <= i:
            ai += 1
        out.append({"frame": i, "t": round(i / FPS, 4), "action_index": ai, "physics_step": f["physics_step"] - tele[f0]["physics_step"],
                    "pose": f["pose"], "attitude_quat_xyzw": f["attitude_quat_xyzw"], "velocity": f["velocity"],
                    "agl_m": f["agl_m"], "camera": f["camera"], "camera_extrinsics": geo.extrinsics(f["camera"]),
                    "px4": f["px4"]})
    return out


def check_instructions(canon, para, short, D):
    num = str(int(round(D)))
    for p in para:
        s = difflib.SequenceMatcher(None, canon.lower(), p.lower()).ratio()
        if s >= SIMILARITY_MAX:
            sys.exit(f"paraphrase too close to the canonical instruction ({s:.2f}): {p}")
        if num not in p:
            sys.exit(f"paraphrase drops the {num} m stand-off: {p}")
    if num not in short:
        sys.exit(f"short instruction drops the {num} m stand-off: {short}")
    return [round(difflib.SequenceMatcher(None, canon.lower(), p.lower()).ratio(), 3) for p in para]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--no-video", action="store_true", help="publish everything except reference/trajectory.mp4")
    a = ap.parse_args()
    work, dest = a.work, a.dest
    video = dest / "reference" / "trajectory.mp4"
    if (dest / "task.json").exists() and not video.exists():
        # published earlier with --no-video: only add the video, and check it matches the published telemetry
        f0, f1, _, _ = episode_window(work / "flight")
        n = len(json.loads((dest / "reference" / "telemetry.json").read_text()))
        if f1 - f0 != n:
            sys.exit(f"episode window has {f1 - f0} frames but the published telemetry has {n}")
        stitch(work, f0, f1, video)
        print(f"added {video} ({n} frames)")
        return
    if dest.exists() and any(dest.iterdir()):
        sys.exit(f"refusing to overwrite non-empty {dest}")
    flight = work / "flight"
    if not (flight / "FLIGHT_OK").exists():
        sys.exit("flight not finished OK")
    meta = json.loads((flight / "meta.json").read_text())
    tele = json.loads((flight / "telemetry.json").read_text())
    fact = json.loads((flight / "action.json").read_text())["actions"]
    mid = meta["mission"]["id"]
    m = meta["mission"]
    bench, tgt = m["bench"], m["bench"]["target"]
    f0, f1, t0, t1 = episode_window(flight)
    fcol = json.loads((flight / "collision.json").read_text())
    ecol = collision_record.episode_record(fcol, tele[f0]["t"], tele[f1 - 1]["t"], tele[f0]["physics_step"])
    if not ecol["monitor_verified"]:
        sys.exit("collision monitor not verified for this flight")
    if ecol["first_prohibited"] or ecol["invalid"]:
        sys.exit(f"the expert touched something during the episode: {ecol['first_prohibited'] or ecol['invalid']}")

    ref = dest / "reference"
    ref.mkdir(parents=True, exist_ok=True)
    (dest / "results").mkdir(exist_ok=True)
    if not a.no_video:
        stitch(work, f0, f1, ref / "trajectory.mp4")
    acts = episode_actions(tele, fact, f0, f1)
    et = episode_telemetry(tele, acts, f0, f1)
    (ref / "action.json").write_text(json.dumps({"actions": acts}, indent=1))
    (ref / "telemetry.json").write_text(json.dumps(et))
    (ref / "collision.json").write_text(json.dumps(ecol, indent=1))

    D = bench["standoff_m"]
    pp = json.loads((HERE / "paraphrases.json").read_text()).get(mid)
    if not pp:
        sys.exit(f"no paraphrases for {mid} in paraphrases.json")
    sims = check_instructions(m["instruction"], pp["paraphrases"], pp["short"], D)
    K = meta["camera_intrinsics"]
    p0, last = et[0], et[-1]
    look = tgt["look_at_xyz"]
    ref_len = geo.path_length(et)
    dur = round(len(et) / FPS, 3)
    t_limit = 10.0 * math.ceil(TIME_LIMIT_FACTOR * dur / 10.0)
    tol = max(STANDOFF_TOL[0], STANDOFF_TOL[1] * D)
    ep_id = f"Easy/Forest/{dest.name}"
    sc = bench["scene"]
    view0 = geo.in_frame(geo.project(p0["camera"], look, K), meta["frame_size"], 0.0)

    task = {
        "episode_id": ep_id,
        "benchmark_version": BENCH_VERSION,
        "difficulty": "Easy",
        "task_type": "inspect (stop at a stand-off distance, facing the target)",
        "instruction": m["instruction"],
        "instruction_paraphrases": pp["paraphrases"],
        "instruction_short": pp["short"],
        "scene": {"environment": "Forest", "scene_id": sc["scene_mission"], "source_simulation": sc["source_simulation"],
                  "rebuild": f"dataset_codebase Forest scene with the Easy v2 props layout of {sc['scene_mission']} "
                             f"(envs.boot('Forest', ..., mission_id='{sc['scene_mission']}'))"},
        "start": {"xyz": [round(v, 3) for v in geo.xyz(p0)], "yaw_deg": round(p0["pose"]["yaw_deg"], 2),
                  "camera_pitch_deg": round(p0["camera"]["pitch_deg"], 1),
                  "state": "in the air, hovering, armed, PX4 Position mode",
                  "note": "the runner flies the take-off and the turn to this heading by itself and hands control to the agent here"},
        "time_limit_s": t_limit,
        "observation": {
            "rgb": {"size": meta["frame_size"], "fps": FPS, "camera_intrinsics": K,
                    "mount": "0.12 m forward of and 0.07 m below the airframe centre, yaw locked to the airframe, stabilised (no roll)"},
            "pose": "drone position (ENU metres) and heading every frame, as in telemetry.json pose",
            "instruction": "text",
        },
        "action": {
            "decision_interval_s": 0.5,
            "sticks": {"x1": "yaw, + = rotate clockwise", "y1": "throttle, + = climb", "x2": "roll, + = move right",
                       "y2": "pitch, + = move forward", "range": [-1.0, 1.0], "quantisation": 0.1,
                       "response_at_full_stick": {"horizontal_mps": 8.0, "climb_mps": 3.0, "descent_mps": 2.0, "yaw_dps": 60.0}},
            "gimbal_pitch_deg": {"range": [0.0, 80.0], "quantisation": GIMBAL_STEP_DEG, "meaning": "positive = looking down",
                                 "slew_dps": 35.0},
            "stop": "ends the episode; the drone holds its position (the agent says: I am done)",
            "collision": "the simulator ends the episode at the first prohibited contact (any contact of the drone with "
                         "anything; this episode has no landing), right after that 240 Hz physics step",
            "autopilot": "PX4 v1.14.3 SITL, Position mode, sticks over MAVLink MANUAL_CONTROL",
        },
        "coordinate_frame": "ENU metres (x east, y north, z up); yaw 0 = +x, counter-clockwise positive; camera pitch positive = down",
    }
    goal = {
        "episode_id": ep_id,
        "hidden": "evaluator only -- never give this file to the agent",
        "target": {"name": tgt["name"], "description": tgt["description"], "kind": tgt["kind"],
                   "xy": [round(v, 3) for v in tgt["xy"]], "point_xyz": [round(v, 3) for v in look],
                   "base_z": round(tgt["base_z"], 2), "top_z": round(tgt["top_z"], 2),
                   "in_view_at_start": view0, "start_turn_from_target_deg": bench["start_turn_from_target_deg"]},
        "success": {
            "rule": (f"the agent says STOP within {t_limit:.0f} s, and at that moment (1) its horizontal distance to the target "
                     f"is {D:.0f} +- {tol:.0f} m, (2) the target point is inside the image with {FRAME_MARGIN:.0%} of the frame "
                     f"clear at every edge, and (3) during the whole episode it never tilted beyond {MAX_TILT_DEG:.0f} deg "
                     f"nor came below {MIN_AGL_M:.0f} m above the ground (no crash, no landing), and (4) no part of the "
                     f"drone (rotors, airframe, landing gear) touched anything -- the PhysX contact monitor's first "
                     f"prohibited contact ends and fails the episode; a flight without verified collision monitoring is invalid"),
            "collision": {"version": ecol["version"], "rule": ecol["rule"], "landing_exception": None,
                          "touch_separation_m": ecol["touch_separation_m"], "physics_hz": ecol["physics"]["hz"]},
            "standoff_m": D, "standoff_tolerance_m": tol, "distance": "horizontal (x, y), drone to target.xy",
            "frame_margin": FRAME_MARGIN, "max_tilt_deg": MAX_TILT_DEG, "min_agl_m": MIN_AGL_M, "time_limit_s": t_limit,
            "not_required": "the route, the speed, the height, how long it hovers -- any flight that ends in the goal state succeeds",
        },
        "metrics": {
            "SR": "success (rule above)",
            "OSR": "oracle success: at SOME frame the drone was in the goal state (distance band + target in frame), even if it did not stop there",
            "NE": "navigation error at STOP = | horizontal distance to target - standoff_m | (0 = exactly on the stand-off ring)",
            "SPL": "SR x reference_path_m / max(path_m, reference_path_m)",
            "heading_error_deg": "angle between the drone heading and the direction to the target at STOP",
        },
        "reference": {"path_m": round(ref_len, 1), "duration_s": dur, "frames": len(et), "actions": len(acts),
                      "stop_xyz": [round(v, 2) for v in geo.xyz(last)],
                      "files": "reference/trajectory.mp4, reference/action.json, reference/telemetry.json",
                      "flown": {"work_dir": str(work), "flight_frames": [f0, f1], "flight_t": [t0, t1],
                                "verification": meta["verification"]},
                      "collision": "reference/collision.json (monitor verified, no prohibited contact)"},
        "instruction_checks": {"paraphrase_similarity_to_canonical": sims, "similarity_limit": SIMILARITY_MAX,
                               "paraphrase_source": pp["source"]},
    }
    (dest / "task.json").write_text(json.dumps(task, indent=1))
    (dest / "goal.json").write_text(json.dumps(goal, indent=1))
    print(f"published {ep_id}: {len(et)} frames ({dur} s), {len(acts)} actions, path {ref_len:.0f} m, "
          f"target in view at start: {view0} -> {dest}")


if __name__ == "__main__":
    main()
