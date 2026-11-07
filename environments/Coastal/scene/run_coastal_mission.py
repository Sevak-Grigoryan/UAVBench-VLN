"""Simulation_0_0_7: an A-to-B coastal sortie to the lighthouse.

    START_PAD (coastal plain)  --fly-->  LIGHTHOUSE (headland)  --orbit-->  END_PAD (elsewhere along the coast)

Same structure and data format as the mountain/city/forest missions: acceleration-rate-limited
flight (flight_controller.py, city-scale speed profile -- this mission is smaller in both
target-height and pad-distance than the mountain's, so its faster profile isn't needed here), a
gimbal that aims at the target, and action.json / meta.json / frames / trajectory.mp4 output, with
track_controller.mp4 built afterwards from the recorded data.

Target is the lighthouse's gallery level (not the bare headland point or the very top) -- orbiting
a real structure, like the mountain's summit lookout. Altitude and orbit radius are derived from
measured built geometry (coastal_scene.build_lighthouse's returned gallery/lamp/finial heights and
a FULL-360-degree orbit-clearance check against the real terrain), not guessed.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import coastal_scene
from encode_video import VideoWriter
from flight_controller import simulate_mission

PLACEHOLDER_INSTRUCTION = "[instruction generation not yet wired up for Simulation_0_0_7]"
MAX_GIMBAL_PITCH_DEG = 55.0

# 28m orbit radius clears the tower/gallery comfortably (checked around the FULL 360-degree ring,
# not a few samples -- the mountain project's real bug). 15deg/s keeps the required tangential
# speed (28m * radians(15deg) = 7.33 m/s) under flight_controller's 8.0 m/s cap.
ORBIT_RADIUS_M = 28.0
FLIGHT_HEIGHT_ABOVE_TARGET_M = 16.0
ORBIT_SPEED_DEG = 15.0


def rss_gb():
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:
                pass
        return total / (1024.0 ** 3)
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/mnt/weka/sgrigoryan/Simulation_0_0_7/data_coastal")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-seconds", type=float, default=220.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir or f"/mnt/weka/sgrigoryan/Simulation_0_0_7/_runs/{os.environ.get('SLURM_JOB_ID','local')}")

    import coastal_bridge

    bridge = coastal_bridge.CoastalBridge(run_dir, seed=args.seed)
    print(f"CoastalBridge booted, render_mode={bridge.current_render_mode()} (RSS {rss_gb():.2f} GB)")

    sx, sy = coastal_scene.START_PAD
    ex, ey = coastal_scene.END_PAD
    info = bridge.coastal_info
    tx, ty, tz = info["lighthouse_x"], info["lighthouse_y"], info["lighthouse_gallery_z"]

    start_xyz = (sx, sy, bridge.terrain_height(sx, sy))
    land_xyz = (ex, ey, bridge.terrain_height(ex, ey))
    target_xyz = (tx, ty, tz)

    print(f"START pad ({sx:.1f}, {sy:.1f}, {start_xyz[2]:.1f})  "
          f"{coastal_scene.headland_r(sx, sy):.0f}m from the lighthouse")
    print(f"TARGET lighthouse ({tx:.1f}, {ty:.1f}, {tz:.1f})  "
          f"ground={info['lighthouse_ground_z']:.1f}m  lamp_top={info['lighthouse_lamp_top_z']:.1f}m  "
          f"finial_top={info['lighthouse_finial_top_z']:.1f}m  "
          f"orbit_radius={ORBIT_RADIUS_M}m  altitude={tz + FLIGHT_HEIGHT_ABOVE_TARGET_M:.1f}m")
    print(f"END pad   ({ex:.1f}, {ey:.1f}, {land_xyz[2]:.1f})  "
          f"{coastal_scene.headland_r(ex, ey):.0f}m from the lighthouse")

    orbit_issues, worst = coastal_scene.validate_orbit_clearance(
        ORBIT_RADIUS_M, tz + FLIGHT_HEIGHT_ABOVE_TARGET_M, seed=args.seed)
    print(f"orbit clearance: worst {worst[0]:.2f}m at {worst[1]}deg (terrain {worst[2]:.1f}m there)")
    if orbit_issues:
        raise RuntimeError("orbit clearance check failed: " + "; ".join(orbit_issues))

    mission_dir = out_dir / "simulation_coastal_lighthouse_A2B_0"
    frames_dir = mission_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    action_path = mission_dir / "action.json"
    meta_path = mission_dir / "meta.json"
    video_path = mission_dir / "trajectory.mp4"

    try:
        t0 = time.time()
        records, meta = simulate_mission(start_xyz, target_xyz, max_seconds=args.max_seconds,
                                          orbit_radius=ORBIT_RADIUS_M,
                                          height_above_target=FLIGHT_HEIGHT_ABOVE_TARGET_M,
                                          orbit_speed_deg=ORBIT_SPEED_DEG,
                                          land_xyz=land_xyz)
        gen_s = time.time() - t0
        phases = {}
        for r in records:
            phases[r["phase"]] = phases.get(r["phase"], 0) + 1
        print(f"trajectory: {len(records)} frames, {meta['duration_sec']}s, phases={phases}")
        if meta.get("hit_max_frames"):
            print("WARNING: hit max_seconds -- mission did not finish landing. Raise --max-seconds.")

        t0 = time.time()
        vw = VideoWriter(video_path, coastal_bridge.hq_bridge.FRAME_SIZE[0],
                         coastal_bridge.hq_bridge.FRAME_SIZE[1], fps=meta["fps"])
        from PIL import Image

        for j, rec in enumerate(records):
            tf0 = time.time()
            p = rec["pose"]
            cam_z = p["z"] + 2.2
            dxy = math.hypot(target_xyz[0] - p["x"], target_xyz[1] - p["y"])
            pitch_deg = math.degrees(math.atan2(cam_z - target_xyz[2], max(dxy, 1e-3)))
            # A perfectly LEVEL (0 deg) camera 1m above open ground looks straight into the sun's
            # grazing-incidence reflection off the terrain -- rendered as a shimmering, wave-like
            # glare on the grass (traj_takeoff.png), which is what actually looked like "starting
            # from the ocean", not real proximity to the sea (the pads are verified >300m from any
            # water in every direction). A real gimbal shot essentially never holds dead-level
            # framing either. MIN_PITCH_DEG=4 plus a slightly higher camera keeps a small
            # downward tilt at all times.
            MIN_PITCH_DEG = 4.0
            pitch_deg = max(MIN_PITCH_DEG, min(MAX_GIMBAL_PITCH_DEG, pitch_deg))
            rec["camera_pitch_deg"] = round(pitch_deg, 3)
            rec["camera_position"] = {"x": round(p["x"], 4), "y": round(p["y"], 4), "z": round(cam_z, 4),
                                       "yaw_deg": p["yaw_deg"], "pitch_deg": round(pitch_deg, 3)}
            bridge.set_camera_pose(p["x"], p["y"], cam_z, p["yaw_deg"], pitch_deg)
            frame = bridge.process_camera_data()
            Image.fromarray(frame).save(frames_dir / f"image_{j}.png")
            vw.write_frame(frame)
            print(f"[0_0_7] frame {j}/{len(records)} ({time.time()-tf0:.2f}s, {rec['phase']}, "
                  f"pitch={pitch_deg:.1f}deg, RSS {rss_gb():.2f} GB)", flush=True)
        vw.close()
        render_s = time.time() - t0
        print(f"[0_0_7] lighthouse: {len(records)} frames in {render_s:.1f}s "
              f"({render_s / max(1, len(records)):.3f}s/frame)")

        action_path.write_text(json.dumps(records, indent=2))
        full_meta = {
            "simulation_id": mission_dir.name,
            "instruction": PLACEHOLDER_INSTRUCTION,
            "mission_type": "point_A_to_point_B",
            "environment": "authored coastline (Poly Haven CC0 PBR textures, real elevation)",
            "environment_info": bridge.coastal_info,
            "environment_seed": args.seed,
            "target_name": "lighthouse",
            "target_category": "coastal_landmark",
            "target_position": {"x": target_xyz[0], "y": target_xyz[1], "z": target_xyz[2]},
            "lighthouse_ground_m": info["lighthouse_ground_z"],
            "start_pad_headland_distance_m": coastal_scene.headland_r(sx, sy),
            "land_pad_headland_distance_m": coastal_scene.headland_r(ex, ey),
            "frame_size": list(coastal_bridge.hq_bridge.FRAME_SIZE),
            "render_mode": bridge.current_render_mode(),
            "render_seconds": render_s,
            "generation_seconds": gen_s,
            **meta,
        }
        meta_path.write_text(json.dumps(full_meta, indent=2))
        print(f"[0_0_7] wrote {action_path} and {meta_path}")
    finally:
        print(f"final RSS: {rss_gb():.2f} GB")
        print("shutting down coastal bridge cleanly...")
        bridge.shutdown()


if __name__ == "__main__":
    main()
