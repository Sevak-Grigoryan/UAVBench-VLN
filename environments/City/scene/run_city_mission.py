"""Simulation_0_0_5: A-to-B city sortie to the tower landmark.

    START_PAD (-93, 93)  --fly-->  TOWER (31, 31)  --orbit-->  END_PAD (93, -31)

Same structure and data format as Simulations_0_0_4's forest mission: smooth rate-limited yaw, a
gimbal that aims at the target, and action.json / meta.json / frames / trajectory.mp4 output, with
track_controller.mp4 built afterwards from the recorded data.

Target is a TOWER rather than a bus station: no bus or shelter model exists in any available
library, so both would have to be faked from boxes and would be the least convincing thing in
frame at orbit range. See city_scene.py's module docstring.

Altitudes are derived from the built geometry, not guessed: the tower is 58m to the parapet and
~71.8m to the mast tip, so the orbit flies ABOVE the roofline of the surrounding blocks (tallest
~34m) and looks down at the tower's upper floors.
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

import city_scene
from encode_video import VideoWriter
from flight_controller import simulate_mission

PLACEHOLDER_INSTRUCTION = "[instruction generation not yet wired up for Simulation_0_0_5]"
MAX_GIMBAL_PITCH_DEG = 55.0

# Orbit the tower's upper third: high enough to clear every surrounding block (tallest ~34m),
# close enough that the facade detail reads.
ORBIT_RADIUS_M = 38.0
TARGET_Z_M = 40.0          # a point on the tower shaft, not its base
FLIGHT_HEIGHT_ABOVE_TARGET_M = 12.0   # => ~52m absolute, ~18m over the tallest neighbouring roof
# 38m radius at 25deg/s would need 16.6 m/s of tangential ground speed. 10deg/s gives ~6.6 m/s,
# which is what a real inspection orbit looks like.
ORBIT_SPEED_DEG = 10.0


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
    ap.add_argument("--out-dir", default="/mnt/weka/sgrigoryan/Simulation_0_0_5/data_city")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-seconds", type=float, default=170.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir or f"/mnt/weka/sgrigoryan/Simulation_0_0_5/_runs/{os.environ.get('SLURM_JOB_ID','local')}")

    import city_bridge

    bridge = city_bridge.CityBridge(run_dir, seed=args.seed)
    print(f"CityBridge booted, render_mode={bridge.current_render_mode()} (RSS {rss_gb():.2f} GB)")

    sx, sy = city_scene.START_PAD
    ex, ey = city_scene.END_PAD
    tx, ty = city_scene.TOWER_POS
    start_xyz = (sx, sy, 0.0)
    land_xyz = (ex, ey, 0.0)
    target_xyz = (tx, ty, TARGET_Z_M)

    print(f"START pad ({sx:.1f}, {sy:.1f})  {city_scene.distance_to_road(sx, sy):.1f}m from any road")
    print(f"TARGET tower ({tx:.1f}, {ty:.1f}, {TARGET_Z_M:.1f})  height={city_scene.TOWER_HEIGHT_M}m  "
          f"orbit_radius={ORBIT_RADIUS_M}m  altitude={TARGET_Z_M + FLIGHT_HEIGHT_ABOVE_TARGET_M:.1f}m")
    print(f"END pad   ({ex:.1f}, {ey:.1f})  {city_scene.distance_to_road(ex, ey):.1f}m from any road")

    mission_dir = out_dir / "simulation_city_tower_A2B_0"
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
        vw = VideoWriter(video_path, city_bridge.hq_bridge.FRAME_SIZE[0],
                         city_bridge.hq_bridge.FRAME_SIZE[1], fps=meta["fps"])
        from PIL import Image

        for j, rec in enumerate(records):
            tf0 = time.time()
            p = rec["pose"]
            cam_z = p["z"] + 1.0
            dxy = math.hypot(target_xyz[0] - p["x"], target_xyz[1] - p["y"])
            pitch_deg = math.degrees(math.atan2(cam_z - target_xyz[2], max(dxy, 1e-3)))
            pitch_deg = max(0.0, min(MAX_GIMBAL_PITCH_DEG, pitch_deg))
            rec["camera_pitch_deg"] = round(pitch_deg, 3)
            rec["camera_position"] = {"x": round(p["x"], 4), "y": round(p["y"], 4), "z": round(cam_z, 4),
                                       "yaw_deg": p["yaw_deg"], "pitch_deg": round(pitch_deg, 3)}
            bridge.set_camera_pose(p["x"], p["y"], cam_z, p["yaw_deg"], pitch_deg)
            frame = bridge.process_camera_data()
            Image.fromarray(frame).save(frames_dir / f"image_{j}.png")
            vw.write_frame(frame)
            print(f"[0_0_5] frame {j}/{len(records)} ({time.time()-tf0:.2f}s, {rec['phase']}, "
                  f"pitch={pitch_deg:.1f}deg, RSS {rss_gb():.2f} GB)", flush=True)
        vw.close()
        render_s = time.time() - t0
        print(f"[0_0_5] tower: {len(records)} frames in {render_s:.1f}s "
              f"({render_s / max(1, len(records)):.3f}s/frame)")

        action_path.write_text(json.dumps(records, indent=2))
        full_meta = {
            "simulation_id": mission_dir.name,
            "instruction": PLACEHOLDER_INSTRUCTION,
            "mission_type": "point_A_to_point_B",
            "environment": "authored city (Poly Haven CC0 PBR textures + street models)",
            "environment_info": bridge.city_info,
            "environment_seed": args.seed,
            "target_name": "tower",
            "target_category": "city_landmark",
            "target_position": {"x": target_xyz[0], "y": target_xyz[1], "z": target_xyz[2]},
            "tower_height_m": city_scene.TOWER_HEIGHT_M,
            "start_pad_road_clearance_m": city_scene.distance_to_road(sx, sy),
            "land_pad_road_clearance_m": city_scene.distance_to_road(ex, ey),
            "frame_size": list(city_bridge.hq_bridge.FRAME_SIZE),
            "render_mode": bridge.current_render_mode(),
            "render_seconds": render_s,
            "generation_seconds": gen_s,
            **meta,
        }
        meta_path.write_text(json.dumps(full_meta, indent=2))
        print(f"[0_0_5] wrote {action_path} and {meta_path}")
    finally:
        print(f"final RSS: {rss_gb():.2f} GB")
        print("shutting down city bridge cleanly...")
        bridge.shutdown()


if __name__ == "__main__":
    main()
