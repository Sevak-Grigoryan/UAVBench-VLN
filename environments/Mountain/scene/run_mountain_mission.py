"""Simulation_0_0_6: an A-to-B mountain sortie to the summit lookout.

    START_PAD (in the valley)  --fly-->  SUMMIT LOOKOUT (peak, ~186m)  --orbit-->  END_PAD (elsewhere)

Same structure and data format as Simulations_0_0_5's city mission and Simulations_0_0_4's forest
mission: acceleration-rate-limited flight (flight_controller.py, shared unchanged), a gimbal that
aims at the target, and action.json / meta.json / frames / trajectory.mp4 output, with
track_controller.mp4 built afterwards from the recorded data.

Target is the summit's timber lookout deck, not the bare peak point -- landing on/hovering at a
built structure is a more realistic mission than orbiting an arbitrary GPS coordinate. Altitude and
orbit radius are derived from measured built geometry (mountain_scene.build_summit's returned
deck/roof/mast heights and the terrain's own slope near the summit), not guessed.
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

import mountain_scene
from encode_video import VideoWriter
from flight_controller import simulate_mission

PLACEHOLDER_INSTRUCTION = "[instruction generation not yet wired up for Simulation_0_0_6]"
MAX_GIMBAL_PITCH_DEG = 55.0

# 24m orbit radius clears the small lookout platform (5.2m deck) comfortably while staying
# proportionate to it (the city's 38m radius suited a 58m tower; this structure is far smaller).
# 14deg/s keeps the required tangential speed (24m * radians(14deg) = 5.86 m/s) safely under
# flight_controller's 8.0 m/s cap -- the same "orbit_radius * omega must not exceed MAX_HORIZONTAL_
# SPEED" check that both the forest and city missions needed after acceleration limiting landed.
ORBIT_RADIUS_M = 24.0
# 26m, not a smaller "just clears the deck" number: the terrain right around the 24m orbit ring
# has real ridged noise (the same relief that makes the mountain look like rock, not a smooth
# dome) -- checked around the FULL ring, not one radial sample, and it peaks ~17m above the deck
# at one point (220deg). 14m of clearance (the first value tried) put the orbit ring 2.7m INSIDE
# the terrain there; this leaves ~9m of clearance at the worst point.
FLIGHT_HEIGHT_ABOVE_TARGET_M = 26.0
ORBIT_SPEED_DEG = 14.0


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
    ap.add_argument("--out-dir", default="/mnt/weka/sgrigoryan/Simulation_0_0_6/data_mountain")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-seconds", type=float, default=300.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir or f"/mnt/weka/sgrigoryan/Simulation_0_0_6/_runs/{os.environ.get('SLURM_JOB_ID','local')}")

    import mountain_bridge

    bridge = mountain_bridge.MountainBridge(run_dir, seed=args.seed)
    print(f"MountainBridge booted, render_mode={bridge.current_render_mode()} (RSS {rss_gb():.2f} GB)")

    sx, sy = mountain_scene.START_PAD
    ex, ey = mountain_scene.END_PAD
    info = bridge.mountain_info
    tx, ty, tz = info["summit_x"], info["summit_y"], info["summit_deck_z"]

    start_xyz = (sx, sy, bridge.terrain_height(sx, sy))
    land_xyz = (ex, ey, bridge.terrain_height(ex, ey))
    target_xyz = (tx, ty, tz)

    print(f"START pad ({sx:.1f}, {sy:.1f}, {start_xyz[2]:.1f})  "
          f"{mountain_scene.peak_r(sx, sy):.0f}m from the peak")
    print(f"TARGET summit lookout ({tx:.1f}, {ty:.1f}, {tz:.1f})  "
          f"peak_height={info['peak_height_m']:.1f}m  roof_top={info['summit_roof_top_z']:.1f}m  "
          f"mast_tip={info['summit_mast_tip_z']:.1f}m  "
          f"orbit_radius={ORBIT_RADIUS_M}m  altitude={tz + FLIGHT_HEIGHT_ABOVE_TARGET_M:.1f}m")
    print(f"END pad   ({ex:.1f}, {ey:.1f}, {land_xyz[2]:.1f})  "
          f"{mountain_scene.peak_r(ex, ey):.0f}m from the peak")

    orbit_issues, worst = mountain_scene.validate_orbit_clearance(
        ORBIT_RADIUS_M, tz + FLIGHT_HEIGHT_ABOVE_TARGET_M, seed=args.seed)
    print(f"orbit clearance: worst {worst[0]:.2f}m at {worst[1]}deg "
          f"(terrain {worst[2]:.1f}m there)")
    if orbit_issues:
        raise RuntimeError("orbit clearance check failed: " + "; ".join(orbit_issues))

    mission_dir = out_dir / "simulation_mountain_summit_A2B_0"
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
        vw = VideoWriter(video_path, mountain_bridge.hq_bridge.FRAME_SIZE[0],
                         mountain_bridge.hq_bridge.FRAME_SIZE[1], fps=meta["fps"])
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
            print(f"[0_0_6] frame {j}/{len(records)} ({time.time()-tf0:.2f}s, {rec['phase']}, "
                  f"pitch={pitch_deg:.1f}deg, RSS {rss_gb():.2f} GB)", flush=True)
        vw.close()
        render_s = time.time() - t0
        print(f"[0_0_6] summit: {len(records)} frames in {render_s:.1f}s "
              f"({render_s / max(1, len(records)):.3f}s/frame)")

        action_path.write_text(json.dumps(records, indent=2))
        full_meta = {
            "simulation_id": mission_dir.name,
            "instruction": PLACEHOLDER_INSTRUCTION,
            "mission_type": "point_A_to_point_B",
            "environment": "authored mountain (Poly Haven CC0 PBR textures, real elevation)",
            "environment_info": bridge.mountain_info,
            "environment_seed": args.seed,
            "target_name": "summit lookout",
            "target_category": "mountain_landmark",
            "target_position": {"x": target_xyz[0], "y": target_xyz[1], "z": target_xyz[2]},
            "peak_height_m": info["peak_height_m"],
            "start_pad_peak_distance_m": mountain_scene.peak_r(sx, sy),
            "land_pad_peak_distance_m": mountain_scene.peak_r(ex, ey),
            "frame_size": list(mountain_bridge.hq_bridge.FRAME_SIZE),
            "render_mode": bridge.current_render_mode(),
            "render_seconds": render_s,
            "generation_seconds": gen_s,
            **meta,
        }
        meta_path.write_text(json.dumps(full_meta, indent=2))
        print(f"[0_0_6] wrote {action_path} and {meta_path}")
    finally:
        print(f"final RSS: {rss_gb():.2f} GB")
        print("shutting down mountain bridge cleanly...")
        bridge.shutdown()


if __name__ == "__main__":
    main()
