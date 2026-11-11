"""Simulations_0_0_4: a point-A-to-point-B forest sortie.

Changes from Simulations_0_0_3:
  * The drone no longer takes off from / lands on the RIVER. 0_0_3 (and every project before it)
    used (0,0) as the pad, but river_y(0) == 0.0 exactly -- that pad sat dead on the river
    centreline, so the drone was starting and landing in the water.
  * The mission ENDS AT A DIFFERENT POINT than it started, like a real transit/delivery sortie:
        START_PAD (-40, 45)  --fly-->  footbridge (72, 8.2)  --orbit-->  END_PAD (120, -55)
    Both pads are verified dry land (51.7m and 59.9m from the river centreline) and each gets its
    own terrain levelling + vegetation clearing (see forest_scene.PADS).
  * Landing descends to the landing pad's actual GROUND HEIGHT rather than world z=0 (the end
    pad's ground is at +1.33m, so the old hardcoded z=0 would have buried the drone).

Everything else -- the HQ RealTimePathTracing renderer, the smooth terrain, the dense vegetation
layer, the gimballed camera, and the action.json / meta.json / frames / trajectory.mp4 output
format -- is unchanged from 0_0_3.
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

import forest_scene
from encode_video import VideoWriter
from flight_controller import simulate_mission

PLACEHOLDER_INSTRUCTION = "[instruction generation not yet wired up for Simulations_0_0_4]"
MAX_GIMBAL_PITCH_DEG = 55.0

# (x, y_expr, z_offset_above_terrain, orbit_radius_m, height_above_target_m) -- see 0_0_3's notes
# on how these altitudes were derived from measured asset/terrain geometry.
# 16m orbit radius at the old 25deg/s default needs 7.0 m/s of tangential ground speed, which is
# above this mission's 4.0 m/s cap -- the drone would fall behind its own orbit. 14deg/s gives
# 3.9 m/s, inside the limit and what a real inspection orbit looks like.
ORBIT_SPEED_DEG = 14.0

FOREST_TARGETS = {
    "footbridge": (72.0, "river", 2.1, 16.0, 14.0),
    "stone_cairn": (180.0, -55.0, 3.5, 14.0, 10.0),
    "fire_lookout": (130.0, 85.0, 10.5, 18.0, 10.0),
}


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
    ap.add_argument("--target", default="footbridge", choices=sorted(FOREST_TARGETS))
    ap.add_argument("--out-dir", default="/mnt/weka/sgrigoryan/Simulations_0_0_4/data_forest")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--seed", type=int, default=5200)
    ap.add_argument("--assets-root", default="/mnt/weka/sgrigoryan/assets/polyhaven-v2")
    ap.add_argument("--max-seconds", type=float, default=140.0)  # longer: A->target->B, not a round trip
    ap.add_argument("--mem-limit-gb", type=float, default=64.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir or f"/mnt/weka/sgrigoryan/Simulations_0_0_4/_runs/{os.environ.get('SLURM_JOB_ID', 'local')}")

    import forest_bridge

    bridge = forest_bridge.ForestBridge(run_dir, assets_root=args.assets_root, seed=args.seed)
    print(f"ForestBridge booted, render_mode={bridge.current_render_mode()} (RSS {rss_gb():.2f} GB)")

    tx_, ty_spec, z_off, orbit_radius, height_above = FOREST_TARGETS[args.target]
    ty_ = bridge.natural_valley.river_y(tx_) if ty_spec == "river" else float(ty_spec)
    terrain_z = bridge.terrain_height(tx_, ty_)
    target_xyz = (tx_, ty_, terrain_z + z_off)

    sx, sy = forest_scene.START_PAD
    ex, ey = forest_scene.END_PAD
    start_xyz = (sx, sy, bridge.terrain_height(sx, sy))
    land_xyz = (ex, ey, bridge.terrain_height(ex, ey))

    def river_offset(x, y):
        return abs(y - bridge.natural_valley.river_y(x))

    print(f"START pad ({sx:.1f}, {sy:.1f}) ground={start_xyz[2]:.2f}  "
          f"river offset={river_offset(sx, sy):.1f}m")
    print(f"TARGET   {args.target} ({tx_:.1f}, {ty_:.1f}, {target_xyz[2]:.2f})  "
          f"orbit_radius={orbit_radius}m  altitude={target_xyz[2] + height_above:.2f}m")
    print(f"END pad   ({ex:.1f}, {ey:.1f}) ground={land_xyz[2]:.2f}  "
          f"river offset={river_offset(ex, ey):.1f}m")
    for label, (px, py) in (("start", (sx, sy)), ("end", (ex, ey))):
        if river_offset(px, py) < forest_scene.RIVER_HALF_WIDTH_M * 1.5:
            raise RuntimeError(f"{label} pad at ({px},{py}) is in/near the river -- refusing to run")

    mission_dir = out_dir / f"simulation_forest_{args.target}_A2B_0"
    frames_dir = mission_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    action_path = mission_dir / "action.json"
    meta_path = mission_dir / "meta.json"
    video_path = mission_dir / "trajectory.mp4"

    try:
        t0 = time.time()
        records, meta = simulate_mission(start_xyz, target_xyz, max_seconds=args.max_seconds,
                                          orbit_radius=orbit_radius, height_above_target=height_above,
                                          orbit_speed_deg=ORBIT_SPEED_DEG,
                                          land_xyz=land_xyz)
        gen_s = time.time() - t0
        phases = {}
        for r in records:
            phases[r["phase"]] = phases.get(r["phase"], 0) + 1
        print(f"trajectory: {len(records)} frames, {meta['duration_sec']}s, phases={phases} "
              f"(generated in {gen_s:.2f}s)")
        if meta.get("hit_max_frames"):
            print("WARNING: mission hit max_seconds -- it did not finish landing. Raise --max-seconds.")

        t0 = time.time()
        vw = VideoWriter(video_path, forest_bridge.hq_bridge.FRAME_SIZE[0],
                         forest_bridge.hq_bridge.FRAME_SIZE[1], fps=meta["fps"])
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
            print(f"[0_0_4] frame {j}/{len(records)} ({time.time()-tf0:.2f}s, "
                  f"{rec['phase']}, pitch={pitch_deg:.1f}deg, RSS {rss_gb():.2f} GB)", flush=True)
        vw.close()
        render_s = time.time() - t0
        print(f"[0_0_4] {args.target}: {len(records)} frames in {render_s:.1f}s "
              f"({render_s / max(1, len(records)):.3f}s/frame)")

        action_path.write_text(json.dumps(records, indent=2))
        full_meta = {
            "simulation_id": mission_dir.name,
            "instruction": PLACEHOLDER_INSTRUCTION,
            "mission_type": "point_A_to_point_B",
            "environment": "natural_valley (forest/river valley, Poly Haven CC0 assets)",
            "environment_info": bridge.valley_info,
            "environment_seed": args.seed,
            "target_name": args.target,
            "target_category": "forest_landmark",
            "target_position": {"x": target_xyz[0], "y": target_xyz[1], "z": target_xyz[2]},
            "target_terrain_z": terrain_z,
            "start_pad_river_offset_m": river_offset(sx, sy),
            "land_pad_river_offset_m": river_offset(ex, ey),
            "frame_size": list(forest_bridge.hq_bridge.FRAME_SIZE),
            "render_mode": bridge.current_render_mode(),
            "render_seconds": render_s,
            "generation_seconds": gen_s,
            **meta,
        }
        meta_path.write_text(json.dumps(full_meta, indent=2))
        print(f"[0_0_4] wrote {action_path} and {meta_path}")
    finally:
        print(f"final RSS: {rss_gb():.2f} GB")
        print("shutting down forest bridge cleanly...")
        bridge.shutdown()


if __name__ == "__main__":
    main()
