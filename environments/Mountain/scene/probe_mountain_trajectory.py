"""Render a handful of REAL frames from the actual mission trajectory, using the exact same
camera-aiming formula run_mountain_mission.py uses (gimbal pitch = atan2 toward the target).

The hand-aimed viewpoints in probe_mountain_view.py got two shots' yaw wrong (pointed away from
the peak entirely) before this was written -- guessing yaw/pitch by hand is unreliable. Sampling
the REAL, already-offline-verified trajectory removes that whole class of mistake: every frame
here is aimed exactly the way the mission will aim it.
"""

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)

OUT = Path("/mnt/weka/sgrigoryan/Simulation_0_0_6/probe_mountain")
MAX_GIMBAL_PITCH_DEG = 55.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import mountain_bridge
    import mountain_scene
    from flight_controller import simulate_mission

    run_dir = Path(f"/mnt/weka/sgrigoryan/Simulation_0_0_6/_runs/probetraj_{os.environ.get('SLURM_JOB_ID','local')}")
    bridge = mountain_bridge.MountainBridge(run_dir, seed=7)
    print(f"booted, render_mode={bridge.current_render_mode()}", flush=True)

    sx, sy = mountain_scene.START_PAD
    ex, ey = mountain_scene.END_PAD
    info = bridge.mountain_info
    tx, ty, tz = info["summit_x"], info["summit_y"], info["summit_deck_z"]
    start_xyz = (sx, sy, bridge.terrain_height(sx, sy))
    land_xyz = (ex, ey, bridge.terrain_height(ex, ey))
    target_xyz = (tx, ty, tz)

    records, meta = simulate_mission(start_xyz, target_xyz, max_seconds=220.0, orbit_radius=24.0,
                                      height_above_target=26.0, orbit_speed_deg=14.0,
                                      land_xyz=land_xyz)
    n = len(records)
    print(f"trajectory: {n} frames, {meta['duration_sec']}s", flush=True)

    # one frame from each phase, plus a couple spread through the orbit
    phase_first = {}
    for i, r in enumerate(records):
        phase_first.setdefault(r["phase"], i)
    picks = list(phase_first.items())
    orbit_idxs = [i for i, r in enumerate(records) if r["phase"] == "orbit"]
    if orbit_idxs:
        picks.append(("orbit_mid", orbit_idxs[len(orbit_idxs) // 2]))
        picks.append(("orbit_late", orbit_idxs[int(len(orbit_idxs) * 0.8)]))
    # a mid-approach frame (closing in on the mountain, treeline visible)
    approach_idxs = [i for i, r in enumerate(records) if r["phase"] == "approach"]
    if approach_idxs:
        picks.append(("approach_mid", approach_idxs[len(approach_idxs) // 2]))

    from PIL import Image

    try:
        for name, idx in picks:
            rec = records[idx]
            p = rec["pose"]
            cam_z = p["z"] + 1.0
            dxy = math.hypot(target_xyz[0] - p["x"], target_xyz[1] - p["y"])
            pitch_deg = math.degrees(math.atan2(cam_z - target_xyz[2], max(dxy, 1e-3)))
            pitch_deg = max(0.0, min(MAX_GIMBAL_PITCH_DEG, pitch_deg))
            bridge.set_camera_pose(p["x"], p["y"], cam_z, p["yaw_deg"], pitch_deg)
            frame = bridge.process_camera_data()
            Image.fromarray(frame).save(OUT / f"traj_{name}.png")
            print(f"wrote traj_{name}.png  (frame {idx}/{n}, phase={rec['phase']}, "
                  f"pos=({p['x']:.0f},{p['y']:.0f},{p['z']:.0f}), yaw={p['yaw_deg']:.0f}, "
                  f"pitch={pitch_deg:.1f})", flush=True)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
