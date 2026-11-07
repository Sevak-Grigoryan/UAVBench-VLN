"""Render a handful of REAL frames from the actual mission trajectory, using the exact same
camera-aiming formula run_coastal_mission.py uses (gimbal pitch = atan2 toward the target).

Hand-aimed probe viewpoints proved unreliable on both the mountain and city projects (wrong yaw,
buried cameras, missed framing) -- sampling the REAL, already offline-verified trajectory removes
that whole class of mistake.
"""

import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)

OUT = Path("/mnt/weka/sgrigoryan/Simulation_0_0_7/probe_coastal")
MAX_GIMBAL_PITCH_DEG = 55.0


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import coastal_bridge
    import coastal_scene
    from flight_controller import simulate_mission

    run_dir = Path(f"/mnt/weka/sgrigoryan/Simulation_0_0_7/_runs/probetraj_{os.environ.get('SLURM_JOB_ID','local')}")
    bridge = coastal_bridge.CoastalBridge(run_dir, seed=7)
    print(f"booted, render_mode={bridge.current_render_mode()}", flush=True)

    sx, sy = coastal_scene.START_PAD
    ex, ey = coastal_scene.END_PAD
    info = bridge.coastal_info
    tx, ty, tz = info["lighthouse_x"], info["lighthouse_y"], info["lighthouse_gallery_z"]
    start_xyz = (sx, sy, bridge.terrain_height(sx, sy))
    land_xyz = (ex, ey, bridge.terrain_height(ex, ey))
    target_xyz = (tx, ty, tz)

    records, meta = simulate_mission(start_xyz, target_xyz, max_seconds=220.0, orbit_radius=28.0,
                                      height_above_target=16.0, orbit_speed_deg=15.0,
                                      land_xyz=land_xyz)
    n = len(records)
    print(f"trajectory: {n} frames, {meta['duration_sec']}s", flush=True)

    phase_first = {}
    for i, r in enumerate(records):
        phase_first.setdefault(r["phase"], i)
    picks = list(phase_first.items())
    orbit_idxs = [i for i, r in enumerate(records) if r["phase"] == "orbit"]
    if orbit_idxs:
        picks.append(("orbit_mid", orbit_idxs[len(orbit_idxs) // 2]))
        picks.append(("orbit_late", orbit_idxs[int(len(orbit_idxs) * 0.8)]))
    approach_idxs = [i for i, r in enumerate(records) if r["phase"] == "approach"]
    if approach_idxs:
        picks.append(("approach_mid", approach_idxs[len(approach_idxs) // 2]))
        picks.append(("approach_early", approach_idxs[int(len(approach_idxs) * 0.25)]))

    from PIL import Image

    try:
        for name, idx in picks:
            rec = records[idx]
            p = rec["pose"]
            # must match run_coastal_mission.py's own camera math exactly -- this probe had its
            # own stale copy (cam_z+1.0, min pitch 0deg) that missed the actual mission's fix,
            # so it kept rendering the OLD grazing-angle framing after the real mission script had
            # already moved past it.
            cam_z = p["z"] + 2.2
            dxy = math.hypot(target_xyz[0] - p["x"], target_xyz[1] - p["y"])
            pitch_deg = math.degrees(math.atan2(cam_z - target_xyz[2], max(dxy, 1e-3)))
            MIN_PITCH_DEG = 4.0
            pitch_deg = max(MIN_PITCH_DEG, min(MAX_GIMBAL_PITCH_DEG, pitch_deg))
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
