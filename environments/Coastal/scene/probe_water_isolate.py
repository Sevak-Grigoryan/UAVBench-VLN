"""Isolate the 'water-like' artifact at the start pad: same (x,y) position, several pitches and
yaws, to determine whether it's purely angle-dependent (grazing incidence) or something else
entirely. Three probe rounds have "fixed" plausible causes with zero visible change -- this
answers the actual question before guessing a fourth."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)

OUT = Path("/mnt/weka/sgrigoryan/Simulation_0_0_7/probe_coastal")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import coastal_bridge
    import coastal_scene

    run_dir = Path(f"/mnt/weka/sgrigoryan/Simulation_0_0_7/_runs/probewater_{os.environ.get('SLURM_JOB_ID','local')}")
    bridge = coastal_bridge.CoastalBridge(run_dir, seed=7)
    print(f"booted, render_mode={bridge.current_render_mode()}", flush=True)

    sx, sy = coastal_scene.START_PAD
    gz = bridge.terrain_height(sx, sy)
    print(f"START_PAD ground z: {gz:.2f}", flush=True)

    shots = [
        ("pitch00_yaw0", sx, sy, gz + 3.0, 0.0, 0.0),
        ("pitch04_yaw0", sx, sy, gz + 3.0, 0.0, 4.0),
        ("pitch20_yaw0", sx, sy, gz + 3.0, 0.0, 20.0),
        ("pitch40_yaw0", sx, sy, gz + 3.0, 0.0, 40.0),
        ("pitch04_yaw90", sx, sy, gz + 3.0, 90.0, 4.0),
        ("pitch04_yaw180", sx, sy, gz + 3.0, 180.0, 4.0),
        ("high_pitch10", sx, sy, gz + 20.0, 0.0, 10.0),
    ]
    from PIL import Image

    try:
        for name, x, y, z, yaw, pitch in shots:
            bridge.set_camera_pose(x, y, z, yaw, pitch)
            frame = bridge.process_camera_data()
            Image.fromarray(frame).save(OUT / f"water_{name}.png")
            print(f"wrote water_{name}.png", flush=True)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
