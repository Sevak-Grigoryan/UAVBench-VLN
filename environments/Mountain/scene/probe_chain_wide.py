"""Two wide shots looking ALONG the chain, to answer one specific question: does this read as a
connected chain of mountains, or an isolated peak with the others out of frame? Positions were
checked against real terrain clearance offline before use (see the comment in each shot)."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)

OUT = Path("/mnt/weka/sgrigoryan/Simulation_0_0_6/probe_mountain")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import mountain_bridge

    run_dir = Path(f"/mnt/weka/sgrigoryan/Simulation_0_0_6/_runs/probewide_{os.environ.get('SLURM_JOB_ID','local')}")
    bridge = mountain_bridge.MountainBridge(run_dir, seed=7)
    print(f"booted, render_mode={bridge.current_render_mode()}", flush=True)

    # verified offline: clearance 386m and 459m respectively
    shots = [
        ("wide_along_chain_1", -307.0, -627.0, 420.0, 52.6, 12.0),
        ("wide_along_chain_2", -90.0, -410.0, 480.0, -178.0, 15.0),
    ]
    from PIL import Image

    try:
        for name, x, y, z, yaw, pitch in shots:
            bridge.set_camera_pose(x, y, z, yaw, pitch)
            frame = bridge.process_camera_data()
            Image.fromarray(frame).save(OUT / f"{name}.png")
            print(f"wrote {name}.png", flush=True)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
