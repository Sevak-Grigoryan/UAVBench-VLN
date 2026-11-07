"""Cheap look at the real city scene: is the far-belt visible, and do the new tinted facades read?

Builds the actual city via CityBridge (same code path as the mission) but renders only a handful
of deliberately chosen viewpoints instead of 2736 trajectory frames -- ~7 minutes instead of ~35.

Viewpoints are chosen to answer specific questions rather than to look nice:
  edge_*   high and pitched down toward the city boundary, where /World/Far must fill the gap
           between the detailed grid and the HDRI horizon
  street_* low over a street, where facade texture and prop density have to hold up close
  cruise   the mission's actual altitude and framing, as a sanity check
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)

OUT = Path("/mnt/weka/sgrigoryan/Simulation_0_0_5/probe_city")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import city_bridge
    import city_scene

    run_dir = Path(f"/mnt/weka/sgrigoryan/Simulation_0_0_5/_runs/probecity_{os.environ.get('SLURM_JOB_ID','local')}")
    bridge = city_bridge.CityBridge(run_dir, seed=7)
    print(f"booted, render_mode={bridge.current_render_mode()}", flush=True)

    # how far does the geometry actually reach? printed so the image can be checked against it
    E = city_scene.CITY_HALF_EXTENT_M
    print(f"detailed city half-extent {E}m; far belt should continue out to ~900m "
          f"over a {city_scene.build_surroundings.__doc__.splitlines()[0]}", flush=True)

    shots = [
        # looking outward across the city edge from high up: the far belt occupies the band
        # between the detailed grid and the horizon, so it is unmissable if it renders.
        ("edge_n", 0.0, -120.0, 150.0, 90.0, 18.0),
        ("edge_ne", -150.0, -150.0, 150.0, 45.0, 18.0),
        ("edge_far", 0.0, 0.0, 220.0, 90.0, 12.0),
        # close over a street: facade texture, tint variety, prop density
        ("street_a", -60.0, 0.0, 22.0, 90.0, 8.0),
        ("street_b", 31.0, -40.0, 30.0, 90.0, 14.0),
        # the mission's own framing
        ("cruise", -20.0, -20.0, 52.0, 45.0, 35.0),
    ]
    from PIL import Image

    try:
        for name, x, y, z, yaw, pitch in shots:
            bridge.set_camera_pose(x, y, z, yaw, pitch)
            frame = bridge.process_camera_data()
            Image.fromarray(frame).save(OUT / f"{name}.png")
            print(f"wrote {name}.png  (pos {x:.0f},{y:.0f},{z:.0f} yaw {yaw:.0f} pitch {pitch:.0f})",
                  flush=True)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
