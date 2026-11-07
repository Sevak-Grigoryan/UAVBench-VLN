"""Cheap look at the real mountain scene, same purpose as Simulation_0_0_5's probe_city_view.py:
answer specific visual questions in ~5 minutes instead of committing to the ~53 minute full mission
render.

Viewpoints, each chosen to check one thing:
  summit_close    the lookout platform itself -- deck/railing/roof geometry, wood colour
  summit_wide     the peak from orbit distance/altitude -- does the target read as a structure
  treeline        mid-slope -- pine forest density, dirt/scree texture, tree placement
  snowline        near the snow overlay boundary -- does the band read, or is there a seam
  ridge_distant   looking across the mountain toward the horizon -- does the far terrain fade
                  naturally or end at a hard edge (the exact failure mode fixed in the city)
  cruise          the mission's own actual framing (orbit altitude, orbit radius)
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)

OUT = Path("/mnt/weka/sgrigoryan/Simulation_0_0_6/probe_mountain")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import mountain_bridge
    import mountain_scene

    run_dir = Path(f"/mnt/weka/sgrigoryan/Simulation_0_0_6/_runs/probe_{os.environ.get('SLURM_JOB_ID','local')}")
    bridge = mountain_bridge.MountainBridge(run_dir, seed=7)
    print(f"booted, render_mode={bridge.current_render_mode()}", flush=True)

    px, py = mountain_scene.PEAK_XY
    info = bridge.mountain_info
    deck_z = info["summit_deck_z"]

    # Every camera position is checked against REAL terrain height before use, not placed by a
    # naive offset formula -- the first version of this probe buried two cameras inside the
    # terrain (summit_close -7.9m clearance, cruise -2.2m) because the ridged noise right around
    # the peak rises well above a flat "target_z + margin" guess. This is the same class of bug
    # validate_orbit_clearance() now checks for the actual mission before every render.
    def clear(x, y, z, min_clear=8.0):
        h = mountain_scene.terrain_height(x, y, seed=7)
        return z if z - h >= min_clear else h + min_clear

    FLIGHT_ALT = deck_z + 26.0  # matches run_mountain_mission.py's validated altitude
    shots = [
        ("summit_close", px - 14.0, py - 14.0, clear(px - 14.0, py - 14.0, deck_z + 16.0), 45.0, 30.0),
        ("summit_wide", px - 24.0, py, clear(px - 24.0, py, FLIGHT_ALT), 90.0, 25.0),
        ("treeline", px - 190.0, py - 40.0, clear(px - 190.0, py - 40.0, 130.0), 60.0, 15.0),
        ("snowline", px + 20.0, py + 110.0, clear(px + 20.0, py + 110.0, 180.0), 200.0, 25.0),
        ("ridge_distant", px, py - 20.0, clear(px, py - 20.0, deck_z + 45.0), 180.0, 8.0),
        ("cruise", px - 24.0 * 0.7, py - 24.0 * 0.7, clear(px - 24.0 * 0.7, py - 24.0 * 0.7, FLIGHT_ALT), 45.0, 30.0),
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
