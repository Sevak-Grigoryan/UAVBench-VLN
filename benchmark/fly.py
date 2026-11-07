"""FLIGHT of one benchmark episode through the unchanged dataset flight runner (PX4 SITL + Isaac Sim physics, no rendering).
    python fly.py --env Mountain --mission vlnbench_mountain_013 --out <work>/flight --vehicle-id 0
Same arguments as dataset_codebase/common/pipeline/run_x500_mission.py; only the mission id is resolved here."""
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import episode_mission as em

em.install()
script = em.CB / "common/pipeline/run_x500_mission.py"
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
