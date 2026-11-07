"""RENDER frames [start, end) of a flown benchmark episode through the unchanged dataset slice renderer.
    python render.py --env Mountain --telemetry <work>/flight/telemetry.json --start S --end E --out <work>/slices/slice_S.mp4 \
        --mission vlnbench_mountain_013
Only RGB is rendered (depth / segmentation are switched off: the benchmark publishes video, actions, telemetry, task, goal)."""
import runpy
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import episode_mission as em

em.install()
import envs  # noqa: E402  (dataset_codebase/common/pipeline, on sys.path after install())

_boot = envs.boot


def boot(*a, **k):
    env = _boot(*a, **k)
    env.aux = False
    return env


envs.boot = boot
script = em.CB / "common/pipeline/render_slice.py"
sys.argv = [str(script)] + sys.argv[1:]
runpy.run_path(str(script), run_name="__main__")
