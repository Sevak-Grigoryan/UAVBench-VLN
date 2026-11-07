"""Fly every Forest mission offline (PX4-like model, no Isaac Sim) and print duration, number of
stick changes, minimum height above ground in transit, touchdown error and the instruction text."""
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
import forest_scene as fs
import missions as fm
import natural_valley as nv
import pilot

fm.apply_pads(fs)
fs.patch_terrain(nv, 5200)
ground = lambda x, y: nv.terrain_height(x, y, 5200)
M, L = fm.missions(nv, 5200, ground)
for k, m in M.items():
    fr, p = pilot.simulate_offline(m["beats"], ground, m["start_xy"], math.radians(m["yaw_deg"]))
    ch = sum(1 for a, b in zip(fr, fr[1:]) if a["sticks"] != b["sticks"]) + 1
    transit = [f["agl"] for f in fr if f["t"] > 8 and not f["beat"].startswith("land")]
    td = [e for e in p.events if e["type"] == "land"]
    print(f"{k:28s} {m['difficulty']:6s} done={p.done} t={fr[-1]['t']:6.1f}s frames={len(fr)} changes={ch} "
          f"minAGL(transit)={min(transit):5.1f} touchdown={td[0].get('touchdown_error_m') if td else None}")
    print("   ", fm.instruction(m, L, ground)[0])
