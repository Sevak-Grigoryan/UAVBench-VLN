"""Check every Easy v2 mission (offline flight + layout) against the tall static obstacles of the valley
(scene/obstacles.json: valley-wall cliffs 26-45 m, cliff gate 14 m). Flight 036 orbited into a cliff."""
import json, math, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
import forest_scene as fs, missions as fm, natural_valley as nv, pilot, easy_v2
fm.apply_pads(fs); fs.patch_terrain(nv, 5200)
ground = lambda x, y: nv.terrain_height(x, y, 5200)
import numpy as np
from scipy.spatial import cKDTree

# surface points of the valley-wall cliffs and the cliff gate (meshes as rendered; 300k-point sample)
PTS = np.load(HERE / "scene/obstacle_points.npy")
TREE = cKDTree(PTS)
OBJ_CLEAR, FLIGHT_CLEAR = 1.5, 6.0


def violations(idx, attempt=None):
    lay = easy_v2.layout(idx, nv, fs, attempt=attempt)
    m, L = easy_v2.mission(idx, lay, nv, fs, ground, attempt=attempt)
    bad = []
    for g in lay["groups"]:
        for o in g["objects"]:
            if o.get("world"):
                continue
            x, y = o["xy"]
            g0 = fs.visible_ground(nv, x, y, 5200)
            for z in np.arange(g0 - 0.5, o["top_z"] + 1.0, 1.0):
                d, _ = TREE.query([x, y, z])
                if d < o["r"] + OBJ_CLEAR:
                    bad.append(("object", o["name"], round(float(d), 1)))
                    break
    fr, p = pilot.simulate_offline(m["beats"], ground, m["start_xy"], math.radians(m["yaw_deg"]))
    P = np.array([[f["x"], f["y"], f["z"]] for f in fr[::3]])
    d, _ = TREE.query(P)
    k = np.nonzero(d < FLIGHT_CLEAR)[0]
    if len(k):
        f = fr[3 * int(k[0])]
        bad.append(("flight", f["beat"], round(f["t"], 1), round(float(d.min()), 1)))
    return bad, p.done and not p.failed


if __name__ == "__main__":
    ids = [int(a) for a in sys.argv[1:]] or range(1, easy_v2.N_SIMS + 1)
    for i in ids:
        bad, ok = violations(i)
        if bad or not ok:
            print(i, ok, bad[:3], len(bad))
