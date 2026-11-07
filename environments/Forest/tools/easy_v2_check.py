"""Offline check of all Forest Easy v2 simulations (no Isaac Sim): layouts, physics checks, missions,
uniqueness, offline flight (PX4-like model): duration, min AGL in transit, touchdown."""
import math, sys, time, json
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
import forest_scene as fs, missions as fm, natural_valley as nv, pilot, easy_v2
fm.apply_pads(fs); fs.patch_terrain(nv, 5200)
ground = lambda x, y: nv.terrain_height(x, y, 5200)
ids = range(1, easy_v2.N_SIMS + 1) if len(sys.argv) < 2 else [int(a) for a in sys.argv[1:]]
sigs, tot, rows = {}, 0.0, []
from collections import Counter
cnt = Counter()
for i in ids:
    t0 = time.time()
    lay = easy_v2.layout(i, nv, fs)
    m, L = easy_v2.mission(i, lay, nv, fs, ground)
    sg = easy_v2.signature(m)
    dup = sigs.get(sg); sigs[sg] = i
    kinds = Counter(o["kind"] for g in lay["groups"] for o in g["objects"])
    unstable = [o["name"] for g in lay["groups"] for o in g["objects"] if "static_check" in o and not o["static_check"]["stable"]]
    fr, p = pilot.simulate_offline(m["beats"], ground, m["start_xy"], math.radians(m["yaw_deg"]))
    transit = [f["agl"] for f in fr if f["t"] > 8 and not f["beat"].startswith("land")]
    td = [e for e in p.events if e["type"] == "land"]
    dur = fr[-1]["t"]; tot += dur
    cnt[m["task_type"]] += 1
    print(f"{m['id']} {m['task_type']:12s} groups={[g['type'][0] for g in lay['groups']]} objs={sum(kinds.values())} "
          f"dur={dur:5.1f}s done={p.done} fail={p.failed} minAGL={min(transit):4.1f} td={td[0].get('touchdown_error_m') if td else None} "
          f"dup={dup} unstable={unstable} ({time.time()-t0:.1f}s)")
    print("    ", m["instruction"])
print("TOTAL video s", round(tot), "mean", round(tot / len(list(ids)), 1), dict(cnt))
