"""Offline check of all Easy Coastal missions: plan, static checks, obstacle + terrain clearance, offline flight."""
import sys, time
from collections import Counter
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
import coastal_v2 as cv, missions as cm, pilot
ids = [int(a) for a in sys.argv[1:]] or range(1, cm.N_SIMS + 1)
tot, cnt, att = 0.0, Counter(), Counter()
for i in ids:
    t0 = time.time()
    pl, m, L = cm.mission_and_plan(i, cv, pilot)
    mv = cv.movables(pl)
    unstable = sum(not x["static_check"]["stable"] for x in mv)
    tot += m["offline_check"]["duration_s"]; cnt[m["task_type"]] += 1; att[m["attempt"]] += 1
    print(f"{m['id']} att={m['attempt']} {m['task_type']:12s} dur={m['offline_check']['duration_s']:5.1f} sites={len(pl['sites'])} ships={len(pl['ships'])} "
          f"movables={len(mv)} unstable={unstable} start={m['start']}{'(ship)' if m['start_on_ship'] else ''} land={m['land_pad']} ({time.time()-t0:.1f}s)", flush=True)
    print("    ", m["instruction"], flush=True)
print("TOTAL video s", round(tot), "mean", round(tot / len(list(ids)), 1), dict(cnt), "attempts", dict(att))
