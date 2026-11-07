"""Offline check of all Easy City missions: plan, physics static checks, obstacle clearance, offline flight."""
import math, sys, time
from pathlib import Path
from collections import Counter
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
import city_v2 as cv, missions as cm, pilot
ids = [int(a) for a in sys.argv[1:]] or range(1, cm.N_SIMS + 1)
tot, cnt, sigs = 0.0, Counter(), {}
for i in ids:
    t0 = time.time()
    pl, m, L = cm.mission_and_plan(i, cv, pilot)
    mv = cv.movables(pl)
    unstable = [x["name"] for x in mv if not x["static_check"]["stable"]]
    kinds = Counter(f["kind"] for f in pl["features"].values())
    sig = (m["start"], m["beats"][-1]["pad"], m["task_type"], tuple(sorted(L)))
    tot += m["offline_check"]["duration_s"]; cnt[m["task_type"]] += 1
    print(f"{m['id']} att={m['attempt']} {m['task_type']:11s} dur={m['offline_check']['duration_s']:5.1f} buildings={len(pl['buildings'])} "
          f"features={len(pl['features'])} movables={len(mv)} unstable={unstable[:3]} dup={sigs.get(sig)} ({time.time()-t0:.1f}s)")
    sigs[sig] = i
    print("    ", m["instruction"])
print("TOTAL video s", round(tot), "mean", round(tot / len(list(ids)), 1), dict(cnt))
