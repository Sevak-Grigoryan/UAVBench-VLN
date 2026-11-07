"""Plan + fly offline every Easy Mountain mission (no Isaac Sim): which attempt succeeds, task, landscape, duration,
instruction. Writes Mountain/_offline/<idx>.json.   python offline_all.py 1 2 3 ..."""
import json, sys, time, traceback
from pathlib import Path
CB = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CB / "common/pipeline"), str(CB / "Mountain"), str(CB / "Mountain/scene")]
import mountain_v2 as mv
import missions as mm
import pilot
OUT = CB / "Mountain/_offline"
BAKE = "--bake" in sys.argv
for a in [x for x in sys.argv[1:] if not x.startswith("--")]:
    i = int(a)
    t0 = time.time()
    try:
        pl, m, L = mm.mission_and_plan(i, mv, pilot)
        rec = dict(idx=i, ok=True, attempt=m["attempt"], task=m["task_type"], landscape=pl["landscape"], season=pl["season"], sky=pl["sky"]["id"],
                   duration_s=m["offline_check"]["duration_s"], instruction=m["instruction"], clauses=[c["text"] for c in m["clauses"]],
                   start=m["start"], land=m["land_pad"], start_words=pl["pads_all"][m["start"]]["words"], land_words=pl["pads_all"][m["land_pad"]]["words"],
                   sites=[s["kind"] for s in pl["sites"]], n_mov=len(pl["movables"]), n_static=len(pl["statics"]), n_veg=len(pl["vegetation"]),
                   cruise=m.get("cruise"), wall_s=round(time.time() - t0, 1))
        if BAKE:
            import mountain_bake
            t1 = time.time()
            rec["bake"] = str(mountain_bake.bake(pl))
            import mountain_build
            rec["far_texture"] = str(mountain_build.far_texture(pl))
            rec["bake_s"] = round(time.time() - t1, 1)
    except Exception as e:
        rec = dict(idx=i, ok=False, error=repr(e), tb=traceback.format_exc()[-1500:], wall_s=round(time.time() - t0, 1))
    (OUT / f"{i:03d}.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: rec.get(k) for k in ("idx", "ok", "attempt", "task", "landscape", "duration_s", "wall_s", "error")}), flush=True)
