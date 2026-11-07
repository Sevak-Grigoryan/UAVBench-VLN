"""Recompute a published Mountain run's pass landmark entries (telemetry landmarks_in_view[pass]) with the test point 6 m
above the saddle (generator fix 2026-10-24) from the run's own depth video, and its task_visibility. Originals are
backed up to All_Simulations_original_metadata_2026-10-21/... before writing.   python fix_pass_point.py 15"""
import json, math, shutil, sys
from pathlib import Path
import numpy as np
sys.path[:0] = ["/mnt/weka/sgrigoryan/dataset_codebase/Mountain", "/mnt/weka/sgrigoryan/dataset_codebase/Mountain/scene",
                "/mnt/weka/sgrigoryan/dataset_codebase/common/pipeline"]
import enrich
K = int(sys.argv[1])
D = Path(f"/mnt/weka/sgrigoryan/All_Simulations/Easy/Mountain/Simulation_{K}")
RUN = Path(f"/mnt/weka/sgrigoryan/dataset_codebase/Mountain/_runs/easy_mountain_{K:03d}")
B = Path(f"/mnt/weka/sgrigoryan/All_Simulations_original_metadata_2026-10-21/Easy/Mountain/Simulation_{K}")
m = json.loads((D / "meta.json").read_text()); T = json.loads((D / "telemetry.json").read_text())
Kc = m["camera_intrinsics"]; W, H = Kc["frame_size"]; sx, sy = W / enrich.AW, H / enrich.AH
passes = {n: l for n, l in m["landmarks"].items() if l["kind"] == "pass"}
changed = 0
for i, dep in enumerate(enrich._frames(RUN / "depth.mkv", enrich.AW * enrich.AH * 2)):
    if i >= len(T):
        break
    R, t = enrich.camera_extrinsics(T[i]["camera"])
    for name, l in passes.items():
        X = (l["xy"][0], l["xy"][1], l["top_z"] + 6.0)
        xc = enrich._cam(R, t, X)
        liv = T[i]["landmarks_in_view"]
        old = liv.pop(name, None)
        if xc[2] > 1.0:
            u, v = Kc["fx_px"] * xc[0] / xc[2] + Kc["cx_px"], Kc["fy_px"] * xc[1] / xc[2] + Kc["cy_px"]
            if 0 <= u < W and 0 <= v < H:
                d = float(dep[min(enrich.AH - 1, int(v / sy)), min(enrich.AW - 1, int(u / sx))])
                ok = d >= 65535 or d / 100.0 >= xc[2] * 0.97 - 4.0
                liv[name] = {"point_px": [round(u, 1), round(v, 1)], "visible": bool(ok), "distance_m": round(math.sqrt(sum(c * c for c in xc)), 1)}
        changed += int(old != liv.get(name))
tv = m["task_visibility"]
notes = []
for v in tv:
    if v["landmark"] in passes:
        f0, f1 = v["frames"]
        seen = sum(1 for i in range(f0, f1 + 1) if (T[i]["landmarks_in_view"].get(v["landmark"]) or {}).get("visible"))
        frac = round(seen / (f1 - f0 + 1), 3); q = "good" if frac >= 0.9 else "partial" if frac >= 0.5 else "poor"
        notes.append(f"{v['landmark']} vf {v['visible_fraction']} -> {frac}")
        v.update(visible_fraction=frac, look_at_quality=q)
B.mkdir(parents=True, exist_ok=True)
for fn in ("meta.json", "telemetry.json", "action.json"):
    if not (B / fn).exists():
        shutil.copy2(D / fn, B / fn)
m["data_corrections"].append({"date": "2026-10-24", "field": "telemetry landmarks_in_view[pass], task_visibility",
                              "change": f"{changed} frame entries; " + "; ".join(notes),
                              "reason": "a pass is tested at a point 6 m above the saddle (the gap the drone flies through); 1 m above the ground the slope rising to the saddle hid the point in alternate frames although the gap filled the frame. Recomputed from this run's depth video"})
for fn, obj, kw in (("telemetry.json", T, {}), ("meta.json", m, {"indent": 1})):
    tmp = D / (fn + ".tmp"); tmp.write_text(json.dumps(obj, **kw)); json.loads(tmp.read_text()); tmp.replace(D / fn)
print(K, "entries changed", changed, notes)
