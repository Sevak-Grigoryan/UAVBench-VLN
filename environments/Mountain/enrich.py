"""
Annotations added to every Easy Mountain sample at publish time (finalize.py calls enrich()). Pure Python on the login
node -- nothing is re-flown or re-rendered. Built to the Coastal review of 2026-10-20 (NEXT_GENERATION_REQUIREMENTS.md
C1-C12):

  C1/C2  task_visibility is measured over the TASK SEGMENT only: the orbit (circle), the hover after the drone faces the
         target (inspect), the stretch of the pass-over where the drone is actually over / next to the target (not the
         approach), the follow beats (river / ridge / cable car / lake), the pass crossing, the final approach + landing.
  C3     instruction_alignment: every clause owns its own beats; frame ranges are contiguous and never overlap
         (checked here; finalize refuses the sample otherwise).
  C4     landmarks_in_view[*].bbox_px is the box of the VISIBLE pixels of the object in the instance segmentation
         (occlusion-aware, full-resolution pixel coordinates); terrain landmarks (summits, passes, lakes) are a point
         tested against the depth buffer (visible / occluded).
  C8     scene_objects: every object has a description (colours, role, where it is).
  C9     meta.sky is always written.
  C10    events cover the whole video: a final "landed" event runs from touchdown to the last frame.
  C11    only trajectory.mp4, action.json, telemetry.json, meta.json are published (segmentation / depth stay internal).
  C12    no train / test split file.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODEBASE = HERE.parent
DATASET_VERSION = "x500-vln mountain-easy v1.0 (2026-10-20): per-simulation landscape type, season, sky, sites and missions"
PARAPHRASES = HERE / "paraphrases_easy.json"
PARAPHRASES_BENCH = HERE / "paraphrases_bench.json"  # benchmark episodes (bench_mountain_<k>)
FF = "/mnt/weka/shared-cache/miniforge3/bin/ffmpeg"
AW, AH = 640, 360

COORDINATE_FRAME = {
    "world": "ENU, metres: x = east, y = north, z = up; z = 0 is the valley floor at the middle of the map (the floor tilts gently "
             "along the valley; summits reach 150-500 m)",
    "heading": "yaw_deg: 0 = facing +x (east), counter-clockwise positive (90 = north)",
    "camera_pitch": "pitch_deg positive = looking down (0 = level, 90 = straight down); no roll (stabilised gimbal)",
    "camera_mount": "0.12 m forward of and 0.07 m below the airframe centre, yaw locked to the airframe",
    "extrinsics": "telemetry[i].camera_extrinsics: R_wc (3x3, world -> camera) and t_wc, OpenCV camera axes (x right, y down, "
                  "z forward): X_cam = R_wc @ X_world + t_wc; pixel u = fx * X/Z + cx, v = fy * Y/Z + cy with meta.camera_intrinsics",
    "landmarks_in_view": "telemetry[i].landmarks_in_view[name]: objects -> bbox_px [u0, v0, u1, v1] of the VISIBLE pixels (occlusion-aware, "
                         "from the renderer's instance segmentation), visible_pixels, centroid_px; terrain landmarks (summits, passes, "
                         "lakes) -> point_px of the landmark point and visible = not hidden behind nearer terrain (depth test)",
}


def _plan_mission(idx):
    sys.path[:0] = [str(CODEBASE / "common/pipeline"), str(HERE), str(HERE / "scene"), str(CODEBASE / "common/vln_simulation")]
    import missions as mm
    import mountain_v2 as mv
    import pilot

    pl, m, L = mm.mission_and_plan(idx, mv, pilot)
    return mv, mm, pl, m


def paraphrases_for(mission_id, canonical):
    src = PARAPHRASES_BENCH if mission_id.startswith("bench_") else PARAPHRASES
    if not src.exists():
        return None
    d = json.loads(src.read_text()).get(mission_id)
    if not d or d.get("canonical") != canonical:
        return None
    return d


def camera_extrinsics(c):
    yaw, pitch = math.radians(c["yaw_deg"]), math.radians(c["pitch_deg"])
    f = (math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch))
    r = (math.sin(yaw), -math.cos(yaw), 0.0)
    d = (f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0])
    R = [list(r), list(d), list(f)]
    C = (c["x"], c["y"], c["z"])
    t = [-sum(R[i][j] * C[j] for j in range(3)) for i in range(3)]
    return R, t


def _cam(R, t, X):
    return [sum(R[i][j] * X[j] for j in range(3)) + t[i] for i in range(3)]


def _frames(path, n_bytes):
    proc = subprocess.Popen([FF, "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray16le", "-"], stdout=subprocess.PIPE)
    while True:
        buf = proc.stdout.read(n_bytes)
        if len(buf) < n_bytes:
            break
        yield np.frombuffer(buf, dtype=np.uint16).reshape(AH, AW)
    proc.wait()


def visibility(run, telemetry, K, reg, landmarks):
    """Per frame: objects -> visible bbox / pixels from segmentation; terrain landmarks -> point + depth test."""
    W, H = K["frame_size"]
    sx, sy = W / AW, H / AH
    obj_lms = {n: l for n, l in landmarks.items() if l["kind"] not in ("summit", "pass", "lake", "tarn")}
    pt_lms = {n: l for n, l in landmarks.items() if n not in obj_lms}
    ids = {}
    for k, (name, _, _, _) in enumerate(reg):
        for lm in obj_lms:
            if name == lm or name.startswith(lm + "/"):
                ids.setdefault(lm, []).append(k + 1)
    luts = {}
    for lm, lst in ids.items():
        lut = np.zeros(len(reg) + 2, dtype=bool)
        lut[lst] = True
        luts[lm] = lut
    seg_p, dep_p = Path(run) / "segmentation.mkv", Path(run) / "depth.mkv"
    segs = _frames(seg_p, AW * AH * 2)
    deps = _frames(dep_p, AW * AH * 2) if dep_p.exists() else None
    out = []
    for i, f in enumerate(telemetry):
        seg = next(segs, None)
        dep = next(deps, None) if deps is not None else None
        R, t = camera_extrinsics(f["camera"])
        vis = {}
        if seg is not None:
            seg = np.minimum(seg, len(reg) + 1)
            present = np.bincount(seg.ravel(), minlength=len(reg) + 2) > 0
            for lm, lut in luts.items():
                if not present[lut].any():
                    continue
                m = lut[seg]
                cnt = int(m.sum())
                if cnt < 2:
                    continue
                rows = np.flatnonzero(m.any(axis=1))
                cols = np.flatnonzero(m.any(axis=0))
                vv, uu = np.nonzero(m)
                vis[lm] = {"bbox_px": [round(cols[0] * sx, 1), round(rows[0] * sy, 1), round((cols[-1] + 1) * sx - 1, 1), round((rows[-1] + 1) * sy - 1, 1)],
                           "visible_pixels": int(cnt * sx * sy), "centroid_px": [round((uu.mean() + 0.5) * sx, 1), round((vv.mean() + 0.5) * sy, 1)]}
        for lm, l in pt_lms.items():
            # a pass is a gap: test a point 6 m above the saddle (the air the drone flies through); 1 m above the ground the
            # slope rising to the saddle hid it in alternate frames although the gap filled the frame (Mountain S015, 2026-10-24)
            X = (l["xy"][0], l["xy"][1], l["top_z"] + (6.0 if l.get("kind") == "pass" else 1.0))
            xc = _cam(R, t, X)
            if xc[2] <= 1.0:
                continue
            u, v = K["fx_px"] * xc[0] / xc[2] + K["cx_px"], K["fy_px"] * xc[1] / xc[2] + K["cy_px"]
            if not (0 <= u < W and 0 <= v < H):
                continue
            ok = True
            if dep is not None:
                d = float(dep[min(AH - 1, int(v / sy)), min(AW - 1, int(u / sx))])
                ok = d >= 65535 or d / 100.0 >= xc[2] * 0.97 - 4.0
            vis[lm] = {"point_px": [round(u, 1), round(v, 1)], "visible": bool(ok), "distance_m": round(math.sqrt(sum(c * c for c in xc)), 1)}
        out.append(vis)
    return out


def _seen(vis, name, lm_kind):
    e = vis.get(name)
    if not e:
        return False
    if "visible_pixels" in e:
        return e["visible_pixels"] >= 60
    return e.get("visible", False)


def enrich(meta, telemetry, run, mask_px=None):
    mid = meta["mission"]["id"]
    idx = int(mid.split("_")[-1]) + (1000 if mid.startswith("bench_") else 0)  # bench_mountain_<k> = idx 1000 + k (missions.BENCH_BASE)
    mv, mm, pl, m = _plan_mission(idx)
    K = meta["camera_intrinsics"]
    fps = meta["fps"]
    n = len(telemetry)
    meta["dataset_version"] = DATASET_VERSION
    meta["coordinate_frame"] = COORDINATE_FRAME
    # ---- C9 sky (always)
    b = meta["environment"].get("build", {})
    meta["sky"] = {"hdri": b.get("hdri"), "description": b.get("sky"), "sun_elevation_deg": b.get("sun_elevation_deg"),
                   "sun_azimuth_deg": b.get("sun_azimuth_deg"), "sun_color": b.get("sun_color")}
    if not meta["sky"]["hdri"]:
        raise SystemExit("no sky in the build info")
    meta["season"] = pl["season"]
    meta["landscape"] = pl["landscape"]
    # ---- instructions
    para = paraphrases_for(mid, meta["instruction"])
    meta["instructions"] = [meta["instruction"]] + (para["paraphrases"] if para else [])
    meta["instruction_short"] = para["short"] if para else None
    meta["instructions_source"] = ("instructions[0]: generated from the mission parameters; instructions[1:] and instruction_short: "
                                   "paraphrases written by an LLM (Claude) from the same parameters, checked automatically: same landmark, "
                                   "places and numbers, similarity to the original < 0.8, no duplicates across simulations") if para else "generated only"
    # ---- C10 events cover the whole video
    ev = list(meta["events"])
    last_t = n / fps  # the time of the last frame (frame i has t = (i + 1) / fps); (n - 1) / fps ended one frame early (audit 2026-10-21)
    if ev and ev[-1]["t_end"] < last_t:
        ev.append({"beat": None, "type": "landed", "label": "on the ground, motors stopped", "t_start": ev[-1]["t_end"], "t_end": round(last_t, 3),
                   "pose": ev[-1].get("pose")})
    if ev and ev[0]["t_start"] > 0:
        ev[0]["t_start"] = 0.0
    for e in ev:
        e["frame_start"] = int(round(e["t_start"] * fps))
        e["frame_end"] = min(n - 1, int(round(e["t_end"] * fps)) - 1) if e is not ev[-1] else n - 1
    for a, b2 in zip(ev, ev[1:]):
        b2["frame_start"] = a["frame_end"] + 1
    meta["events"] = ev
    # ---- C3 alignment: clause -> its own beats -> contiguous, non-overlapping frame ranges
    evb = {e["beat"]: e for e in ev if e["beat"] is not None}
    align = []
    for cl in m["clauses"]:
        es = [evb[b] for b in cl["beats"] if b in evb]
        if not es:
            continue
        align.append({"text": cl["text"], "beats": cl["beats"], "frame_start": min(e["frame_start"] for e in es),
                      "frame_end": max(e["frame_end"] for e in es)})
    if align:
        align[0]["frame_start"] = 0
        align[-1]["frame_end"] = n - 1
        for a, b2 in zip(align, align[1:]):
            if b2["frame_start"] <= a["frame_end"]:
                raise SystemExit(f"alignment overlap: '{a['text']}' / '{b2['text']}'")
            a["frame_end"] = b2["frame_start"] - 1  # contiguous (no gap between clauses)
        for a in align:
            a["t_start"], a["t_end"] = round(a["frame_start"] / fps, 3), round((a["frame_end"] + 1) / fps, 3)
    meta["instruction_alignment"] = align
    meta["instruction_steps"] = [a["text"] for a in align]
    # ---- C8 scene objects
    reg = mv.object_registry(pl)
    lms = pl["landmarks"]
    stat = {st["name"]: st for st in pl["statics"]}
    mov = {f"{x['group']}/{x['name']}": x for x in pl["movables"]}
    objs = []
    for i, (name, paths, cls, words) in enumerate(reg):
        o = {"id": i + 1, "name": name, "class": cls, "description": words}
        if name in stat:
            o["xy"] = [round(v, 2) for v in stat[name]["xy"]]
            o["site"] = name.split("/")[0]
        if name in mov:
            x = mov[name]
            o.update(xy=[round(x["x"], 2), round(x["y"], 2)], kind=x["kind"], mass_kg=x["prop"]["mass"], site=x["group"])
        if name.startswith("pad_"):
            p = pl["pads"][name[4:]]
            o.update(xy=[round(v, 2) for v in p["xy"]])
        objs.append(o)
    if any(not o["description"] for o in objs):
        raise SystemExit("scene object without a description")
    meta["scene_objects"] = objs
    meta["segmentation_classes"] = mv.SEG_CLASSES
    meta["landmarks"] = {k: {"description": l["words"], "kind": l["kind"], "xy": [round(v, 2) for v in l["xy"]], "top_z": round(l["top_z"], 2)}
                         for k, l in lms.items()}
    # ---- C4 visibility per frame
    vis = visibility(run, telemetry, K, reg, {**lms, "water": {"kind": "water", "xy": (0.0, 0.0), "top_z": 0.0, "words": "the river and lakes"}})
    for f, vv in zip(telemetry, vis):
        R, t = camera_extrinsics(f["camera"])
        f["camera_extrinsics"] = {"R_wc": [[round(v, 6) for v in row] for row in R], "t_wc": [round(v, 4) for v in t]}
        f["landmarks_in_view"] = vv
    # ---- C1 / C2 task visibility over the task segment only
    task_vis = []
    for tk in meta.get("tasks", []):
        seg = _task_segment(tk, meta, m, pl, telemetry, fps)
        if seg is None:
            continue
        name, f0, f1, how = seg
        frames = range(f0, f1 + 1)
        # the task record names the landmark its visibility is scored on (Sim 7: tasks[0] had none, the score used another peak)
        if name in lms and not tk.get("landmark"):
            tk["landmark"] = name
            ml = meta["mission"].setdefault("landmarks", [])
            if not any(x["name"] == name for x in ml):
                L = lms[name]
                best = min(math.hypot(r["pose"]["x"] - L["xy"][0], r["pose"]["y"] - L["xy"][1]) for r in telemetry)
                ml.append({"name": name, "description": L["words"], "position_xy": [round(v, 2) for v in L["xy"]],
                           "top_z": round(L["top_z"], 3), "closest_horizontal_approach_m": round(best, 2)})
        kind = lms[name]["kind"] if name in lms else "object"
        if tk["task"] == "follow_ridge":
            frac = crest_ahead_fraction(meta, telemetry, K, pl, f0, f1)
            task_vis.append({"task": tk["task"], "landmark": name, "segment": how + "; scored on the crest line ahead (30-200 m "
                             "along the remaining crest) being inside the frame", "frames": [f0, f1], "visible_fraction": round(frac, 3),
                             "look_at_quality": "good" if frac >= 0.9 else "partial" if frac >= 0.5 else "poor"})
            continue
        if name == "river" or tk["task"] == "cross_lake":
            # water-surface tasks are judged on visible water pixels: one point at the lake centre left the frame while the
            # water filled it (Mountain S050: 45 % on the point, 99.5 % of frames show the lake, 2026-10-29)
            seen = sum(1 for i in frames if vis[i].get("water", {}).get("visible_pixels", 0) >= 400)
            frac = seen / max(1, len(frames))
            task_vis.append({"task": tk["task"], "landmark": name, "segment": how + ("; scored on visible water pixels (>= 400)" if name != "river" else ""), "frames": [f0, f1], "visible_fraction": round(frac, 3),
                             "look_at_quality": "good" if frac >= 0.9 else "partial" if frac >= 0.5 else "poor"})
        elif name in lms:
            seen = sum(1 for i in frames if _seen(vis[i], name, kind))
            frac = seen / max(1, len(frames))
            task_vis.append({"task": tk["task"], "landmark": name, "segment": how, "frames": [f0, f1], "visible_fraction": round(frac, 3),
                             "look_at_quality": "good" if frac >= 0.9 else "partial" if frac >= 0.5 else "poor"})
        else:
            task_vis.append({"task": tk["task"], "landmark": name, "segment": how, "frames": [f0, f1], "visible_fraction": None,
                             "look_at_quality": "not applicable (the goal is an unmarked place on the ground, not one object)"
                             if tk["task"] == "land_at" else "not applicable (a route, not one object)"})
    meta["task_visibility"] = task_vis
    # ---- verification extras, files
    # the flight job builds the scene without the ground-texture bake; the render used it -- record the bake the video shows
    try:
        import mountain_bake as mb_
        bd = mb_.tile_dir(pl)
        if (bd / "DONE").exists():
            meta["environment"]["build"]["bake"] = f"{mb_.VERSION}: {mb_.NT}x{mb_.NT} tiles at {mb_.RES} px ({bd.name})"
    except Exception:
        pass
    meta["verification"]["canopy_height_max_m"] = round(max((mn_h for mn_h in _tree_heights(pl)), default=0.0), 1)
    meta["files"] = {"trajectory.mp4": "onboard gimbal camera video, 30 fps", "action.json": "stick actions, one entry per change",
                     "telemetry.json": "per-frame pose, velocity, camera pose and extrinsics, PX4 state, landmarks in view",
                     "meta.json": "this file"}
    return meta, telemetry


def _tree_heights(pl):
    import mountain_nature as mn
    for k, x, y, z, yaw, sc in pl["vegetation"]:
        if mn.PROTOS[k][4] in ("tree", "young", "broadleaf"):
            yield mn.PROTOS[k][3] * sc


def crest_ahead_fraction(meta, telemetry, K, pl, f0, f1):
    """follow_ridge: share of the crest-flight frames in which the crest AHEAD is in the frame -- the remaining crest line
    (the drone's position, then the crest waypoints still to fly), sampled every 10 m from 30 m to 200 m ahead at ground
    level + 2 m; a frame counts when any sample projects inside it. The end summit is the wrong yardstick: it rises 25-30 deg
    above a drone flying 40 m over the crest, above the frame of a camera that cannot look up (Mountain S032: 15 % on the
    end summit, 2026-10-23); a single near waypoint is under the frame's bottom edge."""
    import mountain_v2 as mv
    beats = meta["mission"]["beats"]
    W, H = K["frame_size"]
    fev = [e for e in meta["events"] if e.get("beat") is not None and beats[e["beat"]].get("task") == "follow_ridge"
           and beats[e["beat"]].get("label") == "follow the ridge"]
    if not fev:
        return 0.0
    seen = n = 0
    for i in range(f0, f1 + 1):
        k = next((j for j, e in enumerate(fev) if e["frame_start"] <= i <= e["frame_end"]), None)
        if k is None:
            continue
        r = telemetry[i]
        poly = [(r["pose"]["x"], r["pose"]["y"])] + [tuple(beats[e["beat"]]["xy"]) for e in fev[k:]]
        R, t = camera_extrinsics(r["camera"])
        n += 1
        hit = False
        for dist in range(30, 201, 10):
            rem = float(dist)
            pt = None
            for (ax, ay), (bx, by) in zip(poly[:-1], poly[1:]):
                L = math.hypot(bx - ax, by - ay)
                if L >= rem:
                    pt = (ax + (bx - ax) * rem / L, ay + (by - ay) * rem / L)
                    break
                rem -= L
            if pt is None:
                break
            xc = _cam(R, t, (pt[0], pt[1], mv.terrain_height(pt[0], pt[1], pl) + 2.0))
            if xc[2] > 1.0:
                u, v = K["fx_px"] * xc[0] / xc[2] + K["cx_px"], K["fy_px"] * xc[1] / xc[2] + K["cy_px"]
                if 0 <= u < W and 0 <= v < H:
                    hit = True
                    break
        seen += int(hit)
    return seen / max(1, n)


def _task_segment(tk, meta, m, pl, telemetry, fps):
    """(landmark, first frame, last frame, definition) of the part of the flight that IS the task."""
    ev = [e for e in meta["events"] if e.get("beat") is not None]
    beats = m["beats"]
    task = tk["task"]
    tb = [e for e in ev if beats[e["beat"]].get("task") == task]
    if not tb:
        return None
    name = tk.get("landmark")
    if task == "circle":
        e = tb[0]
        return name, e["frame_start"], e["frame_end"], "the orbit (one full circle) only"
    if task == "cross_pass":
        # judged on the climb to the pass, while the camera is aimed at the saddle: during the crossing itself the saddle is
        # under the drone, out of the forward camera's view (Mountain S023: 97 % on the climb, 2 % while crossing, 2026-10-22)
        climb = next((x for x in ev if x["beat"] == tb[0]["beat"] - 1 and beats[x["beat"]].get("label") == "climb toward the pass"), None)
        if climb is not None:
            return name, climb["frame_start"], climb["frame_end"], "the climb to the pass (camera aimed at the saddle)"
    if task == "inspect":
        e = tb[0]
        return name, int(round(e.get("aligned_at", e["t_start"]) * fps)), e["frame_end"], "the hover, from the moment the drone faces the target"
    if task == "pass_over":
        e = tb[0]
        b = beats[e["beat"]]
        tx, ty = b["over_xy"]
        r = max(18.0, pl["landmarks"][name].get("size_r", 15.0) * 0.8)
        idx = [i for i in range(e["frame_start"], e["frame_end"] + 1)
               if math.hypot(telemetry[i]["pose"]["x"] - tx, telemetry[i]["pose"]["y"] - ty) <= r]
        if not idx:
            return None
        return name, idx[0], idx[-1], f"only the frames while the drone is within {r:.0f} m (horizontally) of the target, i.e. over it"
    if task == "land_at":
        e = tb[0]
        goto = next(x for x in ev if x["beat"] == e["beat"] - 2)
        # the camera looks forward (5 deg down) while descending and landing (user rule), so the site is judged on the final
        # approach, while the camera is aimed at it
        return name, goto["frame_start"] + (goto["frame_end"] - goto["frame_start"]) // 2, goto["frame_end"], "the final approach to the site (camera aimed at it)"
    f0, f1 = min(x["frame_start"] for x in tb), max(x["frame_end"] for x in tb)
    what = {"follow_valley": "river", "follow_ridge": None, "follow_cable_car": tk.get("landmark"), "cross_lake": tk.get("landmark"),
            "cross_pass": tk.get("landmark")}.get(task)
    if task == "cross_lake":
        what = "lake" if "lake" in pl["landmarks"] else "tarn_1"
    if task == "follow_ridge":
        # the end of the crest flight (ahead of the camera); the start summit is behind it the whole time (Sim 7 scored 0 %)
        what = next((b.get("landmark") for b in beats if b.get("task") == "follow_ridge" and b.get("landmark")), None)
        if what is None:
            ends = [c for c in pl["summits"] + pl["cols"]]
            last = [b for b in beats if b.get("task") == "follow_ridge"][-1]["xy"]
            what = min(ends, key=lambda c: math.hypot(c["x"] - last[0], c["y"] - last[1]))["name"]
    if what is None:
        what = next((b.get("landmark") for b in beats if b.get("label") == "climb to the ridge"), None) or "ridge"
    return what, f0, f1, {"follow_valley": "the follow-the-river beats only", "follow_ridge": "the follow-the-crest beats only",
                          "follow_cable_car": "the climb beside the cable line only", "cross_lake": "the crossing over the water only",
                          "cross_pass": "the crossing of the pass only"}[task]
