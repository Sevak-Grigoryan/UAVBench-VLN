"""
Annotations added to every NEW Coastal sample at publish time (finalize.py calls enrich()), answering the Easy-data review
(dataset_codebase/NEXT_GENERATION_REQUIREMENTS.md). Pure Python on the login node -- nothing is re-flown or re-rendered.

  #1 instructions            canonical instruction + LLM-written paraphrases + a short version (paraphrases_easy.json)
  #2 instruction_alignment   each clause of the canonical instruction -> mission events -> frame range
  #3 target visibility       per frame: is each landmark inside the image (projected box) and how many pixels of it the
                             segmentation shows (occlusion-aware); per task: fraction of task frames the target is seen
  #5 scene_objects           every object of the scene with its segmentation id, class, position and description
  #7 coordinate_frame, per-frame camera extrinsics (OpenCV convention), dataset_version
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODEBASE = HERE.parent
DATASET_VERSION = "x500-vln coastal-easy v2.1 (2026-10-14): varied take-off / landing places, sky / time of day, cruise height and speed"
PARAPHRASES = HERE / "paraphrases_easy.json"

COORDINATE_FRAME = {
    "world": "ENU, metres: x = east, y = north, z = up; sea level is z = 0 (the drawn sea surface swells +-0.3 m around -0.12)",
    "heading": "yaw_deg: 0 = facing +x (east), counter-clockwise positive (90 = north)",
    "camera_pitch": "pitch_deg positive = looking down (0 = level, 90 = straight down); no roll (stabilised gimbal)",
    "camera_mount": "0.12 m forward of and 0.07 m below the airframe centre, yaw locked to the airframe",
    "extrinsics": "telemetry[i].camera_extrinsics: R_wc (3x3, world -> camera) and t_wc, OpenCV camera axes (x right, y down, "
                  "z forward): X_cam = R_wc @ X_world + t_wc; pixel u = fx * X/Z + cx, v = fy * Y/Z + cy with meta.camera_intrinsics",
    "depth": "depth.mkv: 640x360 FFV1 gray16le, value = distance along the optical axis in centimetres, 65535 = sky / beyond 655 m",
    "segmentation": "segmentation.mkv: 640x360 FFV1 gray16le, value = scene_objects id (0 = sky); nearest-neighbour 2x downsample",
    "landmarks_in_view": "telemetry[i].landmarks_in_view[name] (name = a landmark of environment.sites / mission.landmarks): bbox_px [u0, v0, u1, v1] = the landmark's 3-D bounding box projected into the frame and clipped to it (NOT occlusion-aware); centre_px = projection of the box centre (null when it lies behind the camera); mask_pixels = number of VISIBLE pixels of the landmark (all its parts) in the instance segmentation, counted at 640x360 and scaled by 4 to 1280x720 (occlusion-aware). task_visibility counts a frame as visible when mask_pixels > 50.",
}


def _plan_mission(idx):
    sys.path[:0] = [str(CODEBASE / "common/pipeline"), str(HERE), str(HERE / "scene"), str(CODEBASE / "common/vln_simulation")]
    import coastal_v2 as cv
    import missions as cm
    import pilot

    pl, m, L = cm.mission_and_plan(idx, cv, pilot)
    return cv, cm, pl, m


def paraphrases_for(mission_id, canonical):
    if not PARAPHRASES.exists():
        return None
    d = json.loads(PARAPHRASES.read_text()).get(mission_id)
    if not d or d.get("canonical") != canonical:
        return None
    return d


def camera_extrinsics(c):
    yaw, pitch = math.radians(c["yaw_deg"]), math.radians(c["pitch_deg"])
    f = (math.cos(yaw) * math.cos(pitch), math.sin(yaw) * math.cos(pitch), -math.sin(pitch))
    r = (math.sin(yaw), -math.cos(yaw), 0.0)
    d = (f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0])
    R = [list(r), list(d), list(f)]  # rows: camera x, y, z axes in world coordinates -> R_wc
    C = (c["x"], c["y"], c["z"])
    t = [-sum(R[i][j] * C[j] for j in range(3)) for i in range(3)]
    return R, t


def _project(R, t, K, X):
    xc = [sum(R[i][j] * X[j] for j in range(3)) + t[i] for i in range(3)]
    if xc[2] <= 0.1:
        return None
    return (K["fx_px"] * xc[0] / xc[2] + K["cx_px"], K["fy_px"] * xc[1] / xc[2] + K["cy_px"])


def _box_corners(box, base_z):
    cx, cy, yaw, hx, hy, top = box
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for a in (-hx, hx):
        for b in (-hy, hy):
            for z in (base_z, top):
                out.append((cx + c * a - s * b, cy + s * a + c * b, z))
    return out


def _mask_from_seg(seg_path, reg, lms):
    """Per-frame landmark pixel counts from the published segmentation video. A landmark owns its own registry entry and
    every entry named '<landmark>/...' (a shore site = its quay, buildings, cranes and the people/props of that site).
    The video is 640x360 (2x downsampled); counts are scaled by 4 to full-resolution pixels. (The counts the render
    slices wrote matched landmark names against registry names only and stayed empty -- Coastal 2026-10-14.)"""
    import subprocess

    import numpy as np

    if not Path(seg_path).exists():
        return None
    ids = {}
    for k, (name, _, _, _) in enumerate(reg):
        for lm in lms:
            if name == lm or name.startswith(lm + "/"):
                ids.setdefault(lm, []).append(k + 1)
    if not ids:
        return {}
    lut = np.zeros(len(reg) + 2, dtype=np.int32)  # registry id -> landmark index + 1
    names = sorted(ids)
    for j, lm in enumerate(names):
        lut[ids[lm]] = j + 1
    ff = "/mnt/weka/shared-cache/miniforge3/bin/ffmpeg"
    proc = subprocess.Popen([ff, "-loglevel", "error", "-i", str(seg_path), "-f", "rawvideo", "-pix_fmt", "gray16le", "-"],
                            stdout=subprocess.PIPE)
    out, i, fb = {}, 0, 640 * 360 * 2
    while True:
        buf = proc.stdout.read(fb)
        if len(buf) < fb:
            break
        a = np.frombuffer(buf, dtype=np.uint16)
        a = np.minimum(a, len(lut) - 1)
        c = np.bincount(lut[a], minlength=len(names) + 1)
        out[str(i)] = {names[j - 1]: int(c[j]) * 4 for j in range(1, len(names) + 1) if c[j] > 0}
        i += 1
    proc.wait()
    return out


def enrich(meta, telemetry, run, mask_px):
    mid = meta["mission"]["id"]
    idx = int(mid.split("_")[-1])
    cv, cm, pl, m = _plan_mission(idx)
    K = meta["camera_intrinsics"]
    W, H = K["frame_size"]
    # ---- #7 coordinate frame, version
    meta["dataset_version"] = DATASET_VERSION
    meta["coordinate_frame"] = COORDINATE_FRAME
    # ---- #1 instructions
    para = paraphrases_for(mid, meta["instruction"])
    meta["instructions"] = [meta["instruction"]] + (para["paraphrases"] if para else [])
    meta["instruction_short"] = para["short"] if para else None
    meta["instructions_source"] = ("instructions[0]: generated from the mission parameters; instructions[1:] and instruction_short: "
                                   "paraphrases written by an LLM (Claude) from the same parameters, checked automatically for the "
                                   "landmark, pads and numbers -- a human should spot-check a sample") if para else "generated only"
    # ---- #2 alignment: clause -> beats -> events -> frames
    ev = {e["beat"]: e for e in meta["events"]}
    fps = meta["fps"]
    n = len(telemetry)
    align = []
    for cl in m["clauses"]:
        evs = [ev[b] for b in cl["beats"] if b in ev]
        if not evs:
            continue
        t0, t1 = min(e["t_start"] for e in evs), max(e["t_end"] for e in evs)
        align.append({"text": cl["text"], "beats": cl["beats"], "t_start": round(t0, 3), "t_end": round(t1, 3),
                      "frame_start": max(0, int(round(t0 * fps)) - 1), "frame_end": min(n - 1, int(round(t1 * fps)) - 1)})
    if align:
        align[-1]["frame_end"] = n - 1
    meta["instruction_alignment"] = align
    meta["instruction_steps"] = [a["text"] for a in align]
    # ---- #5 scene objects (+ segmentation ids)
    reg = cv.object_registry(pl)
    lms = pl["landmarks"]
    seg_mask = _mask_from_seg(Path(run) / "segmentation.mkv", reg, lms)
    if seg_mask is not None:
        mask_px = seg_mask
    objs = []
    for i, (name, paths, cls, words) in enumerate(reg):
        o = {"id": i + 1, "name": name, "class": cls}
        if name in lms:
            o.update(description=lms[name]["words"], xy=[round(v, 2) for v in lms[name]["xy"]], top_z=round(lms[name]["top_z"], 2),
                     kind=lms[name]["kind"], landmark=True)
        elif words:
            o["description"] = words
        objs.append(o)
    mov = {f"{x['group']}/{x['name']}": x for x in pl["movables"]}
    for o in objs:
        if o["name"] in mov:
            x = mov[o["name"]]
            o.update(xy=[round(x["x"], 2), round(x["y"], 2)], kind=x["kind"], role=x["prop"].get("role"), mass_kg=x["prop"]["mass"])
    meta["scene_objects"] = objs
    meta["segmentation_classes"] = cv.SEG_CLASSES
    # ---- #3 visibility
    boxes = {}
    for name, lm in lms.items():
        if lm.get("box") is None:
            continue
        base = 0.0 if lm["kind"] not in cv.SITE_KINDS else max(cv.terrain_height(lm["xy"][0], lm["xy"][1], pl), 0.0)
        boxes[name] = (_box_corners(lm["box"], base), (lm["xy"][0], lm["xy"][1], (base + lm["top_z"]) / 2))
    for i, f in enumerate(telemetry):
        R, t = camera_extrinsics(f["camera"])
        f["camera_extrinsics"] = {"R_wc": [[round(v, 6) for v in row] for row in R], "t_wc": [round(v, 4) for v in t]}
        vis = {}
        for name, (corners, centre) in boxes.items():
            pts = [p for p in (_project(R, t, K, X) for X in corners) if p is not None]
            if not pts:
                continue
            u0, u1 = max(0.0, min(p[0] for p in pts)), min(W - 1.0, max(p[0] for p in pts))
            v0, v1 = max(0.0, min(p[1] for p in pts)), min(H - 1.0, max(p[1] for p in pts))
            if u0 >= u1 or v0 >= v1:
                continue
            c_ = _project(R, t, K, centre)
            vis[name] = {"bbox_px": [round(u0, 1), round(v0, 1), round(u1, 1), round(v1, 1)],
                         "centre_px": [round(c_[0], 1), round(c_[1], 1)] if c_ else None,
                         "mask_pixels": int(mask_px.get(str(i), {}).get(name, mask_px.get(i, {}).get(name, 0)))}
        f["landmarks_in_view"] = vis
    task_vis = []
    for tk in meta.get("tasks", []):
        name = tk.get("landmark")
        if not name:
            continue
        beat = next((e for e in meta["events"] if e.get("label", "").endswith(name) and e["type"] in ("orbit", "hover", "goto", "land")
                     and (tk["task"] != "pass_over" or e["type"] == "goto")), None)
        if beat is None:
            continue
        a0 = beat.get("aligned_at", beat["t_start"])
        # exact frame indices: int() truncated 21.533*30 = 645.99 to 645, and t_end*fps is the first frame of the NEXT event
        # (data-consistency audit 2026-10-21)
        f0, f1 = int(round(a0 * fps)), min(n - 1, int(round(beat["t_end"] * fps)) - 1)
        frames = telemetry[f0:f1 + 1]
        seen = sum(1 for f in frames if f["landmarks_in_view"].get(name, {}).get("mask_pixels", 0) > 50)
        frac = seen / max(1, len(frames))
        task_vis.append({"task": tk["task"], "landmark": name, "segment": f"the {beat['type']} beat at the target"
                         + (", from the moment the drone faces it" if beat.get("aligned_at") else ""), "frames": [f0, f1],
                         "visible_fraction": round(frac, 3), "look_at_quality": "good" if frac >= 0.9 else "partial" if frac >= 0.5 else "poor"})
    for tk in meta.get("tasks", []):  # a followed shoreline is a route: one entry per task, no fraction
        if tk.get("task") == "follow_shore" and not tk.get("landmark"):
            fe = [e for e in meta["events"] if "shore" in e.get("label", "")]
            if fe:
                task_vis.append({"task": "follow_shore", "landmark": None, "segment": "the follow-the-shoreline beats only",
                                 "frames": [int(round(fe[0]["t_start"] * fps)), min(n - 1, int(round(fe[-1]["t_end"] * fps)) - 1)],
                                 "visible_fraction": None, "look_at_quality": "not applicable (a route, not one object)"})
    meta["task_visibility"] = task_vis
    meta["files"].update({"depth.mkv": COORDINATE_FRAME["depth"], "segmentation.mkv": COORDINATE_FRAME["segmentation"]})
    return meta, telemetry
