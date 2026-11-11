"""One common meta.json structure for every published simulation (data-consistency audit 2026-10-21).

normalize(meta, telemetry) runs in finalize.py just before meta.json is written, for EVERY environment:
  * top-level keys in one fixed order (TOP); a key the environment does not produce is written as null (never guessed)
  * every tasks[] record has the same keys (TASK_KEYS): task, landmark, laps, direction, seconds, route, route_xy
    (follow_road / follow_shore / follow_river: landmark = null, route + route_xy = the mission's follow waypoints)
  * every event has the same keys (EVENT_KEYS) incl. frame_start / frame_end; the events partition the video
    (a final "landed" event covers the frames after touchdown when telemetry shows the drone on the ground, disarmed)
  * every task_visibility entry has the same keys (TV_KEYS)
  * coordinate_frame always present (environments without camera extrinsics get the plain conventions)
check(meta) returns a list of structural problems (finalize refuses to publish when it is not empty).
The same rules were applied to the 313 runs published before 2026-10-21 by report/consistency/fix_data.py.
"""

TOP = ['simulation_id', 'difficulty', 'task_count', 'tasks', 'difficulty_rule', 'task_type', 'environment', 'instruction',
       'instruction_steps', 'mission', 'drone', 'autopilot', 'action_format', 'telemetry_format', 'camera_intrinsics', 'frame_size',
       'fps', 'physics_hz', 'frames', 'duration_s', 'actions', 'events', 'verification', 'props_physics_check', 'flight_wall_time_s',
       'simulator', 'render', 'files', 'dataset_version', 'coordinate_frame', 'sky', 'season', 'landscape', 'instructions',
       'instruction_short', 'instructions_source', 'instruction_alignment', 'scene_objects', 'segmentation_classes', 'landmarks',
       'task_visibility', 'data_corrections']
TASK_KEYS = ["task", "landmark", "laps", "direction", "seconds", "route", "route_xy"]
EVENT_KEYS = ["beat", "type", "label", "t_start", "t_end", "frame_start", "frame_end", "pose", "aligned_at", "arrival_error_m",
              "touchdown_error_m", "swept_deg"]
TV_KEYS = ["task", "landmark", "segment", "frames", "visible_fraction", "look_at_quality"]
FOLLOW = {"follow_road": "road", "follow_shore": "shore", "follow_river": "river"}
FEATURE = {"follow_road": "the street followed by the drone", "follow_shore": "the shoreline followed by the drone",
           "follow_river": "the river followed by the drone"}
CF_PLAIN = {"world": "ENU, metres: x = east, y = north, z = up",
            "heading": "yaw_deg: 0 = facing +x (east), counter-clockwise positive (90 = north)",
            "camera_pitch": "pitch_deg positive = looking down (0 = level, 90 = straight down); no roll (stabilised gimbal)",
            "camera_mount": "0.12 m forward of and 0.07 m below the airframe centre, yaw locked to the airframe",
            "timestamps": "frame i (0-based) has t = (i + 1) / fps in telemetry and action.json; events t_start / t_end in seconds, "
                          "frame_start / frame_end = the video frames of the event",
            "extrinsics": None, "landmarks_in_view": None}


def _ordered(d, keys):
    tmp = {k: d.pop(k) for k in list(d)}
    d.update({k: tmp.get(k) for k in keys})
    d.update({k: v for k, v in tmp.items() if k not in keys})


def follow_routes(meta):
    beats = meta.get("mission", {}).get("beats", [])
    for t in meta.get("tasks", []):
        k = FOLLOW.get(t.get("task"))
        if not k or t.get("route_xy"):
            continue
        pts = [[round(b["xy"][0], 2), round(b["xy"][1], 2)] for b in beats if b.get(k) and b.get("xy")]
        pts = [p for i, p in enumerate(pts) if i == 0 or p != pts[i - 1]]
        if len(pts) >= 2:
            t.update(landmark=None, route=FEATURE[t["task"]], route_xy=pts)


def normalize(meta, telemetry=None):
    n, fps = meta["frames"], meta["fps"]
    follow_routes(meta)
    for t in meta.get("tasks", []):
        _ordered(t, TASK_KEYS)
    if meta.get("task_type") is None and len(meta.get("tasks", [])) == 1:
        meta["task_type"] = meta["tasks"][0]["task"]
    ev = meta.get("events", [])
    if ev and telemetry:
        first_after = int(round(ev[-1]["t_end"] * fps))
        if ev[-1].get("type") != "landed" and first_after <= n - 1:
            tail = telemetry[first_after:]
            if tail and all(r["px4"]["landed_state"] == "ON_GROUND" and not r["px4"]["armed"] for r in tail):
                ev.append({"beat": None, "type": "landed", "label": "on the ground, motors stopped", "t_start": ev[-1]["t_end"],
                           "t_end": round(n / fps, 3), "pose": ev[-1].get("pose")})
    for i, e in enumerate(ev):
        if e.get("frame_start") is None or e.get("frame_end") is None:
            e["frame_end"] = n - 1 if i == len(ev) - 1 and abs(e["t_end"] - n / fps) < 0.02 else min(n - 1, int(round(e["t_end"] * fps)) - 1)
            e["frame_start"] = 0 if i == 0 else ev[i - 1]["frame_end"] + 1
        _ordered(e, EVENT_KEYS)
    if isinstance(meta.get("task_visibility"), list):
        for v in meta["task_visibility"]:
            _ordered(v, TV_KEYS)
    if meta.get("coordinate_frame") is None:
        meta["coordinate_frame"] = dict(CF_PLAIN)
    if meta.get("instructions") is None:
        meta["instructions"] = [meta["instruction"]]
        meta["instructions_source"] = "generated only"
    meta.setdefault("data_corrections", [])
    out = {k: meta.get(k) for k in TOP}
    out.update({k: v for k, v in meta.items() if k not in TOP})
    meta.clear()
    meta.update(out)
    return meta


def check(meta):
    P = []
    if list(meta)[:len(TOP)] != TOP:
        P.append("top-level keys / order")
    for t in meta.get("tasks", []):
        if list(t) != TASK_KEYS:
            P.append(f"task keys {list(t)}")
    for e in meta.get("events", []):
        if list(e)[:len(EVENT_KEYS)] != EVENT_KEYS:
            P.append(f"event keys {list(e)}")
            break
    fe = [(e["frame_start"], e["frame_end"]) for e in meta.get("events", [])]
    if fe and (fe[0][0] != 0 or fe[-1][1] != meta["frames"] - 1 or any(b[0] != a[1] + 1 for a, b in zip(fe, fe[1:]))):
        P.append("events do not partition the video frames")
    tv = meta.get("task_visibility")
    if isinstance(tv, list):
        if len(tv) != len(meta.get("tasks", [])):
            P.append("task_visibility: one entry per task expected")
        for v in tv:
            if list(v) != TV_KEYS:
                P.append(f"task_visibility keys {list(v)}")
            fr = v.get("frames")
            if fr and not (0 <= fr[0] <= fr[1] <= meta["frames"] - 1):
                P.append(f"task_visibility frames {fr}")
    if meta.get("actions") is None or abs(meta.get("duration_s", 0) - meta["frames"] / meta["fps"]) > 0.002:
        P.append("duration_s != frames / fps")
    return P
