"""Checks of one newly published Easy Mountain simulation before it is sent: structure, counts, landing, no pad, pilot spin,
task visibility, events.   python verify_pub.py 10"""
import json, math, sys
sys.path.insert(0, "/mnt/weka/sgrigoryan/dataset_codebase/common/pipeline")
import meta_schema as ms
K = int(sys.argv[1])
D = f"/mnt/weka/sgrigoryan/All_Simulations/Easy/Mountain/Simulation_{K}"
m = json.load(open(D + "/meta.json")); a = json.load(open(D + "/action.json"))["actions"]; T = json.load(open(D + "/telemetry.json"))
P = ms.check(m)
if len(T) != m["frames"] or len(a) != m["actions"]:
    P.append("counts")
v = m["verification"]
if not v["landed_and_disarmed"] or (v["touchdown_error_m"] or 9) > 3:
    P.append(f"landing {v}")
pads = m["environment"]["pads"]
for k in (m["mission"]["start_pad"], m["mission"]["landing_pad"]):
    if not str(pads.get(k, {}).get("kind", "")).startswith("unmarked"):
        P.append(f"uses a painted pad {k}")
spin = 0
beats = m["mission"]["beats"]
for e in m["events"]:
    if e["type"] != "goto" or e.get("beat") is None:
        continue
    b = beats[e["beat"]]
    if "xy" not in b:
        continue
    idx = [k for k in range(e["frame_start"], e["frame_end"] + 1)
           if math.hypot(T[k]["pose"]["x"] - b["xy"][0], T[k]["pose"]["y"] - b["xy"][1]) < max(b.get("radius", 3.0), 4.0)]
    if len(idx) < 2:
        continue
    # over the waypoint the nose must face the look-at point (or hold the heading it arrived with); after 2 s to turn,
    # count the frames more than 40 deg off (the old bug: random / wandering heading while climbing)
    k0 = idx[0]
    for k in idx[60:]:
        if "look_at" in b:
            want = math.degrees(math.atan2(b["look_at"][1] - T[k]["pose"]["y"], b["look_at"][0] - T[k]["pose"]["x"]))
        else:
            want = T[k0]["pose"]["yaw_deg"]
        if abs(((T[k]["pose"]["yaw_deg"] - want) + 180) % 360 - 180) > 40:
            spin += 1
if spin >= 30:
    P.append(f"pilot heading off target over a waypoint: {spin} frames")
print(json.dumps(dict(sim=K, problems=P, task=m["tasks"], task_visibility=m["task_visibility"], duration_s=m["duration_s"],
                      actions=m["actions"], touchdown_m=v["touchdown_error_m"], max_tilt=v["max_tilt_deg"],
                      instruction=m["instruction"], n_instructions=len(m["instructions"]),
                      textured_parts=m["environment"]["build"].get("textured_parts"), bake=m["environment"]["build"].get("bake"),
                      canopy=v.get("canopy_height_max_m")), indent=1))
