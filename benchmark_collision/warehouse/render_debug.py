"""
OPT-IN external-camera DEBUG video of a flown mission (never part of a benchmark episode; written to the work dir only).

    python render_debug.py --env Forest --scene easy_v2_037 --flight <work>/flight --out <work>/debug

The dataset slice renderer draws only the stabilised onboard camera and has no X500 in the scene. Here, per telemetry frame:
  * the X500 (dataset_codebase/common/assets/x500/x500.usd, physics schemas removed: a visual copy) is placed at the recorded
    position with the FULL recorded attitude quaternion (roll, pitch, yaw);
  * EXTERNAL view: one camera at a fixed world position, CAM_BACK_M behind the drone's starting hover position (opposite the
    episode heading) and above it, aimed at that hover position, with a longer lens (DEBUG_FOCAL_MM) so a short sideways
    manoeuvre fills the frame;
  * ONBOARD view: the same stabilised camera pose the dataset renders (the X500 is hidden for it, as in the dataset);
  * overlay: sim time, phase / beat, sticks, roll / pitch / yaw from telemetry, and a 4x inset around the drone.
Bank check: at sample frames the external view is rendered with and without the X500; the image-space angle of the drone's
silhouette (principal axis of the difference mask) is compared with the angle predicted by projecting the recorded attitude.

Collision-test videos (--camera side): instead of the fixed camera, the external camera FOLLOWS the drone at a constant
world-frame offset (constant orientation, so the attitude seen is the real one), from the side of the motion; the offset is
picked from candidates around the drone so the line of sight is not blocked by tree / rock proxies (proxies.json). The test
fixtures (wall, ceiling, pole: collision.json test_notes.fixtures) are re-created as visual-only orange shapes, and the first
prohibited contact (collision.json) is marked: red banner + contact point, last frame held. --max-seconds keeps only the last
part of the episode, --stride renders every n-th frame.

Outputs: <name>.mp4 (side by side), bank_check.json, bank_check_frames/ (mask PNGs) unless --bank-samples 0.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.stdout.reconfigure(line_buffering=True)
CB = Path("/mnt/weka/sgrigoryan/dataset_codebase")
sys.path.insert(0, str(CB / "common/pipeline"))
import envs  # noqa: E402

FPS = 30
DEBUG_FOCAL_MM = 35.0
CAM_BACK_M, CAM_UP_M = 24.0, 4.0
X500_USD = str(CB / "common/assets/x500/x500.usd")
MARGIN_S = 1.0

ap = argparse.ArgumentParser()
ap.add_argument("--env", required=True, choices=envs.ENV_NAMES + ("Warehouse",))
ap.add_argument("--scene", required=True)
ap.add_argument("--flight", required=True, type=Path)
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--bank-samples", type=int, default=16, help="0 = no bank check")
ap.add_argument("--camera", choices=("behind", "side"), default="behind",
                help="behind: fixed camera behind the start hover; side: follows the drone from the side (collision tests)")
ap.add_argument("--max-seconds", type=float, default=0.0, help="only the last N s of the episode (0 = all)")
ap.add_argument("--stride", type=int, default=1, help="render every n-th telemetry frame")
ap.add_argument("--hold-s", type=float, default=0.0, help="hold the last frame this long")
ap.add_argument("--name", default="debug_external_onboard")
a = ap.parse_args()

tele = [r for r in json.loads((a.flight / "telemetry.json").read_text()) if not r.get("terminal")]
# post-contact: frames recorded after the first prohibited contact (mission post_contact_s) are part of the video
ep = [i for i, r in enumerate(tele) if r.get("phase") in ("episode", "post-contact")]
if not ep:
    sys.exit("no episode frames in the telemetry")
lo, hi = max(0, ep[0] - int(MARGIN_S * FPS)), min(len(tele), ep[-1] + 1 + int(MARGIN_S * FPS))
if a.max_seconds > 0 and ep[-1] + 1 - ep[0] > a.max_seconds * FPS:
    lo = ep[-1] + 1 - int(a.max_seconds * FPS)
cpath = a.flight / "collision.json"
col = json.loads(cpath.read_text()) if cpath.exists() else {}
hit = col.get("first_prohibited")
fixtures = (col.get("test_notes") or {}).get("fixtures") or []
r0 = tele[ep[0]]
hover = (r0["pose"]["x"], r0["pose"]["y"], r0["pose"]["z"])
yaw0 = math.radians(r0["pose"]["yaw_deg"])
a.out.mkdir(parents=True, exist_ok=True)
print(f"[debug] frames [{lo},{hi}) episode [{ep[0]},{ep[-1] + 1}) hover {hover} heading {math.degrees(yaw0):.1f}", flush=True)

if a.env == "Warehouse":
    import warehouse_env

    env = warehouse_env.boot(a.out / f"kit_debug_{os.environ.get('SLURM_JOB_ID', 'local')}")
else:
    env = envs.boot(a.env, a.out / f"kit_debug_{os.environ.get('SLURM_JOB_ID', 'local')}", mission_id=a.scene)
bridge = env.bridge
print(f"[debug] booted, render_mode={bridge.current_render_mode()}", flush=True)
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

stage = bridge.stage
# ---- visual X500 -----------------------------------------------------------------------------------------------------
root = UsdGeom.Xform.Define(stage, "/World/DebugX500")
root.GetPrim().GetReferences().AddReference(X500_USD)
x_tr = root.AddTranslateOp()
x_or = root.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
for prim in Usd.PrimRange(root.GetPrim()):
    for api in (UsdPhysics.ArticulationRootAPI, UsdPhysics.RigidBodyAPI, UsdPhysics.MassAPI, UsdPhysics.CollisionAPI):
        if prim.HasAPI(api):
            prim.RemoveAPI(api)
    if prim.IsA(UsdPhysics.Joint):
        prim.SetActive(False)
vis = UsdGeom.Imageable(root.GetPrim())
# ---- test fixtures, visual only (the flight spawned them at run time; the scene file does not have them) ------------------
for fx in fixtures:
    if fx["shape"] == "box":
        g = UsdGeom.Cube.Define(stage, fx["path"])
        g.CreateSizeAttr(1.0)
        g.AddTranslateOp().Set(Gf.Vec3d(*fx["centre"]))
        g.AddRotateZOp().Set(math.degrees(fx["yaw"]))
        g.AddScaleOp().Set(Gf.Vec3f(*fx["size"]))
    else:
        g = UsdGeom.Cylinder.Define(stage, fx["path"])
        g.CreateRadiusAttr(fx["radius"])
        g.CreateHeightAttr(fx["height"])
        g.CreateAxisAttr("Z")
        g.AddTranslateOp().Set(Gf.Vec3d(*fx["centre"]))
    g.CreateDisplayColorAttr([(1.0, 0.45, 0.05)])
    print(f"[debug] fixture {fx['path']} re-created (visual only)", flush=True)

# ---- external camera: fixed, behind + above the starting hover position, aimed at it ---------------------------------
gx, gy = hover[0] - CAM_BACK_M * math.cos(yaw0), hover[1] - CAM_BACK_M * math.sin(yaw0)
cam_xyz = (gx, gy, max(hover[2] + CAM_UP_M, env.ground(gx, gy) + 3.0))
d = math.hypot(hover[0] - cam_xyz[0], hover[1] - cam_xyz[1])
cam_yaw = math.degrees(math.atan2(hover[1] - cam_xyz[1], hover[0] - cam_xyz[0]))
cam_pitch = math.degrees(math.atan2(cam_xyz[2] - hover[2], d))
focal = DEBUG_FOCAL_MM
SIDE_DIST_M, SIDE_FOCAL_MM = (4.5, 18.0) if a.env == "Warehouse" else (12.0, 24.0)
LAMP_Z_M = 5.8
# indoors the racks, walls and ceiling block the view: their boxes (warehouse_layout.json, from the GPU probe) are checked
# like the test fixtures
SCENE_BOXES = [dict(shape="box", centre=[(lo + hi) / 2 for lo, hi in zip(b["min"], b["max"])],
                    size=[hi - lo for lo, hi in zip(b["min"], b["max"])], yaw=0.0)
               for b in (getattr(env, "layout", None) or {}).get("blockers", [])] if a.env == "Warehouse" else []
offset = None


def _blocked(cam, tgt, obst):
    """Number of proxy shapes the segment cam -> tgt passes through (vertical cylinders r = radius_m)."""
    n = 0
    for o in obst:
        for k in range(1, 30):
            u = k / 30.0
            x, y, z = (cam[i] + u * (tgt[i] - cam[i]) for i in range(3))
            if o["base_z"] <= z <= o["base_z"] + o["height_m"] and math.hypot(x - o["xy"][0], y - o["xy"][1]) < o["radius_m"]:
                n += 1
                break
    return n


def _in_fixture(x, y, z, fx, pad=0.0):
    """Is the point inside the test fixture (box: oriented by its yaw; cylinder: vertical), grown by pad?"""
    cx, cy, cz = fx["centre"]
    if fx["shape"] == "box":
        c, s = math.cos(-fx["yaw"]), math.sin(-fx["yaw"])
        lx, ly = c * (x - cx) - s * (y - cy), s * (x - cx) + c * (y - cy)
        return all(abs(v) <= h / 2 + pad for v, h in zip((lx, ly, z - cz), fx["size"]))
    return math.hypot(x - cx, y - cy) <= fx["radius"] + pad and abs(z - cz) <= fx["height"] / 2 + pad


def _fixture_blocked(cam, tgt):
    """Number of test fixtures (and, indoors, scene boxes) the segment cam -> tgt passes through (the last 5 % at the drone
    is skipped: it touches them)."""
    n = 0
    for fx in fixtures + SCENE_BOXES:
        if _in_fixture(*cam, fx, pad=0.5) or any(
                _in_fixture(*(cam[i] + k / 60.0 * (tgt[i] - cam[i]) for i in range(3)), fx) for k in range(1, 57)):
            n += 1
    return n


if a.camera == "side":
    # constant world-frame offset from the drone, chosen once: side of the motion, line of sight clear of proxies
    pts = [np.array([tele[i]["pose"]["x"], tele[i]["pose"]["y"], tele[i]["pose"]["z"]]) for i in (lo, (lo + hi) // 2, hi - 1)]
    mv = pts[-1] - pts[0]
    head = math.atan2(mv[1], mv[0]) if math.hypot(mv[0], mv[1]) > 1.0 else math.radians(tele[hi - 1]["pose"]["yaw_deg"])
    pxf = a.flight / "proxies.json"
    prox = json.loads(pxf.read_text()) if pxf.exists() else []
    near = [o for o in prox if min(math.hypot(o["xy"][0] - q[0], o["xy"][1] - q[1]) for q in pts) < SIDE_DIST_M + 10.0]
    cands = []
    for rel in (90, -90, 60, -60, 120, -120, 30, -30, 150, -150, 180):
        # indoors near the ceiling (ceiling test) every camera above the drone is among the lamps: also try below it
        ups = (0.5, 1.5, 2.5) + ((-2.0, -3.0, -4.0) if max(q[2] for q in pts) > LAMP_Z_M - 1.5 else ())
        for up in (ups if a.env == "Warehouse" else (0.5, 3.0, 6.0, 10.0)):
            az = head + math.radians(rel)
            off = np.array([SIDE_DIST_M * math.cos(az), SIDE_DIST_M * math.sin(az), up])
            under = sum(max(0.0, env.ground((q + off)[0], (q + off)[1]) + 2.0 - (q + off)[2]) for q in pts)
            score = (sum(_blocked(q + off, q, near) for q in pts) * 10 + sum(_fixture_blocked(q + off, q) for q in pts) * 100
                     + under + 0.01 * abs(rel) + 0.05 * abs(up)
                     # indoors: the ceiling lamps hang from z 5.8 m and render as black slabs when the camera is among them
                     + (1000.0 if a.env == "Warehouse" and max((q + off)[2] for q in pts) > LAMP_Z_M - 0.5 else 0.0))
            cands.append((score, rel, up, off))
    score, rel, up, offset = min(cands, key=lambda c: c[0])
    focal = SIDE_FOCAL_MM
    cam_pitch = math.degrees(math.atan2(offset[2], math.hypot(offset[0], offset[1])))
    cam_yaw = math.degrees(math.atan2(-offset[1], -offset[0]))
    print(f"[debug] side camera: {rel:+d} deg from motion, {up} m up, {SIDE_DIST_M} m away, score {score:.2f} "
          f"({len(near)} proxies, {len(fixtures)} test fixtures checked)", flush=True)
cam = UsdGeom.Camera(stage.GetPrimAtPath(bridge.camera_path))
W, H = envs.FRAME_SIZE
H_AP = envs.H_APERTURE_MM


def set_view(external, rec):
    global cam_xyz
    if external:
        if offset is not None:
            p = rec["pose"]
            cam_xyz = (p["x"] + offset[0], p["y"] + offset[1], p["z"] + offset[2])
        cam.GetFocalLengthAttr().Set(focal)
        bridge.set_camera_pose(*cam_xyz, cam_yaw, cam_pitch)
    else:
        cam.GetFocalLengthAttr().Set(envs.FOCAL_MM)
        c = rec["camera"]
        bridge.set_camera_pose(c["x"], c["y"], c["z"], c["yaw_deg"], c["pitch_deg"])


def place(rec):
    p = rec["pose"]
    qx, qy, qz, qw = rec["attitude_quat_xyzw"]
    x_tr.Set(Gf.Vec3d(p["x"], p["y"], p["z"]))
    x_or.Set(Gf.Quatd(qw, qx, qy, qz))


def render():
    f = bridge.process_camera_data()
    for _ in range(10):
        if f.shape[:2] == (H, W):
            break
        f = bridge.process_camera_data()
    if f.shape[:2] != (H, W):
        raise SystemExit("empty render")
    return f[..., :3].copy()


def project(pt):
    """World point -> pixel of the EXTERNAL camera (same pinhole as envs.setup_camera, focal `focal`)."""
    fx = focal / H_AP * W
    cy_, sy_ = math.cos(math.radians(cam_yaw)), math.sin(math.radians(cam_yaw))
    cp_, sp_ = math.cos(math.radians(cam_pitch)), math.sin(math.radians(cam_pitch))
    fwd = np.array([cp_ * cy_, cp_ * sy_, -sp_])
    right = np.array([sy_, -cy_, 0.0])
    up = np.cross(right, fwd)
    v = np.asarray(pt) - np.asarray(cam_xyz)
    z = float(v @ fwd)
    return (W / 2 + fx * float(v @ right) / z, H / 2 - fx * float(v @ up) / z), z


def body_axis_px(rec):
    """Predicted image angle (deg, + = counter-clockwise on screen) of the drone's left-right axis, and its pixel centre."""
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat(rec["attitude_quat_xyzw"])
    p = np.array([rec["pose"]["x"], rec["pose"]["y"], rec["pose"]["z"]])
    (lx, ly), _ = project(p + R.apply([0, 0.25, 0.03]))
    (rx, ry), _ = project(p + R.apply([0, -0.25, 0.03]))
    (cx, cy), z = project(p)
    return math.degrees(math.atan2(-(ly - ry), lx - rx)), (cx, cy), z


def mask_axis(with_d, without_d, centre, win=70):
    """Principal-axis angle of the drone silhouette, only within +-win px of the projected drone (its shadow on the ground
    and path-tracing noise elsewhere would bias it)."""
    diff = np.abs(with_d.astype(np.int16) - without_d.astype(np.int16)).sum(axis=2) > 30
    keep = np.zeros_like(diff)
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    keep[max(0, cy - win):cy + win, max(0, cx - win):cx + win] = True
    diff &= keep
    ys, xs = np.nonzero(diff)
    if len(xs) < 12:
        return None, int(len(xs)), diff
    X = np.stack([xs - xs.mean(), -(ys - ys.mean())])
    w, v = np.linalg.eigh(np.cov(X))
    ax = v[:, int(np.argmax(w))]
    ang = math.degrees(math.atan2(ax[1], ax[0]))
    ang = (ang + 90.0) % 180.0 - 90.0
    return ang, int(len(xs)), diff


from PIL import Image, ImageDraw  # noqa: E402
from encode_video import VideoWriter  # noqa: E402


def contact_mark(img, rec, external, last):
    """Red banner (and, in the external view, a circle at the contact point) once the first prohibited contact happened."""
    # the run stops right after the contact's physics step, so the last recorded frame can precede it by < 1 frame
    if hit is None or (rec["t"] < hit["t"] - 1.0 / FPS and not last):
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, H - 60, W, H], fill=(200, 0, 0))
    d.text((16, H - 48), f"FIRST PROHIBITED CONTACT  t = {hit['t']:.3f} s  {hit['drone_part']} ({hit['drone_collider'].split('/')[-1]})"
                         f"  ->  {hit['other_collider']}   episode FAILED", fill=(255, 255, 255))
    if rec.get("phase") == "post-contact":
        p4 = rec.get("px4") or {}
        d.rectangle([0, H - 92, W, H - 60], fill=(40, 40, 40))
        d.text((16, H - 84), f"AFTER CONTACT  +{rec.get('s_after_first_contact', 0.0):.2f} s   tilt {rec.get('tilt_deg', 0.0):.0f} deg   "
                             f"PX4: {p4.get('system_status')}  armed={p4.get('armed')}  {p4.get('landed_state')}", fill=(255, 255, 255))
    if external and hit.get("points") and rec["t"] <= hit["t"] + 0.5:
        (u, v), z = project(hit["points"][0]["position"])
        if z > 0:
            d.ellipse([u - 18, v - 18, u + 18, v + 18], outline=(255, 0, 0), width=4)
    return np.asarray(im)


def overlay(img, rec, title):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    s = rec.get("sticks") or [None] * 4
    p = rec["pose"]
    lines = [title, f"t = {rec['t']:7.3f} s   frame {rec['frame']}   phase {rec.get('phase')}",
             f"beat: {rec.get('stage')}",
             f"sticks x1 yaw {s[0]:+.1f}  y1 thr {s[1]:+.1f}  x2 roll {s[2]:+.1f}  y2 pitch {s[3]:+.1f}",
             f"roll {p['roll_deg']:+6.2f} deg   pitch {p['pitch_deg']:+6.2f} deg   yaw {p['yaw_deg']:+7.2f} deg"]
    d.rectangle([8, 8, 760, 8 + 26 * len(lines)], fill=(0, 0, 0))
    for i, t in enumerate(lines):
        d.text((16, 12 + 26 * i), t, fill=(255, 255, 255))
    return np.asarray(im)


def inset(img, centre, size=90, zoom=4):
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, y0 = max(0, cx - size // 2), max(0, cy - size // 2)
    crop = img[y0:y0 + size, x0:x0 + size]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return img
    big = np.asarray(Image.fromarray(crop).resize((crop.shape[1] * zoom, crop.shape[0] * zoom), Image.NEAREST))
    out = img.copy()
    hh, ww = big.shape[:2]
    out[H - hh - 10:H - 10, W - ww - 10:W - 10] = big
    out[H - hh - 12:H - hh - 10, W - ww - 12:W - 8] = 255
    return out


half = (W // 2, H // 2)
stride = max(1, a.stride)
vw = VideoWriter(a.out / f"{a.name}.partial.mp4", W, H // 2, fps=FPS / stride)
frames = list(range(lo, hi, stride))
if frames[-1] != hi - 1:
    frames.append(hi - 1)                      # always the last recorded frame (the contact, for a collision test)
samples = set()
if a.bank_samples > 0:
    epw = [i for i in ep if i >= lo]
    samples = {min(frames, key=lambda f: abs(f - x)) for x in np.linspace(epw[0], epw[-1], a.bank_samples)}
    samples.add(min(frames, key=lambda f: abs(f - max(epw, key=lambda i: abs(tele[i]["pose"]["roll_deg"])))))
(a.out / "bank_check_frames").mkdir(exist_ok=True)
bank = []
frame = None
for i in frames:
    rec = tele[i]
    place(rec)
    vis.MakeInvisible()
    set_view(False, rec)
    onboard = render()
    vis.MakeVisible()
    set_view(True, rec)
    ext = render()
    pred, centre, depth = body_axis_px(rec)
    if i in samples:
        vis.MakeInvisible()
        bg = render()
        vis.MakeVisible()
        meas, npx, mask = mask_axis(ext, bg, centre)
        Image.fromarray((mask * 255).astype(np.uint8)).save(a.out / "bank_check_frames" / f"mask_{i:06d}.png")
        row = {"frame": i, "t": rec["t"], "roll_deg": rec["pose"]["roll_deg"], "pitch_deg": rec["pose"]["pitch_deg"],
               "predicted_image_angle_deg": round(pred, 2), "measured_image_angle_deg": None if meas is None else round(meas, 2),
               "mask_pixels": npx, "drone_distance_m": round(depth, 2), "drone_pixel": [round(v, 1) for v in centre]}
        if meas is not None:
            e = (meas - pred + 90.0) % 180.0 - 90.0
            row["error_deg"] = round(e, 2)
        bank.append(row)
        print(f"[debug] bank frame {i}: roll {rec['pose']['roll_deg']:+.2f} predicted {pred:+.2f} measured {meas}", flush=True)
    ext_title = ("EXTERNAL (camera following at a fixed offset, debug only)" if offset is not None
                 else "EXTERNAL (fixed world camera, debug only)")
    ext = contact_mark(inset(overlay(ext, rec, ext_title), centre), rec, True, i == frames[-1])
    onboard = contact_mark(overlay(onboard, rec, "ONBOARD (stabilised, as in the benchmark)"), rec, False, i == frames[-1])
    frame = np.concatenate([np.asarray(Image.fromarray(ext).resize(half)), np.asarray(Image.fromarray(onboard).resize(half))], axis=1)
    vw.write_frame(frame)
    if (i - lo) % 30 == 0:
        print(f"[debug] {i - lo + 1}/{hi - lo} frames", flush=True)
for _ in range(int(a.hold_s * FPS / stride)):
    vw.write_frame(frame)
vw.close()
(a.out / f"{a.name}.partial.mp4").rename(a.out / f"{a.name}.mp4")
errs = [abs(r["error_deg"]) for r in bank if "error_deg" in r]
res = {"external_camera": {"mode": a.camera, "xyz": [round(v, 3) for v in cam_xyz], "yaw_deg": round(cam_yaw, 3),
                           "pitch_deg": round(cam_pitch, 3), "focal_mm": focal,
                           "aimed_at_hover_xyz": None if offset is not None else [round(v, 3) for v in hover],
                           "follow_offset_m": None if offset is None else [round(float(v), 3) for v in offset]},
       "frames": [lo, hi], "stride": stride, "episode_frames": [ep[0], ep[-1] + 1],
       "first_prohibited_contact": None if hit is None else {k: hit[k] for k in ("t", "step", "drone_part", "drone_collider",
                                                                                "other_collider")},
       "fixtures": fixtures,
       "method": "image angle of the X500 silhouette (principal axis of |render with drone - render without| > 30) vs the "
                 "angle of the body left-right axis projected from the recorded attitude quaternion",
       "samples": bank, "max_abs_error_deg": max(errs) if errs else None,
       "mean_abs_error_deg": round(sum(errs) / len(errs), 2) if errs else None,
       "max_abs_roll_deg_in_samples": max((abs(r["roll_deg"]) for r in bank), default=None)}
(a.out / ("bank_check.json" if a.bank_samples > 0 else f"{a.name}.json")).write_text(json.dumps(res, indent=1))
print("[debug] DONE", json.dumps({k: v for k, v in res.items() if k != "samples"}), flush=True)
bridge.shutdown()
