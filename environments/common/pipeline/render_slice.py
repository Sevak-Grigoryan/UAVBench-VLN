"""
RENDER phase: render frames [start, end) of a flown mission from its recorded camera poses.
Many of these run at once, one GPU each; finalize.py stitches the slices into trajectory.mp4.

    python render_slice.py --env Forest --telemetry <run>/flight/telemetry.json \
        --start 0 --end 900 --out <run>/slices/slice_0000.mp4

Valid because the environment is static and envs.boot() builds exactly the scene the flight used
(same code, seed, pads, camera intrinsics). The X500 itself is not in the scene: the camera sits
0.12 m ahead of and 0.07 m below the airframe centre and never sees the frame or props.

Each slice is written as <out>.partial and renamed to <out> only when every frame is in, so a
slice that crashes half-way is never mistaken for a finished one.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(line_buffering=True)
COMMON = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COMMON / "pipeline"))
import envs

FPS = 30

ap = argparse.ArgumentParser()
ap.add_argument("--env", required=True, choices=envs.ENV_NAMES)
ap.add_argument("--telemetry", required=True, type=Path)
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end", type=int, required=True)  # exclusive
ap.add_argument("--out", required=True, type=Path)
ap.add_argument("--mission", default=None, help="mission id, for scenes that depend on it (Forest easy_v2_*)")
a = ap.parse_args()

frames = json.loads(a.telemetry.read_text())
lo, hi = max(0, a.start), min(len(frames), a.end)
if lo >= hi:
    raise SystemExit(f"empty slice [{lo},{hi}) of {len(frames)} frames")
print(f"[slice] [{lo},{hi}) of {len(frames)} frames -> {a.out}", flush=True)

a.out.parent.mkdir(parents=True, exist_ok=True)
env = envs.boot(a.env, a.out.parent / f"kit_{a.out.stem}_{os.environ.get('SLURM_JOB_ID', 'local')}", mission_id=a.mission)
bridge = env.bridge
print(f"[slice] booted, render_mode={bridge.current_render_mode()}", flush=True)

from encode_video import VideoWriter

partial = a.out.with_suffix(".partial.mp4")
vw = VideoWriter(partial, envs.FRAME_SIZE[0], envs.FRAME_SIZE[1], fps=FPS)

# depth + per-object segmentation (environments that ask for it: Coastal v2, Easy-data review item #4). Written as
# lossless 16-bit FFV1 videos at half resolution; per-frame pixel counts of every landmark (target visibility, #3).
AUX = bool(getattr(env, "aux", False)) and hasattr(bridge, "enable_aux")
if AUX:
    import subprocess

    import numpy as np

    FF = "/mnt/weka/shared-cache/miniforge3/bin/ffmpeg"
    AW, AH = envs.FRAME_SIZE[0] // 2, envs.FRAME_SIZE[1] // 2

    def _ffv1(path):
        return subprocess.Popen([FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{AW}x{AH}",
                                 "-r", str(FPS), "-i", "-", "-c:v", "ffv1", "-level", "3", "-g", "1", "-pix_fmt", "gray16le", str(path)],
                                stdin=subprocess.PIPE)
    bridge.enable_aux()
    dpart, spart = a.out.with_suffix(".depth.partial.mkv"), a.out.with_suffix(".seg.partial.mkv")
    dproc, sproc = _ffv1(dpart), _ffv1(spart)
    lm_ids = {name: i + 1 for i, (name, _, _, _) in enumerate(bridge.registry) if name in env.landmarks}
    mask_px = {}
t0 = time.time()
for i in range(lo, hi):
    c = frames[i]["camera"]
    bridge.set_camera_pose(c["x"], c["y"], c["z"], c["yaw_deg"], c["pitch_deg"])
    frame = bridge.process_camera_data()
    tries = 0
    while frame.shape[:2] != (envs.FRAME_SIZE[1], envs.FRAME_SIZE[0]) and tries < 10:
        # an empty frame from the annotator would be skipped by the writer and leave the slice one frame
        # short (hard_river_patrol slice 3600, 899/900); re-render it instead
        tries += 1
        print(f"[slice] frame {i}: empty render {frame.shape}, retry {tries}", flush=True)
        frame = bridge.process_camera_data()
    if frame.shape[:2] != (envs.FRAME_SIZE[1], envs.FRAME_SIZE[0]):
        raise SystemExit(f"frame {i} could not be rendered after {tries} retries")
    vw.write_frame(frame)
    if AUX:
        depth, obj = bridge.get_aux()
        dcm = np.where(np.isfinite(depth), np.clip(depth * 100.0, 1, 65534), 65535).astype(np.uint16)
        dproc.stdin.write(np.ascontiguousarray(dcm[::2, ::2]).tobytes())
        sproc.stdin.write(np.ascontiguousarray(obj[::2, ::2].astype(np.uint16)).tobytes())
        cnt = np.bincount(obj.ravel(), minlength=max(lm_ids.values(), default=0) + 1) if lm_ids else []
        mask_px[i] = {n: int(cnt[k]) for n, k in lm_ids.items() if k < len(cnt) and cnt[k] > 0}
    done = i - lo + 1
    if done % 50 == 0 or i == lo:
        el = time.time() - t0
        print(f"[slice] {done}/{hi - lo} frames, {el:.0f}s, {el / done:.2f}s/frame", flush=True)
vw.close()
if AUX:
    for proc in (dproc, sproc):
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg (depth/segmentation) failed")
    dpart.rename(a.out.with_suffix(".depth.mkv"))
    spart.rename(a.out.with_suffix(".seg.mkv"))
    a.out.with_suffix(".mask.json").write_text(json.dumps(mask_px))
partial.rename(a.out)
el = time.time() - t0
(a.out.with_suffix(".json")).write_text(json.dumps({"start": lo, "end": hi, "frames": hi - lo, "render_s": round(el, 1),
                                                   "renderer": bridge.current_render_mode(),
                                                   "rt_subframes": env.hq_bridge.RT_SUBFRAMES,
                                                   "node": os.uname().nodename}))
print(f"[slice] SLICE DONE [{lo},{hi}) {hi - lo} frames in {el:.0f}s ({el / (hi - lo):.2f}s/frame)", flush=True)
bridge.shutdown()
