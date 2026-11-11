"""Build track_controller.mp4: a composite review video showing, per frame,
  - LEFT   : the rendered camera view (what the drone sees)
  - TOP-R  : the trajectory, full path drawn once + a live marker, home/target marked
  - BOT-R  : the joystick controller state the frame was collected with (both sticks), plus the
             live telemetry readout (phase, time, altitude, distance to target, gimbal pitch)

This is the "which controller inputs produced this footage" view -- it draws the SAME x1/y1/x2/y2
values recorded in action.json, so the panel is literally the collected data, not a re-simulation.

CPU-only (PIL + the existing ffmpeg VideoWriter); no GPU and no Isaac Sim needed. Run it after a
mission has rendered:

    python make_track_controller.py --mission <path to simulation_*/ dir>
"""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from encode_video import VideoWriter

W, H = 1280, 720
CAM_W, CAM_H = 800, 600          # camera view is upscaled to this box on the left
PANEL_X = CAM_W + 16             # right-hand column origin
PANEL_W = W - PANEL_X - 16
TRAJ_H = 360
BG = (18, 20, 24)
FG = (232, 236, 242)
DIM = (120, 128, 140)
ACCENT = (86, 180, 233)          # path / current position
TARGET_C = (233, 122, 86)
HOME_C = (120, 200, 120)
LAND_C = (214, 178, 86)          # touchdown pad (0_0_4 lands somewhere else than it started)
STICK_BG = (30, 34, 40)
STICK_FG = (86, 180, 233)


def _font(size):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_trajectory_panel(draw, records, idx, box, meta):
    """Top-right: bird's-eye path. Drawn in world XY, auto-scaled to fit with margin."""
    x0, y0, w, h = box
    draw.rectangle([x0, y0, x0 + w, y0 + h], fill=STICK_BG)
    xs = [r["pose"]["x"] for r in records]
    ys = [r["pose"]["y"] for r in records]
    tgt = meta.get("target_position", {})
    home = meta.get("home_position", {})
    # 0_0_4 lands at a different pad; there-and-back projects have no land_position, so fall back
    land = meta.get("land_position", home)
    pts_x = xs + [tgt.get("x", 0.0), home.get("x", 0.0), land.get("x", 0.0)]
    pts_y = ys + [tgt.get("y", 0.0), home.get("y", 0.0), land.get("y", 0.0)]
    min_x, max_x = min(pts_x), max(pts_x)
    min_y, max_y = min(pts_y), max(pts_y)
    span = max(max_x - min_x, max_y - min_y, 1.0) * 1.15
    cx, cy = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0
    scale = min(w, h) / span

    def to_px(wx, wy):
        # world +x right, world +y UP on screen (so the plot reads like a map, not flipped)
        return (x0 + w / 2.0 + (wx - cx) * scale,
                y0 + h / 2.0 - (wy - cy) * scale)

    # full path (dim), then the portion already flown (bright)
    full = [to_px(r["pose"]["x"], r["pose"]["y"]) for r in records]
    if len(full) > 1:
        draw.line(full, fill=DIM, width=2)
        if idx > 0:
            draw.line(full[: idx + 1], fill=ACCENT, width=3)

    hx, hy = to_px(home.get("x", 0.0), home.get("y", 0.0))
    draw.ellipse([hx - 6, hy - 6, hx + 6, hy + 6], outline=HOME_C, width=2)
    tx, ty = to_px(tgt.get("x", 0.0), tgt.get("y", 0.0))
    draw.line([tx - 7, ty, tx + 7, ty], fill=TARGET_C, width=2)
    draw.line([tx, ty - 7, tx, ty + 7], fill=TARGET_C, width=2)
    lands_elsewhere = meta.get("lands_elsewhere", False)
    if lands_elsewhere:
        ex, ey = to_px(land.get("x", 0.0), land.get("y", 0.0))
        # filled square = touchdown pad, visually distinct from the hollow start circle
        draw.rectangle([ex - 5, ey - 5, ex + 5, ey + 5], fill=LAND_C)

    # live drone marker + heading tick
    px, py = full[idx]
    draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=ACCENT)
    yaw = math.radians(records[idx]["pose"]["yaw_deg"])
    draw.line([px, py, px + 16 * math.cos(yaw), py - 16 * math.sin(yaw)], fill=FG, width=2)

    f = _font(14)
    draw.text((x0 + 8, y0 + 6), "TRAJECTORY (top-down)", font=f, fill=FG)
    draw.text((x0 + 8, y0 + h - 40), "start", font=_font(12), fill=HOME_C)
    draw.text((x0 + 48, y0 + h - 40), "target", font=_font(12), fill=TARGET_C)
    if lands_elsewhere:
        draw.text((x0 + 94, y0 + h - 40), "land", font=_font(12), fill=LAND_C)
    draw.text((x0 + 8, y0 + h - 22), f"scale ~{span:.0f} m across", font=_font(12), fill=DIM)


def draw_stick(draw, cx, cy, radius, sx, sy, label, f):
    """One joystick gimbal: square gate, cross-hairs, and the live stick position."""
    draw.rectangle([cx - radius, cy - radius, cx + radius, cy + radius], fill=STICK_BG,
                   outline=DIM, width=1)
    draw.line([cx - radius, cy, cx + radius, cy], fill=(60, 66, 74), width=1)
    draw.line([cx, cy - radius, cx, cy + radius], fill=(60, 66, 74), width=1)
    draw.ellipse([cx - radius * 0.72, cy - radius * 0.72, cx + radius * 0.72, cy + radius * 0.72],
                 outline=(52, 58, 66), width=1)
    # stick: +x right, +y UP on screen
    px = cx + sx * radius
    py = cy - sy * radius
    draw.line([cx, cy, px, py], fill=(70, 120, 150), width=2)
    draw.ellipse([px - 7, py - 7, px + 7, py + 7], fill=STICK_FG)
    draw.text((cx - radius, cy + radius + 6), label, font=f, fill=DIM)


def draw_controller_panel(draw, rec, box, meta):
    x0, y0, w, h = box
    f = _font(14)
    fs = _font(12)
    draw.text((x0 + 8, y0 + 2), "CONTROLLER (recorded action.json)", font=f, fill=FG)

    r = 68
    left_cx = x0 + 8 + r + 10
    right_cx = left_cx + 2 * r + 46
    cy = y0 + 34 + r
    legend = meta.get("sticks_legend", {})
    draw_stick(draw, left_cx, cy, r, rec["x1"], rec["y1"], "LEFT  x1=yaw  y1=throttle", fs)
    draw_stick(draw, right_cx, cy, r, rec["x2"], rec["y2"], "RIGHT x2=roll  y2=pitch", fs)

    # numeric readout
    tx = x0 + 8
    ty = cy + r + 28
    p = rec["pose"]
    v = rec["velocity"]
    speed = math.sqrt(v["vx"] ** 2 + v["vy"] ** 2 + v["vz"] ** 2)
    lines = [
        f'phase      {rec["phase"]}',
        f't          {rec["t"]:6.2f} s     frame {rec["frame"]}',
        f'x1 {rec["x1"]:+.2f}  y1 {rec["y1"]:+.2f}   x2 {rec["x2"]:+.2f}  y2 {rec["y2"]:+.2f}',
        f'pos        ({p["x"]:7.1f}, {p["y"]:7.1f}, {p["z"]:6.1f}) m',
        f'yaw        {p["yaw_deg"]:6.1f} deg   speed {speed:5.2f} m/s',
        f'dist->tgt  {rec.get("distance_to_target_m", 0.0):6.1f} m',
    ]
    if "camera_pitch_deg" in rec:
        lines.append(f'gimbal     {rec["camera_pitch_deg"]:5.1f} deg down')
    for i, line in enumerate(lines):
        draw.text((tx, ty + i * 17), line, font=_font(13), fill=FG if i < 3 else DIM)


def build(mission_dir, out_path=None, fps=None, stride=1):
    mission_dir = Path(mission_dir)
    records = json.loads((mission_dir / "action.json").read_text())
    meta = json.loads((mission_dir / "meta.json").read_text())
    frames_dir = mission_dir / "frames"
    out_path = Path(out_path or (mission_dir / "track_controller.mp4"))
    fps = fps or meta.get("fps", 30)

    f_title = _font(16)
    vw = VideoWriter(out_path, W, H, fps=fps / stride)
    import numpy as np

    written = 0
    for idx in range(0, len(records), stride):
        rec = records[idx]
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)

        frame_file = frames_dir / f"image_{idx}.png"
        if frame_file.exists():
            cam = Image.open(frame_file).convert("RGB").resize((CAM_W, CAM_H), Image.BILINEAR)
            canvas.paste(cam, (8, 48))
        draw.text((8, 12), f'{meta.get("simulation_id", mission_dir.name)}', font=f_title, fill=FG)
        draw.text((8, 30), f'{meta.get("environment", "")}', font=_font(12), fill=DIM)
        draw.text((8, 48 + CAM_H + 8), "CAMERA (rendered frame)", font=_font(13), fill=DIM)

        draw_trajectory_panel(draw, records, idx, (PANEL_X, 48, PANEL_W, TRAJ_H), meta)
        draw_controller_panel(draw, rec, (PANEL_X, 48 + TRAJ_H + 12, PANEL_W, H - TRAJ_H - 72), meta)

        vw.write_frame(np.asarray(canvas, dtype=np.uint8))
        written += 1
        if written % 100 == 0:
            print(f"  composited {written} frames...", flush=True)
    vw.close()
    print(f"wrote {out_path} ({written} frames @ {fps/stride:.1f} fps)")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mission", required=True, help="path to a simulation_*/ directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1, help="composite every Nth frame")
    args = ap.parse_args()
    build(args.mission, args.out, args.fps, args.stride)


if __name__ == "__main__":
    main()
