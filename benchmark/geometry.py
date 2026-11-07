"""Camera geometry shared by publish_episode.py and evaluate.py (pure Python, no numpy needed).

World ENU metres (x east, y north, z up). Camera yaw: 0 = +x, counter-clockwise positive. Camera pitch: positive = looking
down. Camera axes follow OpenCV (x right, y down, z forward):  X_cam = R_wc @ X_world + t_wc."""
import math


def rotation_wc(yaw_deg, pitch_deg):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    f = (math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p))   # forward
    r = (math.sin(y), -math.cos(y), 0.0)                                        # right
    d = (f[1] * r[2] - f[2] * r[1], f[2] * r[0] - f[0] * r[2], f[0] * r[1] - f[1] * r[0])  # down = forward x right
    return [list(r), list(d), list(f)]


def extrinsics(cam):
    R = rotation_wc(cam["yaw_deg"], cam["pitch_deg"])
    c = (cam["x"], cam["y"], cam["z"])
    t = [-sum(R[i][j] * c[j] for j in range(3)) for i in range(3)]
    return {"R_wc": [[round(v, 6) for v in row] for row in R], "t_wc": [round(v, 4) for v in t]}


def project(cam, point, K):
    """Pixel (u, v) of a world point seen from camera pose `cam`, or None if it is behind the camera."""
    R = rotation_wc(cam["yaw_deg"], cam["pitch_deg"])
    d = (point[0] - cam["x"], point[1] - cam["y"], point[2] - cam["z"])
    xc, yc, zc = (sum(R[i][j] * d[j] for j in range(3)) for i in range(3))
    if zc <= 0.1:
        return None
    return K["fx_px"] * xc / zc + K["cx_px"], K["fy_px"] * yc / zc + K["cy_px"]


def in_frame(uv, size, margin):
    """True if pixel uv lies inside the image with `margin` (fraction of width / height) kept clear at every edge."""
    if uv is None:
        return False
    w, h = size
    return margin * w <= uv[0] <= (1 - margin) * w and margin * h <= uv[1] <= (1 - margin) * h


def bearing_error_deg(pose, point):
    """Angle between the drone's heading and the horizontal direction to `point`."""
    want = math.atan2(point[1] - pose["y"], point[0] - pose["x"])
    e = want - math.radians(pose["yaw_deg"])
    return abs(math.degrees(math.atan2(math.sin(e), math.cos(e))))


def xyz(rec):
    p = rec["pose"]
    return (p["x"], p["y"], p["z"])


def path_length(tele):
    return sum(math.dist(xyz(a), xyz(b)) for a, b in zip(tele, tele[1:]))
