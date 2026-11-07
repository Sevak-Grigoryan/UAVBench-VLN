"""Offline overlap check for Easy City: oriented footprint rectangles of every movable prop against each other and
against solid static parts of the special blocks, bus shelters, traffic lights (SAT test, 2 cm tolerance)."""
import math, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "common/pipeline"), str(HERE), str(HERE / "scene"), str(HERE.parent / "common/vln_simulation")]
import numpy as np
import city_v2 as cv, missions as cm, pilot, city_props as cp


def rect_of_parts(parts):
    xs, ys = [], []
    for p in parts:
        if not p.get("col"):
            continue
        c = p["c"]
        if p["shape"] == "box":
            hx, hy = p["s"][0] / 2, p["s"][1] / 2
            a = math.radians(p["rot"][2])
            for sx in (-1, 1):
                for sy in (-1, 1):
                    xs.append(c[0] + math.cos(a) * sx * hx - math.sin(a) * sy * hy)
                    ys.append(c[1] + math.sin(a) * sx * hx + math.cos(a) * sy * hy)
        else:
            r = p.get("r", 0.1) * max(p.get("scale", (1, 1, 1))[:2])
            ext = (p.get("h", 0) / 2 if p.get("axis") in ("X", "Y") else 0) + r
            ex = ext if p.get("axis") == "X" else r
            ey = ext if p.get("axis") == "Y" else r
            xs += [c[0] - ex, c[0] + ex]; ys += [c[1] - ey, c[1] + ey]
    return min(xs), max(xs), min(ys), max(ys)


def corners(cx, cy, yaw, r):
    x0, x1, y0, y1 = r
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[cx + c * x - s * y, cy + s * x + c * y] for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))])


def overlap(A, Bq, tol=0.02):
    for P in (A, Bq):
        for i in range(4):
            e = P[(i + 1) % 4] - P[i]
            n = np.array([-e[1], e[0]]) / (np.linalg.norm(e) + 1e-9)
            a, b = A @ n, Bq @ n
            if a.max() < b.min() + tol or b.max() < a.min() + tol:
                return False
    return True


def statics(pl):
    out = []
    for (bx, by), f in pl["features"].items():
        for p in f["parts"]:
            if not p.get("col") or p["shape"] not in ("box", "cyl"):
                continue
            if p["c"][2] - (p["s"][2] / 2 if p["shape"] == "box" else p.get("h", 0) / 2) > 1.6:
                continue  # canopies / roofs above head height
            r = rect_of_parts([p])
            out.append((f["name"], corners(bx, by, f["yaw"], r)))
    for k, (x, y, yaw) in enumerate(pl["shelters"]):
        for p in cp.bus_shelter(0.16):
            if p.get("col") and p["c"][2] < 1.8:
                out.append((f"shelter_{k}", corners(x, y, yaw, rect_of_parts([p]))))
    for k, (x, y, yaw) in enumerate(pl["lights"]):
        out.append((f"light_{k}", corners(x, y, 0, (-0.1, 0.1, -0.1, 0.1))))
    return out


bad_total = 0
for i in ([int(a) for a in sys.argv[1:]] or range(1, 101)):
    pl, m, L = cm.mission_and_plan(i, cv, pilot)
    mv = cv.movables(pl)
    R = [(x["group"] + "/" + x["name"], corners(x["xy"][0], x["xy"][1], x["yaw"], rect_of_parts(x["prop"]["parts"]))) for x in mv]
    S = statics(pl)
    bad = []
    for a in range(len(R)):
        ca = R[a][1].mean(0)
        for b in range(a + 1, len(R)):
            if np.linalg.norm(R[b][1].mean(0) - ca) < 16 and overlap(R[a][1], R[b][1]):
                bad.append((R[a][0], R[b][0]))
        for nm, Sq in S:
            if np.linalg.norm(Sq.mean(0) - ca) < 16 and overlap(R[a][1], Sq):
                bad.append((R[a][0], nm))
    bad_total += len(bad)
    if bad:
        print(i, len(bad), bad[:6])
print("total overlaps", bad_total)
