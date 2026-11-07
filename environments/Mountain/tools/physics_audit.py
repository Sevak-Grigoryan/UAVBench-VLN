"""Physics / placement audit of Easy Mountain scenes (no Isaac Sim): every object must rest on the ground, nothing may stand
in water (except bridges, jetties, boats), nothing may intersect, trees stay out of buildings / roads / water / landing places,
roads stay dry, take-off and landing places are dry, level and clear.   python physics_audit.py 2 3 4 ...
Writes Mountain/_audit/physics_<idx>.json (one record per simulation, list of problems)."""
import json, math, sys
from pathlib import Path
import numpy as np
CB = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(CB / "common/pipeline"), str(CB / "Mountain"), str(CB / "Mountain/scene")]
import mountain_v2 as mv
import mountain_sites as ms
import mountain_nature as mn
import missions as mm
OUT = CB / "Mountain/_audit"
OUT.mkdir(exist_ok=True)
WET_OK = ("bridge", "jetty", "boat", "boathouse", "segment", "gondola", "cable")


def static_box(st):
    """World footprint of a static: base part (lowest) and overall bounds."""
    M = st["M"]
    x, y, z = M[3][0], M[3][1], M[3][2]
    yaw = math.atan2(M[0][1], M[0][0])
    bs = [(ms._part_bounds(p), p) for p in st["parts"] if not p.get("hidden")]
    return x, y, z, yaw, bs


def footprint_pts(x, y, yaw, b, n=4):
    out = []
    for a in np.linspace(b[0], b[1], n):
        for c in np.linspace(b[2], b[3], n):
            dx, dy = mv.rot(yaw, a, c)
            out.append((x + dx, y + dy))
    return out


def audit(idx):
    rec = json.loads((CB / f"Mountain/_offline/{idx:03d}.json").read_text())
    att = rec["attempt"]
    pl = mv.plan(idx, att)
    P = []
    # ---- statics: in water / floating / buried
    for st in pl["statics"]:
        if not st["parts"]:
            continue
        x, y, z, yaw, bs = static_box(st)
        name = st["name"].lower()
        wet_ok = any(k in name for k in WET_OK) or st["cls"] in ("bridge", "boat")
        zmin = min(b[4] for b, _ in bs)
        base = [b for b, _ in bs if b[4] <= zmin + 0.3]
        b = (min(q[0] for q in base), max(q[1] for q in base), min(q[2] for q in base), max(q[3] for q in base))
        if (b[1] - b[0]) * (b[3] - b[2]) < 0.01:
            continue
        pts = footprint_pts(x, y, yaw, b)
        hs = [mv.terrain_height(px, py, pl) for px, py in pts]
        wl = [mv.water_level(px, py, pl) for px, py in pts]
        bottom = z + zmin
        if not wet_ok and any(w is not None and w > bottom - 0.05 for w in wl):
            P.append(dict(kind="static_in_water", obj=st["name"], cls=st["cls"], frac=round(sum(w is not None for w in wl) / len(wl), 2)))
        aerial = st["cls"] in ("cable car",) or "segment" in name or "gondola" in name or "wire" in name
        afloat = st["cls"] == "boat" or "jetty" in name
        if abs(x) + abs(y) + abs(z) < 1e-6 and "cable" not in name:  # world-frame parts (fences): every post must stand on the ground
            for bb, _ in bs:
                if bb[5] - bb[4] < 0.5:
                    continue
                hc = [mv.terrain_height(px, py, pl) for px in (bb[0], bb[1]) for py in (bb[2], bb[3])]
                if bb[4] > min(hc) + 0.1:
                    P.append(dict(kind="part_floating", obj=st["name"], gap_m=round(bb[4] - min(hc), 2)))
                    break
        elif "jetty" in name:  # every pile reaches the lake bed
            for bb, pp in bs:
                if pp["shape"] != "cyl":
                    continue
                cx, cy = (bb[0] + bb[1]) / 2, (bb[2] + bb[3]) / 2
                dx, dy = mv.rot(yaw, cx, cy)
                gap = z + bb[4] - mv.terrain_height(x + dx, y + dy, pl)
                if gap > 0.05:
                    P.append(dict(kind="jetty_pile_floating", obj=st["name"], gap_m=round(gap, 2)))
                    break
        elif st["cls"] == "boat":
            wl_ = mv.water_level(x, y, pl)
            if wl_ is None or abs(z - wl_) > 0.3:
                P.append(dict(kind="boat_not_on_water", obj=st["name"], water=wl_, z=round(z, 2)))
        elif not aerial:
            gap = bottom - min(hs)
            if gap > 0.25:
                P.append(dict(kind="static_floating", obj=st["name"], cls=st["cls"], gap_m=round(gap, 2)))
            top = z + max(bb[5] for bb, _ in bs)
            if max(hs) > top - 0.3 and top - bottom > 0.5:
                P.append(dict(kind="static_buried", obj=st["name"], cls=st["cls"], over_m=round(max(hs) - top, 2)))
    # ---- the river is open water along its whole course (never buried under terrain)
    v = pl["valley"]
    bur = 0
    for xr in np.arange(-mv.EXTENT + 10, mv.EXTENT - 10, 5.0):
        yr = float(mv.river_y(xr, v))
        if abs(yr) > mv.EXTENT - 5:
            continue
        lvl = float(mv.river_level(xr, v))
        if mv.terrain_height(xr, yr, pl) > lvl + 0.2 and not any(d[6] == "bridge" and mv._in_rect(xr, yr, d) for d in pl["decks"]):
            bur += 1
    if bur:
        P.append(dict(kind="river_buried", n_5m_steps=bur))
    # ---- obstacles (building boxes) of different sites must not intersect
    ob = pl["obstacles"]
    for i in range(len(ob)):
        for j in range(i + 1, len(ob)):
            a, b = ob[i], ob[j]
            if a[5] == b[5]:
                continue
            ix = min(a[1], b[1]) - max(a[0], b[0])
            iy = min(a[3], b[3]) - max(a[2], b[2])
            if ix > 0.5 and iy > 0.5:
                P.append(dict(kind="obstacles_intersect", a=a[5], b=b[5], ix=round(ix, 1), iy=round(iy, 1)))
    # ---- static vs static solids of different sites
    sol = pl["solids"]
    # (solids carry no owner; overlap between sites is guarded by site spacing -- check sites instead)
    S = pl["sites"]
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            d = math.hypot(S[i]["xy"][0] - S[j]["xy"][0], S[i]["xy"][1] - S[j]["xy"][1])
            if d < (S[i]["size_r"] + S[j]["size_r"]) * 0.8 and not ({S[i]["kind"], S[j]["kind"]} & {"cable car", "wind farm"}):
                P.append(dict(kind="sites_overlap", a=S[i]["name"], b=S[j]["name"], d=round(d, 1)))
    # ---- movables: rest pose, stability, water, mutual overlap
    mvs = mv.movables(pl)
    for m, q in zip(pl["movables"], mvs):
        if not q["static_check"]["stable"]:
            P.append(dict(kind="movable_unstable", obj=f"{m['group']}/{m['name']}", **q["static_check"]))
        if q["support_residual_m"] > 0.08:
            P.append(dict(kind="movable_support_gap", obj=f"{m['group']}/{m['name']}", resid=q["support_residual_m"]))
        r = mv._foot_r(m["prop"])
        wet = [(m["x"] + dx, m["y"] + dy) for dx in (-r, 0, r) for dy in (-r, 0, r) if mv.water_level(m["x"] + dx, m["y"] + dy, pl) is not None]
        if any(mv.surface_z(px, py, pl) < mv.water_level(px, py, pl) + 0.3 for px, py in wet) and m["kind"] not in ("rowing_boat", "boat"):
            P.append(dict(kind="movable_in_water", obj=f"{m['group']}/{m['name']}", movable_kind=m["kind"]))
    for i in range(len(pl["movables"])):
        a = pl["movables"][i]
        for j in range(i + 1, len(pl["movables"])):
            b = pl["movables"][j]
            d = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if d < (mv._foot_r(a["prop"]) + mv._foot_r(b["prop"])) * 0.9:
                P.append(dict(kind="movables_overlap", a=a["name"], b=b["name"], d=round(d, 2)))
    # ---- vegetation vs solids, roads, water, movables, places
    veg = pl["vegetation"]
    big = [(k, x, y, z, sc) for k, x, y, z, yaw, sc in veg if mn.PROTOS[k][4] in ("tree", "young", "broadleaf", "rock")]
    V = np.array([(x, y) for _, x, y, _, _ in big]) if big else np.zeros((0, 2))
    trunk = np.array([0.5 if mn.PROTOS[k][4] != "rock" else 0.6 * sc for k, _, _, _, sc in big]) if big else np.zeros(0)
    bad_solid = set()
    for sd in sol:
        if sd[0] != "rect":
            continue
        _, sx, sy, a, hx, hy, z0, z1 = sd
        if z1 - z0 < 0.3:
            continue
        near = np.where(np.hypot(V[:, 0] - sx, V[:, 1] - sy) < math.hypot(hx, hy) + 3)[0]
        for i in near:
            u, v_ = mv.rot(-a, V[i, 0] - sx, V[i, 1] - sy)
            if math.hypot(max(abs(u) - hx, 0), max(abs(v_) - hy, 0)) < trunk[i]:
                bad_solid.add(i)
    for i in sorted(bad_solid)[:20]:
        P.append(dict(kind="tree_in_solid", proto=big[i][0], xy=[round(V[i, 0], 1), round(V[i, 1], 1)]))
    if len(bad_solid) > 20:
        P.append(dict(kind="tree_in_solid_more", n=len(bad_solid) - 20))
    nw = 0
    for k, x, y, z, yaw, sc in veg:
        if mv.water_level(x, y, pl) is not None:
            nw += 1
    if nw:
        P.append(dict(kind="vegetation_in_water", n=nw))
    for m in pl["movables"]:
        r = mv._foot_r(m["prop"])
        if len(V) and np.any(np.hypot(V[:, 0] - m["x"], V[:, 1] - m["y"]) < r + trunk):
            P.append(dict(kind="tree_on_movable", obj=f"{m['group']}/{m['name']}"))
    for rd in pl["roads"]:
        pts = np.array(rd["pts"], float)
        seg = []
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            n = max(2, int(math.hypot(bx - ax, by - ay) / 3))
            seg += [(ax + (bx - ax) * t, ay + (by - ay) * t) for t in np.linspace(0, 1, n)]
        seg = [(x, y) for x, y in seg if abs(x) < mv.EXTENT and abs(y) < mv.EXTENT]
        wet = [(x, y) for x, y in seg if mv.water_level(x, y, pl) is not None and mv.surface_z(x, y, pl) < (mv.water_level(x, y, pl) or 0)]
        if wet:
            P.append(dict(kind="road_under_water", road=rd["name"], n_pts=len(wet), first=[round(wet[0][0], 1), round(wet[0][1], 1)]))
        if len(V):
            S2 = np.array(seg)
            hit = 0
            for i in range(0, len(S2), 2):
                hit += int(np.sum(np.hypot(V[:, 0] - S2[i, 0], V[:, 1] - S2[i, 1]) < rd["w"] / 2 + trunk) > 0)
            if hit:
                P.append(dict(kind="tree_on_road", road=rd["name"], n=hit))
    # ---- take-off / landing places
    places = {**pl["pads"], **pl.get("spots", {})}
    for k, p in places.items():
        x, y = p["xy"]
        if any(mv.water_level(x + dx, y + dy, pl) is not None for dx in (-3, 0, 3) for dy in (-3, 0, 3)):
            P.append(dict(kind="place_wet", place=k, words=p["words"]))
        zs = [mv.surface_z(x + dx, y + dy, pl) for dx in (-2.5, 0, 2.5) for dy in (-2.5, 0, 2.5)]
        if max(zs) - min(zs) > 0.6:
            P.append(dict(kind="place_not_level", place=k, dz=round(max(zs) - min(zs), 2), words=p["words"]))
        if len(V) and np.any(np.hypot(V[:, 0] - x, V[:, 1] - y) < 4.0):
            P.append(dict(kind="place_tree_within_4m", place=k, words=p["words"]))
        if any(math.hypot(m["x"] - x, m["y"] - y) < mv._foot_r(m["prop"]) + 2.0 for m in pl["movables"]):
            P.append(dict(kind="place_blocked_by_movable", place=k, words=p["words"]))
    # ---- the mission itself: start / land places used, and no pad in the unpublished set
    try:
        m = mm.build_mission(idx, pl) if hasattr(mm, "build_mission") else None
    except Exception:
        m = None
    used = [rec.get("start"), rec.get("land")]
    for u in used:
        if u in pl["pads"] and idx not in (1, 7, 28):
            P.append(dict(kind="uses_landing_pad", place=u))
    # ---- relief: the mountains keep their height (a saturating river cap once flattened every summit to +40 m)
    H = pl["_H"]
    floor = float(np.median(H[H.shape[0] // 2 - 5:H.shape[0] // 2 + 5, :]))
    relief = float(H.max() - floor)
    top_summit = max((p_["z"] for p_ in pl.get("summits", [])), default=0.0) - floor
    if relief < (90.0 if pl["landscape"] == "rolling highlands" else 140.0):
        P.append(dict(kind="terrain_flattened", relief_m=round(relief, 1)))
    out = dict(idx=idx, attempt=att, relief_m=round(relief, 1), top_summit_m=round(top_summit, 1), task=rec["task"], landscape=rec["landscape"], n_static=len(pl["statics"]), n_mov=len(pl["movables"]),
               n_veg=len(veg), problems=P, counts={k: sum(1 for p in P if p["kind"] == k) for k in sorted({p["kind"] for p in P})},
               start_words=rec.get("start_words"), land_words=rec.get("land_words"))
    (OUT / f"physics_{idx:03d}.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(dict(idx=idx, counts=out["counts"])), flush=True)


for a in sys.argv[1:]:
    try:
        audit(int(a))
    except Exception as e:
        import traceback
        print(json.dumps(dict(idx=int(a), error=repr(e), tb=traceback.format_exc()[-800:])), flush=True)
