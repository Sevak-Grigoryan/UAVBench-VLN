"""
Collision TEST scenarios, flown for real (PX4 SITL + Isaac Sim PhysX) in the Forest scene easy_v2_<idx>, through run_flight.py.
They are tests of the collision monitor / termination / evaluator, not benchmark episodes.

Every scenario: take off from the scene's start clearing and settle (pre-roll; the resting contact on the take-off clearing is the
initialization phase), then the EVALUATED part (phase "episode"), then for the collision-free ones a post-roll landing.
Obstacles that must be hit precisely are spawned at run time relative to the drone's actual pose when the episode starts
(PX4 holds position only to a few cm-dm), as static colliders under /World/TestFixtures.

  wall           3 m/s forward into an upright 0.3 m wall spawned 6 m ahead                 -> prohibited (airframe / rotor)
  rotor_ceiling  slow climb (0.6 m/s) into a flat plate spawned 0.8 m above the drone       -> prohibited, first part = rotor
  tree           approach the nearest suitable Forest tree (vegetation proxy) at mid-height -> prohibited, other = tree proxy
  thin_fast      8 m/s (full stick) into a 2 cm diameter pole spawned 30 m ahead            -> prohibited, between decisions
  wrong_ground   land 8 m off the designated landing clearing                               -> prohibited landing-gear contact
  pad_ok         land on the designated landing clearing                                    -> allowed landing, no failure
  pad_tip        land on the designated clearing, then a 45 N side push tips the X500       -> prohibited rotor/airframe AT the pad
  debug_sideways hover -> move right -> stop (external debug-camera demo), land back        -> no contact in the episode
"""
import math

PRE, EPISODE, POST = "pre-roll", "episode", "post-roll"
NAMES = ("wall", "rotor_ceiling", "tree", "thin_fast", "wrong_ground", "pad_ok", "pad_tip", "debug_sideways")
TAKEOFF_AGL = 12.0
LAND_PAD = "W"
PAD_RADIUS_M = 2.5          # designated touchdown footprint on the terrain (Forest clearings are unmarked; pilot touchdown <= 1.1 m)
FIX = "/World/TestFixtures"
FIX_COLOR = (1.0, 0.45, 0.05)   # orange, so the fixtures stand out in the debug videos


def _common(env, extra_beats, landing=None):
    import missions as fm

    m0 = env.missions[env.scene_mission]
    beats = [dict(type="takeoff", agl=TAKEOFF_AGL, label="take off", phase=PRE),
             dict(type="sticks", sticks=(0.0, 0.0, 0.0, 0.0), s=3.0, label="settle", phase=PRE)] + extra_beats
    m = dict(m0, beats=beats)
    m["tasks"] = fm.tasks([b for b in beats if b["type"] != "sticks"])
    m["landing"] = landing
    return m


def _landing_spec(env):
    import missions as fm

    xy = fm.PADS[LAND_PAD][0]
    return {"pad": LAND_PAD, "xy": [float(xy[0]), float(xy[1])], "radius_m": PAD_RADIUS_M,
            "surface_colliders": ["/World/Terrain"],
            "note": "Forest landing clearings are unmarked spots on the shared terrain mesh: the supporting collider is "
                    "/World/Terrain, so every contact point must also lie within radius_m of the pad centre"}


def _ahead(rt, fwd, right=0.0, up=0.0):
    c, s = math.cos(rt.yaw), math.sin(rt.yaw)
    return (rt.pos[0] + fwd * c + right * s, rt.pos[1] + fwd * s - right * c, rt.pos[2] + up)


def _box(stage, path, centre, size, yaw):
    from pxr import Gf, UsdGeom, UsdPhysics

    g = UsdGeom.Cube.Define(stage, path)
    g.CreateSizeAttr(1.0)
    g.AddTranslateOp().Set(Gf.Vec3d(*centre))
    g.AddRotateZOp().Set(math.degrees(yaw))
    g.AddScaleOp().Set(Gf.Vec3f(*size))
    g.CreateDisplayColorAttr([FIX_COLOR])
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    return path


def _pole(stage, path, centre, radius, height):
    from pxr import Gf, UsdGeom, UsdPhysics

    g = UsdGeom.Cylinder.Define(stage, path)
    g.CreateRadiusAttr(radius)
    g.CreateHeightAttr(height)
    g.CreateAxisAttr("Z")
    g.AddTranslateOp().Set(Gf.Vec3d(*centre))
    g.CreateDisplayColorAttr([FIX_COLOR])
    UsdPhysics.CollisionAPI.Apply(g.GetPrim())
    return path


class Hooks:
    """Called by run_flight.py: on_frame(rt) once per video frame, on_step(rt) after every physics step."""

    def __init__(self, name, env):
        self.name, self.env = name, env
        self.spawned = []
        self.fixtures = []
        self.notes = {}
        self.push = None

    def on_frame(self, rt):
        n = self.name
        if n == "pad_tip":
            self._pad_tip(rt)
            return
        if rt.phase != EPISODE or self.spawned or n in ("wrong_ground", "pad_ok", "debug_sideways"):
            return
        st = rt.stage
        if n == "wall":
            c = _ahead(rt, 6.0 + 0.15)
            self.spawned.append(_box(st, f"{FIX}/wall", c, (0.3, 12.0, 20.0), rt.yaw))
            self.fixtures.append({"shape": "box", "path": f"{FIX}/wall", "centre": c, "size": (0.3, 12.0, 20.0), "yaw": rt.yaw})
        elif n == "rotor_ceiling":
            c = (rt.pos[0], rt.pos[1], rt.pos[2] + 0.8 + 0.05)
            self.spawned.append(_box(st, f"{FIX}/ceiling", c, (3.0, 3.0, 0.1), rt.yaw))
            self.fixtures.append({"shape": "box", "path": f"{FIX}/ceiling", "centre": c, "size": (3.0, 3.0, 0.1), "yaw": rt.yaw})
        elif n == "thin_fast":
            c = _ahead(rt, 30.0)
            self.spawned.append(_pole(st, f"{FIX}/thin_pole", (c[0], c[1], c[2]), 0.01, 30.0))
            self.fixtures.append({"shape": "cylinder", "path": f"{FIX}/thin_pole", "centre": c, "radius": 0.01, "height": 30.0})
            self.notes["pole_diameter_m"] = 0.02
        elif n == "tree":
            self.spawned.append("(no fixture: real Forest tree proxy)")
            self._plan_tree(rt)
        if self.spawned:
            self.notes["fixtures"] = self.fixtures
            self.notes["spawned_at_t"] = round(rt.t, 4)
            self.notes["drone_at_spawn"] = [round(v, 3) for v in rt.pos] + [round(math.degrees(rt.yaw), 2)]
            rt.log(f"[test] {n}: spawned {self.spawned} at t={rt.t:.3f}")

    def _plan_tree(self, rt):
        """Nearest tree proxy with a clear 14 m approach line; replace the placeholder beats with the approach."""
        best = None
        trees = [s for s in rt.proxies if s["kind"] in ("conifer", "broadleaf") and s["height_m"] >= 5.0]
        for tr in sorted(trees, key=lambda s: math.hypot(s["xy"][0] - rt.pos[0], s["xy"][1] - rt.pos[1]))[:200]:
            tx, ty = tr["xy"]
            a = math.atan2(rt.pos[1] - ty, rt.pos[0] - tx)
            ax, ay = tx + 14.0 * math.cos(a), ty + 14.0 * math.sin(a)
            ok = True
            for o in rt.proxies:
                if o is tr:
                    continue
                # distance from o to the segment approach point -> tree, must exceed o's radius + 2 m
                px, py = o["xy"][0] - ax, o["xy"][1] - ay
                vx, vy = tx - ax, ty - ay
                u = max(0.0, min(1.0, (px * vx + py * vy) / (vx * vx + vy * vy)))
                if math.hypot(px - u * vx, py - u * vy) < o["radius_m"] + 2.0 and o["base_z"] + o["height_m"] > tr["base_z"] + 1.0:
                    ok = False
                    break
            if ok:
                best = (tr, ax, ay)
                break
        if best is None:
            raise RuntimeError("no tree with a clear approach found")
        tr, ax, ay = best
        z_mid = tr["base_z"] + 0.5 * tr["height_m"] + 0.227
        look = (tr["xy"][0], tr["xy"][1], z_mid)
        beats = [dict(type="goto", xy=(ax, ay), agl=TAKEOFF_AGL, speed=0.6, radius=1.5, label="fly to the tree approach point",
                      look_at=look, phase=EPISODE),
                 dict(type="descend", z=max(z_mid, rt.ground(ax, ay) + 3.0), look_at=look, label="descend to mid-tree", phase=EPISODE),
                 dict(type="hover", s=2.0, look_at=look, align_first=True, label="face the tree", phase=EPISODE),
                 dict(type="sticks", sticks=(0.0, 0.0, 0.0, 0.2), s=20.0, label="fly into the tree", phase=EPISODE)]
        p = rt.pilot
        p.beats[p.i:] = beats
        p.beat_state = {}
        # the turn toward the approach point saturated the yaw stick (60 deg/s): PX4's heading estimate lagged > 15 deg behind
        # and run_flight aborted 12 m before the tree (px4_heading_mismatch, job 378816). Cap the yaw stick at 0.35 (~21 deg/s).
        yaw_stick = p._yaw_stick
        p._yaw_stick = lambda yaw, desired, ff=0.0: max(-0.35, min(0.35, yaw_stick(yaw, desired, ff)))
        self.notes["yaw_stick_cap"] = 0.35
        self.notes["tree"] = tr
        self.notes["approach_xy"] = [round(ax, 2), round(ay, 2)]

    def _pad_tip(self, rt):
        L = rt.monitor.allowed_landing
        if L is None or self.push is not None:
            return
        if rt.t >= L["t"] + 0.5:
            self.push = {"t_start": round(rt.t, 4), "duration_s": 0.5, "force_n": 45.0,
                         "direction": "airframe right (body -y)", "applied_at": "body origin"}
            rt.log(f"[test] pad_tip: side push 45 N for 0.5 s from t={rt.t:.3f}")
            self.notes["push"] = self.push

    def on_step(self, rt):
        if self.push is None or rt.t > self.push["t_start"] + self.push["duration_s"]:
            return
        f = 45.0
        rt.apply_body_force((f * math.sin(rt.yaw), -f * math.cos(rt.yaw), 0.0))


def build(name, env):
    if name not in NAMES:
        raise ValueError(f"unknown scenario {name}; one of {NAMES}")
    import missions as fm

    L = _landing_spec(env)
    pad = fm.PADS[LAND_PAD][0]
    if name == "wall":
        ep = [dict(type="sticks", sticks=(0.0, 0.0, 0.0, 0.4), s=8.0, label="fly forward 3.2 m/s", phase=EPISODE)]
    elif name == "rotor_ceiling":
        ep = [dict(type="sticks", sticks=(0.0, 0.2, 0.0, 0.0), s=6.0, label="climb 0.6 m/s", phase=EPISODE)]
    elif name == "thin_fast":
        ep = [dict(type="sticks", sticks=(0.0, 0.0, 0.0, 1.0), s=8.0, label="full forward 8 m/s", phase=EPISODE)]
    elif name == "tree":
        ep = [dict(type="sticks", sticks=(0.0, 0.0, 0.0, 0.0), s=1.0, label="(replaced by the tree approach)", phase=EPISODE)]
    elif name in ("wrong_ground", "pad_ok", "pad_tip"):
        xy = (pad[0] + 8.0, pad[1]) if name == "wrong_ground" else pad
        ep = [dict(type="goto", xy=xy, agl=22.0, speed=0.85, radius=4.0, label=f"fly to {LAND_PAD} pad", phase=EPISODE),
              dict(type="land", xy=xy, label=f"land {LAND_PAD}", pad=LAND_PAD, phase=EPISODE)]
    elif name == "debug_sideways":
        ep = [dict(type="sticks", sticks=(0.0, 0.0, 0.0, 0.0), s=3.0, label="hover", phase=EPISODE),
              dict(type="sticks", sticks=(0.0, 0.0, 0.4, 0.0), s=2.5, label="move right 3.2 m/s", phase=EPISODE),
              dict(type="sticks", sticks=(0.0, 0.0, 0.0, 0.0), s=5.0, label="stop", phase=EPISODE)]
        m0 = env.missions[env.scene_mission]
        ep += [dict(type="land", xy=tuple(m0["start_xy"]), label="land back", pad=m0["start"], phase=POST)]
        L = None
    m = _common(env, ep, landing=L)
    m["id"] = f"colltest_{name}"
    m["instruction"] = f"collision test scenario '{name}' (not a benchmark episode)"
    m["test"] = {"name": name, "expected": EXPECTED[name]}
    return m, Hooks(name, env)


EXPECTED = {
    "wall": {"outcome": "collision", "first_prohibited_other_prefix": FIX + "/wall"},
    "rotor_ceiling": {"outcome": "collision", "first_prohibited_part": "rotor", "first_prohibited_other_prefix": FIX + "/ceiling"},
    "tree": {"outcome": "collision", "first_prohibited_other_prefix": "/World/CollisionProxies/"},
    "thin_fast": {"outcome": "collision", "first_prohibited_other_prefix": FIX + "/thin_pole"},
    "wrong_ground": {"outcome": "collision", "first_prohibited_part": "landing_gear", "first_prohibited_other_prefix": "/World/Terrain"},
    "pad_ok": {"outcome": "completed", "allowed_landing": True, "first_prohibited": None},
    "pad_tip": {"outcome": "collision", "allowed_landing": True, "first_prohibited_part_in": ["rotor", "airframe"],
                "first_prohibited_other_prefix": "/World/Terrain"},
    "debug_sideways": {"outcome": "completed", "first_prohibited": None},
}
