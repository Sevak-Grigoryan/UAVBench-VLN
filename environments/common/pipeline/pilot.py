"""
Stick pilot for the X500 missions: flies a list of mission "beats" the way a human pilot does in
PX4 Position mode -- by holding the two sticks still and only moving them when something needs
correcting.

  * Decisions are made at most every DECIDE_EVERY_S of sim time; between decisions the sticks are
    HELD. Each stick is quantised to STICK_STEP, so controller jitter never shows up as a new
    action. That is what lets action.json store only the moments the sticks actually change.
  * Stick convention (same as every earlier project): x1 yaw (+ right/clockwise), y1 throttle
    (+ up), x2 roll (+ right), y2 pitch (+ forward), all in [-1, 1].
  * PX4 Position-mode meaning of the sticks (airframe 10050_isaac_x500: linear, no expo):
    y2/x2 -> body-frame velocity at 8 m/s full deflection, y1 -> climb 3 m/s / descend 2 m/s,
    x1 -> yaw rate 60 deg/s. Centred sticks = PX4 holds position and altitude.

Pure Python (no Isaac Sim) so it can be flown offline against a PX4-like model (simulate_offline)
to check a mission's geometry before spending GPU time on it.

Beat types:
  takeoff   {agl}                                   climb straight up
  goto      {xy, agl | z, label, speed, radius}     turn to face, fly there
  orbit     {center(xyz), radius, z, laps, dir}     circle a point facing it (dir +1 = ccw)
  hover     {s, look_at}                            hold, face a point
  descend   {agl | z}                               change height in place
  land      {xy, label}                             fly over a spot, descend, touch down
"""
import math

DECIDE_EVERY_S = 0.5
STICK_STEP = 0.1
V_XY_MAX = 8.0
V_UP_MAX, V_DN_MAX = 3.0, 2.0
YAW_RATE_MAX = math.radians(60.0)

GIMBAL_TRANSIT_DEG = 15.0
# Take-off and landing look FORWARD (5 deg down, horizon in frame), not at the ground: the 65 deg
# landing pitch of Easy/Forest/Simulation_1 ended the video on a close-up of the grass (user feedback
# 2026-10-09). Simulation_1 itself is kept as published.
GIMBAL_GROUND_DEG = 5.0
GIMBAL_LAND_DEG = GIMBAL_GROUND_DEG
GIMBAL_MAX_DEG = 80.0
GIMBAL_RATE_DPS = 35.0


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _q(v):
    v = max(-1.0, min(1.0, v))
    return round(round(v / STICK_STEP) * STICK_STEP, 2) + 0.0


def compass(dx, dy):
    """ENU vector -> 8-point compass word (x = east, y = north)."""
    names = ["east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east"]
    return names[int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8]


class Pilot:
    def __init__(self, beats, ground_fn):
        self.beats = beats
        self.ground = ground_fn
        self.i = 0
        self.sticks = (0.0, 0.0, 0.0, 0.0)
        self.next_decision = 0.0
        self.beat_t0 = None
        self.beat_state = {}
        self.gimbal = GIMBAL_GROUND_DEG
        self.events = []
        self.done = False
        self.failed = None  # set to a reason string if a beat overruns its time budget

    @property
    def beat(self):
        return self.beats[min(self.i, len(self.beats) - 1)]

    # ---- helpers ---------------------------------------------------------------------------
    def _path_floor(self, x0, y0, x1, y1, n=24):
        return max(self.ground(x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n) for k in range(n + 1))

    def _alt_sticks(self, z, target_z):
        ez = target_z - z
        return ez / 4.0 if ez > 0 else ez / 3.0

    def _yaw_stick(self, yaw, desired, ff=0.0):
        e = _wrap(desired - yaw)
        # x1 + = clockwise = yaw decreasing in ENU
        return -(e / (YAW_RATE_MAX * 0.8)) + ff

    def _body(self, yaw, vx, vy):
        fwd = vx * math.cos(yaw) + vy * math.sin(yaw)
        right = vx * math.sin(yaw) - vy * math.cos(yaw)
        return fwd, right

    def _advance(self, t, s, note=None):
        b = self.beat
        ev = {"beat": self.i, "type": b["type"], "label": b.get("label"), "t_end": round(t, 3),
              "t_start": round(self.beat_t0, 3), "pose": [round(s["x"], 2), round(s["y"], 2), round(s["z"], 2)]}
        if note:
            ev.update(note)
        self.events.append(ev)
        self.i += 1
        self.beat_t0 = t
        self.beat_state = {}
        if self.i >= len(self.beats):
            self.done = True

    # ---- main --------------------------------------------------------------------------------
    def update(self, t, s, dt):
        """s: x y z yaw vx vy vz landed armed. Returns (sticks, gimbal_pitch_deg, beat_label)."""
        if self.beat_t0 is None:
            self.beat_t0 = t
        if not self.done and t + 1e-9 >= self.next_decision:
            self.next_decision = t + DECIDE_EVERY_S
            raw, look = self._decide(t, s)
            if raw is not None:
                self.sticks = tuple(_q(v) for v in raw)
            self._look = look
        look = getattr(self, "_look", ("pitch", GIMBAL_TRANSIT_DEG))
        if look[0] == "point":
            px, py, pz = look[1]
            d = math.hypot(px - s["x"], py - s["y"])
            want = math.degrees(math.atan2(s["z"] - pz, max(d, 0.5)))
        else:
            want = look[1]
        want = max(0.0, min(GIMBAL_MAX_DEG, want))
        step = GIMBAL_RATE_DPS * dt
        self.gimbal += max(-step, min(step, want - self.gimbal))
        return self.sticks, self.gimbal, self.beat.get("label") or self.beat["type"]

    def _budget(self, b, s):
        """Generous time budget per beat; overrunning it means the flight is stuck, not slow."""
        typ = b["type"]
        if typ == "takeoff":
            return 40.0 + b["agl"] / 1.5
        if typ == "goto":
            return 40.0 + math.hypot(b["xy"][0] - s["x"], b["xy"][1] - s["y"]) / 2.0
        if typ == "orbit":
            return 30.0 + 1.8 * b.get("laps", 1.0) * 2 * math.pi * b["radius"] / b.get("speed", 3.0)
        if typ == "hover":
            return b["s"] + 20.0
        if typ == "descend":
            # PX4 Position mode sinks ~0.7 m/s here: scale with the height to lose (a 60 m cruise needs > 40 s,
            # Coastal flight 009 v2.1 was aborted as "stuck" while descending normally)
            drop = max(0.0, s["z"] - (b["z"] if "z" in b else self.ground(s["x"], s["y"]) + b["agl"]))
            return 40.0 + drop / 0.6
        if typ == "land":
            return 150.0
        return 120.0

    def _decide(self, t, s):
        b = self.beat
        typ = b["type"]
        if "budget" not in self.beat_state:
            # budget_extra: missions with large height changes inside one beat (Mountain 2026-10-20) add time; others unchanged
            self.beat_state["budget"] = self._budget(b, s) + b.get("budget_extra", 0.0)
        if t - self.beat_t0 > self.beat_state["budget"]:
            self.failed = f"beat {self.i} ({b.get('label') or typ}) exceeded {self.beat_state['budget']:.0f}s"
            self.done = True
            return (0.0, 0.0, 0.0, 0.0), ("pitch", GIMBAL_TRANSIT_DEG)
        x, y, z, yaw = s["x"], s["y"], s["z"], s["yaw"]
        g_here = self.ground(x, y)
        fwd_v, right_v = self._body(yaw, s["vx"], s["vy"])
        bs = self.beat_state

        if typ == "takeoff":
            target_z = g_here + b["agl"]
            if z >= target_z - 0.6:
                self._advance(t, s)
                return self._decide(t, s) if not self.done else ((0, 0, 0, 0), ("pitch", GIMBAL_TRANSIT_DEG))
            return (0.0, max(0.5, self._alt_sticks(z, target_z)), -right_v / V_XY_MAX, -fwd_v / V_XY_MAX), ("pitch", GIMBAL_GROUND_DEG)

        if typ == "goto":
            tx, ty = b["xy"]
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if "z" in b:
                target_z = b["z"]
            elif b.get("follow_terrain"):
                # Mountain (2026-10-20): height above the ground of the NEXT 200 m of the route, re-evaluated at every decision,
                # so a long transit climbs over a ridge and comes back down after it instead of holding the highest point
                look = min(dist, 200.0)
                ex, ey = x + dx / max(dist, 1e-6) * look, y + dy / max(dist, 1e-6) * look
                target_z = self._path_floor(x, y, ex, ey, n=12) + b.get("agl", 25.0)
            else:
                if "floor" not in bs:
                    bs["floor"] = self._path_floor(x, y, tx, ty)
                target_z = bs["floor"] + b.get("agl", 25.0)
            radius = b.get("radius", 3.0)
            if dist < radius and abs(target_z - z) < 1.5:
                self._advance(t, s, {"arrival_error_m": round(dist, 2)})
                return self._decide(t, s)
            heading = math.atan2(dy, dx)
            if dist < max(radius, 4.0):
                # already over the waypoint (only the height is left): the bearing to a point a few cm away is noise, and
                # steering at it spun the drone to a random heading for the whole climb (Mountain S002 faced away from
                # the pass for 20 s, 2026-10-21) -- face the look-at point, else keep the current heading
                if "look_at" in b:
                    heading = math.atan2(b["look_at"][1] - y, b["look_at"][0] - x)
                elif "heading" in bs:
                    heading = bs["heading"]
                else:
                    heading = yaw
            else:
                bs["heading"] = heading
            e = _wrap(heading - yaw)
            vmax = b.get("speed", 0.8) * V_XY_MAX
            v_des = min(vmax, math.sqrt(2 * 1.2 * max(dist - 0.5 * radius, 0.0)))
            if abs(e) > math.radians(35):
                v_des = min(v_des, 1.0)
            # climb first if well below the transit height (don't plough through the canopy)
            if target_z - z > 6.0 and not b.get("steady_climb"):
                v_des = min(v_des, 1.5)
            elif target_z - z > 6.0:
                # a route that rises steadily (Mountain: up a cable-car line / a ridge), every point of it already clear of the
                # ground: climb and advance together, at the speed the climb rate allows
                v_des = min(v_des, max(1.5, V_UP_MAX * 0.9 * dist / max(target_z - z, 1e-3)))
            if dist < max(radius, 4.0):
                # over the waypoint: small corrections toward it in the body frame, whatever the nose faces
                e_mv = _wrap(math.atan2(dy, dx) - yaw)
                v_des = min(v_des, 1.0)
                fwd = v_des * math.cos(e_mv) - 0.3 * fwd_v * (dist < 1.0)
                lat = v_des * math.sin(-e_mv) - 0.6 * right_v
            else:
                fwd = v_des * math.cos(e)
                lat = v_des * math.sin(-e) - 0.6 * right_v
            look = ("point", b["look_at"]) if "look_at" in b else ("pitch", b.get("gimbal", GIMBAL_TRANSIT_DEG))
            return (self._yaw_stick(yaw, heading), self._alt_sticks(z, target_z), lat / V_XY_MAX, fwd / V_XY_MAX), look

        if typ == "orbit":
            cx, cy, cz = b["center"]
            r = b["radius"]
            d = b.get("dir", 1)
            v_t = b.get("speed", 3.0)
            ang = math.atan2(y - cy, x - cx)
            if "prev" not in bs:
                bs["prev"], bs["swept"] = ang, 0.0
            bs["swept"] += d * _wrap(ang - bs["prev"])
            bs["prev"] = ang
            if bs["swept"] >= 2 * math.pi * b.get("laps", 1.0):
                self._advance(t, s, {"swept_deg": round(math.degrees(bs["swept"]), 1)})
                return self._decide(t, s)
            dist = math.hypot(x - cx, y - cy)
            face = math.atan2(cy - y, cx - x)
            ff = -d * (v_t / r) / YAW_RATE_MAX  # ccw orbit facing centre = yaw increasing = x1 negative
            radial = (dist - r) * 0.4
            return (self._yaw_stick(yaw, face, ff), self._alt_sticks(z, b["z"]), d * v_t / V_XY_MAX,
                    max(-0.4, min(0.4, radial / V_XY_MAX * 4))), ("point", (cx, cy, cz))

        if typ == "hover":
            lx, ly, lz = b["look_at"]
            if b.get("align_first") and "t_aligned" not in bs:
                # the hover clock starts only once the drone faces the target (within 10 deg), so "hover N s looking
                # at it" is true for the whole N s (Easy-data review item #3); Coastal missions set align_first
                if abs(_wrap(math.atan2(ly - y, lx - x) - yaw)) < math.radians(10.0):
                    bs["t_aligned"] = t
            t_ref = bs.get("t_aligned", self.beat_t0 if not b.get("align_first") else None)
            if t_ref is not None and t - t_ref >= b["s"]:
                self._advance(t, s, {"aligned_at": round(t_ref, 3)} if b.get("align_first") else None)
                return self._decide(t, s)
            if "z" not in bs:
                bs["z"] = z
            return (self._yaw_stick(yaw, math.atan2(ly - y, lx - x)), self._alt_sticks(z, bs["z"]),
                    -right_v / V_XY_MAX, -fwd_v / V_XY_MAX), ("point", (lx, ly, lz))

        if typ == "descend":
            target_z = b["z"] if "z" in b else g_here + b["agl"]
            if abs(target_z - z) < 0.6:
                self._advance(t, s)
                return self._decide(t, s)
            look = ("point", b["look_at"]) if "look_at" in b else ("pitch", b.get("gimbal", 30.0))
            return (0.0, self._alt_sticks(z, target_z), -right_v / V_XY_MAX, -fwd_v / V_XY_MAX), look

        if typ == "land":
            tx, ty = b["xy"]
            dx, dy = tx - x, ty - y
            dist = math.hypot(dx, dy)
            if s.get("landed") and not s.get("armed", True):
                self._advance(t, s, {"touchdown_error_m": round(dist, 2)})
                return (0.0, 0.0, 0.0, 0.0), ("pitch", GIMBAL_LAND_DEG)
            if s.get("landed"):
                return (0.0, -1.0, 0.0, 0.0), ("pitch", GIMBAL_LAND_DEG)
            agl = z - self.ground(x, y) - 0.227
            # body-frame position correction (no turning -- a pilot crabs onto the pad)
            pf = dx * math.cos(yaw) + dy * math.sin(yaw)
            pr = dx * math.sin(yaw) - dy * math.cos(yaw)
            v_f = max(-2.0, min(2.0, 0.5 * pf)) - 0.5 * fwd_v
            v_r = max(-2.0, min(2.0, 0.5 * pr)) - 0.5 * right_v
            # descend once roughly over the pad and keep correcting on the way down; waiting for a
            # tight radius first is what stalled the first forest landing at 2.6 m
            if dist > 3.5:
                vz = 0.0 if agl < 25 else -0.3
            else:
                vz = -0.8 if agl > 4.0 else -0.35
            return (0.0, vz, v_r / V_XY_MAX, v_f / V_XY_MAX), ("pitch", GIMBAL_LAND_DEG)

        raise ValueError(f"unknown beat type {typ}")


def simulate_offline(beats, ground_fn, start_xy, start_yaw=0.0, dt=1 / 30, max_s=600.0):
    """Fly the pilot against a first-order model of PX4 Position mode. Returns (frames, pilot)."""
    p = Pilot(beats, ground_fn)
    x, y = start_xy
    z = ground_fn(x, y) + 0.227
    yaw = start_yaw
    vx = vy = vz = 0.0
    landed, armed, land_timer = True, True, 0.0
    frames = []
    t = 0.0
    while t < max_s and not p.done:
        s = dict(x=x, y=y, z=z, yaw=yaw, vx=vx, vy=vy, vz=vz, landed=landed, armed=armed)
        (x1, y1, x2, y2), gimbal, label = p.update(t, s, dt)
        cf, cr = y2 * V_XY_MAX, x2 * V_XY_MAX
        tvx = cf * math.cos(yaw) + cr * math.sin(yaw)
        tvy = cf * math.sin(yaw) - cr * math.cos(yaw)
        tvz = y1 * (V_UP_MAX if y1 > 0 else V_DN_MAX)
        a = dt / 0.6
        vx += (tvx - vx) * a
        vy += (tvy - vy) * a
        vz += (tvz - vz) * dt / 0.3
        yaw = _wrap(yaw - x1 * YAW_RATE_MAX * dt)
        floor = ground_fn(x, y) + 0.227
        if landed:
            vx = vy = 0.0
            if y1 > 0.3:
                landed = False
            vz = max(0.0, vz)
        x += vx * dt
        y += vy * dt
        z += vz * dt
        if z <= floor:
            z = floor
            if not landed and vz < 0 and t > 2:
                landed = True
        if landed and t > 2:
            land_timer += dt
            if land_timer > 2.0:
                armed = False
        else:
            land_timer = 0.0
        frames.append(dict(t=t, x=x, y=y, z=z, yaw=yaw, sticks=(x1, y1, x2, y2), gimbal=gimbal, beat=label,
                           agl=z - floor))
        t += dt
    return frames, p
