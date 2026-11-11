"""
PhysX contact monitor of the X500: every UAV-environment contact, at every physics step, judged by the collision rule.

    mon = ContactMonitor(stage, "/World/x500", landing=spec)     # before world.reset(): applies the contact-report API
    ... world.step(render=False); step += 1
    mon.drain(step, t, ctx)                                       # after EVERY physics step, outside the callback
    if mon.latched: terminate

Capture: omni.physx get_physx_simulation_interface().subscribe_contact_report_events (Isaac Sim 5.1, omni.physx 107.3).
The callback only copies the raw headers / points of the step into a list (PhysX reuses the buffers after the callback);
decoding paths, classifying and judging happen in drain(), called by the runner after each world.step().

Rule (collision-v1):
  * internal pairs (drone part vs drone part) are ignored;
  * a contact "touches" when any of its points has separation <= TOUCH_SEPARATION_M or a non-zero impulse (PhysX also reports
    pairs that are only inside the contact offset; those are kept as "proximity" and never fail an episode);
  * phase "initialization" (spawn -> lift-off): landing gear resting on the take-off surface is expected; anything else
    there is an "initialization violation" (the episode is invalid, not failed by the agent);
  * phase "episode": every touching contact is PROHIBITED, except landing-gear contact with a designated landing surface
    collider inside the designated pad footprint, during the land beat of the designated pad;
  * phase "pre-roll" (after lift-off) / "post-roll": recorded; a pre-roll contact makes the episode invalid.
The first prohibited contact is latched; the runner terminates right after that physics step.
"""
import math

TOUCH_SEPARATION_M = 0.005
CONTACT_OFFSET_M = 0.04          # drone colliders: speculative margin > one 240 Hz step at 8 m/s (3.3 cm)
REST_OFFSET_M = 0.0
MAX_STORED_PER_PAIR = 3          # full records kept per (pair, verdict, phase); the rest is counted
VERSION = "collision-v1"

GEAR = ("skid_left", "skid_right", "leg_left", "leg_right")


def part_class(collider_path, root):
    """landing_gear / rotor / airframe for a drone collider path (unknown drone colliders count as airframe)."""
    rel = collider_path[len(root):]
    name = rel.rsplit("/", 1)[-1]
    if name in GEAR:
        return "landing_gear"
    if "rotor_disc" in name or "/rotor" in rel:
        return "rotor"
    return "airframe"


class ContactMonitor:
    def __init__(self, stage, root, landing=None, log=print):
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema, PhysicsSchemaTools, Usd, UsdPhysics

        self.root = root.rstrip("/")
        self.landing = landing
        self.log = log
        self._int_to_path = PhysicsSchemaTools.intToSdfPath
        self._names = {}
        self.bodies, self.colliders = [], {}
        for prim in Usd.PrimRange(stage.GetPrimAtPath(self.root)):
            p = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                cr = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                cr.CreateThresholdAttr().Set(0.0)
                self.bodies.append(p)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                pc = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                pc.CreateContactOffsetAttr().Set(CONTACT_OFFSET_M)
                pc.CreateRestOffsetAttr().Set(REST_OFFSET_M)
                self.colliders[p] = part_class(p, self.root)
        if not self.bodies or not self.colliders:
            raise RuntimeError(f"no rigid bodies / colliders under {self.root}: contact monitoring impossible")
        self._pending = []
        self._sub = get_physx_simulation_interface().subscribe_contact_report_events(self._on_report)
        self.steps_seen = 0
        self.first_step = self.last_step = None
        self.reports = 0
        self.pairs = {}                 # (verdict, phase, drone collider, other collider) -> summary
        self.events = []                # full records (first MAX_STORED_PER_PAIR per key)
        self.latched = None             # first prohibited contact
        self.invalid = None             # first initialization / pre-roll violation
        self.init_contact_seen = False  # resting contact on the take-off surface was reported (monitor works)
        self.touching_now = 0
        self.allowed_landing = None     # first allowed landing-gear touchdown

    # ---- callback: copy only -------------------------------------------------------------------------------------------
    def _on_report(self, headers, data):
        for h in headers:
            if int(h.type) == 1:        # CONTACT_LOST
                continue
            o, n = h.contact_data_offset, h.num_contact_data
            pts = [((d.position[0], d.position[1], d.position[2]), (d.normal[0], d.normal[1], d.normal[2]),
                    (d.impulse[0], d.impulse[1], d.impulse[2]), d.separation) for d in (data[i] for i in range(o, o + n))]
            self._pending.append((int(h.type), h.actor0, h.actor1, h.collider0, h.collider1, h.proto_index0,
                                  h.proto_index1, pts))

    def _path(self, i):
        p = self._names.get(i)
        if p is None:
            p = self._names[i] = str(self._int_to_path(i))
        return p

    def close(self):
        self._sub = None

    # ---- judge (outside the callback) ----------------------------------------------------------------------------------
    def _verdict(self, cls, other, pts, ctx):
        phase = ctx["phase"]
        if phase == "initialization":
            return "initialization" if cls == "landing_gear" else "initialization_violation"
        if phase == "post-termination":
            return "post_termination"
        if phase == "pre-roll":
            return "pre_roll_contact"
        if phase == "post-roll":
            return "post_roll"
        L, beat = self.landing, ctx.get("beat") or {}
        if (L and cls == "landing_gear" and beat.get("type") == "land" and beat.get("pad") == L["pad"]
                and any(other == s or other.startswith(s + "/") for s in L["surface_colliders"])
                and all(math.hypot(p[0][0] - L["xy"][0], p[0][1] - L["xy"][1]) <= L["radius_m"] for p in pts)):
            return "allowed_landing"
        return "prohibited"

    def drain(self, step, t, ctx):
        """Judge the contacts of physics step `step` (t = s since recording started, None before). ctx: phase, beat."""
        pending, self._pending = self._pending, []
        self.steps_seen += 1
        self.first_step = step if self.first_step is None else self.first_step
        self.last_step = step
        touching = []
        for typ, a0, a1, c0, c1, pi0, pi1, pts in pending:
            self.reports += 1
            c0p, c1p = self._path(c0), self._path(c1)
            d0, d1 = c0p.startswith(self.root + "/"), c1p.startswith(self.root + "/")
            if d0 and d1:
                continue                                  # internal UAV pair
            if not (d0 or d1):
                continue                                  # environment-environment (props on terrain)
            if d0:
                mine, other, actor, proto, sign = c0p, c1p, self._path(a1), pi1, 1.0
            else:
                mine, other, actor, proto, sign = c1p, c0p, self._path(a0), pi0, -1.0
            touch = any(p[3] <= TOUCH_SEPARATION_M or (p[2][0] or p[2][1] or p[2][2]) for p in pts)
            cls = self.colliders.get(mine, part_class(mine, self.root))
            verdict = self._verdict(cls, other, pts, ctx) if touch else "proximity"
            rec = {"step": step, "t": None if t is None else round(t, 6), "phase": ctx["phase"], "beat": ctx.get("beat_index"),
                   "beat_label": (ctx.get("beat") or {}).get("label"), "event": ("found", "lost", "persist")[typ],
                   "verdict": verdict, "drone_part": cls, "drone_collider": mine, "other_collider": other,
                   "other_actor": actor, "other_proto_index": None if proto in (None, 0xFFFFFFFF) else int(proto),
                   "points": [{"position": [round(v, 4) for v in p[0]], "normal": [round(sign * v, 4) for v in p[1]],
                               "impulse": [round(v, 6) for v in p[2]], "separation": round(p[3], 5)} for p in pts],
                   "total_impulse_ns": round(sum(math.sqrt(sum(v * v for v in p[2])) for p in pts), 6)}
            key = (verdict, ctx["phase"], mine, other)
            s = self.pairs.get(key)
            if s is None:
                s = self.pairs[key] = {"verdict": verdict, "phase": ctx["phase"], "drone_collider": mine, "drone_part": cls,
                                       "other_collider": other, "first_step": step, "first_t": rec["t"], "steps": 0,
                                       "max_impulse_ns": 0.0}
            s["steps"] += 1
            s["last_step"], s["last_t"] = step, rec["t"]
            s["max_impulse_ns"] = max(s["max_impulse_ns"], rec["total_impulse_ns"])
            if s["steps"] <= MAX_STORED_PER_PAIR:
                self.events.append(rec)
            if touch:
                touching.append(rec)
            if verdict == "initialization":
                self.init_contact_seen = True
            elif verdict == "allowed_landing" and self.allowed_landing is None:
                self.allowed_landing = rec
            elif verdict in ("initialization_violation", "pre_roll_contact") and self.invalid is None:
                self.invalid = rec
                self.log(f"[contact] INVALID ({verdict}) step {step} t={t} {cls} {mine} vs {other}")
            elif verdict == "prohibited" and self.latched is None:
                self.latched = rec
                self.log(f"[contact] PROHIBITED step {step} t={t} {cls} {mine} vs {other} "
                         f"impulse {rec['total_impulse_ns']:.4f} N s")
        self.touching_now = len(touching)
        return touching

    def summary(self, step_to_t=None):
        """step_to_t(step) -> time relative to the start of recording; fills records drained before recording started."""
        if step_to_t is not None:
            for r in self.events + [self.latched, self.invalid, self.allowed_landing]:
                if r is not None and r.get("t") is None:
                    r["t"] = round(step_to_t(r["step"]), 6)
            for s in self.pairs.values():
                for k in ("first", "last"):
                    if s.get(f"{k}_t") is None:
                        s[f"{k}_t"] = round(step_to_t(s[f"{k}_step"]), 6)
        return {
            "version": VERSION,
            "api": "omni.physx get_physx_simulation_interface().subscribe_contact_report_events (PhysxContactReportAPI, threshold 0)",
            "drone_root": self.root, "drone_bodies": self.bodies, "drone_colliders": self.colliders,
            "touch_separation_m": TOUCH_SEPARATION_M, "drone_contact_offset_m": CONTACT_OFFSET_M,
            "drone_rest_offset_m": REST_OFFSET_M,
            "steps_monitored": self.steps_seen, "first_step": self.first_step, "last_step": self.last_step,
            "contact_reports": self.reports,
            "monitor_verified": self.init_contact_seen,
            "monitor_verified_by": "the resting landing-gear contact on the take-off surface was reported during initialization",
            "landing_exception": self.landing,
            "first_prohibited": self.latched, "first_invalid": self.invalid, "first_allowed_landing": self.allowed_landing,
            "pairs": sorted(self.pairs.values(), key=lambda s: s["first_step"]),
            "events": self.events,
        }
