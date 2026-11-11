"""Offline UNIT test of the contact-rule logic only (fake PhysX reports, no simulator). The collision behaviour itself is
verified by the real simulated flights (tests/check_tests.py); this only guards the classification code."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import contact_monitor as cm  # noqa: E402

R = "/World/x500"
PATHS = {1: f"{R}/body", 2: f"{R}/body/collision/skid_left", 3: f"{R}/body/collision_proxies/rotor_disc_2",
         4: f"{R}/rotor1/collision", 5: f"{R}/body/collision/top_plate", 10: "/World/Terrain", 11: "/World/Terrain",
         12: "/World/CollisionProxies/Nature_Trees/i3_0", 13: f"{R}/rotor1"}
LAND = {"pad": "W", "xy": [-120.0, -45.0], "radius_m": 2.5, "surface_colliders": ["/World/Terrain"]}


def mon(landing=None):
    m = cm.ContactMonitor.__new__(cm.ContactMonitor)
    m.root, m.landing, m.log = R, landing, lambda *a: None
    m._names, m._int_to_path = {}, lambda i: PATHS[i]
    m.colliders = {p: cm.part_class(p, R) for p in PATHS.values() if p.startswith(R + "/")}
    m._pending, m.steps_seen, m.first_step, m.last_step, m.reports = [], 0, None, None, 0
    m.pairs, m.events, m.latched, m.invalid, m.init_contact_seen, m.touching_now, m.allowed_landing = {}, [], None, None, False, 0, None
    return m


def pt(x, y, sep=0.0, imp=0.1):
    return ((x, y, 0.0), (0, 0, 1), (0, 0, imp), sep)


cases = []
m = mon(LAND)
land = {"type": "land", "pad": "W"}
m._pending = [(0, 1, 11, 2, 11, 0xFFFFFFFF, 0xFFFFFFFF, [pt(0, 0)])]
m.drain(1, None, {"phase": "initialization"})
cases.append(("resting gear in initialization", m.init_contact_seen and m.latched is None))
m._pending = [(0, 1, 13, 5, 4, 0, 0, [pt(0, 0)])]
m.drain(2, 0.0, {"phase": "episode"})
cases.append(("internal pair ignored", m.latched is None and m.reports == 2))
m._pending = [(0, 1, 11, 2, 11, 0, 0, [pt(-119.0, -45.5)])]
m.drain(3, 1.0, {"phase": "episode", "beat": land})
cases.append(("gear on designated pad during land beat: allowed", m.allowed_landing is not None and m.latched is None))
m._pending = [(0, 1, 11, 2, 11, 0, 0, [pt(-120.0, -45.0, sep=0.03, imp=0.0)])]
m.drain(4, 1.1, {"phase": "episode"})
cases.append(("contact-offset only (separation 3 cm, no impulse): proximity", m.latched is None))
m._pending = [(0, 1, 11, 3, 11, 0, 0, [pt(-120.0, -45.0)])]
m.drain(5, 1.2, {"phase": "episode", "beat": land})
cases.append(("rotor on the designated pad: prohibited", m.latched is not None and m.latched["drone_part"] == "rotor"))
m2 = mon(LAND)
m2._pending = [(0, 1, 11, 2, 11, 0, 0, [pt(-112.0, -45.0)])]
m2.drain(6, 2.0, {"phase": "episode", "beat": land})
cases.append(("gear 8 m off the pad: prohibited", m2.latched is not None))
m3 = mon(LAND)
m3._pending = [(0, 1, 11, 2, 11, 0, 0, [pt(-120.0, -45.0)])]
m3.drain(7, 2.0, {"phase": "episode", "beat": {"type": "goto"}})
cases.append(("gear on the pad but not in the land beat: prohibited", m3.latched is not None))
m4 = mon(None)
m4._pending = [(0, 1, 12, 3, 12, 0, 5, [pt(0, 0)])]
m4.drain(8, 3.0, {"phase": "pre-roll"})
cases.append(("tree contact in the pre-roll: invalid, not latched", m4.invalid is not None and m4.latched is None))
m5 = mon(None)
m5._pending = [(0, 1, 11, 2, 11, 0, 0, [pt(0, 0)])]
m5.drain(9, 3.0, {"phase": "initialization"})
m5._pending = [(0, 1, 11, 5, 11, 0, 0, [pt(0, 0)])]
m5.drain(10, 3.0, {"phase": "initialization"})
cases.append(("airframe on the ground in initialization: invalid", m5.invalid is not None))
ok = True
for n, r in cases:
    print("PASS" if r else "FAIL", n)
    ok &= bool(r)
sys.exit(0 if ok else 1)
