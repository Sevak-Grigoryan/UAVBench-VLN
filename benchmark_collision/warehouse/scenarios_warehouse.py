"""
Collision TEST scenarios in the warehouse (Isaac Sim warehouse template, see warehouse_env.py), flown for real (PX4 SITL +
Isaac Sim PhysX) through run_flight.py --env Warehouse. Same structure as scenarios.py (Forest): take off and settle
(pre-roll), then the EVALUATED part (phase "episode"), which ends at the first prohibited contact.
Nothing is spawned: the obstacles are the warehouse's own racks and walls (positions from warehouse_layout.json, GPU probe).

  shelf             face rack Shelf_0 from the aisle, fly forward at ~1.3 m/s into its boxes  -> prohibited, other = Shelf_0
  wall              fly forward at ~1.35 m/s into the east wall (x = 9.21)                     -> prohibited, other = east wall
  propeller_strike  hover beside rack Shelf_1 at its beam height, slide sideways slowly (stick 0.15, just past PX4's 0.1 hold
                    deadzone) until a rotor touches the shelf beam                           -> prohibited, first part = rotor
  ceiling           take off to ~5.4 m in a column clear of lamps and roof beams, climb slowly (~0.5 m/s) into the
                    ceiling plate (z 9 m)                                                    -> prohibited, other = ceiling plate
  wall_post         = wall, but after the first contact physics and PX4 run on 3 s and are recorded, with PX4's official
                    Failure Detector enabled (CBRK_FLIGHTTERM = 0)                           -> prohibited, then what really happens
  wall_fast_post    the same from 15 m further back at stick 0.6 (much faster), same 3 s post-contact window
The rotor is modelled as a disc collider (collision_proxies.add_rotor_discs): the contact is detected and stops the run;
blade breakage itself is not simulated.
"""
import math

PRE, EPISODE = "pre-roll", "episode"
NAMES = ("shelf", "wall", "propeller_strike", "ceiling", "wall_post", "wall_fast_post")
TAKEOFF_AGL = 2.4            # the take-off beat ends 0.6 m below its target: the drone settles at ~2 m, the height of the boxes
                             # on the racks' first shelf level (z 1.3-2.3 m)

# start (= take-off point on the floor), heading, evaluated sticks (x1 yaw, y1 throttle, x2 right, y2 forward), seconds
SETUP = {
    "shelf": dict(start_xy=(-5.0, 6.2), yaw_deg=180.0, sticks=(0.0, 0.0, 0.0, 0.2), s=10.0,
                  label="fly forward into rack Shelf_0",
                  target="rack Shelf_0 (boxes on the first shelf level, front face x = -8.97, shelf plank edge x = -8.78)"),
    "wall": dict(start_xy=(4.5, -6.3), yaw_deg=0.0, sticks=(0.0, 0.0, 0.0, 0.25), s=10.0,
                 label="fly forward into the east wall",
                 target="east wall SM_WallA_6M15 (inner face x = 9.21)"),
    # climb_s 0.5: after the usual take-off (settles at ~2.38 m) a 0.5 s throttle pulse of 0.2 (~0.5 m/s, Forest rotor_ceiling
    # telemetry) lifts the drone ~0.2 m, so the rotor discs (body + 0.05..0.07 m, reaching 0.31 m sideways) are at the height
    # of Shelf_1's second-level beam (z 2.455-2.825, front edge x = -0.858): any body height 2.40-2.76 m works, and the rotor
    # reaches the beam 0.36 m before the legs could reach the boxes (face x = -0.67).
    # Earlier tries: 2.38 m (job 381707) -> the rotors passed 1 mm under the beam and the right LEG touched a box;
    # take-off agl 2.58 (job 381792) -> the take-off overshot to 3.03 m and the top plate touched a pallet.
    "propeller_strike": dict(start_xy=(-2.6, 5.4), yaw_deg=90.0, sticks=(0.0, 0.0, 0.15, 0.0), s=12.0, climb_s=0.5,
                             label="slide right slowly into rack Shelf_1",
                             target="front beam of rack Shelf_1's second level (z 2.455-2.825 m, edge x = -0.858), "
                                    "on the drone's right side, at rotor height"),
    # (5.2, -6.7): >= 2 m from every lamp, roof beam, pillar and rack (probe warehouse_meshes.json), >= 0.7 m from the edges of
    # ceiling plate SM_CeilingA_6X17 (x -0.05..5.95, y -12..-6, z 9). Take-off agl 6 ends ~0.6 m short (~5.4 m), so the
    # evaluated climb starts below the lamps (z 5.83) and has ~3.3 m to go.
    "ceiling": dict(start_xy=(5.2, -6.7), yaw_deg=0.0, agl=6.0, sticks=(0.0, 0.2, 0.0, 0.0), s=15.0,
                    label="climb slowly into the ceiling",
                    target="ceiling plate SM_CeilingA_6X17 (z 9.0 m), right above the drone"),
    "wall_post": dict(start_xy=(4.5, -6.3), yaw_deg=0.0, sticks=(0.0, 0.0, 0.0, 0.25), s=10.0, post_contact_s=3.0,
                      label="fly forward into the east wall",
                      target="east wall SM_WallA_6M15 (inner face x = 9.21)"),
    # the corridor y = -6.3 is free at drone height from x = -7.5 to the wall (probe warehouse_meshes.json)
    "wall_fast_post": dict(start_xy=(-6.0, -6.3), yaw_deg=0.0, sticks=(0.0, 0.0, 0.0, 0.6), s=15.0, post_contact_s=3.0,
                           label="fly forward fast into the east wall",
                           target="east wall SM_WallA_6M15 (inner face x = 9.21), 15 m away"),
}
DESCRIPTION = {
    "shelf": "face a storage rack from the aisle and fly forward (~1.3 m/s, stick 0.2) into the boxes on it",
    "wall": "fly forward (~1.35 m/s, stick 0.25) along a free corridor into the warehouse's east wall",
    "propeller_strike": "hover next to a rack at the height of its shelf beam and slide sideways slowly (~0.4 m/s, stick 0.15) until a propeller "
                        "touches the beam",
    "ceiling": "climb straight up slowly (~0.5 m/s, stick 0.2) inside the closed warehouse until the drone touches the ceiling",
    "wall_post": "fly forward (stick 0.25) into the east wall; after the contact the simulation runs on 3 s with PX4's Failure "
                 "Detector enabled, to record what really happens next",
    "wall_fast_post": "fly forward fast (stick 0.6, 15 m run-up) into the east wall; after the contact the simulation runs on 3 s "
                      "with PX4's Failure Detector enabled",
}
EXPECTED = {
    "shelf": {"outcome": "collision", "first_prohibited_other_prefix": "/Root/Shelf_0"},
    "wall": {"outcome": "collision", "first_prohibited_other_prefix": "/Root/SM_WallA_6M15"},
    "propeller_strike": {"outcome": "collision", "first_prohibited_part": "rotor", "first_prohibited_other_prefix": "/Root/Shelf_1"},
    "ceiling": {"outcome": "collision", "first_prohibited_other_prefix": "/Root/SM_CeilingA_6X17"},
    "wall_post": {"outcome": "collision", "first_prohibited_other_prefix": "/Root/SM_WallA_6M15"},
    "wall_fast_post": {"outcome": "collision", "first_prohibited_other_prefix": "/Root/SM_WallA_6M15"},
}


class Hooks:
    """run_flight.py calls on_frame / on_step; nothing to spawn here, only notes for the record."""

    def __init__(self, name, env):
        self.name, self.env = name, env
        self.fixtures = []
        self.notes = {"obstacle": SETUP[name]["target"], "fixtures": []}
        self._noted = False

    def on_frame(self, rt):
        if rt.phase == EPISODE and not self._noted:
            self._noted = True
            self.notes["drone_at_episode_start"] = [round(v, 3) for v in rt.pos] + [round(math.degrees(rt.yaw), 2)]
            rt.log(f"[test] {self.name}: episode starts at {self.notes['drone_at_episode_start']}")

    def on_step(self, rt):
        return


def build(name, env):
    if name not in NAMES:
        raise ValueError(f"unknown warehouse scenario {name}; one of {NAMES}")
    c = SETUP[name]
    beats = [dict(type="takeoff", agl=c.get("agl", TAKEOFF_AGL), label="take off", phase=PRE),
             dict(type="sticks", sticks=(0.0, 0.0, 0.0, 0.0), s=4.0, label="settle", phase=PRE)]
    if c.get("climb_s"):
        beats += [dict(type="sticks", sticks=(0.0, 0.2, 0.0, 0.0), s=c["climb_s"], label="climb to beam height", phase=PRE),
                  dict(type="sticks", sticks=(0.0, 0.0, 0.0, 0.0), s=3.0, label="settle", phase=PRE)]
    beats += [
             dict(type="sticks", sticks=c["sticks"], s=c["s"], label=c["label"], phase=EPISODE)]
    m = {"id": f"colltest_{name}", "start": "warehouse_floor", "start_xy": c["start_xy"], "yaw_deg": c["yaw_deg"],
         "beats": beats, "tasks": [], "landing": None,
         "instruction": f"collision test scenario '{name}' in the warehouse (not a benchmark episode)",
         "test": {"name": name, "expected": EXPECTED[name]},
         "post_contact_s": c.get("post_contact_s", 0.0), "px4_failure_detector": bool(c.get("post_contact_s"))}
    return m, Hooks(name, env)
