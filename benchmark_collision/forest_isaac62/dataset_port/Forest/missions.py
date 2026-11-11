"""
Forest (natural_valley river valley) missions for the Holybro X500 V2.

Landing pads: the two pads Simulations_0_0_4 already had (START_PAD / END_PAD) plus five new ones,
each flattened and kept free of vegetation by forest_scene.PADS (patched in this project's copy of
the forest, see apply_pads()). All pads are dry land >= 35 m from the river centreline.

Landmarks are the ones natural_valley.build_landmarks() actually builds; heights are measured from
the same terrain function the mesh uses, so orbit/hover heights are relative to real geometry.
Transit height is 25 m above the highest ground under the leg -- the tallest trees in this forest
(jacaranda at scale 0.52) are 12.7 m.

Each mission: difficulty, start pad, start heading, beats for pilot.py, and the instruction
(written from the same beat list, so the text and the flight cannot disagree).
"""
import math

# ---- pads (x, y) -------------------------------------------------------------------------------
PADS = {
    "NW": ((-40.0, 45.0), "the north-west clearing"),
    "SE": ((120.0, -55.0), "the south clearing east of the footbridge"),
    "W": ((-120.0, -45.0), "the western clearing"),
    "SW": ((-40.0, -85.0), "the south-west clearing near the survey marker"),
    "CAIRN": ((160.0, -30.0), "the clearing north-west of the stone cairn"),
    "N": ((105.0, 105.0), "the clearing beside the fire lookout"),
    "E": ((225.0, 40.0), "the eastern clearing below the rockslide"),
}


def apply_pads(forest_scene):
    """Extend the forest's flattened, vegetation-free pads with this project's extra pads."""
    forest_scene.PADS = tuple(xy for xy, _ in PADS.values())
    return forest_scene.PADS


# ---- landmarks ---------------------------------------------------------------------------------
def landmarks(nv, seed, ground):
    """name -> dict(xy, top_z, desc). nv = natural_valley (already patched), ground = terrain fn."""
    bx, by = 72.0, nv.river_y(72.0)
    lm = {
        "footbridge": dict(xy=(bx, by), top_z=ground(bx, by) + 3.6 + 1.1,  # deck (forest_scene.BRIDGE_DECK_ABOVE_TERRAIN_M) + rail
                           desc="the timber footbridge that crosses the river", short="the footbridge"),
        "cairn": dict(xy=(180.0, -55.0), top_z=ground(180.0, -55.0) + 4.1,
                      desc="the stone cairn on the south bank", short="the stone cairn"),
        "lookout": dict(xy=(130.0, 85.0), top_z=ground(130.0, 85.0) + 14.3,
                        desc="the wooden fire lookout tower with the orange beacon on the north slope",
                        short="the fire lookout tower"),
        "marker": dict(xy=(-55.0, -95.0), top_z=ground(-55.0, -95.0) + 2.0,
                       desc="the orange survey marker post in the south-west of the valley",
                       short="the survey marker"),
        # the waterfall terrace (forest_scene.WATERFALL_X, 4 m drop); top_z = the lip of the fall
        "waterfall": dict(xy=(262.0, nv.river_y(262.0)), top_z=ground(268.0, nv.river_y(268.0)) + 0.5,
                          desc="the waterfall at the upper end of the river", short="the waterfall"),
        "rockslide": dict(xy=(218.0, 75.0), top_z=ground(218.0, 75.0) + 6.0,
                          desc="the rockslide of fallen boulders on the north slope", short="the rockslide"),
    }
    return lm


def river_points(nv, x_from, x_to, step=40.0):
    n = max(1, int(abs(x_to - x_from) // step))
    return [(x_from + (x_to - x_from) * k / n, nv.river_y(x_from + (x_to - x_from) * k / n)) for k in range(n + 1)]


# ---- missions ----------------------------------------------------------------------------------
def missions(nv, seed, ground):
    L = landmarks(nv, seed, ground)
    P = {k: v[0] for k, v in PADS.items()}

    def orbit(name, radius, above, laps=1.0, d=1, speed=3.0):
        x, y = L[name]["xy"]
        return dict(type="orbit", center=(x, y, L[name]["top_z"]), radius=radius, z=L[name]["top_z"] + above,
                    laps=laps, dir=d, speed=speed, label=f"orbit {name}", landmark=name)

    def goto_lm(name, standoff=0.0, agl=25.0, speed=0.75, look=False):
        x, y = L[name]["xy"]
        b = dict(type="goto", xy=(x, y), agl=agl, speed=speed, radius=max(3.0, standoff), label=f"fly to {name}",
                 landmark=name)
        if look:
            b["look_at"] = (x, y, L[name]["top_z"])
        return b

    def goto_orbit_entry(name, radius, above, from_xy, speed=0.75):
        """Fly to the point on the orbit circle nearest the approach direction, at orbit height."""
        x, y = L[name]["xy"]
        a = math.atan2(from_xy[1] - y, from_xy[0] - x)
        return dict(type="goto", xy=(x + radius * math.cos(a), y + radius * math.sin(a)), agl=25.0, speed=speed,
                    radius=3.0, label=f"fly to {name}", landmark=name, look_at=(x, y, L[name]["top_z"]))

    def hover(name, s=5.0):
        x, y = L[name]["xy"]
        return dict(type="hover", s=s, look_at=(x, y, L[name]["top_z"]), label=f"inspect {name}", landmark=name)

    def land(pad):
        return dict(type="land", xy=P[pad], label=f"land {pad}", pad=pad)

    def river(x_from, x_to, agl=18.0):
        pts = river_points(nv, x_from, x_to)
        return [dict(type="goto", xy=p, agl=agl, speed=0.7, radius=6.0, label="follow river", river=True) for p in pts]

    M = {}
    # ---------------- Easy: one landmark, circle it once, land at a different clearing --------
    M["easy_footbridge"] = dict(
        difficulty="Easy", start="W", yaw_deg=20.0,
        beats=[dict(type="takeoff", agl=25.0, label="take off"),
               goto_orbit_entry("footbridge", 16.0, 14.0, P["W"]), orbit("footbridge", 16.0, 14.0, 1.0, +1),
               land("NW")])
    M["easy_lookout"] = dict(
        difficulty="Easy", start="NW", yaw_deg=-30.0,
        beats=[dict(type="takeoff", agl=25.0, label="take off"),
               goto_orbit_entry("lookout", 18.0, 8.0, P["NW"]), orbit("lookout", 18.0, 8.0, 1.0, -1),
               land("N")])
    M["easy_cairn"] = dict(
        difficulty="Easy", start="SW", yaw_deg=60.0,
        beats=[dict(type="takeoff", agl=25.0, label="take off"),
               goto_orbit_entry("cairn", 14.0, 14.0, P["SW"]), orbit("cairn", 14.0, 14.0, 1.0, +1),
               land("SE")])
    M["easy_marker"] = dict(
        difficulty="Easy", start="N", yaw_deg=-120.0,
        beats=[dict(type="takeoff", agl=25.0, label="take off"),
               goto_orbit_entry("marker", 14.0, 17.0, P["N"]), orbit("marker", 14.0, 17.0, 1.0, -1),
               land("SW")])
    # ---------------- Medium: two landmarks / a route feature, several stages ----------------
    M["medium_bridge_cairn"] = dict(
        difficulty="Medium", start="SE", yaw_deg=150.0,
        beats=[dict(type="takeoff", agl=25.0, label="take off"),
               goto_orbit_entry("footbridge", 16.0, 14.0, P["SE"]), orbit("footbridge", 16.0, 14.0, 1.0, -1),
               goto_orbit_entry("cairn", 14.0, 14.0, L["footbridge"]["xy"]), orbit("cairn", 14.0, 14.0, 1.0, +1),
               land("CAIRN")])
    M["medium_river_lookout"] = dict(
        difficulty="Medium", start="W", yaw_deg=90.0,
        beats=[dict(type="takeoff", agl=18.0, label="take off")]
        + river(-110.0, 110.0)
        + [goto_orbit_entry("lookout", 18.0, 8.0, (110.0, nv.river_y(110.0))), orbit("lookout", 18.0, 8.0, 1.0, +1),
           land("N")])
    M["medium_waterfall_rockslide"] = dict(
        difficulty="Medium", start="CAIRN", yaw_deg=0.0,
        beats=[dict(type="takeoff", agl=25.0, label="take off"),
               dict(type="goto", xy=(236.0, L["waterfall"]["xy"][1] - 4.0), agl=14.0, speed=0.7, radius=3.0,
                    label="fly to waterfall", landmark="waterfall", look_at=(*L["waterfall"]["xy"], L["waterfall"]["top_z"] - 4)),
               hover("waterfall", 5.0),
               dict(type="goto", xy=L["rockslide"]["xy"], agl=30.0, speed=0.6, radius=4.0, label="fly over rockslide",
                    landmark="rockslide", gimbal=55.0),
               land("E")])
    # ---------------- Hard: ordered multi-landmark tasks with specific manoeuvres ----------------
    M["hard_bridge_lookout_cairn"] = dict(
        difficulty="Hard", start="NW", yaw_deg=-90.0,
        beats=[dict(type="takeoff", agl=25.0, label="take off"),
               goto_orbit_entry("footbridge", 12.0, 12.0, P["NW"]), orbit("footbridge", 12.0, 12.0, 1.0, +1),
               goto_orbit_entry("lookout", 18.0, 6.0, L["footbridge"]["xy"]), orbit("lookout", 18.0, 6.0, 2.0, -1),
               dict(type="goto", xy=(L["cairn"]["xy"][0] - 16.0, L["cairn"]["xy"][1] + 6.0), agl=20.0, speed=0.75,
                    radius=3.0, label="fly to cairn", landmark="cairn", look_at=(*L["cairn"]["xy"], L["cairn"]["top_z"])),
               hover("cairn", 5.0),
               land("SW")])
    M["hard_river_patrol"] = dict(
        difficulty="Hard", start="E", yaw_deg=-150.0,
        beats=[dict(type="takeoff", agl=20.0, label="take off"),
               dict(type="goto", xy=(240.0, L["waterfall"]["xy"][1] + 3.0), agl=14.0, speed=0.7, radius=3.0,
                    label="fly to waterfall", landmark="waterfall", look_at=(*L["waterfall"]["xy"], L["waterfall"]["top_z"] - 4)),
               hover("waterfall", 4.0)]
        + river(230.0, -40.0, agl=16.0)
        + [goto_orbit_entry("marker", 14.0, 17.0, (-40.0, nv.river_y(-40.0))), orbit("marker", 14.0, 17.0, 1.0, +1),
           land("NW")])
    M["hard_five_stage"] = dict(
        difficulty="Hard", start="SE", yaw_deg=100.0,
        beats=[dict(type="takeoff", agl=25.0, label="take off"),
               goto_orbit_entry("cairn", 14.0, 14.0, P["SE"]), orbit("cairn", 14.0, 14.0, 0.5, -1),
               dict(type="goto", xy=L["rockslide"]["xy"], agl=30.0, speed=0.7, radius=4.0, label="fly over rockslide",
                    landmark="rockslide", gimbal=55.0),
               goto_orbit_entry("lookout", 18.0, 8.0, L["rockslide"]["xy"]), orbit("lookout", 18.0, 8.0, 1.0, +1),
               dict(type="goto", xy=(L["footbridge"]["xy"][0] + 14.0, L["footbridge"]["xy"][1]), agl=15.0, speed=0.6,
                    radius=3.0, label="fly to footbridge", landmark="footbridge",
                    look_at=(*L["footbridge"]["xy"], L["footbridge"]["top_z"])),
               hover("footbridge", 4.0),
               land("SE")])
    for k, m in M.items():
        m["id"] = k
        m["start_xy"] = P[m["start"]]
        m["tasks"] = tasks(m["beats"])
        rule = difficulty_from_tasks(len(m["tasks"]))
        if m["difficulty"] != rule:
            raise ValueError(f"{k}: labelled {m['difficulty']} but has {len(m['tasks'])} task(s) -> {rule}")
    return M, L


# ---- difficulty = number of tasks (user rule, 2026-10-11) ----------------------------------------------
DIFFICULTY_RULE = "1 task = Easy, 2 tasks = Medium, 3+ tasks = Hard (take-off, transit and landing are not tasks)"


def tasks(beats):
    """The mission's tasks: circling a landmark, inspecting one (hover), following the river (one task however
    many waypoints), or passing over a landmark with the camera down. Take-off, transit legs and landing are
    not counted -- every mission has them."""
    out, prev_river = [], False
    for b in beats:
        t = b["type"]
        if t == "orbit":
            out.append({"task": "circle", "landmark": b["landmark"], "laps": b.get("laps", 1.0), "direction": "counter-clockwise" if b.get("dir", 1) > 0 else "clockwise"})
        elif t == "hover":
            out.append({"task": "inspect", "landmark": b["landmark"], "seconds": b["s"]})
        elif t == "goto" and b.get("river"):
            if not prev_river:
                out.append({"task": "follow_river"})
        elif t == "goto" and "over" in b.get("label", ""):
            out.append({"task": "pass_over", "landmark": b["landmark"]})
        prev_river = t == "goto" and bool(b.get("river"))
    return out


def difficulty_from_tasks(n):
    return "Easy" if n <= 1 else "Medium" if n == 2 else "Hard"


# ---- instructions ------------------------------------------------------------------------------
def _dir(a, b):
    from pilot import compass
    return compass(b[0] - a[0], b[1] - a[1])


def _dist(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _laps(l):
    return {0.5: "half a circle", 1.0: "one full circle", 2.0: "two full circles"}.get(l, f"{l:g} circles")


def _sense(d):
    return "counter-clockwise" if d > 0 else "clockwise"


def instruction(m, L, ground):
    """Instruction text scaled to the task: Easy 1-2 sentences, Medium 3-4, Hard a step-by-step brief."""
    if m.get("instruction"):  # Easy v2 missions carry their own text (easy_v2.mission)
        return m["instruction"], [m["instruction"]]
    pad_desc = {k: v[1] for k, v in PADS.items()}
    beats = m["beats"]
    start = m["start"]
    land_pad = next(b["pad"] for b in beats if b["type"] == "land")
    pos = m["start_xy"]
    steps = []
    i = 0
    while i < len(beats):
        b = beats[i]
        t = b["type"]
        if t == "takeoff":
            steps.append(f"take off from {pad_desc[start]} and climb to about {b['agl']:.0f} m")
        elif t == "goto" and b.get("river"):
            j = i
            while j < len(beats) and beats[j].get("river"):
                j += 1
            first, last = beats[i]["xy"], beats[j - 1]["xy"]
            way = "upstream (east)" if last[0] > first[0] else "downstream (west)"
            steps.append(f"follow the river {way} at about {b['agl']:.0f} m above the water for roughly "
                         f"{abs(last[0] - first[0]):.0f} m")
            pos = last
            i = j
            continue
        elif t == "goto":
            name = b.get("landmark")
            if name is None:
                steps.append(f"fly {_dir(pos, b['xy'])}")
            elif "over" in b["label"]:
                steps.append(f"fly {_dir(pos, b['xy'])} and pass over {L[name]['desc']} with the camera looking down")
            else:
                nxt = beats[i + 1] if i + 1 < len(beats) else {}
                if nxt.get("type") in ("orbit", "hover") and nxt.get("landmark") == name:
                    steps.append(f"fly {_dir(pos, L[name]['xy'])} about {_dist(pos, L[name]['xy']):.0f} m to {L[name]['desc']}")
                else:
                    steps.append(f"fly {_dir(pos, b['xy'])} to {L[name]['desc']}")
            pos = b["xy"]
        elif t == "orbit":
            steps.append(f"circle {L[b['landmark']]['short']} {_sense(b['dir'])} for {_laps(b['laps'])} at a radius of "
                         f"about {b['radius']:.0f} m, keeping the camera on it")
        elif t == "hover":
            steps.append(f"hold position for about {b['s']:.0f} seconds facing {L[b['landmark']]['short']}")
        elif t == "land":
            steps.append(f"fly {_dir(pos, b['xy'])} to {pad_desc[land_pad]} and land there")
            pos = b["xy"]
        i += 1

    d = m["difficulty"]
    if d == "Easy":
        lm = next(b["landmark"] for b in beats if b["type"] == "orbit")
        text = (f"Take off from {pad_desc[start]}, fly to {L[lm]['desc']} and circle it once, "
                f"then land at {pad_desc[land_pad]}.")
    elif d == "Medium":
        body = "; ".join(steps[1:-1])
        text = (f"Take off from {pad_desc[start]}. Then {body}. "
                f"Finish by landing at {pad_desc[land_pad]}, not back where you started.")
        text = text[0].upper() + text[1:]
    else:
        lines = [f"{k + 1}. {s[0].upper() + s[1:]}." for k, s in enumerate(steps)]
        text = ("This is a multi-stage forest inspection; do the stages in this exact order. "
                + " ".join(lines)
                + f" Stay above the treetops between stages and touch down only at {pad_desc[land_pad]}.")
    return text, [s[0].upper() + s[1:] for s in steps]
