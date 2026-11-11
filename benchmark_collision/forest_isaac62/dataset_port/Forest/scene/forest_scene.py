"""Customizations applied on top of the reference natural_valley forest scene.

Three changes, all requested after reviewing the first forest render:

1. SMOOTH TERRAIN (was: "like we [are] in a circle").
   The upstream height field builds a steep-walled trough plus a literal disc around the pad:
       ridge        = 0.0015 * lateral**1.78   -> 5m at 100m out, 17m at 200m, 36m at 300m
       launch_blend = min(1, hypot(x,y)/24)    -> terrain scaled 12%..100% inside a 24m radius,
                                                  which reads on camera as a circular pad/crater
       horizon_rise = up to +95m past 650m     -> a wall closing the horizon
   smooth_terrain_height() replaces all of that with gentle rolling hills, a near-linear (not
   power-curve) lateral rise, and a wide gaussian pad flattening with no hard radius, so there is
   no visible circle and no bowl.

2. MORE VEGETATION (was: "simulate more object i need more trees bushes etc").
   Upstream deliberately clears vegetation 80-90m around the pad, 12m around every landmark and
   35-45m around all ten of ITS OWN mission routes -- which is why the first render was bald
   (measured: 0 trees within 60m of the footbridge). relax_clearings() cuts those back to a small
   pad-only exclusion, and add_dense_vegetation() then layers several thousand extra instances
   (mature trees, saplings, shrubs, ferns, grass, rocks, deadwood) over the mission area.

3. The river channel and a small ring around the bridge are kept clear of trees so the water and
   the mission target stay readable from the air.

Everything here is applied by monkeypatching / adding to the upstream module rather than editing
it: natural_valley.py belongs to the cloned reference repo (VLN-Simulations@1e9b0c8) and is shared
with Simulations_0_0_2's renderer-comparison work, so it stays byte-identical to the pinned commit.
"""

import math

import numpy as np

# Kept clear of trees so they stay readable from the air.
RIVER_HALF_WIDTH_M = 8.5  # river v2: bank top at 6.8 m
PAD_CLEAR_RADIUS_M = 11.0
BRIDGE_CLEAR_RADIUS_M = 15.0

# Simulations_0_0_4: the drone starts at one pad and LANDS AT A DIFFERENT ONE, and neither may be
# in the water. Previous projects used (0,0) as the pad -- but river_y(0) == 0.0 exactly, so that
# pad sat dead on the river centreline: the drone was taking off from and landing ON the river.
# Both pads below are verified dry land (>50m from the centreline) and each gets its own local
# terrain flattening + vegetation clearing.
START_PAD = (-40.0, 45.0)   # 51.7m from the river centreline, ground ~-0.24m
END_PAD = (120.0, -55.0)    # 59.9m from the river centreline, ground ~+1.33m
PADS = (START_PAD, END_PAD)
PAD_FLATTEN_RADIUS_M = 34.0  # gaussian sigma of the local levelling around each pad


def smooth_terrain_height(x, y, seed):
    """Gentle rolling valley -- see module docstring item 1."""
    centre = 7.0 * math.sin(x / 70.0) + 3.0 * math.sin(x / 31.0)  # same river_y as upstream
    lateral = abs(y - centre)
    # near-linear lateral rise: 1.2m at 100m, 3.6m at 300m, 7.2m at 600m (upstream: 5 / 36 / ~100)
    ridge = 0.012 * min(lateral, 600.0)
    # low-frequency rolling hills -- large wavelengths so nothing reads as a ridge or a rim
    rolling = (2.2 * math.sin(x / 130.0 + seed * 0.011)
               + 1.8 * math.sin(y / 155.0 + seed * 0.007)
               + 1.1 * math.sin((x + y) / 190.0))
    river_cut = -1.0 * math.exp(-((y - centre) / 10.0) ** 2)
    base = ridge + rolling + river_cut + river_channel(lateral, x)
    # Local levelling at each landing pad so takeoff/touchdown happen on flat ground. Gaussian
    # (no hard radius) so there is no visible disc edge -- the upstream version's hard 24m
    # launch_blend radius is exactly what read on camera as "we are in a circle".
    level = 0.0
    for px, py in PADS:
        level = max(level, math.exp(-((math.hypot(x - px, y - py) / PAD_FLATTEN_RADIUS_M) ** 2)))
    return base * (1.0 - 0.8 * level) - 0.15 + waterfall_step(x)


# ---- waterfall terrace (X500 dataset fix, 2026-10-10) ------------------------------------------------
# Upstream's "waterfall" was a turquoise box hovering 2.5 m over a flat river (no drop anywhere). The
# valley now steps up WATERFALL_DROP_M across x = WATERFALL_X, so the river genuinely falls over a rock
# lip there; build_waterfall() dresses the step. Everything else (pads, landmarks, all flight paths, which
# stay at x < 245) is unaffected: the step is < 1 cm below x = 245.
WATERFALL_X = 262.0
WATERFALL_DROP_M = 4.0
WATERFALL_WIDTH_M = 2.2  # sigmoid scale: 10%..90% of the drop over ~9.7 m, resolvable by the 8.1 m terrain grid


def waterfall_step(x):
    u = max(-60.0, min(60.0, (x - WATERFALL_X) / WATERFALL_WIDTH_M))  # clamp: distant terrain spans km
    return WATERFALL_DROP_M / (1.0 + math.exp(-u))


def patch_terrain(nv, seed):
    """Point natural_valley's module-global terrain_height at the smooth version, so the terrain
    mesh, the landmarks and the vegetation scatter all agree on one height field."""
    nv.terrain_height = smooth_terrain_height
    patch_river_v2(nv)
    return nv


def _clear_of_pads(x, y, radius=PAD_CLEAR_RADIUS_M):
    """True when (x,y) is outside BOTH landing pads' clearings."""
    return all(math.hypot(x - px, y - py) >= radius for px, py in PADS)


def relax_clearings(nv):
    """Shrink upstream's vegetation exclusions (see module docstring item 2), but keep both
    landing pads clear -- the drone has to actually touch down on open ground at each end."""
    nv.clear_of_launch = lambda x, y, radius: _clear_of_pads(x, y)
    # every built landmark keeps only a small ring, instead of 12m around six landmarks
    nv.safe_from_landmarks = lambda x, y: math.hypot(x - 72.0, y - nv.river_y(72.0)) > BRIDGE_CLEAR_RADIUS_M
    # neutralise the 35-45m clearing around the reference repo's own ten routes: hand
    # scatter_nature a single degenerate segment far outside the map so nothing is ever rejected
    nv.mission_definition = lambda episode_id, seed: {
        "waypoints_enu_m": [(9.0e4, 9.0e4, 0.0, "unused"), (9.0e4 + 1.0, 9.0e4, 0.0, "unused")]
    }
    return nv


def _tree_ok(x, y, nv):
    """Trees avoid the river channel, both landing pads, and the bridge ring."""
    if abs(y - nv.river_y(x)) < RIVER_HALF_WIDTH_M:
        return False
    if not _clear_of_pads(x, y):
        return False
    if math.hypot(x - 72.0, y - nv.river_y(72.0)) < BRIDGE_CLEAR_RADIUS_M:
        return False
    return True


def add_dense_vegetation(stage, nv, seed, assets_root, area_radius_m=260.0,
                          n_mature=260, n_saplings=850, n_groundcover=3000,
                          n_rocks=320, n_debris=140):
    """Layer extra instances over whatever build_environment already scattered.

    Uses upstream's own point_instancer/material helpers so these instances are authored exactly
    like the originals (shared prototypes, spatial partitioning, correct extents) -- this is
    additive, it does not touch or re-scatter the existing /World/Nature instancers.
    """
    rng = np.random.default_rng(seed + 101)
    materials = {
        key: nv.create_asset_material(stage, f"/World/Materials/Dense_{key}", assets_root, asset_id)
        for key, asset_id in nv.ASSET_IDS.items()
    }

    def sample_points(count, r_min, r_max, scale_fn, filter_fn=_tree_ok, protos=1):
        out = []
        attempts = 0
        while len(out) < count and attempts < count * 25:
            attempts += 1
            r = float(rng.uniform(r_min, r_max))
            theta = float(rng.uniform(0.0, 2.0 * math.pi))
            x, y = r * math.cos(theta), r * math.sin(theta)
            if filter_fn and not filter_fn(x, y, nv):
                continue
            z = nv.terrain_height(x, y, seed)
            s = scale_fn()
            out.append((int(rng.integers(0, protos)), (x, y, z),
                        float(rng.uniform(-math.pi, math.pi)), (s, s, s)))
        return out

    counts = {}

    mature = sample_points(n_mature, 14.0, area_radius_m,
                           lambda: float(rng.uniform(0.30, 0.52)), protos=1)
    nv.point_instancer(stage, "/World/DenseNature/MatureTrees", [
        (nv.asset_path(assets_root, "mature_tree"), 1.0, materials["mature_tree"]),
    ], mature)
    counts["dense_mature_trees"] = len(mature)

    saplings = sample_points(n_saplings, 12.0, area_radius_m,
                             lambda: float(rng.uniform(1.6, 3.2)), protos=2)
    nv.point_instancer(stage, "/World/DenseNature/Trees", [
        (nv.asset_path(assets_root, "fir"), 1.0, materials["fir"]),
        (nv.asset_path(assets_root, "pine"), 1.0, materials["pine"]),
    ], saplings)
    counts["dense_saplings"] = len(saplings)

    # groundcover may sit closer in and may line the riverbank, so it uses a looser filter --
    # still keeps the immediate touchdown spot of each pad clear
    def ground_ok(x, y, _nv):
        # river v2: nothing grows in the water (the plants standing in the river in the first videos)
        return _clear_of_pads(x, y, PAD_CLEAR_RADIUS_M * 0.6) and abs(y - _nv.river_y(x)) > _shore_half(x) + 1.0

    ground = sample_points(n_groundcover, 8.0, area_radius_m,
                           lambda: float(rng.uniform(0.8, 2.1)),
                           filter_fn=ground_ok, protos=3)
    nv.point_instancer(stage, "/World/DenseNature/Groundcover", [
        (nv.asset_path(assets_root, "grass"), 1.0, materials["grass"]),
        (nv.asset_path(assets_root, "fern"), 1.0, materials["fern"]),
        (nv.asset_path(assets_root, "shrub"), 1.0, materials["shrub"]),
    ], ground)
    counts["dense_groundcover"] = len(ground)

    rocks = sample_points(n_rocks, 10.0, area_radius_m,
                          lambda: float(rng.uniform(0.5, 1.6)),
                          filter_fn=ground_ok, protos=4)
    nv.point_instancer(stage, "/World/DenseNature/Rocks", [
        (nv.asset_path(assets_root, "small_rock"), 1.0, materials["small_rock"]),
        (nv.asset_path(assets_root, "stone"), 1.0, materials["stone"]),
        (nv.asset_path(assets_root, "rock"), 1.0, materials["rock"]),
        (nv.asset_path(assets_root, "boulder"), 1.0, materials["boulder"]),
    ], rocks)
    counts["dense_rocks"] = len(rocks)

    debris = sample_points(n_debris, 14.0, area_radius_m,
                           lambda: float(rng.uniform(0.7, 1.5)), protos=2)
    nv.point_instancer(stage, "/World/DenseNature/Debris", [
        (nv.asset_path(assets_root, "stump"), 1.0, materials["stump"]),
        (nv.asset_path(assets_root, "dead_log"), 1.0, materials["dead_log"]),
    ], debris)
    counts["dense_debris"] = len(debris)

    counts["dense_total"] = sum(counts.values())
    return counts


BRIDGE_X = 72.0
BRIDGE_DECK_ABOVE_TERRAIN_M = 3.6  # river v2: deep channel; deck 2.65 m above the water, just above the bank tops


def rebuild_footbridge(stage, nv, seed):
    """Replace upstream's footbridge with one that is actually attached to the ground.

    As authored upstream it is 13 loose planks at terrain+2.1m with 0.57m gaps between them, no
    deck, no piers, no abutments, and two railings floating 1.1m OUTSIDE the deck edges (rails sit
    at x = bx +- 2.7 while the deck is only 3.2m wide, i.e. +-1.6m). From the air it reads as
    planks hovering over the river -- "the bridge is flying".

    This removes that prim and rebuilds the landmark as a real timber footbridge: a continuous
    deck, stone abutments carrying the deck down to the bank on each side, four piers standing in
    the riverbed, and railings with balusters along the actual deck edges. Deck height is
    unchanged, so the mission's target z and the measured orbit clearances still hold.
    """
    Gf = nv.Gf
    bx = BRIDGE_X
    by = nv.river_y(bx)
    ground = nv.terrain_height(bx, by, seed)
    deck_top = ground + BRIDGE_DECK_ABOVE_TERRAIN_M

    stage.RemovePrim("/World/Landmarks/Footbridge")

    wood = nv.create_color_material(stage, "/World/Materials/BridgeWood", (0.24, 0.13, 0.06), 0.72)
    dark_wood = nv.create_color_material(stage, "/World/Materials/BridgePier", (0.16, 0.09, 0.04), 0.8)
    stone = nv.create_color_material(stage, "/World/Materials/BridgeStone", (0.33, 0.32, 0.29), 0.9)

    deck_len = 13.0
    deck_w = 3.2
    deck_t = 0.22
    root = "/World/Landmarks/Footbridge"

    # continuous deck (replaces the gapped planks)
    nv.cube(stage, f"{root}/Deck", (bx, by, deck_top - deck_t / 2.0),
            (deck_w, deck_len, deck_t), wood)
    # plank texture on top: thin raised strips, but touching each other so there are no holes
    n_planks = 16
    for i in range(n_planks):
        py = by - deck_len / 2.0 + (i + 0.5) * (deck_len / n_planks)
        nv.cube(stage, f"{root}/Plank_{i:02d}", (bx, py, deck_top + 0.02),
                (deck_w - 0.1, deck_len / n_planks - 0.06, 0.05), wood)

    # stringers under the deck, running the full span
    for side in (-1, 1):
        nv.cube(stage, f"{root}/Stringer_{'W' if side < 0 else 'E'}",
                (bx + side * (deck_w / 2.0 - 0.25), by, deck_top - deck_t - 0.14),
                (0.22, deck_len, 0.28), dark_wood)

    # abutments: carry the deck down to the ground at both ends
    for side, name in ((-1, "South"), (1, "North")):
        ay = by + side * (deck_len / 2.0 + 0.55)
        g = nv.terrain_height(bx, ay, seed)
        height = max(0.4, deck_top - g)
        nv.cube(stage, f"{root}/Abutment_{name}", (bx, ay, g + height / 2.0),
                (deck_w + 0.7, 1.3, height), stone)

    # piers standing in the riverbed, base on the ground, top at the deck underside.
    # NOTE: prim names must be valid USD identifiers -- no '-', so the offsets are named S/N
    # rather than interpolated from their (signed) numeric value.
    for px_side, x_name in ((-1, "W"), (1, "E")):
        for py_off, y_name in ((-2.4, "S"), (2.4, "N")):
            px = bx + px_side * (deck_w / 2.0 - 0.45)
            py = by + py_off
            g = nv.terrain_height(px, py, seed)
            height = max(0.5, (deck_top - deck_t - 0.28) - g)
            nv.cylinder(stage, f"{root}/Pier_{x_name}{y_name}",
                        (px, py, g + height / 2.0), 0.20, height, dark_wood)

    # railings ON the deck edges (upstream put them 1.1m outside it), with balusters
    for side, name in ((-1, "West"), (1, "East")):
        rx = bx + side * (deck_w / 2.0 - 0.09)
        nv.cube(stage, f"{root}/Rail_{name}", (rx, by, deck_top + 1.02),
                (0.09, deck_len, 0.09), wood)
        n_posts = 9
        for i in range(n_posts):
            py = by - deck_len / 2.0 + (i + 0.5) * (deck_len / n_posts)
            nv.cube(stage, f"{root}/Baluster_{name}_{i}", (rx, py, deck_top + 0.53),
                    (0.08, 0.08, 1.02), wood)

    return {
        "bridge_ground_z": ground,
        "bridge_deck_top_z": deck_top,
        "bridge_rebuilt": True,
    }


# ---- fire lookout tower rebuild (X500 dataset fix, 2026-10-10) -----------------------------------------
LOOKOUT_XY = (130.0, 85.0)


def rebuild_lookout(stage, nv, seed):
    """Replace upstream's fire lookout with one that stands on the ground.

    Upstream authors it with nv.cube(), whose cube has size 1, but passes HALF-extents as the scale:
    the four 9 m legs come out 4.5 m tall, centred 4.5 m up -- their feet hang 2.1-2.4 m above the
    ground -- and the "3.2 m" deck is really 3.2 m wide, narrower than the 4.6 m spacing of the legs,
    so the legs don't reach it either. From the drone it reads as a tower floating in the air.

    Rebuilt with full extents: legs from below grade up to the deck, three rings of cross beams, a
    ladder, a 5.6 m deck that sits on the legs, deck railings, the cabin, a roof, and the orange beacon
    on a pole whose top stays at ground + 14.3 m (the landmark height the missions orbit around).
    """
    lx, ly = LOOKOUT_XY
    lz = nv.terrain_height(lx, ly, seed)
    root = "/World/Landmarks/Lookout"
    stage.RemovePrim(root)
    wood = nv.create_color_material(stage, "/World/Materials/LookoutWood", (0.24, 0.13, 0.06), 0.75)
    dark = nv.create_color_material(stage, "/World/Materials/LookoutWoodDark", (0.15, 0.08, 0.035), 0.8)
    canvas = nv.create_color_material(stage, "/World/Materials/LookoutCabin", (0.42, 0.17, 0.06), 0.9)
    roof_m = nv.create_color_material(stage, "/World/Materials/LookoutRoof", (0.12, 0.12, 0.13), 0.6)
    marker = nv.create_color_material(stage, "/World/Materials/LookoutBeacon", (0.92, 0.22, 0.05), 0.42)

    deck_bottom = lz + 8.6
    leg = 2.3
    legs = [(-leg, -leg), (-leg, leg), (leg, -leg), (leg, leg)]
    for i, (dx, dy) in enumerate(legs):
        g = nv.terrain_height(lx + dx, ly + dy, seed) - 0.3  # sink 0.3 m into the ground
        h = deck_bottom - g
        nv.cube(stage, f"{root}/Leg_{i}", (lx + dx, ly + dy, g + h / 2.0), (0.24, 0.24, h), wood)
    # three rings of horizontal beams tying the legs together
    for k, zf in enumerate((0.3, 0.55, 0.8)):
        z = lz + zf * 8.6
        for j, (cx, cy, sx, sy) in enumerate(((0, -leg, 2 * leg, 0.14), (0, leg, 2 * leg, 0.14),
                                               (-leg, 0, 0.14, 2 * leg), (leg, 0, 0.14, 2 * leg))):
            nv.cube(stage, f"{root}/Beam_{k}_{j}", (lx + cx, ly + cy, z), (sx, sy, 0.14), dark)
    # ladder on the south face, from the ground to the deck
    g = nv.terrain_height(lx, ly - leg - 0.35, seed)
    h = deck_bottom + 0.2 - g
    for side in (-0.3, 0.3):
        nv.cube(stage, f"{root}/LadderRail_{'W' if side < 0 else 'E'}", (lx + side, ly - leg - 0.35, g + h / 2.0),
                (0.07, 0.07, h), dark)
    for r in range(int(h / 0.35)):
        nv.cube(stage, f"{root}/LadderRung_{r:02d}", (lx, ly - leg - 0.35, g + 0.3 + r * 0.35), (0.6, 0.05, 0.05), dark)

    deck_w, deck_t = 5.6, 0.2
    nv.cube(stage, f"{root}/Deck", (lx, ly, deck_bottom + deck_t / 2.0), (deck_w, deck_w, deck_t), wood)
    top = deck_bottom + deck_t
    # deck railing: corner posts + top rail on all four sides
    half = deck_w / 2.0 - 0.06
    for i, (dx, dy) in enumerate(((-half, -half), (-half, half), (half, -half), (half, half))):
        nv.cube(stage, f"{root}/RailPost_{i}", (lx + dx, ly + dy, top + 0.5), (0.1, 0.1, 1.0), dark)
    for j, (cx, cy, sx, sy) in enumerate(((0, -half, deck_w, 0.08), (0, half, deck_w, 0.08),
                                           (-half, 0, 0.08, deck_w), (half, 0, 0.08, deck_w))):
        nv.cube(stage, f"{root}/Rail_{j}", (lx + cx, ly + cy, top + 1.0), (sx, sy, 0.08), dark)
        nv.cube(stage, f"{root}/MidRail_{j}", (lx + cx, ly + cy, top + 0.5), (sx, sy, 0.06), dark)

    cabin_w, cabin_h = 3.4, 2.4
    nv.cube(stage, f"{root}/Cabin", (lx, ly, top + cabin_h / 2.0), (cabin_w, cabin_w, cabin_h), canvas)
    roof_t = 0.25
    roof_bottom = top + cabin_h
    nv.cube(stage, f"{root}/Roof", (lx, ly, roof_bottom + roof_t / 2.0), (cabin_w + 0.8, cabin_w + 0.8, roof_t), roof_m)
    beacon_top = lz + 14.3
    pole_bottom = roof_bottom + roof_t
    nv.cylinder(stage, f"{root}/Beacon", (lx, ly, (pole_bottom + beacon_top) / 2.0), 0.25, beacon_top - pole_bottom, marker)
    return {"lookout_ground_z": lz, "lookout_deck_top_z": top, "lookout_beacon_top_z": beacon_top}


# ---- waterfall dressing ------------------------------------------------------------------------------
TERRAIN_GRID_N, TERRAIN_EXTENT = 161, 650.0  # must match natural_valley.build_terrain()


def terrain_mesh_height(nv, x, y, seed):
    """Height of the RENDERED terrain mesh at (x, y): natural_valley's 161x161 grid (8.1 m spacing)
    triangulated as (a,b,d)+(a,d,c), linearly interpolated -- not the analytic height function. Water and
    rocks placed on the analytic height hang above / sink below the mesh wherever it is curved (the first
    waterfall probe showed water slabs floating over the step), so they are placed on this instead."""
    h = 2 * TERRAIN_EXTENT / (TERRAIN_GRID_N - 1)
    fx, fy = (x + TERRAIN_EXTENT) / h, (y + TERRAIN_EXTENT) / h
    c, r = min(max(int(math.floor(fx)), 0), TERRAIN_GRID_N - 2), min(max(int(math.floor(fy)), 0), TERRAIN_GRID_N - 2)
    u, v = fx - c, fy - r
    gx = lambda k: -TERRAIN_EXTENT + k * h
    za, zb = nv.terrain_height(gx(c), gx(r), seed), nv.terrain_height(gx(c + 1), gx(r), seed)
    zc, zd = nv.terrain_height(gx(c), gx(r + 1), seed), nv.terrain_height(gx(c + 1), gx(r + 1), seed)
    if u >= v:
        return za + u * (zb - za) + v * (zd - zb)
    return za + v * (zc - za) + u * (zd - zc)


def _mesh_ribbon(nv, samples, half_width, z_offset, seed, across=5):
    """Ribbon along the river centre line, `across` vertices wide, every vertex on the terrain mesh."""
    G = nv.Gf
    pts, uvs, idx, cnt = [], [], [], []
    for i, x in enumerate(samples):
        yc = nv.river_y(float(x))
        for j in range(across):
            y = yc - half_width + 2 * half_width * j / (across - 1)
            pts.append(G.Vec3f(float(x), float(y), float(terrain_mesh_height(nv, float(x), y, seed) + z_offset)))
            uvs.append(G.Vec2f(i / 5.0, j / (across - 1)))
    for i in range(len(samples) - 1):
        for j in range(across - 1):
            a, b = i * across + j, i * across + j + 1
            c, d = (i + 1) * across + j, (i + 1) * across + j + 1
            idx += [a, b, d, a, d, c]
            cnt += [3, 3]
    return pts, idx, cnt, uvs


def build_waterfall(stage, nv, seed, assets_root):
    """Turn the terrace at WATERFALL_X into a rocky cascade and remove upstream's floating box.

    The original river (water + riverbed ribbons, analytic heights, 10 m sampling) is kept everywhere
    except a SECTION of +-SECTION_M around the drop, which is re-authored on the terrain mesh at 0.25 m
    sampling so the water lies on the ground all the way down the step. In that section:
      * the river's own dark water flows down the drop (no riverbed band -- it would read as a pale strip);
      * thin, semi-transparent white streaks over the steep part, plus small rocks in the stream: rapids;
      * an irregular, see-through foam patch at the foot;
      * rocks and boulders along both banks, seated into the mesh.
    """
    SECTION_M = 22.0
    x0, x1 = WATERFALL_X - SECTION_M, WATERFALL_X + SECTION_M
    stage.RemovePrim("/World/Landmarks/Waterfall")
    for path in ("/World/RiverWater", "/World/Riverbed"):
        stage.RemovePrim(path)
    bed = stage.GetPrimAtPath("/World/Materials/Riverbed")
    water = stage.GetPrimAtPath("/World/Materials/Water")
    bed_m = nv.UsdShade.Material(bed) if bed else nv.create_pbr_material(stage, "/World/Materials/Riverbed", assets_root, "ganges_river_pebbles")
    water_m = nv.UsdShade.Material(water) if water else nv.create_color_material(stage, "/World/Materials/Water", (0.001, 0.018, 0.032), roughness=0.13, opacity=0.58)
    # original river, unchanged, on both sides of the section (one sample of overlap into the section)
    samples = np.linspace(-1200.0, 1600.0, 281)
    for name, part in (("Down", samples[samples <= x0 + 10.0]), ("Up", samples[samples >= x1 - 10.0])):
        nv.create_mesh(stage, f"/World/Riverbed{name}", *nv.ribbon_geometry(part, 7.2, 0.12, seed), bed_m, collision=False)
        nv.create_mesh(stage, f"/World/RiverWater{name}", *nv.ribbon_geometry(part, 2.6, 0.23, seed), water_m, collision=False)

    root = "/World/Landmarks/Waterfall"
    sec = np.arange(x0, x1 + 0.01, 0.25)
    nv.create_mesh(stage, f"{root}/Water", *_mesh_ribbon(nv, sec, 2.6, 0.23, seed), water_m, collision=False)

    rng = np.random.default_rng(seed + 4242)
    G = nv.Gf
    # Transparency does not render in this RTX path (the river's own opacity-0.58 water renders solid),
    # so white water is drawn opaque: churned lighter water over the steep part, short foam dashes on it.
    churn = nv.create_color_material(stage, "/World/Materials/ChurnedWater", (0.035, 0.075, 0.09), roughness=0.25)
    steep = np.arange(WATERFALL_X - 6.0, WATERFALL_X + 5.0, 0.25)
    nv.create_mesh(stage, f"{root}/ChurnedWater", *_mesh_ribbon(nv, steep, 2.5, 0.255, seed), churn, collision=False)
    foam_m = nv.create_color_material(stage, "/World/Materials/Foam", (0.55, 0.62, 0.66), roughness=0.6)

    def blob(path, cx, cy, rx, ry, dz):
        pts = [G.Vec3f(cx, cy, terrain_mesh_height(nv, cx, cy, seed) + dz)]
        n = 16
        for j in range(n):
            a = 2 * math.pi * j / n
            r = 1.0 + 0.3 * math.sin(3 * a + cx)
            px, py = cx + rx * r * math.cos(a), cy + ry * r * math.sin(a)
            pts.append(G.Vec3f(px, py, terrain_mesh_height(nv, px, py, seed) + dz))
        idx, cnt = [], []
        for j in range(n):
            idx += [0, 1 + j, 1 + (j + 1) % n]
            cnt.append(3)
        nv.create_mesh(stage, path, pts, idx, cnt, None, foam_m, collision=False)

    for k in range(22):  # short foam dashes on the churned water, elongated along the flow
        x = WATERFALL_X + rng.uniform(-5.5, 4.5)
        blob(f"{root}/Dash_{k:02d}", x, nv.river_y(x) + rng.uniform(-1.9, 1.9), rng.uniform(0.4, 1.0), rng.uniform(0.07, 0.16), 0.27)
    for k in range(5):  # small foam patches at the foot of the drop
        x = WATERFALL_X - 7.0 + rng.uniform(-1.5, 1.0)
        blob(f"{root}/Foam_{k}", x, nv.river_y(x) + rng.uniform(-1.3, 1.3), rng.uniform(0.4, 0.9), rng.uniform(0.25, 0.55), 0.26)

    rock_m = nv.create_asset_material(stage, "/World/Materials/FallRock", assets_root, nv.ASSET_IDS["rock"])
    boulder_m = nv.create_asset_material(stage, "/World/Materials/FallBoulder", assets_root, nv.ASSET_IDS["boulder"])
    inst = []
    for x in np.arange(WATERFALL_X - 9.0, WATERFALL_X + 8.0, 2.0):  # banks
        for side in (-1, 1):
            if rng.random() < 0.3:
                continue
            y = nv.river_y(float(x)) + side * rng.uniform(3.2, 6.0)
            z = terrain_mesh_height(nv, float(x), y, seed) - 0.4
            s = float(rng.uniform(0.9, 1.8))
            inst.append((int(rng.integers(0, 2)), (float(x), float(y), float(z)), float(rng.uniform(-math.pi, math.pi)), (s, s, s)))
    for k in range(7):  # rocks sticking out of the water on the steep part
        x = WATERFALL_X + rng.uniform(-4.5, 4.5)
        y = nv.river_y(x) + rng.uniform(-1.8, 1.8)
        z = terrain_mesh_height(nv, x, y, seed) - 0.05
        s = float(rng.uniform(0.6, 0.95))
        inst.append((0, (float(x), float(y), float(z)), float(rng.uniform(-math.pi, math.pi)), (s, s, s)))
    nv.point_instancer(stage, f"{root}/Rocks", [
        (nv.asset_path(assets_root, "rock"), 1.0, rock_m),
        (nv.asset_path(assets_root, "boulder"), 1.0, boulder_m),
    ], inst)
    top_z = terrain_mesh_height(nv, WATERFALL_X + 6.0, nv.river_y(WATERFALL_X + 6.0), seed)
    return {"waterfall_x": WATERFALL_X, "waterfall_drop_m": WATERFALL_DROP_M, "waterfall_rocks": len(inst),
            "waterfall_top_z": top_z}


# ==== river v2 (X500 Easy dataset, 2026-10-11) =========================================================
# User feedback on the published Forest videos: "fix the river part", "i dont need that blue strips". The
# upstream river is two flat ribbons laid on the 8.1 m terrain grid: a 5.2 m bright water strip with
# straight edges and a 14.4 m pebble strip that is mostly buried and pokes out as a pale wedge on one bank;
# grass and ferns from the dense layer grow in the water. v2 replaces it with a river that sits IN the land:
#   * the height field gets a real channel (RIVER_DEPTH_M deep, flat bed, sloped banks) -- river_channel();
#   * the coarse 161x161 terrain cannot resolve a 9 m channel, so it is lowered under the river
#     (coarse_terrain_height) and a fine corridor mesh (1 m x 0.5 m) carries the true banks, blending back
#     into the coarse mesh at its edges with the same ground material and UVs (no visible seam);
#   * the water is one FLAT surface per cross-section at RIVER_WATER_ABOVE_BED_M: it meets the sloped
#     banks wherever they cross that level, so the shoreline is where the ground says it is, not a ribbon
#     edge; dark, slightly rough, with small ripples (no mirror-blue sky reflection);
#   * pebble shores with irregular edges on both banks, and stones along the waterline;
#   * no vegetation inside the channel.
RIVER_DEPTH_M = 2.3  # user: the water must sit clearly BELOW the land, banks sloping down to it
RIVER_BED_HALF_M = 2.8      # flat bed half width
RIVER_BANK_TOP_HALF_M = 6.8  # where the bank meets the valley floor
RIVER_WATER_ABOVE_BED_M = 0.95  # -> water surface 1.35 m below the bank tops
CORRIDOR_HALF_M, CORRIDOR_BLEND_M = 26.0, 6.0
COARSE_TROUGH_M, COARSE_TROUGH_SIGMA_M = 4.8, 13.0
RIVER_X_RANGE = (-650.0, 650.0)  # fine corridor; beyond it simple far ribbons (km away from every route)
VEG_RIVER_CLEAR_M = 7.5


def river_bed_half(x):
    """Bed half width varies along the river (1.9 .. 4.1 m) so it does not read as a canal."""
    return RIVER_BED_HALF_M + 0.75 * math.sin(x / 37.0 + 0.4) + 0.45 * math.sin(x / 13.3 + 1.1)


def river_channel(lateral, x=0.0):
    bed = river_bed_half(x)
    top = bed + (RIVER_BANK_TOP_HALF_M - RIVER_BED_HALF_M) + 0.8 * math.sin(x / 21.0 + 2.0)
    if lateral <= bed:
        return -RIVER_DEPTH_M
    if lateral >= top:
        return 0.0
    u = (lateral - bed) / (top - bed)
    return -RIVER_DEPTH_M * (1.0 - u * u * (3.0 - 2.0 * u))


def river_bed_z(nv, x, seed):
    return smooth_terrain_height(x, nv.river_y(x), seed)


def river_water_z(nv, x, seed):
    return river_bed_z(nv, x, seed) + RIVER_WATER_ABOVE_BED_M


def coarse_terrain_height(x, y, seed):
    """What the coarse 161x161 terrain mesh is built from: the true surface, lowered under the river so
    the fine corridor mesh (which carries the real banks) always lies on top of it."""
    lateral = abs(y - (7.0 * math.sin(x / 70.0) + 3.0 * math.sin(x / 31.0)))
    return smooth_terrain_height(x, y, seed) - COARSE_TROUGH_M * math.exp(-(lateral / COARSE_TROUGH_SIGMA_M) ** 2)


def coarse_mesh_height(nv, x, y, seed):
    """Height of the rendered coarse terrain mesh (triangle interpolation of coarse_terrain_height)."""
    h = 2 * TERRAIN_EXTENT / (TERRAIN_GRID_N - 1)
    fx, fy = (x + TERRAIN_EXTENT) / h, (y + TERRAIN_EXTENT) / h
    c, r = min(max(int(math.floor(fx)), 0), TERRAIN_GRID_N - 2), min(max(int(math.floor(fy)), 0), TERRAIN_GRID_N - 2)
    u, v = fx - c, fy - r
    gx = lambda k: -TERRAIN_EXTENT + k * h
    za, zb = coarse_terrain_height(gx(c), gx(r), seed), coarse_terrain_height(gx(c + 1), gx(r), seed)
    zc, zd = coarse_terrain_height(gx(c), gx(r + 1), seed), coarse_terrain_height(gx(c + 1), gx(r + 1), seed)
    if u >= v:
        return za + u * (zb - za) + v * (zd - zb)
    return za + v * (zc - za) + u * (zd - zc)


def visible_ground(nv, x, y, seed):
    """Height of the ground surface you SEE at (x, y): the fine corridor near the river, else the coarse
    mesh. Objects are seated on this (not on the analytic function) so nothing hovers or sinks."""
    lateral = abs(y - nv.river_y(x))
    if lateral >= CORRIDOR_HALF_M or not (RIVER_X_RANGE[0] <= x <= RIVER_X_RANGE[1]):
        return coarse_mesh_height(nv, x, y, seed)
    zt = smooth_terrain_height(x, y, seed)
    if lateral <= CORRIDOR_HALF_M - CORRIDOR_BLEND_M:
        return zt
    w = (lateral - (CORRIDOR_HALF_M - CORRIDOR_BLEND_M)) / CORRIDOR_BLEND_M
    w = w * w * (3 - 2 * w)
    return (1 - w) * zt + w * coarse_mesh_height(nv, x, y, seed)


def patch_river_v2(nv):
    """Coarse terrain from coarse_terrain_height (build_river_v2 replaces upstream's river afterwards)."""
    if not hasattr(nv, "_orig_build_terrain"):
        nv._orig_build_terrain = nv.build_terrain

        def build_terrain(stage, seed, assets_root):
            true_h = nv.terrain_height
            nv.terrain_height = coarse_terrain_height
            try:
                return nv._orig_build_terrain(stage, seed, assets_root)
            finally:
                nv.terrain_height = true_h

        nv.build_terrain = build_terrain
        # upstream build_river still runs (its scene_inventory reads /World/RiverWater); build_river_v2 removes it
    return nv


def _river_noise(x, k):
    return (math.sin(x / 9.7 + 1.3 * k) * 0.55 + math.sin(x / 4.3 + 2.1 * k) * 0.3 + math.sin(x / 23.0 + 0.7 * k) * 0.45)


def build_river_v2(stage, nv, seed, assets_root):
    G = nv.Gf
    for path in ("/World/RiverWater", "/World/Riverbed", "/World/Landmarks/Waterfall"):
        stage.RemovePrim(path)
    ground_m = nv.UsdShade.Material(stage.GetPrimAtPath("/World/Materials/Ground"))
    pebble_m = nv.create_pbr_material(stage, "/World/Materials/RiverPebblesV2", assets_root, "ganges_river_pebbles")
    water_m = nv.create_color_material(stage, "/World/Materials/RiverWaterV2", (0.026, 0.038, 0.026), roughness=0.3)
    xs = np.arange(RIVER_X_RANGE[0], RIVER_X_RANGE[1] + 0.01, 1.0)

    def grid_mesh(path, xs, d_of, n_across, z_of, uv_of, material):
        pts, uvs, idx, cnt = [], [], [], []
        for i, x in enumerate(xs):
            x = float(x)
            yc = nv.river_y(x)
            for j in range(n_across):
                d = d_of(x, j)
                y = yc + d
                pts.append(G.Vec3f(x, float(y), float(z_of(x, y, d))))
                uvs.append(G.Vec2f(*uv_of(x, y)))
        for i in range(len(xs) - 1):
            for j in range(n_across - 1):
                a, b = i * n_across + j, i * n_across + j + 1
                c, d = a + n_across, b + n_across
                # j runs north (+y), i runs east (+x): (a, d, b) + (a, c, d) is counter-clockwise seen from
                # above, i.e. normals point UP like the terrain's. (The first v2 render had them reversed and
                # the banks / water / shore shaded as a bright yellow band, Simulation_6 before re-render.)
                idx += [a, d, b, a, c, d]
                cnt += [3, 3]
        return nv.create_mesh(stage, path, pts, idx, cnt, uvs, material, collision=False)

    # 1. banks: fine corridor ground, same material + UVs as the terrain mesh it blends into
    n_c = int(2 * CORRIDOR_HALF_M / 0.5) + 1
    grid_mesh("/World/River/Banks", xs, lambda x, j: -CORRIDOR_HALF_M + 0.5 * j, n_c,
              lambda x, y, d: visible_ground(nv, x, y, seed),
              lambda x, y: ((x + 650.0) / 28.0, (y + 650.0) / 28.0), ground_m)
    # 2. pebble shore on both banks, irregular outer edge, 1.5 cm above the bank surface
    n_p = 25

    def peb_d(x, j):
        sh = _shore_half(x)
        left = sh + 1.3 + 0.9 * _river_noise(x, 1) + 0.5 * math.sin(x / 2.3 + 0.5)
        right = sh + 1.3 + 0.9 * _river_noise(x, 2) + 0.5 * math.sin(x / 2.9 + 1.7)
        return -left + (left + right) * j / (n_p - 1)

    grid_mesh("/World/River/Shore", xs, peb_d, n_p, lambda x, y, d: smooth_terrain_height(x, y, seed) + 0.015,
              lambda x, y: (x / 3.0, y / 3.0), pebble_m)
    # 3. water: flat across each section (meets the banks at their own height), small ripples
    n_w = 13
    grid_mesh("/World/River/Water", np.arange(RIVER_X_RANGE[0], RIVER_X_RANGE[1] + 0.01, 0.5),
              lambda x, j: -8.5 + 17.0 * j / (n_w - 1), n_w,
              lambda x, y, d: river_water_z(nv, x, seed) + 0.012 * math.sin(1.7 * x + 0.9 * d) + 0.008 * math.sin(3.1 * x - 2.3 * d),
              lambda x, y: (x / 4.0, y / 4.0), water_m)
    # 4. far river beyond the corridor (distant terrain only): simple flat ribbons at water level
    for name, part in (("West", np.linspace(-1200.0, RIVER_X_RANGE[0] + 2.0, 60)), ("East", np.linspace(RIVER_X_RANGE[1] - 2.0, 1600.0, 96))):
        pts, idx, cnt, uvs = nv.ribbon_geometry(part, 3.8, RIVER_WATER_ABOVE_BED_M + 0.05, seed)
        idx = [v for k in range(0, len(idx), 3) for v in (idx[k], idx[k + 2], idx[k + 1])]  # upstream winding faces down
        nv.create_mesh(stage, f"/World/River/Far{name}Water", pts, idx, cnt, uvs, water_m, collision=False)
    # 5. stones along the waterline and a few in the stream
    rng = np.random.default_rng(seed + 777)
    rock_m = {k: nv.create_asset_material(stage, f"/World/Materials/RiverStone_{k}", assets_root, nv.ASSET_IDS[k])
              for k in ("small_rock", "stone", "rock")}
    inst = []
    for x in np.arange(-320.0, 420.0, 3.5):
        for side in (-1, 1):
            if rng.random() < 0.45:
                continue
            xx = float(x + rng.uniform(-1.5, 1.5))
            d = side * (_shore_half(xx) + float(rng.uniform(-0.6, 1.8)))
            y = nv.river_y(xx) + d
            s = float(rng.uniform(0.35, 1.0))
            z = smooth_terrain_height(xx, y, seed) - 0.08 * s
            inst.append((int(rng.integers(0, 3)), (xx, y, z), float(rng.uniform(-math.pi, math.pi)), (s, s, s)))
    for x in np.arange(-320.0, 420.0, 11.0):
        if rng.random() < 0.5:
            continue
        xx = float(x + rng.uniform(-4, 4))
        y = nv.river_y(xx) + float(rng.uniform(-1.0, 1.0)) * (_shore_half(xx) - 1.0)
        s = float(rng.uniform(0.5, 0.9))
        inst.append((2, (xx, y, river_bed_z(nv, xx, seed) + 0.05), float(rng.uniform(-math.pi, math.pi)), (s, s, s)))
    nv.point_instancer(stage, "/World/River/Stones", [
        (nv.asset_path(assets_root, k), 1.0, rock_m[k]) for k in ("small_rock", "stone", "rock")], inst)
    info = build_waterfall_v2(stage, nv, seed, assets_root, water_m)
    info.update({"river_version": "v2-channel", "river_depth_m": RIVER_DEPTH_M, "river_water_width_m": [round(2 * min(_shore_half(x) for x in xs), 1), round(2 * max(_shore_half(x) for x in xs), 1)],
                 "river_stones": len(inst)})
    return info


def _shore_half(x=0.0):
    """Half width of the water surface at x (where the bank crosses the water level)."""
    target = RIVER_WATER_ABOVE_BED_M - RIVER_DEPTH_M
    lo, hi = 0.0, 12.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if river_channel(mid, x) < target:
            lo = mid
        else:
            hi = mid
    return lo


def build_waterfall_v2(stage, nv, seed, assets_root, water_m):
    """Dress the 4 m step at WATERFALL_X: churned water over the steep part, foam, rocks in the stream and
    boulders on both banks. All of it sits on the v2 water surface / banks."""
    G = nv.Gf
    root = "/World/Landmarks/Waterfall"
    rng = np.random.default_rng(seed + 4242)
    # (no flat "churned water" sheet -- it read as a pale panel on the river; foam streaks + rocks only)
    foam_m = nv.create_color_material(stage, "/World/Materials/FoamV2", (0.36, 0.41, 0.40), roughness=0.6)
    half = _shore_half(WATERFALL_X) - 0.4

    def blob(path, cx, cy, rx, ry, dz):
        z = river_water_z(nv, cx, seed) + dz
        p = [G.Vec3f(cx, cy, z)]
        m = 16
        for j in range(m):
            a = 2 * math.pi * j / m
            r = 1.0 + 0.3 * math.sin(3 * a + cx)
            px = cx + rx * r * math.cos(a)
            p.append(G.Vec3f(px, cy + ry * r * math.sin(a), river_water_z(nv, px, seed) + dz))
        ii, cc = [], []
        for j in range(m):
            ii += [0, 1 + j, 1 + (j + 1) % m]
            cc.append(3)
        nv.create_mesh(stage, path, p, ii, cc, None, foam_m, collision=False)

    for k in range(45):  # thin streaks along the flow, densest on the steep part
        x = WATERFALL_X + rng.normal(-0.5, 2.6)
        blob(f"{root}/Dash_{k:02d}", x, nv.river_y(x) + rng.uniform(-half, half), rng.uniform(0.2, 0.5), rng.uniform(0.02, 0.05), 0.03)
    for k in range(10):  # broken foam at the foot of the drop
        x = WATERFALL_X - 6.5 + rng.uniform(-2.5, 1.0)
        blob(f"{root}/Foam_{k}", x, nv.river_y(x) + rng.uniform(-half + 0.5, half - 0.5), rng.uniform(0.2, 0.5), rng.uniform(0.12, 0.3), 0.03)
    rock_m = nv.create_asset_material(stage, "/World/Materials/FallRock", assets_root, nv.ASSET_IDS["rock"])
    boulder_m = nv.create_asset_material(stage, "/World/Materials/FallBoulder", assets_root, nv.ASSET_IDS["boulder"])
    inst = []
    for x in np.arange(WATERFALL_X - 9.0, WATERFALL_X + 8.0, 2.0):
        for side in (-1, 1):
            if rng.random() < 0.3:
                continue
            y = nv.river_y(float(x)) + side * rng.uniform(4.5, 7.5)
            s = float(rng.uniform(0.9, 1.8))
            inst.append((int(rng.integers(0, 2)), (float(x), float(y), smooth_terrain_height(float(x), y, seed) - 0.4),
                         float(rng.uniform(-math.pi, math.pi)), (s, s, s)))
    for k in range(14):  # rocks across the lip and in the rapids
        x = WATERFALL_X + rng.uniform(-3.5, 3.5)
        y = nv.river_y(x) + rng.uniform(-1.0, 1.0) * half
        s = float(rng.uniform(0.6, 0.95))
        inst.append((0, (float(x), float(y), river_bed_z(nv, x, seed) + 0.1), float(rng.uniform(-math.pi, math.pi)), (s, s, s)))
    nv.point_instancer(stage, f"{root}/Rocks", [
        (nv.asset_path(assets_root, "rock"), 1.0, rock_m),
        (nv.asset_path(assets_root, "boulder"), 1.0, boulder_m),
    ], inst)
    return {"waterfall_x": WATERFALL_X, "waterfall_drop_m": WATERFALL_DROP_M, "waterfall_rocks": len(inst),
            "waterfall_top_z": river_water_z(nv, WATERFALL_X + 6.0, seed)}
