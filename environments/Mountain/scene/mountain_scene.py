"""Simulation_0_0_6: an authored mountain, built to the same quality bar as Simulation_0_0_5's
city after its full realism pass (real elevation, verified-detail rock/snow PBR, a puresky HDRI,
distance-haze grading, and acceleration-limited flight -- see that project's README for how each
of those was actually diagnosed, several diagnoses along the way were wrong and are recorded there
so they are not re-attempted here).

Unlike the forest (a gentle river valley, <15m of relief) or the city (flat, authored blocks),
this is real vertical terrain: a single peak rising ~185m over a ~430m radius, built as one height
field so the mesh, the treeline, the snowline and the summit landmark all agree on the same ground.

Layers, low to high:
  * valley floor -- two landing pads, flattened the same gaussian way the forest's pads were
  * lower slopes -- dirt/scree PBR (brown_mud_rocks_01, aerial_rocks_02), pine/fir forest
    (reusing the forest project's own pine_sapling_small/fir_sapling -- same species, same look)
  * rock band -- bare cliff PBR (rock_face, dry_riverbed_rock) above the treeline, boulders
    (reusing the forest's boulder_01/rock_07/rock_09/rock_moss_set_01/stone_01)
  * snow cap -- snow PBR (snow_02, snow_field_aerial) as an overlay mesh above the snowline,
    same overlay pattern the city used for sidewalks-over-roads: a second mesh following the
    identical height field, offset a few cm up, rather than per-vertex texture blending (which
    USD/UsdPreviewSurface does not support without a custom shader)
  * summit -- a small timber lookout platform (the mission target) plus a stone cairn, at the
    peak's own true height, not a guessed altitude

All texture picks were MEASURED (see fetch_mountain_assets.py's check_texture_contrast) before
being trusted -- the city's three near-blank facade textures were caught that way, not by name.
"""

import math

import numpy as np
# NOTE: no top-level `from pxr import ...` -- pxr is only importable AFTER Isaac Sim's
# SimulationApp has started (see hq_bridge.py/city_scene.py, which follow the same convention).
# This module is imported by run_mountain_mission.py before the bridge/SimulationApp exists, so a
# top-level pxr import crashes immediately -- caught here by a probe job failing in 4 seconds
# instead of partway through the real ~54-minute render. build_terrain() imports Vt locally.

# ---- layout -----------------------------------------------------------------------------------

PEAK_XY = (60.0, 40.0)
PEAK_HEIGHT_M = 185.0
BASE_RADIUS_M = 430.0          # radial scale of the PRIMARY peak's cone falloff, still used by
                                # summit_smooth/orbit framing -- NOT the visible terrain extent

# A CHAIN of peaks, not one isolated summit surrounded by flat scree. PEAK_XY (index 2, the
# primary/target) keeps its original position and height so the mission and lookout structure are
# unchanged; four more peaks flank it along a ridge line, spaced (340m) closer than their own
# radii (360-430m) so adjacent cones OVERLAP and the saddle between any two neighbours never
# drops back to open flat ground (checked offline: saddle heights 76-96m, not ~0m). Entries are
# (x, y, height_m, radius_m).
_CHAIN_ANGLE_DEG = 38.0
_CHAIN_SPACING_M = 340.0
PEAKS = [
    (-442.0, -422.0, 148.0, 360.0),
    (-229.5, -141.7, 168.0, 390.0),
    (PEAK_XY[0], PEAK_XY[1], PEAK_HEIGHT_M, BASE_RADIUS_M),   # primary / target, unchanged
    (355.6, 213.9, 158.0, 370.0),
    (558.9, 505.9, 172.0, 400.0),
]

TERRAIN_EXTENT_M = 620.0       # high-detail mesh half-extent
FAR_EXTENT_M = 3200.0          # coarse distant terrain half-extent (natural_valley-style ring)

TREELINE_Z = 92.0              # no trees above this elevation
SNOWLINE_Z = 148.0             # snow overlay mesh above this elevation
ROCKLINE_Z = 60.0              # bare rock PBR starts blending in above this (vs dirt/scree below)

START_PAD = (-315.0, 245.0)    # found by fresh offline search AFTER the single peak became a
END_PAD = (-97.0, 439.0)       # 5-peak CHAIN (see PEAKS below) -- the previous pads (r~390-410m
                                # from the primary, valid for the old isolated-peak terrain) ended
                                # up on a NEIGHBOURING peak's own flank once the chain existed
                                # (167m from peak3's centre, well inside its 370m radius), caught
                                # here rather than at the GPU. These sit in the saddle/foothill
                                # ground between chain peaks: <1m height spread across a 12m
                                # footprint, clear of every peak's own steep cone, 292m apart
                                # (matching the forest's 189m and city's 224m pad-separation scale),
                                # ~428m from the primary/target peak (similar transit distance to
                                # the single-peak version, so the render-time budget is unchanged).
PAD_FLATTEN_RADIUS_M = 55.0    # gaussian sigma of local levelling at each pad
PAD_CLEAR_RADIUS_M = 13.0      # vegetation/rock exclusion radius around each pad centre

SUMMIT_PLATFORM_R_M = 5.5      # footprint radius kept clear for the lookout structure
SUMMIT_CLEAR_R_M = 16.0        # wider vegetation/rock exclusion around the summit


def peak_r(x, y):
    return math.hypot(x - PEAK_XY[0], y - PEAK_XY[1])


def pad_level(x, y):
    """1.0 at a landing-pad centre, decaying to 0.0 away from it (gaussian, no hard edge -- a
    hard-radius flattening disc is exactly what read as 'standing in a circle' earlier this
    session). Governs crushing height toward the valley floor -- NOT used near the summit, see
    summit_smooth below: an earlier version folded both into one 'level' term and it crushed the
    whole cone toward zero at the peak, turning the mountain into a crater (base height 48m where
    it should have been 185m, caught by validate_layout before ever reaching the renderer)."""
    level = 0.0
    for px, py in (START_PAD, END_PAD):
        level = max(level, math.exp(-((math.hypot(x - px, y - py) / PAD_FLATTEN_RADIUS_M) ** 2)))
    return level


def summit_smooth(x, y):
    """1.0 at the true summit, decaying outward -- used only to damp the ROUGH NOISE near the
    peak for a buildable platform. Deliberately does NOT touch the cone term `base`: the summit's
    elevation must stay at the mountain's actual height, not get crushed toward the valley floor
    the way a landing pad does."""
    return math.exp(-((peak_r(x, y) / (SUMMIT_PLATFORM_R_M * 1.6)) ** 2))


def _chain_base(x, y):
    """The chain's combined cone shape: the MAX over every peak's own cone, so the terrain at any
    point is however tall the nearest/tallest contributing peak makes it. This is what turns
    isolated summits into a connected ridge -- with cones wider than half the inter-peak spacing
    (checked offline), the max of two overlapping cones stays well above zero all the way through
    the saddle between them, instead of each peak falling to flat ground before the next begins."""
    best = 0.0
    for px, py, h, r in PEAKS:
        d = math.hypot(x - px, y - py)
        shape = max(0.0, 1.0 - (d / r) ** 1.15) ** 1.55
        best = max(best, h * shape)
    return best


def terrain_height(x, y, seed=0):
    """Real mountain relief: a CHAIN of overlapping cones (see PEAKS/_chain_base) with ridged
    multi-octave noise for rock texture, tapering to gently rolling foothills at the visible
    horizon (never fully flat -- a dead-flat far plane is what produced the city's original hard
    edge before the fix)."""
    r = peak_r(x, y)   # distance to the PRIMARY peak only -- still used below for the noise's
                        # radial falloff and the foothill/rolling blend, which only need to soften
                        # near the one peak that has a built structure on it
    base = _chain_base(x, y)

    # elevation fraction gates how rugged the noise gets: subtle in the valley, pronounced on the
    # rock band and peak -- a mountain does not have valley-floor-grade ruggedness at the summit
    # and vice versa.
    elev_frac = min(1.0, base / PEAK_HEIGHT_M)
    amp = 1.2 + 9.0 * elev_frac
    ridged = amp * (
        1.6 * math.sin(x / 41.0 + seed * 0.019) * math.cos(y / 37.0 - seed * 0.011)
        + 1.1 * math.sin((x + y) / 29.0 + seed * 0.007)
        + 0.7 * math.cos((x - 1.7 * y) / 18.0)
        + 0.35 * math.sin((2.3 * x - y) / 9.0 + seed * 0.031)
    )

    # foothills far from the peak: gentle rolling terrain instead of a dead-flat plain, capped so
    # it never competes with the real mountain
    horizon = min(28.0, 0.02 * max(0.0, r - BASE_RADIUS_M * 0.9))
    rolling = 1.6 * math.sin(x / 210.0 + seed * 0.013) + 1.3 * math.sin(y / 240.0 - seed * 0.017)

    level = pad_level(x, y)
    smooth = max(summit_smooth(x, y), level)   # buildable flat spot at the summit too
    ridged = ridged * (1.0 - 0.95 * smooth)

    h = base + ridged + horizon + rolling * (1.0 - 0.5 * min(1.0, r / BASE_RADIUS_M))
    # crush toward the valley floor ONLY at the landing pads (`level`, from pad_level) -- the
    # summit's `smooth` term above only removed noise, it never entered this blend, so the peak
    # keeps its full `base` height.
    h = h * (1.0 - 0.97 * level) - 0.25 * level
    return max(-1.5, h)


def slope(x, y, seed=0, eps=1.5):
    """Central-difference slope magnitude (rise/run), for keeping trees off cliffs and pads level."""
    hx1 = terrain_height(x + eps, y, seed)
    hx0 = terrain_height(x - eps, y, seed)
    hy1 = terrain_height(x, y + eps, seed)
    hy0 = terrain_height(x, y - eps, seed)
    return math.hypot((hx1 - hx0) / (2 * eps), (hy1 - hy0) / (2 * eps))


def distance_to_pad(x, y):
    return min(math.hypot(x - px, y - py) for px, py in (START_PAD, END_PAD))


def validate_layout(seed=7):
    """CPU-only pre-flight, same discipline as the forest's river-offset check and the city's
    validate_layout(): catch a bad pad or a target underground before spending GPU time."""
    issues = []
    for name, (px, py) in (("START_PAD", START_PAD), ("END_PAD", END_PAD)):
        # A valley pad on a mountain has no reason to sit at literal z=0 -- what matters for a
        # safe landing is LOCAL flatness (slope, and height variance across the actual footprint),
        # not absolute elevation above some arbitrary reference. An absolute-height check made
        # sense for the flat city; it does not here.
        s = slope(px, py, seed)
        if s > 0.08:
            issues.append(f"{name} local slope {s:.3f} is too steep to land on")
        heights = [terrain_height(px + dx, py + dy, seed)
                   for dx in (-6.0, 0.0, 6.0) for dy in (-6.0, 0.0, 6.0)]
        spread = max(heights) - min(heights)
        if spread > 1.5:
            issues.append(f"{name} footprint height varies {spread:.2f}m across 12m -- not flat")
    sx, sy = PEAK_XY
    peak_h = terrain_height(sx, sy, seed)
    if peak_h < PEAK_HEIGHT_M * 0.85:
        issues.append(f"peak height {peak_h:.1f}m is far below the intended {PEAK_HEIGHT_M}m "
                      f"(pad flattening or noise is eating the summit)")
    if math.hypot(START_PAD[0] - END_PAD[0], START_PAD[1] - END_PAD[1]) < 150.0:
        issues.append("START and END pads are too close for a real A->B transit")
    for name, (px, py) in (("START_PAD", START_PAD), ("END_PAD", END_PAD)):
        if peak_r(px, py) < SUMMIT_CLEAR_R_M + 60.0:
            issues.append(f"{name} is too close to the summit platform")
    return issues, peak_h


def validate_orbit_clearance(orbit_radius, flight_altitude, seed=7, margin=3.0, step_deg=1):
    """Check the FULL 360-degree orbit ring against real terrain, not one or a few radial samples.

    A single-ray check (or a few) missed this: the terrain around the peak has real ridged noise
    (the same relief that makes the mountain look like rock), and it is NOT radially symmetric --
    one ray at 23deg showed comfortable clearance while the ring's actual worst point, at 220deg,
    had the orbit flying 2.7m INSIDE the terrain at a flight altitude that looked fine everywhere
    else checked. Call this with the mission's real orbit_radius/flight_altitude before every
    render, not just once during scene design -- a future orbit_radius or target-altitude change
    needs the same check re-run."""
    px, py = PEAK_XY
    worst = None
    for deg in range(0, 360, step_deg):
        ang = math.radians(deg)
        x, y = px + orbit_radius * math.cos(ang), py + orbit_radius * math.sin(ang)
        h = terrain_height(x, y, seed)
        clearance = flight_altitude - h
        if worst is None or clearance < worst[0]:
            worst = (clearance, deg, h)
    issues = []
    if worst[0] < margin:
        issues.append(f"orbit ring (r={orbit_radius}m, altitude={flight_altitude:.1f}m) clears "
                      f"terrain by only {worst[0]:.2f}m at {worst[1]}deg (need >={margin}m)")
    return issues, worst


# ---- shared authoring helpers (same as Simulation_0_0_5/pipeline/city_scene.py) ---------------


def textured_box(nv, stage, path, centre, size, material, tile_m=4.0, collision=False):
    """A box mesh with real per-face UVs in METRES (uv = extent / tile_m). nv.cube() makes a
    UsdGeom.Cube, which has no `st` primvar -- a PBR material on one renders untextured. This was
    the mechanism behind two separate bugs in the city project (blank facades on cubes, blank
    pilasters), so every textured surface here goes through this, never nv.cube with a PBR mat."""
    Gf = nv.Gf
    hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
    cx, cy, cz = centre
    faces = [
        ((0, 0, 1), size[0], size[1]), ((0, 0, -1), size[0], size[1]),
        ((1, 0, 0), size[1], size[2]), ((-1, 0, 0), size[1], size[2]),
        ((0, 1, 0), size[0], size[2]), ((0, -1, 0), size[0], size[2]),
    ]
    corners = {
        (0, 0, 1): [(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)],
        (0, 0, -1): [(-hx, -hy, -hz), (-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz)],
        (1, 0, 0): [(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)],
        (-1, 0, 0): [(-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), (-hx, hy, -hz)],
        (0, 1, 0): [(-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), (hx, hy, -hz)],
        (0, -1, 0): [(-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz)],
    }
    points, uvs, indices = [], [], []
    for k, (normal, u_ext, v_ext) in enumerate(faces):
        base = len(points)
        for i, (ox, oy, oz) in enumerate(corners[normal]):
            points.append(Gf.Vec3f(cx + ox, cy + oy, cz + oz))
            uu = (0.0, 1.0, 1.0, 0.0)[i] * (u_ext / tile_m)
            vv = (0.0, 0.0, 1.0, 1.0)[i] * (v_ext / tile_m)
            uvs.append(Gf.Vec2f(uu, vv))
        indices.extend([base, base + 1, base + 2, base + 3])
    return nv.create_mesh(stage, path, points, indices, [4] * 6, uvs, material, collision)


def create_pbr_material_fixed(stage, nv, path, assets_root, texture_id, uv_scale=1.0, tint=None):
    """PBR material with a correctly decoded normal map (scale/bias/sourceColorSpace) and
    optional diffuse tint via UsdUVTexture.scale, so one texture can serve several looks."""
    Sdf, Gf, UsdShade = nv.Sdf, nv.Gf, nv.UsdShade
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.85)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    st_reader = UsdShade.Shader.Define(stage, f"{path}/TexCoord")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    transform = UsdShade.Shader.Define(stage, f"{path}/Transform2d")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(uv_scale, uv_scale))

    for channel, input_name, output_name in (("diffuse", "diffuseColor", "rgb"),
                                             ("normal", "normal", "rgb"),
                                             ("roughness", "roughness", "r")):
        try:
            filename = nv.texture_path(assets_root, texture_id, channel)
        except FileNotFoundError:
            continue
        tex = UsdShade.Shader.Define(stage, f"{path}/{channel.title()}Texture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(filename)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            transform.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if channel == "diffuse" and tint is not None:
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(float(tint[0]), float(tint[1]), float(tint[2]), 1.0))
        if channel == "normal":
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2.0, 2.0, 2.0, 1.0))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1.0, -1.0, -1.0, 0.0))
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        tex.CreateOutput("r", Sdf.ValueTypeNames.Float)
        surface.CreateInput(input_name, {
            "diffuseColor": Sdf.ValueTypeNames.Color3f,
            "normal": Sdf.ValueTypeNames.Normal3f,
            "roughness": Sdf.ValueTypeNames.Float,
        }[input_name]).ConnectToSource(tex.ConnectableAPI(), output_name)
    surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface.ConnectableAPI(), "surface")
    return material


def _pbr(stage, nv, assets_root, name, texture_id, tint=None):
    """uv_scale is always 1.0: textured_box() already bakes UVs in metres, so tiling is
    controlled solely by each call site's tile_m -- the city's uv_scale-on-top-of-metres bug is
    not repeated here."""
    return create_pbr_material_fixed(stage, nv, f"/World/Materials/Mtn_{name}", assets_root,
                                     texture_id, uv_scale=1.0, tint=tint)


# ---- terrain mesh -------------------------------------------------------------------------------


def _terrain_normal(x, y, seed, eps=1.0):
    """Analytic per-vertex normal from the height field's own gradient (central difference),
    independent of the mesh's triangulation -- normalize((-dh/dx, -dh/dy, 1))."""
    hx1 = terrain_height(x + eps, y, seed)
    hx0 = terrain_height(x - eps, y, seed)
    hy1 = terrain_height(x, y + eps, seed)
    hy0 = terrain_height(x, y - eps, seed)
    dhdx = (hx1 - hx0) / (2 * eps)
    dhdy = (hy1 - hy0) / (2 * eps)
    nx, ny, nz = -dhdx, -dhdy, 1.0
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    return (nx / n, ny / n, nz / n)


def build_terrain(stage, nv, assets_root, seed=7):
    """ONE high-detail terrain mesh, banded dirt/rock/snow by real elevation via USD GeomSubsets
    (one material per face-subset) -- plus a coarse distant ring so the mountain does not end at
    a visible edge.

    This used to be THREE overlapping meshes (dirt base + a snow-cap overlay + a rock-band
    overlay), each offset a few centimetres up, the same 'stack a thin surface above the one
    below it' pattern that worked for the city's sidewalks-over-roads. It does NOT work here: the
    city's roads were dead flat, so a fixed z-offset trivially guaranteed which surface was on
    top everywhere. This mountain's height field has real ~10m-amplitude noise near the peak, and
    the three meshes sampled that noise at THREE DIFFERENT grid resolutions (161/121/81 points
    over three different extents) -- their surfaces don't agree between grid points, so a 3-5cm
    offset is nowhere near enough to guarantee stacking order, and the two overlapping surfaces
    fight for visibility face-by-face. Rendered as an unmistakable shattered-glass mosaic of
    alternating rock/snow triangles (probe traj_orbit_mid.png) -- caught by looking at a real
    frame, not diagnosed from the code. One mesh with per-face subsets has exactly one surface, so
    there is nothing left to fight."""
    from pxr import Vt   # lazy import -- see module-level note on why pxr can't be imported at
                         # module scope here

    rock = _pbr(stage, nv, assets_root, "Rock", "rock_face")
    dirt = _pbr(stage, nv, assets_root, "Dirt", "brown_mud_rocks_01")
    snow = _pbr(stage, nv, assets_root, "Snow", "snow_field_aerial")

    Gf, UsdGeom, UsdShade = nv.Gf, nv.UsdGeom, nv.UsdShade
    grid_n = 221
    coords = np.linspace(-TERRAIN_EXTENT_M, TERRAIN_EXTENT_M, grid_n)
    heights = np.array([[terrain_height(float(x), float(y), seed) for x in coords] for y in coords])
    points, uvs, normals = [], [], []
    for iy, y in enumerate(coords):
        for ix, x in enumerate(coords):
            h = float(heights[iy, ix])
            points.append(Gf.Vec3f(float(x), float(y), h))
            uvs.append(Gf.Vec2f(float((x + TERRAIN_EXTENT_M) / 22.0), float((y + TERRAIN_EXTENT_M) / 22.0)))
            nx, ny, nz = _terrain_normal(float(x), float(y), seed)
            normals.append(Gf.Vec3f(nx, ny, nz))

    indices, counts = [], []
    band_faces = {"dirt": [], "rock": [], "snow": []}
    face_i = 0
    for row in range(grid_n - 1):
        for col in range(grid_n - 1):
            a = row * grid_n + col
            b, c, d = a + 1, a + grid_n, a + grid_n + 1
            indices.extend([a, b, d, a, d, c])
            counts.extend([3, 3])
            # classify this quad by the average height of its 4 corners
            hq = (heights[row, col] + heights[row, col + 1] + heights[row + 1, col]
                  + heights[row + 1, col + 1]) / 4.0
            band = "snow" if hq >= SNOWLINE_Z else ("rock" if hq >= ROCKLINE_Z else "dirt")
            band_faces[band].extend([face_i, face_i + 1])   # the quad's 2 triangles
            face_i += 2

    mesh = nv.create_mesh(stage, "/World/Terrain", points, indices, counts, uvs, None,
                          collision=False)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    for band, material in (("dirt", dirt), ("rock", rock), ("snow", snow)):
        faces = band_faces[band]
        if not faces:
            continue
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, f"Band_{band}", UsdGeom.Tokens.face,
                                                 Vt.IntArray(faces))
        UsdShade.MaterialBindingAPI(subset.GetPrim()).Bind(material)

    band_counts = {k: len(v) // 2 for k, v in band_faces.items()}

    # distant ring: natural_valley's own build_distant_terrain pattern (a coarse mesh with a hole
    # under the high-detail one), tinted toward the sky's horizon colour by distance so the far
    # foothills fade instead of stopping at a hard edge -- computed the same way the city's haze
    # ratio was: measured diffuse mean vs the HDRI horizon colour, not guessed.
    HAZE_RATIO = (1.05, 1.10, 1.32)   # dirt's warm diffuse -> the puresky horizon's cool blue-grey
    far_mats = [_pbr(stage, nv, assets_root, f"FarDirt{k}", "brown_mud_rocks_01",
                     tint=tuple(1.0 + (k / 4.0) * (r - 1.0) for r in HAZE_RATIO))
                for k in range(5)]
    Gf = nv.Gf
    coords = np.linspace(-FAR_EXTENT_M, FAR_EXTENT_M, 161)
    points, uvs = [], []
    for y in coords:
        for x in coords:
            r = math.hypot(float(x), float(y))
            fade = min(1.0, max(0.0, (r - TERRAIN_EXTENT_M * 0.85) / (FAR_EXTENT_M - TERRAIN_EXTENT_M)))
            h = terrain_height(float(x), float(y), seed) * (1.0 - 0.4 * fade) - 0.55
            points.append(Gf.Vec3f(float(x), float(y), float(h)))
            uvs.append(Gf.Vec2f(float((x + FAR_EXTENT_M) / 34.0), float((y + FAR_EXTENT_M) / 34.0)))
    indices, counts = [], []
    for row in range(160):
        for col in range(160):
            x0, x1 = coords[col], coords[col + 1]
            y0, y1 = coords[row], coords[row + 1]
            if max(abs(x0), abs(x1), abs(y0), abs(y1)) < TERRAIN_EXTENT_M * 0.82:
                continue
            a = row * 161 + col
            b, c, d = a + 1, a + 161, a + 161 + 1
            # pick the haze level from distance for this quad's outer corner
            r_out = math.hypot(max(abs(x0), abs(x1)), max(abs(y0), abs(y1)))
            lvl = min(4, int((r_out - TERRAIN_EXTENT_M) / (FAR_EXTENT_M - TERRAIN_EXTENT_M) * 5))
            indices.extend([a, b, d, a, d, c])
            counts.extend([3, 3])
    # single-material distant mesh at the mid haze level keeps this from needing per-quad material
    # assignment (USD GeomSubsets would work but cost more setup for a barely-visible ring);
    # visually the far ring reads as haze because it sits far below the eyeline at flight altitude
    nv.create_mesh(stage, "/World/DistantTerrain", points, indices, counts, uvs, far_mats[2],
                   collision=False)

    return {"terrain_extent_m": TERRAIN_EXTENT_M, "far_extent_m": FAR_EXTENT_M,
            "terrain_faces_dirt": band_counts["dirt"], "terrain_faces_rock": band_counts["rock"],
            "terrain_faces_snow": band_counts["snow"],
            "peak_height_m": terrain_height(PEAK_XY[0], PEAK_XY[1], seed)}


# ---- summit landmark ----------------------------------------------------------------------------


def build_summit(stage, nv, assets_root, forest_assets_root, seed=7):
    """A timber lookout platform at the true summit (the mission target), plus a stone cairn and a
    radio mast beside it -- built with the same relief discipline the city's buildings ended up
    needing (real geometry, not a single flat-shaded box): legs, railings with posts and rails,
    a hipped roof, and a stacked cairn rather than one grey cylinder."""
    Gf = nv.Gf
    px, py = PEAK_XY
    gz = terrain_height(px, py, seed)  # ground at the true summit -- read, not guessed

    # PBR only on surfaces built with textured_box (real UVs): the deck and the roof slabs.
    wood = _pbr(stage, nv, assets_root, "Timber", "brown_mud_rocks_01", tint=(1.25, 0.95, 0.62))
    roof_mat = _pbr(stage, nv, assets_root, "LookoutRoof", "dry_riverbed_rock", tint=(0.82, 0.55, 0.40))
    # Everything else here is built with nv.cylinder/nv.cube -- UsdGeom.Cylinder/Cube, neither of
    # which carries an `st` primvar, so a PBR material on one renders untextured (the same failure
    # mode as the city's blank pilasters). Flat colour for these small structural/trim elements,
    # the same convention the city used for its own trim, cornices and tower mast.
    wood_flat = nv.create_color_material(stage, "/World/Materials/MtnWoodFlat", (0.42, 0.27, 0.15), 0.8)
    metal = nv.create_color_material(stage, "/World/Materials/MtnMetal", (0.42, 0.43, 0.46), 0.5,
                                     metallic=0.65)
    stone_tones = [nv.create_color_material(stage, f"/World/Materials/MtnCairn{i}", c, 0.85)
                  for i, c in enumerate(((0.52, 0.50, 0.48), (0.46, 0.45, 0.44), (0.58, 0.55, 0.51)))]

    deck_w, deck_d, deck_h = 5.2, 5.2, 0.22
    leg_h = 1.7
    deck_z = gz + leg_h

    root = "/World/Summit"
    # four support legs, following the ground (same height here since the summit is levelled)
    for i, (lx, ly) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        nv.cylinder(stage, f"{root}/Leg{i}",
                    (px + lx * (deck_w / 2 - 0.4), py + ly * (deck_d / 2 - 0.4), gz + leg_h / 2.0),
                    0.14, leg_h, wood_flat)
    # deck
    textured_box(nv, stage, f"{root}/Deck", (px, py, deck_z), (deck_w, deck_d, deck_h), wood,
                tile_m=1.4)

    # railing: posts + a top rail on three sides, leaving the fourth (south, -y) open as the
    # approach/entry the way a real lookout platform has a gap for the ladder
    rail_h = 1.0
    post_positions = []
    for side, sx, sy in (("N", 0.0, 1.0), ("E", 1.0, 0.0), ("W", -1.0, 0.0)):
        span = deck_w - 0.5 if side in ("N",) else deck_d - 0.5
        for t in (-0.5, 0.0, 0.5):
            ox = t * span if side == "N" else sx * (deck_w / 2 - 0.15)
            oy = sy * (deck_d / 2 - 0.15) if side != "N" else sy * (deck_d / 2 - 0.15)
            if side in ("E", "W"):
                oy = t * span
            post_positions.append((ox, oy))
    for i, (ox, oy) in enumerate(post_positions):
        nv.cylinder(stage, f"{root}/RailPost{i}", (px + ox, py + oy, deck_z + deck_h / 2 + rail_h / 2),
                    0.045, rail_h, metal)
    for side, sx, sy, span in (("N", 0.0, 1.0, deck_w - 0.3), ("E", 1.0, 0.0, deck_d - 0.3),
                               ("W", -1.0, 0.0, deck_d - 0.3)):
        rx = sx * (deck_w / 2 - 0.15) if side != "N" else 0.0
        ry = sy * (deck_d / 2 - 0.15)
        w = span if side == "N" else 0.08
        d = 0.08 if side == "N" else span
        nv.cube(stage, f"{root}/Rail_{side}", (px + rx, py + ry, deck_z + deck_h / 2 + rail_h),
               (w, d, 0.06), metal)

    # hipped roof: two sloped slabs meeting at a ridge, plus gable end caps -- cheap to build,
    # reads correctly from above (which is the only view this mission ever gets of it)
    roof_z = deck_z + 2.1
    ridge_len = deck_w * 0.5
    slope_len = math.hypot(deck_d / 2 + 0.5, 0.9)
    slope_ang = math.degrees(math.atan2(0.9, deck_d / 2 + 0.5))
    for side, sy in (("N", 1.0), ("S", -1.0)):
        textured_box(nv, stage, f"{root}/RoofSlope{side}",
                    (px, py + sy * (deck_d / 4.2), roof_z - 0.35),
                    (deck_w + 0.6, slope_len, 0.10), roof_mat, tile_m=1.2)
    nv.cube(stage, f"{root}/RoofRidge", (px, py, roof_z + 0.55), (deck_w + 0.5, 0.18, 0.18), wood_flat)
    for i, (lx, ly) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        nv.cylinder(stage, f"{root}/RoofPost{i}",
                    (px + lx * (deck_w / 2 - 0.3), py + ly * (deck_d / 2 - 0.3),
                     deck_z + deck_h / 2 + (roof_z - deck_z) / 2),
                    0.06, roof_z - deck_z - 0.3, wood_flat)

    # radio mast, offset from the platform so it doesn't visually merge with the roof
    mast_x, mast_y = px - deck_w * 0.9, py - deck_d * 0.9
    mast_gz = terrain_height(mast_x, mast_y, seed)
    mast_h = 9.0
    nv.cylinder(stage, f"{root}/Mast", (mast_x, mast_y, mast_gz + mast_h / 2), 0.08, mast_h, metal)
    nv.cylinder(stage, f"{root}/MastTip", (mast_x, mast_y, mast_gz + mast_h + 0.6), 0.03, 1.2, metal)

    # stone cairn beside the platform -- graduated stacked blocks, not one cylinder
    cx, cy = px + deck_w * 0.95, py + deck_d * 0.6
    cgz = terrain_height(cx, cy, seed)
    z = cgz
    for i, (s, h) in enumerate(((1.3, 0.5), (1.0, 0.4), (0.72, 0.35), (0.45, 0.3), (0.22, 0.35))):
        nv.cylinder(stage, f"{root}/Cairn{i}", (cx, cy, z + h / 2), s / 2, h,
                   stone_tones[i % len(stone_tones)])
        z += h

    return {"summit_x": px, "summit_y": py, "summit_ground_z": gz,
            "summit_deck_z": deck_z + deck_h, "summit_roof_top_z": roof_z + 0.65,
            "summit_mast_tip_z": mast_gz + mast_h + 1.2}


# ---- vegetation & rock scatter -------------------------------------------------------------------


def _clear_of_pads(x, y, radius=PAD_CLEAR_RADIUS_M):
    return all(math.hypot(x - px, y - py) >= radius for px, py in (START_PAD, END_PAD))


def _clear_of_summit(x, y, radius=SUMMIT_CLEAR_R_M):
    return peak_r(x, y) >= radius


def build_vegetation(stage, nv, forest_assets_root, seed=13, terrain_seed=7, area_radius_m=560.0):
    """Pine/fir forest on the lower slopes, thinning toward the treeline; boulders on the rock
    band and above. Reuses the forest project's own proven assets (same species look), scattered
    by ELEVATION and SLOPE so the bands read as real terrain zones rather than a uniform sprinkle.

    `seed` (offset from the terrain's own seed) drives the RANDOM SAMPLING only -- distinct from
    the terrain's seed on purpose, so vegetation doesn't sample in lockstep with anything else
    built from the same base seed. `terrain_seed` is the ACTUAL seed the terrain mesh was built
    with, and every terrain_height()/slope() call in this function must use it: passing `seed`
    to those instead (an earlier version did) computes z against a height field the rendered
    terrain does NOT match -- trees and boulders floating above or sunk into the real ground,
    caught in the trajectory probe as an oddly airborne rock cluster, not by inspecting the code.

    Sampling is over a BOUNDING BOX covering the whole chain (`area_radius_m` used as a half-
    extent here), not a radius from the primary peak alone -- radial-from-primary sampling would
    leave the other four peaks in the chain bare of trees/boulders while the primary got all of
    them, which is exactly the kind of empty-looking terrain a mountain CHAIN should not have."""
    rng = np.random.default_rng(seed)
    keys = ("fir", "pine", "boulder", "rock", "small_rock", "moss_rocks", "stone")
    materials = {key: nv.create_asset_material(stage, f"/World/Materials/MtnVeg_{key}",
                                               forest_assets_root, nv.ASSET_IDS[key])
                for key in keys}
    box = min(area_radius_m, TERRAIN_EXTENT_M - 20.0)

    def sample(count, ok_fn, scale_fn, protos, max_tries_factor=45):
        out = []
        tries = 0
        while len(out) < count and tries < count * max_tries_factor:
            tries += 1
            x = float(rng.uniform(-box, box))
            y = float(rng.uniform(-box, box))
            if not (_clear_of_pads(x, y) and _clear_of_summit(x, y)):
                continue
            z = terrain_height(x, y, terrain_seed)
            if not ok_fn(x, y, z):
                continue
            s = scale_fn()
            out.append((int(rng.integers(0, protos)), (x, y, z),
                       float(rng.uniform(-math.pi, math.pi)), (s, s, s)))
        return out

    counts = {}

    # trees: below the treeline, avoid steep rock (slope() is expensive-ish but this is CPU-side
    # authoring, run once, not per-frame)
    def tree_ok(x, y, z):
        return z < TREELINE_Z and slope(x, y, terrain_seed) < 0.75

    # Counts raised for the chain: box-sampling over a fixed area spread across 5 peaks is
    # inherently sparser per-peak than the old radial-around-one-summit sampling at the same
    # count (measured ~22% density drop for an unchanged count) -- raised to compensate and then
    # some, since filling in the ridge/saddle terrain between peaks is the actual point.
    trees = sample(4200, tree_ok, lambda: float(rng.uniform(1.4, 3.4)), 2)
    nv.point_instancer(stage, "/World/Vegetation/Trees", [
        (nv.asset_path(forest_assets_root, "fir"), 1.0, materials["fir"]),
        (nv.asset_path(forest_assets_root, "pine"), 1.0, materials["pine"]),
    ], trees)
    counts["trees"] = len(trees)

    # boulders: on the rock band and above (steeper, higher terrain), plus scattered lower down
    # where a real slope would still show exposed rock
    def rock_ok(x, y, z):
        return z > ROCKLINE_Z - 20.0 or slope(x, y, terrain_seed) > 0.5

    rocks = sample(1600, rock_ok, lambda: float(rng.uniform(0.6, 2.4)), 4)
    nv.point_instancer(stage, "/World/Vegetation/Rocks", [
        (nv.asset_path(forest_assets_root, "boulder"), 1.0, materials["boulder"]),
        (nv.asset_path(forest_assets_root, "rock"), 1.0, materials["rock"]),
        (nv.asset_path(forest_assets_root, "small_rock"), 1.0, materials["small_rock"]),
        (nv.asset_path(forest_assets_root, "moss_rocks"), 1.0, materials["moss_rocks"]),
    ], rocks)
    counts["rocks"] = len(rocks)

    counts["vegetation_total"] = sum(counts.values())
    return counts


# ---- lighting -------------------------------------------------------------------------------------


def build_lighting(stage, nv, assets_root, hdri_id="kloofendal_48d_partly_cloudy_puresky"):
    """Sky-only HDRI (verified the same way the city's replacement was: fetched, tone-mapped, and
    actually looked at) + a sun. Mountain light is harsher/more directional than the city's."""
    from pathlib import Path

    UsdLux, Gf = nv.UsdLux, nv.Gf
    sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
    sky.CreateIntensityAttr(880.0)
    hdr = Path(assets_root) / "hdri" / hdri_id / f"{hdri_id}_2k.hdr"
    if hdr.exists():
        sky.CreateTextureFileAttr().Set(str(hdr))
        sky.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    sky.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 25.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1500.0)
    sun.CreateAngleAttr(0.55)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
    sun.AddRotateXYZOp().Set(Gf.Vec3f(52.0, -12.0, 40.0))
    return {"hdri": hdri_id if hdr.exists() else None}


def build_mountain(stage, nv, assets_root, forest_assets_root, seed=7):
    info = {}
    build_lighting(stage, nv, assets_root)
    info.update(build_terrain(stage, nv, assets_root, seed=seed))
    info.update(build_summit(stage, nv, assets_root, forest_assets_root, seed=seed))
    info.update(build_vegetation(stage, nv, forest_assets_root, seed=seed + 5, terrain_seed=seed))
    info["peak_xy"] = list(PEAK_XY)
    return info
