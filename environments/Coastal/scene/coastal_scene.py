"""Simulation_0_0_7: an authored coastline, built to the same quality bar as the mountain
(Simulation_0_0_6) after ITS realism pass -- real elevation, a CHAIN of headlands (not one
isolated bluff with empty ground everywhere else, exactly the lesson from that project), verified-
detail PBR, a puresky HDRI, distance haze, acceleration-limited flight, authored terrain normals,
and ONE terrain mesh banded by elevation via USD GeomSubset rather than overlapping meshes.

Layout: a meandering shoreline (`shore_y(x)`) with 5 headlands overlapping along it -- ocean on
one side, a continuous coastal bluff on the other, never a flat empty gap between headlands
(checked offline: saddle land-relief 15.6-28.2m, never near 0m -- same discipline that caught the
mountain's flat inter-peak gaps). The tallest, central headland carries the mission target: a real
lighthouse, not a guessed structure -- built with a tapered tower, gallery, railing, and lamp room.

Bands, low to high (elevation-classified, same GeomSubset technique as the mountain's dirt/rock/
snow -- NOT overlapping meshes, which z-fought on the mountain's high-relief terrain):
  * sand   -- beach/tideline, coast_sand_05
  * rock   -- cliff face, cliff_side / coast_land_rocks_02
  * grass  -- clifftop, aerial_grass_rock (reused from the forest's own pack)

Ocean is one large flat plane (real water needs reflections, not surface albedo detail -- a low-
roughness tinted material is the honest choice here, not a texture that would just fight the
requirement to look wet and reflective).
"""

import math

import numpy as np
# NOTE: no top-level `from pxr import ...` -- pxr is only importable AFTER Isaac Sim's
# SimulationApp has started (see hq_bridge.py / mountain_scene.py, same convention). This module
# is imported before the bridge exists, so a top-level pxr import would crash the mission before
# it ever reaches the GPU. build_terrain() imports Vt locally.

# ---- layout -----------------------------------------------------------------------------------


def shore_y(x):
    """Meandering coastline centreline -- land is on the +lateral side, ocean on the -lateral
    side (see terrain_height's `lateral` computation)."""
    return 55.0 * math.sin(x / 300.0) + 22.0 * math.sin(x / 130.0 + 0.6)


# A CHAIN of headlands, not one isolated bluff -- overlapping cones (radius > half the inter-
# headland spacing) so the coastal bluff stays continuous between them (checked offline: saddle
# land-relief 15.6-28.2m, never collapsing to open flat ground). Entries are
# (x, y, height_m, radius_m); the CENTRE headland (index 2) is the primary/target, carrying the
# lighthouse.
HEADLANDS = [
    (-480.0, -21.1, 36.0, 190.0),
    (-230.0, -8.4, 52.0, 220.0),
    (0.0, 72.4, 70.0, 250.0),      # primary / target
    (250.0, 101.5, 46.0, 210.0),
    (480.0, 76.9, 38.0, 185.0),
]
LIGHTHOUSE_XY = (HEADLANDS[2][0], HEADLANDS[2][1])
LIGHTHOUSE_BASE_HEIGHT_M = HEADLANDS[2][2]

BEACH_WIDTH_M = 40.0            # metres over which land blends to sea across the shoreline
SEA_DEPTH_M = -4.0               # shallow -- invisible under the opaque ocean plane regardless
SEA_LEVEL_Z = 0.0

TERRAIN_EXTENT_M = 700.0         # high-detail mesh half-extent
FAR_EXTENT_M = 3200.0            # coarse distant terrain half-extent

SAND_Z = 3.0                     # below this: sand
CLIFFTOP_Z = 34.0                # between SAND_Z and this: rock cliff face; above: grass

START_PAD = (-77.0, 295.0)       # re-found after the original pads (technically on 16m-high dry
END_PAD = (175.0, 371.0)         # land) still read as "starting from the ocean": the ocean plane
                                  # is a giant flat slab under the ENTIRE map, and a level (pitch=0)
                                  # takeoff-phase camera shows the true horizon in every direction
                                  # -- wherever the shoreline curves close enough within the
                                  # camera's view, the ocean becomes visible and, being reflective,
                                  # dominates the lower frame at a grazing angle even from several
                                  # hundred metres away. The old pads had a dip to open water only
                                  # ~320-575m off-axis, well inside the camera's actual field of
                                  # view (traj_takeoff.png/coastal_start_0.png showed it plainly).
                                  # These sit further from the shoreline: verified with a FULL
                                  # 360-degree sweep (not just the camera's nominal facing, since
                                  # the drone's start heading is an arbitrary default) that no
                                  # terrain point below 5m elevation exists within ~315m in ANY
                                  # direction. <1m height spread across a 12m footprint, slope~0,
                                  # 263m apart, 236-346m from the lighthouse.
PAD_FLATTEN_RADIUS_M = 50.0
PAD_CLEAR_RADIUS_M = 13.0

LIGHTHOUSE_PLATFORM_R_M = 6.0
LIGHTHOUSE_CLEAR_R_M = 18.0


def lateral(x, y):
    """Signed distance-like coordinate: positive = inland, negative = out to sea."""
    return y - shore_y(x)


def headland_r(x, y):
    """Distance to the PRIMARY headland (lighthouse site) only."""
    return math.hypot(x - LIGHTHOUSE_XY[0], y - LIGHTHOUSE_XY[1])


def _land_relief(x, y):
    """Max over every headland's own cone -- see HEADLANDS docstring above."""
    best = 0.0
    for px, py, h, r in HEADLANDS:
        d = math.hypot(x - px, y - py)
        shape = max(0.0, 1.0 - (d / r) ** 1.15) ** 1.55
        best = max(best, h * shape)
    return best


def _smoothstep(edge0, edge1, x):
    t = max(0.0, min(1.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def pad_level(x, y):
    if START_PAD is None or END_PAD is None:
        return 0.0
    level = 0.0
    for px, py in (START_PAD, END_PAD):
        level = max(level, math.exp(-((math.hypot(x - px, y - py) / PAD_FLATTEN_RADIUS_M) ** 2)))
    return level


def lighthouse_smooth(x, y):
    """1.0 at the lighthouse site, decaying outward -- damps ROUGH NOISE only (never the base
    land_relief), same summit_smooth pattern that avoided crushing the mountain's peak into a
    crater."""
    return math.exp(-((headland_r(x, y) / (LIGHTHOUSE_PLATFORM_R_M * 1.6)) ** 2))


def _plain_baseline(lat):
    """A modest coastal-plain/bluff elevation that exists EVERYWHERE inland, independent of
    headland proximity -- ramps from 0 at the beach to a 16m plateau further inland. Without this,
    `_land_relief` alone (a max over spatially-LOCALIZED headland cones) decays back toward 0 far
    from every headland even at large positive lateral (deep inland), so the terrain between
    headlands collapsed back to sea level instead of staying dry land -- an offline pad search
    found only 4 flat, dry candidates in the whole map before this fix (all near-water slivers),
    the same 'no landable ground away from the peak' failure the mountain project had, just
    re-introduced here by dropping its `rolling`/`horizon` baseline term."""
    return min(16.0, max(0.0, 0.12 * (lat - BEACH_WIDTH_M)))


def structural_height(x, y):
    """The coastline's shape WITHOUT the ridged rock-texture noise -- headland cones blended to
    sea level across the shoreline, nothing else. Used for the terrain mesh's material-band
    classification (sand/rock/grass): rock must mean a genuinely steep HEADLAND FLANK, not a
    locally noisy patch of otherwise-flat ground. The noise term's own spatial gradient has an
    amplitude around 0.9 (from its ~17m-wavelength component) -- comfortably ABOVE the 0.55 rock
    threshold on its own, so classifying from the noisy height flagged ~3% of quads on the
    genuinely flat coastal plain as "rock" purely from decorative micro-texture (measured near
    END_PAD: 39 of 1156 sampled quads), rendering as a fine checkerboard of fake rock patches
    across open grassland (traj_land.png) -- misdiagnosed twice (as inland sand, then as a
    terrain-mesh/distant-ring seam) before actually comparing noisy vs. noise-free slope."""
    lat = lateral(x, y)
    land = max(_land_relief(x, y), _plain_baseline(lat))
    s = _smoothstep(-BEACH_WIDTH_M, BEACH_WIDTH_M, lat)
    return SEA_DEPTH_M * (1.0 - s) + land * s


def structural_slope(x, y, eps=3.0):
    hx1 = structural_height(x + eps, y)
    hx0 = structural_height(x - eps, y)
    hy1 = structural_height(x, y + eps)
    hy0 = structural_height(x, y - eps)
    return math.hypot((hx1 - hx0) / (2 * eps), (hy1 - hy0) / (2 * eps))


def terrain_height(x, y, seed=0):
    """A real coastline: overlapping headland cones ADDED TO a modest coastal-plain baseline
    (so the whole shore has real dry land, not just the headlands), blended smoothly down to sea
    level across the shoreline, with ridged noise for rock texture and a shallow, flat-ish seabed
    (invisible under the ocean plane) far from shore."""
    lat = lateral(x, y)
    land = max(_land_relief(x, y), _plain_baseline(lat))

    elev_frac = min(1.0, land / max(h for _, _, h, _ in HEADLANDS))
    amp = 1.0 + 7.0 * elev_frac
    ridged = amp * (
        1.5 * math.sin(x / 39.0 + seed * 0.017) * math.cos(y / 35.0 - seed * 0.013)
        + 1.0 * math.sin((x + y) / 27.0 + seed * 0.009)
        + 0.6 * math.cos((x - 1.6 * y) / 17.0)
    )

    # Pads get the SAME treatment as the lighthouse platform: damp the ROUGH NOISE only, never
    # crush the underlying `land` elevation toward a valley floor. The mountain's pads legitimately
    # wanted to sit near z=0 (the valley floor), so crushing toward 0 there was correct; these
    # pads were found on a genuinely flat ~18m coastal plateau, and crushing them the same way
    # pulled the exact pad centre down to ~0.3m (near sea level) while the surrounding footprint
    # stayed near 18m -- a locally IMPOSSIBLE landing surface, caught by validate_layout's own
    # "too close to sea level" check flagging a pad that was, by construction, nowhere near the
    # sea. Do not reintroduce a height-crushing term here.
    level = pad_level(x, y)
    smooth = max(lighthouse_smooth(x, y), level)
    ridged = ridged * (1.0 - 0.95 * smooth)

    land_h = land + ridged
    s = _smoothstep(-BEACH_WIDTH_M, BEACH_WIDTH_M, lat)
    h = SEA_DEPTH_M * (1.0 - s) + land_h * s
    return h


def slope(x, y, seed=0, eps=1.5):
    hx1 = terrain_height(x + eps, y, seed)
    hx0 = terrain_height(x - eps, y, seed)
    hy1 = terrain_height(x, y + eps, seed)
    hy0 = terrain_height(x, y - eps, seed)
    return math.hypot((hx1 - hx0) / (2 * eps), (hy1 - hy0) / (2 * eps))


def _terrain_normal(x, y, seed, eps=1.0):
    hx1 = terrain_height(x + eps, y, seed)
    hx0 = terrain_height(x - eps, y, seed)
    hy1 = terrain_height(x, y + eps, seed)
    hy0 = terrain_height(x, y - eps, seed)
    dhdx = (hx1 - hx0) / (2 * eps)
    dhdy = (hy1 - hy0) / (2 * eps)
    nx, ny, nz = -dhdx, -dhdy, 1.0
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    return (nx / n, ny / n, nz / n)


def validate_layout(seed=7):
    """CPU-only pre-flight, same discipline as the mountain's validate_layout(): catch a bad pad
    or a target underground before spending GPU time."""
    issues = []
    if START_PAD is None or END_PAD is None:
        return ["START_PAD/END_PAD not set"], 0.0
    for name, (px, py) in (("START_PAD", START_PAD), ("END_PAD", END_PAD)):
        s = slope(px, py, seed)
        if s > 0.08:
            issues.append(f"{name} local slope {s:.3f} is too steep to land on")
        heights = [terrain_height(px + dx, py + dy, seed)
                   for dx in (-6.0, 0.0, 6.0) for dy in (-6.0, 0.0, 6.0)]
        spread = max(heights) - min(heights)
        if spread > 1.5:
            issues.append(f"{name} footprint height varies {spread:.2f}m across 12m -- not flat")
        if terrain_height(px, py, seed) < SAND_Z + 2.0:
            issues.append(f"{name} sits too close to sea level -- would be on the beach/underwater")
    lh_x, lh_y = LIGHTHOUSE_XY
    lh_h = terrain_height(lh_x, lh_y, seed)
    if lh_h < LIGHTHOUSE_BASE_HEIGHT_M * 0.85:
        issues.append(f"lighthouse site height {lh_h:.1f}m is far below the intended "
                      f"{LIGHTHOUSE_BASE_HEIGHT_M}m")
    if math.hypot(START_PAD[0] - END_PAD[0], START_PAD[1] - END_PAD[1]) < 150.0:
        issues.append("START and END pads are too close for a real A->B transit")
    for name, (px, py) in (("START_PAD", START_PAD), ("END_PAD", END_PAD)):
        if headland_r(px, py) < LIGHTHOUSE_CLEAR_R_M + 40.0:
            issues.append(f"{name} is too close to the lighthouse platform")
    return issues, lh_h


def validate_orbit_clearance(orbit_radius, flight_altitude, seed=7, margin=3.0, step_deg=1):
    """Check the FULL 360-degree orbit ring against real terrain -- a single radial sample missed
    a real clipping problem on the mountain project; never repeat that shortcut."""
    px, py = LIGHTHOUSE_XY
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


def distance_to_pad(x, y):
    if START_PAD is None or END_PAD is None:
        return 1e9
    return min(math.hypot(x - px, y - py) for px, py in (START_PAD, END_PAD))


# ---- shared authoring helpers (same as the mountain/city projects) ----------------------------


def textured_box(nv, stage, path, centre, size, material, tile_m=4.0, collision=False):
    """A box mesh with real per-face UVs in METRES (uv = extent / tile_m). nv.cube() makes a
    UsdGeom.Cube, which has no `st` primvar -- a PBR material on one renders untextured."""
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


def create_pbr_material_fixed(stage, nv, path, assets_root, texture_id, uv_scale=1.0, tint=None,
                              use_normal=True, roughness=0.85, ior=1.5, roughness_map=True):
    """PBR material with a correctly decoded normal map (optional) and optional diffuse tint.

    `use_normal=False` drops the normal map entirely. Diagnosed with a controlled probe
    (water_pitch00/20/40.png): at 40deg pitch the terrain read as correct grass; at 20deg, the
    NEAR field (steep local incidence) was correct grass while the FAR field (always near-grazing
    incidence to a roughly flat surface, regardless of the camera's own pitch -- distance alone
    guarantees this) showed a glassy, wave-like sheen easily mistaken for standing water. Simply
    reducing the normal map's strength (scale 2.0 -> 1.3) was NOT enough -- verified with the same
    probe, image unchanged. The terrain's own authored smooth vertex normals (from the height
    field's analytic gradient) already provide real large-scale shading variation; the normal MAP
    only adds fine bump detail that is barely visible from flight altitude and is what compounds
    with Fresnel reflection into this artifact at any grazing view. Used for the terrain bands
    (sand/rock/grass); kept on for the lighthouse/cottage, where the camera is normally closer
    and at a steeper angle."""
    Sdf, Gf, UsdShade = nv.Sdf, nv.Gf, nv.UsdShade
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    # ior (index of refraction) sets the STRENGTH of Fresnel grazing-angle brightening, largely
    # independent of roughness -- the standard dielectric default (1.5, ~4% reflectance at normal
    # incidence rising toward 100% at grazing) is calibrated for glass/plastic/water, not dry
    # grass or sand, and was the remaining driver of this project's grazing-angle "water" artifact
    # even after removing the normal map and raising roughness (that suppressed the sharp bump-
    # driven ripple and the specular LOBE shape, but not the underlying Fresnel curve itself,
    # which was still visibly reflecting nearby vegetation and sky). Lower for terrain.
    surface.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(ior)
    st_reader = UsdShade.Shader.Define(stage, f"{path}/TexCoord")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    transform = UsdShade.Shader.Define(stage, f"{path}/Transform2d")
    transform.CreateIdAttr("UsdTransform2d")
    transform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result")
    transform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(uv_scale, uv_scale))

    channels = [("diffuse", "diffuseColor", "rgb")] + ([("roughness", "roughness", "r")] if roughness_map else [])
    if use_normal:
        channels.insert(1, ("normal", "normal", "rgb"))
    for channel, input_name, output_name in channels:
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
            # XY scale reduced from the standard 2.0 (full-strength [0,1]->[-1,1] decode) to 1.3:
            # at the shallow gimbal angles this coastal mission's takeoff/land phases actually use
            # (drone near ground level, camera close to level), full-strength bump normals
            # compounded with grazing-incidence sun reflection into a shimmering, wave-like glare
            # across the grass (traj_takeoff.png) -- easy to mistake for standing water. Z stays
            # at the full decode so the surface still reads as bumpy up close, just less violently
            # at grazing incidence.
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(1.3, 1.3, 2.0, 1.0))
            tex.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-0.65, -0.65, -1.0, 0.0))
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


def _pbr(stage, nv, assets_root, name, texture_id, tint=None, use_normal=True, roughness=0.85,
        ior=1.5):
    """uv_scale always 1.0 -- textured_box() already bakes UVs in metres."""
    return create_pbr_material_fixed(stage, nv, f"/World/Materials/Coast_{name}", assets_root,
                                     texture_id, uv_scale=1.0, tint=tint, use_normal=use_normal,
                                     roughness=roughness, ior=ior)


# ---- terrain mesh -------------------------------------------------------------------------------


def build_terrain(stage, nv, assets_root, seed=7):
    """ONE high-detail terrain mesh, banded sand/rock/grass by real elevation via USD GeomSubsets
    (one material per face-subset -- the mountain project tried overlapping meshes first and they
    z-fought on high-relief terrain; not repeated here), an ocean plane, and a coarse distant ring
    so the coastline does not end at a visible edge."""
    from pxr import Vt   # lazy import -- pxr is only available after SimulationApp starts

    # use_normal=False on all three -- see create_pbr_material_fixed's docstring for the
    # grazing-incidence artifact this was diagnosed against.
    # roughness pushed to near-max and ior dropped toward 1.0 (minimal Fresnel contrast) -- see
    # create_pbr_material_fixed's docstring on the grazing-angle artifact these two, together with
    # use_normal=False, were needed to actually resolve (removing the normal map alone fixed the
    # sharp ripple pattern but left a broad reflective wash, including recognisable mirror
    # reflections of nearby vegetation, that neither change on its own eliminated).
    sand = _pbr(stage, nv, assets_root, "Sand", "coast_sand_05", use_normal=False,
               roughness=0.97, ior=1.05)
    rock = _pbr(stage, nv, assets_root, "Rock", "cliff_side", use_normal=False,
               roughness=0.95, ior=1.1)
    grass = _pbr(stage, nv, assets_root, "Grass", "aerial_grass_rock", use_normal=False,
                roughness=0.98, ior=1.02)

    Gf, UsdGeom, UsdShade = nv.Gf, nv.UsdGeom, nv.UsdShade
    grid_n = 241
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
    band_faces = {"sand": [], "rock": [], "grass": []}
    face_i = 0
    for row in range(grid_n - 1):
        for col in range(grid_n - 1):
            a = row * grid_n + col
            b, c, d = a + 1, a + grid_n, a + grid_n + 1
            indices.extend([a, b, d, a, d, c])
            counts.extend([3, 3])
            h00, h01 = heights[row, col], heights[row, col + 1]
            h10, h11 = heights[row + 1, col], heights[row + 1, col + 1]
            hq = (h00 + h01 + h10 + h11) / 4.0
            qx = (coords[col] + coords[col + 1]) / 2.0
            qy = (coords[row] + coords[row + 1]) / 2.0
            # Classify by STRUCTURAL slope (headland shape only, no decorative rock-texture
            # noise) -- slope from the rendered/noisy height flagged ~3% of the geometrically flat
            # coastal plain as "rock" purely from the noise term's own gradient (its ~17m-
            # wavelength component has slope amplitude ~0.9, above the 0.55 threshold on its own),
            # rendering as a checkerboard of fake rock patches across open grassland. See
            # structural_height()'s docstring -- this was found and fixed AFTER two other
            # plausible-looking but wrong diagnoses (inland sand leakage, a mesh/ring seam).
            local_slope = structural_slope(qx, qy)
            # "sand" also requires being NEAR THE ACTUAL SHORELINE (small |lateral|), not just low
            # elevation: the coastal-plain baseline is only 16m and ridged noise can locally dip a
            # few metres below SAND_Z far inland, at a real bay, painting isolated fake "beach"
            # patches kilometres from any water.
            near_shore = abs(lateral(qx, qy)) < BEACH_WIDTH_M * 1.6
            if hq < SAND_Z and near_shore:
                band = "sand"
            elif local_slope > 0.55:
                band = "rock"
            else:
                band = "grass"
            band_faces[band].extend([face_i, face_i + 1])
            face_i += 2

    mesh = nv.create_mesh(stage, "/World/Terrain", points, indices, counts, uvs, None,
                          collision=False)
    mesh.CreateNormalsAttr(normals)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    for band, material in (("sand", sand), ("rock", rock), ("grass", grass)):
        faces = band_faces[band]
        if not faces:
            continue
        subset = UsdGeom.Subset.CreateGeomSubset(mesh, f"Band_{band}", UsdGeom.Tokens.face,
                                                 Vt.IntArray(faces))
        UsdShade.MaterialBindingAPI(subset.GetPrim()).Bind(material)
    band_counts = {k: len(v) // 2 for k, v in band_faces.items()}

    # ocean: one large flat, low-roughness tinted plane. Real water needs reflections (which
    # RealTimePathTracing gives a low-roughness surface for free), not surface-albedo texture
    # detail -- a texture here would just fight the requirement to look wet.
    ocean_info = build_ocean(stage, nv, seed)

    # distant ring: coarse mesh with a hole under the high-detail one, tinted toward the sky's
    # horizon colour by distance (measured diffuse mean vs the HDRI horizon, same method the city
    # and mountain projects used -- NOT guessed tints, which drifted the wrong direction the first
    # time this was tried on the city).
    #
    # fade must be EXACTLY 0 at r=TERRAIN_EXTENT_M and the hole must be cut at that SAME radius --
    # the first version faded starting at 0.85x the extent but only cut the hole at 0.82x, leaving
    # a ~125m overlap band with two independently-computed heights (a real bug, fixed and kept).
    #
    # FLAT COLOUR, not a PBR texture: a tiled PBR texture (34m/repeat over a 3200m ring) viewed at
    # the shallow gimbal pitch the 'land'/'takeoff' phases actually use (looking toward a target
    # hundreds of metres away, from near ground level) puts a huge span of this ring's own
    # repeating pattern into a few rows of screen pixels near the image horizon -- classic
    # minification aliasing, rendered as a persistent pale checkerboard mosaic (traj_land.png) that
    # survived three separate GEOMETRY/CLASSIFICATION fixes untouched, because none of them were
    # the actual cause: the ring was never misclassified, it was alias ing its own fine tiling.
    # A flat, haze-graded colour has no repeating pattern to alias, at any viewing angle.
    HAZE_RATIO = (0.97, 1.28, 2.01)   # dry_ground_rocks's diffuse mean -> puresky horizon colour
    BASE_FAR_RGB = (0.583, 0.453, 0.307)   # dry_ground_rocks's own measured diffuse mean
    far_mats = [nv.create_color_material(stage, f"/World/Materials/FarLand{k}",
                                         tuple(c * (1.0 + (k / 4.0) * (ratio - 1.0))
                                              for c, ratio in zip(BASE_FAR_RGB, HAZE_RATIO)), 0.85)
                for k in range(5)]
    def _far_h(x, y):
        r = math.hypot(x, y)
        fade = min(1.0, max(0.0, (r - TERRAIN_EXTENT_M) / (FAR_EXTENT_M - TERRAIN_EXTENT_M)))
        return terrain_height(x, y, seed) * (1.0 - 0.5 * fade) - 0.55 * fade

    def _far_normal(x, y, eps=8.0):
        dhdx = (_far_h(x + eps, y) - _far_h(x - eps, y)) / (2 * eps)
        dhdy = (_far_h(x, y + eps) - _far_h(x, y - eps)) / (2 * eps)
        nxr, nyr, nzr = -dhdx, -dhdy, 1.0
        n = math.sqrt(nxr * nxr + nyr * nyr + nzr * nzr)
        return (nxr / n, nyr / n, nzr / n)

    # authored normals, same fix as the high-detail mesh: without them this coarse (161-point over
    # 3200m -> ~40m/cell) mesh shaded as visibly faceted once its flat-colour material stopped
    # masking it behind texture noise (see the flat-colour fix above).
    coords_f = np.linspace(-FAR_EXTENT_M, FAR_EXTENT_M, 161)
    points_f, uvs_f, normals_f = [], [], []
    for y in coords_f:
        for x in coords_f:
            h = _far_h(float(x), float(y))
            points_f.append(Gf.Vec3f(float(x), float(y), float(h)))
            uvs_f.append(Gf.Vec2f(float((x + FAR_EXTENT_M) / 34.0), float((y + FAR_EXTENT_M) / 34.0)))
            nxr, nyr, nzr = _far_normal(float(x), float(y))
            normals_f.append(Gf.Vec3f(nxr, nyr, nzr))
    indices_f, counts_f = [], []
    for row in range(160):
        for col in range(160):
            x0, x1 = coords_f[col], coords_f[col + 1]
            y0, y1 = coords_f[row], coords_f[row + 1]
            if max(abs(x0), abs(x1), abs(y0), abs(y1)) < TERRAIN_EXTENT_M:
                continue
            a = row * 161 + col
            b, c, d = a + 1, a + 161, a + 161 + 1
            indices_f.extend([a, b, d, a, d, c])
            counts_f.extend([3, 3])
    far_mesh = nv.create_mesh(stage, "/World/DistantTerrain", points_f, indices_f, counts_f,
                              uvs_f, far_mats[2], collision=False)
    far_mesh.CreateNormalsAttr(normals_f)
    far_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    return {"terrain_extent_m": TERRAIN_EXTENT_M, "far_extent_m": FAR_EXTENT_M,
            "terrain_faces_sand": band_counts["sand"], "terrain_faces_rock": band_counts["rock"],
            "terrain_faces_grass": band_counts["grass"],
            "lighthouse_ground_m": terrain_height(LIGHTHOUSE_XY[0], LIGHTHOUSE_XY[1], seed),
            **ocean_info}


def _wave_h(x, y, seed):
    """Gentle swell, not a dead-flat mirror -- two long-wavelength components (a real ocean's
    dominant swell direction plus a cross-swell) so the surface has genuine large-scale
    undulation. Small amplitude (~0.4m) and long wavelengths (35-70m) keep it well clear of the
    nearby beach terrain -- checked offline against the coastline's own lowest points -- while
    still being visible from a low-altitude orbit."""
    return (0.28 * math.sin(x / 42.0 + y / 61.0 + seed * 0.021)
            + 0.18 * math.sin(x / 27.0 - y / 33.0 - seed * 0.014)
            + 0.10 * math.cos(x / 71.0 + seed * 0.008))


def _ocean_normal(x, y, seed, eps=6.0):
    dhdx = (_wave_h(x + eps, y, seed) - _wave_h(x - eps, y, seed)) / (2 * eps)
    dhdy = (_wave_h(x, y + eps, seed) - _wave_h(x, y - eps, seed)) / (2 * eps)
    nx, ny, nz = -dhdx, -dhdy, 1.0
    n = math.sqrt(nx * nx + ny * ny + nz * nz)
    return (nx / n, ny / n, nz / n)


def build_ocean(stage, nv, seed=7):
    """A real sea surface, not a dead-flat mirror plane: gentle wave geometry (authored normals,
    same technique as the terrain) plus a foam ribbon that actually follows the meandering
    shoreline curve, not just a uniform colour band.

    Material is less aggressively reflective than the first version (metallic 0.35, roughness
    0.06 -- close to a literal mirror): at the shallow gimbal angles the coastal mission's
    takeoff/land phases actually use, a near-perfect mirror picks up strong Fresnel reflection of
    whatever is overhead (sky, or a nearby lit hillside) and reads as a flat, artificial-looking
    wash rather than water. Toned down to metallic 0.18 / roughness 0.14, with the wave normals
    doing the work of breaking up reflections the way real chop does."""
    Gf, UsdGeom = nv.Gf, nv.UsdGeom
    ocean_mat = nv.create_color_material(stage, "/World/Materials/Ocean", (0.04, 0.16, 0.22),
                                         0.14, metallic=0.18)
    OCEAN_BASE_Z = SEA_LEVEL_Z - 0.20   # extra clearance below the lowest nearby beach terrain

    grid_n = 161
    coords = np.linspace(-FAR_EXTENT_M, FAR_EXTENT_M, grid_n)
    points, uvs, normals = [], [], []
    for y in coords:
        for x in coords:
            h = OCEAN_BASE_Z + _wave_h(float(x), float(y), seed)
            points.append(Gf.Vec3f(float(x), float(y), float(h)))
            uvs.append(Gf.Vec2f(float((x + FAR_EXTENT_M) / 90.0), float((y + FAR_EXTENT_M) / 90.0)))
            nx, ny, nz = _ocean_normal(float(x), float(y), seed)
            normals.append(Gf.Vec3f(nx, ny, nz))
    indices, counts = [], []
    for row in range(grid_n - 1):
        for col in range(grid_n - 1):
            a = row * grid_n + col
            b, c, d = a + 1, a + grid_n, a + grid_n + 1
            indices.extend([a, b, d, a, d, c])
            counts.extend([3, 3])
    ocean_mesh = nv.create_mesh(stage, "/World/Ocean", points, indices, counts, uvs, ocean_mat,
                                collision=False)
    ocean_mesh.CreateNormalsAttr(normals)
    ocean_mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    # foam ribbon: follows the REAL shoreline curve (shore_y), not a straight band, restricted to
    # the high-detail extent where it's actually visible up close. A flat, low-roughness colour --
    # foam is white spray/bubbles, not something that should pick up sky reflections.
    foam_mat = nv.create_color_material(stage, "/World/Materials/Foam", (0.92, 0.94, 0.93), 0.75)
    foam_half_w = 5.0
    n_seg = 140
    fpoints, fuvs, findices, fcounts = [], [], [], []
    for i in range(n_seg + 1):
        x = -TERRAIN_EXTENT_M + 2 * TERRAIN_EXTENT_M * i / n_seg
        sy = shore_y(x)
        z = OCEAN_BASE_Z + _wave_h(x, sy, seed) + 0.06
        fpoints.append(Gf.Vec3f(float(x), float(sy - foam_half_w), float(z)))
        fpoints.append(Gf.Vec3f(float(x), float(sy + foam_half_w), float(z)))
        fuvs.append(Gf.Vec2f(float(i) / 6.0, 0.0))
        fuvs.append(Gf.Vec2f(float(i) / 6.0, 1.0))
    for i in range(n_seg):
        a, b = 2 * i, 2 * i + 1
        c, d = 2 * i + 2, 2 * i + 3
        findices.extend([a, b, d, a, d, c])
        fcounts.extend([3, 3])
    nv.create_mesh(stage, "/World/Foam", fpoints, findices, fcounts, fuvs, foam_mat,
                   collision=False)

    return {"ocean_base_z": OCEAN_BASE_Z}


# ---- lighthouse landmark ------------------------------------------------------------------------


def build_lighthouse(stage, nv, assets_root, seed=7):
    """A real lighthouse: tapered painted tower, gallery with railing posts, black lamp room, a
    domed cap, and a keeper's cottage beside the base -- at the headland's own true height, not a
    guessed altitude.

    The cottage walls/roof use textured_box (real UVs) so their PBR materials actually render;
    the tower/gallery/lamp-room/dome/finial are built from nv.cylinder, which is a UsdGeom.
    Cylinder with NO st primvar -- a PBR material on one renders untextured (same failure mode as
    the mountain's cylinder legs), so those stay flat colour by construction, not by oversight."""
    Gf = nv.Gf
    px, py = LIGHTHOUSE_XY
    gz = terrain_height(px, py, seed)   # ground at the true headland -- read, not guessed

    root = "/World/Lighthouse"

    # painted tower: stacked, tapering cylinder segments, alternating white/red bands
    white = nv.create_color_material(stage, "/World/Materials/LhWhite", (0.92, 0.91, 0.87), 0.55)
    red = nv.create_color_material(stage, "/World/Materials/LhRed", (0.62, 0.10, 0.09), 0.55)
    dark = nv.create_color_material(stage, "/World/Materials/LhDark", (0.10, 0.10, 0.11), 0.4)
    metal = nv.create_color_material(stage, "/World/Materials/LhMetal", (0.45, 0.46, 0.48), 0.5,
                                     metallic=0.6)
    glow = nv.create_color_material(stage, "/World/Materials/LhGlow", (1.0, 0.92, 0.55), 0.2)

    n_segs = 6
    seg_h = 5.0
    r0, r1 = 4.0, 2.6
    z = gz
    for i in range(n_segs):
        t0, t1 = i / n_segs, (i + 1) / n_segs
        r_mid = r0 + (r1 - r0) * (t0 + t1) / 2.0
        mat = red if i % 2 == 1 else white   # alternating bands, red on odd segments
        nv.cylinder(stage, f"{root}/Tower{i}", (px, py, z + seg_h / 2.0), r_mid, seg_h, mat)
        z += seg_h
    gallery_z = z   # top of the tower = the gallery deck level (also the mission target altitude)

    # gallery deck + railing posts (posts only, no ring rail -- see module note: the mission never
    # flies close enough at ground level to need one, same simplification the mountain's lookout
    # made for its own railing top rail on curved runs)
    nv.cylinder(stage, f"{root}/GalleryDeck", (px, py, gallery_z + 0.2), r1 + 0.5, 0.4, dark)
    n_posts = 16
    for i in range(n_posts):
        ang = 2.0 * math.pi * i / n_posts
        rx, ry = px + (r1 + 0.35) * math.cos(ang), py + (r1 + 0.35) * math.sin(ang)
        nv.cylinder(stage, f"{root}/RailPost{i}", (rx, ry, gallery_z + 0.4 + 0.55), 0.05, 1.1, metal)

    # lamp room + domed cap + finial
    lamp_h = 3.0
    nv.cylinder(stage, f"{root}/LampRoom", (px, py, gallery_z + 0.4 + lamp_h / 2.0), 2.0, lamp_h, dark)
    lamp_top = gallery_z + 0.4 + lamp_h
    nv.cylinder(stage, f"{root}/Lantern", (px, py, lamp_top - lamp_h / 2.0), 1.5, 0.3, glow)
    nv.cylinder(stage, f"{root}/Dome0", (px, py, lamp_top + 0.4), 1.6, 0.8, metal)
    nv.cylinder(stage, f"{root}/Dome1", (px, py, lamp_top + 1.3), 0.9, 1.0, metal)
    finial_top = lamp_top + 1.8 + 1.5
    nv.cylinder(stage, f"{root}/Finial", (px, py, lamp_top + 1.8 + 0.75), 0.07, 1.5, metal)

    # keeper's cottage beside the tower base -- real UV-textured walls + roof
    wall_mat = _pbr(stage, nv, assets_root, "Cottage", "cliff_side", tint=(1.55, 1.45, 1.30))
    roof_mat = _pbr(stage, nv, assets_root, "CottageRoof", "coast_land_rocks_02",
                    tint=(0.85, 0.55, 0.42))
    cx, cy = px + 11.0, py - 3.0
    cw, cd, ch = 9.0, 6.0, 4.2
    textured_box(nv, stage, f"{root}/Cottage", (cx, cy, gz + ch / 2.0), (cw, cd, ch), wall_mat,
                tile_m=2.0)
    roof_z = gz + ch + 1.1
    slope_len = math.hypot(cd / 2 + 0.5, 1.3)
    for side, sy in (("N", 1.0), ("S", -1.0)):
        textured_box(nv, stage, f"{root}/CottageRoof{side}", (cx, cy + sy * (cd / 4.0), roof_z - 0.5),
                    (cw + 0.6, slope_len, 0.12), roof_mat, tile_m=1.4)
    nv.cube(stage, f"{root}/CottageRidge", (cx, cy, roof_z + 0.7), (cw + 0.4, 0.2, 0.2), white)

    # door (facing the tower, i.e. the cottage's west wall), two windows, and a chimney -- a
    # blank-walled box reads as a shed, not a keeper's cottage someone actually lives in.
    door_mat = nv.create_color_material(stage, "/World/Materials/CottageDoor", (0.25, 0.16, 0.10), 0.7)
    glass_mat = nv.create_color_material(stage, "/World/Materials/CottageGlass", (0.10, 0.16, 0.20),
                                         0.15, metallic=0.25)
    door_w, door_h = 1.1, 2.15
    nv.cube(stage, f"{root}/CottageDoor", (cx - cw / 2.0 - 0.06, cy, gz + door_h / 2.0),
           (0.12, door_w, door_h), door_mat)
    # two windows spaced ALONG the south wall (varying x, fixed y at the wall's own face) and one
    # centred on the north wall (fixed x, varying y) -- windows on a wall must vary along the
    # wall's own length, not step further into the building.
    win_w, win_h = 1.15, 1.1
    for i, wx in enumerate((cx - cw / 4.0, cx + cw / 4.0)):
        nv.cube(stage, f"{root}/CottageWinS{i}", (wx, cy - cd / 2.0 - 0.06, gz + 2.35),
               (win_w, 0.1, win_h), glass_mat)
    nv.cube(stage, f"{root}/CottageWinN", (cx, cy + cd / 2.0 + 0.06, gz + 2.35),
           (win_w, 0.1, win_h), glass_mat)
    nv.cube(stage, f"{root}/CottageChimney", (cx + cw / 2.0 - 1.3, cy, roof_z + 1.1),
           (0.8, 0.8, 1.6), red)

    return {"lighthouse_x": px, "lighthouse_y": py, "lighthouse_ground_z": gz,
            "lighthouse_gallery_z": gallery_z, "lighthouse_lamp_top_z": lamp_top,
            "lighthouse_finial_top_z": finial_top}


# ---- vegetation & rock scatter -------------------------------------------------------------------


def _clear_of_pads(x, y, radius=PAD_CLEAR_RADIUS_M):
    return all(math.hypot(x - px, y - py) >= radius for px, py in (START_PAD, END_PAD))


def _clear_of_lighthouse(x, y, radius=LIGHTHOUSE_CLEAR_R_M):
    return headland_r(x, y) >= radius


def build_vegetation(stage, nv, forest_assets_root, seed=13, terrain_seed=7, box=560.0):
    """Coastal grass and shrub on the clifftop, thinning toward the cliff edge; boulders on the
    rock band. Reuses the forest's own grass/shrub/rock assets.

    `seed` drives random SAMPLING only; `terrain_seed` MUST equal the terrain mesh's own seed for
    every terrain_height()/slope() call -- the mountain project found 3,550 trees floating/sunk
    because an earlier version conflated the two, so this project keeps them separate from the
    start rather than as an afterthought fix."""
    rng = np.random.default_rng(seed)
    keys = ("grass", "shrub", "boulder", "rock", "small_rock", "moss_rocks", "stone")
    materials = {key: nv.create_asset_material(stage, f"/World/Materials/CoastVeg_{key}",
                                               forest_assets_root, nv.ASSET_IDS[key])
                for key in keys}
    box = min(box, TERRAIN_EXTENT_M - 20.0)

    def sample(count, ok_fn, scale_fn, protos, max_tries_factor=45):
        out = []
        tries = 0
        while len(out) < count and tries < count * max_tries_factor:
            tries += 1
            x = float(rng.uniform(-box, box))
            y = float(rng.uniform(-box, box))
            if not (_clear_of_pads(x, y) and _clear_of_lighthouse(x, y)):
                continue
            z = terrain_height(x, y, terrain_seed)
            if not ok_fn(x, y, z):
                continue
            s = scale_fn()
            out.append((int(rng.integers(0, protos)), (x, y, z),
                       float(rng.uniform(-math.pi, math.pi)), (s, s, s)))
        return out

    counts = {}

    # grass/shrub: dry land, gentler ground, below the cliff band
    # Same slope-based rule as the terrain mesh's own band classification (not an elevation cap --
    # an elevation-only rule would exclude grass from the whole ~16-18m coastal plain, the same
    # bug fixed in build_terrain's own banding).
    # structural_slope (not slope(), which includes the decorative rock-texture noise), same fix
    # as the terrain mesh's own band classification -- otherwise grass gets excluded, and boulders
    # scattered, on noise-driven false "steep" spots across genuinely flat ground.
    def grass_ok(x, y, z):
        return z > SAND_Z and structural_slope(x, y) <= 0.55

    grass = sample(5200, grass_ok, lambda: float(rng.uniform(0.8, 2.0)), 2)
    nv.point_instancer(stage, "/World/Vegetation/GrassShrub", [
        (nv.asset_path(forest_assets_root, "grass"), 1.0, materials["grass"]),
        (nv.asset_path(forest_assets_root, "shrub"), 1.0, materials["shrub"]),
    ], grass)
    counts["grass_shrub"] = len(grass)

    # rocks/boulders: the cliff band and steep ground
    def rock_ok(x, y, z):
        return structural_slope(x, y) > 0.55

    rocks = sample(1800, rock_ok, lambda: float(rng.uniform(0.6, 2.6)), 4)
    nv.point_instancer(stage, "/World/Vegetation/Rocks", [
        (nv.asset_path(forest_assets_root, "boulder"), 1.0, materials["boulder"]),
        (nv.asset_path(forest_assets_root, "rock"), 1.0, materials["rock"]),
        (nv.asset_path(forest_assets_root, "small_rock"), 1.0, materials["small_rock"]),
        (nv.asset_path(forest_assets_root, "moss_rocks"), 1.0, materials["moss_rocks"]),
    ], rocks)
    counts["rocks"] = len(rocks)

    counts["vegetation_total"] = sum(counts.values())
    return counts


def build_beach_props(stage, nv, forest_assets_root, seed=21, terrain_seed=7, box=560.0):
    """Driftwood and scattered stones along the actual sand band -- an empty beach is one of the
    least realistic things a coastline can show. Reuses the forest's dead-log model as driftwood
    (a bare fallen trunk reads as driftwood at this scale and camera distance) and its small-rock
    assets. Placement uses the SAME sand classification as the terrain mesh's own band (near the
    shoreline, low elevation) so props never appear stranded inland or sitting in open water."""
    rng = np.random.default_rng(seed)
    keys = ("dead_log", "small_rock", "stone")
    materials = {key: nv.create_asset_material(stage, f"/World/Materials/Beach_{key}",
                                               forest_assets_root, nv.ASSET_IDS[key])
                for key in keys}
    box = min(box, TERRAIN_EXTENT_M - 20.0)

    def sand_ok(x, y, z):
        return z > 0.0 and z < SAND_Z and abs(lateral(x, y)) < BEACH_WIDTH_M * 1.6

    out = []
    tries = 0
    target_count = 260
    while len(out) < target_count and tries < target_count * 260:
        tries += 1
        x = float(rng.uniform(-box, box))
        y = float(rng.uniform(-box, box))
        if not (_clear_of_pads(x, y) and _clear_of_lighthouse(x, y)):
            continue
        z = terrain_height(x, y, terrain_seed)
        if not sand_ok(x, y, z):
            continue
        proto = int(rng.integers(0, 3))
        s = float(rng.uniform(0.4, 1.1)) if proto == 0 else float(rng.uniform(0.3, 0.9))
        out.append((proto, (x, y, z), float(rng.uniform(-math.pi, math.pi)), (s, s, s)))

    nv.point_instancer(stage, "/World/Vegetation/BeachProps", [
        (nv.asset_path(forest_assets_root, "dead_log"), 1.0, materials["dead_log"]),
        (nv.asset_path(forest_assets_root, "small_rock"), 1.0, materials["small_rock"]),
        (nv.asset_path(forest_assets_root, "stone"), 1.0, materials["stone"]),
    ], out)
    return {"beach_props": len(out)}


# ---- lighting -------------------------------------------------------------------------------------


def build_lighting(stage, nv, assets_root, hdri_id="kloofendal_48d_partly_cloudy_puresky"):
    """Sky-only HDRI (already verified for the mountain project -- fetched, tone-mapped, and
    looked at) + a sun."""
    from pathlib import Path

    UsdLux, Gf = nv.UsdLux, nv.Gf
    sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
    sky.CreateIntensityAttr(900.0)
    hdr = Path(assets_root) / "hdri" / hdri_id / f"{hdri_id}_2k.hdr"
    if hdr.exists():
        sky.CreateTextureFileAttr().Set(str(hdr))
        sky.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    sky.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 15.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    # Reduced from 1450: even with the normal map removed (which fixed the wave-ripple pattern),
    # a bright direct sun still produces real Fresnel brightening on any dielectric surface at
    # grazing incidence -- inherent to distant, roughly-flat terrain viewed from low altitude,
    # not fixable by camera angle. Toning down the light source directly reduces how washed-out
    # that grazing band reads, without touching materials further.
    sun.CreateIntensityAttr(1050.0)
    sun.CreateAngleAttr(0.55)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.92))
    sun.AddRotateXYZOp().Set(Gf.Vec3f(48.0, -14.0, 32.0))
    return {"hdri": hdri_id if hdr.exists() else None}


def build_coastal(stage, nv, assets_root, forest_assets_root, seed=7):
    info = {}
    build_lighting(stage, nv, assets_root)
    info.update(build_terrain(stage, nv, assets_root, seed=seed))
    info.update(build_lighthouse(stage, nv, assets_root, seed=seed))
    info.update(build_vegetation(stage, nv, forest_assets_root, seed=seed + 5, terrain_seed=seed))
    info.update(build_beach_props(stage, nv, forest_assets_root, seed=seed + 9, terrain_seed=seed))
    info["lighthouse_xy"] = list(LIGHTHOUSE_XY)
    return info
