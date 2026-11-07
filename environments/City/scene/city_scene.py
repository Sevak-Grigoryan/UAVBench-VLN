"""City scene for Simulation_0_0_5, authored from real CC0 Poly Haven assets.

WHY AUTHORED RATHER THAN LOADED: NVIDIA's Isaac library has no city building content (ArchVis is
interior furniture, Industrial/Buildings holds only a Warehouse) and its single outdoor
environment, Rivermark, measured ~19 minutes to stream with broken upstream texture references
(probe job 256410). So the city is built the same way the forest got its quality in
Simulations_0_0_4: real PBR materials + real props, on geometry we control -- which also means we
know exactly where the landing pads and the target are, instead of reverse-engineering someone
else's environment.

WHY THE TARGET IS A TOWER AND NOT A BUS STATION: there is no bus model and no shelter model in any
available library. Both would have to be faked from boxes, and the drone orbits at ~35m, close
enough that a boxy bus (real ones have curved roofs, wheels, window bands, mirrors) would be the
least convincing thing on screen. A tower is the opposite case -- real towers ARE prismatic
volumes with window grids, floor bands and a parapet, so authored geometry plus real facade
textures (rectangular_facade_tiles / concrete_tile_facade) reads as a genuine building. It is also
a better VLN landmark: tall, visible from a distance, and natural to orbit.

Everything here uses natural_valley.py's material/instancer helpers against the city asset root,
so the PBR setup is identical to the forest's.
"""

import math

import numpy as np

# ---- layout constants -------------------------------------------------------------------
BLOCK_PITCH_M = 62.0          # street grid spacing (building block + road)
ROAD_HALF_WIDTH_M = 7.0       # 14m carriageway
SIDEWALK_M = 3.2
SIDEWALK_H = 0.16
CITY_HALF_EXTENT_M = 250.0    # ground plane half-size
BLOCK_MARGIN_M = 4.0          # gap between the sidewalk edge and a building footprint
CURB_H = 0.14                 # sidewalks sit on a real curb rather than floating above the road
LANE_MARK_W = 0.16            # painted centre-line width

# Block CENTRES are at odd multiples of BLOCK_PITCH/2 (31m). Multiples of the pitch itself are the
# road centrelines -- putting the tower or a pad there drops it in the middle of an intersection,
# which is exactly what validate_layout() caught on the first attempt.
TOWER_POS = (31.0, 31.0)      # a block centre, so the tower occupies its own block
TOWER_PLAZA_R = 18.0          # block interior half-width is 20.8m, so the plaza fits inside it
TOWER_HEIGHT_M = 58.0
TOWER_W = 16.0
TOWER_D = 16.0

START_PAD = (-93.0, 93.0)     # block centre, 31m clear of any carriageway (see validate_layout)
END_PAD = (93.0, -31.0)       # a DIFFERENT block centre -- the mission ends here, not at START
PAD_CLEAR_R = 16.0

# Only diffuse maps that actually contain detail. Measured std over each diffuse image:
#   brick_4 23.8, concrete_panels 17.7, concrete_wall_008 12.6  -> keep
#   beige_wall_001 0.35, white_stucco 0.92, concrete_tile_facade 2.2, rectangular_facade_tiles 3.9
# Those last four are effectively blank images (Poly Haven wall sets whose detail lives entirely
# in the normal map), which is why most buildings rendered as flat painted boxes even though the
# textures were binding correctly. Per-building tinting supplies the colour variety they used to.
FACADE_TEXTURES = ["brick_4", "concrete_panels", "concrete_wall_008"]
_FACADE_TEXTURES_UNUSED = ["brick_4", "beige_wall_001", "white_stucco", "concrete_panels",
                   "concrete_tile_facade", "concrete_wall_008"]
ROOF_TEXTURES = ["ceramic_roof_01", "clay_roof_tiles"]
TOWER_FACADE = "rectangular_facade_tiles"

PROP_MODELS = {
    "street_lamp_01": "street_lamp_01",
    "street_lamp_02": "street_lamp_02",
    "fire_hydrant": "fire_hydrant",
    "bench": "painted_wooden_bench",
    "seating": "modular_street_seating",
    "trash_can": "metal_trash_can",
    "barrier": "concrete_road_barrier",
    "planter": "planter_box_01",
    "poles": "modular_electricity_poles",
    "car": "covered_car",
    "fire_escape": "modular_fire_escape",
}


def model_path(assets_root, asset_id):
    """Path to a downloaded Poly Haven model. natural_valley.asset_path() only resolves ids that
    live in its own ASSET_IDS table, and the city props are not in it, so resolve directly."""
    from pathlib import Path

    return str(Path(assets_root) / "models" / asset_id / f"{asset_id}_1k.usdc")



def textured_box(nv, stage, path, centre, size, material, tile_m=4.0, collision=False):
    """A box built as a MESH with real per-face UVs, so PBR textures actually appear.

    This exists because of a real bug caught in the first city render (job 256526): the buildings
    came out as flat untextured boxes. create_pbr_material() samples its textures through a
    UsdPrimvarReader_float2 bound to the "st" primvar, but nv.cube() makes a UsdGeom.Cube, which
    carries NO st primvar -- so every textured cube rendered with no valid UV lookup. Upstream only
    ever uses cube() with flat colour materials (wood/stone/canvas), never with textured PBR, which
    is why the bug is not visible there.

    UVs are laid out in METRES divided by tile_m, so a texture repeats every tile_m of real surface
    regardless of how big the box is -- brick stays brick-sized on a 34m tower and on a 9m shed.
    """
    cx, cy, cz = centre
    w, d, h = size
    hx, hy, hz = w / 2.0, d / 2.0, h / 2.0
    Gf = nv.Gf

    faces = [
        # (four corner offsets, u-extent, v-extent)
        ([(-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz)], w, d),      # +Z
        ([(-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz)], w, d),  # -Z
        ([(hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz)], d, h),      # +X
        ([(-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz)], d, h),  # -X
        ([(-hx, hy, -hz), (hx, hy, -hz), (hx, hy, hz), (-hx, hy, hz)], w, h),      # +Y
        ([(hx, -hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (hx, -hy, hz)], w, h),  # -Y
    ]
    points, uvs, indices = [], [], []
    for corners, u_ext, v_ext in faces:
        base = len(points)
        for k, (ox, oy, oz) in enumerate(corners):
            points.append(Gf.Vec3f(cx + ox, cy + oy, cz + oz))
            uu = (0.0, 1.0, 1.0, 0.0)[k] * (u_ext / tile_m)
            vv = (0.0, 0.0, 1.0, 1.0)[k] * (v_ext / tile_m)
            uvs.append(Gf.Vec2f(uu, vv))
        indices.extend([base, base + 1, base + 2, base + 3])
    return nv.create_mesh(stage, path, points, indices, [4] * 6, uvs, material, collision)


def road_centre_offsets(half_extent=CITY_HALF_EXTENT_M, pitch=BLOCK_PITCH_M):
    """World coordinates of the street centrelines, in both axes."""
    n = int(half_extent / pitch)
    return [i * pitch for i in range(-n, n + 1)]


def on_road(x, y, margin=0.0):
    """True if (x,y) lies on a carriageway (either axis of the grid)."""
    half = ROAD_HALF_WIDTH_M + margin
    for c in road_centre_offsets():
        if abs(x - c) <= half or abs(y - c) <= half:
            return True
    return False


def near_centreline(s, margin=0.0):
    """True if the SINGLE coordinate s is within margin of a road centreline. Distinct from
    on_road(), which is an OR over both axes -- using on_road() to test one axis is always True
    at a road centre and silently discards everything."""
    return any(abs(s - c) <= ROAD_HALF_WIDTH_M + margin for c in road_centre_offsets())


def distance_to_road(x, y):
    cs = road_centre_offsets()
    dx = min(abs(x - c) for c in cs)
    dy = min(abs(y - c) for c in cs)
    return min(dx, dy)


def validate_layout():
    """CPU-only sanity checks run before any GPU time is spent -- same discipline as the forest
    (where measuring first caught a pad sitting in the river and a target in a bald clearing)."""
    issues = []
    for name, (px, py) in (("START_PAD", START_PAD), ("END_PAD", END_PAD), ("TOWER", TOWER_POS)):
        if on_road(px, py, margin=SIDEWALK_M):
            issues.append(f"{name} at ({px},{py}) sits on a carriageway")
        if name != "TOWER" and math.hypot(px - TOWER_POS[0], py - TOWER_POS[1]) < TOWER_PLAZA_R + PAD_CLEAR_R:
            issues.append(f"{name} overlaps the tower plaza")
        block_interior_half = BLOCK_PITCH_M / 2.0 - ROAD_HALF_WIDTH_M - SIDEWALK_M
        need = TOWER_PLAZA_R if name == "TOWER" else PAD_CLEAR_R
        if need > block_interior_half:
            issues.append(f"{name} needs {need}m but the block interior is only "
                          f"{block_interior_half:.1f}m -- it would spill onto the street")
    if math.hypot(START_PAD[0] - END_PAD[0], START_PAD[1] - END_PAD[1]) < 80.0:
        issues.append("START and END pads are too close to be a real A->B transit")
    return issues


# ---- scene construction -----------------------------------------------------------------


def create_pbr_material_fixed(stage, nv, path, assets_root, texture_id, uv_scale=1.0, tint=None):
    """PBR material that actually renders textured in Isaac Sim.

    THE BUG THIS FIXES: natural_valley.create_pbr_material() routes texture coordinates through a
    UsdTransform2d node (primvar reader -> UsdTransform2d -> each UsdUVTexture.st). Isaac Sim's
    RTX renderer does not evaluate that node, so the st chain is dead and every texture lookup
    silently falls back to a flat colour. That is why every ground, road, sidewalk and building
    surface rendered as flat paint in the city -- and, checked afterwards, why the forest's
    terrain is flat brown too. The forest got away with it because 4570 photogrammetry plants
    cover its ground; a city has nowhere to hide.

    The proof is in the same upstream file: create_asset_material(), used for the tree and
    street-furniture models, is identical EXCEPT that it connects each texture's `st` DIRECTLY to
    the primvar reader with no UsdTransform2d -- and those models are the one thing that has
    always rendered properly textured. Same .jpg diffuse, same .exr normal/roughness, so the file
    formats are not the issue; the node is.

    So: connect straight to the primvar reader. Nothing is lost, because textured_box() already
    bakes tiling into the `st` primvar in real metres (uv = extent / tile_m) -- which is exactly
    what the transform's scale was being asked to do. uv_scale is accepted and IGNORED for that
    same reason: applying it on top of metre-baked UVs tiled the sidewalk every 5.7cm.

    Also decodes the normal map [0,1] -> [-1,1] (scale/bias), which upstream omits.
    """
    Sdf, Gf, UsdShade = nv.Sdf, nv.Gf, nv.UsdShade
    material = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    st_reader = UsdShade.Shader.Define(stage, f"{path}/TexCoord")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
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
            st_reader.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        if channel == "diffuse" and tint is not None:
            # UsdUVTexture.scale multiplies the sampled value, so a single texture image can
            # serve many differently-coloured buildings. Real streets are not one colour, and
            # with only two facade textures carrying any detail this is what keeps the blocks
            # from reading as repeats of each other.
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(
                Gf.Vec4f(float(tint[0]), float(tint[1]), float(tint[2]), 1.0))
        if channel == "normal":
            # the actual fix: decode [0,1] -> [-1,1]
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


def _pbr(stage, nv, assets_root, name, texture_id, uv_scale=1.0, tint=None):
    """uv_scale is deliberately IGNORED (forced to 1.0).

    This was the real reason the city looked flat and "much simpler" than the forest, and why the
    roads read lavender. textured_box() bakes UVs in METRES already (uv = extent / tile_m), so the
    material must not scale them again. The forest could pass uv_scale because its geometry had
    plain 0..1 UVs; here the two compounded:

        sidewalk: 500m / 2.5m tile = 200 uv units, x44 uv_scale = 8800 repeats -> 5.7cm per tile
        facade:   18m / 3.2m tile  = 5.6 uv units, x5  uv_scale = 28 repeats   -> 64cm per tile

    A cobblestone texture repeated every 5.7cm is far below a pixel at render distance, so every
    PBR surface integrated down to its own average colour -- i.e. flat paint. Tiling is now
    controlled solely by each call site's tile_m, in real metres."""
    return create_pbr_material_fixed(stage, nv, f"/World/Materials/City_{name}", assets_root,
                                     texture_id, uv_scale=1.0, tint=tint)


def build_ground_and_roads(stage, nv, assets_root):
    """Flat city ground: asphalt carriageways in a grid, cobblestone sidewalks flanking them,
    concrete elsewhere. Built as large quads rather than one textured plane so each surface gets
    its own correctly-scaled material."""
    Gf = nv.Gf
    asphalt = _pbr(stage, nv, assets_root, "Asphalt", "asphalt_02")
    walk = _pbr(stage, nv, assets_root, "Sidewalk", "cobblestone_floor_01", uv_scale=44.0)
    base = _pbr(stage, nv, assets_root, "GroundConcrete", "concrete_wall_008", uv_scale=60.0)
    curb = _pbr(stage, nv, assets_root, "Curb", "concrete_panels", uv_scale=6.0)
    paint = nv.create_color_material(stage, "/World/Materials/RoadPaint", (0.88, 0.87, 0.82), 0.6)

    E = CITY_HALF_EXTENT_M
    # base ground slab, slightly below the roads so roads read as a surface on top of it
    textured_box(nv, stage, "/World/City/Ground", (0.0, 0.0, -0.30), (2 * E, 2 * E, 0.5), base, tile_m=9.0)

    counts = {"roads": 0, "sidewalks": 0}
    for c in road_centre_offsets():
        # carriageways (full span in the other axis)
        textured_box(nv, stage, f"/World/City/Roads/NS_{int(c):+05d}".replace("+", "p").replace("-", "n"),
                     (c, 0.0, -0.02), (2 * ROAD_HALF_WIDTH_M, 2 * E, 0.06), asphalt, tile_m=8.0)
        textured_box(nv, stage, f"/World/City/Roads/EW_{int(c):+05d}".replace("+", "p").replace("-", "n"),
                     (0.0, c, -0.02), (2 * E, 2 * ROAD_HALF_WIDTH_M, 0.06), asphalt, tile_m=8.0)
        counts["roads"] += 2
        # sidewalks either side of each carriageway
        for side in (-1, 1):
            off = side * (ROAD_HALF_WIDTH_M + SIDEWALK_M / 2.0)
            tag = "a" if side < 0 else "b"
            textured_box(nv, stage, f"/World/City/Walks/NS_{int(c):+05d}_{tag}".replace("+", "p").replace("-", "n"),
                         (c + off, 0.0, SIDEWALK_H / 2.0), (SIDEWALK_M, 2 * E, SIDEWALK_H), walk, tile_m=2.5)
            textured_box(nv, stage, f"/World/City/Walks/EW_{int(c):+05d}_{tag}".replace("+", "p").replace("-", "n"),
                         (0.0, c + off, SIDEWALK_H / 2.0), (2 * E, SIDEWALK_M, SIDEWALK_H), walk, tile_m=2.5)
            counts["sidewalks"] += 2
            # curb face: a thin vertical strip closing the gap between road and sidewalk, so the
            # sidewalk reads as a kerbed pavement rather than a slab hovering over the asphalt
            cface = c + side * (ROAD_HALF_WIDTH_M + 0.06)
            textured_box(nv, stage, f"/World/City/Curbs/NS_{int(c):+05d}_{tag}".replace("+", "p").replace("-", "n"),
                         (cface, 0.0, CURB_H / 2.0), (0.12, 2 * E, CURB_H), curb, tile_m=1.2)
            textured_box(nv, stage, f"/World/City/Curbs/EW_{int(c):+05d}_{tag}".replace("+", "p").replace("-", "n"),
                         (0.0, cface, CURB_H / 2.0), (2 * E, 0.12, CURB_H), curb, tile_m=1.2)

    # painted lane markings: dashed centre line down every carriageway, plus zebra crossings at
    # the intersections near the flight corridor. Bare unmarked asphalt is one of the strongest
    # "this is not a real street" cues in an aerial shot.
    dash_len, dash_gap = 3.0, 4.0
    marks = 0
    for c in road_centre_offsets():
        k = 0
        s0 = -E + dash_gap
        while s0 + dash_len < E:
            # skip the stretch inside an intersection (tested on the CROSS axis only)
            if not near_centreline(s0 + dash_len / 2.0, margin=SIDEWALK_M):
                nv.cube(stage, f"/World/City/Marks/NS_{int(c):+05d}_{k:03d}".replace("+", "p").replace("-", "n"),
                        (c, s0 + dash_len / 2.0, 0.035), (LANE_MARK_W, dash_len, 0.02), paint)
                nv.cube(stage, f"/World/City/Marks/EW_{int(c):+05d}_{k:03d}".replace("+", "p").replace("-", "n"),
                        (s0 + dash_len / 2.0, c, 0.035), (dash_len, LANE_MARK_W, 0.02), paint)
                marks += 2
            s0 += dash_len + dash_gap
            k += 1

    # zebra crossings: only the intersections within ~150m of the flight corridor, since at 52m
    # altitude the outer ones are never legible and each costs 28 prims.
    inner = [c for c in road_centre_offsets() if abs(c) <= 2 * BLOCK_PITCH_M + 1.0]
    for cx in inner:
        for cy in inner:
            for sgn in (-1, 1):
                for i in range(7):
                    off = -ROAD_HALF_WIDTH_M + 1.3 + i * 1.8
                    sfx = "p" if sgn > 0 else "n"
                    nm = f"{int(cx):+05d}_{int(cy):+05d}".replace("+", "p").replace("-", "n")
                    nv.cube(stage, f"/World/City/Zebra/X{nm}_{sfx}_{i}",
                            (cx + off, cy + sgn * (ROAD_HALF_WIDTH_M - 1.5), 0.035),
                            (0.9, 2.4, 0.02), paint)
                    nv.cube(stage, f"/World/City/Zebra/Y{nm}_{sfx}_{i}",
                            (cx + sgn * (ROAD_HALF_WIDTH_M - 1.5), cy + off, 0.035),
                            (2.4, 0.9, 0.02), paint)
                    marks += 2
    counts["road_markings"] = marks
    return counts


def build_tower(stage, nv, assets_root):
    """The mission landmark: a real-proportioned high-rise -- plaza base, ground-floor podium with
    a different material, a shaft with floor bands and recessed window rows, a parapet, and a
    rooftop mast."""
    facade = _pbr(stage, nv, assets_root, "TowerFacade", TOWER_FACADE, uv_scale=7.0)
    podium = _pbr(stage, nv, assets_root, "TowerPodium", "concrete_tile_facade", uv_scale=4.0)
    plaza = _pbr(stage, nv, assets_root, "TowerPlaza", "cobblestone_floor_01", uv_scale=26.0)
    band = nv.create_color_material(stage, "/World/Materials/TowerBand", (0.72, 0.72, 0.74), 0.35)
    glass = nv.create_color_material(stage, "/World/Materials/TowerGlass", (0.05, 0.08, 0.12), 0.08)
    mast = nv.create_color_material(stage, "/World/Materials/TowerMast", (0.25, 0.26, 0.28), 0.4)

    tx, ty = TOWER_POS
    root = "/World/City/Tower"
    # plaza slab the tower stands on
    textured_box(nv, stage, f"{root}/Plaza", (tx, ty, 0.09),
                 (2 * TOWER_PLAZA_R, 2 * TOWER_PLAZA_R, 0.18), plaza, tile_m=2.5)

    podium_h = 7.0
    textured_box(nv, stage, f"{root}/Podium", (tx, ty, podium_h / 2.0 + 0.18),
                 (TOWER_W + 3.0, TOWER_D + 3.0, podium_h), podium, tile_m=3.0)

    shaft_base = podium_h + 0.18
    shaft_h = TOWER_HEIGHT_M - shaft_base
    textured_box(nv, stage, f"{root}/Shaft", (tx, ty, shaft_base + shaft_h / 2.0),
                 (TOWER_W, TOWER_D, shaft_h), facade, tile_m=3.6)

    # floor bands + recessed glazing rows -- this is what makes a box read as a building
    floor_h = 3.6
    n_floors = int(shaft_h / floor_h)
    for i in range(n_floors):
        z = shaft_base + (i + 0.5) * floor_h
        nv.cube(stage, f"{root}/Band_{i:02d}", (tx, ty, z + floor_h * 0.42),
                (TOWER_W + 0.35, TOWER_D + 0.35, 0.30), band)
        for axis, (dx, dy, w, d) in (
            ("N", (0.0, TOWER_D / 2.0, TOWER_W - 2.2, 0.25)),
            ("S", (0.0, -TOWER_D / 2.0, TOWER_W - 2.2, 0.25)),
            ("E", (TOWER_W / 2.0, 0.0, 0.25, TOWER_D - 2.2)),
            ("W", (-TOWER_W / 2.0, 0.0, 0.25, TOWER_D - 2.2)),
        ):
            nv.cube(stage, f"{root}/Glass_{i:02d}{axis}", (tx + dx, ty + dy, z),
                    (w, d, floor_h * 0.62), glass)

    top = shaft_base + shaft_h
    _parapet(nv, stage, root, "Tower", tx, ty, TOWER_W, TOWER_D, top + 0.3, band, h=1.2, th=0.45)
    textured_box(nv, stage, f"{root}/RoofDeck", (tx, ty, top + 0.15), (TOWER_W, TOWER_D, 0.3), podium, tile_m=3.0)
    nv.cylinder(stage, f"{root}/Mast", (tx, ty, top + 6.5), 0.28, 11.0, mast)
    nv.cylinder(stage, f"{root}/MastTip", (tx, ty, top + 12.6), 0.12, 2.4, mast)
    return {"tower_height_m": TOWER_HEIGHT_M, "tower_floors": n_floors,
            "tower_top_z": top + 13.8}


def _parapet(nv, stage, root, tag, bx, by, w, d, ztop, trim, h=0.55, th=0.35):
    """A parapet is a RIM around the roof edge, not a lid over it.

    It was a single solid cube of (w+0.6, d+0.6, 0.5) sitting above a roof slab of (w+0.3, d+0.3),
    so it covered the whole roof: every rooftop rendered as a blank white plate, the roof's
    ceramic/clay texture was never visible, and the AC units and vents appeared to float on
    nothing. Four thin walls instead, leaving the roof deck open."""
    for axis, ox, oy, sw, sd in (("N", 0.0, (d + th) / 2.0, w + 2 * th, th),
                                 ("S", 0.0, -(d + th) / 2.0, w + 2 * th, th),
                                 ("E", (w + th) / 2.0, 0.0, th, d + 2 * th),
                                 ("W", -(w + th) / 2.0, 0.0, th, d + 2 * th)):
        nv.cube(stage, f"{root}/{tag}_parapet{axis}", (bx + ox, by + oy, ztop + h / 2.0),
                (sw, sd, h), trim)


def _building_tier(nv, stage, root, tag, bx, by, z0, w, d, h, fac, style, floor_h,
                   win_glass, trim, rng):
    """One massing volume of a building, with the relief that makes a box read as architecture:
    recessed glazing (set INTO the wall, not proud of it), a protruding sill/spandrel ledge at
    every floor line, corner pilasters, and a cornice at the top.

    The ledges matter more than the windows from the air: at 52m with the camera pitched down,
    what separates a real facade from a painted box is the horizontal shadow line each floor
    casts, and that needs actual protruding geometry."""
    textured_box(nv, stage, f"{root}/{tag}", (bx, by, z0 + h / 2.0), (w, d, h), fac, tile_m=3.2)

    n_fl = max(1, int((h - 1.0) / floor_h))
    faces = (("N", 0.0, 1.0), ("S", 0.0, -1.0), ("E", 1.0, 0.0), ("W", -1.0, 0.0))
    for f in range(n_fl):
        fz = z0 + 1.5 + f * floor_h
        if fz + 1.0 > z0 + h:
            break
        for axis, sx, sy in faces:
            span = (w - 2.2) if axis in ("N", "S") else (d - 2.2)
            if span <= 1.0:
                continue
            # glazing plane pushed 0.10m INSIDE the wall face so the reveal casts a shadow
            gx = bx + sx * (w / 2.0 - 0.10)
            gy = by + sy * (d / 2.0 - 0.10)
            if style == "ribbon":
                ww = span if axis in ("N", "S") else 0.16
                dd = 0.16 if axis in ("N", "S") else span
                nv.cube(stage, f"{root}/{tag}_g{f:02d}{axis}", (gx, gy, fz),
                        (ww, dd, floor_h * 0.52), win_glass)
            else:
                pane, gap = 1.5, 1.35
                n_pane = max(1, int(span / (pane + gap)))
                st = -(n_pane - 1) * (pane + gap) / 2.0
                for q in range(n_pane):
                    off = st + q * (pane + gap)
                    ox = off if axis in ("N", "S") else 0.0
                    oy = 0.0 if axis in ("N", "S") else off
                    ww = pane if axis in ("N", "S") else 0.16
                    dd = 0.16 if axis in ("N", "S") else pane
                    nv.cube(stage, f"{root}/{tag}_g{f:02d}{axis}{q:02d}",
                            (gx + ox, gy + oy, fz), (ww, dd, floor_h * 0.55), win_glass)
            # sill ledge: one protruding strip per floor per face (4 prims/floor, not 4 per pane)
            lx = bx + sx * (w / 2.0 + 0.11)
            ly = by + sy * (d / 2.0 + 0.11)
            lw = (w + 0.5) if axis in ("N", "S") else 0.26
            ld = 0.26 if axis in ("N", "S") else (d + 0.5)
            nv.cube(stage, f"{root}/{tag}_s{f:02d}{axis}", (lx, ly, fz - floor_h * 0.36),
                    (lw, ld, 0.18), trim)

    # corner pilasters: vertical relief so the massing is not four flat planes. These are
    # textured_box, not nv.cube: nv.cube makes a UsdGeom.Cube, which carries no `st` primvar, so
    # a PBR material on one renders untextured no matter what the material says.
    for ex in (-1, 1):
        for ey in (-1, 1):
            textured_box(nv, stage, f"{root}/{tag}_p{'p' if ex>0 else 'n'}{'p' if ey>0 else 'n'}",
                         (bx + ex * (w / 2.0 - 0.25), by + ey * (d / 2.0 - 0.25), z0 + h / 2.0),
                         (1.0, 1.0, h), fac, tile_m=3.2)
    # cornice capping the tier
    nv.cube(stage, f"{root}/{tag}_cornice", (bx, by, z0 + h - 0.25), (w + 0.7, d + 0.7, 0.5), trim)
    return n_fl


def _rooftop_clutter(nv, stage, root, bx, by, w, d, ztop, trim, metal, rng):
    """AC plant, vents, a stair bulkhead and an antenna. From a drone at 52m pitched down, roofs
    are a large share of every frame -- bare roof slabs were the most obviously fake surface in
    the first render."""
    n = 0
    def spot(mw, md):
        return (bx + float(rng.uniform(-(w / 2 - mw), w / 2 - mw)),
                by + float(rng.uniform(-(d / 2 - md), d / 2 - md)))
    # stair/lift bulkhead
    bw, bd, bh = float(rng.uniform(3.0, 4.6)), float(rng.uniform(2.6, 4.0)), float(rng.uniform(2.4, 3.4))
    if w > bw * 2.4 and d > bd * 2.4:
        sx, sy = spot(bw / 2 + 0.8, bd / 2 + 0.8)
        nv.cube(stage, f"{root}/rt_bulk", (sx, sy, ztop + bh / 2.0), (bw, bd, bh), trim); n += 1
    # AC / chiller units
    for u in range(int(rng.integers(2, 6))):
        uw, ud, uh = float(rng.uniform(1.2, 2.4)), float(rng.uniform(1.0, 1.9)), float(rng.uniform(0.7, 1.3))
        if w <= uw * 3 or d <= ud * 3:
            break
        sx, sy = spot(uw / 2 + 1.0, ud / 2 + 1.0)
        nv.cube(stage, f"{root}/rt_ac{u}", (sx, sy, ztop + uh / 2.0), (uw, ud, uh), metal); n += 1
    # round vents
    for v in range(int(rng.integers(1, 5))):
        r = float(rng.uniform(0.22, 0.45))
        hgt = float(rng.uniform(0.6, 1.4))
        sx, sy = spot(1.2, 1.2)
        nv.cylinder(stage, f"{root}/rt_v{v}", (sx, sy, ztop + hgt / 2.0), r, hgt, metal); n += 1
    # antenna mast on the taller buildings
    if rng.random() < 0.4:
        sx, sy = spot(1.0, 1.0)
        mh = float(rng.uniform(3.0, 7.0))
        nv.cylinder(stage, f"{root}/rt_mast", (sx, sy, ztop + mh / 2.0), 0.09, mh, metal); n += 1
    return n


def build_blocks(stage, nv, assets_root, seed=7, max_buildings=140):
    """Fill the grid blocks with buildings: varied footprints/heights, setbacks on the tall ones,
    real facade + roof PBR, per-floor relief, ground-floor shopfronts and rooftop plant."""
    rng = np.random.default_rng(seed)
    # 3 detail-carrying textures x 8 tints = 24 distinct facade looks across 92 buildings,
    # instead of 5 materials of which 3 were blank.
    FACADE_TINTS = [(1.00, 0.98, 0.95), (0.86, 0.80, 0.72), (0.72, 0.74, 0.78),
                    (0.95, 0.86, 0.70), (0.62, 0.58, 0.56), (0.88, 0.72, 0.62),
                    (0.78, 0.82, 0.80), (1.05, 1.02, 0.96)]
    facades = [_pbr(stage, nv, assets_root, f"Facade{i}_{k}", tex, tint=tint)
               for i, tex in enumerate(FACADE_TEXTURES)
               for k, tint in enumerate(FACADE_TINTS)]
    roofs = [_pbr(stage, nv, assets_root, f"Roof{i}_{k}", tex, tint=tint)
             for i, tex in enumerate(ROOF_TEXTURES)
             for k, tint in enumerate(((1.0, 1.0, 1.0), (0.80, 0.78, 0.76), (0.66, 0.60, 0.58)))]
    trim = nv.create_color_material(stage, "/World/Materials/CityTrim", (0.78, 0.77, 0.74), 0.45)
    metal = nv.create_color_material(stage, "/World/Materials/RoofPlant", (0.55, 0.56, 0.58), 0.55, metallic=0.6)
    win_glass = nv.create_color_material(stage, "/World/Materials/CityGlass", (0.06, 0.09, 0.13), 0.09)
    shop_glass = nv.create_color_material(stage, "/World/Materials/ShopGlass", (0.10, 0.13, 0.16), 0.12)
    awning_cols = [nv.create_color_material(stage, f"/World/Materials/Awning{i}", c, 0.65)
                   for i, c in enumerate(((0.62, 0.18, 0.16), (0.16, 0.34, 0.52),
                                          (0.20, 0.42, 0.26), (0.70, 0.60, 0.26)))]

    cs = road_centre_offsets()
    placed = 0
    stats = {"buildings": 0, "setback_tiers": 0, "rooftop_props": 0, "shopfronts": 0}
    for i in range(len(cs) - 1):
        for j in range(len(cs) - 1):
            if placed >= max_buildings:
                break
            x0 = cs[i] + ROAD_HALF_WIDTH_M + SIDEWALK_M + BLOCK_MARGIN_M
            x1 = cs[i + 1] - ROAD_HALF_WIDTH_M - SIDEWALK_M - BLOCK_MARGIN_M
            y0 = cs[j] + ROAD_HALF_WIDTH_M + SIDEWALK_M + BLOCK_MARGIN_M
            y1 = cs[j + 1] - ROAD_HALF_WIDTH_M - SIDEWALK_M - BLOCK_MARGIN_M
            if x1 - x0 < 8.0 or y1 - y0 < 8.0:
                continue
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if math.hypot(cx - TOWER_POS[0], cy - TOWER_POS[1]) < TOWER_PLAZA_R + 14.0:
                continue
            if any(math.hypot(cx - px, cy - py) < PAD_CLEAR_R + 16.0 for px, py in (START_PAD, END_PAD)):
                continue
            for k in range(int(rng.integers(1, 3))):
                w = float(rng.uniform(11.0, min(22.0, x1 - x0)))
                d = float(rng.uniform(11.0, min(22.0, y1 - y0)))
                h = float(rng.uniform(9.0, 34.0))
                bx = float(rng.uniform(x0 + w / 2.0, x1 - w / 2.0))
                by = float(rng.uniform(y0 + d / 2.0, y1 - d / 2.0))
                fac = facades[int(rng.integers(0, len(facades)))]
                floor_h = float(rng.uniform(3.1, 3.8))
                style = "ribbon" if rng.random() < 0.45 else "punched"
                root = f"/World/City/Blocks/B_{placed:03d}"

                # setback massing on the taller buildings: a narrower upper tier stepped in from
                # the base. A skyline of single-extrusion boxes is the giveaway of a generated
                # city; real blocks step back as they rise.
                setback = h > 22.0 and rng.random() < 0.6
                h_base = h * float(rng.uniform(0.55, 0.72)) if setback else h
                _building_tier(nv, stage, root, "base", bx, by, 0.0, w, d, h_base, fac, style,
                               floor_h, win_glass, trim, rng)
                ztop = h_base
                if setback:
                    w2 = w * float(rng.uniform(0.62, 0.82))
                    d2 = d * float(rng.uniform(0.62, 0.82))
                    # step in toward one corner rather than staying concentric
                    ox = float(rng.uniform(-(w - w2) / 2.0, (w - w2) / 2.0)) * 0.7
                    oy = float(rng.uniform(-(d - d2) / 2.0, (d - d2) / 2.0)) * 0.7
                    # the base's roof is now a terrace
                    textured_box(nv, stage, f"{root}/terrace", (bx, by, h_base + 0.16),
                                 (w + 0.3, d + 0.3, 0.32), roofs[int(rng.integers(0, len(roofs)))],
                                 tile_m=2.2)
                    _building_tier(nv, stage, root, "top", bx + ox, by + oy, h_base, w2, d2,
                                   h - h_base, fac, style, floor_h, win_glass, trim, rng)
                    stats["rooftop_props"] += _rooftop_clutter(nv, stage, root, bx + ox, by + oy,
                                                               w2, d2, h, trim, metal, rng)
                    textured_box(nv, stage, f"{root}/roof", (bx + ox, by + oy, h + 0.16),
                                 (w2 + 0.3, d2 + 0.3, 0.32), roofs[int(rng.integers(0, len(roofs)))],
                                 tile_m=2.2)
                    _parapet(nv, stage, root, "top", bx + ox, by + oy, w2, d2, h + 0.32, trim)
                    stats["setback_tiers"] += 1
                else:
                    stats["rooftop_props"] += _rooftop_clutter(nv, stage, root, bx, by, w, d, h,
                                                               trim, metal, rng)
                    textured_box(nv, stage, f"{root}/roof", (bx, by, h + 0.16),
                                 (w + 0.3, d + 0.3, 0.32), roofs[int(rng.integers(0, len(roofs)))],
                                 tile_m=2.2)
                    _parapet(nv, stage, root, "main", bx, by, w, d, h + 0.32, trim)

                # ground-floor shopfront on the street-facing sides: a taller glazed base with a
                # coloured awning. Real streets are retail at grade and offices above; a facade
                # whose window grid runs uninhandled to the pavement looks like a model.
                if rng.random() < 0.75:
                    for axis, sx, sy in (("N", 0.0, 1.0), ("S", 0.0, -1.0),
                                         ("E", 1.0, 0.0), ("W", -1.0, 0.0)):
                        span = (w - 2.0) if axis in ("N", "S") else (d - 2.0)
                        if span <= 2.0:
                            continue
                        gx = bx + sx * (w / 2.0 - 0.12)
                        gy = by + sy * (d / 2.0 - 0.12)
                        ww = span if axis in ("N", "S") else 0.2
                        dd = 0.2 if axis in ("N", "S") else span
                        nv.cube(stage, f"{root}/shop{axis}", (gx, gy, 1.9), (ww, dd, 3.0), shop_glass)
                        aw = awning_cols[int(rng.integers(0, len(awning_cols)))]
                        ax = bx + sx * (w / 2.0 + 0.55)
                        ay = by + sy * (d / 2.0 + 0.55)
                        aww = span * 0.8 if axis in ("N", "S") else 1.3
                        add = 1.3 if axis in ("N", "S") else span * 0.8
                        nv.cube(stage, f"{root}/awn{axis}", (ax, ay, 3.6), (aww, add, 0.22), aw)
                        stats["shopfronts"] += 1
                placed += 1
                if placed >= max_buildings:
                    break
    stats["buildings"] = placed
    return stats


def road_axis_yaw(x, y):
    """Heading (radians) of the street nearest to (x,y).

    A point closer to a north-south centreline in x sits beside a road that RUNS along y, so
    anything aligned to that kerb points along +/-y."""
    cs = road_centre_offsets()
    dx = min(abs(x - c) for c in cs)
    dy = min(abs(y - c) for c in cs)
    return math.pi / 2.0 if dx <= dy else 0.0


def _prop_instances(rng, count, predicate, scale_range, z_fn=lambda: 0.0, align=False,
                    jitter_deg=0.0):
    """align=True orients the instance along the nearest kerb instead of at a random angle.

    Parked cars and road barriers were being dropped at uniformly random yaw, which puts them
    across the kerb at arbitrary angles -- obviously wrong on any street, and exactly the kind of
    detail that separates this from the forest, where every asset was a naturally-oriented tree."""
    out = []
    attempts = 0
    while len(out) < count and attempts < count * 40:
        attempts += 1
        x = float(rng.uniform(-CITY_HALF_EXTENT_M + 10, CITY_HALF_EXTENT_M - 10))
        y = float(rng.uniform(-CITY_HALF_EXTENT_M + 10, CITY_HALF_EXTENT_M - 10))
        if not predicate(x, y):
            continue
        s = float(rng.uniform(*scale_range))
        if align:
            yaw = road_axis_yaw(x, y) + (math.pi if rng.random() < 0.5 else 0.0)
            yaw += math.radians(float(rng.uniform(-jitter_deg, jitter_deg)))
        else:
            yaw = float(rng.uniform(-math.pi, math.pi))
        out.append((0, (x, y, z_fn()), yaw, (s, s, s)))
    return out


def build_props(stage, nv, assets_root, seed=11):
    """Real street furniture, instanced. Lamps/benches/bins go on sidewalks, cars along curbs,
    barriers/poles on the verge -- all kept off the landing pads."""
    rng = np.random.default_rng(seed)
    mats = {key: nv.create_asset_material(stage, f"/World/Materials/Prop_{key}", assets_root, asset)
            for key, asset in PROP_MODELS.items()}

    def clear_of_pads(x, y):
        return all(math.hypot(x - px, y - py) > PAD_CLEAR_R for px, py in (START_PAD, END_PAD))

    def on_sidewalk(x, y):
        if not clear_of_pads(x, y):
            return False
        d = distance_to_road(x, y)
        return ROAD_HALF_WIDTH_M + 0.4 < d < ROAD_HALF_WIDTH_M + SIDEWALK_M

    def on_curb(x, y):
        if not clear_of_pads(x, y):
            return False
        d = distance_to_road(x, y)
        return ROAD_HALF_WIDTH_M - 3.0 < d < ROAD_HALF_WIDTH_M - 0.5

    counts = {}
    # Counts raised roughly 4x. There are ~18km of sidewalk and ~18km of kerb in this grid, so the
    # previous 110 cars worked out to one parked car every 164m and 120 lamps to one every 150m --
    # a city that reads as deserted. These are PointInstancer prototypes, so the extra instances
    # cost almost nothing: one prototype mesh each, however many copies.
    # align=True on cars/barriers so they sit parallel to the kerb rather than at random angles.
    spec = [
        ("street_lamp_01", on_sidewalk, 460, (0.9, 1.1), SIDEWALK_H, True, 4.0),
        ("street_lamp_02", on_sidewalk, 320, (0.9, 1.1), SIDEWALK_H, True, 4.0),
        ("fire_hydrant", on_sidewalk, 170, (0.9, 1.15), SIDEWALK_H, False, 0.0),
        ("bench", on_sidewalk, 260, (0.95, 1.1), SIDEWALK_H, True, 6.0),
        ("seating", on_sidewalk, 150, (0.9, 1.1), SIDEWALK_H, True, 6.0),
        ("trash_can", on_sidewalk, 240, (0.9, 1.1), SIDEWALK_H, False, 0.0),
        ("planter", on_sidewalk, 230, (0.9, 1.2), SIDEWALK_H, True, 5.0),
        ("poles", on_sidewalk, 130, (0.9, 1.1), SIDEWALK_H, True, 3.0),
        ("car", on_curb, 900, (0.95, 1.05), 0.0, True, 2.5),
        ("barrier", on_curb, 160, (0.9, 1.1), 0.0, True, 3.0),
    ]
    for key, pred, n, srange, z, align, jit in spec:
        inst = _prop_instances(rng, n, pred, srange, z_fn=lambda z=z: z, align=align,
                               jitter_deg=jit)
        nv.point_instancer(stage, f"/World/City/Props/{key}",
                           [(model_path(assets_root, PROP_MODELS[key]), 1.0, mats[key])],
                           inst)
        counts[f"prop_{key}"] = len(inst)
    counts["prop_total"] = sum(counts.values())
    return counts


def build_trees(stage, nv, forest_assets_root, seed=13, n=170):
    """Street trees, reusing the jacaranda already downloaded for the forest project."""
    rng = np.random.default_rng(seed)
    mat = nv.create_asset_material(stage, "/World/Materials/CityTree", forest_assets_root,
                                   nv.ASSET_IDS["mature_tree"])

    def ok(x, y):
        if any(math.hypot(x - px, y - py) < PAD_CLEAR_R for px, py in (START_PAD, END_PAD)):
            return False
        if math.hypot(x - TOWER_POS[0], y - TOWER_POS[1]) < TOWER_PLAZA_R:
            return False
        d = distance_to_road(x, y)
        return ROAD_HALF_WIDTH_M + 0.6 < d < ROAD_HALF_WIDTH_M + SIDEWALK_M + 1.0

    inst = _prop_instances(rng, n, ok, (0.22, 0.34), z_fn=lambda: SIDEWALK_H)
    nv.point_instancer(stage, "/World/City/Trees",
                       [(nv.asset_path(forest_assets_root, "mature_tree"), 1.0, mat)], inst)
    return {"street_trees": len(inst)}


def build_lighting(stage, nv, assets_root, hdri_id="kloofendal_48d_partly_cloudy_puresky", seed=5):
    """Sky dome + sun.

    The HDRI must be a SKY-ONLY ("puresky") capture. cedar_bridge_2, used until now, is a
    ground-level photograph taken on a bridge -- it contains the bridge deck, its railings,
    roadside guardrails and buildings. A DomeLight projects its texture at infinity, so all of
    that was being painted around the horizon: from the drone it read as a warehouse and a
    guardrail looming over the city, and it is what made the environment look like it ended at a
    hard edge. A puresky HDRI has clouds above and a smooth neutral gradient below, which blends
    into the far ground instead of fighting it.

    Verified by tone-mapping both HDRIs and looking at them; an automatic "horizon detail" metric
    scored cedar_bridge_2 as clean, so it was the wrong check."""
    from pathlib import Path

    UsdLux, Gf = nv.UsdLux, nv.Gf
    sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
    sky.CreateIntensityAttr(720.0)
    hdr = Path(assets_root) / "hdri" / hdri_id / f"{hdri_id}_2k.hdr"
    if hdr.exists():
        sky.CreateTextureFileAttr().Set(str(hdr))
        sky.CreateTextureFormatAttr().Set(UsdLux.Tokens.latlong)
    sky.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 40.0))
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(1150.0)
    sun.CreateAngleAttr(0.6)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.95, 0.86))
    sun.AddRotateXYZOp().Set(Gf.Vec3f(48.0, -18.0, 35.0))
    return {"hdri": hdri_id if hdr.exists() else None}


def build_surroundings(stage, nv, assets_root, seed=21):
    """Everything beyond the detailed grid, so the city has no visible edge.

    Without this the 500m slab simply stops and the drone looks out over a hard rectangular
    boundary with sky underneath -- the city's version of the "we are standing in a circle"
    problem the forest had. Three layers:
      1. a 3km ground plane under everything, so the horizon is ground, not a void;
      2. continuation roads running out past the detailed area, so the street grid fades out
         instead of being chopped off;
      3. a belt of low-detail filler blocks from the edge of the detailed city out to ~900m,
         thinning and lowering with distance so the skyline decays naturally into haze.
    Filler blocks are plain textured boxes with no fenestration: at 300m+ and 52m altitude the
    silhouette is all that resolves, and detail there would cost prims for nothing."""
    rng = np.random.default_rng(seed)
    asphalt = _pbr(stage, nv, assets_root, "FarAsphalt", "asphalt_02")
    # four haze levels: blocks near the city edge keep their colour, blocks out at the belt are
    # washed toward the sky so the skyline dissolves into the horizon instead of ending.
    FAR_TINTS = [tuple(1.0 + (k / 5.0) * (r - 1.0) for r in (1.023, 1.106, 1.403))
                 for k in range(6)]
    facades = [_pbr(stage, nv, assets_root, f"FarFacade{i}_{k}", t_, tint=tint)
               for i, t_ in enumerate(FACADE_TEXTURES)
               for k, tint in enumerate(FAR_TINTS)]

    E = CITY_HALF_EXTENT_M
    FAR = 6000.0
    # Distance haze, done geometrically. The sky dome's horizon is a cool grey-blue (sRGB
    # ~0.565/0.579/0.615, measured off the HDRI) while bare ground concrete is warm cream, so a
    # single ground plane meets the sky in a hard, obviously-fake line. Instead: concentric ground
    # slabs, each larger, lower, and tinted one step further toward the sky colour, so the ground
    # fades into the horizon the way atmospheric scattering makes it. Rendering fog was the other
    # option, but RTX fog settings are not reliably honoured in RealTimePathTracing, and geometry
    # always is.
    # Tints are computed, not guessed. concrete_wall_008's diffuse averages sRGB
    # (0.552, 0.524, 0.438) -- warm -- and the dome's horizon is (0.565, 0.579, 0.615) -- cool.
    # The ratio needed to carry one to the other is (1.023, 1.106, 1.403), i.e. mostly a BLUE
    # BOOST. The first attempt used tints below 1.0 on every channel, which darkened the ground
    # toward green and made the outskirts look like desert rather than haze.
    HAZE_RATIO = (1.023, 1.106, 1.403)
    HAZE_EXTENTS = [1000.0, 1700.0, 2600.0, 3600.0, 4700.0, FAR]
    for i, extent in enumerate(HAZE_EXTENTS):
        f = i / (len(HAZE_EXTENTS) - 1.0)
        tint = tuple(1.0 + f * (r - 1.0) for r in HAZE_RATIO)
        mat = _pbr(stage, nv, assets_root, f"FarGroundHaze{i}", "concrete_wall_008", tint=tint)
        # bigger rings sit lower, so each smaller ring draws on top of the one beyond it
        textured_box(nv, stage, f"/World/Far/Ground_{i}", (0.0, 0.0, -0.55 - 0.03 * i),
                     (2 * extent, 2 * extent, 0.5), mat, tile_m=40.0)
    # continuation roads: extend the grid's own centrelines out to the filler belt
    OUT = 2600.0
    for c in road_centre_offsets():
        textured_box(nv, stage, f"/World/Far/RoadNS_{int(c):+05d}".replace("+", "p").replace("-", "n"),
                     (c, 0.0, -0.03), (2 * ROAD_HALF_WIDTH_M, 2 * OUT, 0.06), asphalt, tile_m=10.0)
        textured_box(nv, stage, f"/World/Far/RoadEW_{int(c):+05d}".replace("+", "p").replace("-", "n"),
                     (0.0, c, -0.03), (2 * OUT, 2 * ROAD_HALF_WIDTH_M, 0.06), asphalt, tile_m=10.0)

    n = 0
    tries = 0
    while n < 2600 and tries < 90000:
        tries += 1
        x = float(rng.uniform(-OUT, OUT))
        y = float(rng.uniform(-OUT, OUT))
        r = max(abs(x), abs(y))
        if r < E + 12.0:          # inside the detailed city -- leave it alone
            continue
        # thin out with distance: dense just past the edge, sparse at the far belt
        if rng.random() > max(0.06, 1.0 - (r - E) / (OUT - E)):
            continue
        if on_road(x, y, margin=SIDEWALK_M + 3.0):
            continue
        # and get lower with distance, so the skyline decays rather than ending in a wall
        fade = 1.0 - 0.62 * min(1.0, (r - E) / (OUT - E))
        w = float(rng.uniform(14.0, 34.0))
        d = float(rng.uniform(14.0, 34.0))
        h = float(rng.uniform(8.0, 30.0)) * fade
        # pick the haze level from distance, and the texture at random within it
        lvl = min(len(FAR_TINTS) - 1, int((r - E) / (OUT - E) * len(FAR_TINTS)))
        # keep the sprawl low-rise far out; only near the city do filler blocks have height
        tex_i = int(rng.integers(0, len(FACADE_TEXTURES)))
        textured_box(nv, stage, f"/World/Far/F_{n:04d}", (x, y, h / 2.0), (w, d, h),
                     facades[tex_i * len(FAR_TINTS) + lvl], tile_m=4.5)
        n += 1
    return {"far_filler_blocks": n, "far_ground_extent_m": FAR}


def build_city(stage, nv, assets_root, forest_assets_root, seed=7):
    info = {}
    build_lighting(stage, nv, assets_root)
    info.update(build_surroundings(stage, nv, assets_root, seed=seed + 14))
    info.update(build_ground_and_roads(stage, nv, assets_root))
    info.update(build_blocks(stage, nv, assets_root, seed=seed))
    info.update(build_tower(stage, nv, assets_root))
    info.update(build_props(stage, nv, assets_root, seed=seed + 4))
    info.update(build_trees(stage, nv, forest_assets_root, seed=seed + 6))
    info["city_half_extent_m"] = CITY_HALF_EXTENT_M
    info["block_pitch_m"] = BLOCK_PITCH_M
    return info
