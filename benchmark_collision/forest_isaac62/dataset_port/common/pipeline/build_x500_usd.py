"""
Convert PX4's official X500 model (Tools/simulation/gz/models/x500/model.sdf + its meshes) into a
USD vehicle that Pegasus Simulator can fly, with the same prim layout as Pegasus' own iris.usd:

    /vehicle            ArticulationRoot
      /body             RigidBody + Mass   (base_link: 2.0 kg, Ixx=Iyy=0.02167, Izz=0.04)
      /rotor0..3        RigidBody + Mass   (0.016 kg props, at (+-0.174, +-0.174, 0.06))
        /jointN         RevoluteJoint Z, body0=/vehicle/body, body1=/vehicle/rotorN

Pegasus applies thrust at "/rotorN" and the yaw/drag at "/body", and spins props through the
articulation dofs named "jointN" -- so those names are load-bearing, not cosmetic.

Rotor numbering is PX4's quad-x order, which is also the SDF's and iris.usd's:
    rotor0 front-right (ccw) | rotor1 back-left (ccw) | rotor2 front-left (cw) | rotor3 back-right (cw)

Every number below is read from, or copied verbatim out of, model.sdf -- nothing is tuned by eye.
The NXP_FMUK66_FRONT/rd/nxp decal visuals are dropped: they are NXP HoverGames branding, which a
Holybro X500 V2 does not carry.

Runs with plain usd-core + trimesh (no Isaac Sim needed):
    python build_x500_usd.py --out ../assets/x500/x500.usd
"""
import argparse
import math
import shutil
from pathlib import Path

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "px4_env/PX4-Autopilot/Tools/simulation/gz/models/x500"
MESH_DIR = MODEL_DIR / "meshes"

# ---- values from model.sdf -------------------------------------------------------------------
BODY_MASS = 2.0
BODY_INERTIA = (0.02166666666666667, 0.02166666666666667, 0.04000000000000001)
ROTOR_MASS = 0.016076923076923075
ROTOR_INERTIA = (3.8464910483993325e-07, 2.6115851691700804e-05, 2.649858234714004e-05)
ROTOR_XY = [(0.174, -0.174), (-0.174, 0.174), (0.174, 0.174), (-0.174, -0.174)]
ROTOR_Z = 0.06
ROTOR_PROP = ["1345_prop_ccw.stl", "1345_prop_ccw.stl", "1345_prop_cw.stl", "1345_prop_cw.stl"]
PROP_POSE = (-0.022, -0.14638461538461536, -0.016)
PROP_SCALE = 0.8461538461538461
BELL_Z = -0.032
ROTOR_COLLISION_SIZE = (0.2792307692307692, 0.016923076923076923, 0.0008461538461538462)
FRAME_POSE = (0.0, 0.0, 0.025, math.pi)  # x, y, z, yaw
MOTOR_BASE_POSES = [(0.174, 0.174), (-0.174, 0.174), (0.174, -0.174), (-0.174, -0.174)]
MOTOR_BASE_Z, MOTOR_BASE_YAW = 0.032, -0.45
# (name, centre xyz, roll, size xyz)
BODY_COLLISION = [
    ("top_plate", (0.0, 0.0, 0.007), 0.0, (0.35355339059327373, 0.35355339059327373, 0.05)),
    ("leg_right", (0.0, -0.098, -0.123), -0.35, (0.015, 0.015, 0.21)),
    ("leg_left", (0.0, 0.098, -0.123), 0.35, (0.015, 0.015, 0.21)),
    ("skid_right", (0.0, -0.132, -0.2195), 0.0, (0.25, 0.015, 0.015)),
    ("skid_left", (0.0, 0.132, -0.2195), 0.0, (0.25, 0.015, 0.015)),
]
# Lowest point of the skids below base_link -> spawn height that puts the skids on the ground.
SKID_BOTTOM = -0.2195 - 0.0075

PROP_COLOR = (0.15, 0.15, 0.15)  # Gazebo/DarkGrey


def _material(stage, path, color, roughness=0.6, metallic=0.0, texture=None):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, f"{path}/Shader")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    if texture:
        st = UsdShade.Shader.Define(stage, f"{path}/st")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        tex = UsdShade.Shader.Define(stage, f"{path}/tex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _xform(prim, t=(0, 0, 0), yaw=0.0, roll=0.0, scale=None):
    x = UsdGeom.Xformable(prim)
    x.ClearXformOpOrder()
    x.AddTranslateOp().Set(Gf.Vec3d(*t))
    if yaw or roll:
        x.AddRotateXYZOp().Set(Gf.Vec3f(math.degrees(roll), 0.0, math.degrees(yaw)))
    if scale is not None:
        x.AddScaleOp().Set(Gf.Vec3f(scale, scale, scale))


def _mesh(stage, path, tm: trimesh.Trimesh, material):
    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(tm.vertices, dtype=np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tm.faces), 3, dtype=np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(tm.faces, dtype=np.int32).ravel()))
    m.CreateSubdivisionSchemeAttr("none")
    m.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(tm.vertex_normals, dtype=np.float32)))
    m.SetNormalsInterpolation("vertex")
    uv = getattr(tm.visual, "uv", None)
    if uv is not None and len(uv) == len(tm.vertices):
        pv = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        pv.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, dtype=np.float32)))
    UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def _dae_parts(name):
    """Collada file -> list of (part name, trimesh in file frame, rgba)."""
    scene = trimesh.load(MESH_DIR / name, force="scene")
    parts = []
    for node in scene.graph.nodes_geometry:
        tf, geom_name = scene.graph[node]
        g = scene.geometry[geom_name].copy()
        g.apply_transform(tf)
        rgba = np.asarray(g.visual.material.baseColorFactor, dtype=float) / 255.0
        parts.append((geom_name.replace("-mesh", "").replace(".", "_"), g, rgba))
    return parts


def _collider_box(stage, path, centre, roll, size):
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    x = UsdGeom.Xformable(cube)
    x.AddTranslateOp().Set(Gf.Vec3d(*centre))
    x.AddRotateXYZOp().Set(Gf.Vec3f(math.degrees(roll), 0.0, 0.0))
    x.AddScaleOp().Set(Gf.Vec3f(*size))
    cube.CreatePurposeAttr(UsdGeom.Tokens.guide)  # physics only, never rendered
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())


def build(out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    tex_dir = out.parent / "textures"
    tex_dir.mkdir(exist_ok=True)
    shutil.copy(MESH_DIR / "CF.png", tex_dir / "CF.png")

    stage = Usd.Stage.CreateNew(str(out))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)

    vehicle = UsdGeom.Xform.Define(stage, "/vehicle")
    stage.SetDefaultPrim(vehicle.GetPrim())
    UsdPhysics.ArticulationRootAPI.Apply(vehicle.GetPrim())
    vehicle.GetPrim().SetMetadata("comment", "Holybro X500 V2 -- converted from PX4 v1.14.3 gz model x500")

    looks = "/vehicle/Looks"
    mats = {}

    def mat_for(part, rgba):
        if part not in mats:
            if part == "CarbonFiber":
                mats[part] = _material(stage, f"{looks}/{part}", (1, 1, 1), 0.35, 0.0, "./textures/CF.png")
            elif part == "FMUK66":
                # The real X500 V2 kit ships a Pixhawk 6C (dark grey case), not NXP's white FMUK66.
                mats[part] = _material(stage, f"{looks}/{part}", (0.12, 0.12, 0.13), 0.5)
            else:
                metal = 0.8 if part == "Metal" else 0.0
                mats[part] = _material(stage, f"{looks}/{part}", tuple(rgba[:3]), 0.5, metal)
        return mats[part]

    # ---- body ---------------------------------------------------------------------------------
    body = UsdGeom.Xform.Define(stage, "/vehicle/body").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(body)
    mass = UsdPhysics.MassAPI.Apply(body)
    mass.CreateMassAttr(BODY_MASS)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*BODY_INERTIA))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))

    frame = UsdGeom.Xform.Define(stage, "/vehicle/body/frame").GetPrim()
    _xform(frame, FRAME_POSE[:3], yaw=FRAME_POSE[3])
    for part, g, rgba in _dae_parts("NXP-HGD-CF.dae"):
        _mesh(stage, f"/vehicle/body/frame/{part}", g, mat_for(part, rgba))

    base_parts = _dae_parts("5010Base.dae")
    for i, (x, y) in enumerate(MOTOR_BASE_POSES):
        mb = UsdGeom.Xform.Define(stage, f"/vehicle/body/motor_base_{i}").GetPrim()
        _xform(mb, (x, y, MOTOR_BASE_Z), yaw=MOTOR_BASE_YAW)
        for part, g, rgba in base_parts:
            _mesh(stage, f"/vehicle/body/motor_base_{i}/{part}", g, mat_for("motor_" + part, rgba))

    UsdGeom.Scope.Define(stage, "/vehicle/body/collision")
    for name, c, roll, size in BODY_COLLISION:
        _collider_box(stage, f"/vehicle/body/collision/{name}", c, roll, size)

    # ---- rotors -------------------------------------------------------------------------------
    bell_parts = _dae_parts("5010Bell.dae")
    prop_mat = _material(stage, f"{looks}/prop", PROP_COLOR, 0.4)
    for i, ((x, y), prop_file) in enumerate(zip(ROTOR_XY, ROTOR_PROP)):
        rp = f"/vehicle/rotor{i}"
        rotor = UsdGeom.Xform.Define(stage, rp).GetPrim()
        _xform(rotor, (x, y, ROTOR_Z))
        UsdPhysics.RigidBodyAPI.Apply(rotor)
        rm = UsdPhysics.MassAPI.Apply(rotor)
        rm.CreateMassAttr(ROTOR_MASS)
        rm.CreateDiagonalInertiaAttr(Gf.Vec3f(*ROTOR_INERTIA))

        prop_x = UsdGeom.Xform.Define(stage, f"{rp}/prop").GetPrim()
        _xform(prop_x, PROP_POSE, scale=PROP_SCALE)
        _mesh(stage, f"{rp}/prop/mesh", trimesh.load(MESH_DIR / prop_file, force="mesh"), prop_mat)

        bell = UsdGeom.Xform.Define(stage, f"{rp}/bell").GetPrim()
        _xform(bell, (0, 0, BELL_Z))
        for part, g, rgba in bell_parts:
            _mesh(stage, f"{rp}/bell/{part}", g, mat_for("bell_" + part, rgba))

        _collider_box(stage, f"{rp}/collision", (0, 0, 0), 0.0, ROTOR_COLLISION_SIZE)

        joint = UsdPhysics.RevoluteJoint.Define(stage, f"{rp}/joint{i}")
        joint.CreateAxisAttr("Z")
        joint.CreateBody0Rel().SetTargets(["/vehicle/body"])
        joint.CreateBody1Rel().SetTargets([rp])
        joint.CreateLocalPos0Attr(Gf.Vec3f(x, y, ROTOR_Z))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
        joint.CreateLocalRot0Attr(Gf.Quatf(1, 0, 0, 0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))

    stage.GetRootLayer().customLayerData = {
        "source": "PX4-Autopilot v1.14.3 Tools/simulation/gz/models/x500",
        "spawn_height_m": -SKID_BOTTOM,
        "rotor_order": "0 FR ccw, 1 BL ccw, 2 FL cw, 3 BR cw",
    }
    stage.GetRootLayer().Save()
    return stage


def report(out: Path):
    stage = Usd.Stage.Open(str(out))
    total = 0.0
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.MassAPI):
            m = UsdPhysics.MassAPI(p).GetMassAttr().Get()
            total += m
            print(f"  {p.GetPath()}: {m:.4f} kg")
    bbox = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_]).ComputeWorldBound(stage.GetPrimAtPath("/vehicle"))
    r = bbox.ComputeAlignedRange()
    print(f"  total mass {total:.4f} kg | visual bbox min {r.GetMin()} max {r.GetMax()}")
    print(f"  spawn height {-SKID_BOTTOM:.4f} m (skid bottom at base_link z {SKID_BOTTOM:.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=PROJECT / "assets/x500/x500.usd")
    a = ap.parse_args()
    build(a.out)
    report(a.out)
    print("wrote", a.out)
