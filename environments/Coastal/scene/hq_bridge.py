"""Isaac Sim 5.1 + OmniGibson HQ (RealTimePathTracing) render bridge for Simulations_0_0_3.
Ported from Simulations_0_0_2's hq_bridge.py (same validated app-launch/renderer mechanics),
extended with city object categories (building, bus_station, bus, streetlight, bench, trash_can)
for the OpenFly-scale, many-object city scene this project's scene_setup.py builds.

Two things are combined here, each independently validated:

1. The app-launch/renderer-settings mechanics -- ported near-verbatim from the reference harness
   at VLN-Simulations@1e9b0c8 (experiments/renderer_comparison/run_isaac.py), whose HQ backend
   produced a real, evidenced H100 render (see its committed manifest.json: 15 captures, 280s,
   render_mode=RealTimePathTracing, GPU=H100 80GB HBM3, run by a colleague on this same cluster).
   This piece is: construct SimulationApp with OmniGibson's omnigibson_5_1_0.kit "experience",
   extract-and-execute OmniGibson's actual _set_renderer_settings() function (via AST, not just
   copy-pasted values -- so it can't silently drift from upstream) with ENABLE_HQ_RENDERING=True,
   and reapply the settings it captured AFTER opening our own stage (settings-only application
   before stage-open silently reverted to RaytracedLighting in earlier attempts -- see the guide's
   own troubleshooting section).

2. The scene-building/object-spawning code -- reused directly from
   Simulations_1_0_0/pipeline/isaac_bridge.py (proven working on real hardware 2026-09-10: cars,
   trees, hydrants render with correct materials/colors). Confirmed portable here because
   natural_valley.py (the reference repo's own scene builder) uses the exact same plain
   pxr.UsdGeom/UsdShade/UsdLux authoring -- object spawning is Kit-experience-agnostic USD, only
   the renderer backend/settings differ.

NOT reused from the reference repo: its full PX4/MAVLink/Pegasus mission-generation system
(scripts/run_batch.py, simulation/generate_episode.py). That is a much heavier, separate stack
(built PX4-Autopilot binary, PegasusSimulator extension, a Poly Haven asset pack pinned to the
original authors' own path) which the pasted HQ-rendering guide's own scope note explicitly
excludes: "The completed tests cover rendering of static scenes and camera views, not a full
OmniGibson household task, PX4 flight, or physics/sensor parity." We use only the validated
rendering piece; our own trajectory controller (flight_controller.py) drives the camera, same as
Simulations_1_0_0's and the original Simulations_0_0_1/0_0_2 approach.

Ground-clearance and terrain-extent fixes match the ones just validated in Simulations_1_0_0
(2026-09-10) -- both are pure-geometry issues (camera embedded in solid ground at z=0; ground
plane smaller than the flight envelope), not renderer-specific, so they apply here unchanged.
Lighting intensities are NOT reused from that project as-is: RealTimePathTracing is physically
based and follows a different exposure convention than the simpler RTX real-time mode tuned
there. Starting values here are a first guess pending an actual H100 smoke test.
"""

import ast
import hashlib
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np

FRAME_SIZE = (640, 480)
RT_SUBFRAMES = 16  # matches the validated manifest's implicit default (16 for 'flight'-kind poses;
                    # reference used 64 only for static 'view' poses in the pathtracing-only backend,
                    # not behavior-hq -- 16 is the HQ backend's actual proven value)

BEHAVIOR_ROOT = Path("/mnt/weka/sgrigoryan/Simulations_0_0_2/BEHAVIOR-1K")
ISAAC_PATH = Path("/home/sgrigoryan/.conda/envs/behavior_h100/lib/python3.11/site-packages/isaacsim")

ASSET_SCALE_CM = 0.01  # NVIDIA Vegetation/ArchVis content is authored at 1 unit = 1cm
KNOWN_TREE_ASSETS = {
    "oak_tree": ("NVIDIA/Assets/Vegetation/Trees/Red_Oak.usd", ASSET_SCALE_CM),
    "pine_tree": ("NVIDIA/Assets/Vegetation/Trees/White_Pine.usd", ASSET_SCALE_CM),
    "palm_tree": ("NVIDIA/Assets/Vegetation/Plant_Tropical/Windmill_Palm.usd", ASSET_SCALE_CM),
}

COLOR_RGB = {
    "red": (0.75, 0.08, 0.08),
    "green": (0.1, 0.55, 0.15),
    "blue": (0.1, 0.2, 0.75),
    "yellow": (0.85, 0.75, 0.1),
    "white": (0.85, 0.85, 0.85),
    "black": (0.05, 0.05, 0.05),
}


def extract_rgb(raw, frame_size):
    data = np.asarray(raw)
    w, h = frame_size
    if data.ndim >= 3:
        return data[:, :, :3].astype(np.uint8)
    if data.dtype.names:
        return np.stack([data[c] for c in data.dtype.names[:3]], axis=-1).reshape(h, w, 3).astype(np.uint8)
    n_channels = data.size // (h * w)
    return data.reshape(h, w, n_channels)[:, :, :3].astype(np.uint8)


def yaw_pitch_to_quat_wxyz(yaw_deg, pitch_deg=0.0):
    """Camera orientation as q = Rz(yaw-90) * Rx(90 - pitch).

    yaw is CCW from +X, pitch is POSITIVE DOWN (gimbal tilt), up = +Z, so the resulting forward
    vector is (cos(pitch)cos(yaw), cos(pitch)sin(yaw), -sin(pitch)). pitch=0 reproduces the
    original level-camera derivation from Simulations_1_0_0/pipeline/isaac_bridge.py exactly.

    Pitch was added 2026-09-11: with a level-only camera, a drone orbiting a ground target from
    above puts the target far below the view axis (at 16m radius and 14m up that's ~41deg, well
    outside the ~+-23deg vertical FOV) -- the target simply isn't in frame. A real drone gimbals
    down; this lets the mission aim the camera at what it's inspecting.

    Verified against real USD ComputeLocalToWorldTransform for 20 yaw/pitch combinations before
    being used (same empirical-verification discipline as the original level-only derivation).
    """
    half_yaw = math.radians(yaw_deg - 90.0) / 2.0
    w1, x1, y1, z1 = math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)
    half_tilt = math.radians(90.0 - pitch_deg) / 2.0
    w2, x2, y2, z2 = math.cos(half_tilt), math.sin(half_tilt), 0.0, 0.0
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return w, x, y, z


def yaw_to_quat_wxyz(yaw_deg):
    """Backwards-compatible level-camera helper (city missions call set_camera_pose without pitch)."""
    return yaw_pitch_to_quat_wxyz(yaw_deg, 0.0)


class SettingsRecorder:
    """Wraps carb.settings.get_settings() so every .set*() call OmniGibson's
    _set_renderer_settings() makes is both applied immediately AND recorded, so it can be
    reapplied verbatim after we open our own stage (matches run_isaac.py's mechanism)."""

    def __init__(self, settings, sink):
        self._settings = settings
        self._sink = sink

    def __getattr__(self, key):
        original = getattr(self._settings, key)
        if key.startswith("set"):
            def setter(path, value):
                self._sink[path] = value
                return original(path, value)
            return setter
        return original


class HQBridge:
    def __init__(self, run_dir):
        """run_dir: a unique, writable directory for this job's Kit caches (portable-root,
        private experience/apps copy) -- mirrors launch-h100-hq.sh's RUN_DIR."""
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "kit-cache").mkdir(parents=True, exist_ok=True)

        os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
        os.environ["ISAAC_PORTABLE_ROOT"] = str(self.run_dir / "kit-cache")

        # ---- private experience dir: symlink runtime ext folders, copy the pinned kit file ----
        private = self.run_dir / "experience"
        (private / "apps").mkdir(parents=True, exist_ok=True)
        for folder in ["exts", "extscache", "extsPhysics", "extsUser", "extsDeprecated"]:
            src = ISAAC_PATH / folder
            dst = private / folder
            if src.exists() and not dst.exists():
                dst.symlink_to(src, target_is_directory=True)
        kit_source = BEHAVIOR_ROOT / "OmniGibson/omnigibson/omnigibson_5_1_0.kit"
        kit_path = private / "apps" / kit_source.name
        shutil.copy2(kit_source, kit_path)
        self.experience_sha256 = hashlib.sha256(kit_source.read_bytes()).hexdigest()
        os.environ["MDL_USER_PATH"] = str(BEHAVIOR_ROOT / "OmniGibson/omnigibson/materials")

        launch_args = [
            "--portable-root", str(self.run_dir / "kit-cache"),
            "--/rtx/rendermode=RealTimePathTracing",
            "--/persistent/rtx/modes/rt/enabled=true",
            "--/persistent/rtx/modes/rt2/enabled=true",
            "--/persistent/rtx/modes/pt/enabled=true",
        ]

        from isaacsim import SimulationApp

        self.app = SimulationApp(
            {
                "headless": True,
                "width": FRAME_SIZE[0],
                "height": FRAME_SIZE[1],
                "multi_gpu": False,
                "renderer": "RealTimePathTracing",
                "extra_args": launch_args,
                "limit_cpu_threads": int(os.environ.get("OMP_NUM_THREADS", "8")),
                "fast_shutdown": True,
            },
            experience=str(kit_path),
        )
        print("PHASE app_started", flush=True)

        import carb
        import omni.replicator.core as rep
        import omni.usd
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

        self.rep, self.Gf, self.Sdf = rep, Gf, Sdf
        self.Usd, self.UsdGeom, self.UsdLux, self.UsdShade = Usd, UsdGeom, UsdLux, UsdShade

        self.stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        # ---- extract & run OmniGibson's OWN _set_renderer_settings, recording what it sets ----
        settings = carb.settings.get_settings()
        applied = {}
        sim_source = BEHAVIOR_ROOT / "OmniGibson/omnigibson/simulator.py"
        text = sim_source.read_text()
        self.behavior_source_sha256 = hashlib.sha256(text.encode()).hexdigest()
        fn = next(
            n for n in ast.walk(ast.parse(text))
            if isinstance(n, ast.FunctionDef) and n.name == "_set_renderer_settings"
        )
        scope = {
            "lazy": SimpleNamespace(carb=SimpleNamespace(settings=SimpleNamespace(
                get_settings=lambda: SettingsRecorder(settings, applied)))),
            "gm": SimpleNamespace(ENABLE_HQ_RENDERING=True, ENABLE_FLATCACHE=True),
        }
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(sim_source), "exec"), scope)
        scope["_set_renderer_settings"](SimpleNamespace(get_rendering_dt=lambda: 1 / 60))
        self._applied_settings = applied
        self._settings = settings

        self.assets_root = self._resolve_assets_root()
        self._setup_world()

        # reapply the recorded settings AFTER our stage's own prims exist -- settings applied only
        # before this point were observed reverting to RaytracedLighting in the reference harness.
        for key, value in applied.items():
            settings.set(key, value)

        self.camera_path = "/World/EgoCamera"
        cam = UsdGeom.Camera.Define(self.stage, self.camera_path)
        cam.CreateFocalLengthAttr(24.0)
        cam.CreateHorizontalApertureAttr(20.955)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10000.0))
        xf = UsdGeom.Xformable(cam.GetPrim())
        self._cam_translate_op = xf.AddTranslateOp()
        self._cam_orient_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)

        rep.orchestrator.set_capture_on_play(False)
        self.render_product = rep.create.render_product(self.camera_path, tuple(FRAME_SIZE))
        self.rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.rgb_annotator.attach(self.render_product)

        # warmup: path-traced/HQ shaders need real compile+accumulate passes before first capture
        # (reference harness uses 80 app.update() steps before any capture -- kept here unchanged)
        for step in range(80):
            self.app.update()
            if step % 20 == 19:
                print(f"PHASE warmup {step + 1} {settings.get('/rtx/rendermode')}", flush=True)

    # ---- scene construction (verbatim from Simulations_1_0_0/pipeline/isaac_bridge.py) --------

    def _resolve_assets_root(self):
        try:
            from isaacsim.storage.native import get_assets_root_path
        except ImportError:
            try:
                from omni.isaac.core.utils.nucleus import get_assets_root_path
            except ImportError:
                return None
        try:
            return get_assets_root_path()
        except Exception:
            return None

    def _reference_asset(self, prim_path, rel_path, position, scale=1.0, prim_type="Xform"):
        Gf, UsdGeom = self.Gf, self.UsdGeom
        full_path = f"{self.assets_root.rstrip('/')}/{rel_path}"
        try:
            prim = self.stage.DefinePrim(prim_path, prim_type)
            prim.GetReferences().AddReference(full_path)
            api = UsdGeom.XformCommonAPI(prim)
            api.SetTranslate(Gf.Vec3d(*position))
            if scale != 1.0:
                api.SetScale(Gf.Vec3f(scale, scale, scale))
            return prim
        except Exception:
            if self.stage.GetPrimAtPath(prim_path).IsValid():
                self.stage.RemovePrim(prim_path)
            return None

    def _bind_color_material(self, prim, color, roughness=0.6):
        UsdShade, Sdf, Gf = self.UsdShade, self.Sdf, self.Gf
        mat_path = str(prim.GetPath()) + "_Material"
        mat = UsdShade.Material.Define(self.stage, mat_path)
        shader = UsdShade.Shader.Define(self.stage, mat_path + "/Shader")
        shader.CreateIdAttr("OmniPBR")
        shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)
        mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)

    def _setup_world(self):
        """Ground is a large procedural plane whose top face sits well below z=0 -- see the module
        docstring: both the extent (must cover the full flight envelope) and the clearance (avoid
        the camera embedding in solid geometry at ground-level poses) were real bugs found and
        fixed in Simulations_1_0_0 on 2026-09-10, and are pure geometry, not renderer-specific."""
        UsdGeom, UsdLux, Gf, stage = self.UsdGeom, self.UsdLux, self.Gf, self.stage
        UsdGeom.Xform.Define(stage, "/World")

        GROUND_HALF_EXTENT_M = 220.0  # 0_0_3's city scene spreads to ~130m radius -- needs real margin
        GROUND_TOP_Z = -0.5
        GROUND_HALF_THICKNESS = 0.3
        ground = UsdGeom.Cube.Define(stage, "/World/Ground")
        ground.AddTranslateOp().Set(Gf.Vec3f(0.0, 0.0, GROUND_TOP_Z - GROUND_HALF_THICKNESS))
        ground.AddScaleOp().Set(Gf.Vec3f(GROUND_HALF_EXTENT_M, GROUND_HALF_EXTENT_M, GROUND_HALF_THICKNESS))
        ground_color = (0.45, 0.47, 0.4)
        ground.CreateDisplayColorAttr([Gf.Vec3f(*ground_color)])
        self._bind_color_material(ground.GetPrim(), ground_color, roughness=0.85)

        # Real H100 render (2026-09-10, job 255253) showed the first-guess 1500/3000 values badly
        # overexposed -- the red car rendered nearly pink/white, colors crushed toward white.
        # RealTimePathTracing's physically-based exposure needs much lower intensities than the
        # simpler RTX real-time mode. Cut by 10x as a first correction, still pending re-validation.
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(150.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(300.0)
        UsdGeom.Xformable(sun.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-60.0, 20.0, 0.0))

    def populate_scene(self, scene_objects):
        stage = self.stage
        old = stage.GetPrimAtPath("/World/Objects")
        if old.IsValid():
            stage.RemovePrim("/World/Objects")
        stage.DefinePrim("/World/Objects", "Xform")
        for i, obj in enumerate(scene_objects):
            self._spawn_object(f"/World/Objects/obj_{i}", obj)

    def _spawn_object(self, prim_path, obj):
        category, color, asset_key = obj["category"], obj["color"], obj["asset_key"]
        tx, ty = obj["x"], obj["y"]
        prim = None
        if self.assets_root and asset_key in KNOWN_TREE_ASSETS:
            rel_path, scale = KNOWN_TREE_ASSETS[asset_key]
            prim = self._reference_asset(prim_path, rel_path, (tx, ty, 0.0), scale)
        if prim is not None:
            if color:
                self._tint(prim, color)
        else:
            self._spawn_primitive_object(prim_path, obj)

    def _box(self, prim_path, tx, ty, tz, heading_deg, w, d, h, rgb, roughness=0.6):
        """Axis-aligned-at-rest box, optionally yawed, then non-uniformly scaled to (w,d,h).
        Op order Translate-then-Rotate-then-Scale (code order) matches the translate-must-precede-
        scale rule already proven this session (Simulations_1_0_0/isaac_bridge.py): each later op
        is the INNER (local) transform, so scale/rotate never distort the world-space placement."""
        UsdGeom, Gf, stage = self.UsdGeom, self.Gf, self.stage
        body = UsdGeom.Cube.Define(stage, prim_path)
        xf = UsdGeom.Xformable(body)
        xf.AddTranslateOp().Set(Gf.Vec3f(tx, ty, tz))
        if heading_deg:
            xf.AddRotateZOp().Set(heading_deg)
        xf.AddScaleOp().Set(Gf.Vec3f(w / 2.0, d / 2.0, h / 2.0))  # USD Cube default size=2 (half-extent 1)
        body.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        self._bind_color_material(body.GetPrim(), rgb, roughness=roughness)
        return body

    def _cylinder(self, prim_path, tx, ty, tz, radius, height, rgb, roughness=0.6):
        UsdGeom, Gf, stage = self.UsdGeom, self.Gf, self.stage
        body = UsdGeom.Cylinder.Define(stage, prim_path)
        body.CreateRadiusAttr(radius)
        body.CreateHeightAttr(height)
        UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3f(tx, ty, tz))
        body.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        self._bind_color_material(body.GetPrim(), rgb, roughness=roughness)
        return body

    def _spawn_primitive_object(self, prim_path, obj):
        category, color, tx, ty = obj["category"], obj["color"], obj["x"], obj["y"]
        rgb = COLOR_RGB.get(color, (0.55, 0.55, 0.55))
        if category == "car":
            self._box(prim_path, tx, ty, 0.7, 0.0, 2.2, 1.0, 0.7 * 2, rgb, roughness=0.3)
        elif category == "tree":
            self._cylinder(prim_path + "_trunk", tx, ty, 1.0, 0.2, 2.0, (0.35, 0.22, 0.1), roughness=0.9)
            canopy = self.UsdGeom.Sphere.Define(self.stage, prim_path)
            canopy.CreateRadiusAttr(1.3)
            self.UsdGeom.Xformable(canopy).AddTranslateOp().Set(self.Gf.Vec3f(tx, ty, 2.6))
            canopy.CreateDisplayColorAttr([self.Gf.Vec3f(0.15, 0.45, 0.15)])
            self._bind_color_material(canopy.GetPrim(), (0.15, 0.45, 0.15), roughness=0.9)
        elif category == "hydrant":
            self._cylinder(prim_path, tx, ty, 0.4, 0.25, 0.8, rgb, roughness=0.5)
        elif category == "trash_can":
            self._cylinder(prim_path, tx, ty, 0.5, 0.35, 1.0, (0.2, 0.2, 0.22), roughness=0.6)
        elif category == "bench":
            self._box(prim_path, tx, ty, 0.4, obj.get("heading_deg", 0.0), 1.6, 0.5, 0.8,
                      (0.4, 0.28, 0.15), roughness=0.7)
        elif category == "streetlight":
            self._cylinder(prim_path + "_pole", tx, ty, 3.0, 0.1, 6.0, (0.15, 0.15, 0.15), roughness=0.4)
            self._box(prim_path, tx, ty, 6.1, 0.0, 0.5, 0.5, 0.3, (0.9, 0.85, 0.6), roughness=0.2)
        elif category == "building":
            w, d, h = obj.get("width", 8.0), obj.get("depth", 8.0), obj.get("height", 12.0)
            color_rgb = obj.get("color_rgb", (0.6, 0.6, 0.55))
            self._box(prim_path, tx, ty, h / 2.0, 0.0, w, d, h, color_rgb, roughness=0.5)
        elif category == "bus":
            heading = obj.get("heading_deg", 0.0)
            self._box(prim_path, tx, ty, 1.5, heading, 9.0, 2.5, 3.0, rgb, roughness=0.35)
        elif category == "bus_station":
            heading = obj.get("heading_deg", 0.0)
            # platform
            self._box(prim_path + "_platform", tx, ty, 0.1, heading, 8.0, 3.0, 0.2,
                      (0.5, 0.5, 0.5), roughness=0.8)
            # two support poles + a flat roof, offset along the platform's long (heading) axis
            rad = math.radians(heading)
            along_x, along_y = math.cos(rad), math.sin(rad)
            for sign, suffix in ((-1, "_pole_a"), (1, "_pole_b")):
                px = tx + sign * 3.2 * along_x
                py = ty + sign * 3.2 * along_y
                self._cylinder(prim_path + suffix, px, py, 1.3, 0.12, 2.6, (0.3, 0.3, 0.3), roughness=0.4)
            self._box(prim_path + "_roof", tx, ty, 2.7, heading, 8.5, 3.2, 0.15,
                      (0.75, 0.2, 0.15), roughness=0.4)
        else:
            self._cylinder(prim_path, tx, ty, 0.4, 0.25, 0.8, rgb, roughness=0.5)

    def _tint(self, prim, color_name):
        color = COLOR_RGB.get(color_name)
        if color is None:
            return
        UsdGeom, Gf = self.UsdGeom, self.Gf
        for desc in self.Usd.PrimRange(prim):
            if desc.IsA(UsdGeom.Gprim):
                UsdGeom.Gprim(desc).CreateDisplayColorAttr([Gf.Vec3f(*color)])

    # ---- capture interface -----------------------------------------------------------------

    def set_camera_pose(self, x, y, z, yaw_deg, pitch_deg=0.0):
        """pitch_deg is positive-DOWN gimbal tilt; defaults to 0 (level) so existing callers are
        unaffected."""
        self._cam_translate_op.Set(self.Gf.Vec3d(x, y, z))
        w, qx, qy, qz = yaw_pitch_to_quat_wxyz(yaw_deg, pitch_deg)
        self._cam_orient_op.Set(self.Gf.Quatf(w, qx, qy, qz))

    def process_camera_data(self):
        """rt_subframes=16 matches the validated HQ manifest's proven accumulation depth (see
        module docstring) -- explicit accumulation-subframe count, not a bare repeated .step()
        loop like Simulations_1_0_0's ACCUMULATION_STEPS, since this API supports it directly."""
        rep = self.rep
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, pause_timeline=False)
        pixels = extract_rgb(self.rgb_annotator.get_data(), FRAME_SIZE)
        for _ in range(8):
            if pixels.shape[:2] == (FRAME_SIZE[1], FRAME_SIZE[0]):
                break
            rep.orchestrator.step(rt_subframes=RT_SUBFRAMES, pause_timeline=False)
            pixels = extract_rgb(self.rgb_annotator.get_data(), FRAME_SIZE)
        return pixels

    def current_render_mode(self):
        return self._settings.get("/rtx/rendermode")

    def shutdown(self):
        if self.app is not None:
            self.app.close()
