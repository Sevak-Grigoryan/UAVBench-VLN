"""Fast material probe: which material construction actually renders TEXTURED in Isaac Sim?

Two diagnoses have now been wrong (uv_scale compounding, then UsdTransform2d), each costing a
~35 minute render to disprove. This renders one small scene containing several walls side by
side, each built a different way, so a single ~5 minute job answers it definitively.

Every wall uses brick_4, whose diffuse is vivid red with std 35.9 -- if a variant textures at
all, it is unmistakable against the flat-colour failure mode.

Variants:
  A  UsdPreviewSurface, texture.st -> primvar reader, varname as Token   (what v4 shipped)
  B  UsdPreviewSurface, same but varname as String
  C  OmniPBR MDL + diffuse_texture + texture_scale                       (the path _bind_color_material uses)
  D  OmniPBR MDL + diffuse_texture + project_uvw=True                    (ignores UVs entirely)
  E  a real prop model as a control -- known to render textured
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)

import hq_bridge

VLN_SIMULATION_PATH = "/mnt/weka/sgrigoryan/Simulations_0_0_2/VLN-Simulations/simulation"
CITY_ASSETS = "/mnt/weka/sgrigoryan/assets/polyhaven-city"
TEX = "brick_4"

WALL_W, WALL_H, PITCH = 12.0, 9.0, 15.0
TILE_M = 3.0            # one brick-texture repeat every 3m


class ProbeBridge(hq_bridge.HQBridge):
    def _setup_world(self):
        UsdGeom, UsdLux, Gf, stage = self.UsdGeom, self.UsdLux, self.Gf, self.stage
        UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))

        if VLN_SIMULATION_PATH not in sys.path:
            sys.path.insert(0, VLN_SIMULATION_PATH)
        import natural_valley as nv
        import city_scene as cs

        self.nv, self.cs = nv, cs
        diffuse = nv.texture_path(CITY_ASSETS, TEX, "diffuse")
        print(f"probe texture: {diffuse}", flush=True)

        # flat ground so the walls have something to stand on
        ground = nv.create_color_material(stage, "/World/M/Ground", (0.35, 0.35, 0.35), 0.8)
        cs.textured_box(nv, stage, "/World/Ground", (0.0, 0.0, -0.5), (400.0, 400.0, 1.0), ground)

        variants = [
            ("A_preview_token", self._preview, {"varname_as_string": False}),
            ("B_preview_string", self._preview, {"varname_as_string": True}),
            ("C_omnipbr_uv", self._omnipbr, {"project_uvw": False}),
            ("D_omnipbr_projected", self._omnipbr, {"project_uvw": True}),
        ]
        n = len(variants)
        for i, (name, factory, kw) in enumerate(variants):
            x = (i - (n - 1) / 2.0) * PITCH
            mat = factory(f"/World/M/{name}", diffuse, **kw)
            cs.textured_box(nv, stage, f"/World/Wall_{name}",
                            (x, 0.0, WALL_H / 2.0), (WALL_W, 0.6, WALL_H), mat, tile_m=TILE_M)
            print(f"  wall {name} at x={x:+.1f}", flush=True)

        # control: a real prop model, which has always rendered textured
        try:
            lamp_mat = nv.create_asset_material(stage, "/World/M/E_control_model",
                                                CITY_ASSETS, "street_lamp_01")
            nv.point_instancer(stage, "/World/Wall_E_control_model",
                               [(cs.model_path(CITY_ASSETS, "street_lamp_01"), 1.0, lamp_mat)],
                               [(0, (x + PITCH, 0.0, 0.0), 0.0, (3.0, 3.0, 3.0))])
            print("  control: street_lamp_01 instanced", flush=True)
        except Exception as exc:
            print(f"  control model failed: {exc}", flush=True)

        sky = UsdLux.DomeLight.Define(stage, "/World/Sky")
        sky.CreateIntensityAttr(900.0)
        sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
        sun.CreateIntensityAttr(1400.0)
        sun.CreateAngleAttr(0.6)
        sun.AddRotateXYZOp().Set(Gf.Vec3f(42.0, -15.0, 30.0))

    def _preview(self, path, diffuse, varname_as_string=False):
        UsdShade, Sdf = self.UsdShade, self.Sdf
        mat = UsdShade.Material.Define(self.stage, path)
        surf = UsdShade.Shader.Define(self.stage, f"{path}/Surface")
        surf.CreateIdAttr("UsdPreviewSurface")
        surf.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.82)
        st = UsdShade.Shader.Define(self.stage, f"{path}/TexCoord")
        st.CreateIdAttr("UsdPrimvarReader_float2")
        vt = Sdf.ValueTypeNames.String if varname_as_string else Sdf.ValueTypeNames.Token
        st.CreateInput("varname", vt).Set("st")
        tex = UsdShade.Shader.Define(self.stage, f"{path}/DiffuseTexture")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse)
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        surf.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.ConnectableAPI(), "rgb")
        surf.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(surf.ConnectableAPI(), "surface")
        return mat

    def _omnipbr(self, path, diffuse, project_uvw=False):
        """Same construction HQBridge._bind_color_material already uses successfully, plus a
        diffuse texture. texture_scale is in repeats-per-uv-unit; textured_box bakes UVs in
        metres already, so 1.0 keeps the tile_m tiling."""
        UsdShade, Sdf, Gf = self.UsdShade, self.Sdf, self.Gf
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, f"{path}/Shader")
        sh.CreateIdAttr("OmniPBR")
        sh.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
        sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
        sh.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(diffuse)
        sh.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(0.82)
        sh.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(bool(project_uvw))
        if project_uvw:
            # world-space projection: scale is in world units per repeat
            sh.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
            sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
                Gf.Vec2f(1.0 / TILE_M, 1.0 / TILE_M))
        else:
            sh.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0, 1.0))
        mat.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
        return mat


def main():
    out = Path("/mnt/weka/sgrigoryan/Simulation_0_0_5/probe_materials")
    out.mkdir(parents=True, exist_ok=True)
    import os
    run_dir = Path(f"/mnt/weka/sgrigoryan/Simulation_0_0_5/_runs/probe_{os.environ.get('SLURM_JOB_ID','local')}")
    bridge = ProbeBridge(run_dir)
    print(f"ProbeBridge booted, render_mode={bridge.current_render_mode()}", flush=True)
    from PIL import Image

    # yaw convention is derived empirically rather than assumed: render all four cardinal
    # headings and keep them all, so one of them is guaranteed to face the wall row.
    try:
        for yaw in (0.0, 90.0, 180.0, 270.0):
            bridge.set_camera_pose(0.0, -42.0, 7.0, yaw, 6.0)
            frame = bridge.process_camera_data()
            Image.fromarray(frame).save(out / f"probe_yaw{int(yaw):03d}.png")
            print(f"wrote probe_yaw{int(yaw):03d}.png", flush=True)
        # and a close pass along the row at the best guess, for texture detail
        for i, x in enumerate((-22.5, -7.5, 7.5, 22.5, 37.5)):
            bridge.set_camera_pose(x, -16.0, 5.0, 90.0, 4.0)
            frame = bridge.process_camera_data()
            Image.fromarray(frame).save(out / f"probe_close_{i}.png")
            print(f"wrote probe_close_{i}.png", flush=True)
    finally:
        bridge.shutdown()


if __name__ == "__main__":
    main()
