"""City variant of hq_bridge.HQBridge for Simulation_0_0_5.

Same validated HQ machinery (Isaac Sim 5.1 + OmniGibson kit experience + RealTimePathTracing
settings extraction/reapplication) inherited unchanged from hq_bridge.HQBridge. _setup_world() is
overridden to build the authored city instead of the parent's flat plane + dome/sun pair, because
city_scene.build_city() brings its own ground, roads and HDRI lighting.

natural_valley.py is imported ONLY for its material/instancer helpers (create_pbr_material,
create_asset_material, point_instancer, cube, cylinder). None of its terrain, river, vegetation or
landmark code runs here -- the city is entirely our own geometry.
"""

import sys
from pathlib import Path

import hq_bridge

VLN_SIMULATION_PATH = "/mnt/weka/sgrigoryan/dataset_codebase/common/vln_simulation"
CITY_ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-city"
FOREST_ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-v2"  # street trees reuse the jacaranda


class CityBridge(hq_bridge.HQBridge):
    def __init__(self, run_dir, assets_root=CITY_ASSETS_ROOT,
                  forest_assets_root=FOREST_ASSETS_ROOT, seed=7):
        # set before super().__init__ -- the parent constructor calls self._setup_world()
        self.city_assets_root = str(assets_root)
        self.forest_assets_root = str(forest_assets_root)
        self.city_seed = int(seed)
        self.city_info = None
        super().__init__(run_dir)

    def _setup_world(self):
        UsdGeom, stage = self.UsdGeom, self.stage
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        if VLN_SIMULATION_PATH not in sys.path:
            sys.path.insert(0, VLN_SIMULATION_PATH)
        import natural_valley
        import city_scene

        self.natural_valley = natural_valley
        self.city_scene = city_scene

        manifest = Path(self.city_assets_root) / "asset_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"City asset pack missing at {self.city_assets_root}. Run "
                f"fetch_city_assets.py --root {self.city_assets_root}")

        issues = city_scene.validate_layout()
        if issues:
            raise RuntimeError("city layout invalid: " + "; ".join(issues))

        print(f"building city (seed={self.city_seed}, assets={self.city_assets_root})...", flush=True)
        self.city_info = city_scene.build_city(stage, natural_valley, self.city_assets_root,
                                               self.forest_assets_root, seed=self.city_seed)
        print(f"city built: {self.city_info}", flush=True)

    def populate_scene(self, scene_objects):
        """No-op: the city IS the scene (kept for HQBridge interface compatibility)."""
        return

    def ground_height(self, x, y):
        """The city is flat; sidewalks are the only raised surface and pads sit on open block
        interiors, so ground is z=0 everywhere the drone takes off from or lands on."""
        return 0.0


class CityBridgeV2(CityBridge):
    """Easy City dataset (2026-10-12): builds city_v2.build(plan) -- a different city per simulation --
    instead of the fixed city_scene.build_city()."""

    def __init__(self, run_dir, plan, **kw):
        self.city_plan = plan
        super().__init__(run_dir, **kw)

    def _setup_world(self):
        UsdGeom, stage = self.UsdGeom, self.stage
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        if VLN_SIMULATION_PATH not in sys.path:
            sys.path.insert(0, VLN_SIMULATION_PATH)
        import natural_valley
        import city_v2

        self.natural_valley = natural_valley
        print(f"building city v2 #{self.city_plan['index']} (attempt {self.city_plan['attempt']})...", flush=True)
        self.city_info = city_v2.build(stage, natural_valley, self.city_plan, self.city_assets_root, self.forest_assets_root)
        print(f"city v2 built: { {k: v for k, v in self.city_info.items() if k != 'rigid_bodies'} }", flush=True)
