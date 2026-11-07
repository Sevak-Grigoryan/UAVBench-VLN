"""Coastal variant of hq_bridge.HQBridge for Simulation_0_0_7, same structure as
mountain_bridge.py: the validated HQ machinery is inherited unchanged; only _setup_world() is
overridden to build the coastline instead of the parent's flat plane + dome/sun.
"""

import sys
from pathlib import Path

import hq_bridge

VLN_SIMULATION_PATH = "/mnt/weka/sgrigoryan/dataset_codebase/common/vln_simulation"
COASTAL_ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-coastal"
FOREST_ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-v2"   # grass/shrub/rock props reused


class CoastalBridge(hq_bridge.HQBridge):
    def __init__(self, run_dir, assets_root=COASTAL_ASSETS_ROOT,
                  forest_assets_root=FOREST_ASSETS_ROOT, seed=7):
        self.coastal_assets_root = str(assets_root)
        self.forest_assets_root = str(forest_assets_root)
        self.coastal_seed = int(seed)
        self.coastal_info = None
        super().__init__(run_dir)

    def _setup_world(self):
        UsdGeom, stage = self.UsdGeom, self.stage
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        if VLN_SIMULATION_PATH not in sys.path:
            sys.path.insert(0, VLN_SIMULATION_PATH)
        import natural_valley
        import coastal_scene

        self.natural_valley = natural_valley
        self.coastal_scene = coastal_scene

        manifest = Path(self.coastal_assets_root) / "asset_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Coastal asset pack missing at {self.coastal_assets_root}. Run "
                f"fetch_coastal_assets.py --root {self.coastal_assets_root}")

        # Must pass self.coastal_seed explicitly -- validate_layout()'s own default (seed=7)
        # would silently check a DIFFERENT terrain shape than the one about to be built whenever
        # this bridge is constructed with a different --seed (the mountain project's own bug,
        # fixed there and not repeated here).
        issues, lh_h = coastal_scene.validate_layout(seed=self.coastal_seed)
        if issues:
            raise RuntimeError("coastal layout invalid: " + "; ".join(issues))
        print(f"coastal layout validated: lighthouse height {lh_h:.1f}m", flush=True)

        print(f"building coastline (seed={self.coastal_seed}, assets={self.coastal_assets_root})...",
              flush=True)
        self.coastal_info = coastal_scene.build_coastal(
            stage, natural_valley, self.coastal_assets_root, self.forest_assets_root,
            seed=self.coastal_seed)
        print(f"coastline built: {self.coastal_info}", flush=True)

    def populate_scene(self, scene_objects):
        return

    def terrain_height(self, x, y):
        return self.coastal_scene.terrain_height(x, y, self.coastal_seed)

    def ground_height(self, x, y):
        return self.terrain_height(x, y)
