"""Mountain variant of hq_bridge.HQBridge for Simulation_0_0_6, mirroring city_bridge.py's
structure exactly: the validated HQ machinery (Isaac Sim 5.1 + OmniGibson kit experience +
RealTimePathTracing settings extraction/reapplication) is inherited unchanged; only
_setup_world() is overridden to build the mountain instead of the parent's flat plane + dome/sun.

natural_valley.py is imported ONLY for its material/instancer/mesh helpers (create_pbr_material,
create_asset_material, point_instancer, cube, cylinder, create_mesh, ASSET_IDS, asset_path,
texture_path). None of its own terrain/river/vegetation/landmark code runs here -- the mountain is
entirely mountain_scene.py's own geometry.
"""

import sys
from pathlib import Path

import hq_bridge

VLN_SIMULATION_PATH = "/mnt/weka/sgrigoryan/dataset_codebase/common/vln_simulation"
MOUNTAIN_ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-mountain"
FOREST_ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-v2"  # pine/fir/rock props reused as-is


class MountainBridge(hq_bridge.HQBridge):
    def __init__(self, run_dir, assets_root=MOUNTAIN_ASSETS_ROOT,
                  forest_assets_root=FOREST_ASSETS_ROOT, seed=7):
        self.mountain_assets_root = str(assets_root)
        self.forest_assets_root = str(forest_assets_root)
        self.mountain_seed = int(seed)
        self.mountain_info = None
        super().__init__(run_dir)

    def _setup_world(self):
        UsdGeom, stage = self.UsdGeom, self.stage
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())

        if VLN_SIMULATION_PATH not in sys.path:
            sys.path.insert(0, VLN_SIMULATION_PATH)
        import natural_valley
        import mountain_scene

        self.natural_valley = natural_valley
        self.mountain_scene = mountain_scene

        manifest = Path(self.mountain_assets_root) / "asset_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Mountain asset pack missing at {self.mountain_assets_root}. Run "
                f"fetch_mountain_assets.py --root {self.mountain_assets_root}")

        # Must pass self.mountain_seed explicitly: validate_layout()'s own default (seed=7) would
        # silently check a DIFFERENT terrain shape than the one about to be built whenever this
        # bridge is ever constructed with a different --seed. Same class of bug as the vegetation
        # seed mismatch this session already found and fixed (see build_vegetation's docstring).
        issues, peak_h = mountain_scene.validate_layout(seed=self.mountain_seed)
        if issues:
            raise RuntimeError("mountain layout invalid: " + "; ".join(issues))
        print(f"mountain layout validated: peak height {peak_h:.1f}m", flush=True)

        print(f"building mountain (seed={self.mountain_seed}, assets={self.mountain_assets_root})...",
              flush=True)
        self.mountain_info = mountain_scene.build_mountain(
            stage, natural_valley, self.mountain_assets_root, self.forest_assets_root,
            seed=self.mountain_seed)
        print(f"mountain built: {self.mountain_info}", flush=True)

    def populate_scene(self, scene_objects):
        """No-op: the mountain IS the scene (kept for HQBridge interface compatibility)."""
        return

    def terrain_height(self, x, y):
        return self.mountain_scene.terrain_height(x, y, self.mountain_seed)

    def ground_height(self, x, y):
        return self.terrain_height(x, y)
