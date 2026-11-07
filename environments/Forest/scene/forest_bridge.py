"""Forest/valley variant of hq_bridge.HQBridge for Simulations_0_0_4.

Instead of our own procedural flat ground + primitive objects, this builds the REAL forest scene
from the reference repo's natural_valley.py (VLN-Simulations@1e9b0c8) using the pinned CC0 Poly
Haven asset pack (18 assets, 511MB, SHA256-verified via that repo's own fetch_assets.py +
configs/assets.lock.json): a 161x161 sculpted terrain with a river-cut valley, a 2.4km distant
terrain skirt, a winding river, instanced mature jacaranda trees / fir + pine saplings / ferns /
shrubs / grass / boulders / moss rocks / deadwood / stumps, real photogrammetric cliff faces, and
a mountain HDRI sky with a deterministic sun angle.

Rendering machinery (Isaac Sim 5.1 + OmniGibson kit experience + RealTimePathTracing settings
extraction/reapplication) is inherited unchanged from hq_bridge.HQBridge -- already validated on
H100 for Simulations_0_0_2/0_0_3.

_setup_world() is overridden rather than extended: natural_valley.build_environment() brings its
OWN lighting (HDRI dome + sun) and its own sculpted terrain, so the parent's flat ground plane and
dome/distant-light pair must NOT also be created -- a flat plane would slice through the valley
and the second light set would double-expose the scene.
"""

import sys
from pathlib import Path

import hq_bridge

VLN_SIMULATION_PATH = "/mnt/weka/sgrigoryan/dataset_codebase/common/vln_simulation"
DEFAULT_ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-v2"
DEFAULT_SEED = 5200  # same seed the reference harness used for its published H100 valley renders


class ForestBridge(hq_bridge.HQBridge):
    def __init__(self, run_dir, assets_root=DEFAULT_ASSETS_ROOT, seed=DEFAULT_SEED,
                  smooth_terrain=True, dense_vegetation=True, rebuild_bridge=True, extra_build=None):
        # set before super().__init__ -- the parent constructor calls self._setup_world(), which
        # is where the valley actually gets built, so these must already exist by then.
        self.valley_assets_root = str(assets_root)
        self.valley_seed = int(seed)
        self.valley_info = None
        self.smooth_terrain = smooth_terrain
        self.dense_vegetation = dense_vegetation
        self.rebuild_bridge = rebuild_bridge
        self.extra_build = extra_build  # callable(stage) -> dict, run last (Easy v2 props layout)
        super().__init__(run_dir)

    def _setup_world(self):
        """Build the real valley/forest instead of the parent's flat ground + lights."""
        UsdGeom, stage = self.UsdGeom, self.stage
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())  # natural_valley authors under /World and expects it

        if VLN_SIMULATION_PATH not in sys.path:
            sys.path.insert(0, VLN_SIMULATION_PATH)
        import natural_valley
        import forest_scene

        self.natural_valley = natural_valley
        self.forest_scene = forest_scene
        manifest = Path(self.valley_assets_root) / "asset_manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"Poly Haven asset pack not found at {self.valley_assets_root} (missing "
                f"{manifest.name}). Fetch it with VLN-Simulations/simulation/fetch_assets.py "
                f"--output-root {self.valley_assets_root}"
            )

        # Apply our customizations BEFORE build_environment so the terrain mesh, the landmarks and
        # upstream's own vegetation scatter all use the smooth height field and the relaxed
        # clearings -- patching afterwards would leave the mesh and the scatter disagreeing.
        if self.smooth_terrain:
            forest_scene.patch_terrain(natural_valley, self.valley_seed)
            print("terrain: patched to smooth rolling valley (no bowl, no pad disc)", flush=True)
        if self.dense_vegetation:
            forest_scene.relax_clearings(natural_valley)
            print("clearings: relaxed (pad-only exclusion instead of 80-90m + route clearing)",
                  flush=True)

        print(f"building natural_valley forest (seed={self.valley_seed}, assets={self.valley_assets_root})...",
              flush=True)
        self.valley_info = natural_valley.build_environment(stage, self.valley_seed, self.valley_assets_root)
        print(f"valley built: {self.valley_info.get('environment_version')}, "
              f"{self.valley_info.get('asset_count')} assets, "
              f"terrain {self.valley_info.get('terrain_extent_m')}m", flush=True)

        if self.dense_vegetation:
            print("adding dense vegetation layer...", flush=True)
            dense = forest_scene.add_dense_vegetation(
                stage, natural_valley, self.valley_seed, self.valley_assets_root)
            self.valley_info.update(dense)
            print(f"dense vegetation added: {dense}", flush=True)

        # X500 dataset fixes (2026-10-10): the lookout tower floated 2.1-2.4 m above the ground and the
        # "waterfall" was a turquoise box hovering over a flat river -- see forest_scene.rebuild_lookout()
        # and forest_scene.build_waterfall().
        info = forest_scene.rebuild_lookout(stage, natural_valley, self.valley_seed)
        self.valley_info.update(info)
        # river v2 (2026-10-11): channel, flat water meeting the banks, pebble shores; includes the waterfall
        info = forest_scene.build_river_v2(stage, natural_valley, self.valley_seed, self.valley_assets_root)
        self.valley_info.update(info)
        print(f"lookout rebuilt on the ground, waterfall built: {info}", flush=True)

        if self.rebuild_bridge:
            # must run AFTER build_environment -- that is what creates the original floating one
            info = forest_scene.rebuild_footbridge(stage, natural_valley, self.valley_seed)
            self.valley_info.update(info)
            print(f"footbridge rebuilt on the ground (deck top z={info['bridge_deck_top_z']:.2f}, "
                  f"ground z={info['bridge_ground_z']:.2f})", flush=True)
        if self.extra_build is not None:
            info = self.extra_build(stage)
            self.valley_info["extra_build"] = info
            print(f"extra build: {({k: v for k, v in info.items() if k != 'rigid_bodies'})}", flush=True)

    def populate_scene(self, scene_objects):
        """No-op: the forest IS the scene. Kept so callers written against HQBridge's interface
        (run_mission.py) don't need a special case."""
        return

    def terrain_height(self, x, y):
        return self.natural_valley.terrain_height(x, y, self.valley_seed)
