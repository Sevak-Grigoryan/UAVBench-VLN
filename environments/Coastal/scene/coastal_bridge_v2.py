"""Easy Coastal (2026-10-13): HQBridge that builds coastal_v2.build(plan) -- a different coast per simulation --
and, for the render jobs, also captures depth and per-object instance segmentation from the same camera
(Easy-data review item #4).

Depth: distance_to_image_plane (metres along the optical axis). Segmentation: instance_id_segmentation; every
instance's prim path is mapped to the scene object that owns it (coastal_v2.object_registry: 1-based ids, 0 = sky),
so a pixel's value says which ship / person / building / landmark it shows.
"""
import sys

import numpy as np

import hq_bridge

VLN_SIMULATION_PATH = "/mnt/weka/sgrigoryan/dataset_codebase/common/vln_simulation"
ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-coastal-v2"
FOREST_ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-v2"


class CoastalBridgeV2(hq_bridge.HQBridge):
    def __init__(self, run_dir, plan, assets_root=ASSETS_ROOT, forest_assets_root=FOREST_ASSETS_ROOT):
        self.plan = plan
        self.assets_root_v2 = str(assets_root)
        self.forest_assets_root = str(forest_assets_root)
        self.coastal_info = None
        self._aux = None
        super().__init__(run_dir)

    def _setup_world(self):
        UsdGeom, stage = self.UsdGeom, self.stage
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        if VLN_SIMULATION_PATH not in sys.path:
            sys.path.insert(0, VLN_SIMULATION_PATH)
        import natural_valley
        import coastal_v2

        self.natural_valley = natural_valley
        print(f"building coastal v2 #{self.plan['index']} (attempt {self.plan['attempt']})...", flush=True)
        self.coastal_info = coastal_v2.build(stage, natural_valley, self.plan, self.assets_root_v2, self.forest_assets_root)
        # semantic class labels on every object root (segmentation ids come from the registry below)
        from isaacsim.core.utils.semantics import add_labels

        self.registry = coastal_v2.object_registry(self.plan)
        n = 0
        for name, paths, cls, _ in self.registry:
            for p in paths:
                prim = stage.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    add_labels(prim, [cls], "class")
                    n += 1
        print(f"coastal v2 built: { {k: v for k, v in self.coastal_info.items() if k not in ('rigid_bodies', 'registry')} } "
              f"labelled {n} prims", flush=True)

    def populate_scene(self, scene_objects):
        return

    # ---- depth + segmentation (render jobs only) ------------------------------------------------------------
    def enable_aux(self):
        rep = self.rep
        self._depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self._inst = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
        self._depth.attach(self.render_product)
        self._inst.attach(self.render_product)
        for _ in range(4):
            self.app.update()
        # the first capture after attaching new annotators returns an empty rgb buffer (probe 330506): warm up until a
        # full colour frame comes back, so render_slice never writes an empty frame
        for _ in range(12):
            fr = self.process_camera_data()
            if fr.ndim == 3 and fr.shape[2] >= 3:
                break
        # longest-prefix lookup: prim path -> object id
        self._prefixes = sorted(((p, i + 1) for i, (_, paths, _, _) in enumerate(self.registry) for p in paths),
                                key=lambda t: -len(t[0]))
        self._id_cache = {}

    def _obj_of_path(self, path):
        for p, oid in self._prefixes:
            if path == p or path.startswith(p + "/"):
                return oid
        return 0

    def get_aux(self):
        """-> (depth float32 HxW metres (inf = sky), object-id uint16 HxW). Call after process_camera_data()."""
        d = self._depth.get_data()
        depth = np.asarray(d["data"] if isinstance(d, dict) else d, dtype=np.float32)
        s = self._inst.get_data()
        ids = np.asarray(s["data"]).astype(np.uint32)
        info = s.get("info", {}) if isinstance(s, dict) else {}
        id_to = info.get("idToLabels", {}) or {}
        lut = {}
        for k, v in id_to.items():
            k = int(k)
            if k not in self._id_cache:
                path = v if isinstance(v, str) else (v.get("class") if isinstance(v, dict) else str(v))
                self._id_cache[k] = self._obj_of_path(str(path))
            lut[k] = self._id_cache[k]
        if ids.size and lut:
            mx = int(ids.max())
            table = np.zeros(mx + 1, dtype=np.uint16)
            for k, v in lut.items():
                if k <= mx:
                    table[k] = v
            obj = table[ids]
        else:
            obj = np.zeros(ids.shape, dtype=np.uint16)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return depth, obj
