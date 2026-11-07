"""Easy Mountain (2026-10-20): HQBridge that builds mountain_build.build(plan) -- a different mountain landscape per
simulation -- and, for the render jobs, also captures per-object instance segmentation (used internally for target
visibility and occlusion-aware bounding boxes; NOT published: review item C11, the dataset keeps only the video,
telemetry.json, meta.json and action.json).
"""
import sys

import numpy as np

import hq_bridge

VLN_SIMULATION_PATH = "/mnt/weka/sgrigoryan/dataset_codebase/common/vln_simulation"
ASSETS_ROOT = "/mnt/weka/sgrigoryan/assets/polyhaven-mountain-v2"


class MountainBridgeV2(hq_bridge.HQBridge):
    def __init__(self, run_dir, plan, assets_root=ASSETS_ROOT):
        self.plan = plan
        self.assets_root_v2 = str(assets_root)
        self.mountain_info = None
        super().__init__(run_dir)

    def _setup_world(self):
        UsdGeom, stage = self.UsdGeom, self.stage
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        if VLN_SIMULATION_PATH not in sys.path:
            sys.path.insert(0, VLN_SIMULATION_PATH)
        import natural_valley
        import mountain_build
        import mountain_v2

        self.natural_valley = natural_valley
        print(f"building mountain v2 #{self.plan['index']} (attempt {self.plan['attempt']}, {self.plan['landscape']}, {self.plan['season']})...", flush=True)
        self.mountain_info = mountain_build.build(stage, natural_valley, self.plan, self.assets_root_v2)
        from isaacsim.core.utils.semantics import add_labels

        self.registry = mountain_v2.object_registry(self.plan)
        n = 0
        for name, paths, cls, _ in self.registry:
            for p in paths:
                prim = stage.GetPrimAtPath(p)
                if prim and prim.IsValid():
                    add_labels(prim, [cls], "class")
                    n += 1
        print(f"mountain v2 built: { {k: v for k, v in self.mountain_info.items() if k not in ('rigid_bodies',)} } labelled {n} prims", flush=True)

    def populate_scene(self, scene_objects):
        return

    # ---- segmentation (render jobs only; depth is not captured: nothing uses it) ----------------------------
    def enable_aux(self):
        rep = self.rep
        self._depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self._inst = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
        self._depth.attach(self.render_product)
        self._inst.attach(self.render_product)
        for _ in range(4):
            self.app.update()
        for _ in range(12):
            fr = self.process_camera_data()
            if fr.ndim == 3 and fr.shape[2] >= 3:
                break
        self._prefixes = sorted(((p, i + 1) for i, (_, paths, _, _) in enumerate(self.registry) for p in paths), key=lambda t: -len(t[0]))
        self._id_cache = {}

    def _obj_of_path(self, path):
        for p, oid in self._prefixes:
            if path == p or path.startswith(p + "/"):
                return oid
        return 0

    def get_aux(self):
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
