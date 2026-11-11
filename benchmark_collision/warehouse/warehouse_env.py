"""
Warehouse environment for the collision tests: Isaac Sim's own warehouse template
(Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd, streamed from NVIDIA's public asset server) opened as the
ROOT stage, the same way ~/Desktop/warehouse_Simulation_0_0_8 did it (warehouse_bridge.py / warehouse_scene.py), but without the
walking people. Rendering settings and the ego camera come unchanged from the dataset's HQBridge (the Forest one), so the
renders match the other environments.

    env = warehouse_env.boot(kit_dir)   -> SimpleNamespace(bridge, ground, missions, landmarks, info, intrinsics, ...)

The flight runner adds PhysX colliders to the warehouse geometry (instanced shelving is de-instanced first: PhysX cannot
take a CollisionAPI on instance proxies), see make_solid().
"""
import sys
from pathlib import Path
from types import SimpleNamespace

CB = Path("/mnt/weka/sgrigoryan/dataset_codebase")
WAREHOUSE_REL = "Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd"
LAYOUT = Path(__file__).resolve().parent / "warehouse_layout.json"     # written by probe_warehouse.py (GPU probe)


def _bridge_class():
    sys.path[:0] = [str(CB / "common/pipeline"), str(CB / "Forest/scene")]
    import hq_bridge

    class WarehouseBridge(hq_bridge.HQBridge):
        def _setup_world(self):
            import omni.usd

            full = f"{self.assets_root.rstrip('/')}/{WAREHOUSE_REL}"
            ctx = omni.usd.get_context()
            ok = ctx.open_stage(full)
            print(f"[warehouse] open_stage({full}) -> {ok}", flush=True)
            for _ in range(60):
                self.app.update()
            self.stage = ctx.get_stage()
            root = self.stage.GetRootLayer()
            print(f"[warehouse] root layer {root.identifier} editable={root.permissionToEdit}", flush=True)
            # the template is opened straight from the asset server: its root layer can be read-only, and then the drone,
            # /World and the colliders are never authored. Everything we add goes to the (in-memory) session layer.
            self.stage.SetEditTarget(self.stage.GetSessionLayer())
            if self.UsdGeom.GetStageMetersPerUnit(self.stage) != 1.0:
                self.UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
            if not self.stage.GetPrimAtPath("/World").IsValid():
                self.UsdGeom.Xform.Define(self.stage, "/World")
            print(f"[warehouse] edit target = session layer; /World valid={self.stage.GetPrimAtPath('/World').IsValid()}", flush=True)
            self.warehouse_url = full

        def populate_scene(self, scene_objects):
            return

    return hq_bridge, WarehouseBridge


def make_solid(stage, log=print):
    """De-instance the warehouse template and give every visible mesh a PhysX collider (triangle mesh, static).
    Returns counts for the run's meta / coverage record."""
    from pxr import Usd, UsdGeom, UsdPhysics

    n_inst = 0
    for _ in range(4):                                   # nested instancing: repeat until no instance is left
        inst = [p for p in stage.Traverse() if p.IsInstanceable()]
        if not inst:
            break
        for p in inst:
            p.SetInstanceable(False)
        n_inst += len(inst)
    n_col, n_had = 0, 0
    for prim in stage.Traverse():
        p = str(prim.GetPath())
        if p.startswith(("/World/x500", "/World/EgoCamera", "/World/TestFixtures", "/World/DebugX500")):
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            n_had += 1
            continue
        UsdPhysics.CollisionAPI.Apply(prim)
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
        n_col += 1
    log(f"[warehouse] de-instanced {n_inst} prims; colliders added {n_col}, already present {n_had}")
    return {"deinstanced": n_inst, "colliders_added": n_col, "colliders_present": n_had}


def boot(kit_dir):
    import json

    hq_bridge, WarehouseBridge = _bridge_class()
    hq_bridge.FRAME_SIZE = (1280, 720)
    sys.path.insert(0, str(CB / "common/pipeline"))
    import envs

    bridge = WarehouseBridge(kit_dir)
    intrinsics = envs.setup_camera(bridge)
    lay = json.loads(LAYOUT.read_text()) if LAYOUT.exists() else {}
    floor_z = lay.get("floor_z", 0.0)
    info = {"name": "Warehouse", "description": "Isaac Sim warehouse template (warehouse_multiple_shelves.usd): racking, "
            "pallets, boxes; no people", "asset": getattr(bridge, "warehouse_url", WAREHOUSE_REL),
            "layout": {k: v for k, v in lay.items() if k in ("floor_z", "ceiling_z", "bbox_min", "bbox_max")}}
    return SimpleNamespace(bridge=bridge, ground=lambda x, y: floor_z, missions_mod=None, missions={}, landmarks={},
                           info=info, hq_bridge=hq_bridge, intrinsics=intrinsics, layout=lay, rigid_bodies=[])
