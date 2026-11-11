"""
Environment boot shared by the flight job (run_x500_mission.py) and the render jobs
(render_slice.py), so both build byte-for-byte the same scene: same code, same seed, same pads,
same camera intrinsics. Rendering a slice from recorded camera poses is only valid because of that.

    env = boot("Forest", kit_dir)   -> SimpleNamespace(bridge, ground, missions_mod, missions,
                                                        landmarks, info, hq_bridge)
"""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

COMMON = Path(__file__).resolve().parents[1]
CODEBASE = COMMON.parent
FRAME_SIZE = (1280, 720)
FOCAL_MM, H_APERTURE_MM = 24.0, 20.955
CAM_FWD, CAM_DOWN = 0.12, 0.07  # gimbal camera position relative to base_link (yaw frame)
ENV_NAMES = ("Forest", "City", "Coastal", "Mountain")


def boot(env_name, kit_dir, mission_id=None):
    """mission_id is needed for missions whose scene depends on the mission (Forest Easy v2: every
    simulation has its own props layout)."""
    env_dir = CODEBASE / env_name
    sys.path[:0] = [str(COMMON / "pipeline"), str(env_dir), str(env_dir / "scene"), str(COMMON / "vln_simulation")]
    import hq_bridge

    hq_bridge.FRAME_SIZE = FRAME_SIZE
    if env_name == "Forest":
        import forest_scene
        import missions as missions_mod

        missions_mod.apply_pads(forest_scene)
        import forest_bridge

        lay, extra, holder = None, None, {}
        if mission_id and mission_id.startswith("easy_v2_"):
            import easy_v2

            def extra(stage):
                # runs inside the bridge, after SimulationApp started (natural_valley needs pxr) and after
                # the terrain was patched -- the layout is computed on the final ground
                import natural_valley as _nv

                holder["lay"] = easy_v2.layout(easy_v2.index_of(mission_id), _nv, forest_scene)
                return easy_v2.build(stage, _nv, holder["lay"])
        bridge = forest_bridge.ForestBridge(kit_dir, extra_build=extra)
        lay = holder.get("lay")
        nv, seed = bridge.natural_valley, bridge.valley_seed
        ground = bridge.terrain_height
        missions, landmarks = missions_mod.missions(nv, seed, ground)
        if lay is not None:
            m, lm = easy_v2.mission(lay["index"], lay, nv, forest_scene, ground)
            missions[m["id"]] = m
            landmarks.update(lm)
        info = {"name": "Forest",
                "description": "authored river valley (natural_valley + Poly Haven CC0), timber footbridge",
                "scene_version": "forest-v2 (river channel + Easy v2 props)" if lay is not None else "forest-v1",
                "seed": seed, "valley_info": summarize(bridge.valley_info),
                "pads": {k: {"xy": v[0], "description": v[1], "ground_z": round(ground(*v[0]), 3)}
                         for k, v in missions_mod.PADS.items()}}
    elif env_name == "City":
        # Easy City dataset: the whole city (buildings, special blocks, street life) and the mission are a
        # pure function of the mission id -- see City/missions.py and City/scene/city_v2.py
        import city_v2
        import missions as missions_mod
        import pilot as _pilot

        idx = missions_mod.index_of(mission_id)
        pl, mission, landmarks = missions_mod.mission_and_plan(idx, city_v2, _pilot)
        import city_bridge

        bridge = city_bridge.CityBridgeV2(kit_dir, pl)
        feats = list(pl["features"])
        ground = lambda x, y: city_v2.surface_z(x, y, feats)
        missions = {mission["id"]: mission}
        info = {"name": "City", "description": "authored city (Poly Haven CC0 PBR facades and street furniture), "
                "per-simulation buildings, special blocks and street life",
                "scene_version": "city-v2 (Easy City dataset)", "city_index": idx, "plan_attempt": pl["attempt"],
                "build": {k: v for k, v in bridge.city_info.items() if k != "rigid_bodies"},
                "special_blocks": [{"name": f["name"], "kind": f["kind"], "center_xy": list(b), "description": f["words"]}
                                   for b, f in pl["features"].items()],
                "pads": {k: {"xy": v[0], "description": v[1]} for k, v in city_v2.PADS.items()}}
        intrinsics = setup_camera(bridge)
        rigid = bridge.city_info.get("rigid_bodies", [])

        def make_dynamic(stage, rb):
            from pxr import UsdPhysics
            for r in rb:
                prim = stage.GetPrimAtPath(r["path"])
                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(r["mass_kg"]))
            return len(rb)
        return SimpleNamespace(bridge=bridge, ground=ground, missions_mod=missions_mod, missions=missions,
                               landmarks=landmarks, info=info, hq_bridge=hq_bridge, intrinsics=intrinsics, layout=pl,
                               rigid_bodies=rigid, make_dynamic=make_dynamic)
    elif env_name == "Coastal":
        # Easy Coastal dataset (2026-10-13): coast, sites, ships, pads and mission are a pure function of the mission id
        # -- see Coastal/missions.py and Coastal/scene/coastal_v2.py
        import coastal_v2
        import missions as missions_mod
        import pilot as _pilot

        idx = missions_mod.index_of(mission_id)
        pl, mission, landmarks = missions_mod.mission_and_plan(idx, coastal_v2, _pilot)
        import coastal_bridge_v2

        bridge = coastal_bridge_v2.CoastalBridgeV2(kit_dir, pl)
        ground = lambda x, y: coastal_v2.surface_z(x, y, pl, water=True)  # noqa: E731
        missions = {mission["id"]: mission}
        lm = {k: dict(xy=v["xy"], top_z=v["top_z"], desc=v["words"], kind=v["kind"], box=v.get("box")) for k, v in landmarks.items()}
        info = {"name": "Coastal", "description": "authored coast (Poly Haven CC0 PBR + ships), per-simulation coastline, shore "
                "sites, ships at anchor, people and harbour life", "scene_version": "coastal-v2 (Easy Coastal dataset)",
                "coast_index": idx, "plan_attempt": pl["attempt"],
                "build": {k: v for k, v in bridge.coastal_info.items() if k not in ("rigid_bodies", "registry")},
                "sites": [{"name": s["name"], "kind": s["kind"], "center_xy": [round(v, 2) for v in s["xy"]], "description": s["words"]}
                          for s in pl["sites"]],
                "pads": {k: {"xy": [round(v, 2) for v in p["xy"]], "z": round(p["z"], 3), "description": p["words"],
                             "on_ship": p.get("ship")} for k, p in pl["pads_all"].items()}}
        intrinsics = setup_camera(bridge)
        rigid = bridge.coastal_info.get("rigid_bodies", [])

        def make_dynamic(stage, rb):
            from pxr import PhysxSchema, UsdPhysics
            frozen = idx in coastal_v2.FROZEN
            for r in rb:
                prim = stage.GetPrimAtPath(r["path"])
                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(r["mass_kg"]))
                if not frozen:
                    # tall standing people crept ~5 cm / 5 deg over a 4-min flight (Coastal v2.1 flight 001): more solver
                    # iterations, sleep once at rest, light damping
                    pa = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                    pa.CreateSolverPositionIterationCountAttr(32)
                    pa.CreateSolverVelocityIterationCountAttr(4)
                    # still a slow straight slide of 6-8 cm (docker1/8, flight 001 #2): strong damping (a 72 kg person
                    # still falls fine: terminal speed g/2 = 5 m/s) and sleep as soon as it is at rest
                    pa.CreateSleepThresholdAttr(0.05)
                    pa.CreateStabilizationThresholdAttr(0.01)
                    pa.CreateLinearDampingAttr(2.0)
                    pa.CreateAngularDampingAttr(2.0)
            return len(rb)
        return SimpleNamespace(bridge=bridge, ground=ground, missions_mod=missions_mod, missions=missions,
                               landmarks=lm, info=info, hq_bridge=hq_bridge, intrinsics=intrinsics, layout=pl,
                               rigid_bodies=rigid, make_dynamic=make_dynamic, aux=True)
    elif env_name == "Mountain":
        # Easy Mountain dataset (2026-10-20): landscape type, season, sky, sites, pads and the mission are a pure function of
        # the mission id -- see Mountain/missions.py and Mountain/scene/mountain_v2.py
        import missions as missions_mod
        import mountain_v2
        import pilot as _pilot

        idx = missions_mod.index_of(mission_id)
        pl, mission, landmarks = missions_mod.mission_and_plan(idx, mountain_v2, _pilot)
        import mountain_bridge_v2

        bridge = mountain_bridge_v2.MountainBridgeV2(kit_dir, pl)
        ground = lambda x, y: mountain_v2.surface_z(x, y, pl, water=True)  # noqa: E731
        missions = {mission["id"]: mission}
        lm = {k: dict(xy=v["xy"], top_z=v.get("top_z", 0.0), desc=v["words"], kind=v.get("kind"), box=v.get("box")) for k, v in landmarks.items()}
        info = {"name": "Mountain", "description": "authored mountain landscape (Poly Haven CC0 PBR, trees and skies), per-simulation "
                "landscape type, season, sky, valley, peaks, passes, lakes, sites, people, animals and vehicles",
                "scene_version": "mountain-v2 (Easy Mountain dataset)", "mountain_index": pl["index"], "plan_attempt": pl["attempt"],
                "landscape": pl["landscape"], "season": pl["season"], "snowline_above_valley_floor_m": round(pl["snowline"], 1),
                "treeline_above_valley_floor_m": round(pl["treeline"], 1),
                "build": {k: v for k, v in bridge.mountain_info.items() if k not in ("rigid_bodies",)},
                "sites": [{"name": s["name"], "kind": s["kind"], "center_xy": [round(v, 2) for v in s["xy"]], "description": s["words"]}
                          for s in pl["sites"]],
                "summits": [{"name": p["name"], "xy": [round(p["x"], 1), round(p["y"], 1)], "z": round(p["z"], 1), "description": p["words"]}
                            for p in pl["summits"]],
                "passes": [{"name": c["name"], "xy": [round(c["x"], 1), round(c["y"], 1)], "z": round(c["z"], 1), "description": c["words"]}
                           for c in pl["cols"]],
                "pads": {k: {"xy": [round(v, 2) for v in p["xy"]], "z": round(p["z"], 3), "description": p["words"],
                             "kind": "painted landing pad" if k.startswith("P") else f"unmarked spot ({p.get('spot')})"}
                         for k, p in pl["pads_all"].items()}}
        intrinsics = setup_camera(bridge)
        rigid = bridge.mountain_info.get("rigid_bodies", [])

        def make_dynamic(stage, rb):
            from pxr import PhysxSchema, UsdPhysics
            for r in rb:
                prim = stage.GetPrimAtPath(r["path"])
                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(float(r["mass_kg"]))
                pa = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)  # the Coastal v2.1 settings that kept every prop at rest
                pa.CreateSolverPositionIterationCountAttr(32)
                pa.CreateSolverVelocityIterationCountAttr(4)
                pa.CreateSleepThresholdAttr(0.05)
                pa.CreateStabilizationThresholdAttr(0.01)
                pa.CreateLinearDampingAttr(2.0)
                pa.CreateAngularDampingAttr(2.0)
            return len(rb)
        return SimpleNamespace(bridge=bridge, ground=ground, missions_mod=missions_mod, missions=missions,
                               landmarks=lm, info=info, hq_bridge=hq_bridge, intrinsics=intrinsics, layout=pl,
                               rigid_bodies=rigid, make_dynamic=make_dynamic, aux=True)
    else:
        raise ValueError(env_name)
    intrinsics = setup_camera(bridge)
    if lay is not None:
        import easy_v2

        info["props_layout"] = easy_v2.summary(lay)
        eb = bridge.valley_info.get("extra_build", {})
        info["vegetation_removed_for_props"] = eb.get("vegetation_removed")
    return SimpleNamespace(bridge=bridge, ground=ground, missions_mod=missions_mod, missions=missions,
                           landmarks=landmarks, info=info, hq_bridge=hq_bridge, intrinsics=intrinsics, layout=lay,
                           rigid_bodies=(bridge.valley_info.get("extra_build") or {}).get("rigid_bodies", []))


def summarize(info, max_len=400):
    """Keep the scene summary small: scalars and short values only (the full scene inventory is ~650 KB
    and belongs to the environment code, not to every sample's meta.json)."""
    import json as _json
    return {k: v for k, v in (info or {}).items() if len(_json.dumps(v, default=str)) <= max_len}


def setup_camera(bridge):
    from pxr import Gf, UsdGeom

    v_aperture = H_APERTURE_MM * FRAME_SIZE[1] / FRAME_SIZE[0]
    cam = UsdGeom.Camera(bridge.stage.GetPrimAtPath(bridge.camera_path))
    cam.CreateVerticalApertureAttr(v_aperture)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 10000.0))
    return {
        "model": "pinhole (USD camera)", "focal_length_mm": FOCAL_MM, "horizontal_aperture_mm": H_APERTURE_MM,
        "vertical_aperture_mm": round(v_aperture, 4), "clipping_range_m": [0.05, 10000.0],
        "fov_horizontal_deg": round(math.degrees(2 * math.atan(H_APERTURE_MM / (2 * FOCAL_MM))), 3),
        "fov_vertical_deg": round(math.degrees(2 * math.atan(v_aperture / (2 * FOCAL_MM))), 3),
        "fx_px": round(FOCAL_MM / H_APERTURE_MM * FRAME_SIZE[0], 3), "fy_px": round(FOCAL_MM / v_aperture * FRAME_SIZE[1], 3),
        "cx_px": FRAME_SIZE[0] / 2, "cy_px": FRAME_SIZE[1] / 2, "frame_size": list(FRAME_SIZE),
        "mount": {"forward_m": CAM_FWD, "down_m": CAM_DOWN, "yaw": "locked to airframe heading",
                  "roll/pitch": "stabilised gimbal; pitch commanded per mission stage, positive = looking down"},
    }


def camera_pose(pos, yaw, pitch_deg):
    cx = pos[0] + CAM_FWD * math.cos(yaw)
    cy = pos[1] + CAM_FWD * math.sin(yaw)
    cz = pos[2] - CAM_DOWN
    return cx, cy, cz, math.degrees(yaw), pitch_deg
