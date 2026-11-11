"""
Collision coverage for collision-aware evaluation (flight job only; the render jobs never build these, so videos are unchanged).

  add_rotor_discs()     one static disc per rotor on the airframe body: the dataset X500 rotor collider is the blade itself, a
                        0.28 x 0.017 x 0.0008 m box spinning with the rotor, so an obstacle inside the swept disc is touched only
                        when the blade happens to point at it. The disc (radius = blade half-length, 2 cm thick) covers the disc.
                        It sits on the body, whose mass, inertia and centre of mass are authored explicitly, so it changes no
                        dynamics. Body-rotor pairs are jointed, so PhysX never pairs a disc with its own blade.
  add_vegetation_proxies()  Forest trees, rocks and fallen wood are PointInstancer scatter WITHOUT colliders in the dataset runner
                        (prototype colliders would be instanced at every tree and shoved props). Each instance gets a simple
                        static proxy built from its prototype's bounding box: conifers -> cone, broadleaf -> trunk cylinder +
                        canopy cylinder, rocks / stumps / logs -> oriented box. Proxies are invisible and in a collision group
                        that does not collide with the movable props.
  enable_ccd()          scene + drone bodies: PhysX continuous collision detection (swept tests for fast bodies).
  audit()               what is and is not collidable, for collision.json.
Grass, ferns and shrubs (Groundcover) stay non-colliding: documented as unsupported foliage (< ~2 m tall).
"""
import math

ROTOR_DISC_RADIUS_M = 0.2792307692307692 / 2      # build_x500_usd.ROTOR_COLLISION_SIZE[0] / 2 (blade tip radius)
ROTOR_DISC_HEIGHT_M = 0.02
ROTOR_XY = [(0.174, -0.174), (-0.174, 0.174), (0.174, 0.174), (-0.174, -0.174)]   # build_x500_usd.ROTOR_XY
ROTOR_Z = 0.06

PROXY_ROOT = "/World/CollisionProxies"
GROUP_ROOT = "/World/CollisionGroups"
VEG_ROOTS = ("/World/Nature", "/World/DenseNature")
# instancer name -> proxy kind (by prototype index where it matters)
KIND = {"Trees": "conifer", "DistantTrees": "conifer", "MatureTrees": "broadleaf", "Rocks": "box", "Debris": "box"}
UNSUPPORTED = {"Groundcover": "grass / fern / shrub scatter: no collider (foliage below ~2 m; the 2 m minimum-height rule applies)"}


def add_rotor_discs(stage, root):
    from pxr import Gf, UsdGeom, UsdPhysics

    out = []
    for i, ((x, y), z) in enumerate(zip(ROTOR_XY, [ROTOR_Z] * 4)):
        p = f"{root}/body/collision_proxies/rotor_disc_{i}"
        c = UsdGeom.Cylinder.Define(stage, p)
        c.CreateRadiusAttr(ROTOR_DISC_RADIUS_M)
        c.CreateHeightAttr(ROTOR_DISC_HEIGHT_M)
        c.CreateAxisAttr("Z")
        c.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        c.CreateVisibilityAttr("invisible")
        UsdPhysics.CollisionAPI.Apply(c.GetPrim())
        out.append(p)
    return out


def enable_ccd(stage, drone_root):
    from pxr import PhysxSchema, Usd, UsdPhysics

    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    for s in scenes:
        PhysxSchema.PhysxSceneAPI.Apply(s).CreateEnableCCDAttr().Set(True)
    bodies = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(drone_root)):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr().Set(True)
            bodies.append(str(prim.GetPath()))
    return {"scenes": [str(s.GetPath()) for s in scenes], "bodies": bodies}


def _proto_bounds(stage, inst):
    """prototype index -> (min xyz, max xyz) in the instancer's space at scale 1 (includes the prototype's own scale op)."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    out = []
    for t in inst.GetPrototypesRel().GetTargets():
        r = cache.ComputeLocalBound(stage.GetPrimAtPath(t)).ComputeAlignedRange()
        out.append((tuple(r.GetMin()), tuple(r.GetMax())))
    return out


def add_vegetation_proxies(stage, log=print):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    UsdGeom.Scope.Define(stage, PROXY_ROOT)
    made, skipped, shapes = {}, {}, []
    n = 0
    instancers = []
    for root in VEG_ROOTS:
        rp = stage.GetPrimAtPath(root)
        if rp:
            instancers += [p for p in Usd.PrimRange(rp) if p.IsA(UsdGeom.PointInstancer)]
    for prim in instancers:
        path = str(prim.GetPath())
        seg = path.split("/")[3] if path.count("/") >= 3 else ""
        group = seg if seg in KIND or seg in UNSUPPORTED else None
        key = path.split("/")[2] + "/" + (group or prim.GetName())
        inst = UsdGeom.PointInstancer(prim)
        pos, ori, sc, idx = (inst.GetPositionsAttr().Get() or [], inst.GetOrientationsAttr().Get() or [],
                             inst.GetScalesAttr().Get() or [], inst.GetProtoIndicesAttr().Get() or [])
        if group is None or group in UNSUPPORTED:
            skipped[key] = skipped.get(key, 0) + len(pos)
            continue
        bounds = _proto_bounds(stage, inst)
        base = PROXY_ROOT + "/" + path[len("/World/"):].replace("/", "_")
        UsdGeom.Scope.Define(stage, base)
        for i in range(len(pos)):
            (x0, y0, z0), (x1, y1, z1) = bounds[int(idx[i])]
            s = float(sc[i][0]) if len(sc) else 1.0
            q = ori[i] if len(ori) else None
            yaw = 2.0 * math.atan2(float(q.GetImaginary()[2]), float(q.GetReal())) if q is not None else 0.0
            px, py, pz = float(pos[i][0]), float(pos[i][1]), float(pos[i][2])
            ex, ey, h = (x1 - x0) * s, (y1 - y0) * s, (z1 - z0) * s
            # bound centre offset, rotated by the instance yaw
            ox, oy = 0.5 * (x0 + x1) * s, 0.5 * (y0 + y1) * s
            cx = px + ox * math.cos(yaw) - oy * math.sin(yaw)
            cy = py + ox * math.sin(yaw) + oy * math.cos(yaw)
            zb = pz + z0 * s
            kind = KIND[group]
            parts = []
            if kind == "conifer":
                r = 0.25 * (ex + ey)                          # mean half-extent: branch tips, not the bounding-box corners
                parts.append(("cone", (cx, cy, zb + 0.5 * h), r, h))
            elif kind == "broadleaf":
                r = 0.25 * (ex + ey)
                trunk_r = max(0.15, 0.05 * r)
                parts.append(("cylinder", (px, py, zb + 0.225 * h), trunk_r, 0.45 * h))
                parts.append(("cylinder", (cx, cy, zb + 0.675 * h), 0.9 * r, 0.65 * h))
            else:
                parts.append(("box", (cx, cy, zb + 0.5 * h), (ex, ey, h), yaw))
            for j, part in enumerate(parts):
                p = f"{base}/i{i}_{j}"
                if part[0] == "box":
                    g = UsdGeom.Cube.Define(stage, p)
                    g.CreateSizeAttr(1.0)
                    g.AddTranslateOp().Set(Gf.Vec3d(*part[1]))
                    g.AddRotateZOp().Set(math.degrees(part[3]))
                    g.AddScaleOp().Set(Gf.Vec3f(*[max(v, 0.05) for v in part[2]]))
                else:
                    g = (UsdGeom.Cone if part[0] == "cone" else UsdGeom.Cylinder).Define(stage, p)
                    g.CreateRadiusAttr(max(part[2], 0.05))
                    g.CreateHeightAttr(max(part[3], 0.1))
                    g.CreateAxisAttr("Z")
                    g.AddTranslateOp().Set(Gf.Vec3d(*part[1]))
                g.CreateVisibilityAttr("invisible")
                UsdPhysics.CollisionAPI.Apply(g.GetPrim())
                n += 1
            shapes.append({"instancer": path, "instance": i, "kind": kind, "xy": [round(px, 2), round(py, 2)],
                           "base_z": round(zb, 2), "height_m": round(h, 2), "radius_m": round(0.25 * (ex + ey), 2)})
            made[key] = made.get(key, 0) + 1
    # proxies never touch the movable props (a conservative canopy over a paddock fence would shove it)
    if stage.GetPrimAtPath("/World/Props"):
        UsdGeom.Scope.Define(stage, GROUP_ROOT)
        gp = UsdPhysics.CollisionGroup.Define(stage, f"{GROUP_ROOT}/props")
        gp.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/Props")
        gv = UsdPhysics.CollisionGroup.Define(stage, f"{GROUP_ROOT}/vegetation_proxies")
        gv.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget(PROXY_ROOT)
        gv.CreateFilteredGroupsRel().AddTarget(f"{GROUP_ROOT}/props")
    log(f"[proxies] vegetation proxies: {n} shapes for {sum(made.values())} instances {made}; not collidable: {skipped}")
    return {"instances_by_instancer": made, "shapes": n, "not_collidable": skipped, "instances": shapes}


def audit(stage, excluded_prefixes):
    """Count visible geometry under each subtree the runner leaves without colliders, and what covers it now."""
    from pxr import Usd, UsdGeom, UsdPhysics

    rows = {}
    for prim in Usd.PrimRange(stage.GetPrimAtPath("/World")):
        p = str(prim.GetPath())
        if p.startswith(PROXY_ROOT) or "/x500" in p:
            continue
        if prim.IsA(UsdGeom.PointInstancer):
            k = "/".join(p.split("/")[:4])
            r = rows.setdefault(k, {"kind": "PointInstancer", "count": 0, "with_collider": 0})
            r["count"] += len(UsdGeom.PointInstancer(prim).GetPositionsAttr().Get() or [])
            continue
        if not prim.IsA(UsdGeom.Gprim) or prim.IsInstanceProxy():
            continue
        top = next((e for e in excluded_prefixes if p.startswith(e)), None)
        if top is None:
            continue
        r = rows.setdefault(top, {"kind": "Gprim", "count": 0, "with_collider": 0})
        r["count"] += 1
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False:
            r["with_collider"] += 1
    return rows
