"""
Isaac Sim 6.2 compatibility layer for the 5.1-era drone stack (Pegasus Simulator + dataset x500.py).

Isaac Sim 6.x removed two things that stack needs:
  * omni.isaac.dynamic_control -- Pegasus (vehicle.py, multirotor.py, also its dev_6.0.1 branch) and the dataset X500 read the
    PhysX body pose / velocities and apply the rotor forces through it;
  * the omni.isaac.core extension -- listed as a dependency in Pegasus' extension.toml, so the extension can no longer be enabled.

Instead of editing Pegasus or the drone code (the flight physics code must stay identical to the 5.1 runs, so that only the
Isaac Sim version differs), this module
  1. installs a stand-in `omni.isaac.dynamic_control._dynamic_control` module that provides exactly the calls the stack uses,
     backed by the PhysX tensor API (omni.physics.tensors, numpy frontend), which reads/writes the live PhysX state;
  2. imports Pegasus' python package directly from the dataset's Pegasus checkout (same commit as the dev_6.0.1 branch, with the
     dataset's local PX4-backend fix), without enabling its UI extension.

Semantics kept from dynamic_control:
  get_rigid_body_pose(rb)                    -> .p (x, y, z), .r (qx, qy, qz, qw), world frame
  get_rigid_body_linear/angular_velocity(rb) -> world frame
  apply_body_force(rb, f, pos, global)       -> global=False: force and position in the body frame
  apply_body_torque(rb, tau, global)         -> global=False: torque in the body frame
  set_dof_velocity(dof, v)                   -> sets the joint VELOCITY STATE (rotor spin)
  get_relative_body_poses(rb, others)        -> poses of `others` in rb's frame
An invalid handle is returned as 0 (falsy) while the physics view is not live yet, as dynamic_control did.
"""
import sys
import types
from pathlib import Path

import numpy as np

PEGASUS_EXT = Path("/mnt/weka/sgrigoryan/dataset_codebase/common/px4_env/PegasusSimulator/extensions/pegasus.simulator")
_DEPS = ("omni.physics.tensors", "omni.physx.tensors", "isaacsim.core.utils", "isaacsim.core.api", "isaacsim.core.prims",
         "isaacsim.storage.native")


class _Transform:
    __slots__ = ("p", "r")

    def __init__(self, p, r):
        self.p, self.r = p, r


def _quat_rotate(q, v):
    """Rotate v by unit quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    u = np.array([x, y, z])
    return 2.0 * np.dot(u, v) * u + (w * w - np.dot(u, u)) * np.asarray(v) + 2.0 * w * np.cross(u, v)


def _quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


class _Body:
    """Rigid-body handle. The PhysX tensor view is created lazily: at the first physics step after world.reset() the
    simulation view is often not attached to the stage yet ("Failed to get a valid attached USD stage id"), and in
    Isaac Sim 6.x a physics callback that raises once is dropped for good -- so every call must succeed (no-op /
    neutral value) until the view exists, as dynamic_control's own calls did."""

    def __init__(self, path):
        self.path, self.view = path, None


class _Art:
    def __init__(self, path):
        self.path, self.view = path, None


class _Dof:
    def __init__(self, art, name):
        self.art, self.name, self.index = art, name, None


_IDENTITY = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))


class DynamicControlCompat:
    """The subset of the dynamic_control interface used by Pegasus and the dataset X500, on omni.physics.tensors."""

    def __init__(self):
        self._sim = None
        self._bodies = {}
        self._arts = {}
        self.calls = {"apply_body_force": 0, "apply_body_torque": 0, "view_errors": 0, "not_ready": 0}

    # ---- views -----------------------------------------------------------------------------------------------------
    def _sim_view(self):
        if self._sim is None or not self._sim.is_valid:
            import omni.physics.tensors as tensors

            self._sim = None
            import omni.usd

            # 6.2: the default stage_id=-1 means "the stage PhysX is attached to", and that lookup fails here ("Failed to
            # get a valid attached USD stage id", smoke 381210: every view failed) -> pass the open stage's id
            sid = omni.usd.get_context().get_stage_id()
            sim = tensors.create_simulation_view("numpy", stage_id=sid)
            if sim is None or not sim.is_valid:
                return None
            sim.set_subspace_roots("/")
            self._sim = sim
            for h in list(self._bodies.values()) + list(self._arts.values()):
                h.view = None
        return self._sim

    def _view(self, h, art=False):
        """The live tensor view of a handle, or None while PhysX is not attached yet (never raises)."""
        try:
            if h.view is not None and self._sim is not None and self._sim.is_valid:
                return h.view
            sim = self._sim_view()
            if sim is None:
                self.calls["not_ready"] += 1
                return None
            v = sim.create_articulation_view(h.path) if art else sim.create_rigid_body_view(h.path)
            if v is None or v.count != 1:
                self.calls["not_ready"] += 1
                return None
            h.view = v
            return v
        except Exception as e:
            self.calls["view_errors"] += 1
            if self.calls["view_errors"] <= 3:
                print(f"[isaac62_compat] view error ({h.path}): {type(e).__name__}: {e}", flush=True)
            self._sim = None
            return None

    @staticmethod
    def _prim_exists(path):
        import omni.usd

        st = omni.usd.get_context().get_stage()
        return st is not None and st.GetPrimAtPath(path).IsValid()

    def get_rigid_body(self, path):
        if path not in self._bodies:
            if not self._prim_exists(path):
                return 0
            self._bodies[path] = _Body(path)
        return self._bodies[path]

    def get_articulation(self, path):
        if path not in self._arts:
            if not self._prim_exists(path):
                return 0
            self._arts[path] = _Art(path)
        return self._arts[path]

    # ---- rigid bodies ----------------------------------------------------------------------------------------------
    def get_rigid_body_pose(self, rb):
        v = self._view(rb) if rb else None
        if v is None:
            return _Transform(*_IDENTITY)
        t = v.get_transforms()[0]
        return _Transform((float(t[0]), float(t[1]), float(t[2])), (float(t[3]), float(t[4]), float(t[5]), float(t[6])))

    def get_rigid_body_linear_velocity(self, rb):
        v = self._view(rb) if rb else None
        if v is None:
            return (0.0, 0.0, 0.0)
        w = v.get_velocities()[0]
        return (float(w[0]), float(w[1]), float(w[2]))

    def get_rigid_body_angular_velocity(self, rb):
        v = self._view(rb) if rb else None
        if v is None:
            return (0.0, 0.0, 0.0)
        w = v.get_velocities()[0]
        return (float(w[3]), float(w[4]), float(w[5]))

    def apply_body_force(self, rb, force, pos, global_coordinates=False):
        v = self._view(rb) if rb else None
        if v is None:
            return
        f = np.array([[force[0], force[1], force[2]]], dtype=np.float32)
        p = np.array([[pos[0], pos[1], pos[2]]], dtype=np.float32)
        v.apply_forces_and_torques_at_position(f, None, p, np.array([0], dtype=np.uint32), bool(global_coordinates))
        self.calls["apply_body_force"] += 1

    def apply_body_torque(self, rb, torque, global_coordinates=False):
        v = self._view(rb) if rb else None
        if v is None:
            return
        t = np.array([[torque[0], torque[1], torque[2]]], dtype=np.float32)
        v.apply_forces_and_torques_at_position(None, t, None, np.array([0], dtype=np.uint32), bool(global_coordinates))
        self.calls["apply_body_torque"] += 1

    def get_relative_body_poses(self, rb, others):
        v0 = self._view(rb) if rb else None
        if v0 is None:
            return [_Transform(*_IDENTITY) for _ in others]
        t0 = v0.get_transforms()[0]
        p0, q0 = np.array(t0[:3], dtype=float), np.array(t0[3:], dtype=float)
        out = []
        for o in others:
            v = self._view(o) if o else None
            if v is None:
                out.append(_Transform(*_IDENTITY))
                continue
            t = v.get_transforms()[0]
            p, q = np.array(t[:3], dtype=float), np.array(t[3:], dtype=float)
            pr = _quat_rotate(_quat_conj(q0), p - p0)
            qr = _quat_mul(_quat_conj(q0), q)
            out.append(_Transform(tuple(float(x) for x in pr), tuple(float(x) for x in qr)))
        return out

    # ---- articulation dofs (rotor spin) ----------------------------------------------------------------------------
    def find_articulation_dof(self, art, name):
        return _Dof(art, name) if art else 0

    def set_dof_velocity(self, dof, value):
        if not dof:
            return
        v = self._view(dof.art, art=True)
        if v is None:
            return
        if dof.index is None:
            names = list(v.shared_metatype.dof_names)
            if dof.name not in names:
                return
            dof.index = names.index(dof.name)
        d = np.array(v.get_dof_velocities(), dtype=np.float32).reshape(1, -1)
        d[0, dof.index] = float(value)
        v.set_dof_velocities(d, np.array([0], dtype=np.uint32))


_INSTANCE = None


def acquire_dynamic_control_interface():
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = DynamicControlCompat()
    return _INSTANCE


def install():
    """Register the stand-in modules and the Pegasus package. Call after SimulationApp started, before importing Pegasus."""
    import omni.kit.app

    ext = omni.kit.app.get_app().get_extension_manager()
    for e in _DEPS:
        ext.set_extension_enabled_immediate(e, True)
    for name in ("omni.isaac", "omni.isaac.dynamic_control"):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []
            sys.modules[name] = m
    dcm = types.ModuleType("omni.isaac.dynamic_control._dynamic_control")
    dcm.acquire_dynamic_control_interface = acquire_dynamic_control_interface
    sys.modules["omni.isaac.dynamic_control._dynamic_control"] = dcm
    sys.modules["omni.isaac.dynamic_control"]._dynamic_control = dcm
    # Pegasus' python package without its UI extension (pegasus/simulator/__init__.py imports the UI window)
    root = PEGASUS_EXT / "pegasus"
    for name, path in (("pegasus", root), ("pegasus.simulator", root / "simulator")):
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        m.__file__ = str(path / "__init__.py")
        sys.modules[name] = m
    sys.modules["pegasus"].simulator = sys.modules["pegasus.simulator"]
    return {"dynamic_control": "omni.physics.tensors compatibility layer (isaac62_compat.py)",
            "pegasus": str(PEGASUS_EXT), "enabled_extensions": list(_DEPS)}


def revive_physics_callbacks(world, log=print):
    """Isaac Sim 6.2: physics-step callbacks subscribed before world.reset() / the first steps (Pegasus' vehicle state,
    rotor update and sensor callbacks are added in the vehicle constructor) stop being called after the first steps
    while PhysX attaches its simulation to the stage (smoke job 381207: 2 calls, then none, no exception). Subscribing
    them again once the simulation runs keeps them alive. Call after the first world.step()s."""
    funcs = dict(getattr(world, "_physics_functions", {}) or {})
    for name, fn in funcs.items():
        world.remove_physics_callback(name)
        world.add_physics_callback(name, fn)
    log(f"[isaac62_compat] re-subscribed {len(funcs)} physics callbacks: {sorted(funcs)}")
    return sorted(funcs)
