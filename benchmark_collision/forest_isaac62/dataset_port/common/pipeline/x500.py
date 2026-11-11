"""
Holybro X500 V2 as a Pegasus Simulator vehicle, flown by real PX4 v1.14.3 SITL.

Pegasus ships only Iris; this mirrors pegasus/simulator/logic/vehicles/multirotors/iris.py with every
constant swapped for the X500's own, all taken from PX4's gz model (Tools/simulation/gz/models/x500/
model.sdf) and airframe (4001_gz_x500) so the vehicle behaves like PX4's reference X500:

  thrust per rotor   T = motorConstant * w^2            motorConstant   = 8.54858e-06
  yaw moment         Q = momentConstant * T             momentConstant  = 0.016
                       -> Pegasus' "rolling_moment_coefficient" = 0.016 * 8.54858e-06
  rotor speed        150 .. 1000 rad/s                  SIM_GZ_EC_MIN / maxRotVelocity
                       -> PX4 motor command u in [0,1] maps to w = 150 + 850 u
  spin direction     rotors 0,1 ccw / 2,3 cw            turningDirection
  body drag          rotorDragCoefficient * w_hover * 4 rotors ~= 0.25 N per m/s (x, y)

The PX4 side runs airframe 10050_isaac_x500 (added to ROMFS by this project): 4001_gz_x500's
geometry/motor layout, but over simulator_mavlink -- the transport Pegasus speaks -- instead of
the gz bridge.
"""
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.sensors import GPS, IMU, Barometer, Magnetometer
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig

PROJECT = Path(__file__).resolve().parents[1]
X500_USD = str(PROJECT / "assets/x500/x500.usd")
PX4_DIR = str(PROJECT / "px4_env/PX4-Autopilot")
PX4_AIRFRAME = "isaac_x500"  # -> ROMFS/px4fmu_common/init.d-posix/airframes/10050_isaac_x500

MOTOR_CONSTANT = 8.54858e-06
MOMENT_CONSTANT = 0.016
ROTOR_DRAG_COEFF = 8.06428e-05
W_MIN, W_MAX = 150.0, 1000.0
MASS_KG = 2.0 + 4 * 0.016076923076923075
SPAWN_Z = 0.227  # skid bottom sits 0.227 m below base_link (see build_x500_usd.py)

W_HOVER = math.sqrt(MASS_KG * 9.81 / (4 * MOTOR_CONSTANT))
LINEAR_DRAG_XY = 4 * ROTOR_DRAG_COEFF * W_HOVER

# Real kit hardware, for meta.json (Holybro X500 V2 PX4 Development Kit, px4.io/project/s500-v2-2-2).
KIT = {
    "name": "Holybro X500 V2",
    "flight_controller": "Pixhawk 6C (PX4 v1.14.3 SITL in simulation)",
    "motors": "Holybro 2216 KV920 x4",
    "esc": "BLHeli S 20A x4",
    "propellers": "1045",
    "frame": "X500 V2 carbon fibre, 500 mm motor-to-motor diagonal",
}


def thrust_config():
    return {
        "num_rotors": 4,
        "rotor_constant": [MOTOR_CONSTANT] * 4,
        "rolling_moment_coefficient": [MOMENT_CONSTANT * MOTOR_CONSTANT] * 4,
        "rot_dir": [-1, -1, 1, 1],
        "min_rotor_velocity": [0.0] * 4,
        "max_rotor_velocity": [W_MAX] * 4,
    }


def physics_summary():
    return {
        "mass_kg": round(MASS_KG, 4),
        "inertia_body_kgm2": [0.02166666666666667, 0.02166666666666667, 0.04],
        "motor_constant": MOTOR_CONSTANT,
        "moment_constant": MOMENT_CONSTANT,
        "rotor_speed_range_rad_s": [W_MIN, W_MAX],
        "hover_rotor_speed_rad_s": round(W_HOVER, 1),
        "thrust_to_weight": round(4 * MOTOR_CONSTANT * W_MAX**2 / (MASS_KG * 9.81), 3),
        "linear_drag_xy": round(LINEAR_DRAG_XY, 4),
        "rotor_positions_m": [[0.174, -0.174, 0.06], [-0.174, 0.174, 0.06], [0.174, 0.174, 0.06], [-0.174, -0.174, 0.06]],
        "source": "PX4-Autopilot v1.14.3 gz model x500 + airframe 4001_gz_x500",
    }


class X500Config(MultirotorConfig):
    def __init__(self, vehicle_id=0, px4_autolaunch=True, graphical_sensors=None):
        super().__init__()
        self.stage_prefix = "x500"
        self.usd_file = X500_USD
        self.thrust_curve = QuadraticThrustCurve(thrust_config())
        self.drag = LinearDrag([LINEAR_DRAG_XY, LINEAR_DRAG_XY, 0.0])
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]
        self.graphical_sensors = graphical_sensors or []
        self.backends = [
            PX4MavlinkBackend(
                PX4MavlinkBackendConfig(
                    {
                        "vehicle_id": vehicle_id,
                        "px4_autolaunch": px4_autolaunch,
                        "px4_dir": PX4_DIR,
                        "px4_vehicle_model": PX4_AIRFRAME,
                        "input_offset": [0.0] * 4,
                        "input_scaling": [W_MAX - W_MIN] * 4,
                        "zero_position_armed": [W_MIN] * 4,
                    }
                )
            )
        ]


class X500(Multirotor):
    def __init__(self, stage_prefix, vehicle_id=0, init_pos=(0.0, 0.0, SPAWN_Z), init_orientation=(0.0, 0.0, 0.0, 1.0), config=None):
        config = config or X500Config(vehicle_id)
        super().__init__(stage_prefix, config.usd_file, vehicle_id, list(init_pos), list(init_orientation), config=config)
        self._dc_quat_order = None

    def update_state(self, dt: float):
        """Pegasus' Vehicle.update_state takes POSITION from the live PhysX body but ATTITUDE from the USD
        xform. USD is only synced from physics on a rendering app update, so when physics is stepped with
        World.step(render=False) (as the dataset runner does between path-traced frames) the attitude stays
        frozen at spawn: the simulated magnetometer stops rotating, gyro/accel are rotated into the wrong
        body frame, and PX4 loses heading and eventually attitude (first forest run, job 318291).
        Here everything -- position, attitude, velocities -- comes from the live PhysX body."""
        dc = self.get_dc_interface()
        body = dc.get_rigid_body(self._stage_prefix + "/body")
        pose = dc.get_rigid_body_pose(body)
        r = [float(pose.r[0]), float(pose.r[1]), float(pose.r[2]), float(pose.r[3])]
        # Before the physics view is live (world.reset, first steps) dynamic_control hands back an
        # invalid body / all-zero pose; normalising that gives NaN, which then goes out to PX4 as
        # HIL_SENSOR and stalls the lockstep (smoke job 318385 hung for an hour). Use Pegasus' own
        # USD-based path until the PhysX pose is real.
        if not body or not np.all(np.isfinite(r)) or np.linalg.norm(r) < 0.5:
            return super().update_state(dt)
        if self._dc_quat_order is None:
            # decide once whether dynamic_control returns (x,y,z,w) or (w,x,y,z), against the USD pose,
            # which is still correct at this point (nothing has moved yet)
            from isaacsim.core.utils.xforms import get_world_pose
            _, q_wxyz = get_world_pose(self._stage_prefix + "/body")
            ref = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
            xyzw = np.array(r)
            wxyz_as_xyzw = np.array([r[1], r[2], r[3], r[0]])
            self._dc_quat_order = "xyzw" if abs(np.dot(xyzw, ref)) >= abs(np.dot(wxyz_as_xyzw, ref)) else "wxyz"
            print(f"[x500] dynamic_control quaternion order: {self._dc_quat_order} (physx r={np.round(r, 4).tolist()}, "
                  f"usd xyzw={np.round(ref, 4).tolist()}, p={[round(float(v), 3) for v in pose.p]})", flush=True)
        att = np.array(r) if self._dc_quat_order == "xyzw" else np.array([r[1], r[2], r[3], r[0]])
        att /= np.linalg.norm(att)

        ang_vel = np.array(dc.get_rigid_body_angular_velocity(body))
        linear_vel = np.array(dc.get_rigid_body_linear_velocity(body))
        rot = Rotation.from_quat(att)
        self._state.linear_acceleration = (linear_vel - self._state.linear_velocity) / dt
        self._state.position = np.array([float(pose.p[0]), float(pose.p[1]), float(pose.p[2])])
        self._state.attitude = att
        self._state.linear_velocity = linear_vel
        self._state.linear_body_velocity = rot.inv().apply(linear_vel)
        self._state.angular_velocity = rot.inv().apply(ang_vel)
