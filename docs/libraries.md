# Libraries and documentation — Simulation_0_1_0_X500 (Holybro X500 V2 + PX4 in Isaac Sim)

## Drone
| | Version | Official page | Documentation |
|---|---|---|---|
| Holybro X500 V2 (PX4 Development Kit) | V2 | https://holybro.com/products/x500-v2-kits | https://docs.px4.io/main/en/frames_multicopter/holybro_x500v2_pixhawk6c.html |
| Pixhawk 6C (kit flight controller) | — | https://holybro.com/products/pixhawk-6c | https://docs.px4.io/main/en/flight_controller/pixhawk6c.html |
| PX4 kit project page | — | https://px4.io/project/s500-v2-2-2/ | — |

## Simulation stack
| | Version | Official page | Documentation / source |
|---|---|---|---|
| NVIDIA Isaac Sim | 5.1.0 (already installed, env `behavior_h100`) | https://developer.nvidia.com/isaac/sim | https://docs.isaacsim.omniverse.nvidia.com/ |
| PX4 Autopilot (SITL firmware) | v1.14.3 | https://px4.io | https://docs.px4.io/v1.14/en/ · https://github.com/PX4/PX4-Autopilot/tree/v1.14.3 |
| PX4 SITL simulation overview | — | — | https://docs.px4.io/v1.14/en/simulation/ |
| PX4 X500 reference model (source of all X500 constants) | v1.14.3 | — | https://github.com/PX4/PX4-Autopilot/tree/v1.14.3/Tools/simulation/gz/models/x500 |
| Pegasus Simulator (Isaac Sim <-> PX4 bridge) | 5.1.0 | https://pegasussimulator.github.io/PegasusSimulator/ | https://github.com/PegasusSimulator/PegasusSimulator |
| MAVLink protocol | v2 | https://mavlink.io | https://mavlink.io/en/messages/common.html |
| pymavlink (already in `behavior_h100`) | 2.4.49 | https://pypi.org/project/pymavlink/ | https://github.com/ArduPilot/pymavlink |

## Python packages installed in `px4_build_env/venv` (PX4 build + model conversion)
| Package | Version | Official page | Why |
|---|---|---|---|
| empy | 3.3.4 | https://pypi.org/project/empy/3.3.4/ | PX4 1.14 code generator (empy 4.x breaks it) |
| CMake | 3.31 | https://cmake.org · https://cmake.org/documentation/ | PX4 1.14 does not configure with CMake 4.x |
| Ninja | latest | https://ninja-build.org | PX4 build backend |
| Jinja2 | latest | https://jinja.palletsprojects.com | PX4 code generation |
| PyYAML | latest | https://pyyaml.org | PX4 config parsing |
| kconfiglib | latest | https://github.com/ulfalizer/Kconfiglib | PX4 board config |
| pyros-genmsg | latest | https://pypi.org/project/pyros-genmsg/ | PX4 uORB message generation |
| packaging | latest | https://packaging.pypa.io | PX4 build scripts |
| jsonschema | latest | https://python-jsonschema.readthedocs.io | PX4 parameter validation |
| NumPy | latest | https://numpy.org/doc/ | maths |
| future | latest | https://python-future.org | PX4 build scripts |
| toml | latest | https://pypi.org/project/toml/ | PX4 build scripts |
| OpenUSD (usd-core) | latest | https://openusd.org · https://pypi.org/project/usd-core/ | writes x500.usd |
| trimesh | latest | https://trimesh.org | reads the X500 meshes |
| pycollada | latest | https://pycollada.readthedocs.io | reads the .dae meshes |
| SciPy | latest | https://scipy.org · https://docs.scipy.org/doc/scipy/ | mesh normals |
