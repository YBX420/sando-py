# ros2_bridge — D435i Gazebo closed loop for the headless sando-py planner

Drives the headless **sando-py** planner (LBFGSpp MINCO, **no Gurobi**) from a Gazebo
physics sim with an **Intel RealSense D435i** depth camera as the only world sensor.

```
Gazebo world + obstacles
   │  D435i depth (gazebo realsense plugin, ~100 Hz)
   ▼
depth_to_occupancy.py ── /NX01/occupancy_grid (PointCloud2, map frame)
   ▼
sando_py_bridge.py  ──(ctypes → cpp/capi/sando_capi.so)──► sando-py planner
   │  /NX01/goal  (dynus_interfaces/Goal: p,v,a,yaw)
   ▼
fake_sim ── integrates → drone motion, pushes pose to Gazebo, republishes
   │  /NX01/state (pos,vel,quat) ──────────────────────────────┐
   └──────────────────── feedback to the bridge ───────────────┘
```

## Files
| file | role |
|---|---|
| `sando_py_bridge.py` | ROS2 node: occupancy_grid + term_goal + state → sando-py → goal |
| `depth_to_occupancy.py` | D435 depth cloud → map-frame `occupancy_grid` (+ empty `unknown_grid`) |
| `record_and_plot.py` | record a flight (odom + occupancy) → matplotlib PNG |
| `check_occ.py` | one-shot occupancy_grid centroid checker |
| `minimal_state.world` | minimal Gazebo world (sun+ground+`gazebo_ros_state`) that doesn't crash headless |
| `sando_live.sh` | live demo: sim + gzclient + RViz + goal ping-pong |
| `sando_viz.sh` | headless demo → renders the flight PNG |

## Build the planner lib (once, Linux)
```bash
cd cpp
g++ -O2 -shared -fPIC -std=c++17 -o capi/sando_capi.so capi/sando_capi.cpp \
    -Iinclude -Ithird_party/eigen -Ithird_party
```
(`sando_cpp_bridge.py` auto-loads `cpp/capi/sando_capi.so` on Linux, the Windows `.dll` otherwise.)

## Run (assumes the `sando_ws` ROS2 workspace is built; scripts source it themselves)
```bash
bash ros2_bridge/sando_live.sh      # live RViz + Gazebo
bash ros2_bridge/sando_viz.sh       # headless -> /tmp/sando_flight.png
```
The launchers are location-independent: they resolve the repo from their own path and
default the workspace to `~/code/sando_ws` (override: `SANDO_WS=/path/to/ws bash ...`).
Full laptop migration guide: `docs/UBUNTU22_PORT.md`.

## Notes / TODO (M3 — RGBD safety, not yet done)
- `unknown_grid` is published **empty** → no true partial-observability yet
  (frustum free-space carving needed; the planner's `find_safe_sub_goal` consumes it).
- occupancy is a per-frame snapshot (no temporal accumulation).
- `skip_initial_yawing=True` is set to go straight to TRAVELING; perception-aware
  yaw-to-look-ahead is part of the safety layer.
- the `*.sh` launchers default to `~/code/sando_ws`; set `SANDO_WS` to override.
