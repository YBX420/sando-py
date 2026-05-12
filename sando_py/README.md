# sando_py

Python re-implementation of the MIT-ACL SANDO trajectory planner that
drops into the **upstream ROS 2 simulation world unchanged** — RViz,
`dynamic_forest_node`, `fake_sim`, and `goal_sender` are still the C++
nodes from `~/code/sando_ws`. Only the `sando` planner Node is replaced.
The C++ project is the source of truth for algorithm details; this port
stays as faithful as practical to the C++ types, parameter names, and
per-tick flow.

## Layout

| subpackage | purpose                                                                          |
|------------|----------------------------------------------------------------------------------|
| `core/`    | shared dataclasses + pure-math utilities (no I/O, no algorithms)                 |
| `hgp/`     | global planner — heat-map A* over a voxel occupancy grid                         |
| `decomp/`  | convex decomposition — polyline path -> safety corridors                         |
| `solver/`  | local trajectory optimizer — smooth piecewise polynomial                         |
| `sim_io/`  | ROS adapter layer — `rclpy.Node` + msg converters (only place that imports ROS)  |

Each subpackage has its own `README.md` with the per-file plan and the
input / output contract.

## Architecture

```
       upstream C++ nodes (unchanged, run inside tmuxp world)
   +----------------------+
   | rviz2                | <---- viz markers, polytopes, traj
   | dynamic_forest_node  | ---+
   | fake_sim             | ---+
   | goal_sender          | ---+
   +----------------------+    |
                               |   /trajs (DynTraj)
                               |   state  (State)
                               |   term_goal (PoseStamped)
                               v
                       +-----------------------+
                       | sando_py.sim_io       |
                       |   SandoNode (rclpy)   |
                       |   msg_convert         |
                       +-----------+-----------+
                                   |
                                   | per-tick: call pure-Python pipeline
                                   v
                       +-----------+-----------+
                       | sando_py.core/        |
                       | sando_py.hgp/         |
                       | sando_py.decomp/      |
                       | sando_py.solver/      |
                       +-----------------------+
```

## Per-tick flow (`SandoNode.replanCallback`)

```
config/sando.yaml         /trajs subscription
    |                            |
    v                            v
Parameters             List[DynTraj]  (built incrementally as msgs arrive)
    \                          /
     \                        /
      \____________  ________/
                   \/
   1. A          <- core.findA(committed_traj, t_now + lookahead)
   2. path       <- hgp.plan(A, goal, voxel_map, obstacles_t, params)
   3. polys      <- decomp.sfc(path, voxel_map, obstacles_t, params)
   4. traj_new   <- solver.solve(A, path, polys, params)
   5. commit     <- traj_new
   6. publish    -> goal (timer @ params.dc), viz markers (on commit)

   state machine: YAWING -> TRAVELING -> GOAL_SEEN -> GOAL_REACHED
```

## Entry points

```
# (default) C++ stack — wrapper forwards to ~/code/sando_ws launcher:
python3 src/sando/scripts/run_sim.py -m dynamic -d hard -s install/setup.bash

# Hybrid: same ROS world, but Python SandoNode replaces the C++ one:
python3 src/sando/scripts/run_sim.py -m dynamic -d hard -s install/setup.bash --backend py

# Or run the Python node directly inside an already-sourced ROS shell:
python -m sando_py.sim_io
```

## Status

| subpackage | status                                                                                |
|------------|---------------------------------------------------------------------------------------|
| `core/`    | implemented (port of `sando_type.hpp` + `utils.{hpp,cpp}` + YAML loader)              |
| `hgp/`     | v2 implemented — 3D 26-connected A*, dynamic-heat overlay with temporal sampling, static halo |
| `decomp/`  | v1 implemented — axis-aligned-box safety corridors                                    |
| `solver/`  | v1 (cubic Hermite) + v2 (Gurobi QP — full state anchored, Bezier corridor, time-alloc adapter, parallel factor search) implemented |
| `sim_io/`  | rclpy.Node + msg + viz converters; replan wires hgp → decomp → solver with `findA` lookahead and pushes 4 RViz topics |
| `src/sando/scripts/run_sim.py` | `--backend {cpp,py}` (task #16) — cpp forwards verbatim, py rewrites the upstream tmuxp YAML to swap the C++ planner for the Python module (auto-detects multi-agent namespaces, spawns one Python node per agent) |
