# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Workspace layout

`/home/boxuan/code` contains **three interlinked colcon workspaces** for SANDO (Safe Autonomous Trajectory Planning for Dynamic Unknown Environments — MIT-ACL, ROS 2 Humble):

- `sando_ws/` — main workspace. Real source lives in `sando_ws/src/sando/` (the `mit-acl/sando` repo, branch `main`). The other entries in `sando_ws/src/` are **symlinks into `sando_ws/src/sando/deps/`** (`acl-mapping`, `dynus_interfaces`, `gazebo_ros_pkgs`, `livox_laser_simulation_ros2`, `realsense_gazebo_plugin`, `uav_simulator`, `unitree-go2-ros2`).
- `decomp_ws/` — built separately because `decomp_util` must exist before the rest of `DecompROS2` builds. `decomp_ws/src/DecompROS2` symlinks to `sando_ws/src/sando/deps/DecompROS2`.
- `livox_ws/` — built with the upstream `build.sh humble` script (not plain `colcon`). `livox_ws/src/livox_ros_driver2` symlinks to `sando_ws/src/sando/deps/livox_ros_driver2`. The Livox-SDK2 C library is installed system-wide to `/usr/local/lib`.

Sourcing order matters: `/opt/ros/humble` → `decomp_ws/install/setup.bash` → `sando_ws/install/setup.bash`. `setup.sh` appends all of these (plus `LD_LIBRARY_PATH` for `livox_ws/install/livox_ros_driver2/lib` and Gurobi env) to `~/.bashrc`. `ROS_DOMAIN_ID=20` is also set in `~/.bashrc`.

When editing or grepping, work inside `sando_ws/src/sando/`. The other workspaces' `src/` are just symlinks — edits land in the same files but the workspaces exist so colcon discovers and builds them in the right order.

## Build / rebuild

Initial install (idempotent, safe to re-run): `cd sando_ws/src/sando && ./setup.sh [-j N]`. This installs ROS 2 Humble, Gurobi 11.0.3 (to `/opt/gurobi1103`), system deps, builds Livox-SDK2 + livox_ros_driver2 + DecompROS2, then the SANDO workspace.

Incremental rebuild of just the main package:
```bash
cd ~/code/sando_ws
colcon build --packages-select sando --cmake-args -DCMAKE_BUILD_TYPE=Release
```

The full SANDO `colcon build` invocation (used by `setup.sh`) requires `--allow-overriding gazebo_dev gazebo_msgs gazebo_ros gazebo_ros_pkgs gazebo_plugins` because we ship a patched fork of `gazebo_ros_pkgs` that overrides the apt-installed versions. Drop this flag and the build will fail.

CMake flags: `-O3 -march=native -ffast-math -funroll-loops`, IPO/LTO on, OpenMP required, C++17. Template-heavy files (e.g. `gazebo_ros_camera.cpp`) can use 3–4 GB RAM per `cc1plus` — drop `--parallel-workers` if you hit OOM.

## Lint / format

Pre-commit is configured (`.pre-commit-config.yaml`): `clang-format` (Google base, 100-col, custom include-priority groups in `.clang-format`), `ruff-format` + `ruff` for Python, plus trailing-whitespace / EOF fixers. Install with `pre-commit install` inside `sando_ws/src/sando/`. There are no unit tests in this repo — verification is done via simulation runs and the benchmarking harness.

## Running simulations

Always run from the workspace root (`~/code/sando_ws`) so `install/setup.bash` resolves and the `rviz/sando.rviz` path is correct. The launcher script auto-generates a tmuxp YAML and execs it:

```bash
python3 src/sando/scripts/run_sim.py -m interactive       -s install/setup.bash
python3 src/sando/scripts/run_sim.py -m static  -d easy   -s install/setup.bash
python3 src/sando/scripts/run_sim.py -m dynamic -d hard   -s install/setup.bash
python3 src/sando/scripts/run_sim.py -m unknown_dynamic -d medium -s install/setup.bash
```

Modes: `interactive` (RViz, click 2D Nav Goal or hit `g`), `static` (Gazebo forest), `dynamic` (RViz-only known dynamic obstacles), `unknown_dynamic` (Gazebo + perception). Difficulties `easy/medium/hard` map to 50/100/200 obstacles. There are also `--mode hover-test` and `--mode adversarial-test`. Add `--dry-run` to inspect the generated tmuxp YAML without launching.

The Docker path (`sando_ws/src/sando/docker/Makefile`) wraps the same modes: `make run-interactive`, `make run-demo SCENARIO=static_easy`, `make shell`. Build with `make build BUILD_JOBS=N` (default 2). A WLS Gurobi license must be at `docker/gurobi.lic` — only WLS licenses work inside Docker. Use `GPU=false` to disable nvidia-container-toolkit if it isn't installed (RViz-only modes work fine without GPU).

## Benchmarking

```bash
# Local trajectory benchmark — requires SFCs to be generated once
tmuxp load src/sando/launch/generate_sfc.yaml
cd src/sando/benchmarking && python3 run_benchmark_suite.py

# End-to-end simulation benchmarks
python3 src/sando/scripts/run_benchmark.py -s install/setup.bash --mode rviz-only \
  --cases easy medium hard --config-name dynamic --num-trials 10
```

Outputs land in `src/sando/benchmark_data/` (CSV/JSON + ROS bags). Analysis: `scripts/analyze_dynamic_benchmark.py`, `benchmarking/generate_latex_table.py`, plus a C++ port `analyze_benchmark` (built from `src/tools/analyze_benchmark.cpp`).

## Code architecture

The planner is a single ROS 2 node (`sando_node.cpp` → `SANDO` class in `include/sando/sando.hpp` + `src/sando/sando.cpp`) wrapping a state machine `YAWING → TRAVELING → GOAL_SEEN → GOAL_REACHED` (plus `HOVER_AVOIDING`). Per replan tick it:

1. Picks a start state `A` from the committed trajectory (`findAandAtime`).
2. Runs the **HGP** global planner (`include/hgp/`, `src/hgp/`) — a heat-map A* over a voxel grid plus convex decomposition into safety corridors. `hgp_manager.cpp` owns the voxel/heat map and feeds `hgp_planner.cpp` + `graph_search.cpp`.
3. Hands corridors + initial guess to **`GurobiSolver`** (`include/sando/gurobi_solver.hpp`, `src/sando/gurobi_solver.cpp`) — Hermite-spline parameterized QP with variable elimination and a dynamic time-scaling factor that adapts on convergence failure.
4. Publishes the committed trajectory; the converter nodes (`convert_goal_to_cmd_vel`, `convert_odom_to_state`, `convert_vicon_to_state`, `odom_to_global_state`) bridge between hardware/sim sources and the `dynus_interfaces` State/Goal types.

Other targets in `CMakeLists.txt`:
- `obstacle_tracker_node` — clusters/predicts dynamic obstacles from pointclouds.
- `fake_sim` — lightweight kinematic sim used by RViz-only modes.
- `dynamic_obstacles_world_plugin` (Gazebo world plugin) — single plugin manages all dynamic obstacles, replacing the legacy per-model `move_model` plugin.
- `imu_plugin`, `dynamic_forest_node` — sim-side helpers.
- `corridor_generator_node`, `local_traj_benchmark_node`, `visualize_local_trajs_node`, `temporal_layered_corridor_test_node` — benchmarking/visualization-only entry points that link the same `sando.cpp` core.
- `analyze_benchmark`, `convert_bag` — bag post-processing tools.

## Configuration

All planner params are in `sando_ws/src/sando/config/sando.yaml`, organized as `[CONFIGURE]` (vehicle/env: `v_max`, `a_max`, `j_max`, `drone_bbox`, `z_min`/`z_max`, `num_P`/`num_N`), `[TUNE]` (`inflation_hgp`, `sfc_size`, `goal_seen_radius`, `heat_alpha0/1`, `dynamic_factor_initial_mean`), `[INTERNAL]` (algorithm internals — change only with reason). Hardware deployment uses `config/sando_hw_quadrotor.yaml` whose inline comments call out every delta from sim defaults.

## Things to know before changing code

- `setup.sh` is the source of truth for environment setup — if you change a build dependency, update `setup.sh` too, otherwise fresh installs break.
- The DecompROS2 fork must build `decomp_util` first in isolation, then the rest. This is why it lives in its own workspace.
- `gazebo_ros_pkgs` is a vendored fork; never `apt install` `ros-humble-gazebo-ros-pkgs` over it without re-applying the override.
- `dynus_interfaces` defines the message/service types the planner exchanges; if you add a field, rebuild it before `sando` (colcon will normally handle ordering, but a stale install dir can cause stale-header bugs — `rm -rf build/sando install/sando` and rebuild).
- Gurobi license: build does **not** need it; running does. License goes at `~/gurobi.lic` (or `docker/gurobi.lic` for the container).
