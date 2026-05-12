# sim_io/ — ROS 2 adapter layer

Wraps the pure-Python algorithm code (`core`, `hgp`, `decomp`, `solver`)
in a single `rclpy.Node` so it can drop into the same ROS 2 simulation
world the upstream C++ project uses. Nothing in this folder is
algorithmic — it just shuttles ROS messages in and out of the
Python types defined in `core/`.

## Why not a pure-Python sim?

The original SANDO demo is built on ROS 2 + RViz + a Gazebo / fake_sim
world + a `dynamic_forest_node` that publishes 200 dynamic obstacles
over `/trajs`. Re-doing all of that in matplotlib would be a lot of
throwaway work and would lose the value of comparing against the C++
reference. Instead, this subpackage lets the same world stay in
place; only the **`sando` planner Node** is replaced by a Python
re-implementation. RViz, the obstacles, the fake quadrotor, and the
goal sender are all still the upstream C++ nodes.

## Architecture

```
       upstream C++ nodes (unchanged)
   +-----------------------+
   | rviz2                 | <----+ markers, polytopes, traj viz
   | dynamic_forest_node   | --+  |
   | fake_sim              | --+  |
   | goal_sender           | --+  |
   +-----------------------+   |  |
              | DynTraj msgs   |  |
              | State msgs     |  |
              | PoseStamped    |  |
              v                |  |
   +-----------------------+   |  |
   | sando_py.sim_io.      |   |  |
   | SandoNode (rclpy)     | <-+  |
   |  - subs/pubs/timers   |      |
   |  - msg_convert        | -----+
   |  - calls core/hgp/    |
   |       decomp/solver   |
   +-----------------------+
              |
              v
       pure-Python algorithms
       (core, hgp, decomp, solver) — no ROS imports
```

## ROS I/O contract (mirrors `sando_node.cpp`)

**Subscribes:**
| topic            | msg type                              | callback           |
|------------------|---------------------------------------|--------------------|
| `/trajs`         | `dynus_interfaces/msg/DynTraj`        | trajCallback       |
| `predicted_trajs`| `dynus_interfaces/msg/DynTraj`        | trajCallback       |
| `state`          | `dynus_interfaces/msg/State`          | stateCallback      |
| `term_goal`      | `geometry_msgs/msg/PoseStamped`       | terminalGoalCallback |

**Publishes:**
| topic                      | msg type                                   | rate                     |
|----------------------------|--------------------------------------------|--------------------------|
| `goal`                     | `dynus_interfaces/msg/Goal`                | every `params.dc` s      |
| `actual_traj`              | `visualization_msgs/msg/MarkerArray`       | on commit                |
| `hgp_path_marker`          | `visualization_msgs/msg/MarkerArray`       | on commit                |
| `poly_safe`                | `decomp_ros_msgs/msg/PolyhedronArray`      | on commit                |
| `traj_committed_colored`   | `visualization_msgs/msg/MarkerArray`       | on commit                |

**Timers:**
| timer                  | period               | callback                  |
|------------------------|----------------------|---------------------------|
| `timer_replanning_`    | 10 ms                | `replanCallback`          |
| `timer_goal_`          | `params.dc` (~10 ms) | `publishGoal`             |
| `timer_cleanup_old_trajs_` | 500 ms           | `cleanUpOldTrajsCallback` |

## Files

| file                | status        | role                                                         |
|---------------------|---------------|--------------------------------------------------------------|
| `sando_node.py`     | v2 implemented | `SandoNode(rclpy.Node)` — subs/pubs/timers, wires `hgp → decomp → solver` per replan tick (anchored at the lookahead-evaluated `_find_A`), evaluates the committed trajectory in the goal publisher, fires the 4 viz publishers on each successful plan + a throttled `actual_traj` from the goal-pub timer |
| `msg_convert.py`    | implemented   | `ros2core` / `core2ros` for each planner-input / output message type |
| `viz.py`            | implemented   | core types → `MarkerArray` / `PolyhedronArray` for the 4 RViz topics (Jet colormap, path lines+dots, Bezier polyhedron, history line strip) |
| `__main__.py`       | implemented   | thin `rclpy.init() ; rclpy.spin(SandoNode()) ; rclpy.shutdown()` so `python -m sando_py.sim_io` works |

Covered by `tests/ros/test_sando_node.py` (state machine + end-to-end pipeline), `tests/ros/test_msg_convert.py` (round-trip per msg type), `tests/ros/test_viz.py` (marker / polyhedron contract + publisher wiring), and `tests/ros/test_hover_avoidance.py` (`HOVER_AVOIDING` state + repulsion + return-to-hover + yaw branch).

### Gaps vs the C++ (planned)

* `PointCloud2` fusion is wired (subscribes to both `sensor_point_cloud`
  and `/map_generator/global_cloud`; rasterised into the voxel map at
  replan time with TF-buffer transforms from any source frame). Two
  follow-ups remain: free-space carving (the C++ publishes
  `pub_free_map_` etc. for visualisation; we don't), and a published
  occupancy `MarkerArray` so the rasterised cloud is visible in RViz.
* C++ also publishes `original_hgp_path_marker`, `free_hgp_path_marker`,
  and `poly_whole` (whole-trajectory corridor before time-aware
  shrinking) — the Python port omits those: hgp has no
  "original" vs "smoothed" distinction in v2, and decomp v1 has no
  whole-vs-safe split.
* `findA` adaptive ``k_value`` — implemented in `_find_A` +
  `_record_warmup_sample` + `_advance_warmup_state`. Warm-up uses a
  fixed ``default_k_value`` lookahead while collecting actual replan
  wall-time samples; once ``num_replanning_before_adapt`` are in hand,
  the lookahead switches to ``(int(k_value_factor * t_est / dc) - 1)
  * dc`` with ``t_est`` an EMA filtered by ``alpha_k_value_filtering``.
  Toggle via ``use_adaptive_lookahead`` (defaults to True; off falls
  back to the fixed ``lookahead_replan_time`` knob, which existing
  benchmarks rely on).

## Entry point

```
python -m sando_py.sim_io
```

The `src/sando/scripts/run_sim.py` wrapper will gain a `--backend py`
flag that swaps the C++ `sando` Node in the tmuxp YAML for this command.

## Notes

* `rclpy` and `dynus_interfaces.msg` are imported **inside this
  subpackage only**. The pure-Python algorithm subpackages (`core`,
  `hgp`, `decomp`, `solver`) must not import them, so that
  `pytest` in a conda env (no ROS sourced) still works. The repo
  `conftest.py` actively strips `/opt/ros/humble/...` from `sys.path`
  to enforce this.
* The C++ node fuses an occupancy grid from a `PointCloud2`. In
  `rviz_only` mode it just initializes an empty map — that's what the
  Python port will do first. Real occupancy-grid handling lands in a
  later milestone.
