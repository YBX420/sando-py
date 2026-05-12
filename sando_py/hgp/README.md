# hgp/ — global planner

Heat-map-aware A* over a 3D voxel occupancy grid. Mirrors
`include/hgp/` in the C++ codebase at the `astar_heat` mode level.

## What it does

Given a start state `A`, a goal, the current obstacle snapshot, and the
current time `t_now`, produces a polyline `path: List[np.ndarray]` from
`A.pos` to (near) the goal that
  - avoids inflated static obstacle positions,
  - prefers cells with low **dynamic-heat** — cells that predicted
    dynamic-obstacle trajectories will pass through within the next
    ~`heat_tau_ratio * horizon` seconds,
  - optionally avoids halos around static occupied cells
    (`static_heat_*`).

## Contract

```python
HGPPlanner(params).plan(
    start:       RobotState,
    goal:        np.ndarray,
    obstacles_t: Iterable[DynTraj],
    t_now:       float,
) -> Optional[List[np.ndarray]]
```

Returns waypoints in the world frame, including the start and goal (or
the closest reachable approximation to the goal) as the first and last
points. `None` means A* failed (goal unreachable / map exhausted).

## Files

| file              | status        | role                                                       |
|-------------------|---------------|------------------------------------------------------------|
| `data_utils.py`   | implemented   | tiny helpers ported from `hgp/data_utils.hpp`              |
| `voxel_map.py`    | implemented   | 3D occupancy grid + optional heat overlay                  |
| `astar.py`        | implemented   | 3D 26-connected A* with heat-weighted edge cost            |
| `heat_map.py`     | implemented   | dynamic obstacle heat overlay + static halo                |
| `hgp_planner.py`  | implemented   | top-level `plan(...)` orchestration                        |

Covered by `tests/test_hgp.py` and `tests/test_heat_map.py`.

## v2 implementation notes

* **3D-canonical voxel map**: indices are 3-tuples `(ix, iy, iz)`.
  `VoxelMap.from_bounds(x_lim, y_lim, z_lim, res)` builds a full 3D
  grid; `VoxelMap.from_bounds_2d(x_lim, y_lim, res, z_plane)` builds a
  degenerate `nz = 1` grid for the "agent at fixed altitude" case (and
  for 2D-style demos).
* **3D A***: 26-connected (axis 1×6 + face-diag √2 ×12 + corner-diag
  √3 ×8). Heuristic is the 3D octile distance. Reduces to the 8-conn
  2D case for free when `nz == 1`.
* **Heat overlay**: stored as `vm.heat: np.ndarray[float32]` with the
  same shape as `vm.occupied`. `compute_dynamic_heat` writes a base
  reachable-radius halo (`alpha0 (1 - d/R_reach)^p`) plus a temporal
  tube bonus (`alpha1 · exp(-t/tau) · (1 - d/R_j)^q` max'd across
  samples). `compute_static_heat` walks the boundary cells of occupied
  regions and adds a radial-falloff halo (`alpha (1 - d/rmax)^p`). All
  vectorised over the local AABB per obstacle.
* **Temporal sampling**: `heat_num_samples` evenly spaced times in
  `[t_now, t_now + horizon]`. Per sample, the obstacle is `DynTraj.eval`'d
  to get its predicted centre; the bounding box + growing radius (`R_0
  + heat_gamma · t`) defines the tube. Decay weight is
  `exp(-t / (heat_tau_ratio · horizon))`.
* **A* edge cost** = `geometric_step + heat_weight · vm.heat[nb]`.
  `heat_weight == 0` reduces to pure geometric A*.

## Limitations vs the C++

* The C++ HGP has three modes — `sjps`, `sastar`, `astar_heat`. **All
  three are implemented**: `astar_heat` (default; A* with heat),
  `sastar` (plain A*, heat disabled — same code path with
  `heat_weight=0`), and `sjps` (3D Jump Point Search in `jps.py`).
  `params.global_planner` selects the mode; an unknown value falls
  back to `astar_heat`.
* The C++ JPS uses a precomputed neighbour lookup table
  (`JPS3DNeib`); the Python port computes natural / forced
  neighbours on demand from the parent direction. Algorithmically
  equivalent — same forced-neighbour pruning rules — but ~250 LOC
  instead of ~900.
* `hgp_timeout_duration_ms` is plumbed end-to-end (HGPPlanner →
  `jps()` / `astar()`); a non-zero value caps the per-plan wall
  clock. ``max_num_expansion`` remains the secondary budget.
* The C++ static-heat path supports per-cell variable `rmax`
  (`static_heat_rmax_m` is a default; obstacles can override via a
  user-supplied lambda). The Python port uses a single global `rmax`.

## Parameters consumed (from `core.Parameters`)

  `inflation_hgp`, `drone_radius`, `x_min/x_max`, `y_min/y_max`,
  `z_min/z_max`, `factor_hgp`, `max_num_expansion`,
  `global_planner_heuristic_weight`, `use_free_start`,
  `free_start_factor`, `use_free_goal`, `free_goal_factor`, plus the
  full heat-map group:
  `heat_weight`, `heat_alpha0`, `heat_alpha1`, `heat_p`, `heat_q`,
  `heat_gamma`, `heat_tau_ratio`, `heat_Hmax`,
  `dyn_heat_tube_radius_m`, `heat_num_samples`, `obst_max_vel`,
  `horizon`, and `static_heat_enabled` / `static_heat_alpha` /
  `static_heat_p` / `static_heat_Hmax` / `static_heat_rmax_m`.
