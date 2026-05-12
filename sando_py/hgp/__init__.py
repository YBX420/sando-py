"""Global planner — heat-map-aware A* over a 3D voxel occupancy grid.

Mirrors ``include/hgp/`` in the C++ codebase at the ``astar_heat``
level. See ``hgp/README.md``.

Contract
--------
    HGPPlanner(params).plan(
        start:       RobotState,
        goal:        np.ndarray,
        obstacles_t: Iterable[DynTraj],
        t_now:       float,
    ) -> Optional[List[np.ndarray]]

Returns a polyline path (3D world coordinates) from ``start.pos`` to
``goal``, avoiding inflated obstacle snapshots and biased away from
predicted dynamic-obstacle trajectories. ``None`` means A* failed.

Public API
----------
VoxelMap                — 3D occupancy + heat overlay (voxel_map.py)
astar                   — 3D 26-connected A* with heat-weighted cost (astar.py)
simplify_path_los       — 3D line-of-sight smoothing (astar.py)
compute_dynamic_heat    — dynamic obstacle heat overlay (heat_map.py)
compute_static_heat     — static obstacle halo (heat_map.py)
HGPPlanner              — top-level orchestration (hgp_planner.py)

V2 features (vs v1)
-------------------
* Full 3D — voxel map is ``nx × ny × nz``; A* is 26-connected. A 2D
  scenario is the degenerate case ``nz = 1`` (via
  ``VoxelMap.from_bounds_2d``).
* Heat-map weighting — dynamic obstacles contribute a soft cost to
  nearby cells via ``heat_alpha0/1``, ``heat_p/q``, ``heat_gamma``,
  ``heat_Hmax``. Static obstacles can add a halo via
  ``static_heat_*``. A* edge cost includes ``heat_weight ·
  vm.heat[neighbor]``.
* Temporal sampling — obstacles are sampled at ``heat_num_samples``
  times in ``[t_now, t_now + horizon]`` rather than a single snapshot.

See ``hgp/README.md`` for details and limitations.
"""

from .astar import astar, octile, simplify_path_los
from .jps import jps
from .data_utils import (
    ANSI_COLOR_BLUE,
    ANSI_COLOR_CYAN,
    ANSI_COLOR_GREEN,
    ANSI_COLOR_MAGENTA,
    ANSI_COLOR_RED,
    ANSI_COLOR_RESET,
    ANSI_COLOR_YELLOW,
    total_distance,
    transform_vec,
)
from .heat_map import compute_dynamic_heat, compute_static_heat
from .hgp_planner import HGPPlanner
from .voxel_map import VoxelMap

__all__ = [
    # planner
    "HGPPlanner",
    "VoxelMap",
    "astar",
    "jps",
    "octile",
    "simplify_path_los",
    # heat-map overlays
    "compute_dynamic_heat",
    "compute_static_heat",
    # helpers from data_utils
    "transform_vec",
    "total_distance",
    "ANSI_COLOR_RED",
    "ANSI_COLOR_GREEN",
    "ANSI_COLOR_YELLOW",
    "ANSI_COLOR_BLUE",
    "ANSI_COLOR_MAGENTA",
    "ANSI_COLOR_CYAN",
    "ANSI_COLOR_RESET",
]
