"""Top-level global planner: ``HGPPlanner.plan(...)`` returns a polyline
from the agent's current state to (near) the terminal goal, avoiding
inflated obstacle predictions and biased away from dynamic obstacles'
predicted trajectories via a heat-map overlay.

v2 features (vs v1):
  * Full 3D — voxel map covers ``[z_min, z_max]`` and A* is 26-connected.
    Falls back to a single z layer when ``z_max - z_min < res`` so the
    "agent at fixed altitude" demos still work the same.
  * Heat-map weighting — dynamic obstacles are sampled at multiple
    times along their predicted trajectories and their growing tubes
    contribute a soft cost to nearby cells (``heat_alpha0/1``,
    ``heat_p/q``, ``heat_gamma``, ``heat_Hmax``).
  * Static halo — optional soft cost around occupied cells
    (``static_heat_*``).
  * Temporal sampling — obstacles contribute heat at multiple time
    samples in ``[t_now, t_now + horizon]`` rather than a single
    snapshot at ``t_now``.

Mirrors the C++ ``include/hgp/hgp_planner.cpp`` ``astar_heat`` mode at
the level we need for the trajectory bias.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from sando_py.core import DynTraj, Parameters, RobotState

from .astar import astar, simplify_path_los
from .heat_map import compute_dynamic_heat, compute_static_heat
from .jps import jps
from .voxel_map import VoxelMap


class HGPPlanner:
    """Stateless wrapper — one instance per running node. Each ``plan``
    rebuilds a local 3D grid covering the start/goal AABB plus a
    margin; the most recently built grid is exposed on
    ``last_voxel_map`` so callers (e.g. ``decomp.sfc``) can reuse it
    without re-rasterizing the obstacles."""

    def __init__(self, params: Parameters) -> None:
        self.params = params
        self.last_voxel_map: Optional[VoxelMap] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(
        self,
        start: RobotState,
        goal: np.ndarray,
        obstacles_t: Iterable[DynTraj],
        t_now: float,
        *,
        margin: float = 5.0,
        cloud_points: Optional[np.ndarray] = None,
    ) -> Optional[List[np.ndarray]]:
        """Return a list of 3D waypoints from ``start.pos`` to ``goal``.

        ``cloud_points`` is an optional ``(N, 3)`` array of static-world
        sample points (e.g. a depth-camera ``PointCloud2`` already
        transformed into the map frame). When supplied, each point is
        rasterised into the voxel map as an occupied cell before A*
        runs — this is how full-Gazebo modes feed sensor data to the
        planner. In ``rviz_only`` and ``dynamic`` (RViz-only) demos
        the caller passes ``None`` and the planner sees only the
        ``obstacles_t`` trajectories.

        Returns ``None`` if A* fails. Endpoints are snapped to the exact
        start and goal so the downstream decomp/solver don't have to
        deal with quarter-cell offsets.
        """
        start_pos = np.asarray(start.pos, dtype=float).reshape(3)
        goal_pos = np.asarray(goal, dtype=float).reshape(3)

        obstacles_t = list(obstacles_t)

        vm = self._build_voxel_map(start_pos, goal_pos, margin=margin)
        self._mark_static_obstacles(vm, obstacles_t, t_now)

        # Rasterise sensor cloud (if any). Inflated by the same
        # ``inflation_hgp`` half-cell margin as the obstacle trajectories,
        # so the planner sees a consistent safety buffer.
        if cloud_points is not None and len(cloud_points) > 0:
            infl_m = float(self.params.inflation_hgp or 0.0)
            infl_cells = int(np.ceil(infl_m / vm.res)) if infl_m > 0 else 0
            vm.mark_points(cloud_points, inflation_cells=infl_cells)

        # Free-start / free-goal: C++ ``freeStart`` / ``freeGoal``
        # (hgp_manager.cpp:147-157) clears a sphere of radius
        # ``factor * res_`` around the point — the grid resolution, NOT
        # the drone's geometric radius. The previous Python multiplied
        # by drone_radius, which made the clear-sphere 2-3× larger than
        # C++ and effectively erased nearby obstacles.
        if self.params.use_free_start:
            r = float(self.params.free_start_factor) * vm.res
            if r <= 0:
                r = vm.res * 1.5
            vm.clear_sphere(start_pos, r)

        if self.params.use_free_goal:
            r = float(self.params.free_goal_factor) * vm.res
            if r <= 0:
                r = vm.res * 1.5
            vm.clear_sphere(goal_pos, r)

        # Heat overlay (no-op when the relevant params are zero or the
        # selected mode doesn't use heat — JPS/sastar skip the compute).
        mode = self._planner_mode()
        if mode == "astar_heat" and self._heat_enabled():
            if obstacles_t:
                compute_dynamic_heat(vm, obstacles_t, self.params, t_now)
            if self.params.static_heat_enabled:
                compute_static_heat(vm, self.params)

        self.last_voxel_map = vm

        start_idx = vm.world_to_idx(start_pos)
        goal_idx = vm.world_to_idx(goal_pos)

        h_weight = self.params.global_planner_heuristic_weight or 1.0
        max_exp = self.params.max_num_expansion or 100_000
        timeout_s = max(0.0, float(self.params.hgp_timeout_duration_ms or 0) / 1000.0)

        if mode == "sjps":
            idx_path = jps(
                start_idx, goal_idx, vm,
                heuristic_weight=h_weight,
                max_expansions=max_exp,
                timeout_s=timeout_s,
            )
        else:
            # ``astar_heat`` (uses heat if available) or ``sastar``
            # (force heat_weight=0 → plain A*).
            heat_w = (
                float(self.params.heat_weight or 0.0)
                if mode == "astar_heat" else 0.0
            )
            idx_path = astar(
                start_idx,
                goal_idx,
                vm,
                heuristic_weight=h_weight,
                max_expansions=max_exp,
                heat_weight=heat_w,
                timeout_s=timeout_s,
            )
        if idx_path is None:
            return None

        smoothed = simplify_path_los(idx_path, vm)
        waypoints = [vm.idx_to_world(ix) for ix in smoothed]

        # Snap the endpoints to the exact start / goal so downstream code
        # doesn't have to deal with quarter-cell offsets from the grid.
        if waypoints:
            waypoints[0] = start_pos.copy()
            waypoints[-1] = goal_pos.copy()
        return waypoints

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _heat_enabled(self) -> bool:
        """``True`` if any heat-overlay parameter is non-zero. Saves
        the cost of computing a guaranteed-zero overlay."""
        return (
            (self.params.heat_alpha0 or 0.0) > 0
            or (self.params.heat_alpha1 or 0.0) > 0
            or bool(self.params.static_heat_enabled)
        )

    def _planner_mode(self) -> str:
        """Resolve ``params.global_planner`` to one of the three
        supported modes. Unknown / unset values fall back to
        ``astar_heat`` — the most expressive of the three, matching
        the C++ default when ``global_planner`` is empty."""
        mode = (self.params.global_planner or "astar_heat").strip()
        if mode not in ("astar_heat", "sastar", "sjps"):
            mode = "astar_heat"
        return mode

    def _build_voxel_map(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        margin: float,
    ) -> VoxelMap:
        res = max(0.05, float(self.params.res) * float(self.params.factor_hgp or 1.0))

        x_lo = min(start[0], goal[0]) - margin
        x_hi = max(start[0], goal[0]) + margin
        y_lo = min(start[1], goal[1]) - margin
        y_hi = max(start[1], goal[1]) + margin

        # Clamp to world bounds (skip if unset)
        if self.params.x_min < self.params.x_max:
            x_lo = max(x_lo, self.params.x_min)
            x_hi = min(x_hi, self.params.x_max)
        if self.params.y_min < self.params.y_max:
            y_lo = max(y_lo, self.params.y_min)
            y_hi = min(y_hi, self.params.y_max)

        if x_hi - x_lo < 2 * res:
            x_hi = x_lo + 2 * res
        if y_hi - y_lo < 2 * res:
            y_hi = y_lo + 2 * res

        # Z range: full 3D when params.z_min < params.z_max with at
        # least 2 cells of range; otherwise a single-z-layer 2D map at
        # the agent's altitude (matches the v1 behaviour for a UAV
        # cruising at a fixed band).
        z_min_p = float(self.params.z_min or 0.0)
        z_max_p = float(self.params.z_max or 0.0)
        if z_max_p - z_min_p >= 2 * res:
            return VoxelMap.from_bounds(
                (x_lo, x_hi), (y_lo, y_hi), (z_min_p, z_max_p), res,
            )
        return VoxelMap.from_bounds_2d(
            (x_lo, x_hi), (y_lo, y_hi), res, z_plane=float(start[2]),
        )

    def _mark_static_obstacles(
        self,
        vm: VoxelMap,
        obstacles_t: Iterable[DynTraj],
        t_now: float,
    ) -> None:
        """Mark the obstacle positions at ``t_now`` into the occupancy
        grid (hard occupancy). The heat overlay handles the "future
        position" bias separately.

        ``traj.bbox`` is the obstacle's half-extents along (x, y, z).
        Marks an axis-aligned box rather than a sphere — using a sphere
        with ``max(bbox)`` blows up elongated obstacles (tall trees
        ``[0.4, 0.4, 4.0]`` or horizontal poles ``[0.4, 8.0, 0.4]``)
        into 4-8 m balls that wall off the entire map."""
        infl = float(self.params.inflation_hgp or 0.0)
        for traj in obstacles_t:
            try:
                center = traj.eval(t_now)
            except Exception:  # noqa: BLE001
                center = traj.current_pos
            half = np.asarray(traj.bbox, dtype=float).reshape(-1)
            if half.size < 3:
                half = np.array([
                    float(half[0]) if half.size > 0 else 0.4,
                    float(half[1]) if half.size > 1 else 0.4,
                    float(half[2]) if half.size > 2 else 0.4,
                ])
            self._mark_aabb(vm, center, half[:3], infl)

    @staticmethod
    def _mark_aabb(vm: VoxelMap, center: np.ndarray,
                   half: np.ndarray, infl: float) -> int:
        """Mark every cell in ``[center - half - infl, center + half + infl]``
        as occupied. ``infl`` is the ``inflation_hgp`` safety margin
        (added to every face). Returns the number of cells flipped."""
        if vm is None or vm.occupied.size == 0:
            return 0
        c = np.asarray(center, dtype=float).reshape(3)
        h = np.asarray(half, dtype=float).reshape(3) + float(infl)
        lo = c - h
        hi = c + h
        ix_lo = max(0, int(np.floor((lo[0] - vm.x_min) / vm.res)))
        ix_hi = min(vm.nx, int(np.ceil((hi[0] - vm.x_min) / vm.res)) + 1)
        iy_lo = max(0, int(np.floor((lo[1] - vm.y_min) / vm.res)))
        iy_hi = min(vm.ny, int(np.ceil((hi[1] - vm.y_min) / vm.res)) + 1)
        if vm.nz <= 1:
            iz_lo, iz_hi = 0, 1
        else:
            iz_lo = max(0, int(np.floor((lo[2] - vm.z_min) / vm.res)))
            iz_hi = min(vm.nz, int(np.ceil((hi[2] - vm.z_min) / vm.res)) + 1)
        if ix_lo >= ix_hi or iy_lo >= iy_hi or iz_lo >= iz_hi:
            return 0
        block = vm.occupied[ix_lo:ix_hi, iy_lo:iy_hi, iz_lo:iz_hi]
        before = int(block.sum())
        block[:] = True
        after = int(block.sum())
        if after != before:
            vm._occ_version += 1
        return after - before
