"""HGPPlanner — orchestrates voxel map search, LOS shortcut, and path
simplification. Wraps `GraphSearch` and applies the same post-processing
pipeline as `hgp_planner.cpp`.
"""
from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from ..utils import (
    angle_spacing_filter, collapse_short_edges, remove_collinear,
)
from .graph_search import GraphSearch
from .voxel_map import VoxelMapUtil


class HGPPlanner:
    def __init__(
        self,
        global_planner: str,
        verbose: bool,
        v_max: float,
        a_max: float,
        j_max: float,
        timeout_duration_ms: int,
        w_unknown: float,
        w_align: float,
        decay_len_cells: float,
        w_side: float,
        los_cells: int = 0,
        min_len: float = 0.5,
        min_turn: float = 10.0,
        heat_weight: float = 10.0,
        obstacle_soft_cost: float = 5.0,
    ):
        self.global_planner = global_planner
        self.verbose = verbose
        self.v_max = v_max
        self.a_max = a_max
        self.j_max = j_max
        self.timeout_ms = int(timeout_duration_ms)
        self.w_unknown = w_unknown
        self.w_align = w_align
        self.decay_len_cells = decay_len_cells
        self.w_side = w_side
        self.los_cells = los_cells
        self.min_len = min_len
        self.min_turn_deg = min_turn
        self.heat_weight = heat_weight
        self.obstacle_soft_cost = obstacle_soft_cost
        self.map_util: VoxelMapUtil = None  # type: ignore[assignment]
        self.graph_search: GraphSearch = None  # type: ignore[assignment]
        self.max_expand = 100000
        self.raw_path: List[np.ndarray] = []
        self.path: List[np.ndarray] = []
        self.timings = dict(
            global_planning_time=0.0,
            hgp_static_jps_time=0.0,
            hgp_check_path_time=0.0,
            hgp_dynamic_astar_time=0.0,
            hgp_recover_path_time=0.0,
        )

    def set_map_util(self, map_util: VoxelMapUtil) -> None:
        self.map_util = map_util
        self.graph_search = GraphSearch(
            map_util,
            eps=1.0,
            global_planner=self.global_planner,
            w_unknown=self.w_unknown,
            w_align=self.w_align,
            decay_len_cells=self.decay_len_cells,
            w_side=self.w_side,
            verbose=self.verbose,
            heat_weight=self.heat_weight,
            obstacle_soft_cost=self.obstacle_soft_cost,
        )

    def set_max_expand(self, n: int) -> None:
        self.max_expand = n

    def update_vmax(self, v: float) -> None:
        self.v_max = v

    def plan(
        self,
        start: np.ndarray,
        start_vel: np.ndarray,
        goal: np.ndarray,
        current_time: float,
        eps: float = 1.0,
    ) -> Tuple[bool, float]:
        """Run search; return (success, final_g)."""
        if self.map_util is None or not self.map_util.initialized:
            return False, 0.0

        t0 = time.perf_counter()
        self.graph_search.eps = float(eps)
        start_int = self.map_util.float_to_int(start)
        goal_int = self.map_util.float_to_int(goal)
        initial_g = float(np.linalg.norm(start - self.map_util.int_to_float(start_int)))

        ok = self.graph_search.plan(
            start_int, goal_int, initial_g, start_vel,
            max_expand=self.max_expand, timeout_ms=self.timeout_ms,
        )
        self.timings.update(self.graph_search.last_metrics)

        raw_path = self.graph_search.get_path_world()
        if not raw_path:
            self.raw_path = []
            self.path = []
            return False, 0.0
        # Override start/goal endpoints with their world positions
        raw_path[0] = start.copy()
        raw_path[-1] = goal.copy() if ok else raw_path[-1]
        self.raw_path = raw_path

        # Post-processing pipeline. The point-dropping filters get the map's
        # blocked-check so they never simplify a segment through an obstacle.
        blk = self.map_util.is_blocked
        path = self.short_cut_by_los(raw_path, self.los_cells)
        path = collapse_short_edges(path, self.min_len, is_blocked=blk)
        path = angle_spacing_filter(path, self.min_turn_deg, self.min_len, is_blocked=blk)
        path = self.short_cut_by_los(path, self.los_cells)
        path = self.clean_up_path(path)
        path = self._repair_blocked_segments(path, raw_path)  # safety net
        self.path = path

        final_g = self._path_cost(path)
        self.timings["global_planning_time"] = time.perf_counter() - t0
        return ok, final_g

    # ------------------------------------------------------------------
    def short_cut_by_los(self, points: List[np.ndarray], radius_cells: int) -> List[np.ndarray]:
        if len(points) <= 2:
            return [p.copy() for p in points]
        out = [points[0].copy()]
        i = 0
        n = len(points)
        while i < n - 1:
            j = n - 1
            while j > i + 1:
                if self.map_util.line_of_sight_capsule(points[i], points[j], radius_cells):
                    break
                j -= 1
            out.append(points[j].copy())
            i = j
        return out

    def clean_up_path(self, points: List[np.ndarray]) -> List[np.ndarray]:
        # collinear + corner removal forward + reversed
        path = remove_collinear(points)
        path = self._remove_corner_pts(path)
        path_rev = self._remove_corner_pts(list(reversed(path)))
        return list(reversed(path_rev))

    def _repair_blocked_segments(self, path: List[np.ndarray],
                                 raw: List[np.ndarray]) -> List[np.ndarray]:
        """Safety net: guarantee no simplified segment clips an obstacle.

        The raw A* path is collision-free (adjacent cells). If a simplified
        segment is blocked, splice the raw sub-path between the raw vertices
        nearest the segment's endpoints back in.
        """
        if len(path) < 2 or len(raw) < 2:
            return path

        def nearest_raw(pt: np.ndarray) -> int:
            return min(range(len(raw)), key=lambda r: float(np.linalg.norm(raw[r] - pt)))

        out = [path[0].copy()]
        for k in range(len(path) - 1):
            a, b = path[k], path[k + 1]
            if self.map_util.is_blocked(a, b):
                ia, ib = nearest_raw(a), nearest_raw(b)
                if ia < ib:
                    for q in raw[ia + 1:ib]:
                        out.append(q.copy())
            out.append(b.copy())
        return out

    def _remove_corner_pts(self, points: List[np.ndarray]) -> List[np.ndarray]:
        if len(points) <= 2:
            return [p.copy() for p in points]
        out = [points[0].copy()]
        prev_cost = self._seg_cost(out[-1], points[1])
        prev_peak = self._seg_peak_heat(out[-1], points[1])
        i = 1
        while i < len(points) - 1:
            mid = points[i]
            nxt = points[i + 1]
            cost2 = self._seg_cost(mid, nxt)
            peak2 = self._seg_peak_heat(mid, nxt)
            cost3 = self._seg_cost(out[-1], nxt)
            peak3 = self._seg_peak_heat(out[-1], nxt)
            if cost3 < prev_cost + cost2 and peak3 <= 1.5 * max(prev_peak, peak2):
                prev_cost = cost3
                prev_peak = peak3
            else:
                out.append(mid.copy())
                prev_cost = cost2
                prev_peak = peak2
            i += 1
        out.append(points[-1].copy())
        return out

    def _seg_cost(self, a: np.ndarray, b: np.ndarray) -> float:
        if self.map_util.is_blocked(a, b):
            return float("inf")
        L = float(np.linalg.norm(b - a))
        if L < 1e-9:
            return 0.0
        ds = max(0.5 * self.map_util.res, 0.05)
        n = max(2, int(np.ceil(L / ds)))
        heat_int = 0.0
        for k in range(n + 1):
            t = k / n
            p = a + t * (b - a)
            cell = self.map_util.float_to_int(p)
            heat_int += self.map_util.get_heat(*cell) * (L / n)
        return L + self.heat_weight * heat_int

    def _seg_peak_heat(self, a: np.ndarray, b: np.ndarray) -> float:
        L = float(np.linalg.norm(b - a))
        if L < 1e-9:
            cell = self.map_util.float_to_int(a)
            return self.map_util.get_heat(*cell)
        ds = max(0.5 * self.map_util.res, 0.05)
        n = max(2, int(np.ceil(L / ds)))
        peak = 0.0
        for k in range(n + 1):
            t = k / n
            p = a + t * (b - a)
            cell = self.map_util.float_to_int(p)
            peak = max(peak, self.map_util.get_heat(*cell))
        return peak

    def _path_cost(self, path: List[np.ndarray]) -> float:
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            total += float(np.linalg.norm(b - a))
        return total

    def get_path(self) -> List[np.ndarray]:
        return self.path

    def get_raw_path(self) -> List[np.ndarray]:
        return self.raw_path
