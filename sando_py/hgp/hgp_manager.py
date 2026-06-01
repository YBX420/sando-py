"""HGPManager — owns the map and orchestrates global planning + SFC.

Mirrors include/hgp/hgp_manager.hpp. Convex decomposition uses a pure-Python
ellipsoid → polyhedron routine since DecompROS2's C++ shared library isn't
trivially callable from Python; we expose the same data flow (polytope per
segment with `A x <= b`) so the Gurobi solver can plug in unchanged.
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from ..types import Parameters, Polytope
from .hgp_planner import HGPPlanner
from .voxel_map import VAL_FREE, VAL_OCC, VoxelMapUtil


class HGPManager:
    def __init__(self):
        self.par = Parameters()
        self.map_util = VoxelMapUtil(res=self.par.res)
        self.planner: Optional[HGPPlanner] = None
        self.weight = 1.0
        self.res = self.par.res
        self.drone_radius = 0.2
        self.v_max = 10.0
        self.a_max = 20.0
        self.j_max = 100.0
        self.max_dist_vertexes = 2.0
        self.use_shrinked_box = False
        self.shrinked_box_size = 0.0
        self.vec_o: List[np.ndarray] = []     # occupied
        self.vec_uo: List[np.ndarray] = []    # unknown + occupied
        self.sfc_size: List[float] = [3.0, 3.0, 3.0]
        self.map_initialized = False

    # ------------------------------------------------------------------
    def set_parameters(self, par: Parameters) -> None:
        self.par = par
        self.weight = par.global_planner_heuristic_weight
        self.res = par.factor_hgp * par.res
        self.drone_radius = max(par.drone_bbox) * 0.5 if par.drone_bbox else 0.2
        self.v_max = par.v_max
        self.a_max = par.a_max
        self.j_max = par.j_max
        self.sfc_size = list(par.sfc_size)
        self.max_dist_vertexes = par.max_dist_vertexes
        self.use_shrinked_box = par.use_shrinked_box
        self.shrinked_box_size = par.shrinked_box_size

        self.map_util.res = self.res
        self.map_util.use_heat_map = par.use_heat_map
        self.map_util.dynamic_heat_enabled = par.dynamic_heat_enabled
        self.map_util.dynamic_as_occupied_current = par.dynamic_as_occupied_current
        self.map_util.dynamic_as_occupied_future = par.dynamic_as_occupied_future
        self.map_util.heat_alpha0 = par.heat_alpha0
        self.map_util.heat_alpha1 = par.heat_alpha1
        self.map_util.heat_p = par.heat_p
        self.map_util.heat_q = par.heat_q
        self.map_util.heat_tau_ratio = par.heat_tau_ratio
        self.map_util.heat_gamma = par.heat_gamma
        self.map_util.heat_Hmax = par.heat_Hmax
        self.map_util.dyn_base_inflation_m = par.dyn_base_inflation_m
        self.map_util.dyn_heat_tube_radius_m = par.dyn_heat_tube_radius_m
        self.map_util.heat_num_samples = par.heat_num_samples
        self.map_util.obst_max_vel = par.obst_max_vel
        self.map_util.static_heat_enabled = par.static_heat_enabled
        self.map_util.static_heat_alpha = par.static_heat_alpha
        self.map_util.static_heat_p = par.static_heat_p
        self.map_util.static_heat_Hmax = par.static_heat_Hmax
        self.map_util.static_heat_rmax_m = par.static_heat_rmax_m
        self.map_util.static_heat_default_radius_m = par.static_heat_default_radius_m
        self.map_util.static_heat_boundary_only = par.static_heat_boundary_only
        self.map_util.static_heat_apply_on_unknown = par.static_heat_apply_on_unknown
        self.map_util.static_heat_exclude_dynamic = par.static_heat_exclude_dynamic
        self.map_util.use_soft_cost_obstacles = par.use_soft_cost_obstacles
        self.map_util.obstacle_soft_cost = par.obstacle_soft_cost

    def setup_planner(self) -> None:
        par = self.par
        self.planner = HGPPlanner(
            global_planner=par.global_planner,
            verbose=par.global_planner_verbose,
            v_max=par.v_max,
            a_max=par.a_max,
            j_max=par.j_max,
            timeout_duration_ms=par.hgp_timeout_duration_ms,
            w_unknown=par.w_unknown,
            w_align=par.w_align,
            decay_len_cells=par.decay_len_cells,
            w_side=par.w_side,
            los_cells=par.los_cells,
            min_len=par.min_len,
            min_turn=par.min_turn,
            heat_weight=par.heat_weight,
            obstacle_soft_cost=par.obstacle_soft_cost,
        )
        self.planner.set_max_expand(par.max_num_expansion)
        self.planner.set_map_util(self.map_util)

    # ------------------------------------------------------------------
    def update_map(
        self,
        wdx: float, wdy: float, wdz: float,
        center_map: np.ndarray,
        cloud_occ: np.ndarray,
        cloud_unk: Optional[np.ndarray],
        obst_pos: List[np.ndarray],
        obst_bbox: List[np.ndarray],
        traj_max_time: float,
    ) -> None:
        res = self.res
        cells_x = max(2, int(round(wdx / res)))
        cells_y = max(2, int(round(wdy / res)))
        cells_z = max(2, int(round(wdz / res)))
        self.map_util.read_map(
            cells_x, cells_y, cells_z, center_map,
            cloud_occ, self.par.z_min, self.par.z_max,
            self.par.inflation_hgp, obst_pos, obst_bbox, traj_max_time,
        )
        self.map_initialized = True
        if self.planner is None:
            self.setup_planner()

    # ------------------------------------------------------------------
    def free_start(self, start: np.ndarray, factor: float) -> np.ndarray:
        d = factor * self.res
        self.map_util.set_free_voxel_and_surroundings(start, d)
        return start

    def free_goal(self, goal: np.ndarray, factor: float) -> np.ndarray:
        d = factor * self.res
        self.map_util.set_free_voxel_and_surroundings(goal, d)
        return goal

    def check_if_point_occupied(self, p: np.ndarray) -> bool:
        ix, iy, iz = self.map_util.float_to_int(p)
        return self.map_util.is_occupied(int(ix), int(iy), int(iz))

    def check_if_point_free(self, p: np.ndarray) -> bool:
        ix, iy, iz = self.map_util.float_to_int(p)
        return self.map_util.is_free(int(ix), int(iy), int(iz))

    # ------------------------------------------------------------------
    def solve_hgp(
        self,
        start: np.ndarray, start_vel: np.ndarray, goal: np.ndarray,
        current_time: float,
    ) -> Tuple[bool, float, List[np.ndarray], List[np.ndarray]]:
        if self.planner is None:
            self.setup_planner()
        # Optional free-start / free-goal
        if self.par.use_free_start:
            self.free_start(start, self.par.free_start_factor)
        if self.par.use_free_goal:
            self.free_goal(goal, self.par.free_goal_factor)
        ok, final_g = self.planner.plan(
            start, start_vel, goal, current_time, eps=self.weight,
        )
        path = self.planner.get_path() if ok else []
        raw_path = self.planner.get_raw_path()
        # Densify
        if len(path) >= 2:
            path = self.create_more_vertexes(path, self.max_dist_vertexes)
        return ok, final_g, path, raw_path

    def check_if_path_in_free(self, path: List[np.ndarray]) -> Tuple[bool, List[np.ndarray]]:
        if len(path) < 2:
            return False, []
        out: List[np.ndarray] = [path[0].copy()]
        for p in path[1:]:
            if self.check_if_point_free(p):
                out.append(p.copy())
            else:
                break
        return len(out) >= 2, out

    def push_path_into_free_space(self, path: List[np.ndarray]) -> List[np.ndarray]:
        return [self.map_util.find_closest_free_point(p) for p in path]

    # ------------------------------------------------------------------
    def create_more_vertexes(self, path: List[np.ndarray], d: float) -> List[np.ndarray]:
        if len(path) < 2 or d <= 0.0:
            return [p.copy() for p in path]
        out: List[np.ndarray] = [path[0].copy()]
        for a, b in zip(path[:-1], path[1:]):
            seg = b - a
            L = float(np.linalg.norm(seg))
            if L <= 1e-9:
                continue
            n = int(np.floor(L / d))
            u = seg / L
            for k in range(1, n + 1):
                out.append(a + u * (k * d))
            if np.linalg.norm(out[-1] - b) > 1e-6:
                out.append(b.copy())
        return out

    # ------------------------------------------------------------------
    def cvx_ellipsoid_decomp(
        self,
        path: List[np.ndarray],
        base_uo: List[np.ndarray],
        obst_pos: List[np.ndarray],
        obst_bbox: List[np.ndarray],
        seg_end_times: List[float],
    ) -> Tuple[bool, List[Polytope]]:
        """Per-segment convex decomposition. Returns a list of Polytope (A,b)
        defining `A x <= b` for each segment between consecutive path
        vertices. The Python implementation uses a simple axis-aligned box
        clipped by the local SFC bbox plus per-obstacle half-spaces — it's a
        practical surrogate that the Gurobi QP can consume, not a perfect
        port of DecompROS2's iterative ellipsoid inflation.
        """
        polytopes: List[Polytope] = []
        if len(path) < 2:
            return False, polytopes
        sfc = np.array(self.sfc_size, dtype=np.float64) * 0.5

        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            mid = 0.5 * (a + b)
            # Local AABB centered on midpoint
            lo = mid - sfc
            hi = mid + sfc
            # Extend AABB to include the segment endpoints
            lo = np.minimum(lo, np.minimum(a, b) - 0.5 * self.res)
            hi = np.maximum(hi, np.maximum(a, b) + 0.5 * self.res)
            # Clip to map bounds
            lo[0] = max(lo[0], self.par.x_min)
            lo[1] = max(lo[1], self.par.y_min)
            lo[2] = max(lo[2], self.par.z_min)
            hi[0] = min(hi[0], self.par.x_max)
            hi[1] = min(hi[1], self.par.y_max)
            hi[2] = min(hi[2], self.par.z_max)
            # AABB → 6 half-planes
            A_list = []
            b_list = []
            for axis in range(3):
                e = np.zeros(3); e[axis] = 1.0
                A_list.append(e); b_list.append(hi[axis])
                A_list.append(-e); b_list.append(-lo[axis])

            # Per-obstacle half-spaces — cut the polytope away from the segment side
            te = seg_end_times[i] if i < len(seg_end_times) else 0.0
            for c, hk in zip(obst_pos, obst_bbox):
                infl = hk + self.par.obst_max_vel * te + self.drone_radius
                # Find the segment-side of the obstacle: direction from obstacle to midpoint
                dvec = mid - c
                n = np.linalg.norm(dvec)
                if n < 1e-9:
                    continue
                normal = dvec / n
                # Plane normal*x <= normal*c + (infl projected onto normal)
                proj = float(np.dot(np.abs(normal), infl))
                rhs = float(np.dot(normal, c)) + proj
                # Only add if the segment endpoints both satisfy the inequality
                lhs_a = float(np.dot(normal, a))
                lhs_b = float(np.dot(normal, b))
                if lhs_a > rhs + 0.05 or lhs_b > rhs + 0.05:
                    # Obstacle would clip our path — skip this cut (let the QP fail and replan)
                    continue
                if lhs_a < rhs - 0.0 and lhs_b < rhs - 0.0:
                    # The cut is on the obstacle side — flip it to keep the half containing the path
                    A_list.append(-normal); b_list.append(-rhs)

            # Static occupancy — coarse cull: cells inside the AABB get cut by their tightest plane to path mid
            # (kept simple here; the C++ decomp does an iterative ellipsoid inflation)
            for pt in base_uo:
                if np.any(pt < lo - self.drone_radius) or np.any(pt > hi + self.drone_radius):
                    continue
                dvec = mid - pt
                n = np.linalg.norm(dvec)
                if n < 1e-9:
                    continue
                normal = dvec / n
                rhs = float(np.dot(normal, pt)) + self.drone_radius
                if float(np.dot(normal, a)) > rhs or float(np.dot(normal, b)) > rhs:
                    continue
                A_list.append(-normal); b_list.append(-rhs)

            polytopes.append(Polytope(A=np.asarray(A_list, dtype=np.float64),
                                       b=np.asarray(b_list, dtype=np.float64)))
        return True, polytopes

    def cvx_ellipsoid_decomp_time_layered(
        self,
        path: List[np.ndarray],
        base_uo: List[np.ndarray],
        obst_pos: List[np.ndarray],
        obst_bbox: List[np.ndarray],
        time_end_times: List[float],
    ) -> Tuple[bool, List[List[Polytope]]]:
        """One spatial polytope set per time layer, inflated by each layer's t."""
        layers: List[List[Polytope]] = []
        for te in time_end_times:
            ok, polys = self.cvx_ellipsoid_decomp(path, base_uo, obst_pos, obst_bbox,
                                                  [te] * (len(path) - 1))
            if not ok:
                return False, []
            layers.append(polys)
        return True, layers

    # ------------------------------------------------------------------
    def get_occupied_cells(self) -> List[np.ndarray]:
        return self.map_util.cells_world(VAL_OCC)

    def get_free_cells(self) -> List[np.ndarray]:
        return self.map_util.cells_world(VAL_FREE)

    def is_map_initialized(self) -> bool:
        return self.map_initialized
