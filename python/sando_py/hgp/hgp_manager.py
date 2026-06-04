"""HGPManager — owns the map and orchestrates global planning + SFC.

Mirrors include/hgp/hgp_manager.hpp. Convex decomposition uses a pure-Python
ellipsoid → polyhedron routine since DecompROS2's C++ shared library isn't
trivially callable from Python; we expose the same data flow (polytope per
segment with `A x <= b`) so the Gurobi solver can plug in unchanged.

# ===== 中文说明 =====
# 这个文件是「全局段」对外的总管/门面(facade)。它持有那张体素地图(map_util)和
# 全局规划器(HGPPlanner),对上层暴露几个高层接口:
#   - set_parameters:把 yaml 里的一堆参数灌进地图和规划器;
#   - update_map:每个规划周期用新的点云/障碍重建一次地图;
#   - solve_hgp:跑一次全局规划,返回整理好的路径;
#   - cvx_ellipsoid_decomp*:把路径切成一段段「安全走廊」(凸多面体 A x <= b)。
# SFC = Safe Flight Corridor(安全飞行走廊):一连串凸多面体,把无人机限制在里面飞
#   就不会撞;后面的优化器(轨迹一定要待在这些走廊里)用它当硬约束。
# 注意:这里的凸分解是个纯 Python 的「够用版」近似(轴对齐盒子 + 逐障碍切半空间),
#   不是 C++ DecompROS2 那种精确的椭球迭代膨胀;但输出的数据格式(每段一个 A x <= b)
#   一样,所以下游 Gurobi 求解器不用改就能接上。
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from ..types import Parameters, Polytope
from .hgp_planner import HGPPlanner
from .voxel_map import VAL_FREE, VAL_OCC, VoxelMapUtil


class HGPManager:
    # 全局段总管。初始化用一份默认参数;真正的参数随后由 set_parameters 从 yaml 灌入。
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
        # 把整份 yaml 参数落到本类和地图上。注意:全局规划用的格子边长是 factor_hgp*res,
        # 通常比局部用的地图粗一点(全局只要大致走向,用粗格子搜得更快)。
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

    # 按当前参数新建全局规划器实例,并把地图绑上去
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
        # 每个规划周期重建一次地图。wdx/wdy/wdz = 地图三个方向的物理尺寸(米),
        # 这里换算成格数后交给 read_map 干活。第一次建图后顺手把规划器也建起来。
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
    # 把起点周围挖空一小块,确保无人机当前位置可作为搜索起点(否则可能卡在障碍里搜不出)
    def free_start(self, start: np.ndarray, factor: float) -> np.ndarray:
        d = factor * self.res
        self.map_util.set_free_voxel_and_surroundings(start, d)
        return start

    # 同理,把终点周围挖空一小块
    def free_goal(self, goal: np.ndarray, factor: float) -> np.ndarray:
        d = factor * self.res
        self.map_util.set_free_voxel_and_surroundings(goal, d)
        return goal

    # 查某个世界坐标点所在格是否被占
    def check_if_point_occupied(self, p: np.ndarray) -> bool:
        ix, iy, iz = self.map_util.float_to_int(p)
        return self.map_util.is_occupied(int(ix), int(iy), int(iz))

    # 查某个世界坐标点所在格是否空闲
    def check_if_point_free(self, p: np.ndarray) -> bool:
        ix, iy, iz = self.map_util.float_to_int(p)
        return self.map_util.is_free(int(ix), int(iy), int(iz))

    # ------------------------------------------------------------------
    def solve_hgp(
        self,
        start: np.ndarray, start_vel: np.ndarray, goal: np.ndarray,
        current_time: float,
    ) -> Tuple[bool, float, List[np.ndarray], List[np.ndarray]]:
        # 上层一次完整调用:跑全局规划 -> 拿到路径 -> 按 max_dist_vertexes 加密顶点。
        # 返回 (是否成功, 路径代价, 加密后的整理路径, 原始 A* 路径)。
        if self.planner is None:
            self.setup_planner()
        # Optional free-start / free-goal
        # 可选:先把起点/终点周围挖空,提高搜索成功率
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
        # 加密:瘦身后的路径点可能隔很远,按 max_dist_vertexes 在长段上补点,
        # 给后面的局部曲线优化更均匀的控制点
        if len(path) >= 2:
            path = self.create_more_vertexes(path, self.max_dist_vertexes)
        return ok, final_g, path, raw_path

    # 沿路径从头检查,返回「从起点起一直待在空闲空间」的那一前缀段(碰到第一个非空闲点就截断)
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

    # 把路径上每个点都拉到最近的空闲格中心(用于把卡进障碍的点救回来)
    def push_path_into_free_space(self, path: List[np.ndarray]) -> List[np.ndarray]:
        return [self.map_util.find_closest_free_point(p) for p in path]

    # ------------------------------------------------------------------
    # 在相邻两点间每隔 d 米插一个点,把路径采得更密(段长不足 d 的不动)
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

        中文:把路径每相邻两点之间的「一段」各围出一个凸多面体(安全走廊的一格),
        用一组线性不等式 A x <= b 表示「待在这个多面体内就安全」。
        做法(每段):以该段为中心拉一个轴对齐盒子(6 个面)-> 再针对每个动态障碍、
        每个静态障碍各切一刀(加一个半空间),把走廊从障碍那侧削掉。
        这是个工程上「够用」的近似,不是 C++ DecompROS2 那种精确椭球迭代膨胀,
        但输出格式一样,Gurobi 求解器能直接吃。
        """
        polytopes: List[Polytope] = []
        if len(path) < 2:
            return False, polytopes
        sfc = np.array(self.sfc_size, dtype=np.float64) * 0.5   # 走廊盒子的半尺寸

        for i in range(len(path) - 1):
            a = path[i]
            b = path[i + 1]
            mid = 0.5 * (a + b)
            # Local AABB centered on midpoint
            # 以这段中点为中心拉一个盒子
            lo = mid - sfc
            hi = mid + sfc
            # Extend AABB to include the segment endpoints
            # 撑大盒子保证两个端点都在里面(否则端点可能贴着盒壁/出界)
            lo = np.minimum(lo, np.minimum(a, b) - 0.5 * self.res)
            hi = np.maximum(hi, np.maximum(a, b) + 0.5 * self.res)
            # Clip to map bounds
            # 把盒子裁进地图边界
            lo[0] = max(lo[0], self.par.x_min)
            lo[1] = max(lo[1], self.par.y_min)
            lo[2] = max(lo[2], self.par.z_min)
            hi[0] = min(hi[0], self.par.x_max)
            hi[1] = min(hi[1], self.par.y_max)
            hi[2] = min(hi[2], self.par.z_max)
            # AABB → 6 half-planes
            # 盒子的 6 个面 -> 6 条不等式(每个轴上一个 <=hi、一个 >=lo)
            A_list = []
            b_list = []
            for axis in range(3):
                e = np.zeros(3); e[axis] = 1.0
                A_list.append(e); b_list.append(hi[axis])
                A_list.append(-e); b_list.append(-lo[axis])

            # Per-obstacle half-spaces — cut the polytope away from the segment side
            # 逐个动态障碍切一刀:在障碍和这段路之间加一个平面,把障碍那侧从走廊里切掉。
            # te = 这段对应的时间,障碍按「最大速度*te」往外胀(越晚的段越不确定,胀得越大),
            # 再加上无人机自身半径 drone_radius 当安全余量。
            te = seg_end_times[i] if i < len(seg_end_times) else 0.0
            for c, hk in zip(obst_pos, obst_bbox):
                infl = hk + self.par.obst_max_vel * te + self.drone_radius
                # Find the segment-side of the obstacle: direction from obstacle to midpoint
                # 切平面的法向 = 从障碍中心指向这段中点的方向(即「路在障碍的哪一侧」)
                dvec = mid - c
                n = np.linalg.norm(dvec)
                if n < 1e-9:
                    continue
                normal = dvec / n
                # Plane normal*x <= normal*c + (infl projected onto normal)
                # 把膨胀量投影到法向上,得到切平面的右端值 rhs
                proj = float(np.dot(np.abs(normal), infl))
                rhs = float(np.dot(normal, c)) + proj
                # Only add if the segment endpoints both satisfy the inequality
                # 只有当这段两个端点都不被这刀切掉时才加(否则障碍会切到路上)
                lhs_a = float(np.dot(normal, a))
                lhs_b = float(np.dot(normal, b))
                if lhs_a > rhs + 0.05 or lhs_b > rhs + 0.05:
                    # Obstacle would clip our path — skip this cut (let the QP fail and replan)
                    # 障碍会切到路:干脆不加这刀,让后面的 QP 自己失败再触发重规划
                    continue
                if lhs_a < rhs - 0.0 and lhs_b < rhs - 0.0:
                    # The cut is on the obstacle side — flip it to keep the half containing the path
                    # 取「包含路径的那半边」:翻一下符号变成 -normal*x <= -rhs 加进约束
                    A_list.append(-normal); b_list.append(-rhs)

            # Static occupancy — coarse cull: cells inside the AABB get cut by their tightest plane to path mid
            # (kept simple here; the C++ decomp does an iterative ellipsoid inflation)
            # 静态障碍点同理切刀:只看落在这段盒子附近的障碍点(远的跳过),
            # 在它和路之间加一个留出 drone_radius 余量的切平面;会切到端点的就不加。
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

            # 这一段攒齐所有约束,打包成一个多面体
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
        """One spatial polytope set per time layer, inflated by each layer's t.

        中文:按多个时间层各算一套走廊。每一层用自己的时间 te 去给动态障碍做膨胀
        (越晚的层障碍可能跑得越远,走廊就把那块多让出来)—— 这就是「时空走廊」,
        让轨迹在不同时刻避开动态障碍不同时刻的位置。
        """
        layers: List[List[Polytope]] = []
        for te in time_end_times:
            ok, polys = self.cvx_ellipsoid_decomp(path, base_uo, obst_pos, obst_bbox,
                                                  [te] * (len(path) - 1))
            if not ok:
                return False, []
            layers.append(polys)
        return True, layers

    # ------------------------------------------------------------------
    # 返回所有被占格的世界坐标(给可视化用)
    def get_occupied_cells(self) -> List[np.ndarray]:
        return self.map_util.cells_world(VAL_OCC)

    # 返回所有空闲格的世界坐标(给可视化用)
    def get_free_cells(self) -> List[np.ndarray]:
        return self.map_util.cells_world(VAL_FREE)

    # 地图是否已经建过(没建过别调用规划)
    def is_map_initialized(self) -> bool:
        return self.map_initialized
