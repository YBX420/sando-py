"""HGPPlanner — orchestrates voxel map search, LOS shortcut, and path
simplification. Wraps `GraphSearch` and applies the same post-processing
pipeline as `hgp_planner.cpp`.

# ===== 中文说明 =====
# 这个文件把「全局向导」串成一条完整流水线:调 GraphSearch 跑 A* 搜出原始折线路径,
# 然后做一连串「路径瘦身」把它整理干净再交出去。它的角色 = 全局段的总装配。
# 瘦身流水线(按顺序):
#   1) 视线捷径 short_cut_by_los —— 能直连就跳过中间拐点;
#   2) collapse_short_edges —— 合并太短的小段;
#   3) angle_spacing_filter —— 删掉拐得不够大的多余拐点;
#   4) 再来一次视线捷径;
#   5) clean_up_path —— 去共线点、去多余拐角;
#   6) _repair_blocked_segments —— 安全网:万一某段被瘦身后穿了障碍,把原始 A* 的
#      那一小段补回去(因为原始 A* 路径是逐格相邻、保证不撞的)。
# 输入:起点/终点(米)、当前时间;输出:整理好的路径 path(一串世界坐标点)。
# 注意:所有瘦身步骤都拿地图的 is_blocked 当「这段能不能直连」的硬判据,绝不简化出
#   一条穿障碍的捷径。
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

    # 绑定一张地图,并据此新建一个 A* 搜索器(把各种权重/参数透传进去)
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
        """Run search; return (success, final_g).

        中文:跑全局规划全流程。返回 (是否成功, 整理后路径的代价 final_g)。
        步骤:把起终点的米坐标转成格坐标 -> 调 A* -> 取原始路径 -> 路径瘦身流水线。
        """
        if self.map_util is None or not self.map_util.initialized:
            return False, 0.0

        t0 = time.perf_counter()
        self.graph_search.eps = float(eps)
        start_int = self.map_util.float_to_int(start)
        goal_int = self.map_util.float_to_int(goal)
        # initial_g:真实起点到「起点所在格中心」的距离偏差,作为 A* 起始代价补偿
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
        # 把首尾两点换成真实的起点/终点世界坐标(A* 给的是格中心,会有半格误差)
        raw_path[0] = start.copy()
        raw_path[-1] = goal.copy() if ok else raw_path[-1]
        self.raw_path = raw_path

        # Post-processing pipeline. The point-dropping filters get the map's
        # blocked-check so they never simplify a segment through an obstacle.
        # 路径瘦身流水线(详见文件顶说明)。所有删点滤波都拿 is_blocked 当硬判据,
        # 保证不会把一段简化成穿过障碍的捷径。
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
        # 视线捷径:从当前点 i 出发,尽量找最远的、还能「直线看得见(无障碍)」的点 j 直连,
        # 跳过中间所有拐点。这样把锯齿状折线拉直,路径更短更顺。
        if len(points) <= 2:
            return [p.copy() for p in points]
        out = [points[0].copy()]
        i = 0
        n = len(points)
        while i < n - 1:
            # 从最远点往回试,第一个能直连的 j 就是这一段能跳到的最远点
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
        # 收尾清理:先去掉共线的多余点,再正向、反向各做一遍去拐角,让结果不依赖方向
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

        中文:安全网,保证最终路径没有任何一段穿过障碍。原始 A* 路径逐格相邻、绝对
        不撞;若简化后某段被判定撞障碍,就把原始路径里对应那一小段补回去顶替它。
        """
        if len(path) < 2 or len(raw) < 2:
            return path

        # 在原始路径里找离 pt 最近的那个顶点的下标
        def nearest_raw(pt: np.ndarray) -> int:
            return min(range(len(raw)), key=lambda r: float(np.linalg.norm(raw[r] - pt)))

        out = [path[0].copy()]
        for k in range(len(path) - 1):
            a, b = path[k], path[k + 1]
            # 这一段被障碍挡住:把原始路径里 a、b 之间的那些点插回来绕开
            if self.map_util.is_blocked(a, b):
                ia, ib = nearest_raw(a), nearest_raw(b)
                if ia < ib:
                    for q in raw[ia + 1:ib]:
                        out.append(q.copy())
            out.append(b.copy())
        return out

    def _remove_corner_pts(self, points: List[np.ndarray]) -> List[np.ndarray]:
        # 去拐角:试着把中间点 mid 删掉、让 out[-1] 直连 nxt。只有当「直连更划算」
        # (代价更小)且「直连不会明显更靠近障碍」(峰值热度没涨太多)时才删,
        # 否则保留 mid。这样既拉直路径又不会为了抄近道而贴障碍。
        if len(points) <= 2:
            return [p.copy() for p in points]
        out = [points[0].copy()]
        prev_cost = self._seg_cost(out[-1], points[1])
        prev_peak = self._seg_peak_heat(out[-1], points[1])
        i = 1
        while i < len(points) - 1:
            mid = points[i]
            nxt = points[i + 1]
            cost2 = self._seg_cost(mid, nxt)        # 保留 mid 时后半段代价
            peak2 = self._seg_peak_heat(mid, nxt)
            cost3 = self._seg_cost(out[-1], nxt)    # 删掉 mid 直连的代价
            peak3 = self._seg_peak_heat(out[-1], nxt)
            # 直连更省 且 不会更贴障碍(峰值热度不超过原来的 1.5 倍)-> 删掉 mid
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
        # 一段 a->b 的「带热度代价」= 段长 + heat_weight * (沿线积分的热度)。
        # 撞障碍的段直接给无穷大(等于禁止)。这是和 A* 代价同一套思路,用于路径简化的取舍。
        if self.map_util.is_blocked(a, b):
            return float("inf")
        L = float(np.linalg.norm(b - a))
        if L < 1e-9:
            return 0.0
        # 沿线按约半格的步长采样,把热度近似积分(累加 热度*小段长)
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
        # 一段 a->b 上的「最高热度」(沿线采样取最大)。用来判断这段最贴障碍的地方有多危险。
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

    # 整条路径的纯几何长度(不含热度),作为返回给上层的 final_g
    def _path_cost(self, path: List[np.ndarray]) -> float:
        total = 0.0
        for a, b in zip(path[:-1], path[1:]):
            total += float(np.linalg.norm(b - a))
        return total

    def get_path(self) -> List[np.ndarray]:
        return self.path

    def get_raw_path(self) -> List[np.ndarray]:
        return self.raw_path
