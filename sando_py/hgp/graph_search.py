"""A* / heat-A* graph search over a 3D voxel map.

JPS is *not* ported — astar_heat (the default in sando.yaml) is the version
that matters for SANDO. sjps and sastar fall back to plain A* with use_heat=False.

# ===== 中文说明 =====
# 这个文件是「全局向导」的核心:在 voxel_map 那张栅格地图上跑 A* 搜索,找一条从
# 起点到终点、绕开障碍的折线路径(一串格坐标)。它是规划器的第一步,给后面的局部
# MINCO 曲线优化提供一个大致走向。
# 它的角色 = heat-A*:普通 A* 是「走最短」;heat-A* 在每步代价里额外加上 voxel_map
# 算好的「热度」软成本,所以会主动离障碍远一点、躲开动态障碍的预测路线 —— 这就是
# 「在正确时刻躲人」那条线在全局层面的体现。
# 没有移植 JPS(跳点搜索);只有 astar_heat 这个版本在 SANDO 里真正用到。
# 输入:起点/终点的格坐标、地图;输出:一串格坐标组成的路径(get_path_world 转成米)。
"""
from __future__ import annotations

import heapq
import time
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC


# 优先队列里的一项:按 f(=g+h 总估价)排序;counter 是入队序号,用来打破平局
# (f 相同时谁先入队谁先出,保证排序稳定);state_id 不参与比较只是带着走。
@dataclass(order=True)
class _PQEntry:
    f: float
    counter: int
    state_id: int = field(compare=False)


# A* 里一个搜索节点(对应地图里一个格)。
# g = 从起点到这格已花的真实代价;h = 到终点的启发估计;parent_id = 从哪格走过来的;
# dx/dy/dz = 走到这格的方向;opened/closed = A* 的开/闭集合标记。
@dataclass
class State:
    id: int
    x: int
    y: int
    z: int
    dx: int = 0
    dy: int = 0
    dz: int = 0
    g: float = float("inf")
    h: float = 0.0
    parent_id: int = -1
    opened: bool = False
    closed: bool = False


# 26-connected neighbor offsets (skip the zero motion).
# 26 邻接:一个格周围 3x3x3 减去自己 = 26 个方向(含斜向/对角),搜索时往这些方向扩展
_NEIGHBORS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


class GraphSearch:
    # A* 搜索器。
    # eps = 启发权重(>1 时是加权 A*,搜得更快但不一定最优);
    # use_heat 由 global_planner=="astar_heat" 决定(是否把热度软成本算进代价);
    # use_soft_cost 决定障碍是「软」(加大代价但可穿)还是「硬」(直接不能走)。
    def __init__(
        self,
        map_util: VoxelMapUtil,
        eps: float = 1.0,
        global_planner: str = "astar_heat",
        w_unknown: float = 0.0,
        w_align: float = 0.0,
        decay_len_cells: float = 20.0,
        w_side: float = 0.2,
        verbose: bool = False,
        heat_weight: float = 10.0,
        obstacle_soft_cost: float = 5.0,
    ):
        self.map_util = map_util
        self.eps = float(eps)
        self.global_planner = global_planner
        self.w_unknown = w_unknown
        self.w_align = w_align
        self.decay_len_cells = decay_len_cells
        self.w_side = w_side
        self.verbose = verbose
        self.heat_weight = heat_weight
        self.obstacle_soft_cost = obstacle_soft_cost
        self.use_heat = (global_planner == "astar_heat")
        self.use_soft_cost = map_util.use_soft_cost_obstacles
        self._hm: List[Optional[State]] = []
        self._path: List[State] = []
        self.start_vel = np.zeros(3)
        self.goal_int = np.zeros(3, dtype=np.int64)
        self.last_metrics = dict(global_planning_time=0.0,
                                  hgp_check_path_time=0.0,
                                  hgp_dynamic_astar_time=0.0,
                                  hgp_recover_path_time=0.0)

    # ------------------------------------------------------------------
    # 格坐标 -> 一维 id(直接复用地图的 lin_index,所以 id 就是地图下标)
    def _coord_to_id(self, x: int, y: int, z: int) -> int:
        return self.map_util.lin_index(x, y, z)

    # 启发函数 h:到终点的欧氏距离 * eps(A* 用它估「还剩多远」来引导搜索方向)
    def _heur(self, x: int, y: int, z: int) -> float:
        gx, gy, gz = self.goal_int
        dx, dy, dz = x - gx, y - gy, z - gz
        return self.eps * float(np.sqrt(dx * dx + dy * dy + dz * dz))

    # 取这格对应的 State(没有就懒创建一个并缓存进 _hm)。_hm 是一维数组当哈希表用。
    def _state_at(self, x: int, y: int, z: int) -> State:
        sid = self._coord_to_id(x, y, z)
        s = self._hm[sid]
        if s is None:
            s = State(id=sid, x=x, y=y, z=z, h=self._heur(x, y, z))
            self._hm[sid] = s
        return s

    # 这格是否被硬障碍占住(出界也算被占)
    def _is_blocked(self, x: int, y: int, z: int) -> bool:
        if not self.map_util.in_bounds(x, y, z):
            return True
        return self.map_util.cmap[self._coord_to_id(x, y, z)] == VAL_OCC

    def _corner_cut(self, x: int, y: int, z: int, dx: int, dy: int, dz: int) -> bool:
        """No corner-cutting: a diagonal step must not graze an occupied (or
        out-of-bounds) cell. Checks every shoulder cell formed by a non-empty
        proper subset of the move's nonzero components, so the straight line
        between consecutive cell centers only ever touches free space.

        中文:禁止「抄近角」。走斜向(对角)一步时,如果它紧贴的「肩膀格」是障碍,
        就不允许这步 —— 否则路径会从两个障碍的缝里斜穿过去,看着合法实则贴边/穿角。
        做法:把这步非零分量的各个真子集对应的相邻格都查一遍,有障碍就返回 True 拦下。
        """
        nz = [(i, d) for i, d in enumerate((dx, dy, dz)) if d != 0]
        if len(nz) <= 1:
            return False
        for r in range(1, len(nz)):
            for combo in combinations(nz, r):
                s = [0, 0, 0]
                for i, d in combo:
                    s[i] = d
                if self._is_blocked(x + s[0], y + s[1], z + s[2]):
                    return True
        return False

    def plan(
        self,
        start_int: np.ndarray,
        goal_int: np.ndarray,
        initial_g: float,
        start_vel: np.ndarray,
        max_expand: int = 100000,
        timeout_ms: int = 1000,
    ) -> bool:
        # 中文:跑一次 A* 搜索。返回是否找到通往终点的路。
        # initial_g = 起点格的起始代价(由调用方塞入,通常是真实起点到格中心的偏差);
        # max_expand/timeout_ms = 兜底:展开太多节点或超时就停,返回目前最接近终点的路。
        t0 = time.perf_counter()
        self.goal_int = np.asarray(goal_int, dtype=np.int64)
        self.start_vel = np.asarray(start_vel, dtype=np.float64)
        size = self.map_util.total_size()
        self._hm = [None] * size
        self._path = []

        sx, sy, sz = (int(v) for v in start_int)
        gx, gy, gz = (int(v) for v in goal_int)
        # 起点或终点出界就直接失败
        if not self.map_util.in_bounds(sx, sy, sz) or not self.map_util.in_bounds(gx, gy, gz):
            return False

        start_state = self._state_at(sx, sy, sz)
        start_state.g = initial_g
        start_state.opened = True
        goal_state = self._state_at(gx, gy, gz)

        # pq = 开放集(小顶堆),每次弹出 f 最小的节点来展开
        pq: List[_PQEntry] = []
        counter = 0
        heapq.heappush(pq, _PQEntry(start_state.g + start_state.h, counter, start_state.id))
        counter += 1

        # best_node:记下搜索过程中离终点最近的节点,万一搜不到终点就退而求其次返回它
        best_node = start_state
        best_h = start_state.h
        deadline = t0 + timeout_ms / 1000.0
        expansions = 0
        success = False

        while pq:
            top = heapq.heappop(pq)
            curr = self._hm[top.state_id]
            # 堆里可能有同一节点的旧的、过时的条目;已关闭的直接跳过(惰性删除)
            if curr is None or curr.closed:
                continue
            curr.closed = True
            if curr.h < best_h:
                best_h = curr.h
                best_node = curr
            # 弹到终点 = 成功
            if curr.id == goal_state.id:
                success = True
                break
            expansions += 1
            # 兜底:展开次数或时间超限就停(用当前最佳节点收尾)
            if expansions > max_expand:
                break
            if time.perf_counter() > deadline:
                break

            # 26-connected expansion
            # 向 26 个邻居方向扩展
            for dx, dy, dz in _NEIGHBORS:
                nx, ny, nz = curr.x + dx, curr.y + dy, curr.z + dz
                if not self.map_util.in_bounds(nx, ny, nz):
                    continue
                val = self.map_util.cmap[self._coord_to_id(nx, ny, nz)]
                # 硬障碍模式:被占的格直接不能进
                if val == VAL_OCC and not self.use_soft_cost:
                    continue
                # No corner-cutting through obstacles (only when OCC hard-blocks).
                # 硬障碍模式下还要禁止斜抄障碍的角
                if not self.use_soft_cost and self._corner_cut(curr.x, curr.y, curr.z, dx, dy, dz):
                    continue
                child = self._state_at(nx, ny, nz)
                if child.closed:
                    continue
                # 这步的基础代价 = 走的几何距离(直邻=1,斜向=√2/√3)
                step = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                cost = step
                # 走「未知」格额外加罚(鼓励走看得见的已知空间)
                if val == -1 and self.w_unknown > 0:
                    cost += self.w_unknown * step
                # heat-A* 的关键:把热度软成本加进代价 —— 越靠近障碍/动态障碍预测路线越贵
                if self.use_heat:
                    h_val = self.map_util.get_heat(nx, ny, nz)
                    if h_val > 0 and self.heat_weight > 0:
                        cost += self.heat_weight * h_val
                    # 软障碍模式:被占的格不是禁入,而是收一笔很大的「过路费」(可穿但极不划算)
                    if val == VAL_OCC and self.use_soft_cost:
                        cost += self.heat_weight * self.obstacle_soft_cost
                # 标准 A* 松弛:如果经 curr 到 child 更便宜,就更新 child 并重新入队
                tentative_g = curr.g + cost
                if tentative_g < child.g:
                    child.g = tentative_g
                    child.parent_id = curr.id
                    child.dx, child.dy, child.dz = dx, dy, dz
                    heapq.heappush(pq, _PQEntry(tentative_g + child.h, counter, child.id))
                    counter += 1
                    child.opened = True

        # Recover path
        # 回溯路径:成功就从终点回溯,否则从「最接近终点」的节点回溯
        t_rec = time.perf_counter()
        end_node = goal_state if success else best_node
        # If start==goal, treat as success
        # 起点就是终点的特殊情况,当成功处理
        if end_node is start_state and not success:
            success = True
        self._path = self._recover_path(end_node, start_state)
        self.last_metrics["global_planning_time"] = time.perf_counter() - t0
        self.last_metrics["hgp_recover_path_time"] = time.perf_counter() - t_rec
        return success and len(self._path) >= 1

    # 顺着 parent_id 从终点一路回溯到起点,再翻转成「起点->终点」顺序
    def _recover_path(self, end: State, start: State) -> List[State]:
        path: List[State] = []
        cur = end
        while cur is not None:
            path.append(cur)
            if cur is start or cur.parent_id < 0:
                break
            cur = self._hm[cur.parent_id]
            if cur is None:
                break
        path.reverse()
        return path

    # 把搜出来的格坐标路径转成世界坐标(米),供后续模块使用
    def get_path_world(self) -> List[np.ndarray]:
        return [self.map_util.int_to_float(np.array([s.x, s.y, s.z])) for s in self._path]
