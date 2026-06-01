"""A* / heat-A* graph search over a 3D voxel map.

JPS is *not* ported — astar_heat (the default in sando.yaml) is the version
that matters for SANDO. sjps and sastar fall back to plain A* with use_heat=False.
"""
from __future__ import annotations

import heapq
import time
from itertools import combinations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .voxel_map import VoxelMapUtil, VAL_FREE, VAL_OCC


@dataclass(order=True)
class _PQEntry:
    f: float
    counter: int
    state_id: int = field(compare=False)


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
_NEIGHBORS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


class GraphSearch:
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
    def _coord_to_id(self, x: int, y: int, z: int) -> int:
        return self.map_util.lin_index(x, y, z)

    def _heur(self, x: int, y: int, z: int) -> float:
        gx, gy, gz = self.goal_int
        dx, dy, dz = x - gx, y - gy, z - gz
        return self.eps * float(np.sqrt(dx * dx + dy * dy + dz * dz))

    def _state_at(self, x: int, y: int, z: int) -> State:
        sid = self._coord_to_id(x, y, z)
        s = self._hm[sid]
        if s is None:
            s = State(id=sid, x=x, y=y, z=z, h=self._heur(x, y, z))
            self._hm[sid] = s
        return s

    def _is_blocked(self, x: int, y: int, z: int) -> bool:
        if not self.map_util.in_bounds(x, y, z):
            return True
        return self.map_util.cmap[self._coord_to_id(x, y, z)] == VAL_OCC

    def _corner_cut(self, x: int, y: int, z: int, dx: int, dy: int, dz: int) -> bool:
        """No corner-cutting: a diagonal step must not graze an occupied (or
        out-of-bounds) cell. Checks every shoulder cell formed by a non-empty
        proper subset of the move's nonzero components, so the straight line
        between consecutive cell centers only ever touches free space.
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
        t0 = time.perf_counter()
        self.goal_int = np.asarray(goal_int, dtype=np.int64)
        self.start_vel = np.asarray(start_vel, dtype=np.float64)
        size = self.map_util.total_size()
        self._hm = [None] * size
        self._path = []

        sx, sy, sz = (int(v) for v in start_int)
        gx, gy, gz = (int(v) for v in goal_int)
        if not self.map_util.in_bounds(sx, sy, sz) or not self.map_util.in_bounds(gx, gy, gz):
            return False

        start_state = self._state_at(sx, sy, sz)
        start_state.g = initial_g
        start_state.opened = True
        goal_state = self._state_at(gx, gy, gz)

        pq: List[_PQEntry] = []
        counter = 0
        heapq.heappush(pq, _PQEntry(start_state.g + start_state.h, counter, start_state.id))
        counter += 1

        best_node = start_state
        best_h = start_state.h
        deadline = t0 + timeout_ms / 1000.0
        expansions = 0
        success = False

        while pq:
            top = heapq.heappop(pq)
            curr = self._hm[top.state_id]
            if curr is None or curr.closed:
                continue
            curr.closed = True
            if curr.h < best_h:
                best_h = curr.h
                best_node = curr
            if curr.id == goal_state.id:
                success = True
                break
            expansions += 1
            if expansions > max_expand:
                break
            if time.perf_counter() > deadline:
                break

            # 26-connected expansion
            for dx, dy, dz in _NEIGHBORS:
                nx, ny, nz = curr.x + dx, curr.y + dy, curr.z + dz
                if not self.map_util.in_bounds(nx, ny, nz):
                    continue
                val = self.map_util.cmap[self._coord_to_id(nx, ny, nz)]
                if val == VAL_OCC and not self.use_soft_cost:
                    continue
                # No corner-cutting through obstacles (only when OCC hard-blocks).
                if not self.use_soft_cost and self._corner_cut(curr.x, curr.y, curr.z, dx, dy, dz):
                    continue
                child = self._state_at(nx, ny, nz)
                if child.closed:
                    continue
                step = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                cost = step
                if val == -1 and self.w_unknown > 0:
                    cost += self.w_unknown * step
                if self.use_heat:
                    h_val = self.map_util.get_heat(nx, ny, nz)
                    if h_val > 0 and self.heat_weight > 0:
                        cost += self.heat_weight * h_val
                    if val == VAL_OCC and self.use_soft_cost:
                        cost += self.heat_weight * self.obstacle_soft_cost
                tentative_g = curr.g + cost
                if tentative_g < child.g:
                    child.g = tentative_g
                    child.parent_id = curr.id
                    child.dx, child.dy, child.dz = dx, dy, dz
                    heapq.heappush(pq, _PQEntry(tentative_g + child.h, counter, child.id))
                    counter += 1
                    child.opened = True

        # Recover path
        t_rec = time.perf_counter()
        end_node = goal_state if success else best_node
        # If start==goal, treat as success
        if end_node is start_state and not success:
            success = True
        self._path = self._recover_path(end_node, start_state)
        self.last_metrics["global_planning_time"] = time.perf_counter() - t0
        self.last_metrics["hgp_recover_path_time"] = time.perf_counter() - t_rec
        return success and len(self._path) >= 1

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

    def get_path_world(self) -> List[np.ndarray]:
        return [self.map_util.int_to_float(np.array([s.x, s.y, s.z])) for s in self._path]
