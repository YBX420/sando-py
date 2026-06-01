"""SANDO planner — Python port of include/sando/sando.hpp + src/sando/sando.cpp.

State machine wrapping HGPManager (global path planner + safety corridors) and
SolverGurobi (local trajectory QP). Method names mirror the C++ class so audits
stay mechanical.

Single-process, threaded mode parity caveats:
- The C++ planner runs the per-factor Gurobi solves with std::async, with one
  solver instance per factor. We default to a sequential factor sweep here
  because gurobipy releases the GIL and is rate-limited by the solver, not by
  Python; the parallel speedup would be marginal without writing C extension
  code. The factor adaptation loop (success → re-center, failure → shift up)
  is identical.
- KD-trees from PCL → we use scipy.spatial.cKDTree (loaded lazily).
- The full C++ DecompROS2 ellipsoid decomposition is replaced by the simplified
  surrogate already in HGPManager.cvx_ellipsoid_decomp(). Output API is the
  same so the solver consumes both without changes.
"""
from __future__ import annotations

import math
import time
from collections import deque
from threading import Lock
from typing import Deque, List, Optional, Tuple

import numpy as np

from .hgp.hgp_manager import HGPManager
from .local.avoid_config import default_config as _avoid_default_config
from .local.local_opt import plan_minco, DetourConfig
from .local.minco import MinjerkTraj
from .local.obstacles import AABBObstacle, SphereObstacle
from .solver_gurobi import SolverGurobi
from .types import (
    BasisConverter,
    DroneStatus,
    DynTraj,
    Parameters,
    PieceWisePol,
    Polytope,
    RobotState,
)
from .utils import (
    angle_wrap,
    clamp,
    create_more_vertexes,
    min_time_double_integrator_3d,
    project_point_to_sphere,
    transform_inverse_se3,
    transform_stamped_to_matrix,
)


_RETURN_LAST_VERTEX = 0
_RETURN_INTERSECTION = 1


class QuinticPieceWisePol(PieceWisePol):
    """PieceWisePol carrying degree-5 ASCENDING coefficients (6 per segment).

    The base PieceWisePol is a CUBIC, DESCENDING (a u^3 + b u^2 + c u + d)
    polynomial. The MINCO backbone is a QUINTIC stored ASCENDING
    (c0 + c1 u + ... + c5 u^5). Those two facts (degree 5 vs 3, ascending vs
    descending) make a plain PieceWisePol impossible, so we subclass and
    override ONLY _eval_axis to use a degree-5 ascending power-basis evaluator
    identical to MinjerkTraj._basis / eval_deriv (same `**` power basis, so we
    get bit-parity with the source trajectory). eval / velocity / acceleration /
    clear / get_duration / get_end_time are inherited unchanged (they read only
    .times or call _eval_axis with order 0/1/2).
    """

    def _eval_axis(self, coeffs, t: float, order: int) -> float:
        if not self.times or not coeffs:
            return 0.0
        if t >= self.times[-1]:
            u = self.times[-1] - self.times[-2]
            return _poly5_deriv_ascending(coeffs[-1], u, order)
        if t < self.times[0]:
            return _poly5_deriv_ascending(coeffs[0], 0.0, order)
        for i in range(len(self.times) - 1):
            if self.times[i] <= t < self.times[i + 1]:
                u = t - self.times[i]
                return _poly5_deriv_ascending(coeffs[i], u, order)
        return 0.0


def _poly5_deriv_ascending(c: np.ndarray, u: float, order: int) -> float:
    """`order`-th derivative of a degree-5 ASCENDING polynomial at local u.

    p(u) = sum_{j=0..5} c[j] u^j. Mirrors MinjerkTraj._basis exactly:
      d^o/du^o (u^j) = (prod_{k<o}(j-k)) u^(j-o) for j>=o else 0.
    Uses the same `**` power basis as eval_deriv for bit-parity.
    """
    s = 0.0
    for j in range(order, 6):
        coeff = 1.0
        for k in range(order):
            coeff *= (j - k)
        s += coeff * float(c[j]) * (u ** (j - order))
    return s


def _minjerk_to_pwp(mj: MinjerkTraj, A_time: float) -> QuinticPieceWisePol:
    """Adapter: MinjerkTraj -> eval-equivalent QuinticPieceWisePol.

    Segment boundary wall-clock times are A_time + mj._cum (read DYNAMICALLY;
    plan_minco picks its own M). Per-segment coeffs are mj.c reshaped to
    (M,6,3), kept ASCENDING and copied verbatim. The local-time origins line
    up exactly: QuinticPieceWisePol local u = t - times[i] = (A_time+tau) -
    (A_time+_cum[i]) = tau == the MinjerkTraj local tau, so A_time cancels and
    there is no shift bug.
    """
    pwp = QuinticPieceWisePol()
    pwp.times = [A_time + float(mj._cum[i]) for i in range(mj.M + 1)]
    cseg = mj.c.reshape(mj.M, 6, 3)
    for i in range(mj.M):
        pwp.coeff_x.append(cseg[i, :, 0].copy())
        pwp.coeff_y.append(cseg[i, :, 1].copy())
        pwp.coeff_z.append(cseg[i, :, 2].copy())
    return pwp


class SANDO:
    """Top-level SANDO planner. Holds the state machine, dynamic-obstacle list,
    global path, latest plan deque, and timing/diagnostic counters. The C++
    class uses ~14 std::mutex instances; we collapse to one big lock — Python's
    GIL + the actual contention pattern (per-replan, not per-field) doesn't
    benefit from finer locks.
    """

    def __init__(self, par: Parameters):
        self.par = par

        # HGPManager and shared parameters
        self.hgp_manager = HGPManager()
        self.hgp_manager.set_parameters(par)

        # Factor list (time allocation sweep)
        if par.use_dynamic_factor:
            self.num_dynamic_factors = int(
                (2 * par.dynamic_factor_k_radius) / par.factor_constant_step_size
            ) + 1
            self.factors: List[float] = []
            for i in range(self.num_dynamic_factors):
                f = par.dynamic_factor_initial_mean - par.dynamic_factor_k_radius \
                    + i * par.factor_constant_step_size
                if par.factor_initial <= f <= par.factor_final:
                    self.factors.append(f)
        else:
            self.num_dynamic_factors = int(
                (par.factor_final - par.factor_initial) / par.factor_constant_step_size
            ) + 1
            self.factors = [par.factor_initial + i * par.factor_constant_step_size
                            for i in range(self.num_dynamic_factors)]

        # One solver instance per factor (Gurobi state is per-model so we
        # don't actually share between factor passes — but the C++ keeps them
        # warm to skip re-initialization).
        self.whole_traj_solver_ptrs: List[SolverGurobi] = []
        for _ in range(max(1, self.num_dynamic_factors)):
            s = SolverGurobi()
            s.initialize(par)
            self.whole_traj_solver_ptrs.append(s)

        # Precompute worst trajectory time (used as Th in obstacle horizon)
        tmp_solver = SolverGurobi()
        tmp_solver.initialize(par)
        tmp_start = RobotState()
        tmp_end = RobotState()
        tmp_end.set_pos(par.num_P * par.max_dist_vertexes, 0.0, 0.0)
        tmp_solver.set_X0(tmp_start)
        tmp_solver.set_Xf(tmp_end)
        self.worst_traj_time = tmp_solver.get_initial_dt() * par.num_N

        # MINVO basis (kept for visualization parity, not used inside core loop)
        bc = BasisConverter()
        self.A_rest_pos_basis = bc.A_pos_mv_rest.copy()
        self.A_rest_pos_basis_inverse = bc.A_pos_mv_rest_inv.copy()

        # Per-axis limit vectors
        self.v_max_3d = np.array([par.v_max, par.v_max, par.v_max])
        self.a_max_3d = np.array([par.a_max, par.a_max, par.a_max])
        self.j_max_3d = np.array([par.j_max, par.j_max, par.j_max])
        self.v_max = par.v_max
        self.max_dist_vertexes = par.max_dist_vertexes

        # Drone status
        self._drone_status = DroneStatus.GOAL_REACHED

        # Map size (recomputed every replan in computeMapSize)
        self.wdx = par.initial_wdx
        self.wdy = par.initial_wdy
        self.wdz = par.initial_wdz
        self.map_res = par.res
        self.map_center = np.zeros(3)
        self.map_size_initialized = False
        self.hgp_failure_count = 0

        # Flags
        self.state_initialized = False
        self.terminal_goal_initialized = False
        self.use_adapt_k_value = False
        self.kdtree_map_initialized = False
        self.kdtree_unk_initialized = False

        # Diagnostic data (mirrors retrieveData / retrievePolytopes API)
        self.final_g = 0.0
        self.global_planning_time = 0.0
        self.hgp_static_jps_time = 0.0
        self.hgp_check_path_time = 0.0
        self.hgp_dynamic_astar_time = 0.0
        self.hgp_recover_path_time = 0.0
        self.cvx_decomp_time = 0.0
        self.successful_factor = 0.0
        self.local_traj_computation_time = 0.0
        self.safe_paths_time = 0.0
        self.safety_check_time = 0.0
        self.yaw_sequence_time = 0.0
        self.yaw_fitting_time = 0.0
        self.poly_out_whole: List[Polytope] = []
        self.poly_out_safe: List[Polytope] = []
        self.goal_setpoints: List[RobotState] = []
        self.cps: List[np.ndarray] = []
        self.list_subopt_goal_setpoints: List[List[RobotState]] = []

        # Replanning state machine
        self.state = RobotState()
        self.G = RobotState()
        self.A = RobotState()
        self.A_time = 0.0
        self.E = RobotState()
        self.G_term = RobotState()
        self.plan: Deque[RobotState] = deque()
        self.plan_safe_paths: Deque[List[RobotState]] = deque()
        self.previous_yaw = 0.0
        self.prev_dyaw = 0.0
        self.dyaw_filtered = 0.0
        self.pwp_to_share = PieceWisePol()
        self.obst_pos: List[np.ndarray] = []
        self.obst_bbox: List[np.ndarray] = []
        self.obst_class: List[str] = []          # per-class tag, parallel to obst_pos
        self.obst_vel: List[np.ndarray] = []      # current obstacle velocity, parallel
        self.traj_max_time = 0.0

        # Yawing state
        self.yaw_start_pos = np.zeros(3)
        self.yaw_start_time = time.monotonic()
        self._t0_log = time.monotonic()

        # Hover avoidance
        self.p_hover = np.zeros(3)
        self.hover_avoidance_active = False

        # Counter for replanning failure
        self.replanning_failure_count = 0

        # Adaptive k_value
        self.num_replanning = 0
        self.got_enough_replanning = False
        self.k_value = 0
        self.store_computation_times: List[float] = []
        self.est_comp_time = 0.0

        # Dynamic obstacles
        self.trajs: List[DynTraj] = []

        # Point cloud snapshots (set via update_map_ptr)
        self.pclptr_map: Optional[np.ndarray] = None
        self.pclptr_unk: Optional[np.ndarray] = None
        self._kdtree_map = None
        self._kdtree_unk = None

        # Initial pose transform (hardware only)
        self.init_pose_transform = np.eye(4)
        self.init_pose_transform_inv = np.eye(4)
        self.init_pose_transform_rotation = np.eye(3)
        self.init_pose_transform_rotation_inv = np.eye(3)
        self.yaw_init_offset = 0.0
        self.init_pose_set = False

        # One coarse lock for everything (see class docstring)
        self._mtx = Lock()

    # ------------------------------------------------------------------
    # Status / lifecycle
    # ------------------------------------------------------------------
    def change_drone_status(self, new_status: int) -> None:
        if new_status == self._drone_status:
            return
        names = {
            DroneStatus.YAWING: "YAWING",
            DroneStatus.TRAVELING: "TRAVELING",
            DroneStatus.GOAL_SEEN: "GOAL_SEEN",
            DroneStatus.GOAL_REACHED: "GOAL_REACHED",
            DroneStatus.HOVER_AVOIDING: "HOVER_AVOIDING",
        }
        print(f"Changing DroneStatus from status_={names.get(int(self._drone_status), '?')} "
              f"to status_={names.get(int(new_status), '?')}")
        self._drone_status = new_status

    def get_drone_status(self) -> int:
        return int(self._drone_status)

    def get_hover_pos(self) -> np.ndarray:
        return self.p_hover.copy()

    def get_hover_avoidance_d_trigger(self) -> float:
        return self.par.hover_avoidance_d_trigger

    # ------------------------------------------------------------------
    # Thread-safe getters/setters mirroring the C++ class
    # ------------------------------------------------------------------
    def get_state(self) -> RobotState:
        with self._mtx:
            return self.state.clone()

    def get_gterm(self) -> RobotState:
        with self._mtx:
            return self.G_term.clone()

    def set_gterm(self, G_term: RobotState) -> None:
        with self._mtx:
            self.G_term = G_term.clone()

    def get_G(self) -> RobotState:
        with self._mtx:
            return self.G.clone()

    def set_G(self, G: RobotState) -> None:
        with self._mtx:
            self.G = G.clone()

    def get_E(self) -> RobotState:
        with self._mtx:
            return self.E.clone()

    def get_A(self) -> RobotState:
        with self._mtx:
            return self.A.clone()

    def set_A(self, A: RobotState) -> None:
        with self._mtx:
            self.A = A.clone()

    def get_A_time(self) -> float:
        with self._mtx:
            return self.A_time

    def set_A_time(self, A_time: float) -> None:
        with self._mtx:
            self.A_time = float(A_time)

    def get_last_plan_state(self) -> RobotState:
        with self._mtx:
            if self.plan:
                return self.plan[-1].clone()
        return self.get_state()

    def get_trajs(self) -> List[DynTraj]:
        with self._mtx:
            return list(self.trajs)

    def get_global_path(self) -> List[np.ndarray]:
        with self._mtx:
            return [p.copy() for p in self._global_path]

    def get_original_global_path(self) -> List[np.ndarray]:
        with self._mtx:
            return [p.copy() for p in self._original_global_path]

    def get_free_global_path(self) -> List[np.ndarray]:
        return [p.copy() for p in self._free_global_path]

    def get_pwp(self) -> PieceWisePol:
        return self.pwp_to_share

    def get_map_util_shared_ptr(self):
        return self.hgp_manager.map_util

    # ------------------------------------------------------------------
    # Internal lazy attrs (avoid AttributeError before first replan)
    # ------------------------------------------------------------------
    @property
    def _global_path(self) -> List[np.ndarray]:
        if not hasattr(self, "__global_path__"):
            self.__global_path__ = []
        return self.__global_path__

    @_global_path.setter
    def _global_path(self, value: List[np.ndarray]) -> None:
        self.__global_path__ = value

    @property
    def _original_global_path(self) -> List[np.ndarray]:
        if not hasattr(self, "__original_global_path__"):
            self.__original_global_path__ = []
        return self.__original_global_path__

    @_original_global_path.setter
    def _original_global_path(self, value: List[np.ndarray]) -> None:
        self.__original_global_path__ = value

    @property
    def _free_global_path(self) -> List[np.ndarray]:
        if not hasattr(self, "__free_global_path__"):
            self.__free_global_path__ = []
        return self.__free_global_path__

    @_free_global_path.setter
    def _free_global_path(self, value: List[np.ndarray]) -> None:
        self.__free_global_path__ = value

    # ------------------------------------------------------------------
    # Trajectory bookkeeping
    # ------------------------------------------------------------------
    def clean_up_old_trajs(self, current_time: float) -> None:
        with self._mtx:
            self.trajs = [t for t in self.trajs
                          if (current_time - t.time_received) <= self.par.traj_lifetime]

    def add_traj(self, new_traj: DynTraj, current_time: float) -> None:
        # Update existing by id first, no map/horizon check
        with self._mtx:
            for i, t in enumerate(self.trajs):
                if t.id == new_traj.id:
                    self.trajs[i] = new_traj
                    return
        # New trajectory — apply map + horizon filter
        p = new_traj.eval(current_time)
        if not self.check_point_within_map(p):
            return
        if float(np.linalg.norm(p - self.state.pos)) > self.par.horizon:
            return
        with self._mtx:
            self.trajs.append(new_traj)

    # ------------------------------------------------------------------
    # State update (called from odometry callback)
    # ------------------------------------------------------------------
    def update_state(self, data: RobotState) -> None:
        # Hardware-frame translation
        if self.par.use_hardware and self.par.provide_goal_in_global_frame \
                and not self.par.state_already_in_global_frame:
            homo = np.array([data.pos[0], data.pos[1], data.pos[2], 1.0])
            g = self.init_pose_transform @ homo
            data = data.clone()
            data.pos = g[:3]
            data.vel = self.init_pose_transform_rotation @ data.vel
            data.accel = self.init_pose_transform_rotation @ data.accel
            data.jerk = self.init_pose_transform_rotation @ data.jerk
            data.yaw += self.yaw_init_offset

        with self._mtx:
            self.state = data.clone()

        if (not self.state_initialized) or self._drone_status == DroneStatus.YAWING:
            tmp = RobotState()
            if self._drone_status == DroneStatus.YAWING:
                tmp.pos = self.yaw_start_pos.copy()
            else:
                tmp.pos = data.pos.copy()
            tmp.yaw = data.yaw

            if not self.state_initialized:
                self.previous_yaw = data.yaw

            with self._mtx:
                self.plan.clear()
                self.plan.append(tmp)
                self.A = tmp.clone()
                self.G = tmp.clone()

            self.state_initialized = True

    # ------------------------------------------------------------------
    # Map size / map updates
    # ------------------------------------------------------------------
    def compute_map_size(self, min_pos: np.ndarray, max_pos: np.ndarray) -> None:
        dynamic_buffer = self.par.map_buffer
        dist_x = abs(min_pos[0] - max_pos[0])
        dist_y = abs(min_pos[1] - max_pos[1])
        dist_z = abs(min_pos[2] - max_pos[2])
        self.wdx = max(dist_x + 2 * dynamic_buffer, self.par.min_wdx)
        self.wdy = max(dist_y + 2 * dynamic_buffer, self.par.min_wdy)
        self.wdz = max(dist_z + 2 * dynamic_buffer, self.par.min_wdz)
        self.map_center = (min_pos + max_pos) / 2.0

    def check_point_within_map(self, point: np.ndarray) -> bool:
        if not self.map_size_initialized and not hasattr(self, "_map_seen"):
            # Before the first replan we accept everything (matches C++ behavior
            # — the membership check is only used by add_traj which happens
            # after the first map update).
            return True
        return (abs(point[0] - self.map_center[0]) <= self.wdx / 2.0
                and abs(point[1] - self.map_center[1]) <= self.wdy / 2.0
                and abs(point[2] - self.map_center[2]) <= self.wdz / 2.0)

    def update_map_ptr(self, pclptr_map: Optional[np.ndarray],
                       pclptr_unk: Optional[np.ndarray]) -> None:
        with self._mtx:
            self.pclptr_map = pclptr_map
            self.pclptr_unk = pclptr_unk
        # Bootstrap the kdtree as soon as occ cloud arrives (chicken-and-egg
        # fix from C++: kdtree must be initialized before checkReadyToReplan
        # can pass, so we cannot wait for the first replan tick to build it).
        if pclptr_map is not None and len(pclptr_map) > 0 and not self.kdtree_map_initialized:
            self._build_kdtree_map(pclptr_map)
            self.kdtree_map_initialized = True

        if not self.hgp_manager.is_map_initialized():
            self.update_map(0.0)

    def update_occupancy_map_ptr(self, pclptr_map: Optional[np.ndarray]) -> None:
        """Occupancy-only variant used in fake_sim / rviz_only modes."""
        with self._mtx:
            self.pclptr_map = pclptr_map
        if not self.hgp_manager.is_map_initialized():
            self.update_occupancy_map(0.0)

    def _build_kdtree_map(self, points: np.ndarray) -> None:
        try:
            from scipy.spatial import cKDTree
            self._kdtree_map = cKDTree(points) if len(points) > 0 else None
        except ImportError:
            self._kdtree_map = None

    def _build_kdtree_unk(self, points: np.ndarray) -> None:
        try:
            from scipy.spatial import cKDTree
            self._kdtree_unk = cKDTree(points) if len(points) > 0 else None
        except ImportError:
            self._kdtree_unk = None

    def update_map(self, current_time: float) -> None:
        local_state = self.get_state()
        local_G = self.get_G()
        self.compute_map_size(local_state.pos, local_G.pos)

        obst_pos, obst_bbox, pred_samples, pred_times = [], [], [], []
        self.traj_max_time = self._compute_obst_pos_and_traj_max_time(
            obst_pos, obst_bbox, pred_samples, pred_times, current_time
        )

        # Forward predicted samples to the heat-map (used by VoxelMapUtil)
        if hasattr(self.hgp_manager.map_util, "set_dynamic_predicted_samples"):
            self.hgp_manager.map_util.set_dynamic_predicted_samples(pred_samples, pred_times)

        with self._mtx:
            pcl_map = self.pclptr_map
            pcl_unk = self.pclptr_unk

        self.hgp_manager.update_map(
            self.wdx, self.wdy, self.wdz, self.map_center,
            pcl_map if pcl_map is not None else np.zeros((0, 3)),
            pcl_unk if pcl_unk is not None else np.zeros((0, 3)),
            obst_pos, obst_bbox, self.traj_max_time,
        )
        self.map_size_initialized = True
        self._map_seen = True

        if pcl_map is not None and len(pcl_map) > 0:
            self._build_kdtree_map(pcl_map)
            self.kdtree_map_initialized = True
            self.hgp_manager.vec_o = [np.asarray(p) for p in pcl_map]

        if pcl_unk is not None and len(pcl_unk) > 0:
            self._build_kdtree_unk(pcl_unk)
            self.kdtree_unk_initialized = True
            uo = [np.asarray(p) for p in pcl_unk]
            if pcl_map is not None:
                uo = uo + [np.asarray(p) for p in pcl_map]
            self.hgp_manager.vec_uo = uo

    def update_occupancy_map(self, current_time: float) -> None:
        local_state = self.get_state()
        local_G = self.get_G()
        self.compute_map_size(local_state.pos, local_G.pos)

        obst_pos, obst_bbox, pred_samples, pred_times = [], [], [], []
        self.traj_max_time = self._compute_obst_pos_and_traj_max_time(
            obst_pos, obst_bbox, pred_samples, pred_times, current_time
        )
        if hasattr(self.hgp_manager.map_util, "set_dynamic_predicted_samples"):
            self.hgp_manager.map_util.set_dynamic_predicted_samples(pred_samples, pred_times)

        with self._mtx:
            pcl_map = self.pclptr_map

        self.hgp_manager.update_map(
            self.wdx, self.wdy, self.wdz, self.map_center,
            pcl_map if pcl_map is not None else np.zeros((0, 3)),
            np.zeros((0, 3)),
            obst_pos, obst_bbox, self.traj_max_time,
        )
        self.map_size_initialized = True
        self._map_seen = True

        if pcl_map is not None and len(pcl_map) > 0:
            self._build_kdtree_map(pcl_map)
            self.kdtree_map_initialized = True
            self.hgp_manager.vec_o = [np.asarray(p) for p in pcl_map]

    def _compute_obst_pos_and_traj_max_time(
        self, obst_pos: List[np.ndarray], obst_bbox: List[np.ndarray],
        pred_samples: List[List[np.ndarray]], pred_times: List[float],
        current_time: float,
    ) -> float:
        obst_pos.clear(); obst_bbox.clear(); pred_samples.clear(); pred_times.clear()
        local_trajs = self.get_trajs()

        # per-class snapshot tags (swap point for a real classifier): class keyed on
        # the DynTraj id range (200<=id<300 -> wall/SOFT, else human/HARD) and the
        # current obstacle velocity (so a moving human reaches plan_minco as a moving
        # SphereObstacle -> genuine dynamic, space-time avoidance).
        obst_class: List[str] = []
        obst_vel: List[np.ndarray] = []
        selected: List[DynTraj] = []
        for traj in local_trajs:
            p = traj.eval(current_time)
            if not self.check_point_within_map(p):
                continue
            dist = float(np.linalg.norm(p - self.state.pos))
            if dist > self.par.horizon:
                continue
            obst_pos.append(p)
            obst_bbox.append(np.array(traj.bbox, dtype=float))
            tid = int(getattr(traj, "id", 0))
            obst_class.append("wall" if 200 <= tid < 300 else "human")
            try:
                v = np.asarray(traj.velocity(current_time), dtype=float).reshape(3)
            except Exception:
                v = np.zeros(3)
            obst_vel.append(v)
            selected.append(traj)

        with self._mtx:
            self.obst_pos = [p.copy() for p in obst_pos]
            self.obst_bbox = [b.copy() for b in obst_bbox]
            self.obst_class = list(obst_class)
            self.obst_vel = [v.copy() for v in obst_vel]

        Th = self.worst_traj_time * (self.factors[-1] if self.factors else 1.0)
        if Th <= 0.0 or not selected:
            return Th

        dt = 0.5
        M = int(math.ceil(Th / dt)) + 1
        M = max(5, min(M, 10))
        for j in range(M):
            a = 0.0 if M == 1 else j / (M - 1)
            pred_times.append(float(a * Th))

        for k_idx, traj in enumerate(selected):
            samples_k: List[np.ndarray] = []
            for j in range(M):
                t_abs = current_time + pred_times[j]
                pk = traj.eval(t_abs)
                if not np.all(np.isfinite(pk)):
                    pk = traj.eval(current_time)
                samples_k.append(pk)
            pred_samples.append(samples_k)
        return Th

    # ------------------------------------------------------------------
    # HGP wrappers
    # ------------------------------------------------------------------
    def check_if_point_occupied(self, point: np.ndarray) -> bool:
        return self.hgp_manager.check_if_point_occupied(point)

    def check_if_point_free(self, point: np.ndarray) -> bool:
        return self.hgp_manager.check_if_point_free(point)

    # ------------------------------------------------------------------
    # G / horizon projection
    # ------------------------------------------------------------------
    def compute_G(self, A: RobotState, G_term: RobotState, horizon: float) -> None:
        local_G = RobotState()
        local_G.pos = project_point_to_sphere(A.pos, G_term.pos, horizon)
        d = G_term.pos - local_G.pos
        n = float(np.linalg.norm(d))
        if n > 1e-9:
            d = d / n
            local_G.yaw = math.atan2(d[1], d[0])
        self.set_G(local_G)

    # ------------------------------------------------------------------
    # Replan gating
    # ------------------------------------------------------------------
    def need_replan(self, local_state: RobotState, local_G_term: RobotState,
                    last_plan_state: RobotState) -> bool:
        dist_to_term_G = float(np.linalg.norm(local_state.pos - local_G_term.pos))
        dist_from_last_plan_state_to_term_G = float(np.linalg.norm(
            last_plan_state.pos - local_G_term.pos))
        vel_magnitude = float(np.linalg.norm(local_state.vel))
        max_goal_velocity = 0.1

        # Hover avoidance check first (matches C++)
        if self.par.hover_avoidance_enabled and (
                self._drone_status == DroneStatus.GOAL_REACHED
                or self._drone_status == DroneStatus.HOVER_AVOIDING):
            return True

        if dist_to_term_G < self.par.goal_radius and vel_magnitude < max_goal_velocity:
            if self.par.hover_avoidance_enabled:
                self.p_hover = local_G_term.pos.copy()
                self.change_drone_status(DroneStatus.HOVER_AVOIDING)
                return True
            self.change_drone_status(DroneStatus.GOAL_REACHED)
            self.p_hover = local_G_term.pos.copy()
            return False

        # Don't plan if drone is not traveling (YAWING / GOAL_REACHED)
        if self._drone_status == DroneStatus.GOAL_REACHED \
                or self._drone_status == DroneStatus.YAWING:
            return False

        if dist_to_term_G < self.par.goal_seen_radius:
            self.change_drone_status(DroneStatus.GOAL_SEEN)

        if (self._drone_status == DroneStatus.GOAL_SEEN
                and dist_from_last_plan_state_to_term_G < self.par.goal_radius):
            return False

        return True

    def check_ready_to_replan(self) -> bool:
        map_init = self.hgp_manager.is_map_initialized()
        kdtree_ok = (not self.par.use_hardware) or self.kdtree_map_initialized
        return self.state_initialized and self.terminal_goal_initialized \
            and map_init and kdtree_ok

    def goal_reached_check(self) -> bool:
        return self.check_ready_to_replan() and (
            self._drone_status == DroneStatus.GOAL_REACHED
            or self._drone_status == DroneStatus.HOVER_AVOIDING
        )

    # ------------------------------------------------------------------
    # findAandAtime — pick the start state of the next replan from the plan
    # ------------------------------------------------------------------
    def find_A_and_Atime(self, current_time: float,
                        last_replanning_computation_time: float
                        ) -> Tuple[bool, RobotState, float]:
        with self._mtx:
            plan_size = len(self.plan)
        if plan_size == 0:
            print("plan_size == 0")
            return False, RobotState(), 0.0

        if self.par.use_state_update:
            if not self.use_adapt_k_value:
                self.k_value = max(plan_size - self.par.default_k_value, 0)
                if self.num_replanning != 1:
                    self.store_computation_times.append(last_replanning_computation_time)
            else:
                a = self.par.alpha_k_value_filtering
                self.est_comp_time = a * last_replanning_computation_time \
                    + (1 - a) * self.est_comp_time
                self.k_value = max(
                    plan_size - int(self.par.k_value_factor * self.est_comp_time / self.par.dc),
                    0
                )

            if plan_size - 1 - self.k_value < 0 or plan_size - 1 - self.k_value >= plan_size:
                self.k_value = plan_size - 1

            with self._mtx:
                A = self.plan[plan_size - 1 - self.k_value].clone()
            A_time = current_time + (plan_size - 1 - self.k_value) * self.par.dc
        else:
            A = self.get_state()
            A_time = current_time

        if (A.pos[2] < self.par.z_min or A.pos[2] > self.par.z_max
                or A.pos[0] < self.par.x_min or A.pos[0] > self.par.x_max
                or A.pos[1] < self.par.y_min or A.pos[1] > self.par.y_max):
            print(f"A ({A.pos[0]:.3f}, {A.pos[1]:.3f}, {A.pos[2]:.3f}) is out of the map")
            return False, A, A_time
        return True, A, A_time

    # ------------------------------------------------------------------
    # findSafeSubGoal — truncate global path at unknown intersection
    # ------------------------------------------------------------------
    def find_safe_sub_goal(self, global_path: List[np.ndarray]) -> List[np.ndarray]:
        original = [p.copy() for p in global_path]
        if not original:
            return []
        out: List[np.ndarray] = [original[0].copy()]

        sample_dist = 0.1
        r_inflate = self.par.obst_max_vel * self.traj_max_time
        thr_orig = self.par.drone_radius
        thr_infl = self.par.drone_radius + r_inflate
        thr_orig2 = thr_orig * thr_orig
        thr_infl2 = thr_infl * thr_infl

        def is_within_unknown(pt: np.ndarray, thr2: float) -> bool:
            if self._kdtree_unk is None:
                return False
            d2, _ = self._kdtree_unk.query(pt, k=1)
            return float(d2 * d2) < thr2

        def backtrack(seg_i: int, s_hit: float) -> np.ndarray:
            i = max(0, min(seg_i, len(original) - 2))
            A = original[i]
            B = original[i + 1]
            d_vec = B - A
            L = float(np.linalg.norm(d_vec))
            if L < 1e-9:
                return A
            dir_vec = d_vec / L
            s = max(0.0, min(s_hit, L))
            pt = A + dir_vec * s
            if not is_within_unknown(pt, thr_infl2):
                return pt
            while True:
                s -= sample_dist
                if s >= 0.0:
                    pt = A + dir_vec * s
                else:
                    i -= 1
                    if i < 0:
                        return original[0].copy()
                    A = original[i]
                    B = original[i + 1]
                    d_vec = B - A
                    L = float(np.linalg.norm(d_vec))
                    if L < 1e-9:
                        s = 0.0
                        pt = A.copy()
                        continue
                    dir_vec = d_vec / L
                    s = L + s
                    s = max(0.0, min(s, L))
                    pt = A + dir_vec * s
                if not is_within_unknown(pt, thr_infl2):
                    return pt

        M = len(original)
        for i in range(M - 1):
            cur = original[i]
            nxt = original[i + 1]
            d_vec = nxt - cur
            dist = float(np.linalg.norm(d_vec))
            if dist < 1e-9:
                continue
            dir_vec = d_vec / dist
            num_samples = int(dist / sample_dist)
            for j in range(num_samples + 1):
                sp = cur + dir_vec * (sample_dist * j)
                if is_within_unknown(sp, thr_orig2):
                    s_hit = sample_dist * j
                    safe_pt = backtrack(i, s_hit)
                    if float(np.linalg.norm(safe_pt - out[-1])) > 1e-6:
                        out.append(safe_pt)
                    return out
            out.append(nxt.copy())
        return out

    # ------------------------------------------------------------------
    # Replan pipeline
    # ------------------------------------------------------------------
    def reset_data(self) -> None:
        self.final_g = 0.0
        self.global_planning_time = 0.0
        self.hgp_static_jps_time = 0.0
        self.hgp_check_path_time = 0.0
        self.hgp_dynamic_astar_time = 0.0
        self.hgp_recover_path_time = 0.0
        self.cvx_decomp_time = 0.0
        self.local_traj_computation_time = 0.0
        self.safe_paths_time = 0.0
        self.safety_check_time = 0.0
        self.yaw_sequence_time = 0.0
        self.yaw_fitting_time = 0.0
        self.poly_out_whole = []
        self.poly_out_safe = []
        self.goal_setpoints = []
        self.pwp_to_share.clear()
        self.cps = []

    def replan(self, last_replanning_computation_time: float,
               current_time: float) -> Tuple[bool, bool]:
        t_total = time.perf_counter()
        self.reset_data()

        if not self.check_ready_to_replan():
            if self.par.debug_verbose:
                print("Planner is not ready to replan")
            return False, False

        local_state = self.get_state()
        local_G_term = self.get_gterm()
        last_plan_state = self.get_last_plan_state()

        if not self.need_replan(local_state, local_G_term, last_plan_state):
            return False, False

        if self.par.hover_avoidance_enabled and (
                self._drone_status == DroneStatus.GOAL_REACHED
                or self._drone_status == DroneStatus.HOVER_AVOIDING):
            if not self.check_hover_avoidance(current_time):
                return False, False

        # Global planning
        ok, global_path = self.generate_global_path(current_time,
                                                    last_replanning_computation_time)
        if not ok:
            return False, False

        # Local trajectory
        if not self.plan_local_trajectory(global_path, last_replanning_computation_time):
            return False, True

        # Append to plan
        if not self.append_to_plan():
            return False, True

        if self.par.debug_verbose:
            print(f"\033[1;32mReplanning succeeded (factor={self.successful_factor:.2f})\033[0m")
        self.replanning_failure_count = 0
        _ = t_total  # for symmetry with C++ timer
        return True, True

    def generate_global_path(self, current_time: float,
                             last_replanning_computation_time: float
                             ) -> Tuple[bool, List[np.ndarray]]:
        local_G_term = self.get_gterm()

        ok, local_A, A_time = self.find_A_and_Atime(current_time,
                                                      last_replanning_computation_time)
        if not ok:
            self.replanning_failure_count += 1
            return False, []

        self.set_A(local_A)
        self.set_A_time(A_time)
        self.compute_G(local_A, local_G_term, self.par.horizon)

        if self.par.sim_env in ("fake_sim", "rviz_only"):
            self.update_occupancy_map(current_time)
        else:
            self.update_map(current_time)

        # Set up the HGP planner (idempotent)
        if self.hgp_manager.planner is None:
            self.hgp_manager.setup_planner()

        # Free start/goal
        if self.par.use_free_start:
            self.hgp_manager.free_start(local_A.pos, self.par.free_start_factor)
        if self.par.use_free_goal:
            self.hgp_manager.free_goal(self.get_G().pos, self.par.free_goal_factor)

        local_G = self.get_G()

        # Ground robots fix z
        if self.par.vehicle_type != "uav":
            local_A.pos[2] = 1.0
            local_G.pos[2] = 1.0

        # Direction hint from previous global path
        prev_global = self.get_global_path()
        if len(prev_global) >= 2 and float(np.linalg.norm(prev_global[1] - prev_global[0])) > 1e-8:
            dir_hint = (prev_global[1] - prev_global[0])
            dir_hint = dir_hint / float(np.linalg.norm(dir_hint))
        else:
            dir_hint = local_G.pos - local_A.pos
            n = float(np.linalg.norm(dir_hint))
            if n > 1e-8:
                dir_hint = dir_hint / n
            else:
                dir_hint = np.array([1.0, 0.0, 0.0])
        if self.par.vehicle_type != "uav":
            dir_hint[2] = 0.0

        # Solve HGP
        t0 = time.perf_counter()
        ok_hgp, final_g, global_path, raw_global_path = self.hgp_manager.solve_hgp(
            local_A.pos, dir_hint, local_G.pos, A_time,
        )
        self.global_planning_time = (time.perf_counter() - t0) * 1000.0
        if not ok_hgp:
            if self.par.debug_verbose:
                print("HGP did not find a solution")
            self.hgp_failure_count += 1
            self.replanning_failure_count += 1
            return False, []

        self.final_g = float(final_g)

        with self._mtx:
            self._global_path = [p.copy() for p in global_path]
            self._original_global_path = [p.copy() for p in raw_global_path]

        # Trim to (num_P + 1)
        if len(global_path) > self.par.num_P + 1:
            global_path = global_path[: self.par.num_P + 1]

        # Safe sub-goal truncation
        global_path = self.find_safe_sub_goal(global_path)
        if self.par.debug_verbose:
            print(f"global_path.size(): {len(global_path)}")
        return True, global_path

    # ------------------------------------------------------------------
    # MINCO local solve (per-class avoidance) — additive replacement body
    # ------------------------------------------------------------------
    def _obstacles_from_snapshot(self, obst_pos, obst_bbox, obst_class=None, obst_vel=None):
        """Class source hook: map the obstacle snapshot to per-class avoidance
        obstacles for plan_minco. obst_class[k] in {'human','wall'} decides the
        MECHANISM (the per-class point): human -> SphereObstacle (HARD, carries
        its current velocity so the Stage-4 space-time ALM avoids it at the right
        moment); wall -> AABBObstacle (SOFT EGO field, grazeable).

        The class tag is currently keyed on the DynTraj id range upstream
        (_compute_obst_pos_and_traj_max_time); a real RGBD/tracker classifier
        swaps in there. Missing tag -> 'human'/HARD (fail-safe), missing vel -> 0.

        obst_pos[k] = (3,) box CENTER; obst_bbox[k] = (3,) FULL box extents.
        """
        n = len(obst_pos)
        classes = list(obst_class) if obst_class is not None else []
        vels = list(obst_vel) if obst_vel is not None else []
        obstacles = []
        for k in range(n):
            c = np.asarray(obst_pos[k], dtype=np.float64).copy()
            sz = np.asarray(obst_bbox[k], dtype=np.float64)
            cls = classes[k] if k < len(classes) else "human"
            if cls == "wall":
                obstacles.append(AABBObstacle(
                    lo=c - 0.5 * sz, hi=c + 0.5 * sz, class_name="wall"))
            else:
                vel = (np.asarray(vels[k], dtype=np.float64).reshape(3)
                       if k < len(vels) else np.zeros(3))
                obstacles.append(SphereObstacle(
                    centre0=c, radius=0.5 * float(np.max(sz)),
                    vel=vel, class_name="human"))
        avoid_cfg = _avoid_default_config()
        return obstacles, avoid_cfg

    def plan_local_trajectory_minco(self, global_path: List[np.ndarray],
                                    last_replanning_computation_time: float) -> bool:
        local_A = self.get_A()
        local_G = self.get_G()
        A_time = self.get_A_time()
        local_E = RobotState()

        # Parity preamble: subdivide 2-point paths to >=3 and guard (mirrors the
        # Gurobi path; plan_minco itself only needs >=2 but we keep >=3 for parity)
        while len(global_path) == 2:
            mid = 0.5 * (global_path[0] + global_path[1])
            global_path = [global_path[0], mid, global_path[1]]
        if not global_path or len(global_path) < 3:
            self.replanning_failure_count += 1
            return False

        if self._drone_status in (DroneStatus.GOAL_REACHED, DroneStatus.GOAL_SEEN,
                                   DroneStatus.HOVER_AVOIDING):
            local_E = local_G.clone()
        else:
            local_E.pos = global_path[-1].copy()

        # Build the seed polyline (copy so the z-clamp does not mutate the caller's)
        seed_path = [np.asarray(p, dtype=np.float64).copy() for p in global_path]
        if self.par.vehicle_type != "uav":
            local_A.pos[2] = 1.0
            local_E.pos[2] = 1.0
            for p in seed_path:
                p[2] = 1.0

        # Obstacle snapshot under the lock (mirror the Gurobi path)
        with self._mtx:
            obst_pos = [p.copy() for p in self.obst_pos]
            obst_bbox = [b.copy() for b in self.obst_bbox]
            obst_class = list(self.obst_class)
            obst_vel = [v.copy() for v in self.obst_vel]

        obstacles, avoid_cfg = self._obstacles_from_snapshot(
            obst_pos, obst_bbox, obst_class, obst_vel)

        v0 = local_A.vel.copy()
        a0 = local_A.accel.copy()
        astar_path = np.asarray(seed_path, dtype=np.float64)

        t0 = time.perf_counter()
        try:
            # RT: single warm-started solve from the hgp global path (which is
            # already obstacle-avoiding) -- NO detour multi-start (that was ~7x
            # slower, ~4Hz; single seed is ~30ms, ~30Hz). Multi-start belongs to
            # the offline path / Stage 2 seed refinement, not the replan loop.
            mj, info = plan_minco(astar_path, obstacles, avoid_cfg, v0=v0, a0=a0,
                                  detour_cfg=DetourConfig(enabled=False))
        except Exception:  # plan_minco can RAISE 'all seeds failed'
            self.replanning_failure_count += 1
            return False

        if not info.get("trajectory_valid"):
            self.replanning_failure_count += 1
            return False

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # --- Fill goal_setpoints (LOAD-BEARING; append_to_plan flies THIS) ---
        # Mirror solver_gurobi._fill_goal_setpoints sampling at par.dc.
        dc = float(self.par.dc)
        t_end = float(mj.t_end)
        n_samples = max(2, int(math.ceil(t_end / dc)))
        setpoints: List[RobotState] = []
        for i in range(n_samples):
            t_local = min((i + 1) * dc, t_end - 1e-6)
            s = RobotState()
            s.t = A_time + t_local
            s.pos = np.asarray(mj.eval_deriv(t_local, 0), dtype=np.float64)
            s.vel = np.asarray(mj.eval_deriv(t_local, 1), dtype=np.float64)
            s.accel = np.asarray(mj.eval_deriv(t_local, 2), dtype=np.float64)
            s.jerk = np.asarray(mj.eval_deriv(t_local, 3), dtype=np.float64)
            setpoints.append(s)

        # --- Store EXACTLY in the Gurobi winner fields ---
        self.goal_setpoints = setpoints
        self.pwp_to_share = _minjerk_to_pwp(mj, A_time)
        self.cps = mj.control_points()  # (M,6,3) Bernstein; viz/share-only
        self.local_traj_computation_time = elapsed_ms
        self.successful_factor = 1.0    # sentinel (read by replan log + retrieve_data)
        self.cvx_decomp_time = 0.0
        self.poly_out_safe = []
        self.poly_out_whole = []
        self.list_subopt_goal_setpoints = []
        self._last_minco_traj = mj      # harmless debug attr for tests
        return True

    # ------------------------------------------------------------------
    # plan_local_trajectory — sweep factors, build SFC, run Gurobi, pick winner
    # ------------------------------------------------------------------
    def plan_local_trajectory(self, global_path: List[np.ndarray],
                              last_replanning_computation_time: float) -> bool:
        if self.par.local_solver == "minco":
            return self.plan_local_trajectory_minco(
                global_path, last_replanning_computation_time)
        local_A = self.get_A()
        local_G = self.get_G()
        A_time = self.get_A_time()
        local_E = RobotState()

        # If global path has 2 points subdivide until at least 3
        while len(global_path) == 2:
            mid = 0.5 * (global_path[0] + global_path[1])
            global_path = [global_path[0], mid, global_path[1]]
        if not global_path or len(global_path) < 3:
            self.replanning_failure_count += 1
            return False

        if self._drone_status in (DroneStatus.GOAL_REACHED, DroneStatus.GOAL_SEEN,
                                   DroneStatus.HOVER_AVOIDING):
            local_E = local_G.clone()
        else:
            local_E.pos = global_path[-1].copy()
        if self.par.vehicle_type != "uav":
            local_A.pos[2] = 1.0
            local_E.pos[2] = 1.0

        # Pick base map (occupied / unknown depending on sim mode)
        if self.par.sim_env in ("gazebo", "hardware"):
            base_map = list(self.hgp_manager.vec_uo) if hasattr(self.hgp_manager, "vec_uo") else []
        else:
            base_map = list(self.hgp_manager.vec_o) if hasattr(self.hgp_manager, "vec_o") else []
        # Filter out floor voxels at z_min
        z_floor_thresh = self.par.z_min + self.par.res
        base_map = [pt for pt in base_map if pt[2] > z_floor_thresh]

        # Get obst snapshots
        with self._mtx:
            obst_pos = [p.copy() for p in self.obst_pos]
            obst_bbox = [b.copy() for b in self.obst_bbox]

        # Initial dt
        self.whole_traj_solver_ptrs[0].set_X0(local_A)
        self.whole_traj_solver_ptrs[0].set_Xf(local_E)
        initial_dt = self.whole_traj_solver_ptrs[0].get_initial_dt()
        if initial_dt <= 0.0 or not math.isfinite(initial_dt):
            self.replanning_failure_count += 1
            return False

        sub_goal = [float(local_G.pos[0]), float(local_G.pos[1]), float(local_G.pos[2])]

        # Pre-compute spatial constraints for static / dynamic_worst_case
        use_precomputed = self.par.environment_assumption in ("static", "dynamic_worst_case")
        shared_spatial: List[Polytope] = []
        if use_precomputed:
            P = max(0, len(global_path) - 1)
            if self.par.environment_assumption == "dynamic_worst_case":
                max_time_horizon = self.par.num_N * initial_dt * (
                    self.factors[-1] if self.factors else 1.0)
                seg_end_times = [max_time_horizon] * P
            else:
                seg_end_times = self.compute_worst_seg_end_times_poly(
                    initial_dt, self.factors[0] if self.factors else 1.0, P)
            ok_decomp, shared_spatial = self.hgp_manager.cvx_ellipsoid_decomp(
                global_path, base_map, obst_pos, obst_bbox, seg_end_times,
            )
            if not ok_decomp:
                if self.par.debug_verbose:
                    print("Precomputed spatial convex decomposition failed")
                return False
            self.poly_out_whole = list(shared_spatial)

        # Factor sweep — sequential (see module docstring)
        winner_idx = -1
        winner_setpoints: List[RobotState] = []
        winner_pwp = PieceWisePol()
        winner_cps: List[np.ndarray] = []
        winner_gurobi_time = 0.0
        winner_decomp_time = 0.0
        winner_poly_out_safe: List[Polytope] = []
        sub_setpoints: List[List[RobotState]] = []

        parallel_opt_start = time.perf_counter()
        for i, factor in enumerate(self.factors):
            ok_loc, gurobi_time, decomp_time, poly_safe, setpoints, pwp, cps_b = \
                self._generate_local_trajectory(
                    global_path, local_A, local_E, sub_goal, A_time, factor, initial_dt,
                    obst_pos, obst_bbox, base_map, i,
                    shared_spatial if use_precomputed else None,
                )
            if not ok_loc:
                if setpoints:
                    sub_setpoints.append(setpoints)
                if not self.poly_out_safe and poly_safe:
                    self.poly_out_safe = poly_safe
                continue
            winner_idx = i
            winner_setpoints = setpoints
            winner_pwp = pwp
            winner_cps = cps_b
            winner_gurobi_time = gurobi_time
            winner_decomp_time = decomp_time
            winner_poly_out_safe = poly_safe
            break

        parallel_opt_ms = (time.perf_counter() - parallel_opt_start) * 1000.0

        if winner_idx < 0:
            # Failed: adapt factors upward (C++ behavior on failure)
            if self.par.use_dynamic_factor and self.factors:
                current_mean = float(np.mean(self.factors))
                if current_mean + self.par.factor_constant_step_size > self.par.factor_final:
                    # Reset to initial window
                    self.factors = []
                    for k in range(self.num_dynamic_factors):
                        f = self.par.dynamic_factor_initial_mean - self.par.dynamic_factor_k_radius \
                            + k * self.par.factor_constant_step_size
                        if self.par.factor_initial <= f <= self.par.factor_final:
                            self.factors.append(f)
                else:
                    self.factors = [f + self.par.factor_constant_step_size for f in self.factors]
                    self.factors = [f for f in self.factors if f <= self.par.factor_final]
            return False

        # Success: stash results
        self.goal_setpoints = winner_setpoints
        self.pwp_to_share = winner_pwp
        self.cps = winner_cps
        self.local_traj_computation_time = parallel_opt_ms
        self.cvx_decomp_time = winner_decomp_time
        self.successful_factor = self.factors[winner_idx]
        self.poly_out_safe = winner_poly_out_safe
        self.list_subopt_goal_setpoints = [s for s in sub_setpoints if s]

        # Re-center factor window on the winner
        if self.par.use_dynamic_factor:
            sf = self.factors[winner_idx]
            self.factors = []
            for k in range(self.num_dynamic_factors):
                f = sf - self.par.dynamic_factor_k_radius + k * self.par.factor_constant_step_size
                if self.par.factor_initial <= f <= self.par.factor_final:
                    self.factors.append(f)
        return True

    def _generate_local_trajectory(
        self, global_path: List[np.ndarray], local_A: RobotState, local_E: RobotState,
        sub_goal: List[float], A_time: float, factor: float, initial_dt: float,
        obst_pos: List[np.ndarray], obst_bbox: List[np.ndarray], base_uo: List[np.ndarray],
        solver_idx: int, precomputed_spatial: Optional[List[Polytope]],
    ) -> Tuple[bool, float, float, List[Polytope], List[RobotState], PieceWisePol, List[np.ndarray]]:
        P = max(0, len(global_path) - 1)
        if P == 0:
            return False, 0.0, 0.0, [], [], PieceWisePol(), []
        N = int(self.par.num_N)
        if N == 0:
            return False, 0.0, 0.0, [], [], PieceWisePol(), []

        dt_layer = initial_dt * factor
        time_end_times = [(n + 1) * dt_layer for n in range(N)]

        # Build constraints (spatial-only if precomputed, otherwise time-layered)
        cvx_start = time.perf_counter()
        if precomputed_spatial is not None:
            poly_out_safe = list(precomputed_spatial)
            cvx_decomp_time = 0.0
            layered: Optional[List[List[Polytope]]] = None
        else:
            ok_decomp, layered_polys = self.hgp_manager.cvx_ellipsoid_decomp_time_layered(
                global_path, base_uo, obst_pos, obst_bbox, time_end_times,
            )
            if not ok_decomp:
                return False, 0.0, 0.0, [], [], PieceWisePol(), []
            cvx_decomp_time = (time.perf_counter() - cvx_start) * 1000.0
            poly_out_safe = []
            for n in range(N):
                for p in range(P):
                    poly_out_safe.append(layered_polys[n][p])
            layered = layered_polys

        # Configure solver
        solver = self.whole_traj_solver_ptrs[solver_idx % len(self.whole_traj_solver_ptrs)]
        solver.set_X0(local_A)
        solver.set_Xf(local_E)
        solver.set_T0(A_time)
        solver.set_initial_dt(initial_dt)

        if precomputed_spatial is not None:
            solver.set_polytopes(precomputed_spatial)
        else:
            solver.set_polytopes_time_layered(layered)  # type: ignore[arg-type]

        try:
            ok, gurobi_compute = solver.generate_new_trajectory(factor)
        except Exception as ex:  # noqa: BLE001 — Gurobi errors are runtime-only
            print(f"Gurobi error at factor {factor}: {ex}")
            return False, 0.0, cvx_decomp_time, poly_out_safe, [], PieceWisePol(), []
        if not ok:
            return False, gurobi_compute, cvx_decomp_time, poly_out_safe, [], PieceWisePol(), []

        return True, gurobi_compute, cvx_decomp_time, poly_out_safe, \
            solver.get_goal_setpoints(), solver.get_piecewise_pol(), solver.get_control_points_bezier()

    def compute_worst_seg_end_times_poly(self, initial_dt: float, factor: float,
                                          num_seg: int) -> List[float]:
        out: List[float] = []
        if num_seg == 0:
            return out
        P = max(0, self.par.num_P)
        if P <= 0:
            dt = initial_dt * factor
            t_acc = 0.0
            for _ in range(num_seg):
                t_acc += dt
                out.append(t_acc)
            return out
        max_last_ones = P - 1
        min_one = min(max_last_ones, num_seg)
        segments_per_poly = [0] * P
        for k in range(min_one):
            segments_per_poly[(P - 1) - k] = 1
        assigned_to_last = min_one
        first_segments = num_seg - assigned_to_last
        if first_segments > 0:
            segments_per_poly[0] += first_segments
        dt = initial_dt * factor
        t_acc = 0.0
        produced = 0
        for p in range(P):
            if produced >= num_seg:
                break
            for _ in range(segments_per_poly[p]):
                if produced >= num_seg:
                    break
                t_acc += dt
                out.append(t_acc)
                produced += 1
        while len(out) < num_seg:
            t_acc += dt
            out.append(t_acc)
        return out

    # ------------------------------------------------------------------
    # appendToPlan — splice winner setpoints into the plan deque
    # ------------------------------------------------------------------
    def append_to_plan(self) -> bool:
        if self.par.debug_verbose:
            print(f"goal_setpoints_.size(): {len(self.goal_setpoints)}")

        with self._mtx:
            plan_size = len(self.plan)
            if plan_size < self.k_value:
                if self.par.debug_verbose:
                    print(f"(plan_size - k_value_) = {plan_size - self.k_value} < 0")
                self.k_value = max(1, plan_size - 1)
            else:
                # Remove the last k_value points then append new setpoints
                for _ in range(self.k_value):
                    if self.plan:
                        self.plan.pop()
                self.plan.extend(self.goal_setpoints)

        if not self.got_enough_replanning:
            if len(self.store_computation_times) < self.par.num_replanning_before_adapt:
                self.num_replanning += 1
            else:
                self.start_adapt_k_value()
                self.got_enough_replanning = True
        return True

    def start_adapt_k_value(self) -> None:
        n = max(1, len(self.store_computation_times))
        s = sum(self.store_computation_times)
        self.est_comp_time = s / n
        self.use_adapt_k_value = True

    # ------------------------------------------------------------------
    # getNextGoal — pull the front of the plan, fill yaw, apply frame transform
    # ------------------------------------------------------------------
    def get_next_goal(self) -> Tuple[bool, RobotState]:
        if self._drone_status == DroneStatus.YAWING:
            if not self.state_initialized or not self.terminal_goal_initialized:
                return False, RobotState()
        elif not self.check_ready_to_replan():
            return False, RobotState()

        with self._mtx:
            if not self.plan:
                return False, RobotState()
            local_plan = list(self.plan)

        next_goal = local_plan[0].clone()
        if len(local_plan) > 1:
            with self._mtx:
                if self.plan:
                    self.plan.popleft()

        if self._drone_status != DroneStatus.GOAL_REACHED:
            if (self.replanning_failure_count > self.par.yaw_spinning_threshold
                    and self._drone_status != DroneStatus.HOVER_AVOIDING):
                next_goal.yaw = self.previous_yaw + self.par.yaw_spinning_dyaw * self.par.dc
                next_goal.dyaw = self.par.yaw_spinning_dyaw
                self.previous_yaw = next_goal.yaw
            elif (len(local_plan) < 5 and self._drone_status != DroneStatus.YAWING
                    and self._drone_status != DroneStatus.HOVER_AVOIDING):
                next_goal.yaw = self.previous_yaw
                next_goal.dyaw = 0.0
            else:
                self.get_desired_yaw(next_goal)
            next_goal.dyaw = clamp(next_goal.dyaw, -self.par.w_max, self.par.w_max)
        else:
            next_goal.yaw = self.previous_yaw
            next_goal.dyaw = 0.0

        if self.par.use_hardware and self.par.provide_goal_in_global_frame and self.init_pose_set:
            homo = np.array([next_goal.pos[0], next_goal.pos[1], next_goal.pos[2], 1.0])
            local_pos = self.init_pose_transform_inv @ homo
            next_goal.pos = local_pos[:3]
            next_goal.vel = self.init_pose_transform_rotation_inv @ next_goal.vel
            next_goal.accel = self.init_pose_transform_rotation_inv @ next_goal.accel
            next_goal.jerk = self.init_pose_transform_rotation_inv @ next_goal.jerk
            next_goal.yaw = angle_wrap(next_goal.yaw - self.yaw_init_offset)
        return True, next_goal

    def get_desired_yaw(self, next_goal: RobotState) -> None:
        desired_yaw = 0.0
        ds = self._drone_status
        if ds == DroneStatus.YAWING:
            gterm = self.get_gterm()
            desired_yaw = math.atan2(gterm.pos[1] - self.yaw_start_pos[1],
                                     gterm.pos[0] - self.yaw_start_pos[0])
        elif ds == DroneStatus.HOVER_AVOIDING:
            dx = self.p_hover[0] - next_goal.pos[0]
            dy = self.p_hover[1] - next_goal.pos[1]
            if math.hypot(dx, dy) < 0.3:
                next_goal.yaw = self.previous_yaw
                next_goal.dyaw = 0.0
                return
            desired_yaw = math.atan2(dy, dx)
        elif ds in (DroneStatus.TRAVELING, DroneStatus.GOAL_SEEN):
            speed_xy = math.hypot(next_goal.vel[0], next_goal.vel[1])
            if speed_xy < 0.01:
                next_goal.yaw = self.previous_yaw
                next_goal.dyaw = 0.0
                return
            desired_yaw = math.atan2(next_goal.vel[1], next_goal.vel[0])
        elif ds == DroneStatus.GOAL_REACHED:
            next_goal.yaw = self.previous_yaw
            next_goal.dyaw = 0.0
            return

        if ds == DroneStatus.YAWING:
            local_state = self.get_state()
            diff = angle_wrap(desired_yaw - local_state.yaw)
            if abs(diff) < 0.3:
                self.change_drone_status(DroneStatus.TRAVELING)
            elapsed = time.monotonic() - self.yaw_start_time
            if elapsed > 10.0 and abs(diff) < 1.0:
                self.change_drone_status(DroneStatus.TRAVELING)
            diff_cmd = angle_wrap(desired_yaw - self.previous_yaw)
            max_step = self.par.w_max_yawing * self.par.dc
            step = clamp(diff_cmd, -max_step, max_step)
            next_goal.yaw = self.previous_yaw + step
            next_goal.dyaw = step / self.par.dc
            self.previous_yaw = next_goal.yaw
        elif ds == DroneStatus.HOVER_AVOIDING:
            diff_cmd = angle_wrap(desired_yaw - self.previous_yaw)
            max_step = self.par.w_max_yawing * self.par.dc
            step = clamp(diff_cmd, -max_step, max_step)
            next_goal.yaw = self.previous_yaw + step
            next_goal.dyaw = step / self.par.dc
            self.previous_yaw = next_goal.yaw
        else:
            diff = angle_wrap(desired_yaw - self.previous_yaw)
            self._yaw(diff, next_goal)

    def _yaw(self, diff: float, next_goal: RobotState) -> None:
        step = (1.0 - self.par.alpha_filter_dyaw) * diff
        max_step = self.par.w_max * self.par.dc
        step = clamp(step, -max_step, max_step)
        next_goal.yaw = self.previous_yaw + step
        next_goal.dyaw = step / self.par.dc
        self.previous_yaw = next_goal.yaw

    # ------------------------------------------------------------------
    # setTerminalGoal — drives YAWING / smooth mid-flight goal updates
    # ------------------------------------------------------------------
    def set_terminal_goal(self, term_goal: RobotState) -> None:
        # Skip duplicate
        if self.terminal_goal_initialized:
            cur = self.get_gterm()
            if float(np.linalg.norm(cur.pos - term_goal.pos)) < 0.1:
                return
        local_state = self.get_state()

        # Mid-flight smooth update
        if self.terminal_goal_initialized and self._drone_status in (
                DroneStatus.TRAVELING, DroneStatus.GOAL_SEEN):
            self.set_gterm(term_goal)
            self.p_hover = term_goal.pos.copy()
            with self._mtx:
                self.G.pos = project_point_to_sphere(local_state.pos, term_goal.pos,
                                                    self.par.horizon)
            if self._drone_status == DroneStatus.GOAL_SEEN:
                self.change_drone_status(DroneStatus.TRAVELING)
            return

        # Full re-init
        tmp = RobotState()
        tmp.pos = local_state.pos.copy()
        tmp.vel = local_state.vel.copy()
        tmp.accel = local_state.accel.copy()
        tmp.yaw = local_state.yaw
        with self._mtx:
            self.plan.clear()
            self.plan.append(tmp)
            self.A = tmp.clone()
            self.G = tmp.clone()

        self.set_gterm(term_goal)
        self.p_hover = term_goal.pos.copy()
        self.previous_yaw = local_state.yaw
        self.replanning_failure_count = 0

        with self._mtx:
            self.G.pos = project_point_to_sphere(local_state.pos, term_goal.pos, self.par.horizon)

        self.yaw_start_pos = local_state.pos.copy()
        self.yaw_start_time = time.monotonic()

        if self.par.skip_initial_yawing:
            self.change_drone_status(DroneStatus.TRAVELING)
        else:
            self.change_drone_status(DroneStatus.YAWING)

        if not self.terminal_goal_initialized:
            self.terminal_goal_initialized = True

    # ------------------------------------------------------------------
    # Hover avoidance
    # ------------------------------------------------------------------
    def check_hover_avoidance(self, current_time: float) -> bool:
        local_state = self.get_state()
        local_trajs = self.get_trajs()
        lookahead_window = 15.0
        lookahead_step = 0.5

        def is_threatened(pt: np.ndarray, lookahead: bool = True) -> bool:
            for t in local_trajs:
                if float(np.linalg.norm(pt - t.current_pos)) < self.par.hover_avoidance_d_trigger:
                    return True
                if lookahead:
                    t_end = current_time + lookahead_window
                    if t.mode == "Piecewise" and t.pwp.times:
                        t_end = min(t_end, t.pwp.times[-1])
                    tt = current_time
                    while tt <= t_end:
                        p_obs = t.eval(tt)
                        if float(np.linalg.norm(pt - p_obs)) < self.par.hover_avoidance_d_trigger:
                            return True
                        tt += lookahead_step
            return False

        n_total = np.zeros(3)
        closest_dist = 1e9
        for traj in local_trajs:
            p_obs = traj.current_pos
            r_i = local_state.pos - p_obs
            dist_i = float(np.linalg.norm(r_i))
            closest_dist = min(closest_dist, dist_i)
            if dist_i < self.par.hover_avoidance_d_trigger and dist_i > 1e-6:
                w_i = 1.0 / (dist_i * dist_i)
                n_total += w_i * (r_i / dist_i)

        if float(np.linalg.norm(n_total)) > self.par.hover_avoidance_min_repulsion_norm:
            if self._drone_status != DroneStatus.HOVER_AVOIDING:
                gterm = self.get_gterm()
                self.p_hover = gterm.pos.copy()
                self.change_drone_status(DroneStatus.HOVER_AVOIDING)

            direction = n_total / float(np.linalg.norm(n_total))
            if self.par.hover_avoidance_2d:
                direction[2] = 0.0
            n = float(np.linalg.norm(direction))
            if n < 1e-6:
                direction = np.array([1.0, 0.0, 0.0])
            else:
                direction = direction / n

            p_evasion = local_state.pos + self.par.hover_avoidance_h * direction
            if self.par.hover_avoidance_2d:
                p_evasion[2] = local_state.pos[2]
            else:
                p_evasion[2] = max(self.par.z_min + 0.5,
                                   min(p_evasion[2], self.par.z_max - 0.5))

            if is_threatened(p_evasion, True):
                p_evasion = self._find_safe_evasion(direction, local_state.pos, is_threatened,
                                                    require_unoccupied=False)
                if p_evasion is None:
                    return False

            if self.check_if_point_occupied(p_evasion):
                p_evasion = self._find_safe_evasion(direction, local_state.pos, is_threatened,
                                                    require_unoccupied=True)
                if p_evasion is None:
                    return False

            evasion_goal = RobotState()
            evasion_goal.set_pos(float(p_evasion[0]), float(p_evasion[1]), float(p_evasion[2]))
            self.set_gterm(evasion_goal)
            with self._mtx:
                self.G.pos = project_point_to_sphere(local_state.pos, p_evasion, self.par.horizon)
            return True
        elif self._drone_status == DroneStatus.HOVER_AVOIDING:
            if is_threatened(self.p_hover, False):
                return False
            hover_goal = RobotState()
            hover_goal.set_pos(float(self.p_hover[0]), float(self.p_hover[1]), float(self.p_hover[2]))
            self.set_gterm(hover_goal)
            with self._mtx:
                self.G.pos = project_point_to_sphere(local_state.pos, self.p_hover, self.par.horizon)
            return True
        return False

    def _find_safe_evasion(self, direction: np.ndarray, pos: np.ndarray,
                            is_threatened, require_unoccupied: bool) -> Optional[np.ndarray]:
        angles = [math.pi / 6, -math.pi / 6, math.pi / 3, -math.pi / 3,
                  math.pi / 2, -math.pi / 2, math.pi]
        for angle in angles:
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            rotated = np.array([direction[0] * cos_a - direction[1] * sin_a,
                                direction[0] * sin_a + direction[1] * cos_a,
                                direction[2]])
            n = float(np.linalg.norm(rotated))
            if n < 1e-9:
                continue
            rotated = rotated / n
            cand = pos + self.par.hover_avoidance_h * rotated
            if self.par.hover_avoidance_2d:
                cand[2] = pos[2]
            else:
                cand[2] = max(self.par.z_min + 0.5, min(cand[2], self.par.z_max - 0.5))
            if is_threatened(cand, True):
                continue
            if require_unoccupied and self.check_if_point_occupied(cand):
                continue
            return cand
        return None

    # ------------------------------------------------------------------
    # Initial pose transform (hardware only)
    # ------------------------------------------------------------------
    def set_initial_pose(self, init_pose) -> None:
        self.init_pose_transform = transform_stamped_to_matrix(init_pose)
        self.init_pose_transform_inv = transform_inverse_se3(self.init_pose_transform)
        self.init_pose_transform_rotation = self.init_pose_transform[:3, :3].copy()
        self.init_pose_transform_rotation_inv = self.init_pose_transform_rotation.T
        # Yaw from R[1,0], R[0,0]
        R = self.init_pose_transform_rotation
        self.yaw_init_offset = math.atan2(R[1, 0], R[0, 0])

        # Sanity check: M_inv * init_pos should give (0,0,0)
        t = self.init_pose_transform[:3, 3]
        homo = np.array([t[0], t[1], t[2], 1.0])
        local_origin = self.init_pose_transform_inv @ homo
        err = float(np.linalg.norm(local_origin[:3]))
        if err < 0.1:
            print("****** [SANDO] READY TO FLY ******")
        else:
            print(f"****** [SANDO] TRANSFORM SANITY CHECK FAILED (err={err:.3f}) ******")
        self.init_pose_set = True

    def apply_init_pose_transform(self, pwp: PieceWisePol) -> None:
        for i in range(len(pwp.coeff_x)):
            for j in range(4):
                coeff = np.array([pwp.coeff_x[i][j], pwp.coeff_y[i][j], pwp.coeff_z[i][j], 1.0])
                coeff = self.init_pose_transform @ coeff
                pwp.coeff_x[i][j] = coeff[0]
                pwp.coeff_y[i][j] = coeff[1]
                pwp.coeff_z[i][j] = coeff[2]

    def apply_init_pose_inverse_transform(self, pwp: PieceWisePol) -> None:
        for i in range(len(pwp.coeff_x)):
            for j in range(4):
                coeff = np.array([pwp.coeff_x[i][j], pwp.coeff_y[i][j], pwp.coeff_z[i][j], 1.0])
                coeff = self.init_pose_transform_inv @ coeff
                pwp.coeff_x[i][j] = coeff[0]
                pwp.coeff_y[i][j] = coeff[1]
                pwp.coeff_z[i][j] = coeff[2]

    # ------------------------------------------------------------------
    # Retrieval helpers (mirrors C++ retrieve* API)
    # ------------------------------------------------------------------
    def retrieve_data(self):
        return {
            "final_g": self.final_g,
            "global_planning_time": self.global_planning_time,
            "hgp_static_jps_time": self.hgp_static_jps_time,
            "hgp_check_path_time": self.hgp_check_path_time,
            "hgp_dynamic_astar_time": self.hgp_dynamic_astar_time,
            "hgp_recover_path_time": self.hgp_recover_path_time,
            "cvx_decomp_time": self.cvx_decomp_time,
            "local_traj_computation_time": self.local_traj_computation_time,
            "safety_check_time": self.safety_check_time,
            "safe_paths_time": self.safe_paths_time,
            "yaw_sequence_time": self.yaw_sequence_time,
            "yaw_fitting_time": self.yaw_fitting_time,
            "successful_factor": self.successful_factor,
        }

    def retrieve_polytopes(self) -> Tuple[List[Polytope], List[Polytope]]:
        return list(self.poly_out_whole), list(self.poly_out_safe)

    def retrieve_goal_setpoints(self) -> List[RobotState]:
        return list(self.goal_setpoints)

    def retrieve_list_subopt_goal_setpoints(self) -> List[List[RobotState]]:
        return list(self.list_subopt_goal_setpoints)

    def retrieve_cps(self) -> List[np.ndarray]:
        return list(self.cps)
