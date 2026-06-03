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

================== 中文说明(这个文件是整个规划器的"大脑/总控")==================
这个文件是规划器的最外层状态机和主循环,负责把各个零件串起来反复跑:
  1) 它管一台"状态机":飞机当前处在 转头(YAWING)/赶路(TRAVELING)/看见目标
     (GOAL_SEEN)/到达(GOAL_REACHED)/悬停躲避(HOVER_AVOIDING) 之一,不同状态做不同事。
  2) 每一拍(replan)里,它依次调用:
       - HGPManager:先在体素栅格(把空间切成小方块)上用 heat-A* 算一条全局粗路径
         (整体往哪走、绕开已知大障碍);
       - 局部优化器(两套可切换):
           * 默认的新算法 plan_minco —— MINCO 五次最小 jerk 曲线 + 按障碍"分类"避障
             (人 = 硬约束:凸包+ALM,而且带时空,会在"对的时刻"躲会动的人;
              墙 = 软约束:EGO 势场,可以蹭着边走);
           * 老基线 SolverGurobi —— 凸走廊 + Gurobi 二次规划 QP。
       - 把算出来的一段轨迹"接"到已经在飞的计划队列上(commit/splice)。
  3) "receding-horizon"(滚动时域):不是一次算到底,而是每拍只算未来一小段,
     边飞边重算,这样能对环境变化做反应。

关键名词(尽量讲人话):
  - A:这一拍轨迹的"起点状态"。不是飞机此刻的位置,而是在已提交计划上往前挑一个点,
       这样新轨迹能跟旧轨迹平滑接上(留出计算时间的余量,见 k_value)。
  - A_time:A 这个点对应的"墙上时间"(绝对时间戳)。
  - G:本次规划的局部终点(把最终目标 G_term 投影到 horizon 半径的球面上,飞不到那么远
       就先盯着球面上那个方向点)。
  - G_term:用户给的最终目标。
  - plan:一个队列(deque),里面是按固定时间间隔 dc 采样好的、马上要飞的目标点序列。
  - k_value:每次重规划时"砍掉计划队列尾巴的点数 / 往前挑 A 的偏移量"。它平衡两件事:
       砍多了浪费已算好的轨迹,砍少了新轨迹还没接上飞机就飞过头了。会根据计算耗时自适应。

输入:里程计状态(update_state)、点云地图(update_map_ptr)、动态障碍轨迹(add_traj)、
      用户目标(set_terminal_goal)。
输出:get_next_goal 吐出的下一个要飞的目标点(位置/速度/加速度/jerk/yaw)。
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

    中文:一个"分段多项式轨迹"的容器,专门装 MINCO 算出来的曲线,好让外面共享/可视化的
    代码当成普通 PieceWisePol 来读。
    踩坑点:基类 PieceWisePol 是 3 次、系数"降幂"排列;而 MINCO 是 5 次、系数"升幂"排列
    (c0 + c1*u + ... + c5*u^5)。次数和排列都不一样,所以不能直接用基类,只能继承下来,
    只重写 _eval_axis(按某一段、局部时间 u 求值/求导),其余方法照旧。
    """

    def _eval_axis(self, coeffs, t: float, order: int) -> float:
        # 中文:给定某一根轴(x/y/z)的各段系数,在绝对时间 t 处求 order 阶导(0=位置,
        # 1=速度,2=加速度)。先按 t 落在哪一段,把 t 减去该段起始时间得到局部时间 u,
        # 再用升幂 5 次多项式求值。t 超出范围就钳到首/末段端点。
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

    中文:对升幂 5 次多项式 p(u)=c0+c1*u+...+c5*u^5 在局部时间 u 处求第 order 阶导。
    实现刻意和 MinjerkTraj 里那套一模一样(同样用 `**` 幂运算),保证两边算出来逐位相同。
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

    中文:把 MINCO 算出的轨迹对象(MinjerkTraj)包装成上面那个 QuinticPieceWisePol,
    给可视化/对外共享用。
    - 段边界的绝对时间 = A_time(本段起点的墙上时间)+ mj._cum(从 0 开始的各段累计时长)。
    - 系数直接照搬(升幂、reshape 成 每段 6 个系数 x 3 轴)。
    - 关键:对外求值用的局部时间 u = t - times[i],代进去 A_time 正好抵消,等于 MINCO 自己的
      局部时间 tau,所以不会出现"差一个时间偏移"的对齐 bug。
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

    中文:整个规划器的最外层类,持有所有共享状态——状态机、动态障碍列表、全局路径、
    马上要飞的 plan 队列、各阶段耗时统计。C++ 里用了约 14 把锁分别保护各字段;这里因为
    Python 有 GIL 而且真正的争用是"每拍一次"而非"每字段一次",合并成一把大锁 self._mtx
    就够了。
    """

    def __init__(self, par: Parameters):
        self.par = par

        # HGPManager and shared parameters
        self.hgp_manager = HGPManager()
        self.hgp_manager.set_parameters(par)

        # Factor list (time allocation sweep)
        # 中文:factor = 时间缩放因子。同一条空间路径,把它走得快/慢(每段分配多少时间)
        # 会得到不同难度的轨迹。下面预先生成一组待试的 factor:Gurobi 那套会从小到大逐个试,
        # 第一个能解出可行轨迹的 factor 就用它(failure→整体往大移,success→以赢家为中心重设窗口)。
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
        # 中文:预先估一个"最坏情况下一条轨迹要飞多久"(从原点飞到最远的子目标)。
        # 这个时间 Th 后面用来决定动态障碍要往前预测多远(预测时域)。
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
        # 中文:滚动重规划的核心状态。state=飞机当前真实状态;G=本次局部终点;
        # A=本次轨迹起点(在已提交计划上往前挑的点);A_time=A 的绝对时间;
        # E=轨迹末端目标;G_term=用户最终目标;plan=马上要飞的目标点队列。
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
        # 中文:当前这一拍筛进视野内的障碍快照,下面四个列表一一对应(同一个下标=同一个障碍):
        # obst_pos=中心位置, obst_bbox=包围盒尺寸, obst_class=类别('human'硬/'wall'软),
        # obst_vel=当前速度(给会动的人做时空避障用)。
        self.obst_pos: List[np.ndarray] = []
        self.obst_bbox: List[np.ndarray] = []
        self.obst_class: List[str] = []          # per-class tag, parallel to obst_pos
        self.obst_vel: List[np.ndarray] = []      # current obstacle velocity, parallel
        # obst_accel=当前加速度(给会动的人做 CA 匀加速时空预测;静止/匀速时为 0)
        self.obst_accel: List[np.ndarray] = []    # current obstacle acceleration, parallel
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
        # 中文:k_value 自适应相关。先攒够 num_replanning_before_adapt 拍的真实计算耗时,
        # 之后切到"按预估耗时算 k_value"模式(算得慢就多砍点尾巴,留够时间接轨)。
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
        # 中文:切换状态机状态并打印一行日志(状态没变就直接返回,不刷屏)。
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
    # 中文:下面这一批 get/set 都是"加锁读写共享状态"的小工具,读出来一律返回克隆/拷贝,
    # 避免外面拿到内部对象的引用后偷偷改掉它(线程安全)。
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
        # 中文:取计划队列里最后一个点(也就是当前已规划到的最远处);队列空就退回当前状态。
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
    # 中文:下面三个 property 是"懒初始化"的全局路径缓存——第一次重规划前还没算过路径,
    # 直接访问会报 AttributeError,所以用 property 在首次访问时给个空列表兜底。
    #   _global_path=裁剪/处理后的全局路径; _original_global_path=HGP 原始输出;
    #   _free_global_path=已知自由空间内的那一段(可视化/诊断用)。
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
        # 中文:丢掉太久没更新的动态障碍轨迹(超过 traj_lifetime 没收到新消息就当它过期)。
        with self._mtx:
            self.trajs = [t for t in self.trajs
                          if (current_time - t.time_received) <= self.par.traj_lifetime]

    def add_traj(self, new_traj: DynTraj, current_time: float) -> None:
        """中文:收到一条动态障碍预测轨迹。同 id 的先就地更新(已知障碍只是位置变了,不再过滤);
        全新的障碍才做"在地图内 + 在 horizon 半径内"的过滤,太远的先不管。"""
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
        """中文:里程计回调,刷新飞机当前状态 state。
        第一次拿到状态(或正在转头 YAWING)时,顺便把 plan/A/G 都重置成当前点,
        相当于"原地起步":此时还没规划,先让计划队列里只有当前这个点。"""
        # Hardware-frame translation
        # 中文:真机模式下里程计是本地坐标系,这里乘上初始位姿变换转到全局系。
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
        # 中文:根据"起点和目标"两点框出本次要建图的局部窗口大小,四周再留一圈 buffer,
        # 并保证不小于配置的最小尺寸。地图中心取两点中点。每次重规划都重算一遍(地图跟着飞机走)。
        dynamic_buffer = self.par.map_buffer
        dist_x = abs(min_pos[0] - max_pos[0])
        dist_y = abs(min_pos[1] - max_pos[1])
        dist_z = abs(min_pos[2] - max_pos[2])
        self.wdx = max(dist_x + 2 * dynamic_buffer, self.par.min_wdx)
        self.wdy = max(dist_y + 2 * dynamic_buffer, self.par.min_wdy)
        self.wdz = max(dist_z + 2 * dynamic_buffer, self.par.min_wdz)
        self.map_center = (min_pos + max_pos) / 2.0

    def check_point_within_map(self, point: np.ndarray) -> bool:
        # 中文:判断一个点是否落在当前局部地图窗口里(三个轴各自不超过窗口半宽)。
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
        """中文:收到新点云时调用。pclptr_map=已占据(障碍)点云,pclptr_unk=未知区域点云。
        这里只是把点云存起来,真正建图在 update_map 里做。"""
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

    # 中文:把点云塞进 KD-tree(一种能快速查"离某点最近的点有多远"的数据结构),
    # 后面 find_safe_sub_goal 要靠它判断路径有没有撞进未知/障碍区。scipy 懒加载,没装就置 None。
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
        """中文:真正建图。算好局部窗口大小,把"当前动态障碍此刻位置 + 往前预测的若干采样点"
        喂给 HGPManager 去更新体素/热力图(让 heat-A* 知道哪里要躲),最后把点云重建成 KD-tree。
        gazebo/hardware 用 已占据+未知 两套点云;fake_sim/rviz_only 走下面的 occupancy-only 版本。"""
        local_state = self.get_state()
        local_G = self.get_G()
        self.compute_map_size(local_state.pos, local_G.pos)

        obst_pos, obst_bbox, pred_samples, pred_times = [], [], [], []
        self.traj_max_time = self._compute_obst_pos_and_traj_max_time(
            obst_pos, obst_bbox, pred_samples, pred_times, current_time
        )

        # Forward predicted samples to the heat-map (used by VoxelMapUtil)
        # 中文:把动态障碍未来若干时刻的预测位置告诉热力图,让全局 A* 提前避开它将要去的地方。
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
        """中文:update_map 的精简版,只用"已占据"点云、未知点云传空。
        用在 fake_sim / rviz_only 这种没有真实感知、地图直接给定的模式。"""
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
        """中文:遍历所有动态障碍轨迹,挑出"在地图内且在 horizon 内"的,做成本拍的障碍快照。
        通过传进来的列表(原地填充)返回:每个障碍的当前中心 obst_pos、包围盒 obst_bbox、
        未来预测采样 pred_samples(各障碍在 pred_times 这些时刻的位置)。返回值是预测时域 Th。
        同时把 类别(human/wall)和 速度 存进 self.obst_class/self.obst_vel,供局部避障分类用。"""
        obst_pos.clear(); obst_bbox.clear(); pred_samples.clear(); pred_times.clear()
        local_trajs = self.get_trajs()

        # per-class snapshot tags (swap point for a real classifier): class keyed on
        # the DynTraj id range (200<=id<300 -> wall/SOFT, else human/HARD) and the
        # current obstacle velocity (so a moving human reaches plan_minco as a moving
        # SphereObstacle -> genuine dynamic, space-time avoidance).
        obst_class: List[str] = []
        obst_vel: List[np.ndarray] = []
        obst_accel: List[np.ndarray] = []
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
            # 中文:这里是"分类的临时占位规则"——目前靠障碍 id 区间判类别(200~299 当作墙/软避障,
            # 其余当作人/硬避障)。将来换成真正的 RGBD/跟踪分类器只需替换这一行。
            tid = int(getattr(traj, "id", 0))
            obst_class.append("wall" if 200 <= tid < 300 else "human")
            try:
                v = np.asarray(traj.velocity(current_time), dtype=float).reshape(3)
            except Exception:
                v = np.zeros(3)
            obst_vel.append(v)
            # CA 时空预测用的加速度:和 vel 取同一时刻 current_time(SphereObstacle 局部 t=0
            # 对齐 current_time)。取不到就退回 0(等价于匀速 CV)。
            try:
                acc = np.asarray(traj.accel(current_time), dtype=float).reshape(3)
            except Exception:
                acc = np.zeros(3)
            obst_accel.append(acc)
            selected.append(traj)

        with self._mtx:
            self.obst_pos = [p.copy() for p in obst_pos]
            self.obst_bbox = [b.copy() for b in obst_bbox]
            self.obst_class = list(obst_class)
            self.obst_vel = [v.copy() for v in obst_vel]
            self.obst_accel = [a.copy() for a in obst_accel]

        # 中文:预测时域 Th = 最坏轨迹时长 x 最大缩放因子(即"这条轨迹最久可能飞多久")。
        Th = self.worst_traj_time * (self.factors[-1] if self.factors else 1.0)
        if Th <= 0.0 or not selected:
            return Th

        # 中文:在 [0, Th] 上均匀取 M 个时刻(夹在 5~10 个之间),作为预测采样的时间点。
        dt = 0.5
        M = int(math.ceil(Th / dt)) + 1
        M = max(5, min(M, 10))
        for j in range(M):
            a = 0.0 if M == 1 else j / (M - 1)
            pred_times.append(float(a * Th))

        # 中文:对每个选中的障碍,按上面那些时刻算出它未来会在哪(算出 NaN/Inf 就退回当前位置兜底)。
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
        """中文:算本次的局部终点 G。把最终目标 G_term 沿"从 A 指向 G_term 的方向"投影到
        半径 horizon 的球面上——目标太远就先盯着这个方向、飞到能看见为止;同时把 G 的 yaw
        设成朝向 G_term 的方向。"""
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
        """中文:判断这一拍要不要重新规划,顺带推进状态机。
        大致逻辑:开了悬停躲避且处于到达/躲避态 → 要;已到目标且速度很小 → 不要(切到到达态);
        转头/到达态 → 不要;离目标够近 → 切到 GOAL_SEEN;已看见目标且计划末端也到了 → 不要;
        其余情况 → 要。"""
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
        # 中文:重规划的前置条件都齐了吗——拿到过状态、设过目标、地图建好、(真机还要)KD-tree 建好。
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
        """中文:挑本次轨迹的起点 A 和它的绝对时间 A_time。
        核心思路:不从飞机此刻位置起规划(那样规划还没算完飞机已经飞过去了),而是从计划队列里
        往前数 k_value 个点处起规划——给"规划这段时间内飞机会继续飞"留出余量,新旧轨迹才能接上。
        k_value 怎么定:还没攒够数据时按固定 default_k_value;攒够后按"预估计算耗时/dc"自适应
        (算得越慢,k_value 越大、砍掉的尾巴越多)。最后检查 A 没飞出地图边界。"""
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

            # 中文:防越界——索引算出来不合法就退回"用计划末端那个点"。
            if plan_size - 1 - self.k_value < 0 or plan_size - 1 - self.k_value >= plan_size:
                self.k_value = plan_size - 1

            # 中文:A = 计划队列里第 (末尾-k_value) 个点;A_time = 当前时间 + 这么多个 dc 后的时刻。
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
        """中文:把全局路径在"第一次撞进未知区"的地方截断,返回一段保证安全的子目标路径。
        做法:沿路径每隔 sample_dist 采点,用未知点云的 KD-tree 查"离最近未知点的距离";
        一旦小于 drone_radius(算撞了)就停下,再用 backtrack 往回退到一个真正安全的点
        (退的时候用更保守的、加了"障碍最大速度 x 飞行时间"膨胀半径的阈值)。
        这样飞机只会飞进它当下确定安全的范围,把未知区留到下拍看清楚再说。"""
        original = [p.copy() for p in global_path]
        if not original:
            return []
        out: List[np.ndarray] = [original[0].copy()]

        sample_dist = 0.1
        # 中文:膨胀半径 = 障碍最大速度 x 这条轨迹要飞的时间,代表"飞这段期间障碍最远能挪多远"。
        r_inflate = self.par.obst_max_vel * self.traj_max_time
        thr_orig = self.par.drone_radius
        thr_infl = self.par.drone_radius + r_inflate
        thr_orig2 = thr_orig * thr_orig
        thr_infl2 = thr_infl * thr_infl

        # 中文:某点是否"算落在未知区"——离最近未知点比阈值还近就算。注意 query 返回的已是距离,
        # 这里又平方一次和阈值平方比(等价于比距离),没装 scipy(树为 None)就当作不在未知区。
        def is_within_unknown(pt: np.ndarray, thr2: float) -> bool:
            if self._kdtree_unk is None:
                return False
            d2, _ = self._kdtree_unk.query(pt, k=1)
            return float(d2 * d2) < thr2

        # 中文:从撞点沿路径往回退,逐步后撤(必要时跨回上一段),直到找到一个用膨胀阈值也安全的点。
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

        # 中文:逐段、逐采样点扫过去;一旦某点撞进未知区就回退取安全点并立刻返回,否则把整段末端纳入。
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
        # 中文:每拍重规划开头清空上一拍的诊断数据/耗时/中间结果,重新开始计时统计。
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
        """中文:一拍完整的重规划主流程(整个规划器的"心跳")。依次做:
        检查就绪 → 判断要不要重规划 → (需要时)悬停躲避 → 全局路径(HGP)
        → 局部轨迹优化(MINCO 或 Gurobi)→ 把结果接到计划队列。
        返回 (是否规划成功, 是否真的尝试过规划)——第二个布尔给上层日志/失败计数用。"""
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

        # 中文:到达/悬停态且开了悬停躲避——这种情况下不走正常规划,而是靠斥力场临时挪一下避开。
        if self.par.hover_avoidance_enabled and (
                self._drone_status == DroneStatus.GOAL_REACHED
                or self._drone_status == DroneStatus.HOVER_AVOIDING):
            if not self.check_hover_avoidance(current_time):
                return False, False

        # Global planning
        # 中文:第一步——全局粗路径(heat-A*),决定整体往哪绕。
        ok, global_path = self.generate_global_path(current_time,
                                                    last_replanning_computation_time)
        if not ok:
            return False, False

        # Local trajectory
        # 中文:第二步——把粗路径优化成平滑、可飞、避障的局部轨迹。
        if not self.plan_local_trajectory(global_path, last_replanning_computation_time):
            return False, True

        # Append to plan
        # 中文:第三步——把新轨迹接到正在飞的计划队列上(commit)。
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
        """中文:跑全局规划这一段。流程:挑起点 A 和 A_time → 算局部终点 G → 更新地图
        → 让 HGP 在体素栅格上用 heat-A* 解出一条从 A 到 G 的粗路径 → 裁剪到合适段数
        → 在未知区处截断成安全子目标。返回 (是否成功, 处理后的全局路径点列)。"""
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
        # 中文:给 A* 一个出发方向提示(优先用上一拍全局路径的起始方向,否则用 A→G 的直线方向),
        # 让连续两拍的路径不要忽左忽右地跳,飞得更稳。
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
        # 中文:裁掉太长的路径,只留前 num_P+1 个点(局部优化器一次只处理这么多段)。
        if len(global_path) > self.par.num_P + 1:
            global_path = global_path[: self.par.num_P + 1]

        # Safe sub-goal truncation
        # 中文:在第一次撞进未知区的地方截断,只保留当下确定安全的那一段。
        global_path = self.find_safe_sub_goal(global_path)
        if self.par.debug_verbose:
            print(f"global_path.size(): {len(global_path)}")
        return True, global_path

    # ------------------------------------------------------------------
    # MINCO local solve (per-class avoidance) — additive replacement body
    # ------------------------------------------------------------------
    def _obstacles_from_snapshot(self, obst_pos, obst_bbox, obst_class=None,
                                 obst_vel=None, obst_accel=None):
        """Class source hook: map the obstacle snapshot to per-class avoidance
        obstacles for plan_minco. obst_class[k] in {'human','wall'} decides the
        MECHANISM (the per-class point): human -> SphereObstacle (HARD, carries
        its current velocity so the Stage-4 space-time ALM avoids it at the right
        moment); wall -> AABBObstacle (SOFT EGO field, grazeable).

        The class tag is currently keyed on the DynTraj id range upstream
        (_compute_obst_pos_and_traj_max_time); a real RGBD/tracker classifier
        swaps in there. Missing tag -> 'human'/HARD (fail-safe), missing vel -> 0.
        obst_accel[k] -> the human's current acceleration for the CA (constant-
        acceleration) space-time prediction; missing -> 0 (falls back to CV).

        obst_pos[k] = (3,) box CENTER; obst_bbox[k] = (3,) FULL box extents.

        中文:这是"分类避障"的关键开关——把障碍快照按类别变成两种不同的避障对象,交给 plan_minco:
          - 'human'(人)→ SphereObstacle,走硬约束(凸包+ALM),而且带上它当前速度,
            这样第 4 阶段的"时空"避障会在对的时刻躲开它将要去的位置(真·动态躲人)。
          - 'wall'(墙)→ AABBObstacle(轴对齐包围盒),走软约束(EGO 势场),允许蹭着边走。
        类别拿不到时一律按 'human'/硬约束处理(更保守、更安全),速度拿不到就当 0。
        注意:obst_pos[k] 是包围盒中心,obst_bbox[k] 是包围盒的"全尺寸"(所以半尺寸要乘 0.5)。
        """
        n = len(obst_pos)
        classes = list(obst_class) if obst_class is not None else []
        vels = list(obst_vel) if obst_vel is not None else []
        accels = list(obst_accel) if obst_accel is not None else []
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
                # CA: 给会动的人传当前加速度,做匀加速时空预测;拿不到就当 0(退回匀速 CV)
                acc = (np.asarray(accels[k], dtype=np.float64).reshape(3)
                       if k < len(accels) else np.zeros(3))
                obstacles.append(SphereObstacle(
                    centre0=c, radius=0.5 * float(np.max(sz)),
                    vel=vel, accel=acc, class_name="human"))
        avoid_cfg = _avoid_default_config()
        return obstacles, avoid_cfg

    def plan_local_trajectory_minco(self, global_path: List[np.ndarray],
                                    last_replanning_computation_time: float) -> bool:
        """中文:新算法的局部优化主体——把全局粗路径优化成一条 MINCO 五次最小 jerk 曲线,
        并按类别避障(人=硬,墙=软)。流程:
          1) 整理种子路径(把折线当初值,地面车把 z 钉成 1.0);
          2) 取障碍快照、按类别造避障对象(_obstacles_from_snapshot);
          3) 先用全局路径当种子做一次"热启动"求解(它本来就避障了,常见情况快);
             解不出可行轨迹再退回更慢但更稳的"绕行多起点"模式;
          4) 把求出的曲线按时间间隔 dc 采样成 goal_setpoints(真正拿去飞的就是它),
             并存好对外共享/可视化要用的 pwp、控制点等。
        成功返回 True。注意 goal_setpoints 是"承重字段"——append_to_plan 飞的就是这个。"""
        local_A = self.get_A()
        local_G = self.get_G()
        A_time = self.get_A_time()
        local_E = RobotState()

        # Parity preamble: subdivide 2-point paths to >=3 and guard (mirrors the
        # Gurobi path; plan_minco itself only needs >=2 but we keep >=3 for parity)
        # 中文:只有 2 个点的路径在中间插一个中点凑到 >=3 个(和 Gurobi 那条路保持一致),
        # 不足 3 点就当失败。
        while len(global_path) == 2:
            mid = 0.5 * (global_path[0] + global_path[1])
            global_path = [global_path[0], mid, global_path[1]]
        if not global_path or len(global_path) < 3:
            self.replanning_failure_count += 1
            return False

        # 中文:轨迹末端目标 E——快到/看见/悬停时直接用 G;赶路时用全局路径末端点。
        if self._drone_status in (DroneStatus.GOAL_REACHED, DroneStatus.GOAL_SEEN,
                                   DroneStatus.HOVER_AVOIDING):
            local_E = local_G.clone()
        else:
            local_E.pos = global_path[-1].copy()

        # Build the seed polyline (copy so the z-clamp does not mutate the caller's)
        # 中文:种子折线(优化初值)。这里 copy 一份,免得地面车把 z 钉成 1.0 时改到调用方的数据。
        seed_path = [np.asarray(p, dtype=np.float64).copy() for p in global_path]
        if self.par.vehicle_type != "uav":
            local_A.pos[2] = 1.0
            local_E.pos[2] = 1.0
            for p in seed_path:
                p[2] = 1.0

        # Obstacle snapshot under the lock (mirror the Gurobi path)
        # 中文:加锁拷一份障碍快照(位置/包围盒/类别/速度),拷出来再用,免得求解期间被别的线程改。
        with self._mtx:
            obst_pos = [p.copy() for p in self.obst_pos]
            obst_bbox = [b.copy() for b in self.obst_bbox]
            obst_class = list(self.obst_class)
            obst_vel = [v.copy() for v in self.obst_vel]
            obst_accel = [a.copy() for a in self.obst_accel]

        obstacles, avoid_cfg = self._obstacles_from_snapshot(
            obst_pos, obst_bbox, obst_class, obst_vel, obst_accel)

        # 中文:轨迹起点的初始速度/加速度(让新轨迹和飞机当前运动平滑衔接),以及当种子的折线。
        v0 = local_A.vel.copy()
        a0 = local_A.accel.copy()
        astar_path = np.asarray(seed_path, dtype=np.float64)

        t0 = time.perf_counter()
        try:
            # RT: try ONE warm-started solve from the hgp global path first (it is
            # already obstacle-avoiding) -- NO detour multi-start, so the common
            # case is fast (~30 ms, ~30 Hz; detour was ~7x slower). Only if that
            # single seed is infeasible do we fall back to the multi-start detour
            # (slower, but rare) so hard scenes still get solved.
            mj, info = plan_minco(astar_path, obstacles, avoid_cfg, v0=v0, a0=a0,
                                  detour_cfg=DetourConfig(enabled=False))
            if not info.get("trajectory_valid"):
                mj, info = plan_minco(astar_path, obstacles, avoid_cfg, v0=v0, a0=a0,
                                      detour_cfg=DetourConfig(enabled=True))
        except Exception:  # plan_minco can RAISE 'all seeds failed'
            # 中文:plan_minco 在"所有起点都失败"时会直接抛异常,这里兜住当作本拍规划失败。
            self.replanning_failure_count += 1
            return False

        if not info.get("trajectory_valid"):
            self.replanning_failure_count += 1
            return False

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # --- Fill goal_setpoints (LOAD-BEARING; append_to_plan flies THIS) ---
        # Mirror solver_gurobi._fill_goal_setpoints sampling at par.dc.
        # 中文:把连续曲线 mj 按固定时间间隔 dc 离散成一串目标点(位置/速度/加速度/jerk),
        # 每个点的绝对时间 = A_time + 局部时间。append_to_plan 真正接进计划队列、拿去飞的就是它。
        dc = float(self.par.dc)
        t_end = float(mj.t_end)
        n_samples = max(2, int(math.ceil(t_end / dc)))
        setpoints: List[RobotState] = []
        for i in range(n_samples):
            # 中文:从 dc 开始采(第一个点是 dc 处,不是 0),末点稍微往内缩一点避免落在边界外。
            t_local = min((i + 1) * dc, t_end - 1e-6)
            s = RobotState()
            s.t = A_time + t_local
            s.pos = np.asarray(mj.eval_deriv(t_local, 0), dtype=np.float64)
            s.vel = np.asarray(mj.eval_deriv(t_local, 1), dtype=np.float64)
            s.accel = np.asarray(mj.eval_deriv(t_local, 2), dtype=np.float64)
            s.jerk = np.asarray(mj.eval_deriv(t_local, 3), dtype=np.float64)
            setpoints.append(s)

        # --- Store EXACTLY in the Gurobi winner fields ---
        # 中文:存进和 Gurobi 路径完全相同的那批"赢家字段",这样后续 commit/可视化/诊断代码
        # 不用区分用的是哪套求解器。successful_factor=1.0 只是个占位值(MINCO 不扫 factor)。
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
        """中文:局部轨迹优化的"分流入口"。配置选 minco 就走新算法 plan_local_trajectory_minco;
        否则走下面这套老基线:对每个 factor(时间缩放)依次——做凸走廊分解(SFC)→ 跑 Gurobi QP,
        第一个解出可行轨迹的 factor 就当赢家;全失败则把 factor 窗口整体往大移、下拍再试。"""
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
        # 中文:选做凸走廊分解时当"墙"的点云——真实感知模式(gazebo/hardware)把未知区也当障碍
        # (保守);否则只用已占据点。
        if self.par.sim_env in ("gazebo", "hardware"):
            base_map = list(self.hgp_manager.vec_uo) if hasattr(self.hgp_manager, "vec_uo") else []
        else:
            base_map = list(self.hgp_manager.vec_o) if hasattr(self.hgp_manager, "vec_o") else []
        # Filter out floor voxels at z_min
        # 中文:把贴近地面(z_min)的体素滤掉,免得地板被当成障碍把走廊压扁。
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
        # 中文:静态/"动态最坏情况"两种假设下,空间走廊不随时间变,可以一次性预算好所有 factor 共用;
        # 真·动态(下面 else 分支)则每个 factor 都要按时间分层重算走廊。
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

        # 中文:逐个 factor 顺序尝试(C++ 是并行,这里因 GIL 串行,见文件顶部说明)。
        # 找到第一个能解出可行轨迹的就 break,失败的解当"次优解"留着可视化。
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
            # 中文:全失败——把 factor 窗口整体往大挪(给轨迹更多时间、放慢=更好解);
            # 挪到顶了就重置回初始窗口。下一拍用新窗口再试。
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
        # 中文:成功了——下一拍把 factor 窗口以这次的赢家为中心重设,优先在它附近搜。
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
        """中文:单个 factor 的一次 Gurobi 局部求解。按这个 factor 算各段时间 → 准备空间约束
        (预算好的直接用,否则按时间分层重做凸走廊分解)→ 配置求解器(起点/终点/时间/走廊)
        → 跑 Gurobi 生成轨迹。返回 (是否成功, Gurobi 耗时, 分解耗时, 安全多面体, 目标点, pwp, 控制点)。"""
        P = max(0, len(global_path) - 1)
        if P == 0:
            return False, 0.0, 0.0, [], [], PieceWisePol(), []
        N = int(self.par.num_N)
        if N == 0:
            return False, 0.0, 0.0, [], [], PieceWisePol(), []

        # 中文:这个 factor 下每个时间分层的结束时刻(第 n 层 = (n+1)*缩放后的步长)。
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

        # 中文:真正调 Gurobi 求这条轨迹;求解器报错(运行期才知道)时兜住,当本 factor 失败。
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
        """中文:静态假设下,给每个轨迹段算一个"最坏情况结束时刻",用来给凸走廊膨胀留余量。
        分配策略:尽量把时间堆在第一个多面体上、最后几个多面体各只占 1 段(P-1 个),
        这样保守地估计每段最晚什么时候还可能停在该走廊内。返回各段累计结束时刻的列表。"""
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
        """中文:把新算出的 goal_setpoints "接"到正在飞的计划队列上(commit/splice)。
        做法:先从队尾砍掉 k_value 个旧点(那段还没飞、要被新轨迹替换),再把新点接到后面。
        正好对应 find_A_and_Atime 里"从倒数第 k_value 个点起规划":砍掉的就是从 A 往后那一截。
        队列点数还不够砍时退而求其次,不砍、只调小 k_value。"""
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
                # 中文:砍掉队尾 k_value 个旧点(A 之后那段),再把新轨迹采样点接上去。
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
        # 中文:攒够若干拍后,用历史计算耗时的平均值初始化预估耗时,切到 k_value 自适应模式。
        n = max(1, len(self.store_computation_times))
        s = sum(self.store_computation_times)
        self.est_comp_time = s / n
        self.use_adapt_k_value = True

    # ------------------------------------------------------------------
    # getNextGoal — pull the front of the plan, fill yaw, apply frame transform
    # ------------------------------------------------------------------
    def get_next_goal(self) -> Tuple[bool, RobotState]:
        """中文:控制频率回调,吐出"下一个要飞的目标点"。从计划队列头部取一个点(队列里还剩
        多于 1 个就把它弹出,只剩 1 个就反复用它=悬停),填好 yaw/dyaw,真机模式再转回本地系。
        返回 (是否有有效目标, 目标点)。这是规划器对外的最终输出。"""
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
        # 中文:队列里还有不止一个点才弹出队头;只剩一个就留着(原地悬停在最后一个点)。
        if len(local_plan) > 1:
            with self._mtx:
                if self.plan:
                    self.plan.popleft()

        # 中文:决定这一拍的偏航角 yaw。
        # - 连续规划失败太多次:原地缓慢自转(转着扫描环境,等地图看清楚);
        # - 计划快空了(<5 个点)且不在转头/躲避态:yaw 先不动,免得乱转;
        # - 其余:按 get_desired_yaw 朝着该看的方向转。
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

        # 中文:真机模式——规划是在全局系做的,这里用初始位姿的逆变换把目标点转回飞控用的本地系。
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
        """中文:按当前状态算"该朝哪看"的目标偏航角并平滑过去:
        转头(YAWING)→ 朝最终目标方向;悬停躲避 → 朝悬停点;赶路/看见目标 → 朝飞行速度方向;
        到达 → 不转。算出目标角后用限速(w_max*dc)一步步逼近,避免猛甩头。
        转头态里还顺带判断"基本转到位了就切到 TRAVELING 开始赶路"。"""
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

        # 中文:转头态——朝向基本对上(<0.3rad)就开始赶路;转太久(>10s)且大致对上(<1rad)也放行,
        # 防止卡在原地一直转。
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
        # 中文:赶路时的偏航平滑——先用一阶低通(alpha_filter_dyaw)缩小一步要转的角度,
        # 再用 w_max*dc 限速,得到本拍 yaw 和角速度 dyaw。
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
        """中文:设置/更新用户最终目标 G_term,并据此驱动状态机。三种情况:
        - 新目标和旧的几乎重合(<0.1m)→ 忽略;
        - 飞行途中(赶路/看见目标)换目标 → 平滑更新,不打断当前飞行;
        - 否则(首次设目标或停着设)→ 完全重置:清空计划、回到当前点、进入转头(或跳过转头直接赶路)。"""
        # Skip duplicate
        if self.terminal_goal_initialized:
            cur = self.get_gterm()
            if float(np.linalg.norm(cur.pos - term_goal.pos)) < 0.1:
                return
        local_state = self.get_state()

        # Mid-flight smooth update
        # 中文:飞行途中换目标——只更新 G_term 和 G,保持队列继续飞,不重新转头。
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
        # 中文:完全重置——把计划清空、塞入当前状态当唯一起点,A/G 都设成当前点,
        # 失败计数清零,然后(除非配置跳过)进入转头态先把头转向目标方向。
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
        """中文:悬停躲避——飞机已经到达/在目标附近悬停时,如果有动态障碍(尤其是人)逼近,
        临时挪到一个安全点躲一下,威胁过去再飞回悬停点。
        机制:对每个逼近障碍按 1/距离² 加权求一个合斥力方向;斥力够强就沿这个方向退一步到 p_evasion,
        若该点未来仍被威胁或被占据,就绕着方向转一圈(_find_safe_evasion)找别的安全点;
        都安全且斥力变弱后,再把目标设回原悬停点 p_hover。返回是否成功找到落脚点。"""
        local_state = self.get_state()
        local_trajs = self.get_trajs()
        lookahead_window = 15.0
        lookahead_step = 0.5

        # 中文:某点是否"受威胁"——当前就离障碍很近,或在未来一段时间(lookahead)里障碍会逼近它。
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

        # 中文:对所有"逼近到触发距离内"的障碍,按 1/距离² 加权,把"远离它们"的方向叠加成合斥力 n_total。
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

            # 中文:沿斥力方向退 hover_avoidance_h 距离当落脚点;2D 模式保持原高度,否则把 z 钳在安全层内。
            p_evasion = local_state.pos + self.par.hover_avoidance_h * direction
            if self.par.hover_avoidance_2d:
                p_evasion[2] = local_state.pos[2]
            else:
                p_evasion[2] = max(self.par.z_min + 0.5,
                                   min(p_evasion[2], self.par.z_max - 0.5))

            # 中文:落脚点未来仍会被威胁 → 绕方向转一圈找别的安全点;再被障碍占据 → 要求避开占据再找一次。
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
        # 中文:斥力已经变弱(威胁过去了)且当前在躲避态——只要原悬停点此刻安全,就把目标设回去,飞回悬停。
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
        """中文:在水平面内把初始躲避方向依次旋转一些角度(±30/±60/±90/180度),逐个试落脚点,
        返回第一个"不受威胁(且按需未被占据)"的点;都不行返回 None。"""
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
        """中文:真机专用——记录"本地系→全局系"的初始位姿变换(4x4 矩阵)及其逆、旋转部分、
        yaw 偏移,供 update_state / get_next_goal 在两个坐标系间来回转换。最后做个自检:
        逆变换乘回初始平移应得到原点(误差小才打印 READY TO FLY)。"""
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

    # 中文:把分段多项式轨迹的系数整体做坐标变换(正向=本地转全局,inverse=全局转本地)。
    # 因为多项式各项系数也是点坐标,逐项乘变换矩阵即可。
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
    # 中文:下面这几个 retrieve_* 是对外只读接口(对应 C++ 的 retrieve* API),把本拍的耗时统计、
    # 多面体走廊、目标点序列、控制点等拷一份给可视化/记录用。
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
