"""Python equivalents of the structs in include/sando/sando_type.hpp.

Kept deliberately close to the C++ field names so the port stays auditable.
NumPy arrays replace Eigen vectors/matrices. All vectors are shape (3,) and
matrices are 2D ndarrays unless noted.

中文说明:
这个文件是整个规划器的「数据结构定义文件」。它本身不做任何算法,只是把 C++
基线 (sando_type.hpp) 里的各种结构体翻译成 Python 版本,让其他模块 (全局
heat-A* 向导、MINCO 局部曲线优化、planner 状态机) 都用同一套数据格式。

里面主要有这几类东西:
- 状态/几何小结构: StateDeriv (位置/速度/加速度/jerk)、Polytope (凸多面体)、
  RobotState (无人机当前状态)。
- Parameters: 一大坨可调参数,默认值都写死在这里,运行时可被 ROS 参数覆盖。
- BasisConverter: 不同样条基 (MINVO / Bezier / B-spline) 之间的换算矩阵。
- PieceWisePol: 分段三次多项式轨迹 (规划结果就用它存)。
- DynTraj: 动态障碍物 (或别的无人机) 的轨迹,支持分段多项式或解析表达式两种形式。

字段名故意和 C++ 保持一致,方便对照检查这次移植有没有翻译错。
所有「向量」都是形状 (3,) 的 numpy 数组,「矩阵」是二维 ndarray,除非特别说明。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Simple value structs
# ---------------------------------------------------------------------------

# 一个时刻的运动状态导数: 位置、速度、加速度、jerk(加速度的变化率)
@dataclass
class StateDeriv:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    jerk: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Polytope:
    """Convex polytope A x <= b in half-space form.

    中文: 一个凸多面体,用「半空间」形式表示,即满足 A·x <= b 的所有点。
    每一行 (A 的一行 + b 的一个分量) 就是一个平面,把空间切成「合法/不合法」两边;
    多个平面取交集就围出一块凸的可行区域。
    """
    A: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))


# 无人机状态机的几种状态: 原地转向 -> 飞行 -> 看见目标 -> 到达目标 (HOVER_AVOIDING 是悬停避障)
class DroneStatus(IntEnum):
    YAWING = 0
    TRAVELING = 1
    GOAL_SEEN = 2
    GOAL_REACHED = 3
    HOVER_AVOIDING = 4


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# 规划器的全部可调参数,默认值写死在这里,运行时可被 ROS 参数 (sando.yaml) 覆盖。
# 按功能分组,下面每组的注释说明这一块大致管什么。
@dataclass
class Parameters:
    # Sim environment
    # 仿真环境相关: 用哪个仿真器、是否用全局点云
    sim_env: str = "gazebo"
    use_global_pc: bool = True

    # Vehicle
    vehicle_type: str = "uav"
    provide_goal_in_global_frame: bool = False
    state_already_in_global_frame: bool = False
    use_hardware: bool = False

    # Flight
    flight_mode: str = "terminal_goal"
    visual_level: int = 2

    # Global planner
    # 全局规划器: heat-A* 在体素网格上找一条「大方向」引导路径。
    # heuristic_weight 是 A* 启发权重(>1 更贪心更快但不一定最优);
    # inflation_hgp 是障碍膨胀半径; x/y/z_min/max 是规划空间边界;
    # use_free_start/goal 表示起点/终点附近放宽,允许从被占据格子起步/到达。
    global_planner: str = "astar_heat"
    global_planner_verbose: bool = False
    global_planner_heuristic_weight: float = 2.0
    factor_hgp: float = 1.0
    inflation_hgp: float = 0.45
    x_min: float = -200.0
    x_max: float = 200.0
    y_min: float = -200.0
    y_max: float = 200.0
    z_min: float = 0.5
    z_max: float = 6.0
    drone_radius: float = 0.2
    hgp_timeout_duration_ms: int = 1000
    max_num_expansion: int = 100000
    use_free_start: bool = True
    free_start_factor: float = 2.0
    use_free_goal: bool = False
    free_goal_factor: float = 2.0

    # LoS post processing
    # 全局路径的后处理: LoS = line of sight 视线检查,把能直连的点之间多余拐点去掉,
    # 让路径更短更直; min_len/min_turn 控制保留的最短段长和最小转角。
    los_cells: int = 0
    min_len: float = 1.0
    min_turn: float = 0.0

    # Path push visualization
    use_state_update: bool = True
    use_random_color_for_global_path: bool = False
    use_path_push_for_visualization: bool = False

    # Local solver selection: 'minco' (default, MINCO + per-class avoidance) or
    # 'gurobi' (legacy SFC + factor sweep + Gurobi QP, kept as oracle).
    # 中文: 选用哪个局部轨迹求解器。
    #   'minco' = 新算法(默认): MINCO 最小 jerk 五次曲线 + 按障碍类别分别避障
    #             (人=硬约束凸包, 墙=软场), 解析梯度优化。
    #   'gurobi' = 旧的 SANDO 基线(安全走廊 + 系数扫描 + Gurobi 二次规划), 留着当对照标准。
    local_solver: str = "minco"

    # Decomposition
    # 空间分解(把自由空间切成凸盒/走廊)相关; environment_assumption 说明环境假设
    # (static 静态 / dynamic 动态)。主要给旧 gurobi 路线用。
    environment_assumption: str = "dynamic"
    sfc_size: List[float] = field(default_factory=lambda: [3.0, 3.0, 3.0])
    min_dist_from_agent_to_traj: float = 10.0
    use_shrinked_box: bool = False
    shrinked_box_size: float = 0.0

    # Map
    # 局部地图相关: wdx/wdy/wdz 是地图窗口尺寸, res 是体素分辨率(米/格),
    # 格子越小越精细但越慢。
    map_buffer: float = 1.0
    center_shift_factor: float = 0.5
    initial_wdx: float = 15.0
    initial_wdy: float = 15.0
    initial_wdz: float = 4.0
    min_wdx: float = 15.0
    min_wdy: float = 15.0
    min_wdz: float = 4.0
    res: float = 0.3

    # Communication delay
    # 通信延迟补偿: 别的无人机/障碍信息传过来有延迟, 用这些参数把它们的位置
    # 适当外推/膨胀一点, 防止因为信息滞后撞上。
    use_comm_delay_inflation: bool = True
    comm_delay_inflation_alpha: float = 0.2
    comm_delay_inflation_max: float = 0.1
    comm_delay_filter_alpha: float = 0.9

    # Sim
    depth_camera_depth_max: float = 10.0
    fov_visual_depth: float = 10.0
    fov_visual_x_deg: float = 76.0
    fov_visual_y_deg: float = 47.0

    # Segments
    # 全局路径分段相关: max_dist_vertexes 限制相邻顶点最大间距(太远就细分);
    # heat_weight 等是把「热度」(危险程度)加进路径代价的权重。
    max_dist_vertexes: float = 1.0
    w_unknown: float = 0.0
    w_align: float = 0.0
    decay_len_cells: float = 100.0
    w_side: float = 0.0
    heat_weight: float = 10.0

    # Heat
    # 热度图(heat map): 全局 A* 不只看「能不能过」, 还给每个格子算一个「危险热度」,
    # 越靠近障碍/动态物体热度越高, 路径会自动绕开高热度区。
    #   dynamic_* 是动态障碍(尤其是人)产生的热度; static_* 是静态障碍产生的热度。
    #   alpha/p/q/Hmax/tau 这些是热度公式的形状参数(幅度、衰减次数、上限等)。
    use_heat_map: bool = True
    dynamic_heat_enabled: bool = True
    dynamic_as_occupied_current: bool = True
    dynamic_as_occupied_future: bool = False
    use_only_curr_pos_for_dynamic_obst: bool = False
    heat_alpha0: float = 0.2
    heat_alpha1: float = 1.0
    heat_p: int = 2
    heat_q: int = 2
    heat_tau_ratio: float = 0.5
    heat_gamma: float = 0.0
    heat_Hmax: float = 2.0
    dyn_base_inflation_m: float = 0.1
    dyn_heat_tube_radius_m: float = 0.5
    heat_num_samples: int = 15

    static_heat_enabled: bool = True
    static_heat_alpha: float = 1.0
    static_heat_p: int = 2
    static_heat_Hmax: float = 5.0
    static_heat_rmax_m: float = 1.0
    static_heat_default_radius_m: float = 0.5
    static_heat_boundary_only: bool = True
    static_heat_apply_on_unknown: bool = False
    static_heat_exclude_dynamic: bool = True

    # Soft-cost
    use_soft_cost_obstacles: bool = True
    obstacle_soft_cost: float = 5.0

    # Optimization
    # 局部优化相关的硬指标: horizon 规划时域(秒), dc 采样时间步;
    # v_max/a_max/j_max 是速度/加速度/jerk 上限; drone_bbox 是无人机包围盒尺寸;
    # goal_radius/goal_seen_radius 判定「到达/看见目标」的距离阈值。
    horizon: float = 15.0
    dc: float = 0.01
    dynamic_constraint_type: str = "Linf"
    v_max: float = 10.0
    a_max: float = 20.0
    j_max: float = 100.0
    drone_bbox: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.2])
    goal_radius: float = 0.5
    goal_seen_radius: float = 2.0

    # SANDO
    # 旧 SANDO 基线特有的参数: num_P/num_N 是多项式段数/控制点数;
    # factor_* 是时间缩放因子的扫描设置(收敛失败就放大时间重试);
    # jerk_smooth_weight 是 jerk 平滑代价权重。这些主要给 gurobi 路线用。
    num_P: int = 3
    num_N: int = 5
    use_dynamic_factor: bool = True
    dynamic_factor_k_radius: float = 0.4
    dynamic_factor_initial_mean: float = 1.5
    factor_initial: float = 1.0
    factor_final: float = 2.5
    factor_constant_step_size: float = 0.1
    obst_max_vel: float = 0.5
    obst_position_error: float = 0.0
    inflate_unknown_boundary: bool = True
    max_gurobi_comp_time_sec: float = 1.0
    jerk_smooth_weight: float = 10.0
    using_variable_elimination: bool = True

    # Anytime-feasible / gatekeeper (computation-invariant safety). Defaults OFF so the
    # baseline (and every golden) is byte-identical; the deterministic certificate-first
    # stack activates only when these are set.
    minco_time_budget_ms: float = 0.0   # >0 -> hard per-replan compute deadline for plan_minco
    minco_use_topology: bool = False    # True -> deterministic H-signature passing-side seed
    minco_w_time: float = 10.0          # MINCO time-anchor weight: HIGHER -> faster (pushes time
                                        #   allocation toward v_max/a_max), lower -> smoother/slower
    # speed-aware per-class avoidance: ramp v_max DOWN as a human gets close so the planner can
    # SWERVE AROUND instead of braking to a stop; full speed when clear. Off when slow_vmax<=0.
    minco_human_slow_vmax: float = 0.0  # routing-feasible v_max used right next to a human (0=off)
    minco_human_slow_near: float = 3.0  # m: at/below this human distance -> slow_vmax
    minco_human_slow_far: float = 9.0   # m: at/above this human distance -> full v_max

    # Dynamic obstacles
    traj_lifetime: float = 7.0

    # Dynamic k_value
    # k_value 是「提交点」索引: 重规划时已经飞过的前 k 个点不动(已承诺执行),
    # 从第 k 点之后才接新轨迹(commit/splice 拼接)。这组参数控制 k 怎么自适应调整。
    num_replanning_before_adapt: int = 10
    default_k_value: int = 50
    alpha_k_value_filtering: float = 0.9
    k_value_factor: float = 5.0

    # Yaw
    # 偏航(机头朝向)相关: w_max 是最大角速度, 转向时用 w_max_yawing;
    # spinning 那几个是「原地自转扫视」行为的阈值。
    alpha_filter_dyaw: float = 0.8
    w_max: float = 1.0
    w_max_yawing: float = 0.5
    skip_initial_yawing: bool = False
    yaw_spinning_threshold: int = 10000
    yaw_spinning_dyaw: float = 1.0

    # Sim env / goal
    force_goal_z: bool = True
    default_goal_z: float = 2.0

    # Debug
    debug_verbose: bool = False

    # Hover avoidance
    # 悬停避障: 当别的轨迹/障碍逼近到 d_trigger 内时, 进入 HOVER_AVOIDING 状态,
    # 原地(或 2D 平面内)用一个排斥向量躲一下, 而不是硬挤过去。
    ignore_other_trajs: bool = False
    hover_avoidance_enabled: bool = False
    hover_avoidance_2d: bool = True
    hover_avoidance_d_trigger: float = 4.0
    hover_avoidance_h: float = 3.0
    hover_avoidance_min_repulsion_norm: float = 0.01

    @classmethod
    def from_ros_node(cls, node) -> "Parameters":
        """Populate a Parameters instance from declared rclpy node parameters.

        Reads with declare_parameter; if a value is not declared on the node we
        fall back to the dataclass default. This mirrors how sando_node.cpp
        loads parameters via declare_parameter / get_parameter.

        中文: 从一个 ROS 节点把参数读出来, 填进一个 Parameters 对象。
        做法: 遍历 dataclass 的每个字段, 用上面写死的默认值去 declare_parameter,
        然后再 get_parameter 读回实际值(如果 yaml/命令行里配了, 就用配的, 否则保持默认)。
        和 C++ 基线 sando_node.cpp 加载参数的方式一致。
        """
        p = cls()
        from rclpy.parameter import Parameter as RclParameter  # local import
        # 逐个字段: declare(给默认) -> get(取实际值)。declare 重复声明会抛异常, 所以 try 包住。
        for f in p.__dataclass_fields__.values():  # type: ignore[attr-defined]
            name = f.name
            default = getattr(p, name)
            if isinstance(default, list):
                ros_default = list(default)
            else:
                ros_default = default
            try:
                node.declare_parameter(name, ros_default)
            except Exception:
                pass
            try:
                val = node.get_parameter(name).value
                if val is not None:
                    setattr(p, name, val)
            except Exception:
                pass
        # Map alias used in C++ sando.yaml
        # C++ 那边 yaml 里分辨率叫 sando_map_res, 这里额外认一下这个别名, 读到就覆盖 res。
        try:
            node.declare_parameter("sando_map_res", p.res)
            v = node.get_parameter("sando_map_res").value
            if v is not None:
                p.res = float(v)
        except Exception:
            pass
        return p


# ---------------------------------------------------------------------------
# Basis converter (MINVO / Bezier / B-spline)
# ---------------------------------------------------------------------------
# 中文: 下面这堆 _A_... 矩阵是不同样条「基」之间的换算矩阵(每段都是定义在 [0,1] 上的三次曲线)。
# 同一条曲线可以用 B-spline、MINVO、Bezier 三种控制点来表示, 它们之间是线性变换。
# 为什么要换: B-spline 适合连续性平滑, 但 MINVO/Bezier 的控制点能给出曲线的「紧致凸包」,
# 做避障(让整条段待在某个凸盒里)时用凸包更准。这里的常数是预先算好的变换系数。

_A_pos_mv_rest = np.array([
    [-3.4416308968564117698463178385282, 6.9895481477801393310755884158425, -4.4622887507045296828778191411402, 0.91437149978080234369315348885721],
    [6.6792587327074839365081970754545, -11.845989901556746914934592496138, 5.2523596690684613008670567069203, 0.0],
    [-6.6792587327074839365081970754545, 8.1917862965657040064115790301003, -1.5981560640774179482548333908198, 0.085628500219197656306846511142794],
    [3.4416308968564117698463178385282, -3.3353445427890959784633650997421, 0.80808514571348655231020075007109, -8.4567769453869345852581318467855e-18],
])

_A_vel_mv_rest = np.array([
    [1.4999999992328318931811281800037, -2.3660254034601951866889635311964, 0.9330127021136816189983420599674],
    [-2.9999999984656637863622563600074, 2.9999999984656637863622563600074, 0.0],
    [1.4999999992328318931811281800037, -0.6339745950054685996732928288111, 0.066987297886318325490506708774774],
])

_A_accel_mv_rest = np.array([[-1.0, 1.0], [1.0, 0.0]])

_A_pos_be_rest = np.array([
    [-1.0, 3.0, -3.0, 1.0],
    [3.0, -6.0, 3.0, 0.0],
    [-3.0, 3.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 0.0],
])

_A_pos_bs_seg0 = np.array([
    [-1.0, 3.0, -3.0, 1.0],
    [1.75, -4.5, 3.0, 0.0],
    [-0.9167, 1.5, 0.0, 0.0],
    [0.1667, 0.0, 0.0, 0.0],
])
_A_pos_bs_seg1 = np.array([
    [-0.25, 0.75, -0.75, 0.25],
    [0.5833, -1.25, 0.25, 0.5833],
    [-0.5, 0.5, 0.5, 0.1667],
    [0.1667, 0.0, 0.0, 0.0],
])
_A_pos_bs_rest = np.array([
    [-0.1667, 0.5, -0.5, 0.1667],
    [0.5, -1.0, 0.0, 0.6667],
    [-0.5, 0.5, 0.5, 0.1667],
    [0.1667, 0.0, 0.0, 0.0],
])
_A_pos_bs_seg_last2 = np.array([
    [-0.1667, 0.5, -0.5, 0.1667],
    [0.5, -1.0, 0.0, 0.6667],
    [-0.5833, 0.5, 0.5, 0.1667],
    [0.25, 0.0, 0.0, 0.0],
])
_A_pos_bs_seg_last = np.array([
    [-0.1667, 0.5, -0.5, 0.1667],
    [0.9167, -1.25, -0.25, 0.5833],
    [-1.75, 0.75, 0.75, 0.25],
    [1.0, 0.0, 0.0, 0.0],
])


class BasisConverter:
    """Numpy port of BasisConverter from sando_type.hpp.

    Only the methods actually consumed by the Gurobi solver and visualizers
    are exposed. We compute B-spline→MINVO/Bezier on the fly when needed,
    rather than hard-coding the matrices, because numerical equivalence
    suffices for our purposes (Cholesky / direct inverse is stable).

    中文: BasisConverter 把上面那堆常数矩阵打包起来, 提供「B-spline 转 MINVO / Bezier」
    的换算方法。初始化时预先求好各基矩阵的逆, 用的时候直接矩阵相乘即可。
    只暴露了 Gurobi 求解器和可视化实际会用到的几个方法。
    """

    def __init__(self):
        self.A_pos_mv_rest = _A_pos_mv_rest.copy()
        self.A_pos_mv_rest_inv = np.linalg.inv(self.A_pos_mv_rest)
        self.A_vel_mv_rest = _A_vel_mv_rest.copy()
        self.A_vel_mv_rest_inv = np.linalg.inv(self.A_vel_mv_rest)
        self.A_accel_mv_rest = _A_accel_mv_rest.copy()
        self.A_accel_mv_rest_inv = np.linalg.inv(self.A_accel_mv_rest)
        self.A_pos_be_rest = _A_pos_be_rest.copy()
        self.A_pos_bs_seg0 = _A_pos_bs_seg0.copy()
        self.A_pos_bs_seg1 = _A_pos_bs_seg1.copy()
        self.A_pos_bs_rest = _A_pos_bs_rest.copy()
        self.A_pos_bs_seg_last2 = _A_pos_bs_seg_last2.copy()
        self.A_pos_bs_seg_last = _A_pos_bs_seg_last.copy()

    # ----- A matrices per segment -----
    # 返回每一段的 B-spline 基矩阵。注意头两段和尾两段是「端点段」, 用专门的矩阵
    # (因为 B-spline 在两端要钳制/重复节点保证端点插值), 中间段都用同一个 rest 矩阵。
    def get_A_bspline(self, num_pol: int) -> List[np.ndarray]:
        if num_pol < 4:
            return [self.A_pos_bs_rest.copy() for _ in range(num_pol)]
        out = [self.A_pos_bs_seg0.copy(), self.A_pos_bs_seg1.copy()]
        for _ in range(num_pol - 4):
            out.append(self.A_pos_bs_rest.copy())
        out.append(self.A_pos_bs_seg_last2.copy())
        out.append(self.A_pos_bs_seg_last.copy())
        return out

    def get_A_minvo(self, num_pol: int) -> List[np.ndarray]:
        return [self.A_pos_mv_rest.copy() for _ in range(num_pol)]

    def get_A_bezier(self, num_pol: int) -> List[np.ndarray]:
        return [self.A_pos_be_rest.copy() for _ in range(num_pol)]

    # ----- BS->target converters -----
    # 返回「把 B-spline 控制点换成 MINVO 控制点」的换算矩阵(每段一个)。
    # 公式: bs->mv = A_mv^{-1} @ A_bs, 即先回到多项式系数, 再投到 MINVO 基。
    def get_minvo_pos_converters(self, num_pol: int) -> List[np.ndarray]:
        # bs->mv = A_mv^-1 @ A_bs  (in [0,1] domain)
        As = self.get_A_bspline(num_pol)
        return [self.A_pos_mv_rest_inv @ A for A in As]

    def get_bezier_pos_converters(self, num_pol: int) -> List[np.ndarray]:
        Ainv_be = np.linalg.inv(self.A_pos_be_rest)
        As = self.get_A_bspline(num_pol)
        return [Ainv_be @ A for A in As]

    def get_minvo_vel_converters(self, num_pol: int) -> List[np.ndarray]:
        return [self.A_vel_mv_rest_inv.copy() for _ in range(num_pol)]

    def get_minvo_accel_converters(self, num_pol: int) -> List[np.ndarray]:
        return [self.A_accel_mv_rest_inv.copy() for _ in range(num_pol)]


# ---------------------------------------------------------------------------
# Piecewise polynomial trajectory
# ---------------------------------------------------------------------------

@dataclass
class PieceWisePol:
    """Piecewise cubic, segment i spans [times[i], times[i+1]), local
    parameter u = t - times[i]. Stored coefficients are (a, b, c, d) such that
    p(t) = a u^3 + b u^2 + c u + d on each segment.

    中文: 分段三次多项式轨迹(规划结果就用它存)。
    第 i 段覆盖时间区间 [times[i], times[i+1]), 段内用「局部时间」u = t - times[i]
    (即从这段开头算起的相对时间), 三个轴(x/y/z)各存一组系数 (a,b,c,d):
        p(u) = a·u^3 + b·u^2 + c·u + d
    给定一个绝对时间 t, 先找到它落在哪一段, 再代入该段公式求位置/速度/加速度。
    """
    times: List[float] = field(default_factory=list)
    coeff_x: List[np.ndarray] = field(default_factory=list)
    coeff_y: List[np.ndarray] = field(default_factory=list)
    coeff_z: List[np.ndarray] = field(default_factory=list)

    def clear(self) -> None:
        self.times.clear()
        self.coeff_x.clear()
        self.coeff_y.clear()
        self.coeff_z.clear()

    def get_duration(self) -> float:
        if not self.times:
            return 0.0
        return self.times[-1] - self.times[0]

    def get_end_time(self) -> float:
        return self.times[-1] if self.times else 0.0

    # 在单个轴上求值: order=0 位置, 1 速度, 2 加速度。
    # 边界处理: t 超过末尾就钳到最后一段的终点; t 在起点之前就用第一段的起点。
    def _eval_axis(self, coeffs: List[np.ndarray], t: float, order: int) -> float:
        if not self.times or not coeffs:
            return 0.0
        # t 已经到/超过轨迹末尾: 用最后一段, 局部时间取这段的全长(即段末)
        if t >= self.times[-1]:
            u = self.times[-1] - self.times[-2]
            c = coeffs[-1]
            return _poly3_deriv(c, u, order)
        if t < self.times[0]:
            return _poly3_deriv(coeffs[0], 0.0, order)
        # 线性扫一遍找到 t 落在哪一段
        for i in range(len(self.times) - 1):
            if self.times[i] <= t < self.times[i + 1]:
                u = t - self.times[i]
                return _poly3_deriv(coeffs[i], u, order)
        return 0.0

    def eval(self, t: float) -> np.ndarray:
        return np.array([
            self._eval_axis(self.coeff_x, t, 0),
            self._eval_axis(self.coeff_y, t, 0),
            self._eval_axis(self.coeff_z, t, 0),
        ])

    def velocity(self, t: float) -> np.ndarray:
        return np.array([
            self._eval_axis(self.coeff_x, t, 1),
            self._eval_axis(self.coeff_y, t, 1),
            self._eval_axis(self.coeff_z, t, 1),
        ])

    def acceleration(self, t: float) -> np.ndarray:
        return np.array([
            self._eval_axis(self.coeff_x, t, 2),
            self._eval_axis(self.coeff_y, t, 2),
            self._eval_axis(self.coeff_z, t, 2),
        ])


# 三次多项式 a·u^3+b·u^2+c·u+d 在局部时间 u 处的第 order 阶导数(0=值,1=一阶,...)
def _poly3_deriv(c: np.ndarray, u: float, order: int) -> float:
    a, b, cc, d = float(c[0]), float(c[1]), float(c[2]), float(c[3])
    if order == 0:
        return a * u * u * u + b * u * u + cc * u + d
    if order == 1:
        return 3.0 * a * u * u + 2.0 * b * u + cc
    if order == 2:
        return 6.0 * a * u + 2.0 * b
    if order == 3:
        return 6.0 * a
    return 0.0


# ---------------------------------------------------------------------------
# Dynamic trajectory (piecewise or analytic via sympy)
# ---------------------------------------------------------------------------

class _AnalyticExpr:
    """Tiny lambdified wrapper around a string expression of variable 't'.

    Falls back to numexpr-style eval through Python's compiler when sympy is
    not available — the strings used in the simulator are simple (sin, cos,
    +-*/) so this works in practice without an extra dependency.

    中文: 把一段「以 t 为变量的字符串表达式」(比如 "2*sin(0.5*t)")编译成一个可调用函数。
    仿真器里给动态障碍轨迹经常是这种解析式。这里不依赖 sympy, 直接用 Python 的
    compile/eval, 但把内置函数禁掉、只放行白名单里的数学函数(sin/cos/sqrt 等),
    避免 eval 执行危险代码。表达式都很简单所以够用。
    """
    __slots__ = ("src", "_fn")

    _ALLOWED_NAMES = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan, "asin": np.arcsin,
        "acos": np.arccos, "atan": np.arctan, "atan2": np.arctan2,
        "sqrt": np.sqrt, "exp": np.exp, "log": np.log, "pow": np.power,
        "pi": float(np.pi), "e": float(np.e), "abs": np.abs,
    }

    def __init__(self, src: str):
        self.src = src
        if not src:
            self._fn = None
            return
        # Replace ExprTk-style operators if needed (^ -> **)
        # C++ 那边的表达式用 ^ 表示乘方, Python 里 ^ 是异或, 所以换成 **
        py_src = src.replace("^", "**")
        code = compile(py_src, "<analytic>", "eval")
        names = dict(self._ALLOWED_NAMES)

        # 求值时把 t 塞进命名空间; 禁用 __builtins__ 防止 eval 执行任意代码(安全沙箱)
        def fn(t: float) -> float:
            names["t"] = t
            return float(eval(code, {"__builtins__": {}}, names))

        self._fn = fn

    def __call__(self, t: float) -> float:
        return 0.0 if self._fn is None else self._fn(t)


@dataclass
class DynTraj:
    """Dynamic obstacle (or agent) trajectory.

    Holds either a piecewise cubic (`mode == "Piecewise"`) or analytic
    expressions in `t` (`mode == "Analytic"`) for x, y, z.

    中文: 一个动态障碍物(或别的无人机, is_agent=True)的轨迹。
    两种存法二选一(mode 决定):
      - "Piecewise": 用上面的 PieceWisePol 分段多项式存(一般是感知/预测给出的)。
      - "Analytic":  用 x/y/z 三个解析表达式字符串存(仿真里已知运动规律时用)。
    另外还带 bbox(包围盒)、id、ekf/poly 协方差(不确定度)、通信延迟等元信息,
    给避障时膨胀和加安全裕度用。
    """
    mode: str = "Analytic"  # "Piecewise" | "Analytic"
    pwp: PieceWisePol = field(default_factory=PieceWisePol)
    traj_x: str = ""
    traj_y: str = ""
    traj_z: str = ""
    traj_vx: str = ""
    traj_vy: str = ""
    traj_vz: str = ""
    _expr_x: Optional[_AnalyticExpr] = None
    _expr_y: Optional[_AnalyticExpr] = None
    _expr_z: Optional[_AnalyticExpr] = None
    _expr_vx: Optional[_AnalyticExpr] = None
    _expr_vy: Optional[_AnalyticExpr] = None
    _expr_vz: Optional[_AnalyticExpr] = None
    analytic_compiled: bool = False

    ekf_cov_p: np.ndarray = field(default_factory=lambda: np.zeros(3))
    ekf_cov_q: np.ndarray = field(default_factory=lambda: np.zeros(3))
    poly_cov: np.ndarray = field(default_factory=lambda: np.zeros(3))
    control_points: List[np.ndarray] = field(default_factory=list)  # 3x4 each
    bbox: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5]))
    goal: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    is_agent: bool = False
    id: int = -1
    time_received: float = 0.0
    tracking_utility: float = 0.0
    communication_delay: float = 0.0

    def set_piecewise(self, pwp: PieceWisePol) -> None:
        self.mode = "Piecewise"
        self.pwp = pwp

    # 把解析式字符串编译成可调用函数(只在 mode=="Analytic" 时需要); 速度表达式可选,
    # 没给的话后面 velocity() 会用数值差分代替。编译失败返回 False 并打印原因。
    def compile_analytic(self) -> bool:
        try:
            self._expr_x = _AnalyticExpr(self.traj_x)
            self._expr_y = _AnalyticExpr(self.traj_y)
            self._expr_z = _AnalyticExpr(self.traj_z)
            self._expr_vx = _AnalyticExpr(self.traj_vx) if self.traj_vx else None
            self._expr_vy = _AnalyticExpr(self.traj_vy) if self.traj_vy else None
            self._expr_vz = _AnalyticExpr(self.traj_vz) if self.traj_vz else None
            self.analytic_compiled = True
            return True
        except Exception as ex:  # noqa: BLE001 — surface what failed
            print(f"[DynTraj] compile_analytic failed: {ex}")
            self.analytic_compiled = False
            return False

    # 求 t 时刻障碍的位置; 解析模式下若还没编译就先懒编译一次
    def eval(self, t: float) -> np.ndarray:
        if self.mode == "Piecewise":
            return self.pwp.eval(t)
        if not self.analytic_compiled and self.traj_x:
            self.compile_analytic()
        if not self.analytic_compiled:
            return np.zeros(3)
        return np.array([self._expr_x(t), self._expr_y(t), self._expr_z(t)])

    def velocity(self, t: float) -> np.ndarray:
        if self.mode == "Piecewise":
            return self.pwp.velocity(t)
        if self._expr_vx is not None and self._expr_vy is not None and self._expr_vz is not None:
            return np.array([self._expr_vx(t), self._expr_vy(t), self._expr_vz(t)])
        # Numerical fallback
        # 没给速度表达式: 用中心差分数值求导 (p(t+dt)-p(t-dt))/(2dt) 近似速度
        dt = 1e-3
        return (self.eval(t + dt) - self.eval(t - dt)) / (2 * dt)

    # 加速度: 解析模式没有专门的式子, 一律对速度再做一次中心差分
    def accel(self, t: float) -> np.ndarray:
        if self.mode == "Piecewise":
            return self.pwp.acceleration(t)
        dt = 1e-3
        return (self.velocity(t + dt) - self.velocity(t - dt)) / (2 * dt)


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------

# 无人机自身的完整状态: 时间戳 + 位置/速度/加速度/jerk + 偏航角 yaw 及其角速度 dyaw。
# 提供 set_zero/clone 和一组 set_xxx 便捷设值方法(可传一个三维数组或 x,y,z 三个标量)。
@dataclass
class RobotState:
    t: float = 0.0
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    jerk: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0
    dyaw: float = 0.0
    use_tracking_yaw: bool = False

    def set_zero(self) -> None:
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.accel = np.zeros(3)
        self.jerk = np.zeros(3)
        self.yaw = 0.0
        self.dyaw = 0.0

    def clone(self) -> "RobotState":
        return RobotState(
            t=self.t,
            pos=self.pos.copy(),
            vel=self.vel.copy(),
            accel=self.accel.copy(),
            jerk=self.jerk.copy(),
            yaw=self.yaw,
            dyaw=self.dyaw,
            use_tracking_yaw=self.use_tracking_yaw,
        )

    def set_pos(self, x, y=None, z=None) -> None:
        if y is None:
            self.pos = np.asarray(x, dtype=float).reshape(3)
        else:
            self.pos = np.array([x, y, z], dtype=float)

    def set_vel(self, x, y=None, z=None) -> None:
        if y is None:
            self.vel = np.asarray(x, dtype=float).reshape(3)
        else:
            self.vel = np.array([x, y, z], dtype=float)

    def set_accel(self, x, y=None, z=None) -> None:
        if y is None:
            self.accel = np.asarray(x, dtype=float).reshape(3)
        else:
            self.accel = np.array([x, y, z], dtype=float)

    def set_jerk(self, x, y=None, z=None) -> None:
        if y is None:
            self.jerk = np.asarray(x, dtype=float).reshape(3)
        else:
            self.jerk = np.array([x, y, z], dtype=float)
