"""Python equivalents of the structs in include/sando/sando_type.hpp.

Kept deliberately close to the C++ field names so the port stays auditable.
NumPy arrays replace Eigen vectors/matrices. All vectors are shape (3,) and
matrices are 2D ndarrays unless noted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Simple value structs
# ---------------------------------------------------------------------------

@dataclass
class StateDeriv:
    pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    jerk: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class Polytope:
    """Convex polytope A x <= b in half-space form."""
    A: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    b: np.ndarray = field(default_factory=lambda: np.zeros(0))


class DroneStatus(IntEnum):
    YAWING = 0
    TRAVELING = 1
    GOAL_SEEN = 2
    GOAL_REACHED = 3
    HOVER_AVOIDING = 4


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

@dataclass
class Parameters:
    # Sim environment
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
    los_cells: int = 0
    min_len: float = 1.0
    min_turn: float = 0.0

    # Path push visualization
    use_state_update: bool = True
    use_random_color_for_global_path: bool = False
    use_path_push_for_visualization: bool = False

    # Local solver selection: 'minco' (default, MINCO + per-class avoidance) or
    # 'gurobi' (legacy SFC + factor sweep + Gurobi QP, kept as oracle).
    local_solver: str = "minco"

    # Decomposition
    environment_assumption: str = "dynamic"
    sfc_size: List[float] = field(default_factory=lambda: [3.0, 3.0, 3.0])
    min_dist_from_agent_to_traj: float = 10.0
    use_shrinked_box: bool = False
    shrinked_box_size: float = 0.0

    # Map
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
    max_dist_vertexes: float = 1.0
    w_unknown: float = 0.0
    w_align: float = 0.0
    decay_len_cells: float = 100.0
    w_side: float = 0.0
    heat_weight: float = 10.0

    # Heat
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

    # Dynamic obstacles
    traj_lifetime: float = 7.0

    # Dynamic k_value
    num_replanning_before_adapt: int = 10
    default_k_value: int = 50
    alpha_k_value_filtering: float = 0.9
    k_value_factor: float = 5.0

    # Yaw
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
        """
        p = cls()
        from rclpy.parameter import Parameter as RclParameter  # local import
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

    def _eval_axis(self, coeffs: List[np.ndarray], t: float, order: int) -> float:
        if not self.times or not coeffs:
            return 0.0
        if t >= self.times[-1]:
            u = self.times[-1] - self.times[-2]
            c = coeffs[-1]
            return _poly3_deriv(c, u, order)
        if t < self.times[0]:
            return _poly3_deriv(coeffs[0], 0.0, order)
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
        py_src = src.replace("^", "**")
        code = compile(py_src, "<analytic>", "eval")
        names = dict(self._ALLOWED_NAMES)

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
        dt = 1e-3
        return (self.eval(t + dt) - self.eval(t - dt)) / (2 * dt)

    def accel(self, t: float) -> np.ndarray:
        if self.mode == "Piecewise":
            return self.pwp.acceleration(t)
        dt = 1e-3
        return (self.velocity(t + dt) - self.velocity(t - dt)) / (2 * dt)


# ---------------------------------------------------------------------------
# Robot state
# ---------------------------------------------------------------------------

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
