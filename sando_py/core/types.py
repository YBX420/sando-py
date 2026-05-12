"""Foundational data types ported from include/sando/sando_type.hpp.

Faithful translation: same struct/field names, same defaults.
Eigen vectors/matrices -> numpy ndarrays. Cross-references to C++ lines kept as
sando_type.hpp:NNN tags so we can diff against the original.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


# ----------------------------------------------------------------------
# sando_type.hpp:20  struct StateDeriv
# ----------------------------------------------------------------------
@dataclass
class StateDeriv:
    pos:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    vel:   np.ndarray = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    jerk:  np.ndarray = field(default_factory=lambda: np.zeros(3))


# ----------------------------------------------------------------------
# sando_type.hpp:28  struct Polytope  (Ax <= b)
# ----------------------------------------------------------------------
@dataclass
class Polytope:
    A: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    b: np.ndarray = field(default_factory=lambda: np.zeros((0, 1)))


# ----------------------------------------------------------------------
# sando_type.hpp:988  struct RobotState
# ----------------------------------------------------------------------
@dataclass
class RobotState:
    t:     float       = 0.0
    pos:   np.ndarray  = field(default_factory=lambda: np.zeros(3))
    vel:   np.ndarray  = field(default_factory=lambda: np.zeros(3))
    accel: np.ndarray  = field(default_factory=lambda: np.zeros(3))
    jerk:  np.ndarray  = field(default_factory=lambda: np.zeros(3))
    yaw:   float       = 0.0
    dyaw:  float       = 0.0
    use_tracking_yaw: bool = False

    # --- setters mirroring sando_type.hpp:1003-1049 ---
    def setTimeStamp(self, data: float) -> None:
        self.t = data

    def setPos(self, *args) -> None:
        if len(args) == 1:
            v = np.asarray(args[0], dtype=float).reshape(3)
        else:
            v = np.array(args, dtype=float)
        self.pos = v

    def setVel(self, *args) -> None:
        if len(args) == 1:
            v = np.asarray(args[0], dtype=float).reshape(3)
        else:
            v = np.array(args, dtype=float)
        self.vel = v

    def setAccel(self, *args) -> None:
        if len(args) == 1:
            v = np.asarray(args[0], dtype=float).reshape(3)
        else:
            v = np.array(args, dtype=float)
        self.accel = v

    def setJerk(self, *args) -> None:
        if len(args) == 1:
            v = np.asarray(args[0], dtype=float).reshape(3)
        else:
            v = np.array(args, dtype=float)
        self.jerk = v

    def setState(self, pos, vel, accel, jerk) -> None:
        self.pos   = np.asarray(pos,   dtype=float).reshape(3)
        self.vel   = np.asarray(vel,   dtype=float).reshape(3)
        self.accel = np.asarray(accel, dtype=float).reshape(3)
        self.jerk  = np.asarray(jerk,  dtype=float).reshape(3)

    def setYaw(self, data: float) -> None:
        self.yaw = data

    def setDYaw(self, data: float) -> None:
        self.dyaw = data

    def setZero(self) -> None:
        self.pos[:]   = 0.0
        self.vel[:]   = 0.0
        self.accel[:] = 0.0
        self.jerk[:]  = 0.0
        self.yaw  = 0.0
        self.dyaw = 0.0

    def printPos(self) -> None:
        print(f"Pos= {self.pos}")

    def print(self) -> None:
        print(f"Time= {self.t}")
        print(f"Pos= {self.pos}")
        print(f"Vel= {self.vel}")
        print(f"Accel= {self.accel}")

    def printHorizontal(self) -> None:
        print(
            f"Pos, Vel, Accel, Jerk= "
            f"{self.pos} {self.vel} {self.accel} {self.jerk}"
        )


# ----------------------------------------------------------------------
# sando_type.hpp:34  struct Parameters
# Field-by-field port. Order preserved for diffability.
# ----------------------------------------------------------------------
@dataclass
class Parameters:
    # --- Sim environment --- (sando_type.hpp:35-37)
    sim_env:       str  = ""
    use_global_pc: bool = False

    # --- UAV or Ground robot --- (sando_type.hpp:39-43)
    vehicle_type:                  str  = ""
    provide_goal_in_global_frame:  bool = False
    state_already_in_global_frame: bool = False
    use_hardware:                  bool = False

    # --- Flight mode --- (sando_type.hpp:45-46)
    flight_mode: str = ""

    # --- Visual level --- (sando_type.hpp:48-49)
    visual_level: int = 0

    # --- Global planner parameters --- (sando_type.hpp:51-69)
    global_planner:                  str    = ""
    global_planner_verbose:          bool   = False
    global_planner_heuristic_weight: float  = 0.0
    factor_hgp:                      float  = 0.0
    inflation_hgp:                   float  = 0.0
    x_min:                           float  = 0.0
    x_max:                           float  = 0.0
    y_min:                           float  = 0.0
    y_max:                           float  = 0.0
    z_min:                           float  = 0.0
    z_max:                           float  = 0.0
    drone_radius:                    float  = 0.0
    hgp_timeout_duration_ms:         int    = 0
    max_num_expansion:               int    = 0
    use_free_start:                  bool   = False
    free_start_factor:               float  = 0.0
    use_free_goal:                   bool   = False
    free_goal_factor:                float  = 0.0

    # --- LOS post processing parameters --- (sando_type.hpp:71-74)
    los_cells: int   = 0
    min_len:   float = 0.0
    min_turn:  float = 0.0

    # --- Path push visualization parameters --- (sando_type.hpp:76-79)
    use_state_update:                  bool = False
    use_random_color_for_global_path:  bool = False
    use_path_push_for_visualization:   bool = False

    # --- Decomposition parameters --- (sando_type.hpp:81-86)
    environment_assumption:        str         = ""
    sfc_size:                      List[float] = field(default_factory=list)
    min_dist_from_agent_to_traj:   float       = 0.0
    use_shrinked_box:              bool        = False
    shrinked_box_size:             float       = 0.0

    # --- Map parameters --- (sando_type.hpp:88-97)
    map_buffer:           float = 0.0
    center_shift_factor:  float = 0.0
    initial_wdx:          float = 0.0
    initial_wdy:          float = 0.0
    initial_wdz:          float = 0.0
    min_wdx:              float = 0.0
    min_wdy:              float = 0.0
    min_wdz:              float = 0.0
    res:                  float = 0.0

    # --- Communication delay parameters --- (sando_type.hpp:99-103)
    use_comm_delay_inflation:    bool  = False
    comm_delay_inflation_alpha:  float = 0.0
    comm_delay_inflation_max:    float = 0.0
    comm_delay_filter_alpha:     float = 0.0

    # --- Simulation parameters --- (sando_type.hpp:105-109)
    depth_camera_depth_max: float = 0.0
    fov_visual_depth:       float = 0.0
    fov_visual_x_deg:       float = 0.0
    fov_visual_y_deg:       float = 0.0

    # --- number of segments parameters --- (sando_type.hpp:111-117)
    max_dist_vertexes: float = 0.0
    w_unknown:         float = 0.0
    w_align:           float = 0.0
    decay_len_cells:   float = 0.0
    w_side:            float = 0.0
    heat_weight:       float = 0.0

    # --- Heat map parameters --- (sando_type.hpp:119-120)
    use_heat_map: bool = False

    # --- Dynamic heat --- (sando_type.hpp:122-131)
    dynamic_heat_enabled:                bool  = False
    dynamic_as_occupied_current:         bool  = False
    dynamic_as_occupied_future:          bool  = False
    use_only_curr_pos_for_dynamic_obst:  bool  = False
    heat_alpha0:                         float = 0.0
    heat_alpha1:                         float = 0.0
    heat_p:                              int   = 0
    heat_q:                              int   = 0
    heat_tau_ratio:                      float = 0.0
    heat_gamma:                          float = 0.0
    heat_Hmax:                           float = 0.0
    dyn_base_inflation_m:                float = 0.0
    dyn_heat_tube_radius_m:              float = 0.0
    heat_num_samples:                    int   = 0

    # --- Static heat --- (sando_type.hpp:133-138)
    static_heat_enabled:              bool  = False
    static_heat_alpha:                float = 0.0
    static_heat_p:                    int   = 0
    static_heat_Hmax:                 float = 0.0
    static_heat_rmax_m:               float = 0.0
    static_heat_default_radius_m:     float = 0.0
    static_heat_boundary_only:        bool  = False
    static_heat_apply_on_unknown:     bool  = False
    static_heat_exclude_dynamic:      bool  = False

    # --- Soft-cost --- (sando_type.hpp:140-142)
    use_soft_cost_obstacles: bool  = False
    obstacle_soft_cost:      float = 0.0

    # --- Optimization parameters --- (sando_type.hpp:144-153)
    horizon:                 float       = 0.0
    dc:                      float       = 0.0
    dynamic_constraint_type: str         = "Linf"   # "Linf" | "L1" | "L2"
    v_max:                   float       = 0.0
    a_max:                   float       = 0.0
    j_max:                   float       = 0.0
    drone_bbox:              List[float] = field(default_factory=list)
    goal_radius:             float       = 0.0
    goal_seen_radius:        float       = 0.0

    # --- SANDO specific parameters --- (sando_type.hpp:155-173)
    num_P:                          int   = 0
    num_N:                          int   = 0
    use_dynamic_factor:             bool  = False
    dynamic_factor_k_radius:        float = 0.0
    dynamic_factor_initial_mean:    float = 0.0
    factor_initial:                 float = 1.0
    factor_final:                   float = 5.0
    factor_constant_step_size:      float = 0.1
    obst_max_vel:                   float = 0.0
    obst_position_error:            float = 0.0
    inflate_unknown_boundary:       bool  = True
    max_gurobi_comp_time_sec:       float = 0.0
    jerk_smooth_weight:             float = 0.0
    using_variable_elimination:     bool  = True

    # --- MIQP / time-layered corridor (C++ ``use_time_layered_polytopes``) ---
    # When True, each trajectory segment may choose from any of the
    # ``num_P`` corridor polytopes (binary indicator vars + "at least one
    # active" constraint, MIQP). When False (default), segment ``i``
    # is hard-pinned to polytope ``parent_of_seg[i]``. Mirrors
    # sando.cpp ``createSafeCorridorConstraintsForPolytopeAtleastOne``.
    # OFF by default because MIQP is ~10× slower than QP.
    use_miqp_corridor:              bool  = False

    # --- findA lookahead ---
    # Two modes:
    #  * ``use_adaptive_lookahead=True``  — port of C++ ``findAandAtime``
    #    (sando.cpp:185-235): warm up for ``num_replanning_before_adapt``
    #    replan ticks using ``default_k_value`` as a fixed lookahead, then
    #    switch to an EMA-filtered estimate based on actual replan compute
    #    time (``k_value_factor * est_comp_time / dc``).
    #  * ``use_adaptive_lookahead=False`` — fall back to a fixed,
    #    user-configurable ``lookahead_replan_time`` (Python-only mode,
    #    kept for benchmarks / unit tests that want a deterministic A).
    use_adaptive_lookahead:  bool  = True
    lookahead_replan_time:   float = 0.05   # seconds ahead of t_now to anchor A
    # --- Velocity-heuristic shape ---
    threshold_distance_velocity: float = 5.0  # m; see utils.findVelocitiesInPath

    # --- Dynamic obstacles parameters --- (sando_type.hpp:175-176)
    traj_lifetime: float = 0.0

    # --- Dynamic k_value parameters --- (sando_type.hpp:178-182)
    num_replanning_before_adapt: int   = 0
    default_k_value:             int   = 0
    alpha_k_value_filtering:     float = 0.0
    k_value_factor:              float = 0.0

    # --- Yaw-related parameters --- (sando_type.hpp:184-190)
    alpha_filter_dyaw:       float = 0.0
    w_max:                   float = 0.0
    w_max_yawing:            float = 0.0
    skip_initial_yawing:     bool  = False
    yaw_spinning_threshold:  int   = 0
    yaw_spinning_dyaw:       float = 0.0

    # --- Simulation env parameters --- (sando_type.hpp:192-194)
    force_goal_z:    bool  = False
    default_goal_z:  float = 0.0

    # --- Debug flag --- (sando_type.hpp:196-197)
    debug_verbose: bool = False

    # --- Hover avoidance parameters --- (sando_type.hpp:199-205)
    ignore_other_trajs:                bool  = False
    hover_avoidance_enabled:           bool  = False
    hover_avoidance_2d:                bool  = True
    hover_avoidance_d_trigger:         float = 4.0
    hover_avoidance_h:                 float = 3.0
    hover_avoidance_min_repulsion_norm: float = 0.01
