"""sando_cpp_bridge — drop-in C++ engine for sando_py via ctypes (sando_capi.dll).

Exposes Parameters / RobotState / DynTraj / SANDO with the SAME API the Isaac loop uses,
so isaac_sando_loop.py only swaps its imports:
    from sando_py.types import Parameters, RobotState, DynTraj
    from sando_py.planner import SANDO
  ->
    from sando_cpp_bridge import Parameters, RobotState, DynTraj, SANDO

The heavy lifting (HGP heat-A* + per-class MINCO + orchestrator) runs in the golden-verified
C++ (cpp/include/sando_cpp/*.hpp) behind sando_capi.dll. This file only marshals data.
"""
from __future__ import annotations

import os
import ctypes as C
import numpy as np

# --------------------------------------------------------------------------
# locate + load the DLL (built by: g++ -shared cpp/capi/sando_capi.cpp ...)
# --------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# repo layout: <repo>/python/sando_cpp_bridge.py  +  <repo>/cpp/...
_REPO = os.path.dirname(_HERE)
_CANDIDATES = [
    os.path.join(_HERE, "sando_capi.dll"),
    os.path.join(_REPO, "cpp", "capi", "sando_capi.dll"),
    os.path.join(_REPO, "cpp", "build", "sando_capi.dll"),
]


def _load():
    for p in _CANDIDATES:
        if os.path.isfile(p):
            return C.CDLL(p), p
    raise FileNotFoundError(
        "sando_capi.dll not found. Build it:\n"
        "  cd cpp && g++ -O2 -shared -std=c++17 -o capi/sando_capi.dll capi/sando_capi.cpp "
        "-Iinclude -Ithird_party/eigen -Ithird_party -static -static-libgcc -static-libstdc++\n"
        f"searched: {_CANDIDATES}")


_LIB, DLL_PATH = _load()
_dbl = C.POINTER(C.c_double)


def _sig(name, restype, *argtypes):
    f = getattr(_LIB, name)
    f.restype = restype
    f.argtypes = list(argtypes)
    return f


_params_create = _sig("params_create", C.c_void_p)
_params_destroy = _sig("params_destroy", None, C.c_void_p)
_params_set_double = _sig("params_set_double", None, C.c_void_p, C.c_char_p, C.c_double)
_params_set_bool = _sig("params_set_bool", None, C.c_void_p, C.c_char_p, C.c_int)
_params_set_string = _sig("params_set_string", None, C.c_void_p, C.c_char_p, C.c_char_p)
_params_set_vec3 = _sig("params_set_vec3", None, C.c_void_p, C.c_char_p, C.c_double, C.c_double, C.c_double)
_traj_create = _sig("traj_create", C.c_void_p, C.c_int, C.c_double, C.c_double, C.c_double,
                    C.c_char_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_char_p, C.c_char_p)
_traj_destroy = _sig("traj_destroy", None, C.c_void_p)
_traj_eval = _sig("traj_eval", None, C.c_void_p, C.c_double, _dbl)
_sando_create = _sig("sando_create", C.c_void_p, C.c_void_p)
_sando_destroy = _sig("sando_destroy", None, C.c_void_p)
_sando_update_state = _sig("sando_update_state", None, C.c_void_p, _dbl, _dbl, _dbl, C.c_double)
_sando_update_occupancy = _sig("sando_update_occupancy", None, C.c_void_p, _dbl, C.c_int)
_sando_set_terminal_goal = _sig("sando_set_terminal_goal", None, C.c_void_p, _dbl)
_sando_add_traj = _sig("sando_add_traj", None, C.c_void_p, C.c_void_p, C.c_double)
_sando_replan = _sig("sando_replan", C.c_int, C.c_void_p, C.c_double, C.c_double)
_sando_get_next_goal = _sig("sando_get_next_goal", C.c_int, C.c_void_p, _dbl)
_sando_get_drone_status = _sig("sando_get_drone_status", C.c_int, C.c_void_p)
_sando_get_global_path = _sig("sando_get_global_path", C.c_int, C.c_void_p, _dbl, C.c_int)
_sando_get_corridor = _sig("sando_get_corridor", C.c_int, C.c_void_p, _dbl, C.c_int)


def _p(arr):
    """contiguous float64 -> double* (keeps a ref alive via the returned array)."""
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64).reshape(-1))
    return a, a.ctypes.data_as(_dbl)


# --------------------------------------------------------------------------
# Parameters — mirrors sando_py.types.Parameters dataclass: every real field is a
# known attribute (so hasattr() gates exactly like the dataclass); setattr pushes to C++.
# Defaults are the C++ struct defaults (== the Python dataclass defaults).
# --------------------------------------------------------------------------
_DEFAULTS = {
    # strings
    "sim_env": "gazebo", "vehicle_type": "uav", "flight_mode": "terminal_goal",
    "global_planner": "astar_heat", "local_solver": "minco",
    "environment_assumption": "dynamic", "dynamic_constraint_type": "Linf", "avoid_override": "",
    # bools
    "use_global_pc": True, "provide_goal_in_global_frame": False,
    "state_already_in_global_frame": False, "use_hardware": False,
    "global_planner_verbose": False, "use_free_start": True, "use_free_goal": False,
    "use_state_update": True, "use_random_color_for_global_path": False,
    "use_path_push_for_visualization": False, "use_shrinked_box": False,
    "use_comm_delay_inflation": True, "use_heat_map": True, "dynamic_heat_enabled": True,
    "dynamic_as_occupied_current": True, "dynamic_as_occupied_future": False,
    "use_only_curr_pos_for_dynamic_obst": False, "static_heat_enabled": True,
    "static_heat_boundary_only": True, "static_heat_apply_on_unknown": False,
    "static_heat_exclude_dynamic": True, "use_soft_cost_obstacles": True,
    "use_dynamic_factor": True, "inflate_unknown_boundary": True,
    "using_variable_elimination": True, "skip_initial_yawing": False, "force_goal_z": True,
    "debug_verbose": False, "ignore_other_trajs": False, "hover_avoidance_enabled": False,
    "hover_avoidance_2d": True,
    # ints
    "visual_level": 2, "hgp_timeout_duration_ms": 1000, "max_num_expansion": 100000,
    "los_cells": 0, "heat_p": 2, "heat_q": 2, "heat_num_samples": 15, "static_heat_p": 2,
    "num_P": 3, "num_N": 5, "num_replanning_before_adapt": 10, "default_k_value": 50,
    "yaw_spinning_threshold": 10000,
    # vec3 (lists)
    "sfc_size": [3.0, 3.0, 3.0], "drone_bbox": [0.2, 0.2, 0.2],
    # doubles
    "global_planner_heuristic_weight": 2.0, "factor_hgp": 1.0, "inflation_hgp": 0.45,
    "x_min": -200.0, "x_max": 200.0, "y_min": -200.0, "y_max": 200.0, "z_min": 0.5, "z_max": 6.0,
    "drone_radius": 0.2, "free_start_factor": 2.0, "free_goal_factor": 2.0, "min_len": 1.0,
    "min_turn": 0.0, "min_dist_from_agent_to_traj": 10.0, "shrinked_box_size": 0.0,
    "map_buffer": 1.0, "center_shift_factor": 0.5, "initial_wdx": 15.0, "initial_wdy": 15.0,
    "initial_wdz": 4.0, "min_wdx": 15.0, "min_wdy": 15.0, "min_wdz": 4.0, "res": 0.3,
    "comm_delay_inflation_alpha": 0.2, "comm_delay_inflation_max": 0.1,
    "comm_delay_filter_alpha": 0.9, "depth_camera_depth_max": 10.0, "fov_visual_depth": 10.0,
    "fov_visual_x_deg": 76.0, "fov_visual_y_deg": 47.0, "max_dist_vertexes": 1.0, "w_unknown": 0.0,
    "w_align": 0.0, "decay_len_cells": 100.0, "w_side": 0.0, "heat_weight": 10.0,
    "heat_alpha0": 0.2, "heat_alpha1": 1.0, "heat_tau_ratio": 0.5, "heat_gamma": 0.0,
    "heat_Hmax": 2.0, "dyn_base_inflation_m": 0.1, "dyn_heat_tube_radius_m": 0.5,
    "static_heat_alpha": 1.0, "static_heat_Hmax": 5.0, "static_heat_rmax_m": 1.0,
    "static_heat_default_radius_m": 0.5, "obstacle_soft_cost": 5.0, "horizon": 15.0, "dc": 0.01,
    "v_max": 10.0, "a_max": 20.0, "j_max": 100.0, "goal_radius": 0.5, "goal_seen_radius": 2.0,
    "dynamic_factor_k_radius": 0.4, "dynamic_factor_initial_mean": 1.5, "factor_initial": 1.0,
    "factor_final": 2.5, "factor_constant_step_size": 0.1, "obst_max_vel": 0.5,
    "obst_position_error": 0.0, "max_gurobi_comp_time_sec": 1.0, "jerk_smooth_weight": 10.0,
    "minco_time_budget_ms": 0.0, "minco_use_topology": False, "minco_w_time": 10.0,
    "minco_retime_overshoot": False,
    "minco_epsilon_track": 0.0, "minco_pass_behind": False, "minco_wall_margin": 0.0,
    "minco_w_vel": 100.0, "minco_w_accel": 100.0,
    "minco_human_slow_vmax": 0.0, "minco_human_slow_near": 3.0, "minco_human_slow_far": 9.0,
    "minco_sfc_radius": 0.0, "minco_w_corridor": 0.0, "recovery_enabled": True,
    "inflate_walls_by_body": False, "replan_dt": 0.0, "dynamic_speed_thresh": 0.0,
    "pred_horizon_s": 0.0, "use_spacetime_corridor": False, "stc_d_safe_dyn": 0.5,
    "stc_time_dt": 0.1, "stc_Th": 1.5, "stc_w_time": 0.0,
    "use_st_graph": False, "st_NS": 24, "st_NT": 16, "st_w_wait": 1.0,
    "traj_lifetime": 7.0, "alpha_k_value_filtering": 0.9, "k_value_factor": 5.0,
    "alpha_filter_dyaw": 0.8, "w_max": 1.0, "w_max_yawing": 0.5, "yaw_spinning_dyaw": 1.0,
    "default_goal_z": 2.0, "hover_avoidance_d_trigger": 4.0, "hover_avoidance_h": 3.0,
    "hover_avoidance_min_repulsion_norm": 0.01,
}


class Parameters:
    def __init__(self):
        object.__setattr__(self, "_handle", _params_create())
        object.__setattr__(self, "_vals", dict(_DEFAULTS))

    def __setattr__(self, name, value):
        if name in ("_handle", "_vals"):
            object.__setattr__(self, name, value)
            return
        if name not in _DEFAULTS:
            # match the dataclass: assigning an unknown field is not a real Parameters field;
            # the Isaac loop never does this (it gates on hasattr), but be strict to mirror it.
            raise AttributeError(f"Parameters has no field '{name}'")
        self._vals[name] = value
        h = self._handle
        key = name.encode()
        if isinstance(value, str):
            _params_set_string(h, key, value.encode())
        elif isinstance(value, bool):
            _params_set_bool(h, key, 1 if value else 0)
        elif isinstance(value, (list, tuple, np.ndarray)):
            a = np.asarray(value, dtype=float).reshape(-1)
            _params_set_vec3(h, key, float(a[0]), float(a[1]), float(a[2]))
        elif isinstance(value, (int, float)):
            _params_set_double(h, key, float(value))
        else:
            raise TypeError(f"unsupported Parameters value type for '{name}': {type(value)}")

    def __getattr__(self, name):
        vals = object.__getattribute__(self, "_vals")
        if name in vals:
            return vals[name]
        raise AttributeError(name)

    def __del__(self):
        try:
            _params_destroy(self._handle)
        except Exception:
            pass


# --------------------------------------------------------------------------
# RobotState — plain data holder (pos/vel/accel/jerk/yaw/dyaw), zero-arg ctor.
# --------------------------------------------------------------------------
class RobotState:
    def __init__(self):
        self.t = 0.0
        self.pos = np.zeros(3)
        self.vel = np.zeros(3)
        self.accel = np.zeros(3)
        self.jerk = np.zeros(3)
        self.yaw = 0.0
        self.dyaw = 0.0


# --------------------------------------------------------------------------
# DynTraj — analytic per-class obstacle. Fields set by the loop, then compile_analytic()
# builds the C++ traj; eval(t) and add_traj go through the C++ AnalyticExpr parser.
# --------------------------------------------------------------------------
class DynTraj:
    def __init__(self):
        self.id = -1
        self.mode = "Analytic"
        self.bbox = np.array([0.5, 0.5, 0.5])
        self.traj_x = self.traj_y = self.traj_z = "0.0"
        self.traj_vx = self.traj_vy = self.traj_vz = ""
        self.is_agent = False
        self._handle = None

    def compile_analytic(self):
        if self._handle is not None:
            _traj_destroy(self._handle)
        b = np.asarray(self.bbox, dtype=float).reshape(-1)

        def enc(s):
            return ("" if s is None else str(s)).encode()

        self._handle = _traj_create(int(self.id), float(b[0]), float(b[1]), float(b[2]),
                                    enc(self.traj_x), enc(self.traj_y), enc(self.traj_z),
                                    enc(self.traj_vx), enc(self.traj_vy), enc(self.traj_vz))
        return True

    def eval(self, t):
        if self._handle is None:
            self.compile_analytic()
        out = (C.c_double * 3)()
        _traj_eval(self._handle, float(t), C.cast(out, _dbl))
        return np.array([out[0], out[1], out[2]])

    def __del__(self):
        try:
            if self._handle is not None:
                _traj_destroy(self._handle)
        except Exception:
            pass


class _Goal:
    __slots__ = ("pos", "vel", "accel")

    def __init__(self, pos, vel, accel):
        self.pos = pos
        self.vel = vel
        self.accel = accel


# --------------------------------------------------------------------------
# SANDO — same API the Isaac loop calls.
# --------------------------------------------------------------------------
class SANDO:
    def __init__(self, par: Parameters):
        self._h = _sando_create(par._handle)
        self._gp_buf = (C.c_double * (3 * 4096))()  # global-path scratch
        self._corr_buf = (C.c_double * (9 * 64))()  # space-time corridor scratch (9 doubles/cuboid)

    def update_state(self, data: RobotState):
        _, pp = _p(data.pos)
        _, pv = _p(data.vel)
        _, pa = _p(data.accel)
        _sando_update_state(self._h, pp, pv, pa, float(getattr(data, "yaw", 0.0)))

    def update_occupancy_map_ptr(self, cloud):
        arr = np.ascontiguousarray(np.asarray(cloud, dtype=np.float64).reshape(-1, 3)) \
            if (cloud is not None and len(cloud) > 0) else np.zeros((0, 3))
        n = arr.shape[0]
        ptr = arr.ctypes.data_as(_dbl) if n > 0 else _dbl()
        _sando_update_occupancy(self._h, ptr, int(n))

    def set_terminal_goal(self, term_goal: RobotState):
        _, pp = _p(term_goal.pos)
        _sando_set_terminal_goal(self._h, pp)

    def add_traj(self, new_traj: DynTraj, current_time: float):
        if new_traj._handle is None:
            new_traj.compile_analytic()
        _sando_add_traj(self._h, new_traj._handle, float(current_time))

    def replan(self, last_rt: float, current_time: float):
        bits = _sando_replan(self._h, float(last_rt), float(current_time))
        return (bool(bits & 1), bool(bits & 2))

    def get_next_goal(self):
        out = (C.c_double * 9)()
        ok = _sando_get_next_goal(self._h, C.cast(out, _dbl))
        if not ok:
            return False, RobotState()
        g = _Goal(np.array([out[0], out[1], out[2]]),
                  np.array([out[3], out[4], out[5]]),
                  np.array([out[6], out[7], out[8]]))
        return True, g

    def get_drone_status(self):
        return int(_sando_get_drone_status(self._h))

    def get_global_path(self):
        n = _sando_get_global_path(self._h, C.cast(self._gp_buf, _dbl), 4096)
        return [np.array([self._gp_buf[3 * i], self._gp_buf[3 * i + 1], self._gp_buf[3 * i + 2]])
                for i in range(n)]

    def get_corridor(self):
        """Last committed space-time corridor: list of {lo,hi,t_l,t_u,seg} (empty when off)."""
        n = _sando_get_corridor(self._h, C.cast(self._corr_buf, _dbl), 64)
        b = self._corr_buf
        return [{"lo": (b[9 * i], b[9 * i + 1], b[9 * i + 2]),
                 "hi": (b[9 * i + 3], b[9 * i + 4], b[9 * i + 5]),
                 "t_l": b[9 * i + 6], "t_u": b[9 * i + 7], "seg": int(b[9 * i + 8])}
                for i in range(n)]

    def __del__(self):
        try:
            _sando_destroy(self._h)
        except Exception:
            pass


DroneStatus_GOAL_REACHED = 3  # matches DroneStatus.GOAL_REACHED
