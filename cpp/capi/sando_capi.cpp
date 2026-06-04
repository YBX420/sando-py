// sando_capi.cpp — flat extern "C" ABI over the golden-verified C++ SANDO planner
// (planner.hpp + types.hpp), so Isaac Sim's Python can drive the C++ engine via ctypes —
// a drop-in replacement for sando_py with the SAME API surface (see sando_cpp_bridge.py).
//
// Built as a standalone DLL with statically-linked libstdc++/libgcc/winpthread so it loads
// in any CPython (Isaac's MSVC-built 3.10) with no MinGW runtime on PATH.
//
// No algorithm logic lives here — this only marshals primitives <-> the C++ objects.
#include "sando_cpp/types.hpp"
#include "sando_cpp/planner.hpp"

#include <vector>
#include <string>
#include <cstring>
#include <cstdio>

#if defined(_WIN32)
#define SANDO_API extern "C" __declspec(dllexport)
#else
#define SANDO_API extern "C" __attribute__((visibility("default")))
#endif

using sando::Parameters;
using sando::RobotState;
using sando::DynTraj;
using sando::SANDO;

// ===========================================================================
// Parameters — create, set-by-name (mirrors the Python `setattr(par, k, v)` loop),
// destroy. Unknown names print a warning and are ignored (matches the Python loop).
// ===========================================================================
SANDO_API void* params_create() { return new Parameters(); }
SANDO_API void params_destroy(void* p) { delete static_cast<Parameters*>(p); }

// numeric setter — covers EVERY double & int field of Parameters (int fields are cast).
// Complete coverage so any YAML key that maps to a real field is applied (no silent drops).
SANDO_API void params_set_double(void* ph, const char* name, double v) {
  Parameters& p = *static_cast<Parameters*>(ph);
  std::string k = name;
#define D(field) else if (k == #field) p.field = v
#define I(field) else if (k == #field) p.field = static_cast<int>(v)
  if (k == "global_planner_heuristic_weight") p.global_planner_heuristic_weight = v;
  D(factor_hgp); D(inflation_hgp); D(x_min); D(x_max); D(y_min); D(y_max); D(z_min); D(z_max);
  D(drone_radius); D(free_start_factor); D(free_goal_factor); D(min_len); D(min_turn);
  D(min_dist_from_agent_to_traj); D(shrinked_box_size); D(map_buffer); D(center_shift_factor);
  D(initial_wdx); D(initial_wdy); D(initial_wdz); D(min_wdx); D(min_wdy); D(min_wdz); D(res);
  D(comm_delay_inflation_alpha); D(comm_delay_inflation_max); D(comm_delay_filter_alpha);
  D(depth_camera_depth_max); D(fov_visual_depth); D(fov_visual_x_deg); D(fov_visual_y_deg);
  D(max_dist_vertexes); D(w_unknown); D(w_align); D(decay_len_cells); D(w_side); D(heat_weight);
  D(heat_alpha0); D(heat_alpha1); D(heat_tau_ratio); D(heat_gamma); D(heat_Hmax);
  D(dyn_base_inflation_m); D(dyn_heat_tube_radius_m); D(static_heat_alpha); D(static_heat_Hmax);
  D(static_heat_rmax_m); D(static_heat_default_radius_m); D(obstacle_soft_cost); D(horizon); D(dc);
  D(v_max); D(a_max); D(j_max); D(goal_radius); D(goal_seen_radius); D(dynamic_factor_k_radius);
  D(dynamic_factor_initial_mean); D(factor_initial); D(factor_final); D(factor_constant_step_size);
  D(obst_max_vel); D(obst_position_error); D(max_gurobi_comp_time_sec); D(jerk_smooth_weight);
  D(traj_lifetime); D(alpha_k_value_filtering); D(k_value_factor); D(alpha_filter_dyaw); D(w_max);
  D(minco_time_budget_ms); D(minco_w_time);
  D(minco_human_slow_vmax); D(minco_human_slow_near); D(minco_human_slow_far);
  D(minco_sfc_radius); D(minco_w_corridor);
  D(w_max_yawing); D(yaw_spinning_dyaw); D(default_goal_z); D(hover_avoidance_d_trigger);
  D(hover_avoidance_h); D(hover_avoidance_min_repulsion_norm);
  I(visual_level); I(hgp_timeout_duration_ms); I(max_num_expansion); I(los_cells); I(heat_p);
  I(heat_q); I(heat_num_samples); I(static_heat_p); I(num_P); I(num_N);
  I(num_replanning_before_adapt); I(default_k_value); I(yaw_spinning_threshold);
  else std::printf("[sando_capi][warn] params_set_double: unknown field '%s', skipped\n", name);
#undef D
#undef I
}

SANDO_API void params_set_bool(void* ph, const char* name, int v) {
  Parameters& p = *static_cast<Parameters*>(ph);
  std::string k = name; bool b = (v != 0);
#define B(field) else if (k == #field) p.field = b
  if (k == "use_global_pc") p.use_global_pc = b;
  B(provide_goal_in_global_frame); B(state_already_in_global_frame); B(use_hardware);
  B(global_planner_verbose); B(use_free_start); B(use_free_goal); B(use_state_update);
  B(use_random_color_for_global_path); B(use_path_push_for_visualization); B(use_shrinked_box);
  B(use_comm_delay_inflation); B(use_heat_map); B(dynamic_heat_enabled);
  B(dynamic_as_occupied_current); B(dynamic_as_occupied_future); B(use_only_curr_pos_for_dynamic_obst);
  B(static_heat_enabled); B(static_heat_boundary_only); B(static_heat_apply_on_unknown);
  B(static_heat_exclude_dynamic); B(use_soft_cost_obstacles); B(use_dynamic_factor);
  B(inflate_unknown_boundary); B(using_variable_elimination); B(skip_initial_yawing);
  B(minco_use_topology);
  B(force_goal_z); B(debug_verbose); B(ignore_other_trajs); B(hover_avoidance_enabled);
  B(hover_avoidance_2d);
  else std::printf("[sando_capi][warn] params_set_bool: unknown field '%s', skipped\n", name);
#undef B
}

SANDO_API void params_set_string(void* ph, const char* name, const char* v) {
  Parameters& p = *static_cast<Parameters*>(ph);
  std::string k = name, val = v;
#define S(field) else if (k == #field) p.field = val
  if (k == "sim_env") p.sim_env = val;
  S(vehicle_type); S(flight_mode); S(global_planner); S(local_solver);
  S(environment_assumption); S(dynamic_constraint_type);
  else std::printf("[sando_capi][warn] params_set_string: unknown field '%s', skipped\n", name);
#undef S
}

SANDO_API void params_set_vec3(void* ph, const char* name, double x, double y, double z) {
  Parameters& p = *static_cast<Parameters*>(ph);
  std::string k = name;
  if (k == "drone_bbox") p.drone_bbox = {x, y, z};
  else if (k == "sfc_size") p.sfc_size = {x, y, z};
  else std::printf("[sando_capi][warn] params_set_vec3: unknown field '%s', skipped\n", name);
}

// ===========================================================================
// DynTraj — analytic obstacle handle (id + bbox + 6 expression strings).
// ===========================================================================
SANDO_API void* traj_create(int id, double bx, double by, double bz,
                            const char* tx, const char* ty, const char* tz,
                            const char* vx, const char* vy, const char* vz) {
  DynTraj* d = new DynTraj();
  d->id = id;
  d->mode = "Analytic";
  d->bbox = Eigen::Vector3d(bx, by, bz);
  d->traj_x = tx; d->traj_y = ty; d->traj_z = tz;
  d->traj_vx = vx ? vx : ""; d->traj_vy = vy ? vy : ""; d->traj_vz = vz ? vz : "";
  d->compile_analytic();
  return d;
}
SANDO_API void traj_destroy(void* h) { delete static_cast<DynTraj*>(h); }
SANDO_API void traj_eval(void* h, double t, double* out3) {
  Eigen::Vector3d p = static_cast<DynTraj*>(h)->eval(t);
  out3[0] = p[0]; out3[1] = p[1]; out3[2] = p[2];
}

// ===========================================================================
// SANDO lifecycle + the Isaac-loop API surface.
// ===========================================================================
SANDO_API void* sando_create(void* params_h) {
  return new SANDO(*static_cast<Parameters*>(params_h));
}
SANDO_API void sando_destroy(void* h) { delete static_cast<SANDO*>(h); }

SANDO_API void sando_update_state(void* h, const double* pos3, const double* vel3,
                                  const double* acc3, double yaw) {
  RobotState st;
  st.pos = Eigen::Vector3d(pos3[0], pos3[1], pos3[2]);
  if (vel3) st.vel = Eigen::Vector3d(vel3[0], vel3[1], vel3[2]);
  if (acc3) st.accel = Eigen::Vector3d(acc3[0], acc3[1], acc3[2]);
  st.yaw = yaw;
  static_cast<SANDO*>(h)->update_state(st);
}

SANDO_API void sando_update_occupancy(void* h, const double* pts, int n) {
  std::vector<Eigen::Vector3d> cloud;
  cloud.reserve(n);
  for (int i = 0; i < n; ++i) cloud.emplace_back(pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]);
  static_cast<SANDO*>(h)->update_occupancy_map_ptr(cloud);
}

SANDO_API void sando_set_terminal_goal(void* h, const double* pos3) {
  RobotState G;
  G.pos = Eigen::Vector3d(pos3[0], pos3[1], pos3[2]);
  static_cast<SANDO*>(h)->set_terminal_goal(G);
}

SANDO_API void sando_add_traj(void* h, void* traj_h, double t) {
  static_cast<SANDO*>(h)->add_traj(*static_cast<DynTraj*>(traj_h), t);
}

// replan returns std::pair<bool,bool> (success, attempted); encode as bit0=success, bit1=attempted.
SANDO_API int sando_replan(void* h, double last_rt, double t) {
  auto r = static_cast<SANDO*>(h)->replan(last_rt, t);
  return (r.first ? 1 : 0) | (r.second ? 2 : 0);
}

// out9 = [pos(3), vel(3), accel(3)]; returns 1 if a goal is available
SANDO_API int sando_get_next_goal(void* h, double* out9) {
  auto ng = static_cast<SANDO*>(h)->get_next_goal();
  if (!ng.ok) return 0;
  for (int i = 0; i < 3; ++i) {
    out9[i] = ng.goal.pos[i];
    out9[3 + i] = ng.goal.vel[i];
    out9[6 + i] = ng.goal.accel[i];
  }
  return 1;
}

SANDO_API int sando_get_drone_status(void* h) {
  return static_cast<SANDO*>(h)->get_drone_status();
}

// fills out[3*N] with the global path points; returns N (clamped to max_pts).
SANDO_API int sando_get_global_path(void* h, double* out, int max_pts) {
  auto gp = static_cast<SANDO*>(h)->get_global_path();
  int n = static_cast<int>(gp.size());
  if (n > max_pts) n = max_pts;
  for (int i = 0; i < n; ++i) {
    out[3 * i] = gp[i][0]; out[3 * i + 1] = gp[i][1]; out[3 * i + 2] = gp[i][2];
  }
  return n;
}
