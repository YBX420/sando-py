// planner.hpp — faithful C++ port of the SANDO orchestrator class in
// sando_py/planner.py, MINCO PATH ONLY (the Gurobi / B-spline baseline path is
// out of scope and NOT ported — see the honest "NOT ported" list at the bottom).
//
// 不许欺骗、必须还原: every ported method mirrors planner.py 1:1; no re-derivation
// of any reused header. This is the capstone integration that wires together the
// already-golden-verified stack:
//   minjerk_traj.hpp      (MinjerkTraj)
//   obstacles.hpp         (SphereObstacle / AABBObstacle)
//   avoid_config.hpp      (default_config / AvoidParams)
//   plan_minco.hpp        (plan_minco driver + PlanOptParams + DetourConfig)
//   hgp_manager.hpp       (HGPManager.solve_hgp / update_map / setup_planner)
//   types.hpp             (Parameters / RobotState / DynTraj / PieceWisePol / DroneStatus)
//
// Ported (MINCO path of class SANDO):
//   - ctor: hgp_manager, factors sweep list, DroneStatus, flags
//     (state_initialized / terminal_goal_initialized / map_initialized via
//     hgp_manager), state / G / A / G_term, plan deque, obst snapshot vectors.
//   - update_state, set_terminal_goal, set_A/set_A_time/set_G, get_* accessors,
//     change_drone_status.
//   - need_replan, check_ready_to_replan, DroneStatus state machine.
//   - find_A_and_Atime (k_value logic), compute_G (project_point_to_sphere),
//     find_safe_sub_goal (kdtree-unknown truncation via a simple linear nearest),
//     generate_global_path (-> hgp_manager.solve_hgp), num_P trimming, dir_hint.
//   - _compute_obst_pos_and_traj_max_time (id-range per-class tag 200<=id<300 ->
//     wall, else human; obst_vel/accel from traj), _obstacles_from_snapshot
//     (-> SphereObstacle/AABBObstacle per class), add_traj, update_occupancy_map
//     (rviz_only path).
//   - plan_local_trajectory_minco (-> plan_minco), append_to_plan (splice),
//     get_next_goal (pop deque front), retrieve_goal_setpoints,
//     _minjerk_to_pwp (MinjerkTraj -> QuinticPieceWisePol), get_pwp.
//   - replan (top-level cycle: check_ready -> need_replan ->
//     generate_global_path -> plan_local_trajectory_minco -> append_to_plan).
//
// NOT ported (documented honestly, end of file):
//   - The Gurobi / SolverGurobi factor-sweep local-solver path
//     (plan_local_trajectory non-minco, _generate_local_trajectory,
//      compute_worst_seg_end_times_poly) and SFC cvx_ellipsoid_decomp*.
//   - ROS-node glue, drawables/telemetry (retrieve_data dict, poly_out_*),
//     hover avoidance (check_hover_avoidance / _find_safe_evasion), yaw machine
//     get_desired_yaw / _yaw (mostly wall-clock + Gurobi-coupled), hardware
//     init-pose transforms (set_initial_pose / apply_init_pose_*).
//   - worst_traj_time bootstrap (SolverGurobi.get_initial_dt()*num_N). It only
//     feeds _compute_obst_pos_and_traj_max_time's Th; we expose it as a settable
//     field so the golden can drive that method deterministically.
//   - scipy.cKDTree: find_safe_sub_goal uses a brute-force linear nearest over
//     the unknown cloud (same geometry, same thresholds, same backtrack).
#pragma once

#include "sando_cpp/types.hpp"
#include "sando_cpp/hgp_manager.hpp"
#include "sando_cpp/obstacles.hpp"
#include "sando_cpp/avoid_config.hpp"
#include "sando_cpp/plan_minco.hpp"
#include "sando_cpp/minjerk_traj.hpp"

#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <string>
#include <map>
#include <memory>
#include <cmath>
#include <algorithm>
#include <limits>

namespace sando {

// ===========================================================================
// project_point_to_sphere — 1:1 with utils.py::project_point_to_sphere.
// Project P2 onto the sphere of `radius` centred at P1 (unchanged if inside).
// ===========================================================================
inline Eigen::Vector3d project_point_to_sphere(const Eigen::Vector3d& P1,
                                               const Eigen::Vector3d& P2,
                                               double radius) {
  Eigen::Vector3d diff = P2 - P1;
  double n = diff.norm();
  if (n <= radius) return P2;
  return P1 + diff * (radius / n);
}

// clamp — 1:1 with utils.py::clamp.
inline double clamp_scalar(double x, double lo, double hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

// ===========================================================================
// QuinticPieceWisePol — degree-5 ASCENDING piecewise polynomial carrying the
// MINCO backbone. Subclass of PieceWisePol overriding ONLY the axis evaluator
// (the base is cubic/descending; MINCO is quintic/ascending). 1:1 with
// planner.py::QuinticPieceWisePol + _poly5_deriv_ascending.
//
// We store the 6-per-segment ascending coeffs in a parallel vector since the
// base coeff_x/y/z are Vector4d (cubic). eval/velocity/acceleration are
// overridden to read the quintic coeffs. The base `times` is reused as-is.
// ===========================================================================
inline double poly5_deriv_ascending(const Eigen::Matrix<double, 6, 1>& c,
                                    double u, int order) {
  double s = 0.0;
  for (int j = order; j < 6; ++j) {
    double coeff = 1.0;
    for (int k = 0; k < order; ++k) coeff *= static_cast<double>(j - k);
    s += coeff * c(j) * std::pow(u, j - order);
  }
  return s;
}

class QuinticPieceWisePol : public PieceWisePol {
 public:
  // ascending degree-5 coeffs per segment, parallel to `times` (size M; M+1 times)
  std::vector<Eigen::Matrix<double, 6, 1>> qcoeff_x;
  std::vector<Eigen::Matrix<double, 6, 1>> qcoeff_y;
  std::vector<Eigen::Matrix<double, 6, 1>> qcoeff_z;

  void clear_q() {
    times.clear();
    qcoeff_x.clear();
    qcoeff_y.clear();
    qcoeff_z.clear();
  }

  // order: 0 pos, 1 vel, 2 accel, 3 jerk. Boundary handling mirrors Python.
  double eval_q_axis(const std::vector<Eigen::Matrix<double, 6, 1>>& coeffs,
                     double t, int order) const {
    if (times.empty() || coeffs.empty()) return 0.0;
    const std::size_t n = times.size();
    if (t >= times[n - 1]) {
      double u = times[n - 1] - times[n - 2];
      return poly5_deriv_ascending(coeffs.back(), u, order);
    }
    if (t < times[0]) return poly5_deriv_ascending(coeffs[0], 0.0, order);
    for (std::size_t i = 0; i + 1 < n; ++i) {
      if (times[i] <= t && t < times[i + 1]) {
        double u = t - times[i];
        return poly5_deriv_ascending(coeffs[i], u, order);
      }
    }
    return 0.0;
  }

  Eigen::Vector3d eval_q(double t) const {
    return Eigen::Vector3d(eval_q_axis(qcoeff_x, t, 0), eval_q_axis(qcoeff_y, t, 0),
                           eval_q_axis(qcoeff_z, t, 0));
  }
  Eigen::Vector3d velocity_q(double t) const {
    return Eigen::Vector3d(eval_q_axis(qcoeff_x, t, 1), eval_q_axis(qcoeff_y, t, 1),
                           eval_q_axis(qcoeff_z, t, 1));
  }
  Eigen::Vector3d acceleration_q(double t) const {
    return Eigen::Vector3d(eval_q_axis(qcoeff_x, t, 2), eval_q_axis(qcoeff_y, t, 2),
                           eval_q_axis(qcoeff_z, t, 2));
  }
};

// ---------------------------------------------------------------------------
// _minjerk_to_pwp — MinjerkTraj -> eval-equivalent QuinticPieceWisePol.
//   times[i] = A_time + mj.cum[i]; per-segment ascending coeffs copied verbatim
//   (mj.c is (6M,3) ascending). 1:1 with planner.py::_minjerk_to_pwp.
// ---------------------------------------------------------------------------
inline QuinticPieceWisePol minjerk_to_pwp(const MinjerkTraj& mj, double A_time) {
  QuinticPieceWisePol pwp;
  pwp.times.resize(mj.M + 1);
  for (int i = 0; i <= mj.M; ++i) pwp.times[i] = A_time + mj.cum(i);
  for (int i = 0; i < mj.M; ++i) {
    Eigen::Matrix<double, 6, 1> cx, cy, cz;
    for (int j = 0; j < 6; ++j) {
      cx(j) = mj.c(6 * i + j, 0);
      cy(j) = mj.c(6 * i + j, 1);
      cz(j) = mj.c(6 * i + j, 2);
    }
    pwp.qcoeff_x.push_back(cx);
    pwp.qcoeff_y.push_back(cy);
    pwp.qcoeff_z.push_back(cz);
  }
  return pwp;
}

// ===========================================================================
// SANDO — the orchestrator (MINCO path).
// ===========================================================================
class SANDO {
 public:
  Parameters par;
  HGPManager hgp_manager;

  // factor list (time-allocation sweep). Kept for parity; the MINCO path only
  // reads factors_.back() for the obstacle-horizon Th in _compute_obst_*.
  std::vector<double> factors_;
  int num_dynamic_factors = 0;

  // worst trajectory time (Th = worst_traj_time * factors_.back()). Bootstrap is
  // Gurobi-based (out of scope) -> settable field; default 0 (Th=0 -> no preds).
  double worst_traj_time = 0.0;

  // limits
  Eigen::Vector3d v_max_3d, a_max_3d, j_max_3d;
  double v_max = 0.0;
  double max_dist_vertexes = 1.0;

  // drone status
  DroneStatus drone_status_ = DroneStatus::GOAL_REACHED;

  // map size (recomputed each replan in compute_map_size)
  double wdx = 0, wdy = 0, wdz = 0, map_res = 0;
  Eigen::Vector3d map_center = Eigen::Vector3d::Zero();
  bool map_size_initialized = false;
  bool map_seen = false;
  int hgp_failure_count = 0;

  // flags
  bool state_initialized = false;
  bool terminal_goal_initialized = false;
  bool use_adapt_k_value = false;
  bool kdtree_map_initialized = false;
  bool kdtree_unk_initialized = false;

  // diagnostics actually used on the MINCO path
  double final_g = 0.0;
  double global_planning_time = 0.0;
  double cvx_decomp_time = 0.0;
  double successful_factor = 0.0;
  double local_traj_computation_time = 0.0;
  std::vector<RobotState> goal_setpoints;
  std::vector<Eigen::Matrix<double, 6, 3>> cps;  // Bernstein control pts (M,6,3)

  // replanning state machine
  RobotState state, G, A, E, G_term;
  double A_time = 0.0;
  std::deque<RobotState> plan;
  double previous_yaw = 0.0;

  // shared pwp
  QuinticPieceWisePol pwp_to_share;

  // obstacle snapshot (parallel lists)
  std::vector<Eigen::Vector3d> obst_pos;
  std::vector<Eigen::Vector3d> obst_bbox;
  std::vector<std::string> obst_class;
  std::vector<Eigen::Vector3d> obst_vel;
  std::vector<Eigen::Vector3d> obst_accel;
  double traj_max_time = 0.0;

  // hover
  Eigen::Vector3d p_hover = Eigen::Vector3d::Zero();

  // failure count
  int replanning_failure_count = 0;

  // adaptive k_value
  int num_replanning = 0;
  bool got_enough_replanning = false;
  int k_value = 0;
  std::vector<double> store_computation_times;
  double est_comp_time = 0.0;

  // dynamic obstacles
  std::vector<DynTraj> trajs;

  // point clouds (occupied / unknown) — vectors of world points
  std::vector<Eigen::Vector3d> pclptr_map;
  std::vector<Eigen::Vector3d> pclptr_unk;
  bool has_map_cloud = false;
  bool has_unk_cloud = false;

  // global path caches
  std::vector<Eigen::Vector3d> global_path_;
  std::vector<Eigen::Vector3d> original_global_path_;

  // last minco traj (debug)
  std::shared_ptr<MinjerkTraj> last_minco_traj;

  // -------------------------------------------------------------------------
  explicit SANDO(const Parameters& par_) : par(par_) {
    hgp_manager.set_parameters(par);

    // factor list
    if (par.use_dynamic_factor) {
      num_dynamic_factors =
          static_cast<int>((2.0 * par.dynamic_factor_k_radius) /
                           par.factor_constant_step_size) + 1;
      for (int i = 0; i < num_dynamic_factors; ++i) {
        double f = par.dynamic_factor_initial_mean - par.dynamic_factor_k_radius +
                   i * par.factor_constant_step_size;
        if (par.factor_initial <= f && f <= par.factor_final) factors_.push_back(f);
      }
    } else {
      num_dynamic_factors =
          static_cast<int>((par.factor_final - par.factor_initial) /
                           par.factor_constant_step_size) + 1;
      for (int i = 0; i < num_dynamic_factors; ++i)
        factors_.push_back(par.factor_initial + i * par.factor_constant_step_size);
    }

    v_max_3d = Eigen::Vector3d(par.v_max, par.v_max, par.v_max);
    a_max_3d = Eigen::Vector3d(par.a_max, par.a_max, par.a_max);
    j_max_3d = Eigen::Vector3d(par.j_max, par.j_max, par.j_max);
    v_max = par.v_max;
    max_dist_vertexes = par.max_dist_vertexes;

    drone_status_ = DroneStatus::GOAL_REACHED;
    wdx = par.initial_wdx; wdy = par.initial_wdy; wdz = par.initial_wdz;
    map_res = par.res;
  }

  // ------------------------------------------------------------------
  // Status / lifecycle
  // ------------------------------------------------------------------
  void change_drone_status(DroneStatus new_status) {
    if (new_status == drone_status_) return;
    drone_status_ = new_status;
  }
  int get_drone_status() const { return static_cast<int>(drone_status_); }
  Eigen::Vector3d get_hover_pos() const { return p_hover; }

  // ------------------------------------------------------------------
  // Getters / setters (no locking — single-threaded port)
  // ------------------------------------------------------------------
  RobotState get_state() const { return state; }
  RobotState get_gterm() const { return G_term; }
  void set_gterm(const RobotState& g) { G_term = g; }
  RobotState get_G() const { return G; }
  void set_G(const RobotState& g) { G = g; }
  RobotState get_E() const { return E; }
  RobotState get_A() const { return A; }
  void set_A(const RobotState& a) { A = a; }
  double get_A_time() const { return A_time; }
  void set_A_time(double t) { A_time = t; }

  RobotState get_last_plan_state() const {
    if (!plan.empty()) return plan.back();
    return state;
  }
  std::vector<DynTraj> get_trajs() const { return trajs; }
  std::vector<Eigen::Vector3d> get_global_path() const { return global_path_; }
  const QuinticPieceWisePol& get_pwp() const { return pwp_to_share; }

  // ------------------------------------------------------------------
  // Trajectory bookkeeping
  // ------------------------------------------------------------------
  void clean_up_old_trajs(double current_time) {
    std::vector<DynTraj> kept;
    for (auto& t : trajs)
      if ((current_time - t.time_received) <= par.traj_lifetime) kept.push_back(t);
    trajs = std::move(kept);
  }

  bool check_point_within_map(const Eigen::Vector3d& point) const {
    if (!map_size_initialized && !map_seen) return true;
    return (std::abs(point(0) - map_center(0)) <= wdx / 2.0 &&
            std::abs(point(1) - map_center(1)) <= wdy / 2.0 &&
            std::abs(point(2) - map_center(2)) <= wdz / 2.0);
  }

  // add_traj — update-by-id first (no filter), else map + horizon filter.
  void add_traj(const DynTraj& new_traj, double current_time) {
    for (auto& t : trajs)
      if (t.id == new_traj.id) { t = new_traj; return; }
    DynTraj nt = new_traj;
    Eigen::Vector3d p = nt.eval(current_time);
    if (!check_point_within_map(p)) return;
    if ((p - state.pos).norm() > par.horizon) return;
    trajs.push_back(nt);
  }

  // ------------------------------------------------------------------
  // State update (odometry callback)
  // ------------------------------------------------------------------
  void update_state(const RobotState& data) {
    state = data;
    if (!state_initialized || drone_status_ == DroneStatus::YAWING) {
      RobotState tmp;
      // YAWING branch uses yaw_start_pos (hardware/yaw machine, not ported) ->
      // mirror the common (non-YAWING) path: tmp.pos = data.pos.
      tmp.pos = data.pos;
      tmp.yaw = data.yaw;
      if (!state_initialized) previous_yaw = data.yaw;
      plan.clear();
      plan.push_back(tmp);
      A = tmp;
      G = tmp;
      state_initialized = true;
    }
  }

  // ------------------------------------------------------------------
  // Map size / map updates
  // ------------------------------------------------------------------
  void compute_map_size(const Eigen::Vector3d& min_pos, const Eigen::Vector3d& max_pos) {
    double dynamic_buffer = par.map_buffer;
    double dist_x = std::abs(min_pos(0) - max_pos(0));
    double dist_y = std::abs(min_pos(1) - max_pos(1));
    double dist_z = std::abs(min_pos(2) - max_pos(2));
    wdx = std::max(dist_x + 2 * dynamic_buffer, par.min_wdx);
    wdy = std::max(dist_y + 2 * dynamic_buffer, par.min_wdy);
    wdz = std::max(dist_z + 2 * dynamic_buffer, par.min_wdz);
    map_center = (min_pos + max_pos) / 2.0;
  }

  // update_map_ptr — store occupied + unknown clouds; bootstrap map if needed.
  void update_map_ptr(const std::vector<Eigen::Vector3d>& cloud_map,
                      const std::vector<Eigen::Vector3d>& cloud_unk) {
    pclptr_map = cloud_map; has_map_cloud = true;
    pclptr_unk = cloud_unk; has_unk_cloud = true;
    if (!cloud_map.empty()) kdtree_map_initialized = true;
    if (!hgp_manager.is_map_initialized()) update_map(0.0);
  }

  void update_occupancy_map_ptr(const std::vector<Eigen::Vector3d>& cloud_map) {
    pclptr_map = cloud_map;
    has_map_cloud = true;
    if (!hgp_manager.is_map_initialized()) update_occupancy_map(0.0);
  }

  // update_map — gazebo/hardware path (occupied + unknown clouds).
  void update_map(double current_time) {
    RobotState local_state = get_state();
    RobotState local_G = get_G();
    compute_map_size(local_state.pos, local_G.pos);

    std::vector<Eigen::Vector3d> opos, obbox;
    std::vector<std::vector<Eigen::Vector3d>> pred_samples;
    std::vector<double> pred_times;
    traj_max_time = compute_obst_pos_and_traj_max_time(opos, obbox, pred_samples,
                                                       pred_times, current_time);

    std::vector<Eigen::Vector3d> pcl_map = has_map_cloud ? pclptr_map
                                                         : std::vector<Eigen::Vector3d>();
    std::vector<Eigen::Vector3d> pcl_unk = has_unk_cloud ? pclptr_unk
                                                         : std::vector<Eigen::Vector3d>();
    hgp_manager.update_map(wdx, wdy, wdz, map_center, pcl_map, pcl_unk, opos, obbox,
                           traj_max_time);
    map_size_initialized = true;
    map_seen = true;
    if (!pcl_map.empty()) kdtree_map_initialized = true;
    if (!pcl_unk.empty()) kdtree_unk_initialized = true;
  }

  // update_occupancy_map — rviz_only / fake_sim path (occupied-only, empty unk).
  void update_occupancy_map(double current_time) {
    RobotState local_state = get_state();
    RobotState local_G = get_G();
    compute_map_size(local_state.pos, local_G.pos);

    std::vector<Eigen::Vector3d> opos, obbox;
    std::vector<std::vector<Eigen::Vector3d>> pred_samples;
    std::vector<double> pred_times;
    traj_max_time = compute_obst_pos_and_traj_max_time(opos, obbox, pred_samples,
                                                       pred_times, current_time);

    std::vector<Eigen::Vector3d> pcl_map = has_map_cloud ? pclptr_map
                                                         : std::vector<Eigen::Vector3d>();
    hgp_manager.update_map(wdx, wdy, wdz, map_center, pcl_map,
                           std::vector<Eigen::Vector3d>(), opos, obbox, traj_max_time);
    map_size_initialized = true;
    map_seen = true;
    if (!pcl_map.empty()) kdtree_map_initialized = true;
  }

  // _compute_obst_pos_and_traj_max_time — per-class snapshot + predicted samples.
  double compute_obst_pos_and_traj_max_time(
      std::vector<Eigen::Vector3d>& opos, std::vector<Eigen::Vector3d>& obbox,
      std::vector<std::vector<Eigen::Vector3d>>& pred_samples,
      std::vector<double>& pred_times, double current_time) {
    opos.clear(); obbox.clear(); pred_samples.clear(); pred_times.clear();
    std::vector<DynTraj> local_trajs = get_trajs();

    std::vector<std::string> oclass;
    std::vector<Eigen::Vector3d> ovel, oaccel;
    std::vector<DynTraj> selected;
    for (auto& traj : local_trajs) {
      DynTraj t = traj;
      Eigen::Vector3d p = t.eval(current_time);
      if (!check_point_within_map(p)) continue;
      double dist = (p - state.pos).norm();
      if (dist > par.horizon) continue;
      opos.push_back(p);
      obbox.push_back(t.bbox);
      int tid = t.id;
      oclass.push_back((200 <= tid && tid < 300) ? "wall" : "human");
      ovel.push_back(t.velocity(current_time));
      oaccel.push_back(t.accel(current_time));
      selected.push_back(t);
    }

    obst_pos = opos;
    obst_bbox = obbox;
    obst_class = oclass;
    obst_vel = ovel;
    obst_accel = oaccel;

    double Th = worst_traj_time * (factors_.empty() ? 1.0 : factors_.back());
    if (Th <= 0.0 || selected.empty()) return Th;

    double dt = 0.5;
    int M = static_cast<int>(std::ceil(Th / dt)) + 1;
    M = std::max(5, std::min(M, 10));
    for (int j = 0; j < M; ++j) {
      double a = (M == 1) ? 0.0 : double(j) / double(M - 1);
      pred_times.push_back(a * Th);
    }
    for (size_t k_idx = 0; k_idx < selected.size(); ++k_idx) {
      DynTraj t = selected[k_idx];
      std::vector<Eigen::Vector3d> samples_k;
      for (int j = 0; j < M; ++j) {
        double t_abs = current_time + pred_times[j];
        Eigen::Vector3d pk = t.eval(t_abs);
        if (!pk.allFinite()) pk = t.eval(current_time);
        samples_k.push_back(pk);
      }
      pred_samples.push_back(samples_k);
    }
    return Th;
  }

  // ------------------------------------------------------------------
  // G / horizon projection
  // ------------------------------------------------------------------
  void compute_G(const RobotState& A_in, const RobotState& G_term_in, double horizon) {
    RobotState local_G;
    local_G.pos = project_point_to_sphere(A_in.pos, G_term_in.pos, horizon);
    Eigen::Vector3d d = G_term_in.pos - local_G.pos;
    double n = d.norm();
    if (n > 1e-9) {
      d /= n;
      local_G.yaw = std::atan2(d(1), d(0));
    }
    set_G(local_G);
  }

  // ------------------------------------------------------------------
  // Replan gating
  // ------------------------------------------------------------------
  bool need_replan(const RobotState& local_state, const RobotState& local_G_term,
                   const RobotState& last_plan_state) {
    double dist_to_term_G = (local_state.pos - local_G_term.pos).norm();
    double dist_from_last_plan_state_to_term_G =
        (last_plan_state.pos - local_G_term.pos).norm();
    double vel_magnitude = local_state.vel.norm();
    double max_goal_velocity = 0.1;

    if (par.hover_avoidance_enabled &&
        (drone_status_ == DroneStatus::GOAL_REACHED ||
         drone_status_ == DroneStatus::HOVER_AVOIDING))
      return true;

    if (dist_to_term_G < par.goal_radius && vel_magnitude < max_goal_velocity) {
      if (par.hover_avoidance_enabled) {
        p_hover = local_G_term.pos;
        change_drone_status(DroneStatus::HOVER_AVOIDING);
        return true;
      }
      change_drone_status(DroneStatus::GOAL_REACHED);
      p_hover = local_G_term.pos;
      return false;
    }

    if (drone_status_ == DroneStatus::GOAL_REACHED ||
        drone_status_ == DroneStatus::YAWING)
      return false;

    if (dist_to_term_G < par.goal_seen_radius)
      change_drone_status(DroneStatus::GOAL_SEEN);

    if (drone_status_ == DroneStatus::GOAL_SEEN &&
        dist_from_last_plan_state_to_term_G < par.goal_radius)
      return false;

    return true;
  }

  bool check_ready_to_replan() const {
    bool map_init = hgp_manager.is_map_initialized();
    bool kdtree_ok = (!par.use_hardware) || kdtree_map_initialized;
    return state_initialized && terminal_goal_initialized && map_init && kdtree_ok;
  }

  bool goal_reached_check() const {
    return check_ready_to_replan() &&
           (drone_status_ == DroneStatus::GOAL_REACHED ||
            drone_status_ == DroneStatus::HOVER_AVOIDING);
  }

  // ------------------------------------------------------------------
  // find_A_and_Atime — pick the start state of the next replan from the plan.
  // ------------------------------------------------------------------
  struct AAndAtime { bool ok; RobotState A; double A_time; };

  AAndAtime find_A_and_Atime(double current_time,
                             double last_replanning_computation_time) {
    int plan_size = static_cast<int>(plan.size());
    if (plan_size == 0) return {false, RobotState(), 0.0};

    RobotState A_out;
    double A_time_out;
    if (par.use_state_update) {
      if (!use_adapt_k_value) {
        k_value = std::max(plan_size - par.default_k_value, 0);
        if (num_replanning != 1)
          store_computation_times.push_back(last_replanning_computation_time);
      } else {
        double a = par.alpha_k_value_filtering;
        est_comp_time = a * last_replanning_computation_time + (1 - a) * est_comp_time;
        k_value = std::max(
            plan_size -
                static_cast<int>(par.k_value_factor * est_comp_time / par.dc),
            0);
      }

      if (plan_size - 1 - k_value < 0 || plan_size - 1 - k_value >= plan_size)
        k_value = plan_size - 1;

      A_out = plan[plan_size - 1 - k_value];
      A_time_out = current_time + (plan_size - 1 - k_value) * par.dc;
    } else {
      A_out = get_state();
      A_time_out = current_time;
    }

    if (A_out.pos(2) < par.z_min || A_out.pos(2) > par.z_max ||
        A_out.pos(0) < par.x_min || A_out.pos(0) > par.x_max ||
        A_out.pos(1) < par.y_min || A_out.pos(1) > par.y_max) {
      return {false, A_out, A_time_out};
    }
    return {true, A_out, A_time_out};
  }

  // ------------------------------------------------------------------
  // find_safe_sub_goal — truncate global path at the first unknown intersection.
  //   kdtree replaced by brute-force linear nearest over the unknown cloud.
  // ------------------------------------------------------------------
  // nearest distance from pt to the unknown cloud (inf if empty).
  double nearest_unk_dist(const Eigen::Vector3d& pt) const {
    double best = std::numeric_limits<double>::infinity();
    for (const auto& q : pclptr_unk) {
      double d = (pt - q).norm();
      if (d < best) best = d;
    }
    return best;
  }

  std::vector<Eigen::Vector3d> find_safe_sub_goal(
      const std::vector<Eigen::Vector3d>& global_path) const {
    std::vector<Eigen::Vector3d> original = global_path;
    if (original.empty()) return {};
    std::vector<Eigen::Vector3d> out;
    out.push_back(original[0]);

    const double sample_dist = 0.1;
    double r_inflate = par.obst_max_vel * traj_max_time;
    double thr_orig = par.drone_radius;
    double thr_infl = par.drone_radius + r_inflate;
    double thr_orig2 = thr_orig * thr_orig;
    double thr_infl2 = thr_infl * thr_infl;
    bool unk_empty = pclptr_unk.empty();

    // is_within_unknown: d2 = nearest dist; Python compares (d2*d2) < thr2.
    auto is_within_unknown = [&](const Eigen::Vector3d& pt, double thr2) -> bool {
      if (unk_empty) return false;
      double d2 = nearest_unk_dist(pt);
      return (d2 * d2) < thr2;
    };

    auto backtrack = [&](int seg_i, double s_hit) -> Eigen::Vector3d {
      int M = static_cast<int>(original.size());
      int i = std::max(0, std::min(seg_i, M - 2));
      Eigen::Vector3d Aa = original[i];
      Eigen::Vector3d Bb = original[i + 1];
      Eigen::Vector3d d_vec = Bb - Aa;
      double L = d_vec.norm();
      if (L < 1e-9) return Aa;
      Eigen::Vector3d dir_vec = d_vec / L;
      double s = std::max(0.0, std::min(s_hit, L));
      Eigen::Vector3d pt = Aa + dir_vec * s;
      if (!is_within_unknown(pt, thr_infl2)) return pt;
      for (;;) {
        s -= sample_dist;
        if (s >= 0.0) {
          pt = Aa + dir_vec * s;
        } else {
          i -= 1;
          if (i < 0) return original[0];
          Aa = original[i];
          Bb = original[i + 1];
          d_vec = Bb - Aa;
          L = d_vec.norm();
          if (L < 1e-9) { s = 0.0; pt = Aa; continue; }
          dir_vec = d_vec / L;
          s = L + s;
          s = std::max(0.0, std::min(s, L));
          pt = Aa + dir_vec * s;
        }
        if (!is_within_unknown(pt, thr_infl2)) return pt;
      }
    };

    int M = static_cast<int>(original.size());
    for (int i = 0; i + 1 < M; ++i) {
      Eigen::Vector3d cur = original[i];
      Eigen::Vector3d nxt = original[i + 1];
      Eigen::Vector3d d_vec = nxt - cur;
      double dist = d_vec.norm();
      if (dist < 1e-9) continue;
      Eigen::Vector3d dir_vec = d_vec / dist;
      int num_samples = static_cast<int>(dist / sample_dist);
      for (int j = 0; j <= num_samples; ++j) {
        Eigen::Vector3d sp = cur + dir_vec * (sample_dist * j);
        if (is_within_unknown(sp, thr_orig2)) {
          double s_hit = sample_dist * j;
          Eigen::Vector3d safe_pt = backtrack(i, s_hit);
          if ((safe_pt - out.back()).norm() > 1e-6) out.push_back(safe_pt);
          return out;
        }
      }
      out.push_back(nxt);
    }
    return out;
  }

  // ------------------------------------------------------------------
  // Replan pipeline
  // ------------------------------------------------------------------
  void reset_data() {
    final_g = 0.0;
    global_planning_time = 0.0;
    cvx_decomp_time = 0.0;
    local_traj_computation_time = 0.0;
    goal_setpoints.clear();
    pwp_to_share.clear_q();
    cps.clear();
  }

  // replan — top-level cycle. Returns (planned_ok, attempted).
  std::pair<bool, bool> replan(double last_replanning_computation_time,
                               double current_time) {
    reset_data();

    if (!check_ready_to_replan()) return {false, false};

    RobotState local_state = get_state();
    RobotState local_G_term = get_gterm();
    RobotState last_plan_state = get_last_plan_state();

    if (!need_replan(local_state, local_G_term, last_plan_state)) return {false, false};

    // Hover avoidance branch (check_hover_avoidance) NOT ported — only reached
    // when hover_avoidance_enabled; the MINCO golden runs with it disabled.
    if (par.hover_avoidance_enabled &&
        (drone_status_ == DroneStatus::GOAL_REACHED ||
         drone_status_ == DroneStatus::HOVER_AVOIDING)) {
      return {false, false};
    }

    std::vector<Eigen::Vector3d> global_path;
    bool ok = generate_global_path(current_time, last_replanning_computation_time,
                                   global_path);
    if (!ok) return {false, false};

    if (!plan_local_trajectory_minco(global_path, last_replanning_computation_time))
      return {false, true};

    if (!append_to_plan()) return {false, true};

    replanning_failure_count = 0;
    return {true, true};
  }

  // generate_global_path — returns ok; fills `out_global_path`.
  bool generate_global_path(double current_time,
                            double last_replanning_computation_time,
                            std::vector<Eigen::Vector3d>& out_global_path) {
    RobotState local_G_term = get_gterm();

    AAndAtime aa = find_A_and_Atime(current_time, last_replanning_computation_time);
    if (!aa.ok) {
      replanning_failure_count += 1;
      out_global_path.clear();
      return false;
    }
    RobotState local_A = aa.A;
    double A_time_local = aa.A_time;

    set_A(local_A);
    set_A_time(A_time_local);
    compute_G(local_A, local_G_term, par.horizon);

    if (par.sim_env == "fake_sim" || par.sim_env == "rviz_only")
      update_occupancy_map(current_time);
    else
      update_map(current_time);

    if (!hgp_manager.planner) hgp_manager.setup_planner();

    if (par.use_free_start) hgp_manager.free_start(local_A.pos, par.free_start_factor);
    if (par.use_free_goal) hgp_manager.free_goal(get_G().pos, par.free_goal_factor);

    RobotState local_G = get_G();

    if (par.vehicle_type != "uav") {
      local_A.pos(2) = 1.0;
      local_G.pos(2) = 1.0;
    }

    // dir_hint from previous global path else A->G
    std::vector<Eigen::Vector3d> prev_global = get_global_path();
    Eigen::Vector3d dir_hint;
    if (prev_global.size() >= 2 &&
        (prev_global[1] - prev_global[0]).norm() > 1e-8) {
      dir_hint = prev_global[1] - prev_global[0];
      dir_hint /= dir_hint.norm();
    } else {
      dir_hint = local_G.pos - local_A.pos;
      double n = dir_hint.norm();
      if (n > 1e-8) dir_hint /= n;
      else dir_hint = Eigen::Vector3d(1.0, 0.0, 0.0);
    }
    if (par.vehicle_type != "uav") dir_hint(2) = 0.0;

    HGPManager::SolveResult sr =
        hgp_manager.solve_hgp(local_A.pos, dir_hint, local_G.pos, A_time_local);
    if (!sr.success) {
      hgp_failure_count += 1;
      replanning_failure_count += 1;
      out_global_path.clear();
      return false;
    }
    final_g = sr.final_g;

    std::vector<Eigen::Vector3d> gpath = sr.path;
    global_path_ = gpath;
    original_global_path_ = sr.raw_path;

    // trim to (num_P + 1)
    if (static_cast<int>(gpath.size()) > par.num_P + 1)
      gpath.resize(par.num_P + 1);

    // safe sub-goal truncation
    gpath = find_safe_sub_goal(gpath);
    out_global_path = gpath;
    return true;
  }

  // ------------------------------------------------------------------
  // _obstacles_from_snapshot — per-class avoidance obstacles for plan_minco.
  //   human -> SphereObstacle (HARD, carries vel + accel for space-time ALM);
  //   wall  -> AABBObstacle (SOFT EGO field). Missing tag -> human (fail-safe).
  // The returned obstacles own their storage via `owned`; raw pointers in `out`.
  // ------------------------------------------------------------------
  void obstacles_from_snapshot(const std::vector<Eigen::Vector3d>& opos,
                               const std::vector<Eigen::Vector3d>& obbox,
                               const std::vector<std::string>& oclass,
                               const std::vector<Eigen::Vector3d>& ovel,
                               const std::vector<Eigen::Vector3d>& oaccel,
                               std::vector<std::shared_ptr<Obstacle>>& owned,
                               std::vector<const Obstacle*>& out,
                               std::map<std::string, AvoidParams>& avoid_cfg) const {
    owned.clear();
    out.clear();
    size_t n = opos.size();
    for (size_t k = 0; k < n; ++k) {
      Eigen::Vector3d c = opos[k];
      Eigen::Vector3d sz = obbox[k];
      std::string cls = (k < oclass.size()) ? oclass[k] : std::string("human");
      if (cls == "wall") {
        auto o = std::make_shared<AABBObstacle>(c - 0.5 * sz, c + 0.5 * sz, "wall");
        owned.push_back(o);
        out.push_back(o.get());
      } else {
        Eigen::Vector3d vel = (k < ovel.size()) ? ovel[k] : Eigen::Vector3d::Zero();
        Eigen::Vector3d acc = (k < oaccel.size()) ? oaccel[k] : Eigen::Vector3d::Zero();
        double rad = 0.5 * sz.maxCoeff();
        auto o = std::make_shared<SphereObstacle>(c, rad, vel, "human", acc);
        owned.push_back(o);
        out.push_back(o.get());
      }
    }
    avoid_cfg = default_config();
  }

  // ------------------------------------------------------------------
  // plan_local_trajectory_minco — MINCO local solve (per-class avoidance).
  // ------------------------------------------------------------------
  bool plan_local_trajectory_minco(std::vector<Eigen::Vector3d> global_path,
                                   double /*last_replanning_computation_time*/) {
    RobotState local_A = get_A();
    RobotState local_G = get_G();
    double A_time_local = get_A_time();
    RobotState local_E;

    // subdivide 2-point paths to >=3 (parity with Gurobi path)
    while (global_path.size() == 2) {
      Eigen::Vector3d mid = 0.5 * (global_path[0] + global_path[1]);
      std::vector<Eigen::Vector3d> np = {global_path[0], mid, global_path[1]};
      global_path = np;
    }
    if (global_path.empty() || global_path.size() < 3) {
      replanning_failure_count += 1;
      return false;
    }

    if (drone_status_ == DroneStatus::GOAL_REACHED ||
        drone_status_ == DroneStatus::GOAL_SEEN ||
        drone_status_ == DroneStatus::HOVER_AVOIDING) {
      local_E = local_G;
    } else {
      local_E.pos = global_path.back();
    }

    // seed polyline (copy so the z-clamp does not mutate the caller's)
    std::vector<Eigen::Vector3d> seed_path = global_path;
    if (par.vehicle_type != "uav") {
      local_A.pos(2) = 1.0;
      local_E.pos(2) = 1.0;
      for (auto& p : seed_path) p(2) = 1.0;
    }

    // obstacle snapshot
    std::vector<Eigen::Vector3d> opos = obst_pos, obbox = obst_bbox, ovel = obst_vel,
                                 oaccel = obst_accel;
    std::vector<std::string> oclass = obst_class;

    std::vector<std::shared_ptr<Obstacle>> owned;
    std::vector<const Obstacle*> obstacles;
    std::map<std::string, AvoidParams> avoid_cfg;
    obstacles_from_snapshot(opos, obbox, oclass, ovel, oaccel, owned, obstacles,
                            avoid_cfg);

    Eigen::Vector3d v0 = local_A.vel;
    Eigen::Vector3d a0 = local_A.accel;

    // astar_path matrix (M,3)
    Eigen::MatrixXd astar_path(static_cast<int>(seed_path.size()), 3);
    for (int i = 0; i < static_cast<int>(seed_path.size()); ++i)
      astar_path.row(i) = seed_path[i].transpose();

    PlanOptParams opt;  // defaults mirror local_opt.py:OptParams
    std::shared_ptr<MinjerkTraj> mj_ptr;
    PlanInfo info;
    try {
      DetourConfig dc_off; dc_off.enabled = false;
      auto pr = plan_minco(astar_path, obstacles, avoid_cfg, opt, dc_off, v0, a0);
      mj_ptr = std::make_shared<MinjerkTraj>(pr.first);
      info = pr.second;
      if (!info.trajectory_valid) {
        DetourConfig dc_on; dc_on.enabled = true;
        auto pr2 = plan_minco(astar_path, obstacles, avoid_cfg, opt, dc_on, v0, a0);
        mj_ptr = std::make_shared<MinjerkTraj>(pr2.first);
        info = pr2.second;
      }
    } catch (const std::exception&) {
      replanning_failure_count += 1;
      return false;
    }
    if (!mj_ptr || !info.trajectory_valid) {
      replanning_failure_count += 1;
      return false;
    }
    MinjerkTraj& mj = *mj_ptr;

    // Fill goal_setpoints (load-bearing).
    double dc = par.dc;
    double t_end = mj.t_end;
    int n_samples = std::max(2, static_cast<int>(std::ceil(t_end / dc)));
    std::vector<RobotState> setpoints;
    for (int i = 0; i < n_samples; ++i) {
      double t_local = std::min((i + 1) * dc, t_end - 1e-6);
      RobotState s;
      s.t = A_time_local + t_local;
      s.pos = mj.eval_deriv(t_local, 0);
      s.vel = mj.eval_deriv(t_local, 1);
      s.accel = mj.eval_deriv(t_local, 2);
      s.jerk = mj.eval_deriv(t_local, 3);
      setpoints.push_back(s);
    }

    goal_setpoints = setpoints;
    pwp_to_share = minjerk_to_pwp(mj, A_time_local);
    cps = mj.control_points();
    successful_factor = 1.0;
    cvx_decomp_time = 0.0;
    last_minco_traj = mj_ptr;
    return true;
  }

  // dispatch: local_solver == "minco" -> MINCO path; otherwise out of scope.
  bool plan_local_trajectory(const std::vector<Eigen::Vector3d>& global_path,
                             double last_replanning_computation_time) {
    if (par.local_solver == "minco")
      return plan_local_trajectory_minco(global_path, last_replanning_computation_time);
    // Gurobi path NOT ported.
    return false;
  }

  // ------------------------------------------------------------------
  // append_to_plan — splice winner setpoints into the plan deque.
  // ------------------------------------------------------------------
  bool append_to_plan() {
    int plan_size = static_cast<int>(plan.size());
    if (plan_size < k_value) {
      k_value = std::max(1, plan_size - 1);
    } else {
      for (int i = 0; i < k_value; ++i)
        if (!plan.empty()) plan.pop_back();
      for (const auto& s : goal_setpoints) plan.push_back(s);
    }

    if (!got_enough_replanning) {
      if (static_cast<int>(store_computation_times.size()) <
          par.num_replanning_before_adapt) {
        num_replanning += 1;
      } else {
        start_adapt_k_value();
        got_enough_replanning = true;
      }
    }
    return true;
  }

  void start_adapt_k_value() {
    int n = std::max<int>(1, static_cast<int>(store_computation_times.size()));
    double s = 0.0;
    for (double v : store_computation_times) s += v;
    est_comp_time = s / n;
    use_adapt_k_value = true;
  }

  // ------------------------------------------------------------------
  // get_next_goal — pull the front of the plan; fill yaw (subset of yaw machine).
  //   The full get_desired_yaw / _yaw + hardware transform are NOT ported; we
  //   port the deque pop + the GOAL_REACHED / low-plan yaw-hold branches that the
  //   MINCO golden exercises. Returns (ok, next_goal).
  // ------------------------------------------------------------------
  struct NextGoal { bool ok; RobotState goal; };

  NextGoal get_next_goal() {
    if (drone_status_ == DroneStatus::YAWING) {
      if (!state_initialized || !terminal_goal_initialized)
        return {false, RobotState()};
    } else if (!check_ready_to_replan()) {
      return {false, RobotState()};
    }

    if (plan.empty()) return {false, RobotState()};
    std::vector<RobotState> local_plan(plan.begin(), plan.end());

    RobotState next_goal = local_plan[0];
    if (local_plan.size() > 1) {
      if (!plan.empty()) plan.pop_front();
    }

    if (drone_status_ != DroneStatus::GOAL_REACHED) {
      if (replanning_failure_count > par.yaw_spinning_threshold &&
          drone_status_ != DroneStatus::HOVER_AVOIDING) {
        next_goal.yaw = previous_yaw + par.yaw_spinning_dyaw * par.dc;
        next_goal.dyaw = par.yaw_spinning_dyaw;
        previous_yaw = next_goal.yaw;
      } else if (local_plan.size() < 5 && drone_status_ != DroneStatus::YAWING &&
                 drone_status_ != DroneStatus::HOVER_AVOIDING) {
        next_goal.yaw = previous_yaw;
        next_goal.dyaw = 0.0;
      } else {
        // get_desired_yaw (TRAVELING/GOAL_SEEN speed-direction branch).
        get_desired_yaw_traveling(next_goal);
      }
      next_goal.dyaw = clamp_scalar(next_goal.dyaw, -par.w_max, par.w_max);
    } else {
      next_goal.yaw = previous_yaw;
      next_goal.dyaw = 0.0;
    }
    return {true, next_goal};
  }

  // get_desired_yaw — the TRAVELING / GOAL_SEEN branch + _yaw smoothing. The
  // YAWING / HOVER_AVOIDING branches need wall-clock timing and are not ported.
  void get_desired_yaw_traveling(RobotState& next_goal) {
    DroneStatus ds = drone_status_;
    if (ds == DroneStatus::TRAVELING || ds == DroneStatus::GOAL_SEEN) {
      double speed_xy = std::hypot(next_goal.vel(0), next_goal.vel(1));
      if (speed_xy < 0.01) {
        next_goal.yaw = previous_yaw;
        next_goal.dyaw = 0.0;
        return;
      }
      double desired_yaw = std::atan2(next_goal.vel(1), next_goal.vel(0));
      double diff = angle_wrap(desired_yaw - previous_yaw);
      yaw_smooth(diff, next_goal);
    } else {
      next_goal.yaw = previous_yaw;
      next_goal.dyaw = 0.0;
    }
  }

  // angle_wrap — 1:1 with utils.py: (a + pi) % (2*pi) - pi, where % is Python's
  // FLOORED modulo (result has the sign of the divisor, here always >= 0).
  static double angle_wrap(double a) {
    const double pi = 3.141592653589793;
    const double two_pi = 2.0 * pi;
    double x = a + pi;
    double m = std::fmod(x, two_pi);
    if (m < 0.0) m += two_pi;  // floored modulo
    return m - pi;
  }

  void yaw_smooth(double diff, RobotState& next_goal) {
    double step = (1.0 - par.alpha_filter_dyaw) * diff;
    double max_step = par.w_max * par.dc;
    step = clamp_scalar(step, -max_step, max_step);
    next_goal.yaw = previous_yaw + step;
    next_goal.dyaw = step / par.dc;
    previous_yaw = next_goal.yaw;
  }

  // ------------------------------------------------------------------
  // set_terminal_goal — drives YAWING / smooth mid-flight goal updates.
  // ------------------------------------------------------------------
  void set_terminal_goal(const RobotState& term_goal) {
    if (terminal_goal_initialized) {
      RobotState cur = get_gterm();
      if ((cur.pos - term_goal.pos).norm() < 0.1) return;
    }
    RobotState local_state = get_state();

    // mid-flight smooth update
    if (terminal_goal_initialized &&
        (drone_status_ == DroneStatus::TRAVELING ||
         drone_status_ == DroneStatus::GOAL_SEEN)) {
      set_gterm(term_goal);
      p_hover = term_goal.pos;
      G.pos = project_point_to_sphere(local_state.pos, term_goal.pos, par.horizon);
      if (drone_status_ == DroneStatus::GOAL_SEEN)
        change_drone_status(DroneStatus::TRAVELING);
      return;
    }

    // full re-init
    RobotState tmp;
    tmp.pos = local_state.pos;
    tmp.vel = local_state.vel;
    tmp.accel = local_state.accel;
    tmp.yaw = local_state.yaw;
    plan.clear();
    plan.push_back(tmp);
    A = tmp;
    G = tmp;

    set_gterm(term_goal);
    p_hover = term_goal.pos;
    previous_yaw = local_state.yaw;
    replanning_failure_count = 0;

    G.pos = project_point_to_sphere(local_state.pos, term_goal.pos, par.horizon);

    if (par.skip_initial_yawing)
      change_drone_status(DroneStatus::TRAVELING);
    else
      change_drone_status(DroneStatus::YAWING);

    if (!terminal_goal_initialized) terminal_goal_initialized = true;
  }

  // ------------------------------------------------------------------
  // Retrieval helpers
  // ------------------------------------------------------------------
  std::vector<RobotState> retrieve_goal_setpoints() const { return goal_setpoints; }
  std::vector<Eigen::Matrix<double, 6, 3>> retrieve_cps() const { return cps; }
};

}  // namespace sando
