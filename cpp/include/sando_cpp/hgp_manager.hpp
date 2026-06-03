// HGPManager — faithful C++ port of sando_py/hgp/hgp_manager.py:HGPManager.
//
// Facade over the global-planning stack: owns a VoxelMapUtil (the planning grid)
// and an HGPPlanner (heat-A* + path post-processing), and exposes the high-level
// interface the upper layers call:
//   - set_parameters(par): push the yaml Parameters into map_util + planner knobs.
//                          NOTE: the HGP grid edge length is factor_hgp*res (the
//                          global grid is usually coarser than the local one), and
//                          drone_radius = max(drone_bbox)*0.5 (0.2 if bbox empty).
//   - setup_planner():     (re)build HGPPlanner from par and bind map_util.
//   - update_map(...):     rebuild the grid via map_util.read_map(...) each cycle;
//                          first build also constructs the planner.
//   - free_start / free_goal(p, factor): carve a free cube of half-size factor*res
//                          around p (set_free_voxel_and_surroundings).
//   - check_if_point_occupied / check_if_point_free(p): float_to_int -> query.
//   - solve_hgp(start, start_vel, goal, current_time):
//          optional free-start/free-goal, planner.plan(), get_path()/get_raw_path(),
//          densify the processed path by max_dist_vertexes (create_more_vertexes).
//          Returns (success, final_g, processed_path, raw_path).
//   - check_if_path_in_free(path): longest free prefix from the start.
//   - push_path_into_free_space(path): snap each point to nearest free cell center.
//   - create_more_vertexes(path, d): insert a vertex every d metres on long edges.
//   - get_occupied_cells / get_free_cells: world cell-centers (visualization).
//
// REUSES the already golden-verified headers (no re-implementation of grid math):
//   voxel_map.hpp (VoxelMapUtil read_map/float_to_int/is_free/is_occupied/set_free),
//   voxel_map_helpers.hpp (set_free_voxel_and_surroundings / find_closest_free_point
//                          / cells_world FREE functions in sando::helpers),
//   hgp_planner.hpp (HGPPlanner plan/get_path/get_raw_path),
//   types.hpp (Parameters).
//
// NOT ported (documented honestly):
//   - cvx_ellipsoid_decomp / cvx_ellipsoid_decomp_time_layered (the SFC / Safe
//     Flight Corridor convex-decomposition surrogate). These build per-segment
//     A x <= b polytopes for the Gurobi QP; they pull in the Polytope type and are
//     out of scope for this port. Skipped per instruction (NON-trivial, not ported).
//   - is_map_initialized() is ported (trivial flag); time.perf_counter timing is
//     non-reproducible and not used by HGPManager (no timings dict in Python here).
//
// 必须还原:每个公式照 Python 原样移植,无 stub/近似;golden 逐项验证,对不上就 FAIL。
#pragma once
#include "voxel_map.hpp"
#include "voxel_map_helpers.hpp"
#include "hgp_planner.hpp"
#include "types.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

namespace sando {

class HGPManager {
 public:
  // ---- owned state (mirror Python __init__) ----
  Parameters par;                          // self.par
  VoxelMapUtil map_util;                    // self.map_util (res defaults to par.res)
  std::unique_ptr<HGPPlanner> planner;      // self.planner (None until setup)

  double weight = 1.0;                      // global_planner_heuristic_weight (eps)
  double res;                               // HGP grid edge = factor_hgp*par.res
  double drone_radius = 0.2;
  double v_max = 10.0;
  double a_max = 20.0;
  double j_max = 100.0;
  double max_dist_vertexes = 2.0;
  bool use_shrinked_box = false;
  double shrinked_box_size = 0.0;
  std::vector<double> sfc_size = {3.0, 3.0, 3.0};
  bool map_initialized = false;

  // Python initializes map_util with res=par.res (the default Parameters().res).
  HGPManager() : map_util(par.res), res(par.res) {}

  // ------------------------------------------------------------------
  // set_parameters — push the full yaml Parameters into this class + map_util.
  //   self.res = factor_hgp * res  (coarser global grid), drone_radius from bbox.
  // ------------------------------------------------------------------
  void set_parameters(const Parameters& p) {
    par = p;
    weight = p.global_planner_heuristic_weight;
    res = p.factor_hgp * p.res;
    // drone_radius = max(par.drone_bbox)*0.5 if drone_bbox else 0.2
    if (!p.drone_bbox.empty()) {
      double mx = p.drone_bbox[0];
      for (double v : p.drone_bbox) mx = std::max(mx, v);
      drone_radius = mx * 0.5;
    } else {
      drone_radius = 0.2;
    }
    v_max = p.v_max;
    a_max = p.a_max;
    j_max = p.j_max;
    sfc_size = p.sfc_size;
    max_dist_vertexes = p.max_dist_vertexes;
    use_shrinked_box = p.use_shrinked_box;
    shrinked_box_size = p.shrinked_box_size;

    map_util.res = res;
    map_util.use_heat_map = p.use_heat_map;
    map_util.dynamic_heat_enabled = p.dynamic_heat_enabled;
    map_util.dynamic_as_occupied_current = p.dynamic_as_occupied_current;
    map_util.dynamic_as_occupied_future = p.dynamic_as_occupied_future;
    map_util.heat_alpha0 = p.heat_alpha0;
    map_util.heat_alpha1 = p.heat_alpha1;
    map_util.heat_p = p.heat_p;
    map_util.heat_q = p.heat_q;
    map_util.heat_tau_ratio = p.heat_tau_ratio;
    map_util.heat_gamma = p.heat_gamma;
    map_util.heat_Hmax = p.heat_Hmax;
    map_util.dyn_base_inflation_m = p.dyn_base_inflation_m;
    map_util.dyn_heat_tube_radius_m = p.dyn_heat_tube_radius_m;
    map_util.heat_num_samples = p.heat_num_samples;
    map_util.obst_max_vel = p.obst_max_vel;
    map_util.static_heat_enabled = p.static_heat_enabled;
    map_util.static_heat_alpha = p.static_heat_alpha;
    map_util.static_heat_p = p.static_heat_p;
    map_util.static_heat_Hmax = p.static_heat_Hmax;
    map_util.static_heat_rmax_m = p.static_heat_rmax_m;
    map_util.static_heat_default_radius_m = p.static_heat_default_radius_m;
    map_util.static_heat_boundary_only = p.static_heat_boundary_only;
    map_util.static_heat_apply_on_unknown = p.static_heat_apply_on_unknown;
    map_util.static_heat_exclude_dynamic = p.static_heat_exclude_dynamic;
    map_util.use_soft_cost_obstacles = p.use_soft_cost_obstacles;
    map_util.obstacle_soft_cost = p.obstacle_soft_cost;
  }

  // ------------------------------------------------------------------
  // setup_planner — (re)create the HGPPlanner from par and bind map_util.
  // ------------------------------------------------------------------
  void setup_planner() {
    planner.reset(new HGPPlanner(
        par.global_planner, par.global_planner_verbose, par.v_max, par.a_max,
        par.j_max, par.hgp_timeout_duration_ms, par.w_unknown, par.w_align,
        par.decay_len_cells, par.w_side, par.los_cells, par.min_len, par.min_turn,
        par.heat_weight, par.obstacle_soft_cost));
    planner->set_max_expand(par.max_num_expansion);
    planner->set_map_util(map_util);
  }

  // ------------------------------------------------------------------
  // update_map — rebuild the grid each cycle. wdx/wdy/wdz = physical map size (m);
  // converted to cell counts (>=2) and handed to read_map. First build also
  // constructs the planner.
  // ------------------------------------------------------------------
  void update_map(double wdx, double wdy, double wdz,
                  const Eigen::Vector3d& center_map,
                  const std::vector<Eigen::Vector3d>& cloud_occ,
                  const std::vector<Eigen::Vector3d>& /*cloud_unk*/,
                  const std::vector<Eigen::Vector3d>& obst_pos,
                  const std::vector<Eigen::Vector3d>& obst_bbox,
                  double traj_max_time) {
    double r = res;
    // max(2, int(round(w/res))). Python round() = round-half-to-even.
    long cells_x = std::max<long>(2, helpers::py_round_to_long(wdx / r));
    long cells_y = std::max<long>(2, helpers::py_round_to_long(wdy / r));
    long cells_z = std::max<long>(2, helpers::py_round_to_long(wdz / r));
    map_util.read_map(cells_x, cells_y, cells_z, center_map, cloud_occ, par.z_min,
                      par.z_max, par.inflation_hgp, obst_pos, obst_bbox,
                      traj_max_time);
    map_initialized = true;
    if (!planner) setup_planner();
  }

  // ------------------------------------------------------------------
  // free_start / free_goal — carve a free cube of half-size d=factor*res around p.
  // ------------------------------------------------------------------
  Eigen::Vector3d free_start(const Eigen::Vector3d& start, double factor) {
    double d = factor * res;
    helpers::set_free_voxel_and_surroundings(map_util, start, d);
    return start;
  }
  Eigen::Vector3d free_goal(const Eigen::Vector3d& goal, double factor) {
    double d = factor * res;
    helpers::set_free_voxel_and_surroundings(map_util, goal, d);
    return goal;
  }

  // ------------------------------------------------------------------
  // point occupancy / free queries.
  // ------------------------------------------------------------------
  bool check_if_point_occupied(const Eigen::Vector3d& p) const {
    Eigen::Array<long, 3, 1> c = map_util.float_to_int(p);
    return map_util.is_occupied(c(0), c(1), c(2));
  }
  bool check_if_point_free(const Eigen::Vector3d& p) const {
    Eigen::Array<long, 3, 1> c = map_util.float_to_int(p);
    return map_util.is_free(c(0), c(1), c(2));
  }

  // ------------------------------------------------------------------
  // solve_hgp — one full global call:
  //   optional free-start/free-goal, planner.plan(), get_path()/get_raw_path(),
  //   densify processed path (>=2 pts) by max_dist_vertexes.
  // Returns (success, final_g, processed_path, raw_path).
  // ------------------------------------------------------------------
  struct SolveResult {
    bool success = false;
    double final_g = 0.0;
    std::vector<Eigen::Vector3d> path;       // densified processed path
    std::vector<Eigen::Vector3d> raw_path;   // raw A* path
  };

  SolveResult solve_hgp(const Eigen::Vector3d& start,
                        const Eigen::Vector3d& start_vel,
                        const Eigen::Vector3d& goal, double current_time) {
    if (!planner) setup_planner();
    if (par.use_free_start) free_start(start, par.free_start_factor);
    if (par.use_free_goal) free_goal(goal, par.free_goal_factor);

    std::pair<bool, double> r =
        planner->plan(start, start_vel, goal, current_time, weight);
    bool ok = r.first;
    double final_g = r.second;

    std::vector<Eigen::Vector3d> path = ok ? planner->get_path()
                                           : std::vector<Eigen::Vector3d>();
    std::vector<Eigen::Vector3d> raw_path = planner->get_raw_path();

    if (path.size() >= 2) path = create_more_vertexes(path, max_dist_vertexes);

    SolveResult out;
    out.success = ok;
    out.final_g = final_g;
    out.path = std::move(path);
    out.raw_path = std::move(raw_path);
    return out;
  }

  // ------------------------------------------------------------------
  // check_if_path_in_free — longest free prefix from the start (stop at first
  // non-free point). Returns (>=2 kept, prefix).
  // ------------------------------------------------------------------
  std::pair<bool, std::vector<Eigen::Vector3d>> check_if_path_in_free(
      const std::vector<Eigen::Vector3d>& path) const {
    if (path.size() < 2) return {false, {}};
    std::vector<Eigen::Vector3d> out;
    out.push_back(path[0]);
    for (size_t i = 1; i < path.size(); ++i) {
      if (check_if_point_free(path[i]))
        out.push_back(path[i]);
      else
        break;
    }
    return {out.size() >= 2, out};
  }

  // ------------------------------------------------------------------
  // push_path_into_free_space — snap each point to the nearest free cell center.
  // ------------------------------------------------------------------
  std::vector<Eigen::Vector3d> push_path_into_free_space(
      const std::vector<Eigen::Vector3d>& path) const {
    std::vector<Eigen::Vector3d> out;
    out.reserve(path.size());
    for (const auto& p : path)
      out.push_back(helpers::find_closest_free_point(map_util, p));
    return out;
  }

  // ------------------------------------------------------------------
  // create_more_vertexes — insert a vertex every d metres on each edge of the
  // polyline. Edges shorter than d are left as-is; the trailing endpoint is added
  // when the last inserted point hasn't already reached it (>1e-6).
  // ------------------------------------------------------------------
  std::vector<Eigen::Vector3d> create_more_vertexes(
      const std::vector<Eigen::Vector3d>& path, double d) const {
    if (path.size() < 2 || d <= 0.0) {
      // [p.copy() for p in path]
      return path;
    }
    std::vector<Eigen::Vector3d> out;
    out.push_back(path[0]);
    for (size_t e = 0; e + 1 < path.size(); ++e) {
      const Eigen::Vector3d& a = path[e];
      const Eigen::Vector3d& b = path[e + 1];
      Eigen::Vector3d seg = b - a;
      double L = seg.norm();
      if (L <= 1e-9) continue;
      long n = static_cast<long>(std::floor(L / d));
      Eigen::Vector3d u = seg / L;
      for (long k = 1; k <= n; ++k) out.push_back(a + u * (static_cast<double>(k) * d));
      if ((out.back() - b).norm() > 1e-6) out.push_back(b);
    }
    return out;
  }

  // ------------------------------------------------------------------
  // visualization helpers.
  // ------------------------------------------------------------------
  std::vector<Eigen::Vector3d> get_occupied_cells() const {
    return helpers::cells_world(map_util, VAL_OCC);
  }
  std::vector<Eigen::Vector3d> get_free_cells() const {
    return helpers::cells_world(map_util, VAL_FREE);
  }

  bool is_map_initialized() const { return map_initialized; }
};

}  // namespace sando
