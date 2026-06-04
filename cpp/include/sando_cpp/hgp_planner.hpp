// HGPPlanner — faithful C++ port of sando_py/hgp/hgp_planner.py:HGPPlanner.
//
// Orchestrates voxel-map heat-A* search (GraphSearch) + the path post-processing
// pipeline, 1:1 with the Python:
//   plan(start, start_vel, goal, current_time, eps):
//     - float_to_int(start/goal), initial_g = ||start - int_to_float(start_int)||,
//     - GraphSearch.plan(start_int, goal_int, initial_g, start_vel, max_expand),
//     - raw_path = get_path_world(); raw_path[0]=start; raw_path[-1]=goal if ok,
//     - post-processing pipeline (all blocked-checks use is_blocked so no
//       simplification ever cuts through an obstacle):
//         1) short_cut_by_los(raw, los_cells)         [line_of_sight_capsule]
//         2) collapse_short_edges(path, min_len, blk)
//         3) angle_spacing_filter(path, min_turn_deg, min_len, blk)
//         4) short_cut_by_los(path, los_cells)
//         5) clean_up_path(path)  = remove_collinear -> _remove_corner_pts
//                                   forward + reversed (cost-aware, heat-integral
//                                   + 1.5x peak-heat guardrail)
//         6) _repair_blocked_segments(path, raw)       [safety net]
//     - final_g = _path_cost(path) (pure geometric length).
//
// REUSES the already golden-verified headers:
//   voxel_map.hpp (VoxelMapUtil), voxel_map_helpers.hpp (is_blocked /
//   line_of_sight_capsule free fns), graph_search.hpp (GraphSearch).
//
// NOT ported (documented honestly):
//   - wall-clock timeout (Python time.perf_counter deadline inside GraphSearch
//     and HGPPlanner.timings). The C++ GraphSearch already drops the timeout
//     (deterministic golden); HGPPlanner mirrors that and exposes timeout_ms as a
//     stored field only. Golden is generated with a huge timeout that never trips.
//   - self.timings dict (wall-clock profiling, non-reproducible).
//   - graph_search.last_metrics propagation (wall-clock only).
//
// 必须还原:每个公式照 Python 原样移植,无 stub/近似;golden 逐项验证,对不上就 FAIL。
#pragma once
#include "voxel_map.hpp"
#include "voxel_map_helpers.hpp"
#include "graph_search.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>

namespace sando {

class HGPPlanner {
 public:
  // ---- constructor knobs (mirror Python __init__) ----
  std::string global_planner = "astar_heat";
  bool verbose = false;
  double v_max = 0.0;
  double a_max = 0.0;
  double j_max = 0.0;
  long timeout_ms = 1000;        // stored only; NOT enforced (see header note)
  double w_unknown = 0.0;
  double w_align = 0.0;
  double decay_len_cells = 20.0;
  double w_side = 0.2;
  long los_cells = 0;
  double min_len = 0.5;
  double min_turn_deg = 10.0;    // Python `min_turn` -> self.min_turn_deg
  double heat_weight = 10.0;
  double obstacle_soft_cost = 5.0;

  // ---- search wiring ----
  const VoxelMapUtil* map_util = nullptr;
  GraphSearch* graph_search = nullptr;   // owned via graph_search_storage_
  long max_expand = 100000;

  // ---- outputs ----
  std::vector<Eigen::Vector3d> raw_path;
  std::vector<Eigen::Vector3d> path;

  // ------------------------------------------------------------------
  HGPPlanner(const std::string& global_planner_, bool verbose_, double v_max_,
             double a_max_, double j_max_, long timeout_duration_ms,
             double w_unknown_, double w_align_, double decay_len_cells_,
             double w_side_, long los_cells_ = 0, double min_len_ = 0.5,
             double min_turn_ = 10.0, double heat_weight_ = 10.0,
             double obstacle_soft_cost_ = 5.0)
      : global_planner(global_planner_),
        verbose(verbose_),
        v_max(v_max_),
        a_max(a_max_),
        j_max(j_max_),
        timeout_ms(timeout_duration_ms),
        w_unknown(w_unknown_),
        w_align(w_align_),
        decay_len_cells(decay_len_cells_),
        w_side(w_side_),
        los_cells(los_cells_),
        min_len(min_len_),
        min_turn_deg(min_turn_),
        heat_weight(heat_weight_),
        obstacle_soft_cost(obstacle_soft_cost_) {}

  ~HGPPlanner() { delete graph_search; }

  // Bind a map and (re)create the GraphSearch with the planner knobs forwarded.
  void set_map_util(const VoxelMapUtil& mu) {
    map_util = &mu;
    delete graph_search;
    graph_search = new GraphSearch(mu, 1.0, global_planner, w_unknown, w_align,
                                   decay_len_cells, w_side, verbose, heat_weight,
                                   obstacle_soft_cost);
  }

  void set_max_expand(long n) { max_expand = n; }
  void update_vmax(double v) { v_max = v; }

  // ------------------------------------------------------------------
  // plan — returns (success, final_g). 1:1 with Python plan().
  // ------------------------------------------------------------------
  std::pair<bool, double> plan(const Eigen::Vector3d& start,
                               const Eigen::Vector3d& start_vel,
                               const Eigen::Vector3d& goal, double /*current_time*/,
                               double eps = 1.0) {
    if (map_util == nullptr || !map_util->initialized) return {false, 0.0};

    graph_search->eps = eps;
    Eigen::Array<long, 3, 1> start_int_a = map_util->float_to_int(start);
    Eigen::Array<long, 3, 1> goal_int_a = map_util->float_to_int(goal);
    long start_int[3] = {start_int_a(0), start_int_a(1), start_int_a(2)};
    long goal_int[3] = {goal_int_a(0), goal_int_a(1), goal_int_a(2)};
    // initial_g: offset of the true start from its cell center (A* start cost comp).
    double initial_g =
        (start - map_util->int_to_float(start_int_a)).norm();

    bool ok = graph_search->plan(start_int, goal_int, initial_g, start_vel, max_expand);

    std::vector<Eigen::Vector3d> raw = graph_search->get_path_world();
    if (raw.empty()) {
      raw_path.clear();
      path.clear();
      return {false, 0.0};
    }
    // Override start/goal endpoints with their world positions.
    raw[0] = start;
    if (ok) raw.back() = goal;
    raw_path = raw;

    std::vector<Eigen::Vector3d> p = short_cut_by_los(raw, los_cells);
    p = collapse_short_edges_blk(p, min_len);
    p = angle_spacing_filter_blk(p, min_turn_deg, min_len);
    p = short_cut_by_los(p, los_cells);
    p = clean_up_path(p);
    p = repair_blocked_segments(p, raw);
    path = p;

    double final_g = path_cost(path);
    return {ok, final_g};
  }

  // ------------------------------------------------------------------
  // short_cut_by_los: greedily connect the farthest line-of-sight-visible point.
  // ------------------------------------------------------------------
  std::vector<Eigen::Vector3d> short_cut_by_los(
      const std::vector<Eigen::Vector3d>& points, long radius_cells) const {
    if (points.size() <= 2) return points;
    std::vector<Eigen::Vector3d> out;
    out.push_back(points[0]);
    long i = 0;
    long n = static_cast<long>(points.size());
    while (i < n - 1) {
      long j = n - 1;
      while (j > i + 1) {
        if (helpers::line_of_sight_capsule(*map_util, points[size_t(i)],
                                           points[size_t(j)],
                                           static_cast<int>(radius_cells)))
          break;
        --j;
      }
      out.push_back(points[size_t(j)]);
      i = j;
    }
    return out;
  }

  // clean_up_path: collinear removal, then corner removal forward + reversed.
  std::vector<Eigen::Vector3d> clean_up_path(
      const std::vector<Eigen::Vector3d>& points) const {
    std::vector<Eigen::Vector3d> p = remove_collinear(points);
    p = remove_corner_pts(p);
    std::vector<Eigen::Vector3d> rev(p.rbegin(), p.rend());
    std::vector<Eigen::Vector3d> p_rev = remove_corner_pts(rev);
    std::vector<Eigen::Vector3d> out(p_rev.rbegin(), p_rev.rend());
    return out;
  }

  // ------------------------------------------------------------------
  // _repair_blocked_segments: splice raw A* sub-paths back where a simplified
  // segment is blocked. nearest_raw replicates Python min(range(...), key=...)
  // first-minimizer-wins tie-break.
  // ------------------------------------------------------------------
  std::vector<Eigen::Vector3d> repair_blocked_segments(
      const std::vector<Eigen::Vector3d>& p,
      const std::vector<Eigen::Vector3d>& raw) const {
    if (p.size() < 2 || raw.size() < 2) return p;

    auto nearest_raw = [&](const Eigen::Vector3d& pt) -> long {
      long best = 0;
      double best_d = (raw[0] - pt).norm();
      for (long r = 1; r < static_cast<long>(raw.size()); ++r) {
        double d = (raw[size_t(r)] - pt).norm();
        if (d < best_d) {  // strict < => first minimizer wins (Python min())
          best_d = d;
          best = r;
        }
      }
      return best;
    };

    std::vector<Eigen::Vector3d> out;
    out.push_back(p[0]);
    for (size_t k = 0; k + 1 < p.size(); ++k) {
      const Eigen::Vector3d& a = p[k];
      const Eigen::Vector3d& b = p[k + 1];
      if (helpers::is_blocked(*map_util, a, b, VAL_OCC)) {
        long ia = nearest_raw(a), ib = nearest_raw(b);
        if (ia < ib) {
          for (long q = ia + 1; q < ib; ++q) out.push_back(raw[size_t(q)]);
        }
      }
      out.push_back(b);
    }
    return out;
  }

  // ------------------------------------------------------------------
  // _remove_corner_pts: cost-aware corner removal with heat-integral cost and a
  // 1.5x peak-heat guardrail.
  // ------------------------------------------------------------------
  std::vector<Eigen::Vector3d> remove_corner_pts(
      const std::vector<Eigen::Vector3d>& points) const {
    if (points.size() <= 2) return points;
    std::vector<Eigen::Vector3d> out;
    out.push_back(points[0]);
    double prev_cost = seg_cost(out.back(), points[1]);
    double prev_peak = seg_peak_heat(out.back(), points[1]);
    size_t i = 1;
    while (i < points.size() - 1) {
      const Eigen::Vector3d& mid = points[i];
      const Eigen::Vector3d& nxt = points[i + 1];
      double cost2 = seg_cost(mid, nxt);
      double peak2 = seg_peak_heat(mid, nxt);
      double cost3 = seg_cost(out.back(), nxt);
      double peak3 = seg_peak_heat(out.back(), nxt);
      if (cost3 < prev_cost + cost2 &&
          peak3 <= 1.5 * std::max(prev_peak, peak2)) {
        prev_cost = cost3;
        prev_peak = peak3;
      } else {
        out.push_back(mid);
        prev_cost = cost2;
        prev_peak = peak2;
      }
      ++i;
    }
    out.push_back(points.back());
    return out;
  }

  // ------------------------------------------------------------------
  // _seg_cost: length + heat_weight * heat-line-integral. inf if blocked.
  // ------------------------------------------------------------------
  double seg_cost(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
    if (helpers::is_blocked(*map_util, a, b, VAL_OCC))
      return std::numeric_limits<double>::infinity();
    double L = (b - a).norm();
    if (L < 1e-9) return 0.0;
    double ds = std::max(0.5 * map_util->res, 0.05);
    long n = std::max<long>(2, static_cast<long>(std::ceil(L / ds)));
    double heat_int = 0.0;
    for (long k = 0; k <= n; ++k) {
      double t = static_cast<double>(k) / static_cast<double>(n);
      Eigen::Vector3d pt = a + t * (b - a);
      Eigen::Array<long, 3, 1> cell = map_util->float_to_int(pt);
      heat_int += map_util->get_heat(cell(0), cell(1), cell(2)) *
                  (L / static_cast<double>(n));
    }
    return L + heat_weight * heat_int;
  }

  // _seg_peak_heat: max heat sampled along a->b.
  double seg_peak_heat(const Eigen::Vector3d& a, const Eigen::Vector3d& b) const {
    double L = (b - a).norm();
    if (L < 1e-9) {
      Eigen::Array<long, 3, 1> cell = map_util->float_to_int(a);
      return map_util->get_heat(cell(0), cell(1), cell(2));
    }
    double ds = std::max(0.5 * map_util->res, 0.05);
    long n = std::max<long>(2, static_cast<long>(std::ceil(L / ds)));
    double peak = 0.0;
    for (long k = 0; k <= n; ++k) {
      double t = static_cast<double>(k) / static_cast<double>(n);
      Eigen::Vector3d pt = a + t * (b - a);
      Eigen::Array<long, 3, 1> cell = map_util->float_to_int(pt);
      peak = std::max(peak, map_util->get_heat(cell(0), cell(1), cell(2)));
    }
    return peak;
  }

  // _path_cost: pure geometric length of the path.
  double path_cost(const std::vector<Eigen::Vector3d>& pp) const {
    double total = 0.0;
    for (size_t i = 0; i + 1 < pp.size(); ++i) total += (pp[i + 1] - pp[i]).norm();
    return total;
  }

  const std::vector<Eigen::Vector3d>& get_path() const { return path; }
  const std::vector<Eigen::Vector3d>& get_raw_path() const { return raw_path; }

  // ==================================================================
  // utils.py path-simplification helpers (used by the pipeline). Ported here
  // since they are free functions in Python (sando_py.utils) and only HGPPlanner
  // consumes them in this module.
  // ==================================================================

  // remove_collinear: drop interior points where a->b and b->c are ~collinear
  // (sin(angle) = |cross|/(|ab||bc|) <= tol).
  static std::vector<Eigen::Vector3d> remove_collinear(
      const std::vector<Eigen::Vector3d>& points, double tol = 1e-3) {
    if (points.size() <= 2) return points;
    std::vector<Eigen::Vector3d> out;
    out.push_back(points[0]);
    for (size_t i = 1; i + 1 < points.size(); ++i) {
      const Eigen::Vector3d& a = out.back();
      const Eigen::Vector3d& b = points[i];
      const Eigen::Vector3d& c = points[i + 1];
      Eigen::Vector3d ab = b - a;
      Eigen::Vector3d bc = c - b;
      double n_ab = ab.norm();
      double n_bc = bc.norm();
      if (n_ab < 1e-9 || n_bc < 1e-9) continue;
      double cross = ab.cross(bc).norm();
      if (cross / (n_ab * n_bc) > tol) out.push_back(b);
    }
    out.push_back(points.back());
    return out;
  }

  // collapse_short_edges: drop points closer than min_len to the last kept point,
  // unless dropping would merge across an obstacle (is_blocked guard). Uses this
  // planner's map_util for the is_blocked callback.
  std::vector<Eigen::Vector3d> collapse_short_edges_blk(
      const std::vector<Eigen::Vector3d>& points, double min_len_) const {
    if (points.empty()) return {};
    std::vector<Eigen::Vector3d> out;
    out.push_back(points[0]);
    long n = static_cast<long>(points.size());
    for (long i = 1; i < n; ++i) {
      const Eigen::Vector3d& pcur = points[size_t(i)];
      if ((pcur - out.back()).norm() >= min_len_) {
        out.push_back(pcur);
      } else if (i + 1 < n &&
                 helpers::is_blocked(*map_util, out.back(), points[size_t(i + 1)],
                                     VAL_OCC)) {
        out.push_back(pcur);
      }
    }
    if (out.size() >= 2 && (out.back() - points.back()).norm() > 1e-9) {
      out.back() = points.back();
    } else if (out.size() == 1) {
      out.push_back(points.back());
    }
    return out;
  }

  // angle_spacing_filter: drop interior points with tiny turn or short adjacent
  // segments, unless the merged segment a->c would clip an obstacle.
  std::vector<Eigen::Vector3d> angle_spacing_filter_blk(
      const std::vector<Eigen::Vector3d>& points, double min_turn_deg_,
      double min_seg_len) const {
    if (points.size() <= 2 || min_turn_deg_ <= 0) return points;
    // math.radians(deg): CPython computes x * (pi/180.0) with the (pi/180.0)
    // constant folded first — replicate that exact eval order so cos_thr matches
    // bit-for-bit (M_PI not portable on MinGW; use math.pi value explicitly).
    constexpr double kPi = 3.141592653589793;
    constexpr double kDegToRad = kPi / 180.0;
    double cos_thr = std::cos(min_turn_deg_ * kDegToRad);
    std::vector<Eigen::Vector3d> out;
    out.push_back(points[0]);
    for (size_t i = 1; i + 1 < points.size(); ++i) {
      const Eigen::Vector3d& a = out.back();
      const Eigen::Vector3d& b = points[i];
      const Eigen::Vector3d& c = points[i + 1];
      Eigen::Vector3d ab = b - a;
      Eigen::Vector3d bc = c - b;
      double nab = ab.norm();
      double nbc = bc.norm();
      bool want_skip = false;
      if (nab < min_seg_len || nbc < min_seg_len) {
        want_skip = true;
      } else {
        double cosang = ab.dot(bc) / (nab * nbc);
        if (cosang > cos_thr) want_skip = true;
      }
      if (want_skip && !helpers::is_blocked(*map_util, a, c, VAL_OCC)) continue;
      out.push_back(b);
    }
    out.push_back(points.back());
    return out;
  }
};

}  // namespace sando
