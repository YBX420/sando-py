// GraphSearch — faithful C++ port of sando_py/hgp/graph_search.py:GraphSearch.
// heat-A* over a 3D voxel grid (26-connected). 1:1 with the Python:
//   - same 26-neighbor iteration order (dx,dy,dz in (-1,0,1)^3 minus (0,0,0)),
//   - same priority-queue ordering: f = g + h ascending, tie-break by insertion
//     `counter` ascending (FIFO) — replicates Python's @dataclass(order=True)
//     _PQEntry(f, counter, state_id[compare=False]) EXACTLY so the path is
//     deterministic and identical to Python,
//   - same heat cost addition (heat_weight*heat when use_heat),
//   - same Euclidean heuristic * eps,
//   - same lazy-deletion closed-set, best_node fallback, corner-cut rule,
//   - same recover_path backtracking.
// Uses VoxelMapUtil from voxel_map.hpp (occupancy cmap + heat field).
// 必须还原:每个公式对照 Python 原样移植;golden 逐项验证,对不上就 FAIL。
#pragma once
#include "voxel_map.hpp"
#include <Eigen/Dense>
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
#include <cstdint>

namespace sando {

// One A* search node (one grid cell). Mirrors Python `State` dataclass.
struct GSState {
  long id = -1;
  long x = 0, y = 0, z = 0;
  long dx = 0, dy = 0, dz = 0;
  double g = std::numeric_limits<double>::infinity();
  double h = 0.0;
  long parent_id = -1;
  bool opened = false;
  bool closed = false;
  bool exists = false;  // mirrors Python `_hm[id] is None` (None => exists=false)
};

class GraphSearch {
 public:
  const VoxelMapUtil* map_util = nullptr;
  double eps = 1.0;
  std::string global_planner = "astar_heat";
  double w_unknown = 0.0;
  double w_align = 0.0;
  double decay_len_cells = 20.0;
  double w_side = 0.2;
  bool verbose = false;
  double heat_weight = 10.0;
  double obstacle_soft_cost = 5.0;

  bool use_heat = true;        // (global_planner == "astar_heat")
  bool use_soft_cost = true;   // map_util.use_soft_cost_obstacles

  // Search state.
  std::vector<GSState> hm_;    // _hm: size = total_size(); exists flag = "not None"
  std::vector<GSState*> path_; // _path (pointers into hm_)
  Eigen::Vector3d start_vel = Eigen::Vector3d::Zero();
  long goal_int[3] = {0, 0, 0};

  // ------------------------------------------------------------------
  GraphSearch(const VoxelMapUtil& mu, double eps_ = 1.0,
              const std::string& gp = "astar_heat", double w_unknown_ = 0.0,
              double w_align_ = 0.0, double decay_len_cells_ = 20.0,
              double w_side_ = 0.2, bool verbose_ = false,
              double heat_weight_ = 10.0, double obstacle_soft_cost_ = 5.0)
      : map_util(&mu),
        eps(eps_),
        global_planner(gp),
        w_unknown(w_unknown_),
        w_align(w_align_),
        decay_len_cells(decay_len_cells_),
        w_side(w_side_),
        verbose(verbose_),
        heat_weight(heat_weight_),
        obstacle_soft_cost(obstacle_soft_cost_) {
    use_heat = (global_planner == "astar_heat");
    use_soft_cost = mu.use_soft_cost_obstacles;
  }

  // ------------------------------------------------------------------
  long coord_to_id(long x, long y, long z) const {
    return map_util->lin_index(x, y, z);
  }

  // Heuristic h: Euclidean distance to goal * eps.
  double heur(long x, long y, long z) const {
    double dx = double(x - goal_int[0]);
    double dy = double(y - goal_int[1]);
    double dz = double(z - goal_int[2]);
    return eps * std::sqrt(dx * dx + dy * dy + dz * dz);
  }

  // Lazy state accessor (creates+caches on first access).
  GSState& state_at(long x, long y, long z) {
    long sid = coord_to_id(x, y, z);
    GSState& s = hm_[size_t(sid)];
    if (!s.exists) {
      s.exists = true;
      s.id = sid;
      s.x = x; s.y = y; s.z = z;
      s.h = heur(x, y, z);
      s.g = std::numeric_limits<double>::infinity();
      s.parent_id = -1;
      s.dx = s.dy = s.dz = 0;
      s.opened = false;
      s.closed = false;
    }
    return s;
  }

  bool is_blocked(long x, long y, long z) const {
    if (!map_util->in_bounds(x, y, z)) return true;
    return map_util->cmap[size_t(coord_to_id(x, y, z))] == VAL_OCC;
  }

  // No corner-cutting: a diagonal step must not graze an occupied/out-of-bounds
  // cell. Checks every shoulder cell formed by a non-empty proper subset of the
  // move's nonzero components. Mirrors Python _corner_cut using itertools.
  bool corner_cut(long x, long y, long z, long dx, long dy, long dz) const {
    // nz = list of (axis index, delta) for nonzero components, in axis order.
    long axes[3];
    long deltas[3];
    int nnz = 0;
    long comps[3] = {dx, dy, dz};
    for (int i = 0; i < 3; ++i) {
      if (comps[i] != 0) {
        axes[nnz] = i;
        deltas[nnz] = comps[i];
        ++nnz;
      }
    }
    if (nnz <= 1) return false;
    // for r in range(1, nnz): for each r-subset (combinations preserve order):
    for (int r = 1; r < nnz; ++r) {
      // iterate combinations of size r over indices [0..nnz)
      std::vector<int> idx(r);
      for (int i = 0; i < r; ++i) idx[i] = i;
      while (true) {
        long s[3] = {0, 0, 0};
        for (int i = 0; i < r; ++i) s[axes[idx[i]]] = deltas[idx[i]];
        if (is_blocked(x + s[0], y + s[1], z + s[2])) return true;
        // advance combination
        int pos = r - 1;
        while (pos >= 0 && idx[pos] == nnz - r + pos) --pos;
        if (pos < 0) break;
        ++idx[pos];
        for (int i = pos + 1; i < r; ++i) idx[i] = idx[i - 1] + 1;
      }
    }
    return false;
  }

  // ------------------------------------------------------------------
  // Priority queue entry: ordered by (f ascending, counter ascending) to match
  // Python heapq with @dataclass(order=True) _PQEntry(f, counter, state_id).
  struct PQEntry {
    double f;
    long counter;
    long state_id;
  };
  struct PQCompare {
    // std::priority_queue is a MAX-heap; return true when a has LOWER priority
    // than b, i.e. a should come out AFTER b. We want the smallest (f, counter)
    // out first, so a has lower priority when (a.f, a.counter) > (b.f, b.counter).
    bool operator()(const PQEntry& a, const PQEntry& b) const {
      if (a.f != b.f) return a.f > b.f;
      return a.counter > b.counter;
    }
  };

  // ------------------------------------------------------------------
  // plan — faithful to Python plan(start_int, goal_int, initial_g, start_vel,
  // max_expand, timeout_ms). timeout is NOT enforced here (deterministic golden;
  // Python uses wall-clock which is non-reproducible) — we expose it but, like
  // the golden, rely on max_expand for the deterministic stop. Set a large
  // max_expand to match Python runs that never time out.
  bool plan(const long start_int[3], const long goal_int_[3], double initial_g,
            const Eigen::Vector3d& start_vel_, long max_expand = 100000) {
    goal_int[0] = goal_int_[0];
    goal_int[1] = goal_int_[1];
    goal_int[2] = goal_int_[2];
    start_vel = start_vel_;
    long size = map_util->total_size();
    hm_.assign(size_t(size), GSState());
    path_.clear();

    long sx = start_int[0], sy = start_int[1], sz = start_int[2];
    long gx = goal_int_[0], gy = goal_int_[1], gz = goal_int_[2];
    if (!map_util->in_bounds(sx, sy, sz) || !map_util->in_bounds(gx, gy, gz))
      return false;

    GSState& start_state = state_at(sx, sy, sz);
    start_state.g = initial_g;
    start_state.opened = true;
    long start_id = start_state.id;
    GSState& goal_state = state_at(gx, gy, gz);
    long goal_id = goal_state.id;

    std::priority_queue<PQEntry, std::vector<PQEntry>, PQCompare> pq;
    long counter = 0;
    pq.push(PQEntry{start_state.g + start_state.h, counter, start_state.id});
    ++counter;

    long best_id = start_id;
    double best_h = hm_[size_t(start_id)].h;
    long expansions = 0;
    bool success = false;

    while (!pq.empty()) {
      PQEntry top = pq.top();
      pq.pop();
      GSState& curr = hm_[size_t(top.state_id)];
      // stale / closed entries: lazy deletion.
      if (!curr.exists || curr.closed) continue;
      curr.closed = true;
      if (curr.h < best_h) {
        best_h = curr.h;
        best_id = curr.id;
      }
      if (curr.id == goal_id) {
        success = true;
        break;
      }
      ++expansions;
      if (expansions > max_expand) break;

      long cx = curr.x, cy = curr.y, cz = curr.z;
      double cg = curr.g;
      // 26-connected expansion: dx,dy,dz in (-1,0,1) skipping (0,0,0), in the
      // exact Python nesting order (dx outer, dy mid, dz inner).
      for (long dx = -1; dx <= 1; ++dx) {
        for (long dy = -1; dy <= 1; ++dy) {
          for (long dz = -1; dz <= 1; ++dz) {
            if (dx == 0 && dy == 0 && dz == 0) continue;
            long nx = cx + dx, ny = cy + dy, nz = cz + dz;
            if (!map_util->in_bounds(nx, ny, nz)) continue;
            int8_t val = map_util->cmap[size_t(coord_to_id(nx, ny, nz))];
            if (val == VAL_OCC && !use_soft_cost) continue;
            if (!use_soft_cost && corner_cut(cx, cy, cz, dx, dy, dz)) continue;
            GSState& child = state_at(nx, ny, nz);
            if (child.closed) continue;
            double step =
                std::sqrt(double(dx * dx + dy * dy + dz * dz));
            double cost = step;
            if (val == -1 && w_unknown > 0) cost += w_unknown * step;
            if (use_heat) {
              double h_val = map_util->get_heat(nx, ny, nz);
              if (h_val > 0 && heat_weight > 0) cost += heat_weight * h_val;
              if (val == VAL_OCC && use_soft_cost)
                cost += heat_weight * obstacle_soft_cost;
            }
            double tentative_g = cg + cost;
            if (tentative_g < child.g) {
              child.g = tentative_g;
              child.parent_id = curr.id;
              child.dx = dx; child.dy = dy; child.dz = dz;
              pq.push(PQEntry{tentative_g + child.h, counter, child.id});
              ++counter;
              child.opened = true;
            }
          }
        }
      }
    }

    // Recover path.
    long end_id = success ? goal_id : best_id;
    if (end_id == start_id && !success) success = true;
    path_ = recover_path(end_id, start_id);
    return success && (path_.size() >= 1);
  }

  // Backtrack parent_id chain from end to start, then reverse.
  std::vector<GSState*> recover_path(long end_id, long start_id) {
    std::vector<GSState*> path;
    long cur = end_id;
    while (cur >= 0) {
      GSState& s = hm_[size_t(cur)];
      if (!s.exists) break;
      path.push_back(&s);
      if (cur == start_id || s.parent_id < 0) break;
      cur = s.parent_id;
      if (cur < 0 || !hm_[size_t(cur)].exists) break;
    }
    std::reverse(path.begin(), path.end());
    return path;
  }

  // World-coordinate path (cell centers), like Python get_path_world.
  std::vector<Eigen::Vector3d> get_path_world() const {
    std::vector<Eigen::Vector3d> out;
    out.reserve(path_.size());
    for (const GSState* s : path_)
      out.push_back(map_util->int_to_float(s->x, s->y, s->z));
    return out;
  }
};

}  // namespace sando
