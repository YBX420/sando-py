// VoxelMapUtil geometric query helpers — faithful C++ port of the methods in
// sando_py/hgp/voxel_map.py that were NOT in the first voxel_map.hpp port:
//   set_free_voxel_and_surroundings, find_closest_free_point, is_blocked (DDA),
//   line_of_sight_capsule (inflated-capsule), cells_world, has_non_free_neighbor.
//
// This header does NOT edit voxel_map.hpp. It provides these as FREE FUNCTIONS in
// namespace sando::helpers, each taking a (const) VoxelMapUtil& (or &) plus args,
// 1:1 with the Python member methods. Reuses the already-ported, golden-verified
// VoxelMapUtil (cmap/origin/res/dim, float_to_int/int_to_float/in_bounds/lin_index/
// is_free/set_free, VAL_FREE/VAL_OCC/VAL_UNK).
//
// 必须还原:每个公式照 Python 原样移植,无 stub/近似;golden 逐项验证。
#pragma once
#include "sando_cpp/voxel_map.hpp"
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

namespace sando {
namespace helpers {

// ------------------------------------------------------------------
// Python's int(round(x)) — round-half-to-even (banker's rounding), then truncate
// toward zero into a long. Python 3 `round` returns the nearest even on ties.
// We only ever call this on x = d/res + 0.5 (>= 0 in practice), but implement the
// full banker's-rounding semantics so it matches Python exactly for any input.
// ------------------------------------------------------------------
inline long py_round_to_long(double x) {
  // std::nearbyint with the default rounding mode (round-to-nearest-even) matches
  // Python 3's round() tie-breaking. The default FE rounding mode is to-nearest-even.
  double r = std::nearbyint(x);
  return static_cast<long>(r);
}

// ------------------------------------------------------------------
// set_free_voxel_and_surroundings(center, d):
//   n = int(round(d / res + 0.5)); cx,cy,cz = float_to_int(center);
//   for dx,dy,dz in [-n, n]: set_free(cx+dx, cy+dy, cz+dz)
// 中文:把 center 周围 d 米范围内的一坨格子都挖空(确保起点/终点附近是空的)。
// ------------------------------------------------------------------
inline void set_free_voxel_and_surroundings(VoxelMapUtil& vm,
                                            const Eigen::Vector3d& center,
                                            double d) {
  long n = py_round_to_long(d / vm.res + 0.5);
  Eigen::Array<long, 3, 1> c = vm.float_to_int(center);
  long cx = c(0), cy = c(1), cz = c(2);
  for (long dx = -n; dx <= n; ++dx)
    for (long dy = -n; dy <= n; ++dy)
      for (long dz = -n; dz <= n; ++dz)
        vm.set_free(cx + dx, cy + dy, cz + dz);
}

// ------------------------------------------------------------------
// find_closest_free_point(point) -> world cell-center of nearest free voxel.
// If point's own cell is free, return its center. Else expand search radius from
// 1.0 m, step +0.5 m up to 5.0 m; for the first radius that finds any free cell,
// return the (Euclidean-)nearest such cell center. If none found, return point.
// 中文:把「卡在障碍里」的点拉回到最近的自由格中心。
// ------------------------------------------------------------------
inline Eigen::Vector3d find_closest_free_point(const VoxelMapUtil& vm,
                                               const Eigen::Vector3d& point) {
  Eigen::Array<long, 3, 1> pi = vm.float_to_int(point);
  long px = pi(0), py = pi(1), pz = pi(2);
  if (vm.is_free(px, py, pz)) {
    return vm.int_to_float(px, py, pz);
  }
  Eigen::Vector3d best = Eigen::Vector3d::Zero();
  bool have_best = false;
  double best_d = std::numeric_limits<double>::infinity();
  double radius_m = 1.0;
  while (radius_m <= 5.0) {
    long r = static_cast<long>(radius_m / vm.res);  // Python int() truncates toward 0
    for (long dx = -r; dx <= r; ++dx) {
      for (long dy = -r; dy <= r; ++dy) {
        for (long dz = -r; dz <= r; ++dz) {
          long xx = px + dx, yy = py + dy, zz = pz + dz;
          if (!vm.is_free(xx, yy, zz)) continue;
          Eigen::Vector3d wp = vm.int_to_float(xx, yy, zz);
          double dd = (wp - point).norm();
          if (dd < best_d) {
            best_d = dd;
            best = wp;
            have_best = true;
          }
        }
      }
    }
    if (have_best) return best;
    radius_m += 0.5;
  }
  return point;  // point.copy() in Python
}

// ------------------------------------------------------------------
// is_blocked(p1, p2, val=VAL_OCC): exact voxel traversal (Amanatides-Woo DDA).
// True if the straight segment p1->p2 passes through any cell with cmap >= val.
// Includes the tied-axis edge/corner-graze probing exactly as the Python code.
// 中文:沿直线一格一格走(DDA),判断线段有没有穿过任何「值 >= val 的格」。
// ------------------------------------------------------------------
inline bool is_blocked(const VoxelMapUtil& vm, const Eigen::Vector3d& p1,
                       const Eigen::Vector3d& p2, int val = VAL_OCC) {
  const double res = vm.res;
  Eigen::Vector3d a, b;
  for (int i = 0; i < 3; ++i) {
    a(i) = (p1(i) - vm.origin(i)) / res;
    b(i) = (p2(i) - vm.origin(i)) / res;
  }
  Eigen::Array<long, 3, 1> cur, end;
  for (int i = 0; i < 3; ++i) {
    cur(i) = static_cast<long>(std::floor(a(i)));
    end(i) = static_cast<long>(std::floor(b(i)));
  }
  Eigen::Vector3d d = b - a;
  const double INF = std::numeric_limits<double>::infinity();
  double tmax[3] = {INF, INF, INF};
  double tdelta[3] = {INF, INF, INF};
  long step[3] = {1, 1, 1};
  for (int i = 0; i < 3; ++i) {
    if (d(i) > 0) {
      step[i] = 1;
      tmax[i] = (static_cast<double>(cur(i)) + 1.0 - a(i)) / d(i);
      tdelta[i] = 1.0 / d(i);
    } else if (d(i) < 0) {
      step[i] = -1;
      tmax[i] = (static_cast<double>(cur(i)) - a(i)) / d(i);
      tdelta[i] = -1.0 / d(i);
    }
    // d(i) == 0 -> tmax/tdelta stay +inf, step stays 1 (matches Python np.ones init)
  }

  auto blocked = [&](const Eigen::Array<long, 3, 1>& c) -> bool {
    long x = c(0), y = c(1), z = c(2);
    if (!vm.in_bounds(x, y, z)) return false;
    return vm.cmap[vm.lin_index(x, y, z)] >= val;
  };

  if (blocked(cur)) return true;
  const double eps = 1e-9;
  // iteration bound: |end-cur|_1 + 4  (Python int(np.abs(end-cur).sum()) + 4)
  long iter_bound = 0;
  for (int i = 0; i < 3; ++i) iter_bound += std::labs(end(i) - cur(i));
  iter_bound += 4;
  for (long it = 0; it < iter_bound; ++it) {
    if (cur(0) == end(0) && cur(1) == end(1) && cur(2) == end(2)) break;
    double tmin = std::min(tmax[0], std::min(tmax[1], tmax[2]));
    // tied: axes whose tmax <= tmin+eps AND tdelta is finite
    bool tied[3] = {false, false, false};
    int ntied = 0;
    for (int i = 0; i < 3; ++i) {
      if (tmax[i] <= tmin + eps && std::isfinite(tdelta[i])) {
        tied[i] = true;
        ++ntied;
      }
    }
    if (ntied >= 2) {
      // edge/corner crossing: probe stepping each tied axis alone
      for (int i = 0; i < 3; ++i) {
        if (!tied[i]) continue;
        Eigen::Array<long, 3, 1> probe = cur;
        probe(i) += step[i];
        if (blocked(probe)) return true;
      }
    }
    for (int i = 0; i < 3; ++i) {
      if (!tied[i]) continue;
      cur(i) += step[i];
      tmax[i] += tdelta[i];
    }
    if (blocked(cur)) return true;
  }
  return false;
}

// ------------------------------------------------------------------
// line_of_sight_capsule(a, b, inflate_radius_cells): exact center-line DDA check,
// then the same check on a ball of parallel offset lines within `radius` cells
// (Euclidean ball ix*ix+iy*iy+iz*iz <= r^2). Returns True iff all are clear.
// 中文:判断 a->b 这条「带半径的胶囊」是否全程畅通(中心线 + 一圈平行偏移线)。
// ------------------------------------------------------------------
inline bool line_of_sight_capsule(const VoxelMapUtil& vm, const Eigen::Vector3d& a,
                                  const Eigen::Vector3d& b, int inflate_radius_cells) {
  if ((b - a).norm() < 1e-9) return true;
  // exact center-line check (uses VAL_OCC, matching Python self.is_blocked(a,b,VAL_OCC))
  if (is_blocked(vm, a, b, VAL_OCC)) return false;
  int radius = std::max(0, inflate_radius_cells);
  if (radius == 0) return true;
  long r2 = static_cast<long>(radius) * static_cast<long>(radius);
  for (int ix = -radius; ix <= radius; ++ix) {
    for (int iy = -radius; iy <= radius; ++iy) {
      for (int iz = -radius; iz <= radius; ++iz) {
        if ((ix == 0 && iy == 0 && iz == 0) ||
            (static_cast<long>(ix) * ix + static_cast<long>(iy) * iy +
                 static_cast<long>(iz) * iz >
             r2))
          continue;
        Eigen::Vector3d shift(ix * vm.res, iy * vm.res, iz * vm.res);
        if (is_blocked(vm, a + shift, b + shift, VAL_OCC)) return false;
      }
    }
  }
  return true;
}

// ------------------------------------------------------------------
// cells_world(value): world cell-centers of all cells whose cmap == value.
// Iteration order matches Python np.where (linear index order 0..total-1), then
// decode lin -> (x,y,z) and int_to_float. 中文:返回所有等于 value 的格中心。
// ------------------------------------------------------------------
inline std::vector<Eigen::Vector3d> cells_world(const VoxelMapUtil& vm, int value) {
  std::vector<Eigen::Vector3d> out;
  long dimX = vm.dimX, dimY = vm.dimY;
  long dimXY = dimX * dimY;
  long total = vm.total_size();
  for (long lin = 0; lin < total; ++lin) {
    if (vm.cmap[lin] == value) {
      long x = lin % dimX;
      long y = (lin / dimX) % dimY;
      long z = lin / dimXY;
      out.push_back(vm.int_to_float(x, y, z));
    }
  }
  return out;
}

// ------------------------------------------------------------------
// has_non_free_neighbor(p): among the 26 neighbors of p's cell, any cell that is
// in-bounds and not exactly VAL_FREE? 中文:点 p 所在格的 26 邻里有没有非空闲格。
// ------------------------------------------------------------------
inline bool has_non_free_neighbor(const VoxelMapUtil& vm, const Eigen::Vector3d& p) {
  Eigen::Array<long, 3, 1> pi = vm.float_to_int(p);
  long ix = pi(0), iy = pi(1), iz = pi(2);
  for (int dx = -1; dx <= 1; ++dx) {
    for (int dy = -1; dy <= 1; ++dy) {
      for (int dz = -1; dz <= 1; ++dz) {
        if (dx == 0 && dy == 0 && dz == 0) continue;
        long xx = ix + dx, yy = iy + dy, zz = iz + dz;
        if (!vm.in_bounds(xx, yy, zz)) continue;
        if (vm.cmap[vm.lin_index(xx, yy, zz)] != VAL_FREE) return true;
      }
    }
  }
  return false;
}

}  // namespace helpers
}  // namespace sando
