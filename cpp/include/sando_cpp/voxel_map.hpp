// VoxelMapUtil — faithful C++ port of sando_py/hgp/voxel_map.py:VoxelMapUtil.
// 3D voxel occupancy grid (int8 cmap: UNK=-1 / FREE=0 / OCC=100) + float heat field.
// 1:1 with the Python: same layout cmap[x + dimX*(y + dimY*z)], same float_to_int
// (floor convention) / int_to_float (cell-center) conventions, same read_map 9-step
// build, same _rasterize_aabb / _rasterize_cells, same _compose_dynamic_heat /
// _compose_static_heat formulas. 必须还原:每个公式对照 Python 原样移植;golden 逐项验证。
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>

namespace sando {

// ---- cell value codes (match Python VAL_FREE/VAL_OCC/VAL_UNK) ----
constexpr int8_t VAL_FREE = 0;
constexpr int8_t VAL_OCC = 100;
constexpr int8_t VAL_UNK = -1;

class VoxelMapUtil {
 public:
  // res = edge length of each voxel (m).
  double res = 0.3;
  Eigen::Vector3d origin = Eigen::Vector3d::Zero();
  long dimX = 0, dimY = 0, dimZ = 0;  // self.dim
  std::vector<int8_t> cmap;           // size dimX*dimY*dimZ, layout x + dimX*(y + dimY*z)
  std::vector<float> heat;            // same layout (empty when heat disabled)
  std::vector<uint8_t> dyn_occ_mask;  // bool, set during read_map (0/1)
  bool initialized = false;

  // ---- Heat / soft-cost knobs (mirrored from Parameters; Python defaults) ----
  bool use_heat_map = true;
  bool dynamic_heat_enabled = true;
  bool dynamic_as_occupied_current = true;
  bool dynamic_as_occupied_future = false;
  double heat_alpha0 = 0.2;
  double heat_alpha1 = 1.0;
  int heat_p = 2;
  int heat_q = 2;
  double heat_tau_ratio = 0.5;
  double heat_gamma = 0.0;
  double heat_Hmax = 2.0;
  double dyn_base_inflation_m = 0.1;
  double dyn_heat_tube_radius_m = 0.5;
  int heat_num_samples = 15;
  double obst_max_vel = 0.5;

  bool static_heat_enabled = true;
  double static_heat_alpha = 1.0;
  int static_heat_p = 2;
  double static_heat_Hmax = 5.0;
  double static_heat_rmax_m = 1.0;
  double static_heat_default_radius_m = 0.5;
  bool static_heat_boundary_only = true;
  bool static_heat_apply_on_unknown = true;
  bool static_heat_exclude_dynamic = true;

  bool use_soft_cost_obstacles = true;
  double obstacle_soft_cost = 5.0;

  // dyn_pred_samples[k][j] = predicted center of obstacle k at sample j (optional).
  // dyn_pred_times = explicit prediction timestamps (optional). Empty => Python None.
  std::vector<std::vector<Eigen::Vector3d>> dyn_pred_samples;
  std::vector<double> dyn_pred_times;
  bool has_dyn_pred_samples = false;  // mirrors Python `is not None`
  bool has_dyn_pred_times = false;

  explicit VoxelMapUtil(double res_ = 0.3) : res(res_) {}

  // ------------------------------------------------------------------
  // Coordinate conversions (must match Python exactly)
  // ------------------------------------------------------------------
  long total_size() const { return dimX * dimY * dimZ; }

  long lin_index(long x, long y, long z) const {
    return x + dimX * (y + dimY * z);
  }

  bool in_bounds(long x, long y, long z) const {
    return (0 <= x && x < dimX && 0 <= y && y < dimY && 0 <= z && z < dimZ);
  }

  // World point -> the cell that contains it (floor convention).
  Eigen::Array<long, 3, 1> float_to_int(const Eigen::Vector3d& pt) const {
    Eigen::Array<long, 3, 1> v;
    for (int i = 0; i < 3; ++i)
      v(i) = static_cast<long>(std::floor((pt(i) - origin(i)) / res));
    return v;
  }

  // Cell-center world position.
  Eigen::Vector3d int_to_float(const Eigen::Array<long, 3, 1>& pn) const {
    Eigen::Vector3d out;
    for (int i = 0; i < 3; ++i)
      out(i) = (static_cast<double>(pn(i)) + 0.5) * res + origin(i);
    return out;
  }
  Eigen::Vector3d int_to_float(long x, long y, long z) const {
    Eigen::Array<long, 3, 1> p;
    p << x, y, z;
    return int_to_float(p);
  }

  // ------------------------------------------------------------------
  // Queries
  // ------------------------------------------------------------------
  bool is_free(long x, long y, long z) const {
    if (!in_bounds(x, y, z)) return false;
    return cmap[lin_index(x, y, z)] <= VAL_FREE;
  }
  bool is_occupied(long x, long y, long z) const {
    if (!in_bounds(x, y, z)) return true;
    return cmap[lin_index(x, y, z)] == VAL_OCC;
  }
  bool is_unknown(long x, long y, long z) const {
    if (!in_bounds(x, y, z)) return false;
    return cmap[lin_index(x, y, z)] == VAL_UNK;
  }
  // heat accessor / getHeatWeight equivalent (Python get_heat).
  double get_heat(long x, long y, long z) const {
    if (heat.empty() || !in_bounds(x, y, z)) return 0.0;
    return static_cast<double>(heat[lin_index(x, y, z)]);
  }

  // ------------------------------------------------------------------
  // Mutation helpers
  // ------------------------------------------------------------------
  void set_free(long x, long y, long z) {
    if (in_bounds(x, y, z)) cmap[lin_index(x, y, z)] = VAL_FREE;
  }

  // ------------------------------------------------------------------
  // read_map — build the planning grid
  // ------------------------------------------------------------------
  // cloud_occ: (M,3) point cloud (each row a world point). obst_pos/obst_bbox:
  // dynamic obstacle centers / half-extents.
  void read_map(long cells_x, long cells_y, long cells_z,
                const Eigen::Vector3d& center_map,
                const std::vector<Eigen::Vector3d>& cloud_occ,
                double z_ground, double z_max, double inflation,
                const std::vector<Eigen::Vector3d>& obst_pos,
                const std::vector<Eigen::Vector3d>& obst_bbox,
                double traj_max_time) {
    const double r = res;
    // 1) Padding so the inflation kernel fits inside the map
    long pad = static_cast<long>(std::ceil(5.0 * inflation / r));
    long dX = cells_x + pad;
    long dY = cells_y + pad;
    long dZ = cells_z;

    // 2) Z clamp: keep within [z_ground, z_max]
    long halfZ = dZ / 2;  // Python integer floor division (dZ >= 0)
    long down, up;
    if ((center_map(2) - static_cast<double>(halfZ) * r) < z_ground)
      down = std::max<long>(static_cast<long>(std::floor((center_map(2) - z_ground) / r)), 0);
    else
      down = halfZ;
    if ((center_map(2) + static_cast<double>(halfZ) * r) > z_max)
      up = std::max<long>(static_cast<long>(std::floor((z_max - center_map(2)) / r)), 0);
    else
      up = halfZ;
    dZ = std::max<long>(2, down + up);

    // 3) Origin (lower-corner of cell (0,0,0))
    Eigen::Vector3d o;
    o(0) = center_map(0) - static_cast<double>(dX) * r * 0.5;
    o(1) = center_map(1) - static_cast<double>(dY) * r * 0.5;
    o(2) = center_map(2) - static_cast<double>(down) * r;
    o(2) = std::max(z_ground, std::min(o(2), z_max - static_cast<double>(dZ) * r));

    dimX = dX;
    dimY = dY;
    dimZ = dZ;
    origin = o;

    // 4) Fill with UNKNOWN
    long total = dimX * dimY * dimZ;
    cmap.assign(static_cast<size_t>(total), VAL_UNK);

    // 5) Rasterize occupancy from the point cloud with cubic inflation
    long m = static_cast<long>(std::floor(inflation / r));
    if (!cloud_occ.empty()) {
      std::vector<Eigen::Array<long, 3, 1>> idx;
      idx.reserve(cloud_occ.size());
      for (const auto& p : cloud_occ) {
        if (p(2) >= z_ground && p(2) <= z_max) {
          Eigen::Array<long, 3, 1> ci;
          for (int i = 0; i < 3; ++i)
            ci(i) = static_cast<long>(std::floor((p(i) - origin(i)) / r));
          idx.push_back(ci);
        }
      }
      if (!idx.empty()) _rasterize_cells(idx, m);
    }

    // 6) Dynamic obstacles as occupied (current)
    dyn_occ_mask.assign(static_cast<size_t>(total), 0);
    if (dynamic_as_occupied_current) {
      size_t n = std::min(obst_pos.size(), obst_bbox.size());
      for (size_t k = 0; k < n; ++k) {
        Eigen::Vector3d half;
        double add = std::max(dyn_base_inflation_m, inflation);
        for (int i = 0; i < 3; ++i)
          half(i) = std::max(obst_bbox[k](i) + add, 0.0);
        _rasterize_aabb(obst_pos[k], half, true);
      }
    }

    // 7) Dynamic obstacles as occupied (future cone)
    if (dynamic_as_occupied_future && traj_max_time > 0) {
      size_t n = std::min(obst_pos.size(), obst_bbox.size());
      for (size_t k = 0; k < n; ++k) {
        Eigen::Vector3d half;
        for (int i = 0; i < 3; ++i)
          half(i) = obst_bbox[k](i) + obst_max_vel * traj_max_time;
        _rasterize_aabb(obst_pos[k], half, true);
      }
    }

    // 8) y-boundary walls (only in y)
    long wall_cells = static_cast<long>(std::ceil(inflation / r));
    for (long z = 0; z < dimZ; ++z) {
      long base = z * dimX * dimY;
      for (long j = 0; j < std::min(wall_cells, dimY); ++j) {
        long row_start = base + j * dimX;
        for (long i = 0; i < dimX; ++i) cmap[row_start + i] = VAL_OCC;
      }
      for (long j = std::max<long>(0, dimY - wall_cells); j < dimY; ++j) {
        long row_start = base + j * dimX;
        for (long i = 0; i < dimX; ++i) cmap[row_start + i] = VAL_OCC;
      }
    }

    // 9) Heat map composition
    if (dynamic_heat_enabled || static_heat_enabled)
      heat.assign(static_cast<size_t>(total), 0.0f);
    else
      heat.clear();
    if (dynamic_heat_enabled) _compose_dynamic_heat(obst_pos, obst_bbox, traj_max_time);
    if (static_heat_enabled) _compose_static_heat();

    initialized = true;
  }

  // ------------------------------------------------------------------
  // Helpers used by read_map
  // ------------------------------------------------------------------
  void _rasterize_cells(const std::vector<Eigen::Array<long, 3, 1>>& idx, long infl) {
    for (const auto& c : idx) {
      long ix = c(0), iy = c(1), iz = c(2);
      if (!(0 <= ix && ix < dimX && 0 <= iy && iy < dimY && 0 <= iz && iz < dimZ))
        continue;
      for (long dx = -infl; dx <= infl; ++dx) {
        long xx = ix + dx;
        if (!(0 <= xx && xx < dimX)) continue;
        for (long dy = -infl; dy <= infl; ++dy) {
          long yy = iy + dy;
          if (!(0 <= yy && yy < dimY)) continue;
          for (long dz = -infl; dz <= infl; ++dz) {
            long zz = iz + dz;
            if (!(0 <= zz && zz < dimZ)) continue;
            cmap[xx + dimX * (yy + dimY * zz)] = VAL_OCC;
          }
        }
      }
    }
  }

  void _rasterize_aabb(const Eigen::Vector3d& center, const Eigen::Vector3d& half,
                       bool mark_dyn) {
    const double r = res;
    Eigen::Array<long, 3, 1> lo = float_to_int(center - half);
    Eigen::Array<long, 3, 1> hi = float_to_int(center + half);
    long z0 = std::max<long>(lo(2), 0), z1 = std::min<long>(hi(2), dimZ - 1);
    long y0 = std::max<long>(lo(1), 0), y1 = std::min<long>(hi(1), dimY - 1);
    long x0 = std::max<long>(lo(0), 0), x1 = std::min<long>(hi(0), dimX - 1);
    for (long iz = z0; iz <= z1; ++iz) {
      for (long iy = y0; iy <= y1; ++iy) {
        for (long ix = x0; ix <= x1; ++ix) {
          Eigen::Vector3d cc = int_to_float(ix, iy, iz);
          bool inside = true;
          for (int i = 0; i < 3; ++i)
            if (!(std::abs(cc(i) - center(i)) <= half(i) + r * 0.5)) { inside = false; break; }
          if (inside) {
            long lin = lin_index(ix, iy, iz);
            cmap[lin] = VAL_OCC;
            if (mark_dyn) dyn_occ_mask[lin] = 1;
          }
        }
      }
    }
  }

  void _compose_dynamic_heat(const std::vector<Eigen::Vector3d>& obst_pos,
                             const std::vector<Eigen::Vector3d>& obst_bbox,
                             double traj_max_time) {
    if (obst_pos.empty()) return;
    double Th = std::max(0.0, traj_max_time);
    long Msamp = std::max<long>(2, heat_num_samples);
    double tau_w = std::max(1e-3, heat_tau_ratio * std::max(1e-3, Th));
    // times: explicit dyn_pred_times if available with >=2 entries, else linspace(0,Th,M)
    std::vector<double> times;
    if (has_dyn_pred_times && dyn_pred_times.size() >= 2) {
      times = dyn_pred_times;
    } else {
      times.resize(static_cast<size_t>(Msamp));
      if (Msamp == 1) {
        times[0] = 0.0;
      } else {
        for (long i = 0; i < Msamp; ++i)
          times[static_cast<size_t>(i)] =
              Th * static_cast<double>(i) / static_cast<double>(Msamp - 1);
      }
    }
    size_t Nt = times.size();
    std::vector<double> weights(Nt), Rs(Nt);
    double R0 = dyn_heat_tube_radius_m;
    for (size_t j = 0; j < Nt; ++j) {
      weights[j] = std::exp(-times[j] / tau_w);
      Rs[j] = R0 + heat_gamma * times[j];
    }
    int p = heat_p;
    int q = heat_q;

    size_t nobst = std::min(obst_pos.size(), obst_bbox.size());
    for (size_t k = 0; k < nobst; ++k) {
      Eigen::Vector3d ck = obst_pos[k];
      Eigen::Vector3d hk = obst_bbox[k];
      double hk_max = hk.maxCoeff();
      double Rreach = hk_max + obst_max_vel * Th;
      if (Rreach <= 0) continue;

      // AABB around the obstacle: extent = Rreach + max(R0, max(hk)) + obst_max_vel*Th
      double ext = Rreach + std::max(R0, hk_max) + obst_max_vel * Th;
      Eigen::Vector3d extent(ext, ext, ext);
      Eigen::Array<long, 3, 1> lo = float_to_int(ck - extent);
      Eigen::Array<long, 3, 1> hi = float_to_int(ck + extent);
      long x0 = std::max<long>(lo(0), 0);
      long y0 = std::max<long>(lo(1), 0);
      long z0 = std::max<long>(lo(2), 0);
      long x1 = std::min<long>(hi(0), dimX - 1);
      long y1 = std::min<long>(hi(1), dimY - 1);
      long z1 = std::min<long>(hi(2), dimZ - 1);
      if (x1 < x0 || y1 < y0 || z1 < z0) continue;

      const std::vector<Eigen::Vector3d>* samples_k = nullptr;
      if (has_dyn_pred_samples && k < dyn_pred_samples.size())
        samples_k = &dyn_pred_samples[k];

      for (long ix = x0; ix <= x1; ++ix) {
        double CX = (static_cast<double>(ix) + 0.5) * res + origin(0);
        for (long iy = y0; iy <= y1; ++iy) {
          double CY = (static_cast<double>(iy) + 0.5) * res + origin(1);
          for (long iz = z0; iz <= z1; ++iz) {
            double CZ = (static_cast<double>(iz) + 0.5) * res + origin(2);
            // base reachability blob: distance from cell to obstacle AABB
            double dx = std::max(std::abs(CX - ck(0)) - hk(0), 0.0);
            double dy = std::max(std::abs(CY - ck(1)) - hk(1), 0.0);
            double dz = std::max(std::abs(CZ - ck(2)) - hk(2), 0.0);
            double d_box = std::sqrt(dx * dx + dy * dy + dz * dz);
            double Hbase = 0.0;
            if (d_box <= Rreach)
              Hbase = heat_alpha0 *
                      std::pow(1.0 - std::min(d_box / Rreach, 1.0), static_cast<double>(p));
            // trajectory tube: max over predicted time samples
            double tube_max = 0.0;
            for (size_t j = 0; j < Nt; ++j) {
              Eigen::Vector3d cj = ck;
              if (samples_k != nullptr && j < samples_k->size()) cj = (*samples_k)[j];
              double ex = std::max(std::abs(CX - cj(0)) - hk(0), 0.0);
              double ey = std::max(std::abs(CY - cj(1)) - hk(1), 0.0);
              double ez = std::max(std::abs(CZ - cj(2)) - hk(2), 0.0);
              double d_j = std::sqrt(ex * ex + ey * ey + ez * ez);
              double contrib = 0.0;
              double Rj = Rs[j];
              if (d_j <= Rj)
                contrib = weights[j] *
                          std::pow(1.0 - std::min(d_j / Rj, 1.0), static_cast<double>(q));
              if (contrib > tube_max) tube_max = contrib;
            }
            double Hk = Hbase + heat_alpha1 * tube_max;
            if (heat_Hmax > 0) Hk = std::min(Hk, heat_Hmax);
            long lin = ix + dimX * (iy + dimY * iz);
            if (!use_soft_cost_obstacles) {
              // original skipped occupied cells (heat>=0 so max-with-0 is a no-op there)
              if (cmap[lin] > VAL_FREE) Hk = 0.0;
            }
            // np.maximum.at: scatter-max into heat
            float hf = static_cast<float>(Hk);
            if (hf > heat[lin]) heat[lin] = hf;
          }
        }
      }
    }
  }

  void _compose_static_heat() {
    if (heat.empty()) return;
    long Rcell = static_cast<long>(std::ceil(static_heat_rmax_m / res));
    if (Rcell <= 0) return;
    // Precompute offsets (dx,dy,dz) within rmax (Euclidean), with their metric distance.
    struct Off { long dx, dy, dz; double d_m; };
    std::vector<Off> offsets;
    for (long dx = -Rcell; dx <= Rcell; ++dx)
      for (long dy = -Rcell; dy <= Rcell; ++dy)
        for (long dz = -Rcell; dz <= Rcell; ++dz) {
          if (dx == 0 && dy == 0 && dz == 0) continue;
          double d_m = res * std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
          if (d_m <= static_heat_rmax_m) offsets.push_back({dx, dy, dz, d_m});
        }
    if (offsets.empty()) return;

    double Rm = static_heat_default_radius_m;
    int p = static_heat_p;
    double alpha = static_heat_alpha;
    long total = dimX * dimY * dimZ;

    // Collect seeds = occupied cells, optionally excluding dynamic-occupied cells.
    std::vector<long> seed_idxs;
    for (long i = 0; i < total; ++i) {
      if (cmap[i] == VAL_OCC) {
        if (static_heat_exclude_dynamic && !dyn_occ_mask.empty() && dyn_occ_mask[i])
          continue;
        seed_idxs.push_back(i);
      }
    }

    // boundary_only: keep seeds with at least one non-occupied 6-neighbor (or out of bounds).
    if (static_heat_boundary_only && !seed_idxs.empty()) {
      auto occ = [&](long x, long y, long z) -> bool {
        if (!(0 <= x && x < dimX && 0 <= y && y < dimY && 0 <= z && z < dimZ)) return false;
        return cmap[x + dimX * (y + dimY * z)] == VAL_OCC;
      };
      std::vector<long> kept;
      kept.reserve(seed_idxs.size());
      for (long lin : seed_idxs) {
        long x = lin % dimX;
        long y = (lin / dimX) % dimY;
        long z = lin / (dimX * dimY);
        bool interior = occ(x + 1, y, z) && occ(x - 1, y, z) &&
                        occ(x, y + 1, z) && occ(x, y - 1, z) &&
                        occ(x, y, z + 1) && occ(x, y, z - 1);
        if (!interior) kept.push_back(lin);  // boundary = occupied and not interior
      }
      seed_idxs.swap(kept);
    }
    if (seed_idxs.empty()) return;

    bool apply_on_unknown = static_heat_apply_on_unknown;
    double Hmax = static_heat_Hmax;

    // For each offset, compute one weight w (depends only on d_m), then scatter-max into heat.
    for (const Off& off : offsets) {
      if (off.d_m > Rm) continue;
      double u = off.d_m / Rm;
      double w = alpha * std::pow(1.0 - u, static_cast<double>(p));
      if (Hmax > 0) w = std::min(w, Hmax);
      float wf = static_cast<float>(w);
      for (long lin : seed_idxs) {
        long sx = lin % dimX;
        long sy = (lin / dimX) % dimY;
        long sz = lin / (dimX * dimY);
        long xx = sx + off.dx, yy = sy + off.dy, zz = sz + off.dz;
        if (!(xx >= 0 && xx < dimX && yy >= 0 && yy < dimY && zz >= 0 && zz < dimZ)) continue;
        long tlin = xx + dimX * (yy + dimY * zz);
        if (!apply_on_unknown && cmap[tlin] == VAL_UNK) continue;
        if (wf > heat[tlin]) heat[tlin] = wf;
      }
    }
  }
};

}  // namespace sando
