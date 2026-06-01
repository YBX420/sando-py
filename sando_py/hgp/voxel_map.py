"""3D voxel occupancy + heat map.

Port of include/hgp/map_util.hpp (the inline MapUtil<3> / VoxelMapUtil
implementation in the C++ source) — same layout, same value codes, same
floatToInt / intToFloat conventions, so a Python-rasterized map is
behaviorally interchangeable with the C++ one for planning purposes.

Layout:
  cmap[x + dim_x * y + dim_x * dim_y * z]  with int8 cell values
    val_free      =   0
    val_occupied  = 100
    val_unknown   =  -1
  heat[i] is float32 in the same layout (0 when heat disabled).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


VAL_FREE = np.int8(0)
VAL_OCC = np.int8(100)
VAL_UNK = np.int8(-1)


class VoxelMapUtil:
    def __init__(self, res: float = 0.3):
        self.res = float(res)
        self.dim = np.array([0, 0, 0], dtype=np.int64)
        self.origin = np.zeros(3, dtype=np.float64)
        self.cmap: np.ndarray = np.zeros(0, dtype=np.int8)
        self.heat: np.ndarray = np.zeros(0, dtype=np.float32)
        self.dyn_occ_mask: Optional[np.ndarray] = None  # bool, set during readMap

        # Heat / soft-cost knobs (mirrored from Parameters; set via setHeatParams)
        self.use_heat_map: bool = True
        self.dynamic_heat_enabled: bool = True
        self.dynamic_as_occupied_current: bool = True
        self.dynamic_as_occupied_future: bool = False
        self.heat_alpha0: float = 0.2
        self.heat_alpha1: float = 1.0
        self.heat_p: int = 2
        self.heat_q: int = 2
        self.heat_tau_ratio: float = 0.5
        self.heat_gamma: float = 0.0
        self.heat_Hmax: float = 2.0
        self.dyn_base_inflation_m: float = 0.1
        self.dyn_heat_tube_radius_m: float = 0.5
        self.heat_num_samples: int = 15
        self.obst_max_vel: float = 0.5

        self.static_heat_enabled: bool = True
        self.static_heat_alpha: float = 1.0
        self.static_heat_p: int = 2
        self.static_heat_Hmax: float = 5.0
        self.static_heat_rmax_m: float = 1.0
        self.static_heat_default_radius_m: float = 0.5
        self.static_heat_boundary_only: bool = True
        # True: the Python map never carves free space (interior stays UNKNOWN),
        # so static heat must be allowed onto UNKNOWN cells or it never appears.
        self.static_heat_apply_on_unknown: bool = True
        self.static_heat_exclude_dynamic: bool = True

        self.use_soft_cost_obstacles: bool = True
        self.obstacle_soft_cost: float = 5.0

        self.dyn_pred_samples: Optional[List[List[np.ndarray]]] = None
        self.dyn_pred_times: Optional[np.ndarray] = None

        self.initialized: bool = False

    # ------------------------------------------------------------------
    # Coordinate conversions (must match C++ MapUtil exactly)
    # ------------------------------------------------------------------
    def total_size(self) -> int:
        return int(self.dim[0] * self.dim[1] * self.dim[2])

    def lin_index(self, x: int, y: int, z: int) -> int:
        return int(x + self.dim[0] * (y + self.dim[1] * z))

    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return (0 <= x < self.dim[0] and 0 <= y < self.dim[1] and 0 <= z < self.dim[2])

    def float_to_int(self, pt: np.ndarray) -> np.ndarray:
        """World point -> the cell that contains it (floor convention).

        Unified convention: write (point-cloud + AABB rasterization) and read
        (queries) all map a point to the cell that geometrically contains it.
        intToFloat returns that cell's center, so write and read always agree.
        (Deliberately diverges from C++ MapUtil's -0.5 truncate, which left
        rasterize/query ~1 cell apart.)
        """
        v = np.floor((np.asarray(pt, dtype=np.float64) - self.origin) / self.res)
        return v.astype(np.int64)

    def int_to_float(self, pn: np.ndarray) -> np.ndarray:
        """Cell-center world position."""
        return (np.asarray(pn, dtype=np.float64) + 0.5) * self.res + self.origin

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    def is_free(self, x: int, y: int, z: int) -> bool:
        if not self.in_bounds(x, y, z):
            return False
        return self.cmap[self.lin_index(x, y, z)] <= VAL_FREE

    def is_occupied(self, x: int, y: int, z: int) -> bool:
        if not self.in_bounds(x, y, z):
            return True
        return self.cmap[self.lin_index(x, y, z)] == VAL_OCC

    def is_unknown(self, x: int, y: int, z: int) -> bool:
        if not self.in_bounds(x, y, z):
            return False
        return self.cmap[self.lin_index(x, y, z)] == VAL_UNK

    def get_heat(self, x: int, y: int, z: int) -> float:
        if self.heat.size == 0 or not self.in_bounds(x, y, z):
            return 0.0
        return float(self.heat[self.lin_index(x, y, z)])

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    def set_free(self, x: int, y: int, z: int) -> None:
        if self.in_bounds(x, y, z):
            self.cmap[self.lin_index(x, y, z)] = VAL_FREE

    def set_free_voxel_and_surroundings(self, center: np.ndarray, d: float) -> None:
        n = int(round(d / self.res + 0.5))
        cx, cy, cz = self.float_to_int(center)
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                for dz in range(-n, n + 1):
                    self.set_free(int(cx + dx), int(cy + dy), int(cz + dz))

    def find_closest_free_point(self, point: np.ndarray) -> np.ndarray:
        """Return cell-center of nearest free voxel; mirrors C++ findClosestFreePoint."""
        px, py, pz = self.float_to_int(point)
        if self.is_free(int(px), int(py), int(pz)):
            return self.int_to_float(np.array([px, py, pz]))
        best = None
        best_d = float("inf")
        radius_m = 1.0
        while radius_m <= 5.0:
            r = int(radius_m / self.res)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        xx = int(px + dx)
                        yy = int(py + dy)
                        zz = int(pz + dz)
                        if not self.is_free(xx, yy, zz):
                            continue
                        wp = self.int_to_float(np.array([xx, yy, zz]))
                        d = float(np.linalg.norm(wp - point))
                        if d < best_d:
                            best_d = d
                            best = wp
            if best is not None:
                return best
            radius_m += 0.5
        return point.copy()

    # ------------------------------------------------------------------
    # readMap — build the planning grid from sensor pointclouds + dynamic obstacles
    # ------------------------------------------------------------------
    def read_map(
        self,
        cells_x: int,
        cells_y: int,
        cells_z: int,
        center_map: np.ndarray,
        cloud_occ: np.ndarray,           # (M,3)
        z_ground: float,
        z_max: float,
        inflation: float,
        obst_pos: List[np.ndarray],
        obst_bbox: List[np.ndarray],
        traj_max_time: float,
    ) -> None:
        res = self.res
        # 1) Padding so the inflation kernel fits inside the map
        pad = int(np.ceil(5.0 * inflation / res))
        dimX = int(cells_x + pad)
        dimY = int(cells_y + pad)
        dimZ = int(cells_z)

        # 2) Z clamp: keep within [z_ground, z_max]
        halfZ = dimZ // 2
        if (center_map[2] - halfZ * res) < z_ground:
            down = max(int(np.floor((center_map[2] - z_ground) / res)), 0)
        else:
            down = halfZ
        if (center_map[2] + halfZ * res) > z_max:
            up = max(int(np.floor((z_max - center_map[2]) / res)), 0)
        else:
            up = halfZ
        dimZ = max(2, down + up)

        # 3) Origin (lower-corner of cell (0,0,0))
        origin = np.array([
            center_map[0] - dimX * res * 0.5,
            center_map[1] - dimY * res * 0.5,
            center_map[2] - down * res,
        ], dtype=np.float64)
        origin[2] = max(z_ground, min(origin[2], z_max - dimZ * res))

        self.dim = np.array([dimX, dimY, dimZ], dtype=np.int64)
        self.origin = origin

        # 4) Fill with UNKNOWN
        total = dimX * dimY * dimZ
        self.cmap = np.full(total, VAL_UNK, dtype=np.int8)

        # 5) Rasterize occupancy from the point cloud with cubic inflation
        m = int(np.floor(inflation / res))
        if cloud_occ is not None and len(cloud_occ) > 0:
            pts = np.asarray(cloud_occ, dtype=np.float64)
            mask = (pts[:, 2] >= z_ground) & (pts[:, 2] <= z_max)
            pts = pts[mask]
            if len(pts):
                idx = np.floor((pts - origin) / res).astype(np.int64)
                self._rasterize_cells(idx, m)

        # 6) Dynamic obstacles as occupied (current)
        self.dyn_occ_mask = np.zeros(total, dtype=bool)
        if self.dynamic_as_occupied_current:
            for c, hk in zip(obst_pos, obst_bbox):
                half = np.maximum(hk + max(self.dyn_base_inflation_m, inflation), 0.0)
                self._rasterize_aabb(np.asarray(c, dtype=np.float64), half, mark_dyn=True)

        # 7) Dynamic obstacles as occupied (future cone)
        if self.dynamic_as_occupied_future and traj_max_time > 0:
            for c, hk in zip(obst_pos, obst_bbox):
                half = hk + self.obst_max_vel * traj_max_time
                self._rasterize_aabb(np.asarray(c, dtype=np.float64), half, mark_dyn=True)

        # 8) y-boundary walls (only in y per C++ readMap step 8b)
        y_min_world = origin[1]
        y_max_world = origin[1] + dimY * res
        wall_cells = int(np.ceil(inflation / res))
        for j in range(min(wall_cells, dimY)):
            self.cmap[j * dimX:(j + 1) * dimX].reshape(-1)[:] = self.cmap[j * dimX:(j + 1) * dimX]  # noop, kept for clarity
        # easier: flatten per-z slab
        for z in range(dimZ):
            base = z * dimX * dimY
            for j in range(min(wall_cells, dimY)):
                row_start = base + j * dimX
                self.cmap[row_start:row_start + dimX] = VAL_OCC
            for j in range(max(0, dimY - wall_cells), dimY):
                row_start = base + j * dimX
                self.cmap[row_start:row_start + dimX] = VAL_OCC

        # 9) Heat map composition
        self.heat = np.zeros(total, dtype=np.float32) if (self.dynamic_heat_enabled or self.static_heat_enabled) else np.zeros(0, dtype=np.float32)
        if self.dynamic_heat_enabled:
            self._compose_dynamic_heat(obst_pos, obst_bbox, traj_max_time)
        if self.static_heat_enabled:
            self._compose_static_heat()

        self.initialized = True

    # ------------------------------------------------------------------
    # Helpers used by read_map
    # ------------------------------------------------------------------
    def _rasterize_cells(self, idx: np.ndarray, infl: int) -> None:
        """Mark each cell index (and a cubic inflation around it) as occupied."""
        dimX, dimY, dimZ = self.dim
        for ix, iy, iz in idx:
            if not (0 <= ix < dimX and 0 <= iy < dimY and 0 <= iz < dimZ):
                continue
            for dx in range(-infl, infl + 1):
                xx = ix + dx
                if not (0 <= xx < dimX):
                    continue
                for dy in range(-infl, infl + 1):
                    yy = iy + dy
                    if not (0 <= yy < dimY):
                        continue
                    for dz in range(-infl, infl + 1):
                        zz = iz + dz
                        if not (0 <= zz < dimZ):
                            continue
                        self.cmap[xx + dimX * (yy + dimY * zz)] = VAL_OCC

    def _rasterize_aabb(self, center: np.ndarray, half: np.ndarray, mark_dyn: bool) -> None:
        dimX, dimY, dimZ = self.dim
        res = self.res
        lo = self.float_to_int(center - half)
        hi = self.float_to_int(center + half)
        for iz in range(int(max(lo[2], 0)), int(min(hi[2], dimZ - 1)) + 1):
            for iy in range(int(max(lo[1], 0)), int(min(hi[1], dimY - 1)) + 1):
                for ix in range(int(max(lo[0], 0)), int(min(hi[0], dimX - 1)) + 1):
                    cell_center = self.int_to_float(np.array([ix, iy, iz]))
                    if np.all(np.abs(cell_center - center) <= half + res * 0.5):
                        lin = self.lin_index(ix, iy, iz)
                        self.cmap[lin] = VAL_OCC
                        if mark_dyn:
                            self.dyn_occ_mask[lin] = True

    def _compose_dynamic_heat(self, obst_pos, obst_bbox, traj_max_time):
        if not obst_pos:
            return
        Th = max(0.0, float(traj_max_time))
        M = max(2, self.heat_num_samples)
        tau_w = max(1e-3, self.heat_tau_ratio * max(1e-3, Th))
        times = (self.dyn_pred_times
                 if self.dyn_pred_times is not None and len(self.dyn_pred_times) >= 2
                 else np.linspace(0.0, Th, M))
        weights = np.exp(-times / tau_w)
        R0 = self.dyn_heat_tube_radius_m
        Rs = R0 + self.heat_gamma * times
        p = self.heat_p
        q = self.heat_q

        dimX, dimY, dimZ = self.dim

        for k, (ck, hk) in enumerate(zip(obst_pos, obst_bbox)):
            ck = np.asarray(ck, dtype=np.float64)
            hk = np.asarray(hk, dtype=np.float64)
            Rreach = float(np.max(hk)) + self.obst_max_vel * Th
            if Rreach <= 0:
                continue

            # Iterate cells within a generous AABB around the obstacle
            extent = np.array([Rreach, Rreach, Rreach]) + max(R0, np.max(hk)) + self.obst_max_vel * Th
            lo = self.float_to_int(ck - extent)
            hi = self.float_to_int(ck + extent)
            x0, y0, z0 = (int(max(v, 0)) for v in lo)
            x1, y1, z1 = (int(min(v, d - 1)) for v, d in zip(hi, (dimX, dimY, dimZ)))

            samples_k = self.dyn_pred_samples[k] if (self.dyn_pred_samples is not None and k < len(self.dyn_pred_samples)) else None

            for iz in range(z0, z1 + 1):
                for iy in range(y0, y1 + 1):
                    for ix in range(x0, x1 + 1):
                        lin = ix + dimX * (iy + dimY * iz)
                        if self.cmap[lin] > VAL_FREE and not self.use_soft_cost_obstacles:
                            continue
                        cell = self.int_to_float(np.array([ix, iy, iz]))
                        # Distance to AABB
                        diff = np.maximum(np.abs(cell - ck) - hk, 0.0)
                        d_box = float(np.linalg.norm(diff))
                        Hbase = 0.0
                        if d_box <= Rreach:
                            u = min(d_box / Rreach, 1.0)
                            Hbase = self.heat_alpha0 * (1.0 - u) ** p
                        tube_max = 0.0
                        for j, (tj, Rj, wj) in enumerate(zip(times, Rs, weights)):
                            cj = samples_k[j] if (samples_k is not None and j < len(samples_k)) else ck
                            diff_j = np.maximum(np.abs(cell - cj) - hk, 0.0)
                            d_j = float(np.linalg.norm(diff_j))
                            if d_j <= Rj:
                                uj = min(d_j / Rj, 1.0)
                                tube_max = max(tube_max, wj * (1.0 - uj) ** q)
                        Hk = Hbase + self.heat_alpha1 * tube_max
                        if self.heat_Hmax > 0:
                            Hk = min(Hk, self.heat_Hmax)
                        if Hk > self.heat[lin]:
                            self.heat[lin] = Hk

    def _compose_static_heat(self):
        if self.heat.size == 0:
            return
        Rcell = int(np.ceil(self.static_heat_rmax_m / self.res))
        if Rcell <= 0:
            return
        offsets = []
        for dx in range(-Rcell, Rcell + 1):
            for dy in range(-Rcell, Rcell + 1):
                for dz in range(-Rcell, Rcell + 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    d_m = self.res * np.sqrt(dx * dx + dy * dy + dz * dz)
                    if d_m <= self.static_heat_rmax_m:
                        offsets.append((dx, dy, dz, d_m))
        if not offsets:
            return

        dimX, dimY, dimZ = self.dim
        Rm = self.static_heat_default_radius_m
        p = self.static_heat_p
        alpha = self.static_heat_alpha

        # Collect seeds
        seed_idxs = np.where(self.cmap == VAL_OCC)[0]
        if self.static_heat_exclude_dynamic and self.dyn_occ_mask is not None:
            seed_idxs = seed_idxs[~self.dyn_occ_mask[seed_idxs]]

        # boundary_only: keep seeds with at least one non-occupied 6-neighbor
        if self.static_heat_boundary_only and len(seed_idxs) > 0:
            keep = []
            for lin in seed_idxs:
                ix = int(lin % dimX)
                iy = int((lin // dimX) % dimY)
                iz = int(lin // (dimX * dimY))
                boundary = False
                for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                    xx, yy, zz = ix + dx, iy + dy, iz + dz
                    if not (0 <= xx < dimX and 0 <= yy < dimY and 0 <= zz < dimZ):
                        boundary = True
                        break
                    if self.cmap[xx + dimX * (yy + dimY * zz)] != VAL_OCC:
                        boundary = True
                        break
                if boundary:
                    keep.append(lin)
            seed_idxs = np.array(keep, dtype=np.int64) if keep else np.zeros(0, dtype=np.int64)

        for lin in seed_idxs:
            ix = int(lin % dimX)
            iy = int((lin // dimX) % dimY)
            iz = int(lin // (dimX * dimY))
            for dx, dy, dz, d_m in offsets:
                if d_m > Rm:
                    continue
                xx, yy, zz = ix + dx, iy + dy, iz + dz
                if not (0 <= xx < dimX and 0 <= yy < dimY and 0 <= zz < dimZ):
                    continue
                tlin = xx + dimX * (yy + dimY * zz)
                if not self.static_heat_apply_on_unknown and self.cmap[tlin] == VAL_UNK:
                    continue
                u = d_m / Rm
                w = alpha * (1.0 - u) ** p
                if self.static_heat_Hmax > 0:
                    w = min(w, self.static_heat_Hmax)
                if w > self.heat[tlin]:
                    self.heat[tlin] = w

    # ------------------------------------------------------------------
    # Ray-trace / line-of-sight
    # ------------------------------------------------------------------
    def is_blocked(self, p1: np.ndarray, p2: np.ndarray, val: int = 100) -> bool:
        """Exact voxel traversal (Amanatides-Woo DDA): True if the straight
        segment p1->p2 passes through any cell with cmap >= val. Unlike point
        sampling, this never skips a cell the segment actually crosses.
        """
        res = self.res
        a = (np.asarray(p1, dtype=np.float64) - self.origin) / res
        b = (np.asarray(p2, dtype=np.float64) - self.origin) / res
        cur = np.floor(a).astype(np.int64)
        end = np.floor(b).astype(np.int64)
        d = b - a
        tmax = np.array([np.inf, np.inf, np.inf])
        tdelta = np.array([np.inf, np.inf, np.inf])
        step = np.ones(3, dtype=np.int64)
        for i in range(3):
            if d[i] > 0:
                step[i] = 1
                tmax[i] = (cur[i] + 1 - a[i]) / d[i]
                tdelta[i] = 1.0 / d[i]
            elif d[i] < 0:
                step[i] = -1
                tmax[i] = (cur[i] - a[i]) / d[i]
                tdelta[i] = -1.0 / d[i]

        def blocked(c) -> bool:
            if not self.in_bounds(int(c[0]), int(c[1]), int(c[2])):
                return False
            return self.cmap[self.lin_index(int(c[0]), int(c[1]), int(c[2]))] >= val

        if blocked(cur):
            return True
        eps = 1e-9
        for _ in range(int(np.abs(end - cur).sum()) + 4):
            if cur[0] == end[0] and cur[1] == end[1] and cur[2] == end[2]:
                break
            tmin = float(tmax.min())
            tied = [i for i in range(3) if tmax[i] <= tmin + eps and np.isfinite(tdelta[i])]
            if len(tied) >= 2:
                # Edge/corner crossing: the segment grazes the cells reachable by
                # stepping each tied axis alone — check them so a corner graze
                # through an obstacle is not missed.
                for i in tied:
                    probe = cur.copy()
                    probe[i] += step[i]
                    if blocked(probe):
                        return True
            for i in tied:
                cur[i] += step[i]
                tmax[i] += tdelta[i]
            if blocked(cur):
                return True
        return False

    def line_of_sight_capsule(self, a: np.ndarray, b: np.ndarray, inflate_radius_cells: int) -> bool:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if np.linalg.norm(b - a) < 1e-9:
            return True
        # Exact center-line check (DDA) — never skips a grazed cell.
        if self.is_blocked(a, b, VAL_OCC):
            return False
        radius = max(0, int(inflate_radius_cells))
        if radius == 0:
            return True
        # Clearance margin: same exact check on parallel offset lines.
        r2 = radius * radius
        for ix in range(-radius, radius + 1):
            for iy in range(-radius, radius + 1):
                for iz in range(-radius, radius + 1):
                    if (ix == 0 and iy == 0 and iz == 0) or (ix * ix + iy * iy + iz * iz) > r2:
                        continue
                    shift = np.array([ix * self.res, iy * self.res, iz * self.res])
                    if self.is_blocked(a + shift, b + shift, VAL_OCC):
                        return False
        return True

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------
    def cells_world(self, value: int) -> List[np.ndarray]:
        idxs = np.where(self.cmap == value)[0]
        if len(idxs) == 0:
            return []
        dimX = int(self.dim[0])
        dimXY = dimX * int(self.dim[1])
        xs = idxs % dimX
        ys = (idxs // dimX) % int(self.dim[1])
        zs = idxs // dimXY
        ijk = np.stack([xs, ys, zs], axis=1)
        return [self.int_to_float(p) for p in ijk]

    def has_non_free_neighbor(self, p: np.ndarray) -> bool:
        ix, iy, iz = self.float_to_int(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    xx, yy, zz = int(ix + dx), int(iy + dy), int(iz + dz)
                    if not self.in_bounds(xx, yy, zz):
                        continue
                    if self.cmap[self.lin_index(xx, yy, zz)] != VAL_FREE:
                        return True
        return False
