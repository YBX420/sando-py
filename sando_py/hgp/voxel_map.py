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

# ===== 中文说明 =====
# 这个文件是整个规划器的「地图」。它把世界切成一格一格的小立方体(体素),
# 每一格记两样东西:
#   1) 占据状态 cmap:这格是空的(0)、被障碍占了(100)、还是没看见过/不确定(-1)。
#   2) 热度 heat:一个越靠近障碍越大的「软成本」(浮点数),给后面的 A* 当
#      「这里不太想走」的惩罚分,而不是硬挡住。
# 它在规划器里的角色:全局向导 heat-A* 跑搜索时,就是查这张表来判断能不能走、
#   走这格要扣多少分。
# 主要输入:传感器点云(静态障碍)、动态障碍的位置/尺寸(以及预测轨迹)。
# 主要输出:cmap(占据栅格)、heat(热度场),供 graph_search 查询。
# 关键约定:世界坐标 <-> 整数格坐标 的换算(float_to_int / int_to_float),
#   写地图和读地图必须用同一套换算,否则会「画在这格、查到那格」对不上。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np


# 三种格子状态的数值编码:空闲=0、被占=100、未知=-1(和 C++ 完全一致)
VAL_FREE = np.int8(0)
VAL_OCC = np.int8(100)
VAL_UNK = np.int8(-1)


class VoxelMapUtil:
    # 体素地图本体。res = 每个格子的边长(米)。其余字段大多是「热度场」的调参旋钮,
    # 由 HGPManager.set_parameters 从 yaml 配置同步进来。
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
    # 整张地图一共多少格(x*y*z)
    def total_size(self) -> int:
        return int(self.dim[0] * self.dim[1] * self.dim[2])

    # 把 (x,y,z) 三维格坐标压成一维数组下标(cmap/heat 都是一维平铺存的)
    def lin_index(self, x: int, y: int, z: int) -> int:
        return int(x + self.dim[0] * (y + self.dim[1] * z))

    # 这个格坐标是否落在地图范围内
    def in_bounds(self, x: int, y: int, z: int) -> bool:
        return (0 <= x < self.dim[0] and 0 <= y < self.dim[1] and 0 <= z < self.dim[2])

    def float_to_int(self, pt: np.ndarray) -> np.ndarray:
        """World point -> the cell that contains it (floor convention).

        Unified convention: write (point-cloud + AABB rasterization) and read
        (queries) all map a point to the cell that geometrically contains it.
        intToFloat returns that cell's center, so write and read always agree.
        (Deliberately diverges from C++ MapUtil's -0.5 truncate, which left
        rasterize/query ~1 cell apart.)

        中文:世界坐标(米)-> 包含它的那个格子的整数坐标。用 floor(向下取整)。
        关键:写地图(画障碍)和读地图(查询)都用这同一套换算,所以「画哪、查哪」
        永远对得上。这里故意没照搬 C++ 的 -0.5 截断写法 —— 那个会让画和查差约一格。
        """
        v = np.floor((np.asarray(pt, dtype=np.float64) - self.origin) / self.res)
        return v.astype(np.int64)

    def int_to_float(self, pn: np.ndarray) -> np.ndarray:
        """Cell-center world position.

        中文:反过来 —— 给格坐标,返回这格「中心点」的世界坐标(米)。
        """
        return (np.asarray(pn, dtype=np.float64) + 0.5) * self.res + self.origin

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    # 这格是否空闲(<=0 算空闲)。注意:出界一律当「不空闲」,保守。
    def is_free(self, x: int, y: int, z: int) -> bool:
        if not self.in_bounds(x, y, z):
            return False
        return self.cmap[self.lin_index(x, y, z)] <= VAL_FREE

    # 这格是否被占。出界一律当「被占」,保守(宁可绕,不要撞)。
    def is_occupied(self, x: int, y: int, z: int) -> bool:
        if not self.in_bounds(x, y, z):
            return True
        return self.cmap[self.lin_index(x, y, z)] == VAL_OCC

    # 这格是否「没看见过/不确定」
    def is_unknown(self, x: int, y: int, z: int) -> bool:
        if not self.in_bounds(x, y, z):
            return False
        return self.cmap[self.lin_index(x, y, z)] == VAL_UNK

    # 查这格的热度(软成本)。没启用热度场或出界都返回 0。
    def get_heat(self, x: int, y: int, z: int) -> float:
        if self.heat.size == 0 or not self.in_bounds(x, y, z):
            return 0.0
        return float(self.heat[self.lin_index(x, y, z)])

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------
    # 把这格强制设为空闲
    def set_free(self, x: int, y: int, z: int) -> None:
        if self.in_bounds(x, y, z):
            self.cmap[self.lin_index(x, y, z)] = VAL_FREE

    # 把 center 周围 d 米范围内的一坨格子都挖空。用途:确保起点/终点附近是空的,
    # 不然无人机当前就「站在障碍里」会搜不出路。
    def set_free_voxel_and_surroundings(self, center: np.ndarray, d: float) -> None:
        n = int(round(d / self.res + 0.5))
        cx, cy, cz = self.float_to_int(center)
        for dx in range(-n, n + 1):
            for dy in range(-n, n + 1):
                for dz in range(-n, n + 1):
                    self.set_free(int(cx + dx), int(cy + dy), int(cz + dz))

    def find_closest_free_point(self, point: np.ndarray) -> np.ndarray:
        """Return cell-center of nearest free voxel; mirrors C++ findClosestFreePoint.

        中文:给一个点,如果它本身就在空闲格里就直接返回;否则一圈圈往外找,
        返回最近的那个空闲格中心。用来把「卡在障碍里」的点拉回到自由空间。
        """
        px, py, pz = self.float_to_int(point)
        if self.is_free(int(px), int(py), int(pz)):
            return self.int_to_float(np.array([px, py, pz]))
        best = None
        best_d = float("inf")
        # 搜索半径从 1 米起步,找不到就每次扩 0.5 米,最多到 5 米
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
        # 中文:从头建一张规划用的栅格地图。这是本文件最核心的函数。
        # 流程:定地图尺寸/原点 -> 全填「未知」-> 把点云画成障碍(带膨胀)->
        #       把动态障碍也画成障碍 -> 加边界墙 -> 叠热度场。
        # 入参:cells_*=各方向格数;center_map=地图中心世界坐标;cloud_occ=占据点云(M,3);
        #       z_ground/z_max=高度上下限;inflation=障碍膨胀(米,给无人机留余量);
        #       obst_pos/obst_bbox=动态障碍中心/半尺寸;traj_max_time=本次轨迹的总时长。
        res = self.res
        # 1) Padding so the inflation kernel fits inside the map
        # 1) 多留点边(padding),好让障碍「膨胀」时不会被地图边界截掉
        pad = int(np.ceil(5.0 * inflation / res))
        dimX = int(cells_x + pad)
        dimY = int(cells_y + pad)
        dimZ = int(cells_z)

        # 2) Z clamp: keep within [z_ground, z_max]
        # 2) 高度方向裁一下,别让地图伸到地面以下或飞行上限以上
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
        # 3) 算原点:就是 (0,0,0) 这格的「最小角」世界坐标,后面所有换算都基于它
        origin = np.array([
            center_map[0] - dimX * res * 0.5,
            center_map[1] - dimY * res * 0.5,
            center_map[2] - down * res,
        ], dtype=np.float64)
        origin[2] = max(z_ground, min(origin[2], z_max - dimZ * res))

        self.dim = np.array([dimX, dimY, dimZ], dtype=np.int64)
        self.origin = origin

        # 4) Fill with UNKNOWN
        # 4) 先把整张地图填成「未知」。注意:这个 Python 版本不会主动把内部挖空成
        #    free,没被点云打到的地方一直是 UNKNOWN(这点影响静态热度的处理,见下面)
        total = dimX * dimY * dimZ
        self.cmap = np.full(total, VAL_UNK, dtype=np.int8)

        # 5) Rasterize occupancy from the point cloud with cubic inflation
        # 5) 把传感器点云「画」成障碍格,并按 inflation 做立方体膨胀(给无人机留安全余量)
        m = int(np.floor(inflation / res))
        if cloud_occ is not None and len(cloud_occ) > 0:
            pts = np.asarray(cloud_occ, dtype=np.float64)
            mask = (pts[:, 2] >= z_ground) & (pts[:, 2] <= z_max)
            pts = pts[mask]
            if len(pts):
                idx = np.floor((pts - origin) / res).astype(np.int64)
                self._rasterize_cells(idx, m)

        # 6) Dynamic obstacles as occupied (current)
        # 6) 把动态障碍「当前位置」也画成被占。dyn_occ_mask 记下哪些格是动态障碍占的,
        #    后面叠静态热度时好把它们排掉(动态障碍另有动态热度处理)。
        self.dyn_occ_mask = np.zeros(total, dtype=bool)
        if self.dynamic_as_occupied_current:
            for c, hk in zip(obst_pos, obst_bbox):
                half = np.maximum(hk + max(self.dyn_base_inflation_m, inflation), 0.0)
                self._rasterize_aabb(np.asarray(c, dtype=np.float64), half, mark_dyn=True)

        # 7) Dynamic obstacles as occupied (future cone)
        # 7) 可选:把动态障碍「未来可能到达的范围」也当成被占(按最大速度*时长往外胀)
        if self.dynamic_as_occupied_future and traj_max_time > 0:
            for c, hk in zip(obst_pos, obst_bbox):
                half = hk + self.obst_max_vel * traj_max_time
                self._rasterize_aabb(np.asarray(c, dtype=np.float64), half, mark_dyn=True)

        # 8) y-boundary walls (only in y per C++ readMap step 8b)
        # 8) 在 y 方向的两端贴一层「墙」(设为被占),防止规划路径冲出地图边界。
        #    只在 y 方向加,和 C++ readMap 的第 8b 步保持一致。
        y_min_world = origin[1]
        y_max_world = origin[1] + dimY * res
        wall_cells = int(np.ceil(inflation / res))
        for j in range(min(wall_cells, dimY)):
            self.cmap[j * dimX:(j + 1) * dimX].reshape(-1)[:] = self.cmap[j * dimX:(j + 1) * dimX]  # noop, kept for clarity
        # easier: flatten per-z slab
        # 按每个 z 层(slab)逐行刷:y 最小侧几行和 y 最大侧几行全设为障碍
        for z in range(dimZ):
            base = z * dimX * dimY
            for j in range(min(wall_cells, dimY)):
                row_start = base + j * dimX
                self.cmap[row_start:row_start + dimX] = VAL_OCC
            for j in range(max(0, dimY - wall_cells), dimY):
                row_start = base + j * dimX
                self.cmap[row_start:row_start + dimX] = VAL_OCC

        # 9) Heat map composition
        # 9) 叠热度场:动态障碍周围的「软成本」+ 静态障碍边界周围的「软成本」。
        #    热度只是让 A* 不想靠近,不是硬挡;两类分别由下面两个 _compose 函数算。
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
        """Mark each cell index (and a cubic inflation around it) as occupied.

        中文:把一批格坐标设为被占,并在每个格周围 infl 个格的立方体范围一起设占
        (这就是「膨胀」:把障碍按无人机半径胀大,留出安全距离)。
        """
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
        # 中文:把一个轴对齐立方体盒子(中心 center,半边长 half)范围内的格子设为被占。
        # 主要给动态障碍用。mark_dyn=True 时同时在 dyn_occ_mask 里标记「这是动态障碍占的」。
        dimX, dimY, dimZ = self.dim
        res = self.res
        lo = self.float_to_int(center - half)
        hi = self.float_to_int(center + half)
        for iz in range(int(max(lo[2], 0)), int(min(hi[2], dimZ - 1)) + 1):
            for iy in range(int(max(lo[1], 0)), int(min(hi[1], dimY - 1)) + 1):
                for ix in range(int(max(lo[0], 0)), int(min(hi[0], dimX - 1)) + 1):
                    cell_center = self.int_to_float(np.array([ix, iy, iz]))
                    # 用格中心到盒子的距离判断是否真在盒内(留半格容差),避免边角误判
                    if np.all(np.abs(cell_center - center) <= half + res * 0.5):
                        lin = self.lin_index(ix, iy, iz)
                        self.cmap[lin] = VAL_OCC
                        if mark_dyn:
                            self.dyn_occ_mask[lin] = True

    def _compose_dynamic_heat(self, obst_pos, obst_bbox, traj_max_time):
        # 中文:给每个动态障碍周围铺一层「软成本」热度。直觉:离障碍越近、越在它
        # 未来要经过的路线上,热度越高,A* 就越不想走那里(但不硬挡,所以叫软成本)。
        # 热度由两部分相加:
        #   Hbase = 障碍「可达范围」内的基础排斥(离得越近越大);
        #   tube  = 沿障碍预测轨迹的「管子」,在多个未来时间采样点上取最大值,
        #           越早的时间权重越大(weights 按 exp 衰减),体现「先躲眼前的」。
        if not obst_pos:
            return
        Th = max(0.0, float(traj_max_time))            # 本次轨迹总时长(预测看多远)
        M = max(2, self.heat_num_samples)              # 未来时间采样点数
        tau_w = max(1e-3, self.heat_tau_ratio * max(1e-3, Th))  # 时间权重的衰减尺度
        # 优先用外部喂进来的预测时间戳;没有就在 [0, Th] 均匀采 M 个点
        times = (self.dyn_pred_times
                 if self.dyn_pred_times is not None and len(self.dyn_pred_times) >= 2
                 else np.linspace(0.0, Th, M))
        weights = np.exp(-times / tau_w)               # 越近的未来时刻权重越大
        R0 = self.dyn_heat_tube_radius_m               # 管子基础半径
        Rs = R0 + self.heat_gamma * times              # 管子半径随时间增大(预测越不准越胖)
        p = self.heat_p
        q = self.heat_q

        dimX, dimY, dimZ = self.dim

        for k, (ck, hk) in enumerate(zip(obst_pos, obst_bbox)):
            ck = np.asarray(ck, dtype=np.float64)      # 障碍中心
            hk = np.asarray(hk, dtype=np.float64)      # 障碍半尺寸
            # Rreach:障碍在 Th 时间内最远能到多远(用来界定要计算热度的范围)
            Rreach = float(np.max(hk)) + self.obst_max_vel * Th
            if Rreach <= 0:
                continue

            # Iterate cells within a generous AABB around the obstacle
            # 只在障碍周围一个够大的盒子里算热度,盒子外热度必然是 0,省得扫全图
            extent = np.array([Rreach, Rreach, Rreach]) + max(R0, np.max(hk)) + self.obst_max_vel * Th
            lo = self.float_to_int(ck - extent)
            hi = self.float_to_int(ck + extent)
            x0, y0, z0 = (int(max(v, 0)) for v in lo)
            x1, y1, z1 = (int(min(v, d - 1)) for v, d in zip(hi, (dimX, dimY, dimZ)))

            samples_k = self.dyn_pred_samples[k] if (self.dyn_pred_samples is not None and k < len(self.dyn_pred_samples)) else None

            if x1 < x0 or y1 < y0 or z1 < z0:
                continue
            # Vectorised over the region AABB (was a Python triple-loop -> seconds
            # for big/fast obstacles whose reach covers most of the grid). Same
            # math, same int_to_float cell centres -> numerically equivalent.
            # 中文:用 numpy 向量化算整块盒子(以前是 Python 三重循环,大/快的障碍
            # 覆盖大半张图时要好几秒)。算的是同样的格中心、同样的公式,结果等价。
            axs = (np.arange(x0, x1 + 1) + 0.5) * self.res + self.origin[0]
            ays = (np.arange(y0, y1 + 1) + 0.5) * self.res + self.origin[1]
            azs = (np.arange(z0, z1 + 1) + 0.5) * self.res + self.origin[2]
            CX, CY, CZ = np.meshgrid(axs, ays, azs, indexing="ij")
            # base reachability blob: distance from cell to the obstacle AABB
            # 基础排斥团:每个格中心到障碍盒子的距离(在盒内则距离为 0),
            # 距离越近热度越高,超过 Rreach 就归 0
            dx = np.maximum(np.abs(CX - ck[0]) - hk[0], 0.0)
            dy = np.maximum(np.abs(CY - ck[1]) - hk[1], 0.0)
            dz = np.maximum(np.abs(CZ - ck[2]) - hk[2], 0.0)
            d_box = np.sqrt(dx * dx + dy * dy + dz * dz)
            Hbase = np.where(d_box <= Rreach,
                             self.heat_alpha0 * (1.0 - np.minimum(d_box / Rreach, 1.0)) ** p,
                             0.0)
            # trajectory tube: max over predicted time samples
            # 轨迹管子:对每个未来时间采样点算一个排斥团,所有采样点上取最大值。
            # cj = 该时刻障碍预测到的中心(没有预测就退回用当前中心 ck)
            tube_max = np.zeros_like(d_box)
            for j, (tj, Rj, wj) in enumerate(zip(times, Rs, weights)):
                cj = samples_k[j] if (samples_k is not None and j < len(samples_k)) else ck
                cj = np.asarray(cj, dtype=np.float64)
                ex = np.maximum(np.abs(CX - cj[0]) - hk[0], 0.0)
                ey = np.maximum(np.abs(CY - cj[1]) - hk[1], 0.0)
                ez = np.maximum(np.abs(CZ - cj[2]) - hk[2], 0.0)
                d_j = np.sqrt(ex * ex + ey * ey + ez * ez)
                contrib = np.where(d_j <= Rj,
                                   wj * (1.0 - np.minimum(d_j / Rj, 1.0)) ** q, 0.0)
                tube_max = np.maximum(tube_max, contrib)
            # 总热度 = 基础排斥 + 管子排斥(各自有权重),再封顶到 Hmax
            Hk = Hbase + self.heat_alpha1 * tube_max
            if self.heat_Hmax > 0:
                Hk = np.minimum(Hk, self.heat_Hmax)
            IX, IY, IZ = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1),
                                     np.arange(z0, z1 + 1), indexing="ij")
            lin = (IX + dimX * (IY + dimY * IZ)).ravel()
            Hk_flat = Hk.ravel()
            if not self.use_soft_cost_obstacles:
                # original skipped occupied cells entirely; heat>=0 so max-with-0 is a no-op
                # 不用软成本时:被占的格不铺热度(反正它们已经被硬挡了)
                Hk_flat = np.where(self.cmap[lin] > VAL_FREE, 0.0, Hk_flat)
            # 用「取最大」而不是相加来叠加:多个障碍重叠时取最危险的那个值
            np.maximum.at(self.heat, lin, Hk_flat)

    def _compose_static_heat(self):
        # 中文:给静态障碍(墙、树等点云障碍)的边界周围也铺一层软成本热度,
        # 让 A* 路径自然离墙远一点。做法:找出障碍的「边界格」当种子,以每个种子
        # 为中心,在半径 rmax 内的格按距离衰减地刷热度(同一格被多个种子刷时取最大)。
        if self.heat.size == 0:
            return
        Rcell = int(np.ceil(self.static_heat_rmax_m / self.res))  # 影响半径换算成格数
        if Rcell <= 0:
            return
        # 预算好半径内所有偏移量(dx,dy,dz)及其实际距离,后面每个种子复用
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
        # 种子 = 所有被占的格;可选地把动态障碍占的格排掉(它们已有动态热度)
        seed_idxs = np.where(self.cmap == VAL_OCC)[0]
        if self.static_heat_exclude_dynamic and self.dyn_occ_mask is not None:
            seed_idxs = seed_idxs[~self.dyn_occ_mask[seed_idxs]]

        # boundary_only: keep seeds with at least one non-occupied 6-neighbor
        # 只留「边界种子」:六个直接邻居里至少有一个不是障碍(即障碍的外壳)。
        # 这样只在障碍表面往外铺热度,不浪费算力刷障碍内部。
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
                # 默认允许往 UNKNOWN 格刷热度。原因(见文件顶 apply_on_unknown 注释):
                # 这个 Python 地图不主动挖 free,障碍周边大多是 UNKNOWN,不允许就根本不显热
                if not self.static_heat_apply_on_unknown and self.cmap[tlin] == VAL_UNK:
                    continue
                # 距离越近热度越大((1-u)^p),封顶 Hmax;同一格取所有种子里的最大值
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

        中文:精确地「沿直线一格一格走」(Amanatides-Woo DDA 算法),判断从 p1 到 p2
        这条直线段有没有穿过任何「值 >= val 的格」(默认 val=100 即障碍)。
        比「在线上隔点采样」靠谱:它绝不会跳过线真正穿过的格,所以不会漏判碰撞。
        用途:给后面做「视线捷径」(short_cut_by_los)判断两点能不能直连。
        """
        res = self.res
        a = (np.asarray(p1, dtype=np.float64) - self.origin) / res
        b = (np.asarray(p2, dtype=np.float64) - self.origin) / res
        cur = np.floor(a).astype(np.int64)   # 当前所在格
        end = np.floor(b).astype(np.int64)   # 终点所在格
        d = b - a
        # tmax[i]:沿 i 轴走到下一格边界还要多少参数 t;tdelta[i]:每跨一格 t 增量;step:走向(+1/-1)
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
        # 每次挑「最先到达边界」的那个轴跨一格,直到走到终点格
        for _ in range(int(np.abs(end - cur).sum()) + 4):
            if cur[0] == end[0] and cur[1] == end[1] and cur[2] == end[2]:
                break
            tmin = float(tmax.min())
            # tied:多个轴几乎同时到边界 = 直线正好擦过格的棱/角
            tied = [i for i in range(3) if tmax[i] <= tmin + eps and np.isfinite(tdelta[i])]
            if len(tied) >= 2:
                # Edge/corner crossing: the segment grazes the cells reachable by
                # stepping each tied axis alone — check them so a corner graze
                # through an obstacle is not missed.
                # 擦棱/擦角:额外探一下「只沿其中一个轴跨一格」能到的那些格,
                # 防止直线从障碍的棱角穿过却被漏判
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
        # 中文:判断 a 到 b 这条「胶囊」(带半径的线段)是否畅通无障碍。
        # 先查中心线;再查半径 inflate_radius_cells 内一圈平行偏移线 —— 相当于检查
        # 这条线周围留有 radius 格的安全余量。全通才返回 True。
        # 用途:路径简化时判断「能不能把两个点直接连起来、跳过中间拐点」。
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if np.linalg.norm(b - a) < 1e-9:
            return True
        # Exact center-line check (DDA) — never skips a grazed cell.
        # 先精确查中心线本身
        if self.is_blocked(a, b, VAL_OCC):
            return False
        radius = max(0, int(inflate_radius_cells))
        if radius == 0:
            return True
        # Clearance margin: same exact check on parallel offset lines.
        # 安全余量:对一圈平行偏移的线做同样的精确检查
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
    # 返回所有等于 value 的格子的「世界坐标中心点」列表。主要给可视化用(画占据/空闲格)。
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

    # 点 p 所在格的 26 个邻居里,是否有任何一个不是空闲(即贴着障碍/未知区)
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
